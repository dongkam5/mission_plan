"""
Validator v1.0 - 규칙 기반 임무 검증기
근거: AFDP5-0 COA Analysis & Wargaming, JP3-30 ACO, DAFMAN11-260

검증 항목:
1. 공역 충돌  - 경로가 NFZ 통과 금지 (JP3-30 ACO)
2. 위협 침범  - 경로가 SAM/RADAR 반경 내 통과 금지
3. 최소 고도  - AGL 200m 이상 유지 (FMI3-04.155)
4. 자산 충돌  - 동일 공역 동시 비행 금지
5. 임무 순서  - ISR → SEAD → STRIKE 순서 준수 (JP3-30 MAAP)
6. 항속 거리  - 총 경로 거리 ≤ 최대 항속거리 (DAFMAN11-260)
7. MUM-T 비율 - 무인기 ≥ 2 × 전투기 (FMI3-04.155)
"""
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from modules.config import (
    ASSET_PERFORMANCE, MIN_ALTITUDE_AGL,
    MISSION_ASSET_REQUIREMENTS, MUMT_RATIO
)


# ================================================================
# 검증 결과 데이터클래스
# ================================================================

@dataclass
class ValidationIssue:
    """개별 검증 항목 결과"""
    rule_id: str                     # 규칙 ID (예: "NFZ_VIOLATION")
    severity: str                    # "ERROR" / "WARNING" / "INFO"
    asset_id: Optional[str]          # 관련 자산 ID (없으면 전체)
    message: str                     # 한국어 메시지
    doctrine_ref: str                # 교리 참조
    suggestion: str = ""             # 개선 권고사항

    @property
    def icon(self) -> str:
        return {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(self.severity, "⚪")


@dataclass
class ValidationReport:
    """전체 검증 보고서"""
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_rules: List[str] = field(default_factory=list)
    validate_time_ms: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")

    def summary(self) -> str:
        if self.is_valid:
            return f"✅ 검증 통과 | 경고 {self.warning_count}건"
        return f"❌ 검증 실패 | 오류 {self.error_count}건 | 경고 {self.warning_count}건"

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "rule_id": i.rule_id,
                    "severity": i.severity,
                    "asset_id": i.asset_id,
                    "message": i.message,
                    "doctrine_ref": i.doctrine_ref,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "checked_rules": self.checked_rules,
            "validate_time_ms": self.validate_time_ms,
        }


# ================================================================
# 검증 규칙 (개별 함수)
# ================================================================

def _dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리 (km), 위경도 근사"""
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    return math.sqrt(dlat ** 2 + dlon ** 2) * 111.0


def _path_length_km(path: List) -> float:
    """경로 총 거리 (km)"""
    total = 0.0
    for i in range(1, len(path)):
        total += _dist_km(path[i-1][0], path[i-1][1], path[i][0], path[i][1])
    return total


def check_nfz_violation(
    asset_id: str,
    path: List,
    threats: List[Dict]
) -> List[ValidationIssue]:
    """
    Rule 1: 비행금지구역(NFZ) 침범 검사
    근거: JP3-30 ACO (Airspace Control Order)
    """
    issues = []
    nfz_list = [t for t in threats if t.get("type") == "NFZ"]
    if not nfz_list or not path:
        return issues

    for point in path[::5]:  # 5포인트 간격으로 샘플링
        lat, lon = point[0], point[1]
        for nfz in nfz_list:
            if (nfz.get("lat_min", 99) <= lat <= nfz.get("lat_max", -99) and
                    nfz.get("lon_min", 999) <= lon <= nfz.get("lon_max", -999)):
                issues.append(ValidationIssue(
                    rule_id="NFZ_VIOLATION",
                    severity="ERROR",
                    asset_id=asset_id,
                    message=f"[{asset_id}] 비행금지구역 '{nfz.get('name', 'NFZ')}' 침범 감지",
                    doctrine_ref="JP3-30 ACO (Airspace Control Order)",
                    suggestion="경로를 NFZ 외곽으로 재계획하거나 안전 마진을 증가시키세요."
                ))
                break  # 동일 NFZ 중복 보고 방지

    return issues


def check_threat_penetration(
    asset_id: str,
    asset_type: str,
    path: List,
    threats: List[Dict],
    margin_km: float
) -> List[ValidationIssue]:
    """
    Rule 2: 위협 반경 침범 검사
    근거: JP3-30 ROE, FMI3-04.155 자산 보호
    """
    issues = []
    active_threats = [t for t in threats if t.get("type") in ("SAM", "RADAR") and t.get("lat")]
    if not active_threats or not path:
        return issues

    rcs = ASSET_PERFORMANCE.get(asset_type, {}).get("rcs", 1.0)

    for point in path[::5]:
        lat, lon = point[0], point[1]
        for threat in active_threats:
            dist = _dist_km(lat, lon, threat["lat"], threat["lon"])
            radius = threat.get("radius_km", 30.0) + margin_km
            if dist < radius * rcs:
                severity = "ERROR" if dist < radius * 0.5 else "WARNING"
                issues.append(ValidationIssue(
                    rule_id="THREAT_PENETRATION",
                    severity=severity,
                    asset_id=asset_id,
                    message=(f"[{asset_id}] {threat.get('type')} '{threat.get('name', '?')}' "
                             f"위협 반경 내 비행 감지 (거리: {dist:.1f}km, 반경: {radius:.1f}km)"),
                    doctrine_ref="JP3-30 ROE / FMI3-04.155 자산 보호",
                    suggestion=f"안전 마진을 {margin_km + 5:.0f}km 이상으로 증가하거나 경로를 우회하세요."
                ))
                break

    return issues


def check_min_altitude(
    asset_id: str,
    path: List,
    min_agl: float = MIN_ALTITUDE_AGL
) -> List[ValidationIssue]:
    """
    Rule 3: 최소 고도 검사 (3D 경로만)
    근거: FMI3-04.155 최소 AGL 200m
    """
    issues = []
    if not path or len(path[0]) < 3:
        return issues  # 2D 경로는 스킵

    violations = [p for p in path if p[2] < min_agl]
    if violations:
        min_alt = min(p[2] for p in violations)
        issues.append(ValidationIssue(
            rule_id="MIN_ALTITUDE",
            severity="WARNING",
            asset_id=asset_id,
            message=(f"[{asset_id}] 최소 고도 미달 구간 {len(violations)}개 "
                     f"(최저 고도: {min_alt:.0f}m, 기준: {min_agl}m AGL)"),
            doctrine_ref="FMI3-04.155 최소 AGL 200m 기준",
            suggestion="3D 알고리즘의 MIN_ALTITUDE_AGL 설정을 확인하세요."
        ))

    return issues


def check_asset_collision(
    formation_paths: Dict[str, List],
    min_sep_km: float = 2.0
) -> List[ValidationIssue]:
    """
    Rule 4: 자산 간 충돌 위험 검사
    근거: FMI3-04.155 측방/수직/시간 분리 원칙
    """
    issues = []
    asset_ids = list(formation_paths.keys())

    for i in range(len(asset_ids)):
        for j in range(i + 1, len(asset_ids)):
            id_a = asset_ids[i]
            id_b = asset_ids[j]
            path_a = formation_paths[id_a]
            path_b = formation_paths[id_b]

            if not path_a or not path_b:
                continue

            # 두 경로에서 샘플링해서 최근접 거리 계산
            min_dist = float('inf')
            sample_a = path_a[::max(1, len(path_a) // 20)]
            sample_b = path_b[::max(1, len(path_b) // 20)]

            for pa in sample_a:
                for pb in sample_b:
                    d = _dist_km(pa[0], pa[1], pb[0], pb[1])
                    min_dist = min(min_dist, d)

            if min_dist < min_sep_km:
                issues.append(ValidationIssue(
                    rule_id="ASSET_COLLISION",
                    severity="WARNING",
                    asset_id=f"{id_a}/{id_b}",
                    message=(f"[{id_a}] ↔ [{id_b}] 최근접 거리 {min_dist:.1f}km "
                             f"(기준: {min_sep_km}km 이상)"),
                    doctrine_ref="FMI3-04.155 측방/수직/시간 분리 원칙",
                    suggestion="출발 시간 간격 조정 또는 고도 분리를 통해 충돌을 방지하세요."
                ))

    return issues


def check_mission_sequence(
    mission_sequence: List[str]
) -> List[ValidationIssue]:
    """
    Rule 5: 임무 수행 순서 검사
    근거: JP3-30 MAAP - ISR → SEAD → STRIKE 순서 준수
    """
    issues = []
    if not mission_sequence:
        return issues

    # 교리 기준 순서
    DOCTRINE_ORDER = {"ISR": 0, "SEAD": 1, "STRIKE": 2, "CAS": 3}

    # 현재 순서 검사
    for i in range(len(mission_sequence) - 1):
        curr = mission_sequence[i]
        next_ = mission_sequence[i + 1]
        curr_order = DOCTRINE_ORDER.get(curr, 99)
        next_order = DOCTRINE_ORDER.get(next_, 99)

        if curr_order > next_order:
            issues.append(ValidationIssue(
                rule_id="MISSION_SEQUENCE",
                severity="WARNING",
                asset_id=None,
                message=f"임무 순서 오류: {curr} → {next_} (교리: ISR → SEAD → STRIKE)",
                doctrine_ref="JP3-30 MAAP 임무 우선순위 원칙",
                suggestion=f"교리 준수 순서: {' → '.join(sorted(mission_sequence, key=lambda x: DOCTRINE_ORDER.get(x, 99)))}"
            ))

    # STRIKE 전 SEAD 필수 검사
    if "STRIKE" in mission_sequence and "SEAD" not in mission_sequence:
        issues.append(ValidationIssue(
            rule_id="SEAD_REQUIRED",
            severity="WARNING",
            asset_id=None,
            message="STRIKE 임무 수행 시 SEAD(방공제압) 선행 권고",
            doctrine_ref="JP3-30 MAAP - SEAD는 STRIKE 패키지 보호를 위한 필수 요소",
            suggestion="SEAD 임무를 추가하거나, 위협이 없음을 확인하세요."
        ))

    return issues


def check_range_limit(
    asset_id: str,
    asset_type: str,
    path_in: List,
    path_out: List
) -> List[ValidationIssue]:
    """
    Rule 6: 항속거리 초과 검사
    근거: DAFMAN11-260 연료 계획 원칙
    """
    issues = []
    max_range = ASSET_PERFORMANCE.get(asset_type, {}).get("range_km", 999)

    total_dist = _path_length_km(path_in) + _path_length_km(path_out)
    if total_dist > max_range * 0.9:  # 90% 초과 시 경고
        severity = "ERROR" if total_dist > max_range else "WARNING"
        issues.append(ValidationIssue(
            rule_id="RANGE_EXCEEDED",
            severity=severity,
            asset_id=asset_id,
            message=(f"[{asset_id}] 총 비행거리 {total_dist:.0f}km "
                     f"(최대 항속: {max_range}km, {total_dist/max_range*100:.0f}%)"),
            doctrine_ref="DAFMAN11-260 연료 계획 - 최소 10% 예비 연료 필요",
            suggestion="경유지를 추가하거나 목표를 변경하세요. 또는 더 긴 항속 자산으로 교체하세요."
        ))

    return issues


def check_mumt_ratio(
    n_fighter: int,
    n_uav_total: int
) -> List[ValidationIssue]:
    """
    Rule 7: MUM-T 비율 검사
    근거: FMI3-04.155 Lead/Wingman (유인1:무인2)
    """
    issues = []
    if n_fighter == 0:
        return issues

    ratio = n_uav_total / n_fighter
    if ratio < MUMT_RATIO:
        issues.append(ValidationIssue(
            rule_id="MUMT_RATIO",
            severity="WARNING",
            asset_id=None,
            message=(f"MUM-T 비율 미달: 전투기 {n_fighter}대 : UAV {n_uav_total}대 "
                     f"(현재 1:{ratio:.1f}, 권장 1:{MUMT_RATIO:.0f})"),
            doctrine_ref="FMI3-04.155 Lead/Wingman - 유인 1 : 무인 2 권장",
            suggestion=f"UAV를 {int(n_fighter * MUMT_RATIO - n_uav_total)}대 이상 추가하세요."
        ))

    return issues


# ================================================================
# MissionValidator - 통합 검증기
# ================================================================

class MissionValidator:
    """
    임무 계획 전체 검증기
    근거: AFDP5-0 Step 4 COA Analysis & Wargaming

    사용법:
        validator = MissionValidator()
        report = validator.validate(
            formation_result=...,
            formation_paths=...,
            threats=...,
            mission_sequence=...,
            margin_km=5.0
        )
    """

    def validate(
        self,
        formation_result=None,   # FormationResult
        formation_paths: Dict = None,    # {asset_id: path}
        threats: List[Dict] = None,
        mission_sequence: List[str] = None,
        margin_km: float = 5.0
    ) -> ValidationReport:

        import time as _time
        start_t = _time.time()

        report = ValidationReport()
        threats = threats or []
        formation_paths = formation_paths or {}
        mission_sequence = mission_sequence or []

        # ── Rule 5: 임무 순서 ──
        report.checked_rules.append("MISSION_SEQUENCE")
        report.issues.extend(check_mission_sequence(mission_sequence))

        # ── Rule 7: MUM-T 비율 ──
        if formation_result:
            report.checked_rules.append("MUMT_RATIO")
            n_uav = formation_result.n_recon_uav + formation_result.n_attack_uav
            report.issues.extend(check_mumt_ratio(formation_result.n_fighter, n_uav))

        # ── 자산별 검증 ──
        if formation_result and formation_paths:
            for asset in (formation_result.assets or []):
                path_in  = formation_paths.get(asset.asset_id, {}).get("in",  [])
                path_out = formation_paths.get(asset.asset_id, {}).get("out", [])
                full_path = path_in + path_out

                # Rule 1: NFZ
                report.checked_rules.append(f"NFZ_{asset.asset_id}")
                report.issues.extend(check_nfz_violation(asset.asset_id, full_path, threats))

                # Rule 2: 위협 침범
                report.checked_rules.append(f"THREAT_{asset.asset_id}")
                report.issues.extend(check_threat_penetration(
                    asset.asset_id, asset.asset_type, full_path, threats, margin_km
                ))

                # Rule 3: 최소 고도
                report.checked_rules.append(f"ALT_{asset.asset_id}")
                report.issues.extend(check_min_altitude(asset.asset_id, full_path))

                # Rule 6: 항속거리
                report.checked_rules.append(f"RANGE_{asset.asset_id}")
                report.issues.extend(check_range_limit(
                    asset.asset_id, asset.asset_type, path_in, path_out
                ))

        # ── Rule 4: 자산 충돌 ──
        if len(formation_paths) > 1:
            report.checked_rules.append("ASSET_COLLISION")
            path_dict = {k: v.get("in", []) for k, v in formation_paths.items()}
            report.issues.extend(check_asset_collision(path_dict))

        # ── 최종 유효성 판단 ──
        report.is_valid = report.error_count == 0
        report.validate_time_ms = round((_time.time() - start_t) * 1000, 2)

        return report
