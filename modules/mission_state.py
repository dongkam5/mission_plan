"""
미션 상태 관리 v2.0
- Asset, Formation 데이터클래스 추가
- 편대 구성 결과 상태 관리
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
from modules.config import (
    DEFAULT_SAFETY_MARGIN, DEFAULT_STPT_GAP,
    LOG_DIR, ENABLE_LOGGING, DEFAULT_ALGORITHM, ENABLE_3D
)
import os


@dataclass
class MissionParams:
    """미션 파라미터"""
    start: str = "부산(Busan)"
    target_lat: float = 39.000
    target_lon: float = 125.700
    target_name: str = "PY-Core"
    rtb: bool = True
    margin: float = DEFAULT_SAFETY_MARGIN
    waypoint: Optional[str] = None
    stpt_gap: int = DEFAULT_STPT_GAP
    algorithm: str = DEFAULT_ALGORITHM
    enable_3d: bool = ENABLE_3D

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class Threat:
    """
    위협 객체
    alt: 지형 고도 + 안테나 높이 (자동 설정됨)
    """
    name: str
    type: str  # "SAM", "RADAR", "NFZ"
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_km: Optional[float] = None
    alt: float = 0.0
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict):
        if 'alt' not in data and data.get('type') in ['SAM', 'RADAR']:
            data['alt'] = 0.0
        return cls(**data)


# ================================================================
# 편대 구성 관련 데이터클래스 (신규)
# ================================================================

@dataclass
class Asset:
    """
    개별 자산 (전투기 / 정찰UAV / 자폭UAV)
    근거: FMI3-04.155 UAS Operations
    """
    asset_id: str                          # 고유 ID (예: "F-01", "RUAV-01")
    asset_type: str                        # "fighter" / "recon_uav" / "attack_uav"
    callsign: str                          # 콜사인 (예: "Eagle-1")
    base: str = "부산(Busan)"             # 출발 기지
    assigned_mission: Optional[str] = None # 배정된 임무 ("ISR"/"SEAD"/"STRIKE"/"CAS")
    assigned_target_idx: Optional[int] = None  # 배정된 목표 인덱스 (헝가리안 결과)
    path: List = field(default_factory=list)   # 계산된 경로 웨이포인트

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class FormationResult:
    """
    MILP + 헝가리안 편대 구성 결과
    근거: JP3-30 Apportionment/Allocation, FMI3-04.155 MUM-T
    """
    # MILP 결과 - 자산 대수
    n_fighter: int = 0      # 전투기 대수
    n_recon_uav: int = 0    # 정찰UAV 대수
    n_attack_uav: int = 0   # 자폭UAV 대수

    # 임무 시퀀스
    mission_sequence: List[str] = field(default_factory=list)  # ["ISR","SEAD","STRIKE"]

    # 헝가리안 배정 결과
    assets: List[Asset] = field(default_factory=list)

    # 최적화 메타 정보
    total_cost: float = 0.0       # 총 비용
    is_feasible: bool = False     # MILP 해 존재 여부
    solver_status: str = ""       # "Optimal" / "Infeasible" / "Error"
    solve_time_ms: float = 0.0    # 계산 시간
    utilization_pct: int = 0      # N_max 대비 자산 활용률 (%)

    def total_assets(self) -> int:
        return self.n_fighter + self.n_recon_uav + self.n_attack_uav

    def to_dict(self):
        d = asdict(self)
        d['total_assets'] = self.total_assets()
        return d

    def summary(self) -> str:
        """편대 구성 요약 문자열"""
        return (
            f"전투기 {self.n_fighter}대 | "
            f"정찰UAV {self.n_recon_uav}대 | "
            f"자폭UAV {self.n_attack_uav}대 | "
            f"총 {self.total_assets()}대"
        )


# ================================================================
# MissionState - 편대 구성 상태 추가
# ================================================================

class MissionState:
    """미션 전체 상태 관리 v2.0"""

    def __init__(self):
        self.params = MissionParams()
        self.threats: List[Threat] = [
            Threat(name="Default SAM", type="SAM", lat=37.200, lon=127.800, radius_km=20, alt=0)
        ]
        self.chat_history: List[Dict[str, str]] = [
            {"role": "assistant", "content": "작전관님, 명령을 대기 중입니다.", "reasoning": ""}
        ]
        # 편대 구성 결과 (formation_optimizer 실행 후 채워짐)
        self.formation: Optional[FormationResult] = None

    def add_threat(self, threat: Threat):
        """위협 추가"""
        self.threats.append(threat)

    def remove_threat(self, name: str):
        """위협 삭제"""
        self.threats = [t for t in self.threats if t.name != name]

    def add_chat_message(self, role: str, content: str, reasoning: str = ""):
        """채팅 메시지 추가"""
        self.chat_history.append({"role": role, "content": content, "reasoning": reasoning})

    def set_formation(self, result: FormationResult):
        """편대 구성 결과 저장"""
        self.formation = result

    def save_to_file(self, filename: str):
        """상태 저장"""
        if ENABLE_LOGGING:
            os.makedirs(LOG_DIR, exist_ok=True)
            filepath = os.path.join(LOG_DIR, filename)
            data = {
                "timestamp": datetime.now().isoformat(),
                "params": self.params.to_dict(),
                "threats": [t.to_dict() for t in self.threats],
                "chat_history": self.chat_history,
                "formation": self.formation.to_dict() if self.formation else None
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filename: str):
        """저장된 상태 복원"""
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = cls()
        state.params = MissionParams.from_dict(data["params"])
        state.threats = [Threat.from_dict(t) for t in data["threats"]]
        state.chat_history = data["chat_history"]
        return state
