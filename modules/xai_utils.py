"""
XAI (Explainable AI) 유틸리티 v2.8 (Strict NFZ + 3D 통합 + 중첩 확률 논리 수정)
- 모든 거리 계산 3D 통합 (알고리즘 무관)
- NFZ: 진입 시 즉시 리스크 1.0 (Strict No-Go)
- 레이더(Total_PD)와 SAM(Total_PK)의 독립 결합 확률 산출 후 최종 리스크(AND 조건) 계산
- 최종 리스크 = max(total_pd * total_pk, NFZ 리스크)
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from modules.config import HEATMAP_RESOLUTION, MAP_BOUNDS

# [중요] 레이더 음영 체크 함수 임포트
try:
    from modules.radar_shadow import check_line_of_sight
except ImportError:
    def check_line_of_sight(*args, **kwargs): return True

LAT_TO_KM = 110.57


class XAIUtils:
    """XAI 관련 기능 v2.8"""

    @staticmethod
    def calculate_risk_score(
        lat: float,
        lon: float,
        threats: List[dict],
        margin: float,
        terrain_loader=None,
        target_alt: Optional[float] = None
    ) -> float:
        """
        특정 위치의 위험도 점수 계산
        - 레이더 탐지 확률(Total_PD)과 SAM 피격 확률(Total_PK)의 곱으로 무기체계 리스크 산출
        """
        if not threats: return 0.0

        # 고도가 없어도 지형 기반 기본 고도(500m AGL)를 생성하여 3D로 고정
        if target_alt is None:
            try:
                ground_elev = terrain_loader.get_elevation(lat, lon) if terrain_loader else 0.0
                target_alt = ground_elev + 500.0
            except: 
                target_alt = 500.0

        # 독립 시행 결합 확률을 위한 '탐지/피격되지 않을 확률' 초기값
        mul_not_p_d = 1.0  # 모든 레이더로부터 탐지되지 않을 생존 확률
        mul_not_p_k = 1.0  # 모든 SAM으로부터 피격되지 않을 생존 확률
        nfz_risk = 0.0     # NFZ 리스크 초기화

        for t in threats:
            # --- NFZ 처리: 엄격한 진입 금지 ---
            if t["type"] == "NFZ":
                margin_deg = margin / LAT_TO_KM
                if (t.get("lat_min", 0) - margin_deg <= lat <= t.get("lat_max", 0) + margin_deg and
                    t.get("lon_min", 0) - margin_deg <= lon <= t.get("lon_max", 0) + margin_deg):
                    nfz_risk = 1.0 
                continue

            # --- 무기 체계(SAM/RADAR) 3D 거리 계산 ---
            dist_2d_km = math.sqrt(((lat - t["lat"]) * LAT_TO_KM) ** 2 +
                                   ((lon - t["lon"]) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2)
            threat_alt_msl = float(t.get("alt", 0))
            alt_diff_km = (target_alt - threat_alt_msl) / 1000.0
            dist_3d_km = math.sqrt(dist_2d_km**2 + alt_diff_km**2)

            threat_radius = float(t.get("radius_km", 0))
            if dist_3d_km >= threat_radius + margin: 
                continue

            # 레이더 음영 체크 (LOS 반영)
            if terrain_loader:
                is_visible = check_line_of_sight(
                    (t["lat"], t["lon"], threat_alt_msl), 
                    (lat, lon, target_alt), 
                    terrain_loader
                )
                if not is_visible: 
                    continue

            # --- 위협 유형별 확률 계산 ---
            if t["type"] == "RADAR":
                loss, rcs_m2, pd_k = float(t.get("loss", 21.1)), float(t.get("rcs_m2", 2.5)), float(t.get("pd_k", 0.4))
                R_D = threat_radius + margin
                logit_pd = math.log(0.1 / 0.9)
                snr_req = 10.0 ** (logit_pd / max(pd_k, 1e-9) / 10.0)
                snr0 = snr_req * ((R_D * 1000.0) ** 4) * (loss / max(rcs_m2, 1e-12))
                snr = snr0 * (rcs_m2 / loss) / ((dist_3d_km * 1000.0) ** 4)
                p_d = 1.0 / (1.0 + math.exp(-pd_k * 10.0 * math.log10(max(snr, 1e-30))))
                
                # 중첩 계산: 탐지되지 않을 확률을 계속 곱함
                mul_not_p_d *= (1.0 - p_d)

            elif t["type"] == "SAM":
                sskp = float(t.get("sskp", 0.75))
                d0, sigma = float(t.get("pk_peak_km", 0.35 * threat_radius)), float(t.get("pk_sigma_km", 0.20 * threat_radius))
                if sigma > 0:
                    p_k = sskp * math.exp(-((dist_3d_km - d0) ** 2) / (2.0 * sigma ** 2))
                    
                    # 중첩 계산: 피격되지 않을 확률을 계속 곱함
                    mul_not_p_k *= (1.0 - p_k)

        # --- 최종 리스크 산출 ---
        # 1 - (모두 생존할 확률) = 하나라도 탐지/피격될 확률
        total_pd = 1.0 - mul_not_p_d 
        total_pk = 1.0 - mul_not_p_k
        
        # 무기체계 리스크 = 탐지됨 AND 피격됨 (두 확률의 곱)
        weapon_risk = total_pd * total_pk
        
        # 최종 리스크 = max(무기체계 위험도, NFZ 위험도)
        total_risk = max(weapon_risk, nfz_risk)
        
        return float(min(1.0, total_risk))

    @staticmethod
    def generate_heatmap_data(threats, margin, terrain_loader=None):
        min_lat, max_lat = MAP_BOUNDS["min_lat"], MAP_BOUNDS["max_lat"]
        min_lon, max_lon = MAP_BOUNDS["min_lon"], MAP_BOUNDS["max_lon"]
        heatmap_data = []
        step_lat = (max_lat - min_lat) / HEATMAP_RESOLUTION
        step_lon = (max_lon - min_lon) / HEATMAP_RESOLUTION

        for i in range(HEATMAP_RESOLUTION):
            for j in range(HEATMAP_RESOLUTION):
                lat = min_lat + i * step_lat
                lon = min_lon + j * step_lon
                risk = XAIUtils.calculate_risk_score(lat, lon, threats, margin, terrain_loader)
                if risk > 0.01:
                    heatmap_data.append([lat, lon, risk])
        return heatmap_data

    @staticmethod
    def analyze_path_risk(path, threats, margin, terrain_loader=None):
        if not path:
            return {"avg_risk": 0, "max_risk": 0, "high_risk_segments": 0, "total_length_km": 0}

        risks = []
        total_length = 0.0
        for i, p in enumerate(path):
            lat, lon = p[0], p[1]
            target_alt = p[2] if len(p) >= 3 else None

            risk = XAIUtils.calculate_risk_score(lat, lon, threats, margin, terrain_loader, target_alt)
            risks.append(risk)

            if i > 0:
                d2d = math.sqrt(((lat - path[i-1][0]) * LAT_TO_KM) ** 2 + 
                                 ((lon - path[i-1][1]) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2)
                alt_cur = p[2] if len(p) >= 3 else (terrain_loader.get_elevation(lat, lon) + 500 if terrain_loader else 500)
                alt_prev = path[i-1][2] if len(path[i-1]) >= 3 else (terrain_loader.get_elevation(path[i-1][0], path[i-1][1]) + 500 if terrain_loader else 500)
                total_length += math.sqrt(d2d**2 + ((alt_cur - alt_prev)/1000.0)**2)

        return {
            "avg_risk": float(np.mean(risks)) if risks else 0.0,
            "max_risk": float(max(risks)) if risks else 0.0,
            "high_risk_segments": sum(1 for r in risks if r > 0.7),
            "total_length_km": total_length
        }