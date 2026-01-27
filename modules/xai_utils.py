"""
XAI (Explainable AI) 유틸리티 - 레이더 음영 시각화 포함
"""
import numpy as np
import math
from typing import List, Dict, Tuple
from modules.config import HEATMAP_RESOLUTION, MAP_BOUNDS

# [중요] 레이더 음영 체크 함수 임포트
try:
    from modules.radar_shadow import check_line_of_sight
except ImportError:
    def check_line_of_sight(*args, **kwargs): return True

LAT_TO_KM = 110.57

class XAIUtils:
    """XAI 관련 기능"""
    
    @staticmethod
    def calculate_risk_score(lat: float, lon: float, threats: List[dict], margin: float, terrain_loader=None) -> float:
        """
        특정 위치의 위험도 점수 계산 (음영 고려)
        """
        if not threats:
            return 0.0
        
        max_risk = 0.0
        
        # 지형 고도 가져오기 (음영 계산용)
        my_alt = 500 # 기본 비행 고도 가정 (히트맵용)
        if terrain_loader:
            try:
                my_alt = terrain_loader.get_elevation(lat, lon) + 200 # AGL 200m 가정
            except:
                pass

        for t in threats:
            # SAM 또는 RADAR
            if t['type'] in ["SAM", "RADAR"]:
                dist_km = math.sqrt(
                    ((lat - t['lat']) * LAT_TO_KM) ** 2 + 
                    ((lon - t['lon']) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
                )
                
                threat_radius = t['radius_km']
                
                # 1. 거리상 안전하면 패스
                if dist_km >= threat_radius + margin:
                    continue

                # 2. [시각화 핵심] 레이더 음영 체크
                # terrain_loader가 전달된 경우에만 정밀 계산
                if terrain_loader:
                    threat_alt = t.get('alt', 0)
                    if threat_alt == 0: # 고도 없으면 대략 지형+10m
                        try:
                            threat_alt = terrain_loader.get_elevation(t['lat'], t['lon']) + 10
                        except:
                            threat_alt = 100
                    
                    # 가시선 체크 (보이면 True, 가려지면 False)
                    is_visible = check_line_of_sight(
                        radar_pos=(t['lat'], t['lon'], threat_alt),
                        aircraft_pos=(lat, lon, my_alt),
                        terrain_loader=terrain_loader,
                        samples=5 # 히트맵은 속도를 위해 샘플링 줄임
                    )
                    
                    if not is_visible:
                        # 산에 가려짐 -> 위험도 0 (안전)
                        continue

                # 3. 위험도 산출 (거리 비례)
                if dist_km < threat_radius:
                    risk = 1.0 # 반경 내
                else:
                    # 마진 구역 (1.0 -> 0.0 점진적 감소)
                    risk = 1.0 - (dist_km - threat_radius) / margin
                
                max_risk = max(max_risk, risk)
            
            # NFZ
            elif t['type'] == "NFZ":
                margin_deg = margin / LAT_TO_KM
                if ((t['lat_min'] - margin_deg <= lat <= t['lat_max'] + margin_deg) and
                    (t['lon_min'] - margin_deg <= lon <= t['lon_max'] + margin_deg)):
                    if (t['lat_min'] <= lat <= t['lat_max'] and 
                        t['lon_min'] <= lon <= t['lon_max']):
                        risk = 1.0
                    else:
                        risk = 0.5
                    max_risk = max(max_risk, risk)
        
        return min(max_risk, 1.0)
    
    @staticmethod
    def generate_heatmap_data(threats: List[dict], margin: float, terrain_loader=None) -> List[Tuple[float, float, float]]:
        """
        위협 히트맵 데이터 생성 (지형 로더 받아서 음영 처리)
        """
        min_lat, max_lat = MAP_BOUNDS["min_lat"], MAP_BOUNDS["max_lat"]
        min_lon, max_lon = MAP_BOUNDS["min_lon"], MAP_BOUNDS["max_lon"]
        
        heatmap_data = []
        
        # 해상도 조정 (너무 느리면 HEATMAP_RESOLUTION을 40정도로 낮추세요)
        step_lat = (max_lat - min_lat) / HEATMAP_RESOLUTION
        step_lon = (max_lon - min_lon) / HEATMAP_RESOLUTION
        
        for i in range(HEATMAP_RESOLUTION):
            for j in range(HEATMAP_RESOLUTION):
                lat = min_lat + i * step_lat
                lon = min_lon + j * step_lon
                
                # terrain_loader를 넘겨줘야 음영 계산이 됨
                risk = XAIUtils.calculate_risk_score(lat, lon, threats, margin, terrain_loader)
                
                if risk > 0.01:
                    heatmap_data.append([lat, lon, risk])
        
        return heatmap_data
    
    @staticmethod
    def analyze_path_risk(path: List[Tuple[float, float]], threats: List[dict], margin: float) -> Dict:
        # (기존 코드 유지)
        if not path:
            return {"avg_risk": 0, "max_risk": 0, "high_risk_segments": 0, "total_length_km": 0}
        
        risks = []
        total_length = 0.0
        high_risk_count = 0
        
        for i, (lat, lon) in enumerate(path):
            # 경로 분석은 정밀하게 하되, 여기서는 지형 로더 없이 거리 기반으로만 빠르게 (옵션)
            # 필요하면 terrain_loader 추가 가능
            risk = XAIUtils.calculate_risk_score(lat, lon, threats, margin) 
            risks.append(risk)
            
            if risk > 0.7: high_risk_count += 1
            if i > 0:
                dist = math.sqrt(((lat-path[i-1][0])*LAT_TO_KM)**2 + ((lon-path[i-1][1])*LAT_TO_KM*math.cos(math.radians(lat)))**2)
                total_length += dist
        
        return {
            "avg_risk": np.mean(risks) if risks else 0,
            "max_risk": max(risks) if risks else 0,
            "high_risk_segments": high_risk_count,
            "total_length_km": total_length
        }