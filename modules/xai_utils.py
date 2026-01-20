"""
XAI (Explainable AI) 유틸리티
위협 히트맵, 위험도 계산, 설명 생성
"""
import numpy as np
import math
from typing import List, Dict, Tuple
from modules.config import HEATMAP_RESOLUTION, GRID_SIZE, MAP_BOUNDS
LAT_TO_KM=110.57

class XAIUtils:
    """XAI 관련 기능"""
    
    @staticmethod
    def calculate_risk_score(lat: float, lon: float, threats: List[dict], margin: float) -> float:
        """
        특정 위치의 위험도 점수 계산
        
        Args:
            lat, lon: 위경도
            threats: 위협 리스트
            margin: 안전 마진
            
        Returns:
            위험도 (0.0 ~ 1.0)
        """
        if not threats:
            return 0.0
        
        max_risk = 0.0
        
        for t in threats:
            if t['type'] == "SAM":
                # SAM 중심으로부터 거리 (km)
                dist_km = math.sqrt(
                    ((lat - t['lat']) * LAT_TO_KM) ** 2 + 
                    ((lon - t['lon']) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
                )
                
                # 위협 반경 내부: 위험도 1.0
                # 위협 반경 ~ (반경 + 마진): 점진적 감소
                threat_radius = t['radius_km']
                if dist_km < threat_radius:
                    risk = 1.0
                elif dist_km < threat_radius + margin:
                    risk = 1.0 - (dist_km - threat_radius) / margin
                else:
                    risk = 0.0
                
                max_risk = max(max_risk, risk)
            
            elif t['type'] == "NFZ":
                margin_deg = margin / LAT_TO_KM
                if ((t['lat_min'] - margin_deg <= lat <= t['lat_max'] + margin_deg) and
                    (t['lon_min'] - margin_deg <= lon <= t['lon_max'] + margin_deg)):
                    # NFZ 내부
                    if (t['lat_min'] <= lat <= t['lat_max'] and 
                        t['lon_min'] <= lon <= t['lon_max']):
                        risk = 1.0
                    else:
                        # 마진 영역
                        risk = 0.5
                    max_risk = max(max_risk, risk)
        
        return min(max_risk, 1.0)
    
    @staticmethod
    def generate_heatmap_data(threats: List[dict], margin: float) -> List[Tuple[float, float, float]]:
        """
        위협 히트맵 데이터 생성
        
        Returns:
            [(lat, lon, risk_score), ...]
        """
        min_lat, max_lat = MAP_BOUNDS["min_lat"], MAP_BOUNDS["max_lat"]
        min_lon, max_lon = MAP_BOUNDS["min_lon"], MAP_BOUNDS["max_lon"]
        
        heatmap_data = []
        
        for i in range(HEATMAP_RESOLUTION):
            for j in range(HEATMAP_RESOLUTION):
                lat = min_lat + (i / HEATMAP_RESOLUTION) * (max_lat - min_lat)
                lon = min_lon + (j / HEATMAP_RESOLUTION) * (max_lon - min_lon)
                
                risk = XAIUtils.calculate_risk_score(lat, lon, threats, margin)
                
                # 위험도가 0보다 큰 지점만 포함 (성능 최적화)
                if risk > 0.01:
                    heatmap_data.append([lat, lon, risk])
        
        return heatmap_data
    
    @staticmethod
    def analyze_path_risk(path: List[Tuple[float, float]], threats: List[dict], margin: float) -> Dict:
        """
        경로의 위험도 분석
        
        Returns:
            {
                "avg_risk": 평균 위험도,
                "max_risk": 최대 위험도,
                "high_risk_segments": 고위험 구간 수,
                "total_length_km": 총 경로 길이
            }
        """
        if not path:
            return {"avg_risk": 0, "max_risk": 0, "high_risk_segments": 0, "total_length_km": 0}
        
        risks = []
        total_length = 0.0
        high_risk_count = 0
        
        for i, (lat, lon) in enumerate(path):
            risk = XAIUtils.calculate_risk_score(lat, lon, threats, margin)
            risks.append(risk)
            
            if risk > 0.7:
                high_risk_count += 1
            
            if i > 0:
                prev_lat, prev_lon = path[i-1]
                segment_length = math.sqrt(
                    ((lat - prev_lat) * LAT_TO_KM) ** 2 + 
                    ((lon - prev_lon) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
                )
                total_length += segment_length
        
        return {
            "avg_risk": np.mean(risks) if risks else 0,
            "max_risk": max(risks) if risks else 0,
            "high_risk_segments": high_risk_count,
            "total_length_km": total_length
        }
