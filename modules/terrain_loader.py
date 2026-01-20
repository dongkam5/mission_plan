"""
SRTM 지형 데이터 로더
실제 데이터가 없을 경우 합성 지형 생성
"""
import numpy as np
import os
from modules.config import SRTM_DATA_DIR, MAP_BOUNDS, GRID_SIZE, ALTITUDE_MAX


class TerrainLoader:
    """지형 데이터 로드 및 관리"""
    
    def __init__(self):
        self.elevation_grid = None
        self.bounds = [
            MAP_BOUNDS["min_lat"],
            MAP_BOUNDS["max_lat"],
            MAP_BOUNDS["min_lon"],
            MAP_BOUNDS["max_lon"]
        ]
        self._load_terrain()
    
    def _load_terrain(self):
        """
        지형 데이터 로드
        실제 SRTM 파일이 없으면 합성 데이터 생성
        """
        srtm_file = os.path.join(SRTM_DATA_DIR, "korea_elevation.npy")
        
        if os.path.exists(srtm_file):
            # 실제 SRTM 데이터 로드
            self.elevation_grid = np.load(srtm_file)
            print(f"✅ 지형 데이터 로드: {srtm_file}")
        else:
            # 합성 지형 생성 (연구/테스트용)
            print("⚠️ SRTM 데이터 없음. 합성 지형 생성 중...")
            self.elevation_grid = self._generate_synthetic_terrain()
            
            # 저장 (다음번에 재사용)
            os.makedirs(SRTM_DATA_DIR, exist_ok=True)
            np.save(srtm_file, self.elevation_grid)
            print(f"✅ 합성 지형 저장: {srtm_file}")
    
    def _generate_synthetic_terrain(self):
        """
        합성 지형 생성 (Perlin Noise 스타일)
        한반도 실제 지형과 유사하게 산악 지역 시뮬레이션
        """
        # 기본 평지 (해발 50m)
        terrain = np.ones((GRID_SIZE, GRID_SIZE)) * 50
        
        # 산악 지역 추가 (중북부)
        for _ in range(20):
            center_x = np.random.randint(30, 90)
            center_y = np.random.randint(40, 80)
            radius = np.random.randint(10, 25)
            height = np.random.randint(500, 2000)
            
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        terrain[y, x] += height * (1 - dist / radius) ** 2
        
        # 노이즈 추가 (자연스러운 기복)
        noise = np.random.normal(0, 20, (GRID_SIZE, GRID_SIZE))
        terrain += noise
        
        # 해수면 이하 제거
        terrain = np.maximum(terrain, 0)
        
        # 최대 고도 제한
        terrain = np.minimum(terrain, ALTITUDE_MAX)
        
        return terrain
    
    def get_elevation(self, lat: float, lon: float) -> float:
        """
        특정 위경도의 고도 반환
        
        Args:
            lat, lon: 위경도
            
        Returns:
            고도 (m)
        """
        min_lat, max_lat, min_lon, max_lon = self.bounds
        
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return 0.0
        
        # 위경도 → 그리드 인덱스 변환
        y = int((lat - min_lat) / ((max_lat - min_lat) / GRID_SIZE))
        x = int((lon - min_lon) / ((max_lon - min_lon) / GRID_SIZE))
        
        y = max(0, min(GRID_SIZE - 1, y))
        x = max(0, min(GRID_SIZE - 1, x))
        
        return float(self.elevation_grid[y, x])
    
    def get_elevation_grid(self):
        """전체 고도 그리드 반환"""
        return self.elevation_grid
