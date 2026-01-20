"""
경로탐색 엔진 - 2D/3D A* 통합
"""
import math
import heapq
import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional
from modules.config import GRID_SIZE, MAP_BOUNDS, SMOOTHING_FACTOR, ALTITUDE_LEVELS, ALTITUDE_MIN, ALTITUDE_MAX, MIN_ALTITUDE_AGL
LAT_TO_KM=110.57


class AStarPathfinder:
    """2D A* 알고리즘 (기존)"""
    
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.bounds = [
            MAP_BOUNDS["min_lat"],
            MAP_BOUNDS["max_lat"],
            MAP_BOUNDS["min_lon"],
            MAP_BOUNDS["max_lon"]
        ]
        
    def to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """위경도 → 그리드 좌표 변환"""
        min_lat, max_lat, min_lon, max_lon = self.bounds
        
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return -1, -1
        
        y = int((lat - min_lat) / ((max_lat - min_lat) / self.grid_size))
        x = int((lon - min_lon) / ((max_lon - min_lon) / self.grid_size))
        
        y = max(0, min(self.grid_size - 1, y))
        x = max(0, min(self.grid_size - 1, x))
        
        return x, y
    
    def to_latlon(self, x: int, y: int) -> Tuple[float, float]:
        """그리드 좌표 → 위경도 변환"""
        min_lat, max_lat, min_lon, max_lon = self.bounds
        
        lat = min_lat + (y * ((max_lat - min_lat) / self.grid_size))
        lon = min_lon + (x * ((max_lon - min_lon) / self.grid_size))
        
        return lat, lon
    
    def is_collision(self, lat: float, lon: float, threats: List[dict], margin: float) -> bool:
        """위협 충돌 체크"""
        margin_deg = margin / LAT_TO_KM
        
        for t in threats:
            if t['type'] == "SAM":
                dist_km = math.sqrt(
                    ((lat - t['lat']) * LAT_TO_KM) ** 2 + 
                    ((lon - t['lon']) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
                )
                if dist_km < (t['radius_km'] + margin):
                    return True
                    
            elif t['type'] == "NFZ":
                if ((t['lat_min'] - margin_deg <= lat <= t['lat_max'] + margin_deg) and
                    (t['lon_min'] - margin_deg <= lon <= t['lon_max'] + margin_deg)):
                    return True
        
        return False
    
    def find_path(
        self,
        start: List[float],
        end: List[float],
        threats: List[dict],
        safety_margin: float
    ) -> List[Tuple[float, float]]:
        """2D A* 경로탐색"""
        start_grid = self.to_grid(start[0], start[1])
        end_grid = self.to_grid(end[0], end[1])
        
        if start_grid == (-1, -1) or end_grid == (-1, -1):
            return []
        
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end_grid:
                path = []
                while current in came_from:
                    path.append(self.to_latlon(current[0], current[1]))
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.grid_size and 
                        0 <= neighbor[1] < self.grid_size):
                    continue
                
                n_lat, n_lon = self.to_latlon(neighbor[0], neighbor[1])
                
                if self.is_collision(n_lat, n_lon, threats, safety_margin):
                    continue
                
                move_cost = math.sqrt(dx**2 + dy**2)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    h = math.sqrt(
                        (neighbor[0] - end_grid[0]) ** 2 + 
                        (neighbor[1] - end_grid[1]) ** 2
                    )
                    
                    f_score = tentative_g_score + h
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return []


class AStarPathfinder3D:
    """3D A* 알고리즘 (지형 고려)"""
    
    def __init__(self, terrain_loader, grid_size: int = GRID_SIZE, altitude_levels: int = ALTITUDE_LEVELS):
        self.terrain = terrain_loader
        self.grid_size = grid_size
        self.altitude_levels = altitude_levels
        self.bounds = [
            MAP_BOUNDS["min_lat"],
            MAP_BOUNDS["max_lat"],
            MAP_BOUNDS["min_lon"],
            MAP_BOUNDS["max_lon"]
        ]
        
    def to_grid_3d(self, lat: float, lon: float, alt: float) -> Tuple[int, int, int]:
        """위경도고도 → 3D 그리드 좌표"""
        min_lat, max_lat, min_lon, max_lon = self.bounds
        
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return -1, -1, -1
        
        y = int((lat - min_lat) / ((max_lat - min_lat) / self.grid_size))
        x = int((lon - min_lon) / ((max_lon - min_lon) / self.grid_size))
        z = int((alt - ALTITUDE_MIN) / ((ALTITUDE_MAX - ALTITUDE_MIN) / self.altitude_levels))
        
        y = max(0, min(self.grid_size - 1, y))
        x = max(0, min(self.grid_size - 1, x))
        z = max(0, min(self.altitude_levels - 1, z))
        
        return x, y, z
    
    def to_latlonalt(self, x: int, y: int, z: int) -> Tuple[float, float, float]:
        """3D 그리드 좌표 → 위경도고도"""
        min_lat, max_lat, min_lon, max_lon = self.bounds
        
        lat = min_lat + (y * ((max_lat - min_lat) / self.grid_size))
        lon = min_lon + (x * ((max_lon - min_lon) / self.grid_size))
        alt = ALTITUDE_MIN + (z * ((ALTITUDE_MAX - ALTITUDE_MIN) / self.altitude_levels))
        
        return lat, lon, alt
    
    def is_collision_3d(self, lat: float, lon: float, alt: float, threats: List[dict], margin: float) -> bool:
        """3D 위협 충돌 체크 (고도 고려)"""
        margin_deg = margin / LAT_TO_KM
        
        for t in threats:
            if t['type'] == "SAM":
                # 수평 거리
                dist_km = math.sqrt(
                    ((lat - t['lat']) * LAT_TO_KM) ** 2 + 
                    ((lon - t['lon']) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
                )
                
                # SAM 고도 영향 (간단 모델: 반경 내 + 5km 이하 고도)
                if dist_km < (t['radius_km'] + margin) and alt < 5000:
                    # 고도에 따른 위협 감쇄 (높을수록 안전)
                    threat_effectiveness = max(0, 1 - alt / 5000)
                    if threat_effectiveness > 0.3:  # 30% 이상 위협도면 충돌
                        return True
                    
            elif t['type'] == "NFZ":
                # NFZ는 모든 고도에서 금지
                if ((t['lat_min'] - margin_deg <= lat <= t['lat_max'] + margin_deg) and
                    (t['lon_min'] - margin_deg <= lon <= t['lon_max'] + margin_deg)):
                    return True
        
        return False
    
    def is_terrain_collision(self, lat: float, lon: float, alt: float) -> bool:
        """지형 충돌 체크"""
        terrain_elevation = self.terrain.get_elevation(lat, lon)
        
        # 최소 지상고도(AGL) 유지
        if alt < terrain_elevation + MIN_ALTITUDE_AGL:
            return True
        
        return False
    
    def find_path_3d(
        self,
        start: List[float],  # [lat, lon, alt]
        end: List[float],    # [lat, lon, alt]
        threats: List[dict],
        safety_margin: float
    ) -> List[Tuple[float, float, float]]:
        """3D A* 경로탐색"""
        
        # 시작/끝 지점이 2D인 경우 고도 추가 (지형 + 500m)
        if len(start) == 2:
            start_elev = self.terrain.get_elevation(start[0], start[1])
            start = [start[0], start[1], start_elev + 500]
        
        if len(end) == 2:
            end_elev = self.terrain.get_elevation(end[0], end[1])
            end = [end[0], end[1], end_elev + 500]
        
        start_grid = self.to_grid_3d(start[0], start[1], start[2])
        end_grid = self.to_grid_3d(end[0], end[1], end[2])
        
        if start_grid == (-1, -1, -1) or end_grid == (-1, -1, -1):
            return []
        
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        
        # 26방향 이동 (x±1, y±1, z±1 조합)
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    directions.append((dx, dy, dz))
        
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1
            
            # 탐색 제한 (성능)
            if nodes_explored > 50000:
                print("⚠️ 3D 탐색 시간 초과 (50k 노드)")
                break
            
            if current == end_grid:
                # 경로 복원
                path = []
                while current in came_from:
                    lat, lon, alt = self.to_latlonalt(current[0], current[1], current[2])
                    path.append((lat, lon, alt))
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy, dz in directions:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                
                # 범위 체크
                if not (0 <= neighbor[0] < self.grid_size and 
                        0 <= neighbor[1] < self.grid_size and
                        0 <= neighbor[2] < self.altitude_levels):
                    continue
                
                n_lat, n_lon, n_alt = self.to_latlonalt(neighbor[0], neighbor[1], neighbor[2])
                
                # 위협 충돌
                if self.is_collision_3d(n_lat, n_lon, n_alt, threats, safety_margin):
                    continue
                
                # 지형 충돌
                if self.is_terrain_collision(n_lat, n_lon, n_alt):
                    continue
                
                # 비용 계산
                move_cost = math.sqrt(dx**2 + dy**2 + (dz * 0.5)**2)  # 수직 이동 비용 감소
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # 휴리스틱
                    h = math.sqrt(
                        (neighbor[0] - end_grid[0]) ** 2 + 
                        (neighbor[1] - end_grid[1]) ** 2 +
                        (neighbor[2] - end_grid[2]) ** 2
                    )
                    
                    f_score = tentative_g_score + h
                    heapq.heappush(open_set, (f_score, neighbor))
        
        print(f"⚠️ 3D 경로탐색 실패 ({nodes_explored} 노드 탐색)")
        return []


def smooth_path(path_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """2D 경로 평탄화"""
    if not path_coords or len(path_coords) < 3:
        return path_coords
    
    try:
        lat = [p[0] for p in path_coords]
        lon = [p[1] for p in path_coords]
        
        clean_lat, clean_lon = [], []
        for i in range(len(lat)):
            if i == 0 or (lat[i] != lat[i-1] or lon[i] != lon[i-1]):
                clean_lat.append(lat[i])
                clean_lon.append(lon[i])
        
        if len(clean_lat) < 3:
            return path_coords
        
        tck, u = splprep([clean_lat, clean_lon], s=SMOOTHING_FACTOR, per=False)
        u_new = np.linspace(u.min(), u.max(), len(path_coords) * 5)
        new_lat, new_lon = splev(u_new, tck)
        
        return list(zip(new_lat, new_lon))
        
    except Exception as e:
        print(f"⚠️ 경로 평탄화 실패: {str(e)}")
        return path_coords


def smooth_path_3d(path_coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """3D 경로 평탄화"""
    if not path_coords or len(path_coords) < 3:
        return path_coords
    
    try:
        lat = [p[0] for p in path_coords]
        lon = [p[1] for p in path_coords]
        alt = [p[2] for p in path_coords]
        
        # 중복 제거
        clean_lat, clean_lon, clean_alt = [], [], []
        for i in range(len(lat)):
            if i == 0 or (lat[i] != lat[i-1] or lon[i] != lon[i-1] or alt[i] != alt[i-1]):
                clean_lat.append(lat[i])
                clean_lon.append(lon[i])
                clean_alt.append(alt[i])
        
        if len(clean_lat) < 3:
            return path_coords
        
        tck, u = splprep([clean_lat, clean_lon, clean_alt], s=SMOOTHING_FACTOR * 2, per=False)
        u_new = np.linspace(u.min(), u.max(), len(path_coords) * 5)
        new_lat, new_lon, new_alt = splev(u_new, tck)
        
        return list(zip(new_lat, new_lon, new_alt))
        
    except Exception as e:
        print(f"⚠️ 3D 경로 평탄화 실패: {str(e)}")
        return path_coords
