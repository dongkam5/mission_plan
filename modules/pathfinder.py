"""
경로탐색 엔진 - 안정화 버전 (스무딩 오류 제거 + 저고도 침투 강화)
"""
import math
import heapq
import numpy as np
from modules.config import (
    GRID_SIZE, MAP_BOUNDS, SMOOTHING_FACTOR, 
    ALTITUDE_LEVELS, ALTITUDE_MIN, ALTITUDE_MAX, MIN_ALTITUDE_AGL
)
LAT_TO_KM = 110.57

class AStarPathfinder:
    """2D A* 알고리즘"""
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.bounds = [MAP_BOUNDS["min_lat"], MAP_BOUNDS["max_lat"], MAP_BOUNDS["min_lon"], MAP_BOUNDS["max_lon"]]
    
    def to_grid(self, lat, lon):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon): return -1, -1
        y = int((lat - min_lat) / ((max_lat - min_lat) / self.grid_size))
        x = int((lon - min_lon) / ((max_lon - min_lon) / self.grid_size))
        return max(0, min(self.grid_size-1, x)), max(0, min(self.grid_size-1, y))
    
    def to_latlon(self, x, y):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        lat = min_lat + (y * ((max_lat - min_lat) / self.grid_size))
        lon = min_lon + (x * ((max_lon - min_lon) / self.grid_size))
        return lat, lon

    def is_collision(self, lat, lon, threats, margin):
        for t in threats:
            if t['type'] == "SAM":
                dist_km = math.sqrt(((lat - t['lat']) * LAT_TO_KM) ** 2 + ((lon - t['lon']) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2)
                if dist_km < (t['radius_km'] + margin): return True
            elif t['type'] == "NFZ":
                m = margin / LAT_TO_KM
                if (t['lat_min']-m <= lat <= t['lat_max']+m) and (t['lon_min']-m <= lon <= t['lon_max']+m): return True
        return False

    def find_path(self, start, end, threats, safety_margin):
        start_grid = self.to_grid(*start)
        end_grid = self.to_grid(*end)
        if start_grid == (-1, -1) or end_grid == (-1, -1): return []
        
        open_set = [(0, start_grid)]
        came_from, g_score = {}, {start_grid: 0}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end_grid:
                path = []
                while current in came_from:
                    path.append(self.to_latlon(*current))
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                neighbor = (current[0]+dx, current[1]+dy)
                if not (0<=neighbor[0]<self.grid_size and 0<=neighbor[1]<self.grid_size): continue
                
                n_lat, n_lon = self.to_latlon(*neighbor)
                if self.is_collision(n_lat, n_lon, threats, safety_margin): continue
                
                cost = math.sqrt(dx**2+dy**2)
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = math.sqrt((neighbor[0]-end_grid[0])**2 + (neighbor[1]-end_grid[1])**2)
                    heapq.heappush(open_set, (tentative_g+h, neighbor))
        return []

class AStarPathfinder3D:
    """3D A* 알고리즘"""
    def __init__(self, terrain_loader, grid_size: int = GRID_SIZE, altitude_levels: int = ALTITUDE_LEVELS):
        self.terrain = terrain_loader
        self.grid_size = grid_size
        self.altitude_levels = altitude_levels
        self.bounds = [MAP_BOUNDS["min_lat"], MAP_BOUNDS["max_lat"], MAP_BOUNDS["min_lon"], MAP_BOUNDS["max_lon"]]
        
    def to_grid_3d(self, lat, lon, alt):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return -1, -1, -1
        y = int((lat - min_lat) / ((max_lat - min_lat) / self.grid_size))
        x = int((lon - min_lon) / ((max_lon - min_lon) / self.grid_size))
        z = int((alt - ALTITUDE_MIN) / ((ALTITUDE_MAX - ALTITUDE_MIN) / self.altitude_levels))
        return max(0, min(self.grid_size-1, x)), max(0, min(self.grid_size-1, y)), max(0, min(self.altitude_levels-1, z))
    
    def to_latlonalt(self, x, y, z):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        lat = min_lat + (y * ((max_lat - min_lat) / self.grid_size))
        lon = min_lon + (x * ((max_lon - min_lon) / self.grid_size))
        alt = ALTITUDE_MIN + (z * ((ALTITUDE_MAX - ALTITUDE_MIN) / self.altitude_levels))
        return lat, lon, alt

    def is_collision_3d(self, lat, lon, alt, threats, margin):
            """
            3D 위협 충돌 체크 (현실적 전술 모델 적용)
            - Kill Zone: 중심부는 고도 상관없이 위험
            - Radar Zone: 외곽은 저고도(AGL) 비행 시 회피 가능
            """
            # 현재 위치의 지형 고도 가져오기 (AGL 계산용)
            try:
                terrain_h = self.terrain.get_elevation(lat, lon)
            except:
                terrain_h = 0
                
            agl = alt - terrain_h  # 지상고도 (Above Ground Level)

            for t in threats:
                if t['type'] == "SAM":
                    # 수평 거리 계산
                    dist_km = math.sqrt(((lat - t['lat']) * LAT_TO_KM) ** 2 + 
                                    ((lon - t['lon']) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2)
                    
                    threat_radius = t['radius_km'] + margin
                    
                    # 1. 위협 범위 밖이면 안전
                    if dist_km >= threat_radius:
                        continue

                    # 2. [Kill Zone] SAM 바로 위 (반경의 30% 이내)
                    # 광학 장비나 단거리 미사일 사거리 내이므로 고도 상관없이 무조건 위험
                    kill_zone_radius = t['radius_km'] * 0.3
                    if dist_km < kill_zone_radius:
                        return True # 회피 불가능 구역

                    # 3. [Radar Zone] 외곽 지역 (반경의 30% ~ 100%)
                    # 고도가 낮으면 지형 잡음(Clutter)으로 인해 탐지 안 됨
                    # AGL(지상고) 300m 이하를 "전술적 저고도 침투"로 간주
                    if agl < 300: 
                        return False # 안전 (Terrain Masking 성공)
                    else:
                        return True  # 탐지됨 (고고도 비행)

                elif t['type'] == "NFZ":
                    m = margin / LAT_TO_KM
                    if (t['lat_min']-m <= lat <= t['lat_max']+m) and (t['lon_min']-m <= lon <= t['lon_max']+m): 
                        return True
                        
            return False

    def find_path_3d(self, start, end, threats, safety_margin):
        # 2D 좌표면 지형 고도 + 200m로 시작
        if len(start) == 2: start = [*start, self.terrain.get_elevation(*start) + 200]
        if len(end) == 2: end = [*end, self.terrain.get_elevation(*end) + 200]
        
        start_grid = self.to_grid_3d(*start)
        end_grid = self.to_grid_3d(*end)
        
        if start_grid == (-1, -1, -1) or end_grid == (-1, -1, -1):
            print("❌ 좌표 범위 오류")
            return []
        
        open_set = [(0, start_grid)]
        came_from, g_score = {}, {start_grid: 0}
        
        nodes_explored = 0
        MAX_NODES = 30000 
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1
            if nodes_explored > MAX_NODES: 
                print("⚠️ 탐색 타임아웃")
                break
            
            if current == end_grid:
                path = []
                while current in came_from:
                    path.append(self.to_latlonalt(*current))
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            # 26방향 탐색
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        if dx==0 and dy==0 and dz==0: continue
                        neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
                        
                        if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size and 0 <= neighbor[2] < self.altitude_levels): continue
                        
                        n_lat, n_lon, n_alt = self.to_latlonalt(*neighbor)
                        
                        # 지형 충돌
                        terrain_h = self.terrain.get_elevation(n_lat, n_lon)
                        if n_alt < terrain_h + MIN_ALTITUDE_AGL: continue
                        
                        # 위협 충돌
                        if self.is_collision_3d(n_lat, n_lon, n_alt, threats, safety_margin): continue
                        
                        # [비용 함수 수정] 저고도 비행을 유도하는 비용
                        dist_cost = math.sqrt(dx**2 + dy**2 + (dz * 1.5)**2)
                        
                        # 고도가 낮을수록 비용이 적음 (Terrain Following 유도)
                        altitude_cost = (n_alt / 5000.0) * 1.0 
                        
                        tentative_g = g_score[current] + dist_cost + altitude_cost
                        
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            # 휴리스틱
                            h = math.sqrt((neighbor[0]-end_grid[0])**2 + (neighbor[1]-end_grid[1])**2 + (neighbor[2]-end_grid[2])**2)
                            heapq.heappush(open_set, (tentative_g + h, neighbor))
        
        return []

# [중요] 스무딩 함수 간소화 (경로 튀는 문제 해결)
def smooth_path(path):
    return path # 2D는 원본 반환 (안정성 우선)

def smooth_path_3d(path):
    """
    3D 경로 단순화 (B-Spline 제거하여 튀는 현상 방지)
    대신 중간 점들을 적당히 솎아내어 부드럽게 보이게 함
    """
    if not path or len(path) < 3: return path
    
    # 단순히 짝수 번째 점만 추출하거나, 원본을 그대로 씁니다.
    # 복잡한 커브 피팅이 경로를 망치고 있으므로 Raw Path를 반환합니다.
    # 필요하다면 추후 이동평균법(Moving Average) 적용 가능
    return path

