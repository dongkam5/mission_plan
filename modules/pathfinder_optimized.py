"""
A* 3D Optimized Pathfinder
- TerrainCacheFast 기반 고속 DEM 접근
- Altitude quantization + near-goal termination
- Fully compatible with AStarPathfinder3D interface
"""

import math
import heapq
import numpy as np
from typing import List, Tuple
from modules.config import MAP_BOUNDS, GRID_SIZE, ALTITUDE_MIN, ALTITUDE_MAX, MIN_ALTITUDE_AGL

LAT_TO_KM = 110.57


class TerrainCacheFast:
    """RAM 기반 고속 DEM 캐시"""

    def __init__(self, terrain_loader):
        self.terrain = terrain_loader
        self.cache = terrain_loader.cache
        self.grids = {}
        self.bounds = {}
        self.pixel_size = {}

        for name, ds in terrain_loader.datasets.items():
            data = ds.read(1)  # numpy array 전체 로드
            self.grids[name] = data
            self.bounds[name] = ds.bounds
            self.pixel_size[name] = (
                (ds.bounds.right - ds.bounds.left) / ds.width,
                (ds.bounds.top - ds.bounds.bottom) / ds.height,
            )

        print(f"✅ TerrainCacheFast 초기화 완료 ({len(self.grids)} tiles loaded)")

    def get(self, lat: float, lon: float) -> float:
        """빠른 DEM 조회"""
        key = (round(lat, 3), round(lon, 3))
        if key in self.cache:
            return self.cache[key]

        for name, b in self.bounds.items():
            if b.bottom <= lat <= b.top and b.left <= lon <= b.right:
                pw, ph = self.pixel_size[name]
                col = int((lon - b.left) / pw)
                row = int((b.top - lat) / ph)
                elev = float(self.grids[name][row, col])
                self.cache[key] = elev
                return elev
        self.cache[key] = 100.0
        return 100.0


class AStarPathfinder3DOptimized:
    """고속 A* 3D 경로탐색"""

    def __init__(self, terrain_cache, grid_size: int = GRID_SIZE, altitude_levels: int = 10):
        self.terrain = terrain_cache
        self.grid_size = grid_size
        self.altitude_levels = altitude_levels
        self.bounds = [
            MAP_BOUNDS["min_lat"],
            MAP_BOUNDS["max_lat"],
            MAP_BOUNDS["min_lon"],
            MAP_BOUNDS["max_lon"],
        ]

    # --- 좌표 변환 ---
    def to_grid_3d(self, lat, lon, alt):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        y = int((lat - min_lat) / ((max_lat - min_lat) / self.grid_size))
        x = int((lon - min_lon) / ((max_lon - min_lon) / self.grid_size))
        z = int((alt - ALTITUDE_MIN) / ((ALTITUDE_MAX - ALTITUDE_MIN) / self.altitude_levels))
        return max(0, min(self.grid_size - 1, x)), max(0, min(self.grid_size - 1, y)), max(0, min(self.altitude_levels - 1, z))

    def to_latlonalt(self, x, y, z):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        lat = min_lat + (y * ((max_lat - min_lat) / self.grid_size))
        lon = min_lon + (x * ((max_lon - min_lon) / self.grid_size))
        alt = ALTITUDE_MIN + (z * ((ALTITUDE_MAX - ALTITUDE_MIN) / self.altitude_levels))
        return lat, lon, alt

    # --- 유틸 ---
    def heuristic(self, a, b, alt_weight=0.1):
        dx, dy, dz = abs(a[0]-b[0]), abs(a[1]-b[1]), abs(a[2]-b[2])
        return math.sqrt(dx**2 + dy**2 + (dz * alt_weight)**2)

    def distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def reconstruct(self, came_from, current):
        path = []
        while current in came_from:
            path.append(self.to_latlonalt(*current))
            current = came_from[current]
        return path[::-1]

    # --- 충돌 검사 ---
    def is_collision_3d(self, lat, lon, alt, threats, margin):
        margin_deg = margin / LAT_TO_KM
        for t in threats:
            if t["type"] == "SAM":
                dist_km = math.sqrt(
                    ((lat - t["lat"]) * LAT_TO_KM) ** 2
                    + ((lon - t["lon"]) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
                )
                if dist_km < (t["radius_km"] + margin) and alt < 5000:
                    threat_effectiveness = max(0, 1 - alt / 5000)
                    if threat_effectiveness > 0.3:
                        return True
            elif t["type"] == "NFZ":
                if ((t["lat_min"] - margin_deg <= lat <= t["lat_max"] + margin_deg) and
                    (t["lon_min"] - margin_deg <= lon <= t["lon_max"] + margin_deg)):
                    return True
        return False

    # --- 메인 경로탐색 ---
    def find_path_3d_fast(self, start, end, threats, safety_margin):
        start_grid = self.to_grid_3d(*start)
        end_grid = self.to_grid_3d(*end)
        open_set = [(0, start_grid)]
        came_from, g_score = {}, {start_grid: 0}
        directions = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==dy==dz==0)]

        nodes_explored = 0
        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1
            if nodes_explored % 1000 == 0:
                print(f"🔹 탐색 진행중: {nodes_explored} nodes")

            # 근접 도달 허용
            if self.distance(current, end_grid) < 2:
                print(f"✅ 목표 근접 도달 ({nodes_explored} nodes explored)")
                return self.reconstruct(came_from, current)

            for dx, dy, dz in directions:
                neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size and 0 <= neighbor[2] < self.altitude_levels):
                    continue

                n_lat, n_lon, n_alt = self.to_latlonalt(*neighbor)
                elev = self.terrain.get(n_lat, n_lon)
                if n_alt < elev + MIN_ALTITUDE_AGL:
                    continue
                if self.is_collision_3d(n_lat, n_lon, n_alt, threats, safety_margin):
                    continue

                move_cost = math.sqrt(dx**2 + dy**2 + (dz * 0.5)**2)
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, end_grid)
                    heapq.heappush(open_set, (f, neighbor))

            if nodes_explored > 50000:
                print("⚠️ 탐색 제한 초과 (50k nodes)")
                break

        print("❌ 경로 탐색 실패")
        return []
