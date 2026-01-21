from modules.terrain_loader import TerrainLoader
from modules.pathfinder import AStarPathfinder3D

terrain = TerrainLoader()
pathfinder = AStarPathfinder3D(terrain)

# 서울 → 대전 (짧은 거리)
start = [37.5665, 126.9780, 800]
end = [36.3504, 127.3845, 800]
threats = []

path = pathfinder.find_path_3d(start, end, threats, 5.0)
print(f"경로 포인트: {len(path) if path else 0}")
if path:
    print(f"첫 포인트: {path[0]}")
    print(f"끝 포인트: {path[-1]}")