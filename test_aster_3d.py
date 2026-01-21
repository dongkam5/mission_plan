# test_astar_3d.py
from modules.terrain_loader import TerrainLoader
from modules.pathfinder import AStarPathfinder3D
import time

def test_3d_path():
    """3D 경로계획 통합 테스트"""
    
    # 지형 로더
    terrain = TerrainLoader()
    
    # 3D 경로계획기
    pathfinder = AStarPathfinder3D(terrain, grid_size=100, altitude_levels=10)
    
    # 서울 → 부산
    start = [37.5665, 126.9780, 800]  # 500m 고도
    end = [35.1796, 129.0756, 800]
    
    # 대구 근처 SAM
    threats = [{
        "type": "SAM",
        "lat": 35.8714,
        "lon": 128.6014,
        "radius_km": 30
    }]
    
    print("=" * 70)
    print("3D 경로계획 테스트 (실제 지형)")
    print("=" * 70)
    print(f"출발: 서울 ({start[0]:.4f}, {start[1]:.4f}, {start[2]}m)")
    print(f"목표: 부산 ({end[0]:.4f}, {end[1]:.4f}, {end[2]}m)")
    print(f"위협: 대구 SAM (반경 30km)")
    print("-" * 70)
    
    start_time = time.time()
    path_3d = pathfinder.find_path_3d(start, end, threats, safety_margin=5.0)
    calc_time = time.time() - start_time
    
    if path_3d:
        print(f"✅ 경로 생성 성공!")
        print(f"   포인트 수: {len(path_3d)}")
        print(f"   계산 시간: {calc_time:.2f}초")
        
        # 고도 분석
        altitudes = [p[2] for p in path_3d]
        print(f"   고도 범위: {min(altitudes):.0f}m ~ {max(altitudes):.0f}m")
        
        # 지형 충돌 체크
        collisions = 0
        for lat, lon, alt in path_3d:
            terrain_elev = terrain.get_elevation(lat, lon)
            if alt < terrain_elev + 200:
                collisions += 1
        
        if collisions > 0:
            print(f"   ⚠️  지형 충돌: {collisions}개 포인트")
        else:
            print(f"   ✅ 지형 안전: 모든 포인트 200m AGL 이상")
        
        return True
    else:
        print(f"❌ 경로 생성 실패 ({calc_time:.2f}초)")
        return False

if __name__ == "__main__":
    test_3d_path()
