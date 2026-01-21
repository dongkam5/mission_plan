# test_terrain.py (전체 테스트)
from modules.terrain_loader import TerrainLoader

def test_korea_elevations():
    """한국 주요 지점 고도 테스트"""
    terrain = TerrainLoader()
    
    test_points = [
        ("서울", 37.5665, 126.9780, 40, 30),
        ("부산", 35.1796, 129.0756, 20, 30),
        ("대구", 35.8714, 128.6014, 50, 50),  
        ("인천", 37.4563, 126.7052, 40, 40), 
        ("대전", 36.3504, 127.3845, 80, 50),
        ("광주", 35.1595, 126.8526, 30, 50),  
        ("강릉", 37.7519, 128.8761, 100, 100), 
        ("설악산", 38.1197, 128.4656, 1700, 200),  
        ("한라산", 33.3617, 126.5292, 1950, 200)
    ]
    
    print("=" * 70)
    print("한국 지형 고도 테스트")
    print("=" * 70)
    
    passed = 0
    for name, lat, lon, expected, tolerance in test_points:
        actual = terrain.get_elevation(lat, lon)
        error = abs(actual - expected)
        
        if error <= tolerance:
            status = "✅"
            passed += 1
        else:
            status = "❌"
        
        print(f"{status} {name:8s}: 예상={expected:4.0f}m, "
              f"실제={actual:6.1f}m, 오차={error:5.1f}m")
    
    print("=" * 70)
    print(f"결과: {passed}/{len(test_points)} 통과")
    return passed == len(test_points)

if __name__ == "__main__":
    test_korea_elevations()
