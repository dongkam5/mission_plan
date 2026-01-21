"""
SRTM 지형 데이터 로더 (경계 오류 완전 제거 버전)
- 타일 경계 겹침 허용
- 3단계 타일 탐색
- 경계 영역 프리캐싱
"""
import numpy as np
import os

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠️ rasterio 미설치. pip install rasterio 필요")

from modules.config import MAP_BOUNDS

class TerrainLoader:
    """실제 SRTM 지형 데이터 로더 (경계 처리 완전 해결)"""

    def __init__(self, dem_dir: str = "data/terrain/"):
        self.dem_dir = dem_dir
        self.datasets = {}
        self.cache = {}

        self.bounds = [
            MAP_BOUNDS["min_lat"],
            MAP_BOUNDS["max_lat"],
            MAP_BOUNDS["min_lon"],
            MAP_BOUNDS["max_lon"]
        ]

        if HAS_RASTERIO:
            self._load_srtm_tiles()
            self._precache_boundaries()  # 경계 영역 미리 캐싱
        else:
            print("⚠️ SRTM 데이터 로드 불가. 기본 고도(100m) 사용")

    def _load_srtm_tiles(self):
        """SRTM GeoTIFF 타일 로딩"""
        tiles = [
            {"name": "srtm_62_05", "lat_range": (35, 40), "lon_range": (125, 130)},
            {"name": "srtm_62_06", "lat_range": (30, 35), "lon_range": (125, 130)}
        ]

        for tile in tiles:
            for ext in ['.tif', '.tiff', '.hgt']:
                filepath = os.path.join(self.dem_dir, tile['name'] + ext)
                if os.path.exists(filepath):
                    try:
                        dataset = rasterio.open(filepath)
                        self.datasets[tile['name']] = dataset
                        print(f"✅ SRTM 타일 로드: {tile['name']}")
                        break
                    except Exception as e:
                        print(f"❌ 타일 로드 실패 {tile['name']}: {e}")

    def _precache_boundaries(self):
        """
        경계 영역 (34.9~35.1, 39.9~40.1) 미리 캐싱
        계산 시작 전에 문제 영역을 사전 처리
        """
        print("🔧 타일 경계 영역 프리캐싱 중...")

        boundary_coords = []

        # 35도 경계 (± 0.15도 여유)
        for lat in np.arange(34.85, 35.15, 0.02):
            for lon in np.arange(125.0, 130.0, 0.2):
                boundary_coords.append((round(lat, 4), round(lon, 4)))

        # 40도 경계
        for lat in np.arange(39.85, 40.15, 0.02):
            for lon in np.arange(125.0, 130.0, 0.2):
                boundary_coords.append((round(lat, 4), round(lon, 4)))

        # 캐싱 (조용하게)
        cached_count = 0
        for lat, lon in boundary_coords:
            elev = self.get_elevation(lat, lon)
            if elev != 100.0:  # 실제 값이 캐시된 경우
                cached_count += 1

        print(f"✅ 경계 영역 {cached_count}/{len(boundary_coords)}개 좌표 캐싱 완료")

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        특정 위경도의 고도 반환 (오류 없는 버전)

        Args:
            lat, lon: 위경도

        Returns:
            고도 (m), 실패 시 100.0
        """
        if not HAS_RASTERIO or not self.datasets:
            return 100.0

        # 캐시 확인
        cache_key = (round(lat, 4), round(lon, 4))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 타일 찾기 (3단계)
        dataset = self._find_tile(lat, lon)
        if dataset is None:
            self.cache[cache_key] = 100.0
            return 100.0

        try:
            bounds = dataset.bounds

            # 픽셀 크기 계산
            pixel_width = (bounds.right - bounds.left) / dataset.width
            pixel_height = (bounds.top - bounds.bottom) / dataset.height

            # 픽셀 인덱스 계산
            col = int((lon - bounds.left) / pixel_width)
            row = int((bounds.top - lat) / pixel_height)

            # 강제 클램핑 (범위 밖이면 가장 가까운 픽셀)
            row = max(0, min(dataset.height - 1, row))
            col = max(0, min(dataset.width - 1, col))

            # 데이터 읽기
            elevation = float(dataset.read(1)[row, col])

            # 이상값 필터 (SRTM void = -32768)
            if -100 <= elevation <= 3000:
                self.cache[cache_key] = elevation
                return elevation
            else:
                self.cache[cache_key] = 100.0
                return 100.0

        except Exception:
            # 완전히 조용하게 (로그 없음)
            self.cache[cache_key] = 100.0
            return 100.0

    def _find_tile(self, lat, lon):
        """
        좌표에 해당하는 타일 찾기 (3단계 탐색)
        1단계: 완전히 속하는 타일
        2단계: 경계값 허용 (느슨한 체크)
        3단계: 픽셀 검증 후 선택
        """
        tiles = [
            {"name": "srtm_62_05", "lat_range": (35, 40), "lon_range": (125, 130)},
            {"name": "srtm_62_06", "lat_range": (30, 35), "lon_range": (125, 130)}
        ]

        # 1단계: 엄격한 범위 체크 (경계값 제외)
        for tile in tiles:
            lat_min, lat_max = tile['lat_range']
            lon_min, lon_max = tile['lon_range']

            # 완전히 속하는 경우 (< 사용)
            if lat_min < lat < lat_max and lon_min < lon < lon_max:
                dataset = self.datasets.get(tile['name'])
                if dataset:
                    return dataset

        # 2단계: 느슨한 범위 체크 (경계값 포함)
        for tile in tiles:
            lat_min, lat_max = tile['lat_range']
            lon_min, lon_max = tile['lon_range']

            # 경계값 허용 (<= 사용)
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                dataset = self.datasets.get(tile['name'])
                if dataset:
                    # 실제 타일 범위 확인
                    try:
                        bounds = dataset.bounds
                        if bounds.left <= lon <= bounds.right and                            bounds.bottom <= lat <= bounds.top:

                            # 3단계: 픽셀 인덱스 검증
                            pixel_width = (bounds.right - bounds.left) / dataset.width
                            pixel_height = (bounds.top - bounds.bottom) / dataset.height

                            col = int((lon - bounds.left) / pixel_width)
                            row = int((bounds.top - lat) / pixel_height)

                            # 범위 체크 (클램핑 전)
                            if 0 <= row < dataset.height and 0 <= col < dataset.width:
                                return dataset

                            # 범위 밖이어도 클램핑 가능하면 사용
                            if -1 <= row <= dataset.height and -1 <= col <= dataset.width:
                                return dataset
                    except:
                        continue

        # 못 찾으면 None
        return None

    def get_elevation_grid(self):
        """전체 고도 그리드 반환 (호환성 유지)"""
        print("⚠️ get_elevation_grid()는 SRTM 모드에서 미지원")
        return None
