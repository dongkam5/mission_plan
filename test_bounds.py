# test_tile_bounds.py (새 파일)
import rasterio

# 타일 1
with rasterio.open("data/terrain/srtm_62_05.tif") as src:
    print("srtm_62_05 범위:", src.bounds)
    print("  left (경도 min):", src.bounds.left)
    print("  right (경도 max):", src.bounds.right)
    print("  bottom (위도 min):", src.bounds.bottom)
    print("  top (위도 max):", src.bounds.top)

print()

# 타일 2
with rasterio.open("data/terrain/srtm_62_06.tif") as src:
    print("srtm_62_06 범위:", src.bounds)
    print("  left (경도 min):", src.bounds.left)
    print("  right (경도 max):", src.bounds.right)
    print("  bottom (위도 min):", src.bounds.bottom)
    print("  top (위도 max):", src.bounds.top)
