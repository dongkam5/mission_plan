"""
중앙 설정 파일 - 3D 및 XAI 확장
"""

# LLM 설정
LLM_MODEL = "llama3.1"
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 30
LLM_ENABLE_REASONING = True  # XAI 설명 활성화

# 맵 설정 (2D/3D 공통)
GRID_SIZE = 120
MAP_BOUNDS = {
    "min_lat": 33.0,
    "max_lat": 43.0,
    "min_lon": 124.0,
    "max_lon": 132.0
}

# 3D 설정 (NEW)
ENABLE_3D = True  # 3D 모드 활성화
ALTITUDE_LEVELS = 5  # 고도 레벨 수
ALTITUDE_MIN = 0  # 최소 고도 (m)
ALTITUDE_MAX = 5000  # 최대 고도 (m)
TERRAIN_FOLLOWING = True  # 지형 추종 모드
MIN_ALTITUDE_AGL = 250  # 최소 지상고도 (m, Above Ground Level)

# SRTM 데이터 경로
SRTM_DATA_DIR = "data/elevation"
SRTM_RESOLUTION = 90  # 해상도 (m)

# 경로 설정
DEFAULT_SAFETY_MARGIN = 5.0  # km
DEFAULT_STPT_GAP = 10
SMOOTHING_FACTOR = 0.0002

# XAI 설정 (NEW)
ENABLE_HEATMAP = True  # 위협 히트맵 표시
HEATMAP_RESOLUTION = 50  # 히트맵 grid 해상도
RISK_THRESHOLD_HIGH = 0.7  # 고위험 임계값
RISK_THRESHOLD_MEDIUM = 0.4  # 중위험 임계값

# 공항 데이터베이스 (고도 정보 추가)
AIRPORTS = {
    "서산(Seosan)": {"coords": [36.776, 126.493], "elevation": 20},
    "오산(Osan)": {"coords": [37.090, 127.030], "elevation": 38},
    "원주(Wonju)": {"coords": [37.342, 127.920], "elevation": 90},
    "강릉(Gangneung)": {"coords": [37.751, 128.876], "elevation": 6},
    "충주(Chungju)": {"coords": [36.991, 127.926], "elevation": 114},
    "청주(Cheongju)": {"coords": [36.642, 127.489], "elevation": 57},
    "대구(Daegu)": {"coords": [35.871, 128.601], "elevation": 35},
    "광주(Gwangju)": {"coords": [35.159, 126.852], "elevation": 39},
    "부산(Busan)": {"coords": [35.179, 129.075], "elevation": 3},
    "수원(Suwon)": {"coords": [37.240, 127.000], "elevation": 25},
    "사천(Sacheon)": {"coords": [35.088, 128.070], "elevation": 5},
    "서울(Seoul)": {"coords": [37.463, 126.924], "elevation": 48}
}

# UI 설정
MAP_CENTER = [38.0, 128.0]
MAP_ZOOM = 6
CHAT_CONTAINER_HEIGHT = 350

# 알고리즘 선택
AVAILABLE_ALGORITHMS = ["A*", "A* 3D", "RRT", "RRT*"]
DEFAULT_ALGORITHM = "A*"

# 로깅
LOG_DIR = "logs"
ENABLE_LOGGING = True
