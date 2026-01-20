"""
미션 상태 관리 - 3D 파라미터 추가
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json
from datetime import datetime
from modules.config import DEFAULT_SAFETY_MARGIN, DEFAULT_STPT_GAP, LOG_DIR, ENABLE_LOGGING, DEFAULT_ALGORITHM, ENABLE_3D
import os


@dataclass
class MissionParams:
    """미션 파라미터 (3D 확장)"""
    start: str = "부산(Busan)"
    target_lat: float = 39.000
    target_lon: float = 125.700
    target_name: str = "PY-Core"
    rtb: bool = True
    margin: float = DEFAULT_SAFETY_MARGIN
    waypoint: Optional[str] = None
    stpt_gap: int = DEFAULT_STPT_GAP
    algorithm: str = DEFAULT_ALGORITHM  # NEW
    enable_3d: bool = ENABLE_3D  # NEW
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class Threat:
    """위협 객체"""
    name: str
    type: str  # "SAM" or "NFZ"
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_km: Optional[float] = None
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class MissionState:
    """미션 전체 상태 관리"""
    
    def __init__(self):
        self.params = MissionParams()
        self.threats: List[Threat] = [
            Threat(name="Default SAM", type="SAM", lat=37.200, lon=127.800, radius_km=20)
        ]
        self.chat_history: List[Dict[str, str]] = [
            {"role": "assistant", "content": "작전관님, 명령을 대기 중입니다.", "reasoning": ""}
        ]
        self.click_mode = False
        self.last_clicked = None
        
    def add_threat(self, threat: Threat):
        """위협 추가"""
        self.threats.append(threat)
        
    def remove_threat(self, name: str):
        """위협 삭제"""
        self.threats = [t for t in self.threats if t.name != name]
        
    def add_chat_message(self, role: str, content: str, reasoning: str = ""):
        """채팅 메시지 추가 (reasoning 포함)"""
        self.chat_history.append({"role": role, "content": content, "reasoning": reasoning})
        
    def save_to_file(self, filename: str):
        """상태 저장"""
        if ENABLE_LOGGING:
            os.makedirs(LOG_DIR, exist_ok=True)
            filepath = os.path.join(LOG_DIR, filename)
            data = {
                "timestamp": datetime.now().isoformat(),
                "params": self.params.to_dict(),
                "threats": [t.to_dict() for t in self.threats],
                "chat_history": self.chat_history
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str):
        """저장된 상태 복원"""
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = cls()
        state.params = MissionParams.from_dict(data["params"])
        state.threats = [Threat.from_dict(t) for t in data["threats"]]
        state.chat_history = data["chat_history"]
        return state
