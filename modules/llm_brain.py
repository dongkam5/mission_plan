"""
LLM Brain 모듈 - Pydantic 기반 구조화된 출력
"""
import ollama
from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from modules.config import LLM_MODEL, LLM_TEMPERATURE, AIRPORTS

# --- Pydantic 스키마 정의 ---
class MissionUpdateParams(BaseModel):
    safety_margin_km: Optional[float] = Field(None, ge=0.0, le=50.0, description="안전 마진(km)")
    rtb: Optional[bool] = Field(None, description="복귀(Return To Base) 여부")
    waypoint_name: Optional[str] = Field(None, description="경유할 공항 이름")
    stpt_gap: Optional[int] = Field(None, ge=1, le=50, description="STPT 표시 간격")
    algorithm: Optional[Literal["A*", "A* 3D", "RRT", "RRT*"]] = Field(None, description="알고리즘")
    enable_3d: Optional[bool] = Field(None, description="3D 지형 고려 여부")

class LLMResponse(BaseModel):
    action: Literal["UPDATE", "CHAT"]
    update_params: MissionUpdateParams
    response_text: str
    reasoning: str = Field(..., description="AI의 판단 근거 (Why, What, How)")

class LLMBrain:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model = model_name
        self.temperature = LLM_TEMPERATURE
        
    def parse_tactical_command(self, user_msg: str, current_state: dict, path_analysis: dict = None) -> dict:
        state_desc = (
            f"Margin: {current_state['margin']}km, "
            f"RTB: {current_state['rtb']}, "
            f"Algorithm: {current_state.get('algorithm', 'A*')}, "
            f"3D: {current_state.get('enable_3d', False)}"
        )
        
        path_info = ""
        if path_analysis:
            path_info = f"Path Risk: Max {path_analysis.get('max_risk', 0):.2f}"

        system_prompt = f"""
You are an intelligent Mission Planning AI.
Current State: {state_desc}
{path_info}

Analyze the command. Output MUST be JSON matching the schema.
Reasoning MUST be in Korean.
Available Airports: {list(AIRPORTS.keys())}
"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_msg}
                ],
                format=LLMResponse.model_json_schema(), # 구조 강제
                options={'temperature': self.temperature}
            )
            
            # Pydantic 파싱
            validated = LLMResponse.model_validate_json(response['message']['content'])
            result = validated.model_dump()
            
            # 공항 유효성 2차 검증
            wp = result['update_params'].get('waypoint_name')
            if wp and wp not in AIRPORTS:
                result['update_params']['waypoint_name'] = None
                result['response_text'] += f" (⚠️ '{wp}' 공항 없음)"
                
            return result
            
        except Exception as e:
            return {
                "action": "CHAT",
                "response_text": "명령 처리 중 오류가 발생했습니다.",
                "reasoning": f"시스템 오류: {str(e)}",
                "update_params": {}
            }