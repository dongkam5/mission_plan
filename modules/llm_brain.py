"""
LLM Brain 모듈 - XAI reasoning 강화
"""
import ollama
import json
from typing import Dict
from modules.config import LLM_MODEL, LLM_TEMPERATURE, AIRPORTS, LLM_ENABLE_REASONING


class LLMBrain:
    """LLM 인터페이스 - XAI 강화 버전"""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model = model_name
        self.temperature = LLM_TEMPERATURE
        
    def parse_tactical_command(self, user_msg: str, current_state: Dict, path_analysis: Dict = None) -> Dict:
        """
        자연어 명령 파싱 (XAI reasoning 포함)
        
        Args:
            user_msg: 사용자 입력
            current_state: 현재 미션 상태
            path_analysis: 현재 경로 분석 결과 (XAI 설명용)
        """
        state_desc = (
            f"Margin: {current_state['margin']}km, "
            f"RTB: {current_state['rtb']}, "
            f"Waypoint: {current_state['waypoint']}, "
            f"STPT_Gap: {current_state['stpt_gap']}, "
            f"Algorithm: {current_state.get('algorithm', 'A*')}, "
            f"3D Mode: {current_state.get('enable_3d', False)}"
        )
        
        # 경로 분석 정보 추가 (XAI용)
        path_info = ""
        if path_analysis:
            path_info = f"""
Current Path Analysis:
- Average Risk: {path_analysis.get('avg_risk', 0):.2f}
- Max Risk: {path_analysis.get('max_risk', 0):.2f}
- High Risk Segments: {path_analysis.get('high_risk_segments', 0)}
- Total Length: {path_analysis.get('total_length_km', 0):.1f} km
"""
        
        reasoning_instruction = ""
        if LLM_ENABLE_REASONING:
            reasoning_instruction = """
**CRITICAL: Provide detailed reasoning in Korean**
You MUST include a "reasoning" field explaining:
1. WHY you made this decision
2. WHAT factors you considered (threats, distance, risk, etc.)
3. HOW this improves the mission safety/efficiency
"""
        
        system_prompt = f"""
You are a Tactical Mission Planning AI with Explainable AI capabilities.

Current State: {state_desc}
{path_info}

Available Actions:
1. Safety Margin: Adjust 'safety_margin_km' (float, 0.0~50.0)
2. RTB: Set 'rtb' (bool)
3. Waypoint: Set 'waypoint_name' (must be in {list(AIRPORTS.keys())} or null)
4. STPT Gap: Adjust 'stpt_gap' (int, 1~50). Higher = FEWER points
5. Algorithm: Set 'algorithm' (one of: "A*", "A* 3D", "RRT", "RRT*")
6. 3D Mode: Set 'enable_3d' (bool)

{reasoning_instruction}

Output JSON:
{{
    "action": "UPDATE" or "CHAT",
    "update_params": {{
        "safety_margin_km": float/null,
        "rtb": bool/null,
        "waypoint_name": string/null,
        "stpt_gap": int/null,
        "algorithm": string/null,
        "enable_3d": bool/null
    }},
    "response_text": "Brief confirmation in Korean",
    "reasoning": "Detailed explanation in Korean (WHY, WHAT, HOW)"
}}
"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_msg}
                ],
                format='json',
                options={'temperature': self.temperature}
            )
            
            result = json.loads(response['message']['content'])
            validated = self._validate_output(result)
            return validated
            
        except Exception as e:
            return {
                "action": "CHAT",
                "response_text": f"❌ AI 오류: {str(e)}",
                "reasoning": "시스템 오류가 발생했습니다.",
                "update_params": {}
            }
    
    def _validate_output(self, result: Dict) -> Dict:
        """출력 검증 (기존과 동일 + algorithm, enable_3d 추가)"""
        params = result.get("update_params", {})
        
        # Safety Margin
        if params.get("safety_margin_km") is not None:
            margin = params["safety_margin_km"]
            params["safety_margin_km"] = max(0.0, min(50.0, margin))
        
        # STPT Gap
        if params.get("stpt_gap") is not None:
            gap = params["stpt_gap"]
            params["stpt_gap"] = max(1, min(50, gap))
        
        # Waypoint
        if params.get("waypoint_name") and params["waypoint_name"] not in AIRPORTS:
            params["waypoint_name"] = None
            result["response_text"] += " (⚠️ 존재하지 않는 공항)"
        
        # Algorithm
        from modules.config import AVAILABLE_ALGORITHMS
        if params.get("algorithm") and params["algorithm"] not in AVAILABLE_ALGORITHMS:
            params["algorithm"] = None
            result["response_text"] += " (⚠️ 지원하지 않는 알고리즘)"
        
        # Reasoning 기본값
        if "reasoning" not in result or not result["reasoning"]:
            result["reasoning"] = "자동 처리되었습니다."
        
        result["update_params"] = params
        return result
