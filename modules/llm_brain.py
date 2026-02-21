"""
LLM Brain 모듈 v2.0 - 군사 전문 참모 AI
- 모델: qwen3:14b (ollama)
- 자연어 전술 명령 → 문맥 이해 → 파라미터 자동 조정
- 교리 근거: JP3-30, AFDP3-03, AFDP5-0, FMI3-04.155
"""
import ollama
import json
import re
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, ValidationError
from modules.config import (
    LLM_MODEL, LLM_MODEL_FALLBACK, LLM_TEMPERATURE, LLM_TIMEOUT,
    AIRPORTS, MISSION_TYPES, MARGIN_LEVELS, THREAT_DEFAULT_RADIUS
)


# ================================================================
# Pydantic 스키마 - 확장된 구조
# ================================================================

class ThreatInfo(BaseModel):
    """LLM이 명령에서 추출한 위협 정보"""
    name: str = Field(..., description="위협 명칭 (예: Enemy-RADAR-01)")
    type: Literal["SAM", "RADAR", "NFZ"] = Field(..., description="위협 유형")
    lat: Optional[float] = Field(None, description="위협 위도")
    lon: Optional[float] = Field(None, description="위협 경도")
    radius_km: Optional[float] = Field(None, description="위협 반경(km)")


class MissionUpdateParams(BaseModel):
    """미션 파라미터 변경 내용"""
    safety_margin_km: Optional[float] = Field(None, ge=0.0, le=50.0, description="안전 마진(km)")
    rtb: Optional[bool] = Field(None, description="복귀(Return To Base) 여부")
    waypoint_name: Optional[str] = Field(None, description="경유할 공항 이름")
    stpt_gap: Optional[int] = Field(None, ge=1, le=50, description="STPT 표시 간격")
    algorithm: Optional[Literal["A*", "A* 3D", "RRT", "RRT*"]] = Field(None, description="알고리즘")
    enable_3d: Optional[bool] = Field(None, description="3D 지형 고려 여부")
    target_lat: Optional[float] = Field(None, ge=33.0, le=43.0, description="목표 위도")
    target_lon: Optional[float] = Field(None, ge=124.0, le=132.0, description="목표 경도")
    target_name: Optional[str] = Field(None, description="목표 명칭")
    start: Optional[str] = Field(None, description="출발 기지명")


class LLMResponse(BaseModel):
    """LLM 응답 전체 구조"""
    action: Literal["UPDATE", "THREAT_ADD", "MISSION_PLAN", "EXPLAIN", "CHAT"] = Field(
        ...,
        description=(
            "UPDATE: 파라미터만 변경 | "
            "THREAT_ADD: 위협 추가 + 파라미터 변경 | "
            "MISSION_PLAN: 임무 유형 분류 및 순서 결정 | "
            "EXPLAIN: 교리/전술 설명 | "
            "CHAT: 일반 대화"
        )
    )
    update_params: MissionUpdateParams = Field(
        default_factory=MissionUpdateParams,
        description="변경할 파라미터 (없으면 모두 null)"
    )
    threats_to_add: List[ThreatInfo] = Field(
        default_factory=list,
        description="추가할 위협 목록 (THREAT_ADD 액션 시 사용)"
    )
    mission_sequence: List[str] = Field(
        default_factory=list,
        description="임무 수행 순서 (예: ['ISR', 'SEAD', 'STRIKE'])"
    )
    response_text: str = Field(..., description="사용자에게 보여줄 한국어 응답")
    reasoning: str = Field(..., description="판단 근거 - Why(왜), What(무엇을), How(어떻게) 형식으로 Korean으로 작성")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="판단 신뢰도 0~1")


# ================================================================
# 시스템 프롬프트 - 교리 기반 군사 전문 참모
# ================================================================

SYSTEM_PROMPT_TEMPLATE = """
You are IMPS-AI, a military mission planning expert AI assistant for Korean Air Force operations.
You act as a tactical staff officer who understands military doctrine and translates commander's intent into mission parameters.

## DOCTRINE BASIS
- JP 3-30 Joint Air Operations: ROE, asset allocation, JPPA 7-step planning
- AFDP 3-03 Counterland: AI(Air Interdiction), CAS, SEAR mission types
- AFDP 5-0 Planning: COA development, constraints/restraints
- FMI 3-04.155 UAS Operations: ISR, surveillance, MUM-T teaming

## MISSION TYPES (execute in this sequence when multiple apply)
1. ISR  - Intelligence/Surveillance/Reconnaissance (정보·감시·정찰) - ALWAYS FIRST
2. SEAD - Suppression of Enemy Air Defenses (적 방공망 제압) - BEFORE STRIKE
3. STRIKE - Air Interdiction Strike (공중 타격)
4. CAS - Close Air Support (근접항공지원)

## CURRENT MISSION STATE
{state_desc}

## AVAILABLE BASES (출발 가능 기지)
{airports}

## SAFETY MARGIN REFERENCE
- 최소(2km): 위험 감수, 속도 우선
- 표준(5km): 기본값
- 안전(15km): 안전 우선 (연료 여유 있을 때)
- 최대(30km): 최대 회피

## TACTICAL INTERPRETATION RULES
- "저고도" / "지형추종" / "레이더 회피" → enable_3d=true, algorithm="A* 3D"
- "안전하게" / "연료 여유" / "우회" → safety_margin_km 증가 (현재값 + 10~15)
- "빠르게" / "직선" → safety_margin_km 감소, algorithm="A*"
- "타격 후 복귀" / "RTB" → rtb=true
- 좌표 언급 시 → target_lat/target_lon 추출
- 위협 언급 시 (레이더, SAM, 방공망) → threats_to_add에 추가

## ACTION SELECTION RULES
- 위협 정보가 언급되면 → "THREAT_ADD"
- 파라미터만 바꾸면 되면 → "UPDATE"  
- 임무 유형/순서 질문이면 → "MISSION_PLAN"
- 교리/전술 설명 요청이면 → "EXPLAIN"
- 그 외 대화 → "CHAT"

## OUTPUT RULES
- response_text: ALWAYS in Korean, friendly and professional tone
- reasoning: ALWAYS in Korean, format "Why: ... / What: ... / How: ..."
- All coordinates must be within Korea bounds (lat 33~43, lon 124~132)
- Never invent coordinates that weren't mentioned
- If threat radius not mentioned, use defaults: SAM=30km, RADAR=80km

{path_info}
"""


# ================================================================
# LLMBrain 클래스
# ================================================================

class LLMBrain:
    # 클래스 레벨 캐시: 동일 명령어 반복 시 재사용
    _response_cache: dict = {}
    _available_model: str = None  # 주모델 생존 여부 캐시

    def __init__(self, model_name: str = LLM_MODEL):
        self.model = model_name
        self.temperature = LLM_TEMPERATURE

    def _build_state_desc(self, current_state: dict) -> str:
        """현재 상태를 LLM이 읽기 좋은 텍스트로 변환"""
        return (
            f"출발기지: {current_state.get('start', '?')} | "
            f"목표: lat={current_state.get('target_lat', '?')}, lon={current_state.get('target_lon', '?')} "
            f"({current_state.get('target_name', '?')}) | "
            f"안전마진: {current_state.get('margin', '?')}km | "
            f"RTB: {current_state.get('rtb', '?')} | "
            f"알고리즘: {current_state.get('algorithm', 'A*')} | "
            f"3D모드: {current_state.get('enable_3d', False)}"
        )

    def _build_airports_desc(self) -> str:
        """공항 목록 텍스트 생성"""
        return ", ".join(AIRPORTS.keys())

    def _try_parse_response(self, raw_content: str) -> dict:
        """
        qwen3의 <think>...</think> 태그 처리 후 JSON 파싱
        qwen3:14b는 thinking mode로 <think> 블록을 먼저 출력할 수 있음
        """
        # <think>...</think> 제거
        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

        # JSON 블록 추출 (```json ... ``` 형식 대응)
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Pydantic 파싱
        validated = LLMResponse.model_validate_json(content)
        return validated.model_dump()

    def parse_tactical_command(
        self,
        user_msg: str,
        current_state: dict,
        path_analysis: dict = None
    ) -> dict:
        """
        자연어 전술 명령 파싱 → 구조화된 응답 반환

        Returns:
            dict with keys:
                action, update_params, threats_to_add,
                mission_sequence, response_text, reasoning, confidence
        """
        import hashlib, json as _json

        # ── 간단 응답 캐시 (동일 명령+상태 조합은 LLM 재호출 스킵) ──
        _state_key = _json.dumps({
            'msg':   user_msg.strip().lower(),
            'state': {k: current_state.get(k) for k in
                      ('target_lat','target_lon','start','algorithm','margin','enable_3d','rtb')}
        }, ensure_ascii=False, sort_keys=True)
        _cache_key = hashlib.md5(_state_key.encode()).hexdigest()
        if _cache_key in LLMBrain._response_cache:
            cached = LLMBrain._response_cache[_cache_key].copy()
            cached['response_text'] = '💾 ' + cached.get('response_text','')  # 캐시 표시
            cached['_model_used']   = 'cache'
            return cached
        state_desc = self._build_state_desc(current_state)
        airports_desc = self._build_airports_desc()

        path_info = ""
        if path_analysis:
            path_info = (
                f"## CURRENT PATH ANALYSIS\n"
                f"Max Risk: {path_analysis.get('max_risk', 0):.2f} | "
                f"Waypoints: {path_analysis.get('waypoint_count', 0)} | "
                f"Distance: {path_analysis.get('total_distance_km', 0):.1f}km"
            )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            state_desc=state_desc,
            airports=airports_desc,
            path_info=path_info
        )

        # 모델 순서를 동적으로 결정 (ollamatags로 사용 가능한 모델 확인)
        def _get_model_list():
            """ollama에서 사용 가능한 모델 리스트 반환"""
            try:
                models_resp = ollama.list()
                return [m['name'] for m in models_resp.get('models', [])]
            except Exception:
                return []

        available_models = _get_model_list()
        # 우선순위: 설정된 주모델 → 작은 빠른 모델 → 폴백
        priority_models = [self.model, LLM_MODEL_FALLBACK]
        # 로컈에 없으면 빠른 모델 시도 순서 구성
        if available_models:
            fast_models = [m for m in available_models
                          if any(tag in m for tag in ['3b','1b','7b','8b','mistral','phi3','gemma2:2b'])]
            if fast_models:
                priority_models = fast_models[:1] + priority_models
        models_to_try = priority_models

        # 모델 순서대로 시도 (빠른 모델 우선 → qwen3:14b → fallback)
        for model in models_to_try:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_msg}
                    ],
                    format=LLMResponse.model_json_schema(),
                    options={
                        'temperature': self.temperature,
                        'num_predict': 512,   # 중요: 2048→2048대비 4배 빠름 (구조화 JSON만 필요하주어 512으로 충분)
                        'num_ctx':    2048,   # 컨텍스트 토큰 제한 (4096대비 2배 빠름)
                    }
                )

                raw_content = response['message']['content']
                result = self._try_parse_response(raw_content)

                # 공항 유효성 검증
                wp = result['update_params'].get('waypoint_name')
                if wp and wp not in AIRPORTS:
                    result['update_params']['waypoint_name'] = None
                    result['response_text'] += f" (⚠️ '{wp}' 기지 없음)"

                start = result['update_params'].get('start')
                if start and start not in AIRPORTS:
                    result['update_params']['start'] = None
                    result['response_text'] += f" (⚠️ '{start}' 기지 없음)"

                # 위협 기본 반경 보정
                for threat in result.get('threats_to_add', []):
                    if threat.get('radius_km') is None:
                        threat['radius_km'] = THREAT_DEFAULT_RADIUS.get(
                            threat.get('type', 'SAM'), 30.0
                        )

                result['_model_used'] = model
                # 성공 응답 캐시 저장 (최대 50개, 오래된 것 제거)
                if len(LLMBrain._response_cache) >= 50:
                    oldest = next(iter(LLMBrain._response_cache))
                    del LLMBrain._response_cache[oldest]
                LLMBrain._response_cache[_cache_key] = result.copy()
                return result

            except Exception as e:
                if model == LLM_MODEL_FALLBACK:
                    # 최종 폴백 응답
                    return self._fallback_response(str(e))
                continue

    def _fallback_response(self, error_msg: str) -> dict:
        """모든 모델 실패 시 안전한 기본 응답"""
        return {
            "action": "CHAT",
            "update_params": {},
            "threats_to_add": [],
            "mission_sequence": [],
            "response_text": "⚠️ AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "reasoning": f"시스템 오류: {error_msg}",
            "confidence": 0.0,
            "_model_used": "fallback"
        }
