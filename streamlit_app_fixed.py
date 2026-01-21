"""
통합 임무계획 시스템 v10.0
3D 지형 + XAI + 알고리즘 비교 (A*, A* 3D, RRT, RRT*)
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import pandas as pd
import time

from modules.config import AIRPORTS, MAP_CENTER, MAP_ZOOM, CHAT_CONTAINER_HEIGHT, AVAILABLE_ALGORITHMS
from modules.mission_state import MissionState, Threat
from modules.llm_brain import LLMBrain
from modules.pathfinder import AStarPathfinder, AStarPathfinder3D, smooth_path, smooth_path_3d
from modules.pathfinder_rrt import RRTPathfinder, RRTStarPathfinder
from modules.terrain_loader import TerrainLoader
from modules.xai_utils import XAIUtils

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="IMPS v10.0 (3D+XAI)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🚁 통합 임무계획 시스템 v10.0 (3D + XAI + 알고리즘 비교)")

# ===== 상태 초기화 =====
if "mission" not in st.session_state:
    st.session_state.mission = MissionState()

if "terrain" not in st.session_state:
    with st.spinner("🌍 지형 데이터 로딩 중..."):
        st.session_state.terrain = TerrainLoader()

mission = st.session_state.mission
terrain = st.session_state.terrain

# ===== 레이아웃 =====
col_left, col_right = st.columns([1, 2])

with col_left:
    tab_ops, tab_intel, tab_xai, tab_debug = st.tabs([
        "💬 작전 통제",
        "⚠️ 위협 관리",
        "🤔 AI 판단 근거",
        "🔧 디버그"
    ])

    # ===== 작전 통제 탭 =====
    with tab_ops:
        with st.expander("⚙️ 미션 프로파일", expanded=True):
            p = mission.params

            # 기본 설정
            p.start = st.selectbox(
                "출발 기지",
                list(AIRPORTS.keys()),
                index=list(AIRPORTS.keys()).index(p.start)
            )

            st.caption("🎯 타겟 좌표")
            c1, c2 = st.columns(2)
            p.target_lat = c1.number_input("Lat", 33.0, 43.0, p.target_lat, format="%.4f")
            p.target_lon = c2.number_input("Lon", 124.0, 132.0, p.target_lon, format="%.4f")

            # 경로 설정
            c_rtb, c_algo = st.columns([1, 1])
            p.rtb = c_rtb.checkbox("Strike & RTB", value=p.rtb)
            p.algorithm = c_algo.selectbox("알고리즘", AVAILABLE_ALGORITHMS, index=AVAILABLE_ALGORITHMS.index(p.algorithm))

            p.enable_3d = st.checkbox("3D 모드 (지형 고려)", value=p.enable_3d)
            p.margin = st.slider("안전 마진(km)", 0.0, 50.0, p.margin)
            p.stpt_gap = st.slider("STPT 표시 간격", 1, 50, p.stpt_gap)

            if st.button("🔄 경로 재계산", type="primary"):
                st.rerun()

        # ===== 채팅 인터페이스 =====
        st.divider()
        chat_container = st.container(height=CHAT_CONTAINER_HEIGHT)

        for msg in mission.chat_history:
            with chat_container.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("명령 입력 (예: 3D 모드로 바꿔줘, 위험해 보이는데?)"):
            mission.add_chat_message("user", user_input)
            with chat_container.chat_message("user"):
                st.write(user_input)

            with st.spinner("🧠 AI 분석 중..."):
                brain = LLMBrain()

                # 현재 경로 분석 (XAI용)
                path_analysis = None
                if hasattr(st.session_state, 'current_path') and st.session_state.current_path:
                    path_2d = [(p[0], p[1]) for p in st.session_state.current_path]
                    path_analysis = XAIUtils.analyze_path_risk(
                        path_2d,
                        [t.to_dict() for t in mission.threats],
                        mission.params.margin
                    )

                result = brain.parse_tactical_command(
                    user_input,
                    mission.params.to_dict(),
                    path_analysis
                )

                # 파라미터 업데이트
                if result["action"] == "UPDATE":
                    u = result["update_params"]
                    if u.get("safety_margin_km") is not None:
                        mission.params.margin = u["safety_margin_km"]
                    if u.get("rtb") is not None:
                        mission.params.rtb = u["rtb"]
                    if u.get("stpt_gap") is not None:
                        mission.params.stpt_gap = u["stpt_gap"]
                    if u.get("waypoint_name"):
                        mission.params.waypoint = u["waypoint_name"]
                    if u.get("algorithm"):
                        mission.params.algorithm = u["algorithm"]
                    if u.get("enable_3d") is not None:
                        mission.params.enable_3d = u["enable_3d"]

                ai_msg = result["response_text"]
                reasoning = result.get("reasoning", "")
                mission.add_chat_message("assistant", ai_msg, reasoning)

                with chat_container.chat_message("assistant"):
                    st.write(ai_msg)

                st.rerun()

    # ===== 위협 관리 탭 =====
    with tab_intel:
        st.subheader("위협 추가")

        add_type = st.radio('유형', ["원형 (SAM)", "사각형 (NFZ)"], horizontal=True, label_visibility="collapsed")
        t_name = st.text_input("명칭", value="Threat")

        if add_type == "원형 (SAM)":
            c1, c2 = st.columns(2)
            t_lat = c1.number_input("Lat", 33.0, 43.0, 38.0)
            t_lon = c2.number_input("Lon", 124.0, 132.0, 127.0)
            t_rad = st.slider("Radius(km)", 5, 50, 20)

            if st.button("➕ SAM 추가"):
                mission.add_threat(Threat(
                    name=t_name, type="SAM",
                    lat=t_lat, lon=t_lon, radius_km=t_rad
                ))
                st.rerun()
        else:
            c1, c2 = st.columns(2)
            l_min = c1.number_input("Min Lat", 33.0, 43.0, 37.5)
            l_max = c2.number_input("Max Lat", 33.0, 43.0, 37.8)
            ln_min = c1.number_input("Min Lon", 124.0, 132.0, 127.5)
            ln_max = c2.number_input("Max Lon", 124.0, 132.0, 127.8)

            if st.button("➕ NFZ 추가"):
                mission.add_threat(Threat(
                    name=t_name, type="NFZ",
                    lat_min=l_min, lat_max=l_max,
                    lon_min=ln_min, lon_max=ln_max
                ))
                st.rerun()

        st.divider()

        # 위협 목록
        st.subheader("위협 목록")
        if mission.threats:
            threat_df = pd.DataFrame([t.to_dict() for t in mission.threats])
            st.dataframe(threat_df, hide_index=True, use_container_width=True)

            del_name = st.selectbox("삭제할 위협", [t.name for t in mission.threats])
            if st.button("🗑️ 삭제"):
                mission.remove_threat(del_name)
                st.rerun()

    # ===== XAI 탭 =====
    with tab_xai:
        st.subheader("🤔 AI 판단 근거 (Explainable AI)")

        # 최근 AI 응답의 reasoning 표시
        if mission.chat_history:
            last_ai_msgs = [m for m in mission.chat_history if m["role"] == "assistant"]
            if last_ai_msgs:
                last_reasoning = last_ai_msgs[-1].get("reasoning", "설명 없음")
                if last_reasoning == '':
                    last_reasoning = "설명 없음"
                st.info(f"**최근 AI 판단:**\n\n{last_reasoning}")

        st.divider()

        # 경로 위험도 분석
        if hasattr(st.session_state, 'current_path') and st.session_state.current_path:
            st.subheader("📊 현재 경로 위험도 분석")

            path_2d = [(p[0], p[1]) for p in st.session_state.current_path]
            analysis = XAIUtils.analyze_path_risk(
                path_2d,
                [t.to_dict() for t in mission.threats],
                mission.params.margin
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("평균 위험도", f"{analysis['avg_risk']:.2%}")
            col2.metric("최대 위험도", f"{analysis['max_risk']:.2%}")
            col3.metric("고위험 구간", f"{analysis['high_risk_segments']}개")
            col4.metric("총 경로 길이", f"{analysis['total_length_km']:.1f}km")

            # 위험도 등급
            if analysis['max_risk'] > 0.7:
                st.error("⚠️ 경고: 고위험 경로입니다. 안전 마진 증가를 권장합니다.")
            elif analysis['max_risk'] > 0.4:
                st.warning("⚡ 주의: 중간 위험도 경로입니다.")
            else:
                st.success("✅ 안전: 저위험 경로입니다.")

            st.divider()

            # 히트맵 토글
            show_heatmap = st.checkbox("위험도 히트맵 표시", value=True)
            if show_heatmap:
                st.caption("지도에 위험도 히트맵이 표시됩니다 (빨강=위험, 초록=안전)")

    # ===== 디버그 탭 =====
    with tab_debug:
        st.subheader("🔧 디버그 & 실험 재현")

        save_name = st.text_input("시나리오 이름", value="scenario_01.json")
        if st.button("💾 현재 상태 저장"):
            mission.save_to_file(save_name)
            st.success(f"✅ {save_name} 저장 완료")

        st.caption("저장된 시나리오는 `logs/` 폴더에서 확인 가능")

        st.divider()

        # 지형 정보
        if mission.params.enable_3d:
            st.subheader("🌍 지형 정보")
            test_lat = st.number_input("테스트 위도", 33.0, 43.0, 37.5)
            test_lon = st.number_input("테스트 경도", 124.0, 132.0, 127.0)
            elev = terrain.get_elevation(test_lat, test_lon)
            st.metric("해당 위치 고도", f"{elev:.1f}m")

        st.divider()
        st.json(mission.params.to_dict())

# ===== 경로 계산 =====
with col_right:
    st.subheader(f"🗺️ 전술 지도 ({mission.params.algorithm})")

    # 알고리즘 선택 및 초기화
    if mission.params.algorithm == "A*":
        pathfinder = AStarPathfinder()
    elif mission.params.algorithm == "A* 3D":
        pathfinder = AStarPathfinder3D(terrain)
    elif mission.params.algorithm == "RRT":
        pathfinder = RRTPathfinder(max_iterations=3000)
    elif mission.params.algorithm == "RRT*":
        pathfinder = RRTStarPathfinder(max_iterations=3000)

    # 시작/목표 좌표
    start_coord = AIRPORTS[mission.params.start]
    if isinstance(start_coord, dict):
        start_coord = start_coord["coords"]
    target_coord = [mission.params.target_lat, mission.params.target_lon]

    # 🔧 3D 모드면 고도 추가 (핵심 수정)
    if mission.params.enable_3d or mission.params.algorithm == "A* 3D":
        if len(start_coord) == 2:
            start_coord = [start_coord[0], start_coord[1], 800]  # 고도 800m
        if len(target_coord) == 2:
            target_coord = [target_coord[0], target_coord[1], 800]

    threats_dict = [t.to_dict() for t in mission.threats]

    # 경로 계산 시작
    calc_start_time = time.time()

    # Ingress 경로
    wp_coord = None
    if mission.params.waypoint and mission.params.waypoint in AIRPORTS:
        wp = AIRPORTS[mission.params.waypoint]
        wp_coord = wp["coords"] if isinstance(wp, dict) else wp

        # Waypoint도 3D 변환
        if mission.params.enable_3d and isinstance(wp_coord, list) and len(wp_coord) == 2:
            wp_coord = [wp_coord[0], wp_coord[1], 800]

    raw_in = []
    if wp_coord:
        # Waypoint 경유
        if hasattr(pathfinder, 'find_path_3d') and mission.params.enable_3d:
            p1 = pathfinder.find_path_3d(start_coord, wp_coord, threats_dict, mission.params.margin)
            p2 = pathfinder.find_path_3d(wp_coord, target_coord, threats_dict, mission.params.margin)
        else:
            p1 = pathfinder.find_path(start_coord[:2], wp_coord[:2], threats_dict, mission.params.margin)
            p2 = pathfinder.find_path(wp_coord[:2], target_coord[:2], threats_dict, mission.params.margin)

        if p1 and p2:
            raw_in = p1 + p2[1:]
    else:
        # 직접 경로
        if hasattr(pathfinder, 'find_path_3d') and mission.params.enable_3d:
            raw_in = pathfinder.find_path_3d(start_coord, target_coord, threats_dict, mission.params.margin)
        else:
            raw_in = pathfinder.find_path(start_coord[:2], target_coord[:2], threats_dict, mission.params.margin)

    # 경로 평탄화
    if raw_in:
        if len(raw_in[0]) == 3:  # 3D 경로
            final_in = smooth_path_3d(raw_in)
        else:  # 2D 경로
            final_in = smooth_path(raw_in)
    else:
        final_in = []

    # Egress 경로 (RTB)
    final_out = []
    if mission.params.rtb and final_in:
        if len(final_in[-1]) == 3:  # 3D 경로
            egress_start = final_in[-1]
            egress_end = start_coord if len(start_coord) == 3 else start_coord + [800]
        else:  # 2D 경로
            egress_start = target_coord[:2]
            egress_end = start_coord[:2]

        if hasattr(pathfinder, 'find_path_3d') and mission.params.enable_3d:
            raw_out = pathfinder.find_path_3d(egress_start, egress_end, threats_dict, mission.params.margin)
            if raw_out:
                final_out = smooth_path_3d(raw_out)
        else:
            raw_out = pathfinder.find_path(egress_start, egress_end, threats_dict, mission.params.margin)
            if raw_out:
                final_out = smooth_path(raw_out)

    calc_time = time.time() - calc_start_time

    # 현재 경로 저장 (XAI용)
    st.session_state.current_path = final_in

    # 계산 정보 표시
    st.caption(f"⏱️ 계산 시간: {calc_time:.2f}초 | 경로 포인트: {len(final_in)}개")

    # ===== 지도 시각화 =====
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)

    # 공항 마커
    for name, data in AIRPORTS.items():
        coord = data["coords"] if isinstance(data, dict) else data
        color = "blue" if name == mission.params.start else "gray"
        folium.Marker(
            coord,
            icon=folium.Icon(color=color, icon="plane"),
            tooltip=name
        ).add_to(m)

    # 타겟 마커
    folium.Marker(
        target_coord[:2],  # 2D 좌표만 사용
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        tooltip=f"TARGET: {mission.params.target_name}"
    ).add_to(m)

    # 위협 시각화
    for t in mission.threats:
        if t.type == "SAM":
            folium.Circle(
                [t.lat, t.lon],
                radius=t.radius_km * 1000,
                color="crimson",
                fill=True,
                fill_opacity=0.3,
                tooltip=t.name
            ).add_to(m)
        elif t.type == "NFZ":
            folium.Rectangle(
                [[t.lat_min, t.lon_min], [t.lat_max, t.lon_max]],
                color="orange",
                fill=True,
                fill_opacity=0.3,
                tooltip=t.name
            ).add_to(m)

    # 히트맵 (XAI)
    if 'show_heatmap' in locals() and show_heatmap and mission.threats:
        with st.spinner("히트맵 생성 중..."):
            heatmap_data = XAIUtils.generate_heatmap_data(
                threats_dict,
                mission.params.margin
            )

            if heatmap_data:
                HeatMap(
                    heatmap_data,
                    min_opacity=0.2,
                    radius=15,
                    blur=20,
                    gradient={0.0: 'green', 0.4: 'yellow', 0.7: 'orange', 1.0: 'red'}
                ).add_to(m)

    # 경로 시각화
    if final_in:
        path_2d = [(p[0], p[1]) for p in final_in]
        folium.PolyLine(
            path_2d,
            color="blue",
            weight=4,
            opacity=0.8,
            tooltip="Ingress"
        ).add_to(m)

    if final_out:
        path_2d = [(p[0], p[1]) for p in final_out]
        folium.PolyLine(
            path_2d,
            color="orange",
            weight=4,
            dash_array="5, 5",
            opacity=0.8,
            tooltip="Egress (RTB)"
        ).add_to(m)

    # 지도 표시
    st_folium(m, width="100%", height=700)

    # ===== STPT 리스트 =====
    if final_in:
        st.divider()
        st.subheader("📋 Steer Point List")

        gap = mission.params.stpt_gap

        # 2D/3D 분기
        if len(final_in[0]) == 3:  # 3D
            data_in = [
                {
                    "Type": "Ingress",
                    "Seq": i+1,
                    "Lat": f"{p[0]:.4f}",
                    "Lon": f"{p[1]:.4f}",
                    "Alt(m)": f"{p[2]:.0f}"
                }
                for i, p in enumerate(final_in[::gap])
            ]
        else:  # 2D
            data_in = [
                {
                    "Type": "Ingress",
                    "Seq": i+1,
                    "Lat": f"{p[0]:.4f}",
                    "Lon": f"{p[1]:.4f}"
                }
                for i, p in enumerate(final_in[::gap])
            ]

        data_out = []
        if final_out:
            if len(final_out[0]) == 3:
                data_out = [
                    {
                        "Type": "Egress",
                        "Seq": i+1,
                        "Lat": f"{p[0]:.4f}",
                        "Lon": f"{p[1]:.4f}",
                        "Alt(m)": f"{p[2]:.0f}"
                    }
                    for i, p in enumerate(final_out[::gap])
                ]
            else:
                data_out = [
                    {
                        "Type": "Egress",
                        "Seq": i+1,
                        "Lat": f"{p[0]:.4f}",
                        "Lon": f"{p[1]:.4f}"
                    }
                    for i, p in enumerate(final_out[::gap])
                ]

        stpt_df = pd.DataFrame(data_in + data_out)
        st.dataframe(stpt_df, use_container_width=True, hide_index=True)

        # CSV 다운로드
        csv = stpt_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 STPT CSV 다운로드",
            csv,
            f"steer_points_{mission.params.algorithm.replace(' ', '_')}.csv",
            "text/csv"
        )

        # 알고리즘 성능 요약
        col1, col2, col3 = st.columns(3)
        col1.metric("알고리즘", mission.params.algorithm)
        col2.metric("총 포인트", len(final_in) + len(final_out))
        col3.metric("계산 시간", f"{calc_time:.2f}초")
    else:
        st.warning("⚠️ 경로를 찾을 수 없습니다. 위협 마진을 조정하거나 목표 좌표를 변경하세요.")
        st.info("💡 팁: 알고리즘을 RRT 또는 RRT*로 변경하거나, 안전 마진을 줄여보세요.")
