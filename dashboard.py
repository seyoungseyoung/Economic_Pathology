"""
US Economic Diagnostic System - Streamlit Dashboard
실시간 경제 진단 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# 로컬 모듈 import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import EconomicDataFetcher
from vital_signs import VitalSignsMonitor
from diagnostic_system import EconomicICU

# Streamlit 페이지 설정
st.set_page_config(
    page_title="US Economic Diagnostic System",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.critical {
    border-left-color: #d62728 !important;
}
.warning {
    border-left-color: #ff7f0e !important;
}
.normal {
    border-left-color: #2ca02c !important;
}
.diagnosis-box {
    background-color: #fff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ddd;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # 5분 캐시
def load_economic_data():
    """경제 데이터 로드 (캐시됨)"""
    try:
        fetcher = EconomicDataFetcher()
        return fetcher.get_all_data()
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

def create_vital_signs_chart(vital_data):
    """바이탈 사인 차트 생성"""
    # 레이더 차트용 데이터 준비
    categories = []
    values = []
    normal_ranges = []
    
    vital_names = {
        'temperature': '인플레이션',
        'pulse': '변동성(VIX)',
        'blood_pressure': '신용스프레드', 
        'respiration': '통화유통속도',
        'oxygen': '유동성(TED)'
    }
    
    for vital, data in vital_data.items():
        if vital in vital_names and 'value' in data and data['value'] is not None:
            categories.append(vital_names[vital])
            values.append(data['value'])
            # 정상범위 중간값
            normal_range = data.get('normal_range', (0, 100))
            normal_ranges.append((normal_range[0] + normal_range[1]) / 2)
    
    if not categories:
        return None
    
    # 레이더 차트 생성
    fig = go.Figure()
    
    # 현재 값
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='현재 값',
        line_color='rgb(255, 0, 0)',
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    
    # 정상 범위 (참고용)
    fig.add_trace(go.Scatterpolar(
        r=normal_ranges,
        theta=categories,
        fill='toself',
        name='정상 범위 중간값',
        line_color='rgb(0, 255, 0)',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(values), max(normal_ranges)) * 1.2]
            )
        ),
        showlegend=True,
        title="경제 바이탈 사인",
        height=400
    )
    
    return fig

def create_time_series_chart(data):
    """시계열 차트 생성 (더미 데이터)"""
    # 실제로는 historical 데이터를 사용해야 함
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('VIX', 'TED Spread', 'S&P 500', '10Y-2Y Spread'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 더미 시계열 데이터 생성
    np.random.seed(42)
    vix_data = 15 + np.cumsum(np.random.randn(30) * 0.5)
    ted_data = 20 + np.cumsum(np.random.randn(30) * 0.2)
    sp500_data = 4000 + np.cumsum(np.random.randn(30) * 10)
    spread_data = 0.5 + np.cumsum(np.random.randn(30) * 0.05)
    
    fig.add_trace(go.Scatter(x=dates, y=vix_data, name='VIX'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ted_data, name='TED'), row=1, col=2)
    fig.add_trace(go.Scatter(x=dates, y=sp500_data, name='S&P500'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=spread_data, name='Spread'), row=2, col=2)
    
    fig.update_layout(height=500, showlegend=False)
    return fig

def main():
    """메인 대시보드"""
    
    # 헤더
    st.title("🏥 US Economic Diagnostic System")
    st.markdown("### 실시간 경제 건강 진단 시스템")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # FRED API 키 입력
        fred_api_key = st.text_input("FRED API Key (선택사항)", type="password")
        
        # 자동 새로고침 설정
        auto_refresh = st.checkbox("자동 새로고침 (5분)", value=True)
        if auto_refresh:
            st.rerun()
        
        # 수동 새로고침 버튼
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        **시스템 정보:**
        - 바이탈 사인: 실시간 모니터링
        - 진단 엔진: AI 기반 분석
        - 데이터 소스: FRED, Yahoo Finance
        """)
    
    # 데이터 로드
    with st.spinner("경제 데이터 로딩 중..."):
        data = load_economic_data()
    
    if not data:
        st.error("데이터를 불러올 수 없습니다. API 키를 확인하거나 잠시 후 다시 시도해주세요.")
        return
    
    # 바이탈 사인 모니터링 시스템 초기화
    monitor = VitalSignsMonitor()
    vital_result = monitor.monitor(data)
    
    # ICU 시스템 초기화
    icu = EconomicICU()
    icu.add_patient("USA", data)
    dashboard_data = icu.get_dashboard()
    
    # 메인 대시보드 레이아웃
    
    # 1. 전체 상태 개요
    st.header("📊 경제 상태 개요")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = vital_result['vital_status']['overall_score']
        st.metric(
            label="전체 건강 점수",
            value=f"{overall_score:.1f}/100",
            delta=None
        )
        
    with col2:
        overall_status = vital_result['vital_status']['overall_status']
        status_color = {
            'Healthy': 'Good',
            'Stable': 'Stable', 
            'Stressed': 'Warning',
            'Critical': 'Critical'
        }.get(overall_status, 'Unknown')
        st.metric(
            label="전체 상태",
            value=f"{status_color}"
        )
        
    with col3:
        urgency = vital_result['diagnosis']['urgency'].upper()
        st.metric(
            label="응급도",
            value=urgency
        )
        
    with col4:
        alert_count = len(vital_result.get('alerts', []))
        st.metric(
            label="활성 알림",
            value=alert_count
        )
    
    # 2. 바이탈 사인 상세
    st.header("🫀 경제 바이탈 사인")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 레이더 차트
        radar_chart = create_vital_signs_chart(data['vital_signs'])
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
        else:
            st.warning("바이탈 사인 차트를 생성할 수 없습니다.")
    
    with col2:
        # 바이탈 사인 상세 정보
        st.subheader("상세 지표")
        for vital, vital_data in vital_result['vital_status']['individual_vitals'].items():
            status = vital_data['status']
            value = vital_data['value']
            name = vital_data['name']
            unit = vital_data['unit']
            
            # 상태에 따른 색상
            if status == 'critical':
                color = '🔴'
            elif status in ['hypotensive', 'hypertensive']:
                color = '🟡'
            else:
                color = '🟢'
                
            st.markdown(f"""
            <div class="metric-card {'critical' if status == 'critical' else 'warning' if status in ['hypotensive', 'hypertensive'] else 'normal'}">
                <strong>{color} {name}</strong><br>
                값: {value:.2f if isinstance(value, (int, float)) else 'N/A'} {unit}<br>
                상태: {status}
            </div>
            """, unsafe_allow_html=True)
    
    # 3. 진단 결과
    st.header("🩺 진단 결과")
    
    diagnosis = vital_result['diagnosis']
    
    st.markdown(f"""
    <div class="diagnosis-box">
        <h4>주 진단</h4>
        <p><strong>{diagnosis['primary_diagnosis']}</strong></p>
        <p><strong>응급도:</strong> {diagnosis['urgency'].upper()}</p>
        <p><strong>전체 점수:</strong> {diagnosis['overall_score']:.1f}/100</p>
    </div>
    """, unsafe_allow_html=True)
    
    if diagnosis['warnings']:
        st.subheader("⚠️ 주의사항")
        for warning in diagnosis['warnings']:
            st.warning(warning)
    
    if diagnosis['recommendations']:
        st.subheader("💡 권고사항")
        for rec in diagnosis['recommendations']:
            st.info(rec)
    
    # 4. 시장 데이터
    st.header("📈 시장 데이터")
    
    col1, col2, col3 = st.columns(3)
    
    market_data = data.get('market_data', {})
    
    with col1:
        if 'sp500' in market_data and 'last' in market_data['sp500']:
            sp_data = market_data['sp500']
            st.metric(
                label="S&P 500",
                value=f"{sp_data['last']:.2f}",
                delta=f"{sp_data.get('change_1d', 0):.2f}%"
            )
    
    with col2:
        if 'dollar_index' in market_data and 'last' in market_data['dollar_index']:
            dxy_data = market_data['dollar_index']
            st.metric(
                label="달러 지수",
                value=f"{dxy_data['last']:.2f}",
                delta=f"{dxy_data.get('change_1m', 0):.2f}%"
            )
    
    with col3:
        if 'gold' in market_data and 'last' in market_data['gold']:
            gold_data = market_data['gold']
            st.metric(
                label="금 가격",
                value=f"${gold_data['last']:.2f}",
                delta=f"{gold_data.get('change_1m', 0):.2f}%"
            )
    
    # 5. 시계열 차트
    st.header("📊 추세 분석")
    time_chart = create_time_series_chart(data)
    st.plotly_chart(time_chart, use_container_width=True)
    
    # 6. 원시 데이터 (접기 가능)
    with st.expander("🔍 원시 데이터 보기"):
        st.json(data)
    
    # 푸터
    st.markdown("---")
    st.markdown(f"**마지막 업데이트:** {data['timestamp']}")
    st.markdown("**데이터 소스:** FRED, Yahoo Finance | **개발:** US Economic Diagnostic System")

if __name__ == "__main__":
    main()