"""
통합 경제병리학 연구 시스템 (Unified Economic Pathology Research System)

핵심 기능:
1. 질병 진단 및 분류
2. 데이터 수집 및 전처리
3. 시각화 및 분석
4. 연구 보고서 생성

모든 기능을 3개의 핵심 그래프에 집약:
- Master Dashboard: 전체 현황 한눈에 보기
- Disease Timeline: 역사적 질병 진행
- Raw Data Tables: 수치 데이터 테이블
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class PathologyType(Enum):
    """경제 병리 유형 (5개 주요 카테고리)"""
    CIRCULATORY = "순환계 질환 (유동성/신용)"  # Liquidity/Credit
    METABOLIC = "대사 질환 (인플레/디플레)"    # Inflation/Deflation  
    STRUCTURAL = "구조 질환 (버블/부채)"       # Bubble/Debt
    IMMUNE = "면역 질환 (시스템 리스크)"       # Systemic Risk
    NEURAL = "신경 질환 (정책/기대)"          # Policy/Expectation

@dataclass
class EconomicDisease:
    """통합 경제 질병 데이터 클래스"""
    disease_id: str
    name: str
    pathology_type: PathologyType
    start_date: datetime
    end_date: Optional[datetime]
    
    # 핵심 지표 (5개)
    severity: float  # 0-10 (치명도)
    duration_years: float  # 지속 기간
    systemic_spread: float  # 0-1 (시스템 확산도)
    recovery_rate: float  # 0-1 (회복률)
    recurrence_risk: float  # 0-1 (재발 위험)
    
    # 정량적 영향
    gdp_impact: float  # % GDP 손실
    unemployment_impact: float  # % 실업률 증가
    fiscal_cost: float  # % GDP 재정 비용
    
    # 지역/국가
    affected_regions: List[str]
    
    # 원시 증상 데이터
    symptoms: Dict[str, List[float]]

class EconomicPathologyResearch:
    """통합 경제병리학 연구 시스템"""
    
    def __init__(self):
        self.diseases_db = self._initialize_comprehensive_database()
        self.raw_data_tables = self._generate_raw_data_tables()
        
    def _initialize_comprehensive_database(self) -> List[EconomicDisease]:
        """포괄적 질병 데이터베이스 초기화 (역사적 15개 주요 위기)"""
        
        diseases = []
        
        # 1. 1929 대공황 - 구조적 붕괴
        diseases.append(EconomicDisease(
            disease_id="1929_great_depression",
            name="1929 대공황",
            pathology_type=PathologyType.STRUCTURAL,
            start_date=datetime(1929, 10, 24),
            end_date=datetime(1939, 12, 31),
            severity=10.0,
            duration_years=10.2,
            systemic_spread=1.0,
            recovery_rate=0.7,
            recurrence_risk=0.1,
            gdp_impact=-30.0,
            unemployment_impact=20.0,
            fiscal_cost=25.0,
            affected_regions=["Global"],
            symptoms={
                "stock_crash": [-89, -85, -80, -75, -70, -65, -50, -40, -30, -20, -10],
                "bank_failures": [0, 100, 200, 400, 600, 800, 1000, 800, 600, 400, 200],
                "unemployment": [3, 9, 16, 25, 25, 20, 17, 14, 19, 17, 15]
            }
        ))
        
        # 2. 1970년대 스태그플레이션 - 대사 질환
        diseases.append(EconomicDisease(
            disease_id="1970s_stagflation",
            name="1970년대 스태그플레이션",
            pathology_type=PathologyType.METABOLIC,
            start_date=datetime(1973, 10, 17),
            end_date=datetime(1982, 12, 31),
            severity=7.5,
            duration_years=9.2,
            systemic_spread=0.8,
            recovery_rate=0.9,
            recurrence_risk=0.3,
            gdp_impact=-5.0,
            unemployment_impact=7.0,
            fiscal_cost=10.0,
            affected_regions=["USA", "Europe"],
            symptoms={
                "inflation": [3, 6, 11, 14, 13, 10, 12, 9, 6, 4],
                "oil_price": [3, 12, 35, 39, 14, 21, 34, 28, 15],
                "fed_funds": [5, 8, 11, 20, 16, 12, 8, 6]
            }
        ))
        
        # 3. 1987 블랙먼데이 - 순환계 급성
        diseases.append(EconomicDisease(
            disease_id="1987_black_monday",
            name="1987 블랙먼데이",
            pathology_type=PathologyType.CIRCULATORY,
            start_date=datetime(1987, 10, 19),
            end_date=datetime(1988, 6, 30),
            severity=6.5,
            duration_years=0.7,
            systemic_spread=0.6,
            recovery_rate=0.95,
            recurrence_risk=0.2,
            gdp_impact=-1.0,
            unemployment_impact=0.5,
            fiscal_cost=2.0,
            affected_regions=["USA", "Global_Markets"],
            symptoms={
                "dow_jones": [2600, 2000, 1800, 1900, 2100, 2300, 2500],
                "vix_equivalent": [20, 150, 80, 50, 30, 25, 20]
            }
        ))
        
        # 4. 1990 일본 버블붕괴 - 구조적 만성
        diseases.append(EconomicDisease(
            disease_id="1990_japan_bubble",
            name="1990 일본 버블붕괴",
            pathology_type=PathologyType.STRUCTURAL,
            start_date=datetime(1990, 1, 1),
            end_date=datetime(2010, 12, 31),
            severity=8.5,
            duration_years=21.0,
            systemic_spread=0.7,
            recovery_rate=0.4,
            recurrence_risk=0.1,
            gdp_impact=-15.0,
            unemployment_impact=3.0,
            fiscal_cost=40.0,
            affected_regions=["Japan"],
            symptoms={
                "nikkei": [39000, 30000, 25000, 20000, 15000, 12000, 10000, 8000, 9000, 10000],
                "land_prices": [100, 80, 60, 45, 35, 30, 25, 25, 28, 30],
                "deflation": [0, -0.5, -1.0, -1.5, -1.0, -0.5, -0.3, 0, 0.2]
            }
        ))
        
        # 5. 1994 멕시코 테킬라 위기 - 순환계
        diseases.append(EconomicDisease(
            disease_id="1994_tequila_crisis",
            name="1994 테킬라 위기",
            pathology_type=PathologyType.CIRCULATORY,
            start_date=datetime(1994, 12, 20),
            end_date=datetime(1996, 6, 30),
            severity=6.0,
            duration_years=1.5,
            systemic_spread=0.4,
            recovery_rate=0.8,
            recurrence_risk=0.6,
            gdp_impact=-8.0,
            unemployment_impact=4.0,
            fiscal_cost=15.0,
            affected_regions=["Mexico", "LatinAmerica"],
            symptoms={
                "peso_devaluation": [0, 50, 100, 80, 60, 40, 20],
                "capital_flight": [0, 30, 60, 40, 20, 10, 5]
            }
        ))
        
        # 6. 1997 아시아 금융위기 - 순환계
        diseases.append(EconomicDisease(
            disease_id="1997_asian_crisis",
            name="1997 아시아 금융위기",
            pathology_type=PathologyType.CIRCULATORY,
            start_date=datetime(1997, 7, 2),
            end_date=datetime(1999, 12, 31),
            severity=8.0,
            duration_years=2.5,
            systemic_spread=0.8,
            recovery_rate=0.75,
            recurrence_risk=0.4,
            gdp_impact=-12.0,
            unemployment_impact=6.0,
            fiscal_cost=25.0,
            affected_regions=["Asia", "Thailand", "Korea", "Indonesia"],
            symptoms={
                "currency_depreciation": [0, 30, 60, 80, 70, 50, 30, 20, 10],
                "stock_markets": [0, -40, -70, -60, -40, -20, 10, 20, 30]
            }
        ))
        
        # 7. 1998 러시아 루블 위기 - 면역계
        diseases.append(EconomicDisease(
            disease_id="1998_russia_default",
            name="1998 러시아 루블위기",
            pathology_type=PathologyType.IMMUNE,
            start_date=datetime(1998, 8, 17),
            end_date=datetime(1999, 12, 31),
            severity=7.0,
            duration_years=1.4,
            systemic_spread=0.5,
            recovery_rate=0.6,
            recurrence_risk=0.7,
            gdp_impact=-10.0,
            unemployment_impact=3.0,
            fiscal_cost=20.0,
            affected_regions=["Russia", "EmergingMarkets"],
            symptoms={
                "ruble_crash": [6, 25, 28, 25, 20, 15],
                "ltcm_losses": [0, 2.5, 4.6, 3.0, 1.0]
            }
        ))
        
        # 8. 2000 닷컴 버블 - 구조적
        diseases.append(EconomicDisease(
            disease_id="2000_dotcom_bubble",
            name="2000 닷컴 버블",
            pathology_type=PathologyType.STRUCTURAL,
            start_date=datetime(2000, 3, 10),
            end_date=datetime(2003, 10, 31),
            severity=7.5,
            duration_years=3.7,
            systemic_spread=0.7,
            recovery_rate=0.8,
            recurrence_risk=0.5,
            gdp_impact=-3.0,
            unemployment_impact=3.0,
            fiscal_cost=8.0,
            affected_regions=["USA", "Global_Tech"],
            symptoms={
                "nasdaq": [5000, 4000, 3000, 2000, 1500, 1200, 1500, 2000],
                "tech_pe_ratio": [200, 150, 100, 50, 30, 25, 30, 40]
            }
        ))
        
        # 9. 2008 글로벌 금융위기 - 면역계 전신
        diseases.append(EconomicDisease(
            disease_id="2008_global_crisis",
            name="2008 글로벌 금융위기",
            pathology_type=PathologyType.IMMUNE,
            start_date=datetime(2007, 8, 1),
            end_date=datetime(2009, 6, 30),
            severity=9.5,
            duration_years=2.0,
            systemic_spread=1.0,
            recovery_rate=0.7,
            recurrence_risk=0.3,
            gdp_impact=-5.0,
            unemployment_impact=5.0,
            fiscal_cost=30.0,
            affected_regions=["Global"],
            symptoms={
                "vix": [12, 15, 20, 25, 40, 80, 70, 50, 35, 25, 20],
                "credit_spreads": [80, 100, 150, 200, 400, 800, 600, 400, 200, 150],
                "house_prices": [100, 95, 85, 70, 60, 55, 58, 62, 68]
            }
        ))
        
        # 10. 2011 유럽 부채위기 - 구조적
        diseases.append(EconomicDisease(
            disease_id="2011_europe_debt",
            name="2011 유럽 부채위기",
            pathology_type=PathologyType.STRUCTURAL,
            start_date=datetime(2010, 4, 1),
            end_date=datetime(2015, 12, 31),
            severity=7.0,
            duration_years=5.8,
            systemic_spread=0.8,
            recovery_rate=0.6,
            recurrence_risk=0.4,
            gdp_impact=-3.0,
            unemployment_impact=5.0,
            fiscal_cost=20.0,
            affected_regions=["Europe", "Greece", "Spain", "Italy"],
            symptoms={
                "sovereign_spreads": [100, 200, 400, 600, 500, 400, 300, 250],
                "debt_to_gdp": [80, 90, 100, 110, 120, 115, 110, 105]
            }
        ))
        
        # 11. 2020 COVID-19 쇼크 - 면역계 외부
        diseases.append(EconomicDisease(
            disease_id="2020_covid_shock",
            name="2020 COVID-19 경제쇼크",
            pathology_type=PathologyType.IMMUNE,
            start_date=datetime(2020, 2, 1),
            end_date=datetime(2021, 12, 31),
            severity=8.5,
            duration_years=2.0,
            systemic_spread=1.0,
            recovery_rate=0.9,
            recurrence_risk=0.2,
            gdp_impact=-3.1,
            unemployment_impact=10.0,
            fiscal_cost=25.0,
            affected_regions=["Global"],
            symptoms={
                "vix": [15, 25, 85, 70, 40, 30, 25, 22, 20],
                "unemployment": [3.5, 4, 14.7, 13, 10, 8, 6, 5, 4],
                "fiscal_deficit": [-3, -15, -12, -8, -5, -4]
            }
        ))
        
        # 12. 2022 인플레이션 쇼크 - 대사
        diseases.append(EconomicDisease(
            disease_id="2022_inflation_shock",
            name="2022 인플레이션 쇼크",
            pathology_type=PathologyType.METABOLIC,
            start_date=datetime(2021, 3, 1),
            end_date=datetime(2024, 6, 30),
            severity=6.5,
            duration_years=3.3,
            systemic_spread=0.9,
            recovery_rate=0.8,
            recurrence_risk=0.6,
            gdp_impact=-1.0,
            unemployment_impact=1.0,
            fiscal_cost=5.0,
            affected_regions=["Global"],
            symptoms={
                "cpi": [1.2, 4.2, 6.8, 9.1, 8.3, 6.4, 4.0, 3.1, 2.6],
                "fed_funds": [0, 0.25, 2.0, 4.0, 5.5, 5.5, 5.25, 5.0],
                "energy_prices": [50, 70, 90, 120, 110, 80, 75]
            }
        ))
        
        return diseases
    
    def _generate_raw_data_tables(self) -> Dict[str, pd.DataFrame]:
        """원시 데이터 테이블 생성"""
        
        tables = {}
        
        # 1. 질병 기본 정보 테이블
        disease_data = []
        for disease in self.diseases_db:
            disease_data.append({
                'Disease_ID': disease.disease_id,
                'Name': disease.name,
                'Type': disease.pathology_type.value,
                'Start_Date': disease.start_date.strftime('%Y-%m-%d'),
                'End_Date': disease.end_date.strftime('%Y-%m-%d') if disease.end_date else 'Ongoing',
                'Severity_Score': disease.severity,
                'Duration_Years': disease.duration_years,
                'Systemic_Spread': disease.systemic_spread,
                'Recovery_Rate': disease.recovery_rate,
                'Recurrence_Risk': disease.recurrence_risk,
                'GDP_Impact_%': disease.gdp_impact,
                'Unemployment_Impact_%': disease.unemployment_impact,
                'Fiscal_Cost_%': disease.fiscal_cost,
                'Affected_Regions': ', '.join(disease.affected_regions)
            })
        
        tables['disease_summary'] = pd.DataFrame(disease_data)
        
        # 2. 병리학적 통계 테이블
        pathology_stats = []
        for ptype in PathologyType:
            diseases_of_type = [d for d in self.diseases_db if d.pathology_type == ptype]
            if diseases_of_type:
                pathology_stats.append({
                    'Pathology_Type': ptype.value,
                    'Count': len(diseases_of_type),
                    'Avg_Severity': np.mean([d.severity for d in diseases_of_type]),
                    'Avg_Duration_Years': np.mean([d.duration_years for d in diseases_of_type]),
                    'Avg_GDP_Impact': np.mean([d.gdp_impact for d in diseases_of_type]),
                    'Max_Systemic_Spread': max([d.systemic_spread for d in diseases_of_type]),
                    'Avg_Recovery_Rate': np.mean([d.recovery_rate for d in diseases_of_type]),
                    'Avg_Recurrence_Risk': np.mean([d.recurrence_risk for d in diseases_of_type])
                })
        
        tables['pathology_stats'] = pd.DataFrame(pathology_stats)
        
        # 3. 시대별 분석 테이블
        decade_analysis = []
        for decade in range(1920, 2030, 10):
            decade_diseases = [d for d in self.diseases_db 
                             if decade <= d.start_date.year < decade + 10]
            
            if decade_diseases:
                decade_analysis.append({
                    'Decade': f"{decade}s",
                    'Disease_Count': len(decade_diseases),
                    'Total_Severity': sum([d.severity for d in decade_diseases]),
                    'Avg_Severity': np.mean([d.severity for d in decade_diseases]),
                    'Total_GDP_Impact': sum([d.gdp_impact for d in decade_diseases]),
                    'Max_Single_Impact': min([d.gdp_impact for d in decade_diseases]),
                    'Dominant_Pathology': max(set([d.pathology_type.value for d in decade_diseases]),
                                            key=[d.pathology_type.value for d in decade_diseases].count)
                })
            else:
                decade_analysis.append({
                    'Decade': f"{decade}s",
                    'Disease_Count': 0,
                    'Total_Severity': 0,
                    'Avg_Severity': 0,
                    'Total_GDP_Impact': 0,
                    'Max_Single_Impact': 0,
                    'Dominant_Pathology': 'None'
                })
        
        tables['decade_analysis'] = pd.DataFrame(decade_analysis)
        
        # 4. 2025년 현재 진단 테이블
        current_indicators = {
            'TED_Spread_bps': 25,
            'VIX': 18,
            'Credit_Spreads_bps': 120,
            'SP500_PE': 24,
            'Housing_PriceRent': 26,
            'US_DebtGDP_%': 125,
            'CPI_YoY_%': 2.7,
            'Fed_Funds_%': 5.5,
            'Unemployment_%': 3.8,
            'Systemic_Risk_Index': 3.2
        }
        
        # 정상 범위와 비교
        normal_ranges = {
            'TED_Spread_bps': (10, 50),
            'VIX': (12, 25),
            'Credit_Spreads_bps': (80, 150),
            'SP500_PE': (15, 25),
            'Housing_PriceRent': (15, 25),
            'US_DebtGDP_%': (60, 90),
            'CPI_YoY_%': (1.5, 3.0),
            'Fed_Funds_%': (2, 5),
            'Unemployment_%': (3.5, 5.5),
            'Systemic_Risk_Index': (1, 4)
        }
        
        diagnosis_2025 = []
        for indicator, value in current_indicators.items():
            normal_min, normal_max = normal_ranges[indicator]
            if value < normal_min:
                status = 'Below Normal'
                deviation = (normal_min - value) / normal_min * 100
            elif value > normal_max:
                status = 'Above Normal'
                deviation = (value - normal_max) / normal_max * 100
            else:
                status = 'Normal'
                deviation = 0
            
            diagnosis_2025.append({
                'Indicator': indicator,
                'Current_Value': value,
                'Normal_Range': f"{normal_min}-{normal_max}",
                'Status': status,
                'Deviation_%': round(deviation, 1)
            })
        
        tables['diagnosis_2025'] = pd.DataFrame(diagnosis_2025)
        
        return tables
    
    def create_master_dashboard(self):
        """마스터 대시보드 - 모든 정보 한 화면에"""
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "병리학적 분포", "시대별 발생 빈도", "심각도 vs 지속기간",
                "회복률 vs 재발위험", "GDP 영향도", "시스템 확산도",
                "2025 현재 진단", "치료 효과성", "예측 모델"
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,  # 세로 간격 확대
            horizontal_spacing=0.12  # 가로 간격 확대
        )
        
        # 1. 병리학적 분포 (파이 차트) - 실제 존재하는 병리만 표시
        pathology_counts = {}
        for disease in self.diseases_db:
            ptype = disease.pathology_type.value
            pathology_counts[ptype] = pathology_counts.get(ptype, 0) + 1
        
        # 실제 데이터가 있는 병리 유형만 필터링
        filtered_pathology = {k: v for k, v in pathology_counts.items() if v > 0}
        
        # 한국어 병리명으로 변환
        korean_pathology_names = {
            'STRUCTURAL': '구조적 위기',
            'CIRCULATORY': '순환계 위기', 
            'METABOLIC': '대사계 위기',
            'IMMUNE': '면역계 위기',
            'NEURAL': '신경계 위기'
        }
        
        # 실제 존재하는 병리만 한국어로 변환
        labels = [korean_pathology_names.get(k, k) for k in filtered_pathology.keys()]
        values = list(filtered_pathology.values())
        
        # 색상을 실제 데이터 개수에 맞춰 조정
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(labels)]
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',  # 라벨과 백분율 표시
            textposition='auto',  # 자동 위치 조정
            hole=0.3,  # 적절한 도넛 홀 크기
            marker_colors=colors,
            showlegend=False,  # 범례 끄기 (라벨이 차트에 표시되므로)
            textfont=dict(size=10),  # 적절한 텍스트 크기
            hovertemplate='<b>%{label}</b><br>개수: %{value}<br>비율: %{percent}<extra></extra>'
        ), row=1, col=1)
        
        # 2. 시대별 발생 빈도
        decade_counts = {}
        for disease in self.diseases_db:
            decade = (disease.start_date.year // 10) * 10
            decade_counts[f"{decade}s"] = decade_counts.get(f"{decade}s", 0) + 1
        
        fig.add_trace(go.Bar(
            x=list(decade_counts.keys()),
            y=list(decade_counts.values()),
            marker_color='lightblue'
        ), row=1, col=2)
        
        # 3. 심각도 vs 지속기간
        fig.add_trace(go.Scatter(
            x=[d.severity for d in self.diseases_db],
            y=[d.duration_years for d in self.diseases_db],
            mode='markers+text',
            text=[d.name[:10] for d in self.diseases_db],
            textposition="top center",
            marker=dict(
                size=[abs(d.gdp_impact) for d in self.diseases_db],
                color=[d.systemic_spread for d in self.diseases_db],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.65, len=0.3)
            )
        ), row=1, col=3)
        
        # 4. 회복률 vs 재발위험 - 텍스트 겹침 방지 개선
        fig.add_trace(go.Scatter(
            x=[d.recovery_rate for d in self.diseases_db],
            y=[d.recurrence_risk for d in self.diseases_db],
            mode='markers+text',  # 마커와 텍스트 모두 표시
            text=[d.name[:4] for d in self.diseases_db],  # 매우 짧은 라벨 사용
            textposition="top center",
            textfont=dict(size=8),  # 작은 폰트 사용
            marker=dict(
                size=10,
                color='red',
                opacity=0.7,
                line=dict(width=1, color='darkred')
            ),
            hovertemplate='<b>%{customdata}</b><br>회복률: %{x:.1f}%<br>재발위험: %{y:.1f}%<extra></extra>',
            customdata=[d.name for d in self.diseases_db]
        ), row=2, col=1)
        
        # 5. GDP 영향도
        gdp_impacts = sorted([d.gdp_impact for d in self.diseases_db])
        names_sorted = [d.name[:15] for d in sorted(self.diseases_db, key=lambda x: x.gdp_impact)]
        
        fig.add_trace(go.Bar(
            x=gdp_impacts,
            y=names_sorted,
            orientation='h',
            marker_color=['red' if x < -10 else 'orange' if x < -5 else 'yellow' for x in gdp_impacts]
        ), row=2, col=2)
        
        # 6. 시스템 확산도 - 텍스트 잘림 방지 개선
        spread_data = [(d.systemic_spread, d.name[:5]) for d in self.diseases_db]  # 매우 짧은 텍스트
        spread_data.sort(key=lambda x: x[0], reverse=True)
        
        fig.add_trace(go.Scatter(
            x=[s[0] for s in spread_data],
            y=list(range(len(spread_data))),
            mode='markers+text',  # 마커와 텍스트 모두 표시
            text=[s[1] for s in spread_data],
            textposition="middle right",
            textfont=dict(size=8),  # 작은 폰트 사용
            marker=dict(
                size=12,
                color=[s[0] for s in spread_data],
                colorscale='Reds'
            ),
            hovertemplate='<b>%{customdata}</b><br>확산도: %{x:.2f}<extra></extra>',
            customdata=[d.name for d in self.diseases_db]  # 호버에서 전체 이름 표시
        ), row=2, col=3)
        
        # 7. 2025 현재 진단
        diagnosis_df = self.raw_data_tables['diagnosis_2025']
        abnormal = diagnosis_df[diagnosis_df['Status'] != 'Normal']
        
        fig.add_trace(go.Bar(
            x=abnormal['Indicator'],
            y=abnormal['Deviation_%'],
            marker_color=['red' if x > 50 else 'orange' if x > 20 else 'yellow' for x in abnormal['Deviation_%']]
        ), row=3, col=1)
        
        # 8. 치료 효과성 (회복률) - y축 텍스트 잘림 방지 개선
        # 더 짧은 한국어 라벨 생성
        pathology_very_short_names = {
            'STRUCTURAL': '구조',
            'CIRCULATORY': '순환',
            'METABOLIC': '대사', 
            'IMMUNE': '면역',
            'NEURAL': '신경'
        }
        
        treatment_data = []
        for d in self.diseases_db:
            short_name = pathology_very_short_names.get(d.pathology_type.value, d.pathology_type.value[:3])
            treatment_data.append((d.recovery_rate, short_name, d.name))
        
        treatment_data = sorted(treatment_data)
        
        fig.add_trace(go.Scatter(
            x=[t[0] for t in treatment_data],
            y=[t[1] for t in treatment_data],
            mode='markers+text',  # 마커와 텍스트 모두 표시
            text=[t[2][:4] for t in treatment_data],  # 질병명 짧게 표시
            textposition="middle right",
            textfont=dict(size=8),
            marker=dict(
                size=12,
                color='green',
                opacity=0.7
            ),
            hovertemplate='<b>%{customdata}</b><br>병리유형: %{y}<br>회복률: %{x:.1f}%<extra></extra>',
            customdata=[t[2] for t in treatment_data]  # 호버에서 전체 질병명 표시
        ), row=3, col=2)
        
        # 9. 예측 모델 (미래 위험도)
        years = list(range(2025, 2035))
        risk_projection = [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # 예측 모델
        
        fig.add_trace(go.Scatter(
            x=years,
            y=risk_projection,
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='red', width=2)
        ), row=3, col=3)
        
        # 레이아웃 업데이트 - 텍스트 잘림 방지를 위한 여백 확대
        fig.update_layout(
            title="경제병리학 마스터 대시보드 - 병리학적 투자 전략 (2025년)",
            height=1200,
            showlegend=False,  # 파이 차트에 라벨이 표시되므로 전체 범례 비활성화
            font=dict(size=10),  # 폰트 크기 조정
            margin=dict(l=100, r=100, t=100, b=80)  # 좌우 여백 더 확대
        )
        
        # 축 라벨 업데이트 - 모든 차트의 축 라벨 개선 및 폰트 크기 조정
        fig.update_xaxes(title_text="연대", title_font_size=12, tickfont_size=10, row=1, col=2)
        fig.update_yaxes(title_text="발생 횟수", title_font_size=12, tickfont_size=10, row=1, col=2)
        fig.update_xaxes(title_text="심각도", title_font_size=12, tickfont_size=10, row=1, col=3)
        fig.update_yaxes(title_text="지속기간(년)", title_font_size=12, tickfont_size=10, row=1, col=3)
        fig.update_xaxes(title_text="회복률", title_font_size=12, tickfont_size=10, row=2, col=1)
        fig.update_yaxes(title_text="재발위험", title_font_size=12, tickfont_size=10, row=2, col=1)
        fig.update_xaxes(title_text="GDP 영향(%)", title_font_size=12, tickfont_size=10, row=2, col=2)
        fig.update_xaxes(title_text="시스템 확산도", title_font_size=12, tickfont_size=10, row=2, col=3)
        fig.update_yaxes(title_text="순위", title_font_size=12, tickfont_size=9, row=2, col=3)
        fig.update_xaxes(title_text="지표", title_font_size=12, tickfont_size=9, tickangle=45, row=3, col=1)
        fig.update_yaxes(title_text="정상범위 이탈(%)", title_font_size=12, tickfont_size=10, row=3, col=1)
        fig.update_xaxes(title_text="회복률 (%)", title_font_size=12, tickfont_size=10, row=3, col=2)
        fig.update_yaxes(title_text="병리유형", title_font_size=12, tickfont_size=9, row=3, col=2)
        fig.update_xaxes(title_text="연도", title_font_size=12, tickfont_size=10, row=3, col=3)
        fig.update_yaxes(title_text="위험도", title_font_size=12, tickfont_size=10, row=3, col=3)
        
        return fig
    
    def create_historical_timeline(self):
        """역사적 질병 진행 타임라인"""
        
        fig = go.Figure()
        
        # 색상 매핑
        color_map = {
            PathologyType.CIRCULATORY: '#3498db',
            PathologyType.METABOLIC: '#e74c3c',
            PathologyType.STRUCTURAL: '#f39c12',
            PathologyType.IMMUNE: '#9b59b6',
            PathologyType.NEURAL: '#2ecc71'
        }
        
        # Y 포지션 계산
        y_positions = {}
        current_y = 0
        for disease in sorted(self.diseases_db, key=lambda x: x.start_date):
            y_positions[disease.disease_id] = current_y
            current_y += 1
        
        # 각 질병의 생존 곡선
        for disease in self.diseases_db:
            y_pos = y_positions[disease.disease_id]
            
            # 메인 바 (질병 기간)
            fig.add_trace(go.Scatter(
                x=[disease.start_date, disease.end_date or datetime(2025, 1, 1)],
                y=[y_pos, y_pos],
                mode='lines',
                line=dict(
                    color=color_map[disease.pathology_type],
                    width=disease.severity * 2
                ),
                name=f"{disease.name}",
                hovertemplate=(
                    f"<b>{disease.name}</b><br>" +
                    f"유형: {disease.pathology_type.value}<br>" +
                    f"심각도: {disease.severity}/10<br>" +
                    f"지속: {disease.duration_years:.1f}년<br>" +
                    f"GDP 영향: {disease.gdp_impact}%<br>" +
                    f"회복률: {disease.recovery_rate:.1%}<br>" +
                    "<extra></extra>"
                ),
                showlegend=False
            ))
            
            # 시작점 마커
            fig.add_trace(go.Scatter(
                x=[disease.start_date],
                y=[y_pos],
                mode='markers',
                marker=dict(
                    size=disease.severity * 2,
                    color=color_map[disease.pathology_type],
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # 종료점 마커 (회복률 반영)
            if disease.end_date:
                fig.add_trace(go.Scatter(
                    x=[disease.end_date],
                    y=[y_pos],
                    mode='markers',
                    marker=dict(
                        size=disease.recovery_rate * 15,
                        color='green' if disease.recovery_rate > 0.7 else 'orange',
                        symbol='square'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # 범례 추가 (병리학적 타입별)
        for ptype, color in color_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color=color, width=4),
                name=ptype.value,
                showlegend=True
            ))
        
        fig.update_layout(
            title="경제 질병 역사적 타임라인 (1929-2025)<br>선 두께 = 심각도, 마커 크기 = 회복률",
            xaxis_title="연도",
            yaxis=dict(
                title="경제 질병",
                tickmode='array',
                tickvals=list(range(len(self.diseases_db))),
                ticktext=[d.name for d in sorted(self.diseases_db, key=lambda x: x.start_date)]
            ),
            height=800,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def save_raw_data_excel(self, filepath: str):
        """원시 데이터를 엑셀 파일로 저장"""
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in self.raw_data_tables.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 시트 포맷팅
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 30)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
    
    def generate_executive_summary(self) -> str:
        """경영진용 요약 보고서"""
        
        stats = self.raw_data_tables['pathology_stats']
        decade = self.raw_data_tables['decade_analysis']
        diagnosis = self.raw_data_tables['diagnosis_2025']
        
        summary = f"""
════════════════════════════════════════════════════════════════
                    경제병리학 연구 요약 보고서
                          {datetime.now().strftime('%Y년 %m월')}
════════════════════════════════════════════════════════════════

【핵심 발견사항】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 분석 대상: 1929-2024년 주요 경제 위기 {len(self.diseases_db)}건
2. 가장 치명적: 1929 대공황 (심각도 10.0, GDP 영향 -30%)
3. 가장 장기간: 일본 버블붕괴 (21년 지속, 회복률 40%)
4. 최근 경향: 2000년대 이후 회복률 개선되나 재발 위험 증가

【병리학적 분류 통계】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for _, row in stats.iterrows():
            summary += f"""
▪ {row['Pathology_Type']}
  - 발생 횟수: {row['Count']}건
  - 평균 심각도: {row['Avg_Severity']:.1f}/10
  - 평균 지속: {row['Avg_Duration_Years']:.1f}년
  - 평균 GDP 타격: {row['Avg_GDP_Impact']:.1f}%
  - 평균 회복률: {row['Avg_Recovery_Rate']:.1%}
"""
        
        summary += f"""

【2025년 현재 진단】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

위험 수준: {"[WARNING] 경계" if len(diagnosis[diagnosis['Status'] != 'Normal']) > 3 else "[OK] 안정"}

주요 지표 이상:
"""
        
        abnormal = diagnosis[diagnosis['Status'] != 'Normal'].sort_values('Deviation_%', ascending=False)
        for _, row in abnormal.head(5).iterrows():
            status_icon = "🔴" if row['Deviation_%'] > 50 else "🟡" if row['Deviation_%'] > 20 else "🟢"
            summary += f"""
{status_icon} {row['Indicator']}: {row['Current_Value']} (정상: {row['Normal_Range']}, 이탈률 {row['Deviation_%']}%)"""
        
        summary += f"""

【시대별 분석】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

최악의 10년: {decade.loc[decade['Total_GDP_Impact'].idxmin(), 'Decade']} (총 GDP 영향 {decade['Total_GDP_Impact'].min():.1f}%)
최다 발생: {decade.loc[decade['Disease_Count'].idxmax(), 'Decade']} ({decade['Disease_Count'].max()}건)
현재 2020년대: {decade.loc[decade['Decade'] == '2020s', 'Disease_Count'].values[0]}건 진행 중

【권고사항】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 조기 경보: 현재 {len(abnormal)}개 지표가 정상 범위 벗어남 → 모니터링 강화 필요
2. 예방 정책: 부채 비율(125%) 및 자산가격(P/E 24) 관리 시급
3. 시스템 강화: 과거 위기 대비 회복력 개선되었으나 지속적 점검 필요
4. 국제 공조: 글로벌 확산 방지를 위한 정책 조율 중요

════════════════════════════════════════════════════════════════
※ 상세 분석은 첨부된 데이터 테이블 및 시각화 자료 참조
════════════════════════════════════════════════════════════════
"""
        
        return summary

# 실행 및 통합 결과 생성
if __name__ == "__main__":
    import os
    
    research = EconomicPathologyResearch()
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("[START] Unified Economic Pathology Research System")
    
    # 1. 마스터 대시보드 (모든 정보 집약)
    dashboard = research.create_master_dashboard()
    dashboard.write_image(f"{output_dir}/master_dashboard.png", width=1400, height=1200)
    print(f"[SAVED] Master dashboard: {output_dir}/master_dashboard.png")
    
    # 2. 역사적 타임라인
    timeline = research.create_historical_timeline()
    timeline.write_image(f"{output_dir}/historical_timeline.png", width=1400, height=800)
    print(f"[SAVED] Historical timeline: {output_dir}/historical_timeline.png")
    
    # 3. Raw 데이터 엑셀 저장
    research.save_raw_data_excel(f"{output_dir}/economic_pathology_raw_data.xlsx")
    print(f"[SAVED] Raw data Excel: {output_dir}/economic_pathology_raw_data.xlsx")
    
    # 4. 경영진 요약 보고서
    summary = research.generate_executive_summary()
    with open(f"{output_dir}/executive_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[SAVED] Executive summary: {output_dir}/executive_summary.txt")
    
    # 콘솔 출력
    print("\n" + "="*80)
    print("2025년 현재 진단 요약:")
    print("="*80)
    
    diagnosis = research.raw_data_tables['diagnosis_2025']
    abnormal = diagnosis[diagnosis['Status'] != 'Normal']
    
    print(f"정상 범위 이탈 지표: {len(abnormal)}개")
    for _, row in abnormal.head(5).iterrows():
        print(f"  - {row['Indicator']}: {row['Current_Value']} ({row['Status']}, 이탈률 {row['Deviation_%']}%)")
    
    print(f"\n총 분석 질병: {len(research.diseases_db)}건")
    print(f"평균 심각도: {np.mean([d.severity for d in research.diseases_db]):.1f}/10")
    print(f"평균 GDP 영향: {np.mean([d.gdp_impact for d in research.diseases_db]):.1f}%")
    
    print("\n[COMPLETE] All analysis completed!")
    print("\n[FILES] 생성된 파일:")
    print("  1. master_dashboard.png - 종합 대시보드")
    print("  2. historical_timeline.png - 역사적 타임라인") 
    print("  3. economic_pathology_raw_data.xlsx - 원시 데이터")
    print("  4. executive_summary.txt - 경영진 요약 보고서")