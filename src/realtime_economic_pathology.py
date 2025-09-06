"""
실시간 경제병리학 진단 시스템 (Real-time Economic Pathology System)
FRED, IMF, Yahoo Finance 등 실시간 데이터 연동
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# API Libraries
from fredapi import Fred
import yfinance as yf
import sdmx
from dataclasses import dataclass
from enum import Enum

# Load environment variables
load_dotenv()

@dataclass
class DataPoint:
    """데이터 포인트 클래스"""
    value: float
    source: str
    timestamp: datetime
    confidence: float = 1.0

class DataQuality(Enum):
    """데이터 품질 등급"""
    HIGH = "HIGH"       # 95%+ 신뢰도
    MEDIUM = "MEDIUM"   # 80-95% 신뢰도  
    LOW = "LOW"         # 60-80% 신뢰도
    UNRELIABLE = "UNRELIABLE"  # <60% 신뢰도

class MultiSourceValidator:
    """다중 소스 데이터 검증기"""
    
    def __init__(self):
        # 데이터 소스 우선순위 (신뢰도 순)
        self.source_priority = {
            'FRED': 1.0,      # 가장 신뢰할 수 있는 공식 정부 데이터
            'IMF': 0.95,      # 국제기구 데이터
            'Yahoo': 0.85,    # 시장 데이터 제공업체
            'Manual': 0.70    # 수동 입력 데이터
        }
        
        # 허용 가능한 편차 임계값 (%)
        self.deviation_thresholds = {
            'CPI_YOY': 0.5,          # 인플레이션 ±0.5%
            'FED_FUNDS': 0.25,       # 금리 ±0.25%
            'UNEMPLOYMENT': 0.3,     # 실업률 ±0.3%
            'DEBT_GDP': 2.0,         # 부채/GDP ±2%
            'VIX': 5.0,              # VIX ±5점
            'SPY_PRICE': 3.0,        # 주가 ±3%
            'default': 10.0          # 기본 ±10%
        }
    
    def cross_validate_data(self, indicator: str, data_points: List[DataPoint]) -> Dict:
        """다중 소스 데이터 교차 검증"""
        if len(data_points) < 1:
            return {
                'validated_value': None,
                'confidence': 0.0,
                'quality': DataQuality.UNRELIABLE,
                'sources_used': [],
                'validation_method': 'no_data'
            }
        
        if len(data_points) == 1:
            # 단일 소스의 경우 소스 신뢰도에 따라 품질 결정
            dp = data_points[0]
            source_confidence = self.source_priority.get(dp.source, 0.5)
            
            if source_confidence >= 0.8 and dp.confidence >= 0.7:
                quality = DataQuality.HIGH
            elif source_confidence >= 0.6 and dp.confidence >= 0.5:
                quality = DataQuality.MEDIUM
            elif source_confidence >= 0.4:
                quality = DataQuality.LOW
            else:
                quality = DataQuality.UNRELIABLE
            
            return {
                'validated_value': dp.value,
                'confidence': source_confidence * dp.confidence,
                'quality': quality,
                'sources_used': [dp.source],
                'validation_method': 'single_source_trusted'
            }
        
        # 1. 소스별 가중치 적용
        weighted_values = []
        total_weight = 0
        
        for dp in data_points:
            weight = self.source_priority.get(dp.source, 0.5) * dp.confidence
            weighted_values.append(dp.value * weight)
            total_weight += weight
        
        if total_weight == 0:
            return {
                'validated_value': None,
                'confidence': 0.0,
                'quality': DataQuality.UNRELIABLE,
                'sources_used': [],
                'validation_method': 'no_valid_sources'
            }
        
        weighted_average = sum(weighted_values) / total_weight
        
        # 2. 편차 분석
        threshold = self.deviation_thresholds.get(indicator, self.deviation_thresholds['default'])
        deviations = []
        
        for dp in data_points:
            deviation = abs(dp.value - weighted_average) / max(abs(weighted_average), 0.01) * 100
            deviations.append(deviation)
        
        # 3. 일치도 계산
        consistent_sources = sum(1 for dev in deviations if dev <= threshold)
        consistency_ratio = consistent_sources / len(data_points)
        
        # 4. 품질 등급 결정
        if consistency_ratio >= 0.8 and len(data_points) >= 3:
            quality = DataQuality.HIGH
            confidence = min(0.95, 0.7 + consistency_ratio * 0.25)
        elif consistency_ratio >= 0.6 and len(data_points) >= 2:
            quality = DataQuality.MEDIUM
            confidence = min(0.85, 0.6 + consistency_ratio * 0.25)
        elif consistency_ratio >= 0.4:
            quality = DataQuality.LOW
            confidence = min(0.75, 0.4 + consistency_ratio * 0.35)
        else:
            quality = DataQuality.UNRELIABLE
            confidence = 0.3
        
        return {
            'validated_value': weighted_average,
            'confidence': confidence,
            'quality': quality,
            'sources_used': [dp.source for dp in data_points],
            'validation_method': 'multi_source_weighted',
            'consistency_ratio': consistency_ratio,
            'deviations': deviations,
            'raw_values': [dp.value for dp in data_points]
        }
    
    def get_fallback_value(self, indicator: str, historical_data: pd.Series = None) -> Optional[float]:
        """Fallback 값 제공"""
        # 역사적 평균 또는 경제학적 상식 기반 기본값
        fallback_values = {
            'CPI_YOY': 2.5,        # 목표 인플레이션
            'FED_FUNDS': 3.0,      # 중성 금리 추정
            'UNEMPLOYMENT': 4.0,    # 자연실업률 추정
            'VIX': 18.0,           # 역사적 평균
            'DEBT_GDP': 100.0,     # 현재 수준 근사치
            'FIN_STRESS': 0.0,     # 정상 상태
            'TERM_SPREAD': 100.0,  # 정상 수익률 곡선
            'POLICY_UNCERTAINTY': 100.0  # 평균 수준
        }
        
        if historical_data is not None and len(historical_data) > 0:
            # 최근 12개월 평균 사용
            return float(historical_data.tail(12).mean())
        
        return fallback_values.get(indicator, 0.0)

class RealTimeDataConnector:
    """실시간 데이터 연결기"""
    
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.imf_enabled = os.getenv('IMF_SDMX_ENABLED', 'true').lower() == 'true'
        
        # 다중 소스 검증기 초기화
        self.validator = MultiSourceValidator()
        
        # FRED 연결
        self.fred = None
        if self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
                print("[SUCCESS] FRED API 연결 성공")
            except Exception as e:
                print(f"[ERROR] FRED API 연결 실패: {e}")
        
        # IMF SDMX 연결
        self.imf_client = None
        if self.imf_enabled:
            try:
                self.imf_client = sdmx.Client('IMF')
                print("[SUCCESS] IMF SDMX 연결 성공")
            except Exception as e:
                print(f"[ERROR] IMF SDMX 연결 실패: {e}")
        
        # 핵심 지표 매핑
        self.fred_indicators = {
            # 순환계 (유동성/신용) - TED_SPREAD 대신 SOFR-OIS 사용 (LIBOR 폐지로 인함)
            'SOFR_OIS': 'SOFR',  # Secured Overnight Financing Rate
            'LIBOR_OIS': 'USD3MTD156N', 
            'REPO_RATE': 'REPOULSTSRV',
            'BANK_CDS': 'BAMLHYH0A0HYM2TRIV',  # High Yield Option-Adjusted Spread
            
            # 대사계 (인플레이션/디플레이션)
            'CPI_CORE': 'CPILFESL',
            'PCE_CORE': 'PCEPILFE', 
            'CPI_YOY': 'CPIAUCSL',
            'PPI_YOY': 'PPIFIS',
            '5Y5Y_INFLATION': 'T5YIE',
            
            # 구조계 (버블/부채)
            'DEBT_GDP': 'GFDEGDQ188S',  # Federal Debt/GDP
            'CORP_DEBT_GDP': 'NCBDBIQ027S',  # Corporate Debt/GDP
            'HH_DEBT_INCOME': 'HDTGPDUSQ163N',  # Household Debt/Income
            'HOUSE_PRICE': 'CSUSHPINSA',  # Case-Shiller Home Price
            
            # 면역계 (시스템 리스크)
            'FIN_STRESS': 'STLFSI4',  # Financial Stress Index
            'SYSTEMIC_RISK': 'NFCI',  # Financial Conditions Index
            'CREDIT_SPREAD_AAA': 'AAA10Y',
            'CREDIT_SPREAD_BAA': 'BAA10Y',
            
            # 신경계 (정책/기대)
            'FED_FUNDS': 'FEDFUNDS',
            'POLICY_UNCERTAINTY': 'USEPUINDXD',
            'CONSUMER_SENTIMENT': 'UMCSENT',
            'TERM_SPREAD': 'T10Y2Y',  # 10Y-2Y Spread
            
            # 기본 경제 지표
            'GDP_GROWTH': 'GDPC1',
            'UNEMPLOYMENT': 'UNRATE',
            'CAPACITY_UTIL': 'TCU',
            'INDUSTRIAL_PROD': 'INDPRO'
        }
        
        self.yahoo_tickers = {
            'VIX': '^VIX',
            'SPY': 'SPY',
            'QQQ': 'QQQ',
            'TLT': 'TLT',  # 20+ Year Treasury
            'HYG': 'HYG',  # High Yield Corporate Bond ETF
            'USD_INDEX': 'DX-Y.NYB',
            'OIL': 'CL=F',
            'GOLD': 'GC=F'
        }
        
        self.imf_indicators = {
            'GLOBAL_CPI': 'CPI',
            'GLOBAL_GDP': 'NGDP_RPCH',
            'UNEMPLOYMENT_RATE': 'LUR',
            'CURRENT_ACCOUNT': 'BCA_NGDPD'
        }
    
    def fetch_fred_data(self, indicator: str, start_date: datetime = None) -> pd.Series:
        """FRED 데이터 조회"""
        if not self.fred or indicator not in self.fred_indicators:
            return pd.Series()
        
        try:
            fred_id = self.fred_indicators[indicator]
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365*2)  # 2년
            
            data = self.fred.get_series(fred_id, observation_start=start_date.strftime('%Y-%m-%d'))
            return data.dropna()
        except Exception as e:
            print(f"FRED 데이터 조회 실패 ({indicator}): {e}")
            return pd.Series()
    
    def fetch_yahoo_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Yahoo Finance 데이터 조회"""
        try:
            if ticker in self.yahoo_tickers:
                symbol = self.yahoo_tickers[ticker]
            else:
                symbol = ticker
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Yahoo Finance 데이터 조회 실패 ({ticker}): {e}")
            return pd.DataFrame()
    
    def fetch_imf_data(self, indicator: str, countries: List[str] = ['USA'], start_year: int = 2020) -> pd.DataFrame:
        """IMF SDMX 데이터 조회"""
        if not self.imf_client or indicator not in self.imf_indicators:
            return pd.DataFrame()
        
        try:
            dataset = self.imf_indicators[indicator]
            key = '+'.join(countries) + '..' if countries else ''
            
            data_msg = self.imf_client.data(
                dataset, 
                key=key, 
                params={'startPeriod': start_year}
            )
            
            df = sdmx.to_pandas(data_msg)
            return df
        except Exception as e:
            print(f"IMF 데이터 조회 실패 ({indicator}): {e}")
            return pd.DataFrame()
    
    def get_multi_source_data(self, indicator: str) -> List[DataPoint]:
        """다중 소스에서 지표 데이터 수집"""
        data_points = []
        current_time = datetime.now()
        
        # 1. FRED 데이터 시도
        if indicator in self.fred_indicators:
            fred_data = self.fetch_fred_data(indicator)
            if not fred_data.empty:
                if indicator == 'CPI_YOY':
                    # CPI 연간 변화율 계산
                    if len(fred_data) >= 12:
                        latest_value = fred_data.iloc[-1]
                        year_ago_value = fred_data.iloc[-13]
                        cpi_yoy = ((latest_value - year_ago_value) / year_ago_value) * 100
                        data_points.append(DataPoint(cpi_yoy, 'FRED', current_time, 1.0))
                else:
                    data_points.append(DataPoint(float(fred_data.iloc[-1]), 'FRED', current_time, 1.0))
        
        # 2. Yahoo Finance 데이터 시도
        if indicator in ['VIX', 'SPY_PRICE']:
            ticker = 'VIX' if indicator == 'VIX' else 'SPY'
            yahoo_data = self.fetch_yahoo_data(ticker, period="5d")
            if not yahoo_data.empty:
                value = float(yahoo_data['Close'].iloc[-1])
                data_points.append(DataPoint(value, 'Yahoo', current_time, 0.9))
        
        # 3. IMF 데이터 시도 (선택적)
        # 실시간성이 떨어지므로 특정 지표만 사용
        if indicator in ['CPI_YOY', 'UNEMPLOYMENT'] and self.imf_client:
            try:
                imf_data = self.fetch_imf_data('CPI' if indicator == 'CPI_YOY' else 'LUR')
                if not imf_data.empty:
                    # 최신 데이터 추출 (복잡한 IMF 데이터 구조 처리 필요)
                    data_points.append(DataPoint(float(imf_data.iloc[-1]), 'IMF', current_time, 0.8))
            except:
                pass  # IMF 데이터 실패시 무시
        
        return data_points
    
    def get_current_indicators(self) -> Dict[str, float]:
        """다중 소스 검증을 통한 현재 주요 지표 조회"""
        print("[DATA] 다중 소스 실시간 데이터 수집 및 검증 중...")
        
        # 주요 지표 목록 - TED_SPREAD -> SOFR_OIS로 대체 (LIBOR 폐지)
        key_indicators = [
            'SOFR_OIS', 'CPI_YOY', 'FED_FUNDS', 'UNEMPLOYMENT',
            'DEBT_GDP', 'FIN_STRESS', 'TERM_SPREAD', 'POLICY_UNCERTAINTY',
            'VIX', 'SPY_PRICE'
        ]
        
        validated_indicators = {}
        validation_report = {}
        
        for indicator in key_indicators:
            # 다중 소스에서 데이터 수집
            data_points = self.get_multi_source_data(indicator)
            
            # 교차 검증 수행
            validation_result = self.validator.cross_validate_data(indicator, data_points)
            
            # 검증된 값 사용 또는 fallback
            if validation_result['validated_value'] is not None and validation_result['confidence'] > 0.3:
                validated_indicators[indicator] = validation_result['validated_value']
            else:
                # Fallback 값 사용
                fallback_value = self.validator.get_fallback_value(indicator)
                validated_indicators[indicator] = fallback_value
                validation_result['used_fallback'] = True
                validation_result['fallback_value'] = fallback_value
            
            validation_report[indicator] = validation_result
        
        # 검증 요약 출력
        high_quality_count = sum(1 for r in validation_report.values() if r.get('quality') == DataQuality.HIGH)
        total_count = len(validation_report)
        
        print(f"[DATA] 데이터 품질 요약: {high_quality_count}/{total_count}개 지표가 HIGH 품질")
        
        # 낮은 품질의 지표 경고
        for indicator, result in validation_report.items():
            if result.get('quality') in [DataQuality.LOW, DataQuality.UNRELIABLE]:
                sources = result.get('sources_used', [])
                print(f"[WARNING] {indicator}: {result['quality'].value} 품질 (소스: {sources})")
        
        return validated_indicators

class RealTimePathologyDiagnoser:
    """실시간 경제병리 진단기"""
    
    def __init__(self):
        self.data_connector = RealTimeDataConnector()
        
        # 정상 범위 (과거 데이터 기반) - TED_SPREAD 대신 SOFR_OIS 사용
        self.normal_ranges = {
            'SOFR_OIS': (4.0, 6.0),      # SOFR rate percentage
            'VIX': (12, 25),             # volatility index
            'CPI_YOY': (1.5, 3.0),       # % year-over-year
            'FED_FUNDS': (2.0, 5.0),     # %
            'UNEMPLOYMENT': (3.5, 5.5),   # %
            'DEBT_GDP': (60, 90),        # %
            'FIN_STRESS': (-1, 1),       # standardized index
            'TERM_SPREAD': (50, 200),    # basis points
            'POLICY_UNCERTAINTY': (50, 150), # index
            'SPY_PRICE': (300, 500)      # approximation
        }
        
        # 질병 진단 기준 - TED_SPREAD -> SOFR_OIS 대체
        self.pathology_criteria = {
            'CIRCULATORY_DYSFUNCTION': {
                'primary': ['SOFR_OIS', 'FIN_STRESS'],
                'thresholds': {'SOFR_OIS': 6.5, 'FIN_STRESS': 2},
                'severity_multiplier': 1.5
            },
            'METABOLIC_DISORDER': {
                'primary': ['CPI_YOY', 'FED_FUNDS'],
                'thresholds': {'CPI_YOY': 5, 'FED_FUNDS': 6},
                'severity_multiplier': 1.2
            },
            'STRUCTURAL_IMBALANCE': {
                'primary': ['DEBT_GDP', 'SPY_PRICE'],
                'thresholds': {'DEBT_GDP': 120, 'SPY_PRICE': 500},
                'severity_multiplier': 1.0
            },
            'IMMUNE_DEFICIENCY': {
                'primary': ['VIX', 'FIN_STRESS'],
                'thresholds': {'VIX': 30, 'FIN_STRESS': 1.5},
                'severity_multiplier': 1.8
            },
            'NEURAL_PARALYSIS': {
                'primary': ['POLICY_UNCERTAINTY', 'TERM_SPREAD'],
                'thresholds': {'POLICY_UNCERTAINTY': 200, 'TERM_SPREAD': 0},
                'severity_multiplier': 1.1
            }
        }
    
    def diagnose_current_state(self) -> Dict:
        """현재 경제 상태 실시간 진단"""
        
        # 실시간 데이터 수집
        current_data = self.data_connector.get_current_indicators()
        
        if not current_data:
            return {
                'status': 'ERROR',
                'message': 'No real-time data available',
                'timestamp': datetime.now().isoformat()
            }
        
        # 진단 결과
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': ['FRED', 'Yahoo Finance'],
            'current_indicators': current_data,
            'detected_pathologies': [],
            'overall_health_score': 100,
            'risk_level': 'NORMAL',
            'recommendations': []
        }
        
        # 각 지표별 이상 여부 체크
        abnormal_indicators = []
        for indicator, value in current_data.items():
            if indicator in self.normal_ranges:
                normal_min, normal_max = self.normal_ranges[indicator]
                
                if value < normal_min or value > normal_max:
                    deviation = abs(value - (normal_min + normal_max)/2) / ((normal_max - normal_min)/2)
                    # deviation_severity를 더 보수적으로 계산 (로그 스케일 적용)
                    deviation_severity = min(np.log(deviation + 1), 2.0)  # 최대 2.0으로 제한
                    abnormal_indicators.append({
                        'indicator': indicator,
                        'value': value,
                        'normal_range': self.normal_ranges[indicator],
                        'deviation_severity': deviation_severity
                    })
        
        # 병리학적 진단
        for pathology, criteria in self.pathology_criteria.items():
            pathology_score = 0
            detected_indicators = []
            
            for indicator in criteria['primary']:
                if indicator in current_data:
                    value = current_data[indicator]
                    threshold = criteria['thresholds'].get(indicator, float('inf'))
                    
                    if indicator in ['SOFR_OIS', 'VIX', 'CPI_YOY', 'FED_FUNDS', 'DEBT_GDP', 'POLICY_UNCERTAINTY']:
                        if value > threshold:
                            severity = (value - threshold) / threshold
                            pathology_score += severity * criteria['severity_multiplier']
                            detected_indicators.append(indicator)
                    elif indicator in ['TERM_SPREAD']:
                        if value < threshold:
                            severity = abs(value - threshold) / 100  # basis points
                            pathology_score += severity * criteria['severity_multiplier']
                            detected_indicators.append(indicator)
                    elif indicator in ['FIN_STRESS']:
                        if value > threshold:
                            severity = value - threshold
                            pathology_score += severity * criteria['severity_multiplier']
                            detected_indicators.append(indicator)
            
            # 병리학적 상태 감지
            if pathology_score > 0.5:  # 임계값
                diagnosis['detected_pathologies'].append({
                    'pathology': pathology,
                    'severity_score': min(pathology_score, 10),
                    'confidence': min(pathology_score / 2, 1.0),
                    'affected_indicators': detected_indicators
                })
        
        # 전체 건강 점수 계산 (더 보수적으로 계산)
        if abnormal_indicators:
            # 평균 deviation_severity 기반으로 계산 (개별 합산 대신)
            avg_deviation = np.mean([abs_ind['deviation_severity'] for abs_ind in abnormal_indicators])
            total_deductions = min(avg_deviation * 15, 70)  # 최대 70점까지만 차감
        else:
            total_deductions = 0
        
        diagnosis['overall_health_score'] = max(30, 100 - total_deductions)  # 최소 30점 보장
        
        # 위험 수준 결정
        if diagnosis['overall_health_score'] > 80:
            diagnosis['risk_level'] = 'NORMAL'
        elif diagnosis['overall_health_score'] > 60:
            diagnosis['risk_level'] = 'ELEVATED'  
        elif diagnosis['overall_health_score'] > 40:
            diagnosis['risk_level'] = 'HIGH'
        else:
            diagnosis['risk_level'] = 'CRITICAL'
        
        # 권고사항 생성
        diagnosis['recommendations'] = self._generate_recommendations(diagnosis)
        
        return diagnosis
    
    def _generate_recommendations(self, diagnosis: Dict) -> List[str]:
        """진단 결과 기반 권고사항 생성"""
        recommendations = []
        
        risk_level = diagnosis['risk_level']
        pathologies = diagnosis['detected_pathologies']
        
        if risk_level == 'CRITICAL':
            recommendations.append("[CRITICAL] 즉시 비상 대응 체계 가동 필요")
            recommendations.append("[TREATMENT] 긴급 유동성 공급 및 정책 조율 검토")
        elif risk_level == 'HIGH':
            recommendations.append("[WARNING] 강화된 모니터링 체계 운영")
            recommendations.append("[DATA] 주요 지표 일일 점검")
        elif risk_level == 'ELEVATED':
            recommendations.append("[MONITOR] 예방적 조치 검토")
            recommendations.append("[TREND] 시장 동향 면밀 관찰")
        
        # 병리별 권고
        for pathology_info in pathologies:
            pathology = pathology_info['pathology']
            
            if pathology == 'CIRCULATORY_DYSFUNCTION':
                recommendations.append("[LIQUIDITY] 유동성 공급 라인 점검 (중앙은행 스왑, 레포 오퍼레이션)")
            elif pathology == 'METABOLIC_DISORDER':  
                recommendations.append("[TEMP] 인플레이션 대응 정책 준비 (금리, 통화정책)")
            elif pathology == 'STRUCTURAL_IMBALANCE':
                recommendations.append("[STRUCT] 구조적 불균형 해소 방안 검토 (부채, 자산가격)")
            elif pathology == 'IMMUNE_DEFICIENCY':
                recommendations.append("[IMMUNE] 시스템 리스크 관리 강화 (거시건전성)")
            elif pathology == 'NEURAL_PARALYSIS':
                recommendations.append("[BRAIN] 정책 소통 및 기대 관리 개선")
        
        return recommendations
    
    def create_realtime_dashboard(self, diagnosis: Dict) -> go.Figure:
        """실시간 진단 대시보드"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"현재 건강 점수: {diagnosis['overall_health_score']:.0f}/100",
                f"위험 수준: {diagnosis['risk_level']}",
                "주요 지표 현황", 
                "감지된 병리 상태"
            ],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. 건강 점수 게이지
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=diagnosis['overall_health_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "경제 건강 점수"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 60], 'color': "orange"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=1, col=1)
        
        # 2. 위험 수준 표시 (개선된 스케일 설명 포함)
        risk_colors = {
            'NORMAL': 'green',
            'ELEVATED': 'yellow', 
            'HIGH': 'orange',
            'CRITICAL': 'red'
        }
        
        risk_values = {
            'NORMAL': 1,
            'ELEVATED': 2,
            'HIGH': 3, 
            'CRITICAL': 4
        }
        
        risk_descriptions = {
            'NORMAL': '정상 (1단계)',
            'ELEVATED': '주의 (2단계)', 
            'HIGH': '경고 (3단계)',
            'CRITICAL': '위험 (4단계)'
        }
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_values[diagnosis['risk_level']],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"위험 수준<br>{risk_descriptions[diagnosis['risk_level']]}"},
            gauge={
                'axis': {'range': [1, 4], 'tickvals': [1,2,3,4], 'ticktext': ['정상','주의','경고','위험']},
                'bar': {'color': risk_colors[diagnosis['risk_level']]},
                'steps': [
                    {'range': [1, 1.5], 'color': "lightgreen"},
                    {'range': [1.5, 2.5], 'color': "lightyellow"},
                    {'range': [2.5, 3.5], 'color': "orange"},
                    {'range': [3.5, 4], 'color': "lightcoral"}
                ]
            },
            number={'font': {'size': 20}}
        ), row=1, col=2)
        
        # 3. 주요 지표 현황
        indicators = diagnosis['current_indicators']
        indicator_names = list(indicators.keys())[:8]  # 상위 8개
        indicator_values = [indicators[name] for name in indicator_names]
        
        # 정상 범위와 비교하여 색상 결정
        colors = []
        for name in indicator_names:
            if name in self.normal_ranges:
                value = indicators[name]
                normal_min, normal_max = self.normal_ranges[name]
                if normal_min <= value <= normal_max:
                    colors.append('green')
                else:
                    colors.append('red')
            else:
                colors.append('blue')
        
        fig.add_trace(go.Bar(
            x=indicator_names,
            y=indicator_values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in indicator_values],
            textposition='auto'
        ), row=2, col=1)
        
        # 4. 감지된 병리 상태
        if diagnosis['detected_pathologies']:
            pathology_names = [p['pathology'].replace('_', ' ') for p in diagnosis['detected_pathologies']]
            pathology_scores = [p['severity_score'] for p in diagnosis['detected_pathologies']]
            
            fig.add_trace(go.Bar(
                x=pathology_names,
                y=pathology_scores,
                marker_color='red',
                text=[f"{s:.1f}" for s in pathology_scores],
                textposition='auto'
            ), row=2, col=2)
        else:
            fig.add_trace(go.Bar(
                x=['No Pathologies'],
                y=[0],
                marker_color='green'
            ), row=2, col=2)
        
        # 레이아웃 업데이트 - y축 라벨과 인덱스 표시 개선
        fig.update_layout(
            title=f"실시간 경제병리 진단 대시보드<br><sub>{diagnosis['timestamp']}</sub>",
            height=800,
            showlegend=False,
            font=dict(size=12)
        )
        
        # x축 개선: 각도 조정 및 텍스트 크기 설정
        fig.update_xaxes(
            tickangle=45, 
            row=2, col=1,
            title_text="경제 지표",
            title_font=dict(size=14),
            tickfont=dict(size=10)
        )
        fig.update_xaxes(
            tickangle=45, 
            row=2, col=2,
            title_text="병리 유형", 
            title_font=dict(size=14),
            tickfont=dict(size=10)
        )
        
        # y축 개선: 라벨 추가 및 포맷 설정
        fig.update_yaxes(
            row=2, col=1,
            title_text="지표 값",
            title_font=dict(size=14),
            tickfont=dict(size=10),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            row=2, col=2,
            title_text="심각도 (0-10)",
            title_font=dict(size=14), 
            tickfont=dict(size=10),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[0, 10]  # 심각도 스케일 명시
        )
        
        return fig
    
    def save_diagnosis_report(self, diagnosis: Dict, filepath: str):
        """진단 보고서 저장"""
        
        report = f"""
════════════════════════════════════════════════════════════════
                    실시간 경제병리 진단 보고서
                    {diagnosis['timestamp']}
════════════════════════════════════════════════════════════════

【진단 요약】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
경제 건강 점수: {diagnosis['overall_health_score']:.0f}/100
위험 수준: {diagnosis['risk_level']}
데이터 출처: {', '.join(diagnosis['data_sources'])}

【현재 주요 지표】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for indicator, value in diagnosis['current_indicators'].items():
            normal_range = self.normal_ranges.get(indicator, 'N/A')
            status = "[SUCCESS] 정상" if indicator in self.normal_ranges and self.normal_ranges[indicator][0] <= value <= self.normal_ranges[indicator][1] else "[WARNING] 비정상"
            report += f"{indicator}: {value:.2f} (정상범위: {normal_range}) {status}\n"
        
        report += f"""

【감지된 병리 상태】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if diagnosis['detected_pathologies']:
            for i, pathology in enumerate(diagnosis['detected_pathologies'], 1):
                report += f"""
{i}. {pathology['pathology'].replace('_', ' ')}
   - 심각도: {pathology['severity_score']:.1f}/10
   - 신뢰도: {pathology['confidence']:.1%}
   - 영향 지표: {', '.join(pathology['affected_indicators'])}
"""
        else:
            report += "\n현재 감지된 병리 상태 없음 [SUCCESS]"
        
        report += f"""

【권고사항】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, recommendation in enumerate(diagnosis['recommendations'], 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"""

【면책 조항】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이 진단은 연구 목적으로 제공되며, 투자 조언이 아닙니다.
실제 정책 결정 시에는 추가적인 전문가 분석이 필요합니다.

데이터 출처: FRED (Federal Reserve Economic Data), Yahoo Finance, IMF
================================================================================
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

# 실행 코드
if __name__ == "__main__":
    import os
    
    diagnoser = RealTimePathologyDiagnoser()
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n[HOSPITAL] 실시간 경제병리 진단 시작")
    print("="*60)
    
    # 실시간 진단 실행
    diagnosis = diagnoser.diagnose_current_state()
    
    if diagnosis.get('status') == 'ERROR':
        print(f"[ERROR] 진단 실패: {diagnosis.get('message')}")
    else:
        # 결과 출력
        print(f"[DATA] 경제 건강 점수: {diagnosis['overall_health_score']:.0f}/100")
        print(f"[CRITICAL] 위험 수준: {diagnosis['risk_level']}")
        print(f"[ANALYSIS] 감지된 병리: {len(diagnosis['detected_pathologies'])}개")
        
        if diagnosis['detected_pathologies']:
            print("\n감지된 병리 상태:")
            for pathology in diagnosis['detected_pathologies']:
                print(f"  - {pathology['pathology'].replace('_', ' ')}: {pathology['severity_score']:.1f}/10")
        
        print(f"\n주요 지표 ({len(diagnosis['current_indicators'])}개):")
        for indicator, value in list(diagnosis['current_indicators'].items())[:5]:
            print(f"  - {indicator}: {value:.2f}")
        
        # 대시보드 생성
        dashboard = diagnoser.create_realtime_dashboard(diagnosis)
        dashboard.write_image(f"{output_dir}/realtime_diagnosis_dashboard.png", width=1200, height=800)
        print(f"\n[SAVED] 대시보드 저장: {output_dir}/realtime_diagnosis_dashboard.png")
        
        # 진단 보고서 저장
        diagnoser.save_diagnosis_report(diagnosis, f"{output_dir}/realtime_diagnosis_report.txt")
        print(f"[SAVED] 진단 보고서 저장: {output_dir}/realtime_diagnosis_report.txt")
        
        print("\n[SUCCESS] 실시간 진단 완료!")
        
        # 권고사항 출력
        if diagnosis['recommendations']:
            print("\n[RECOMMENDATIONS] 권고사항:")
            for i, rec in enumerate(diagnosis['recommendations'], 1):
                print(f"  {i}. {rec}")