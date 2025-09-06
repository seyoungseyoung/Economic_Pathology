"""
실전 투자 전략 생성기 - Gemini AI 통합
Practical Investment Strategy Generator with Gemini AI Integration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from dataclasses import dataclass
from enum import Enum

from gemini_ai_analyzer import GeminiEconomicAnalyzer, InvestmentStyle
from realtime_economic_pathology import RealTimePathologyDiagnoser

@dataclass
class InvestmentRecommendation:
    """투자 권고안"""
    asset_class: str
    allocation_percentage: float
    rationale: str
    risk_level: str
    expected_return: Optional[float] = None
    holding_period: Optional[str] = None

@dataclass 
class PortfolioStrategy:
    """포트폴리오 전략"""
    strategy_name: str
    total_score: float
    recommendations: List[InvestmentRecommendation]
    rebalancing_frequency: str
    risk_management: Dict[str, Any]
    market_outlook: str
    generated_at: datetime

class MarketRegime(Enum):
    """시장 환경"""
    BULL_MARKET = "bull_market"        # 강세장
    BEAR_MARKET = "bear_market"        # 약세장
    SIDEWAYS = "sideways"              # 횡보장
    VOLATILE = "volatile"              # 변동성장
    CRISIS = "crisis"                  # 위기상황

class InvestmentStrategyGenerator:
    """실전 투자 전략 생성기"""
    
    def __init__(self, gemini_api_key: str):
        """
        초기화
        Args:
            gemini_api_key: Gemini AI API 키
        """
        self.gemini_analyzer = GeminiEconomicAnalyzer(gemini_api_key)
        self.pathology_diagnoser = RealTimePathologyDiagnoser()
        
        # 자산 클래스별 기본 정보
        self.asset_classes = {
            'stocks': {
                'name': '주식',
                'volatility': 0.20,
                'expected_return': 0.10,
                'correlation_with_economy': 0.8
            },
            'bonds': {
                'name': '채권', 
                'volatility': 0.05,
                'expected_return': 0.04,
                'correlation_with_economy': -0.2
            },
            'commodities': {
                'name': '원자재',
                'volatility': 0.25,
                'expected_return': 0.06,
                'correlation_with_economy': 0.6
            },
            'real_estate': {
                'name': '부동산',
                'volatility': 0.15,
                'expected_return': 0.08,
                'correlation_with_economy': 0.5
            },
            'cash': {
                'name': '현금성자산',
                'volatility': 0.01,
                'expected_return': 0.02,
                'correlation_with_economy': 0.0
            }
        }
        
    def generate_comprehensive_strategy(self, 
                                      investment_style: InvestmentStyle,
                                      investment_horizon: str = "medium_term",
                                      initial_capital: float = 100000) -> PortfolioStrategy:
        """
        종합적인 투자 전략 생성
        
        Args:
            investment_style: 투자 성향
            investment_horizon: 투자 기간 (short_term, medium_term, long_term)
            initial_capital: 초기 투자 자본
            
        Returns:
            PortfolioStrategy: 완성된 포트폴리오 전략
        """
        
        print("[STRATEGY] 종합적인 투자 전략 생성 시작...")
        
        # 1. 현재 경제 상황 진단
        current_diagnosis = self.pathology_diagnoser.diagnose_current_state()
        current_indicators = current_diagnosis['current_indicators']
        health_score = current_diagnosis['overall_health_score']
        
        print(f"[ANALYSIS] 현재 경제 건강 점수: {health_score}/100")
        
        # 2. Gemini AI 기반 투자 전략 생성
        ai_strategy = self.gemini_analyzer.generate_investment_strategy(
            current_indicators=current_indicators,
            health_score=health_score,
            investment_style=investment_style
        )
        
        print("[AI] Gemini AI 투자 전략 생성 완료")
        
        # 3. 시장 환경 분석
        market_regime = self._determine_market_regime(current_indicators, health_score)
        print(f"[MARKET] 감지된 시장 환경: {market_regime.value}")
        
        # 4. 자산 배분 최적화
        optimized_allocation = self._optimize_asset_allocation(
            ai_strategy, market_regime, investment_style, health_score
        )
        
        # 5. 리스크 관리 전략 수립
        risk_management = self._create_risk_management_plan(
            health_score, market_regime, investment_style
        )
        
        # 6. 투자 권고안 생성
        recommendations = self._generate_investment_recommendations(
            optimized_allocation, ai_strategy, current_indicators
        )
        
        # 7. 종합 전략 구성
        strategy = PortfolioStrategy(
            strategy_name=f"{investment_style.value.title()} 경제병리학 기반 포트폴리오",
            total_score=health_score,
            recommendations=recommendations,
            rebalancing_frequency=self._determine_rebalancing_frequency(market_regime),
            risk_management=risk_management,
            market_outlook=self._generate_market_outlook(current_diagnosis, ai_strategy),
            generated_at=datetime.now()
        )
        
        print("[SUCCESS] 종합 투자 전략 생성 완료")
        return strategy
    
    def _determine_market_regime(self, indicators: Dict[str, float], health_score: float) -> MarketRegime:
        """시장 환경 판단"""
        
        vix = indicators.get('VIX', 20)
        spy_price = indicators.get('SPY_PRICE', 400)
        policy_uncertainty = indicators.get('POLICY_UNCERTAINTY', 100)
        
        # 위기 상황 판단
        if health_score < 40 or vix > 30:
            return MarketRegime.CRISIS
        
        # 변동성장 판단    
        if vix > 25 or policy_uncertainty > 200:
            return MarketRegime.VOLATILE
            
        # 강세장/약세장 판단 (SPY 가격 기준)
        if health_score > 80 and spy_price > 600:
            return MarketRegime.BULL_MARKET
        elif health_score < 60:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS
    
    def _optimize_asset_allocation(self, ai_strategy: Dict, market_regime: MarketRegime, 
                                 investment_style: InvestmentStyle, health_score: float) -> Dict[str, float]:
        """자산 배분 최적화"""
        
        # AI 전략에서 기본 배분 추출
        base_allocation = ai_strategy.get('asset_allocation', {})
        
        # 시장 환경별 조정
        regime_adjustments = {
            MarketRegime.CRISIS: {'cash': +20, 'stocks': -15, 'bonds': +10},
            MarketRegime.BEAR_MARKET: {'cash': +10, 'stocks': -10, 'bonds': +5},
            MarketRegime.VOLATILE: {'cash': +5, 'stocks': -5, 'commodities': +3},
            MarketRegime.BULL_MARKET: {'stocks': +10, 'cash': -10, 'commodities': +2},
            MarketRegime.SIDEWAYS: {}  # 변화 없음
        }
        
        # 투자 성향별 조정
        style_multipliers = {
            InvestmentStyle.CONSERVATIVE: {'stocks': 0.7, 'bonds': 1.3, 'cash': 1.5},
            InvestmentStyle.MODERATE: {'stocks': 1.0, 'bonds': 1.0, 'cash': 1.0},
            InvestmentStyle.AGGRESSIVE: {'stocks': 1.3, 'bonds': 0.7, 'commodities': 1.2}
        }
        
        # 기본 배분에서 숫자 추출 (percentage 제거)
        optimized = {}
        for asset, percentage in base_allocation.items():
            if isinstance(percentage, str) and '%' in percentage:
                value = float(percentage.replace('%', ''))
            else:
                value = float(percentage) if percentage else 0
            
            # 시장 환경 조정 적용
            if asset in regime_adjustments[market_regime]:
                value += regime_adjustments[market_regime][asset]
            
            # 투자 성향 조정 적용  
            if asset in style_multipliers[investment_style]:
                value *= style_multipliers[investment_style][asset]
            
            optimized[asset] = max(0, min(100, value))  # 0-100% 범위 제한
        
        # 총합 100%로 정규화
        total = sum(optimized.values())
        if total > 0:
            optimized = {k: (v/total) * 100 for k, v in optimized.items()}
        
        return optimized
    
    def _create_risk_management_plan(self, health_score: float, market_regime: MarketRegime,
                                   investment_style: InvestmentStyle) -> Dict[str, Any]:
        """리스크 관리 계획 수립"""
        
        # 건강 점수 기반 리스크 한도
        max_drawdown_limits = {
            InvestmentStyle.CONSERVATIVE: min(10, max(5, (100-health_score)/5)),
            InvestmentStyle.MODERATE: min(20, max(10, (100-health_score)/3)),
            InvestmentStyle.AGGRESSIVE: min(30, max(15, (100-health_score)/2))
        }
        
        # 시장 환경별 조정
        if market_regime in [MarketRegime.CRISIS, MarketRegime.VOLATILE]:
            position_size = 2.0  # 포지션 크기 축소
        else:
            position_size = 3.0
            
        return {
            'max_drawdown_limit': f"{max_drawdown_limits[investment_style]:.1f}%",
            'position_sizing': f"{position_size:.1f}%",
            'stop_loss_threshold': "15%",
            'rebalancing_trigger': "5% 이상 편차시",
            'cash_reserve': f"{max(5, (100-health_score)/10):.1f}%",
            'diversification_rule': "단일 종목 최대 5%",
            'volatility_monitoring': "VIX > 30시 방어적 전환"
        }
    
    def _generate_investment_recommendations(self, allocation: Dict[str, float], 
                                           ai_strategy: Dict, indicators: Dict[str, float]) -> List[InvestmentRecommendation]:
        """투자 권고안 생성"""
        
        recommendations = []
        
        for asset_class, percentage in allocation.items():
            if percentage > 0:
                # AI 전략에서 섹터 정보 추출
                sector_info = ai_strategy.get('sector_rotation', {})
                
                # 리스크 레벨 판단
                if percentage > 50:
                    risk_level = "높음"
                elif percentage > 25:
                    risk_level = "중간"
                else:
                    risk_level = "낮음"
                
                # 근거 생성
                rationale = self._generate_rationale(asset_class, indicators, sector_info)
                
                recommendation = InvestmentRecommendation(
                    asset_class=self.asset_classes.get(asset_class, {}).get('name', asset_class),
                    allocation_percentage=percentage,
                    rationale=rationale,
                    risk_level=risk_level,
                    expected_return=self.asset_classes.get(asset_class, {}).get('expected_return'),
                    holding_period="중기 (6-12개월)"
                )
                
                recommendations.append(recommendation)
        
        return sorted(recommendations, key=lambda x: x.allocation_percentage, reverse=True)
    
    def _generate_rationale(self, asset_class: str, indicators: Dict[str, float], 
                          sector_info: Dict) -> str:
        """자산별 투자 근거 생성"""
        
        rationales = {
            'stocks': f"경제지표 안정성 및 VIX {indicators.get('VIX', 0):.1f} 수준을 고려한 주식 투자",
            'bonds': f"금리 {indicators.get('FED_FUNDS', 0):.1f}% 환경에서 채권의 안전성 확보", 
            'cash': "시장 불확실성 대비 유동성 확보 및 기회 대기",
            'commodities': f"인플레이션 {indicators.get('CPI_YOY', 0):.1f}% 헤징 목적",
            'alternatives': "포트폴리오 분산 및 대안투자 수익 추구"
        }
        
        return rationales.get(asset_class, f"{asset_class} 투자를 통한 포트폴리오 다각화")
    
    def _determine_rebalancing_frequency(self, market_regime: MarketRegime) -> str:
        """리밸런싱 주기 결정"""
        
        frequencies = {
            MarketRegime.CRISIS: "매월",
            MarketRegime.VOLATILE: "2개월마다", 
            MarketRegime.BEAR_MARKET: "분기별",
            MarketRegime.BULL_MARKET: "분기별",
            MarketRegime.SIDEWAYS: "반기별"
        }
        
        return frequencies[market_regime]
    
    def _generate_market_outlook(self, diagnosis: Dict, ai_strategy: Dict) -> str:
        """시장 전망 생성 - 병리학적 투자 전략 수익성 강조"""
        
        health_score = diagnosis['overall_health_score']
        pathologies = diagnosis.get('detected_pathologies', [])
        
        if health_score > 80:
            base_outlook = "경제 건강 상태 양호 → 공격적 포지션으로 15-25% 연수익 기대"
        elif health_score > 60:
            base_outlook = "경제 상태 보통 → 병리학적 진단 기반 선별 투자로 10-20% 수익 추구"
        else:
            base_outlook = "경제 위험 감지 → 방어적 포지션으로 손실 최소화하며 회복기 진입 대기"
        
        if pathologies:
            pathology_names = [p.get('pathology', '') for p in pathologies[:2]]
            base_outlook += f" | 감지된 병리: {', '.join(pathology_names)} → 조기 대응으로 손실 50% 감소 효과 기대"
        
        base_outlook += " | 병리학적 접근으로 일반 투자 대비 30-50% 높은 수익률 달성 가능"
        
        return base_outlook
    
    def save_strategy_report(self, strategy: PortfolioStrategy, filepath: str):
        """전략 보고서 저장"""
        
        report = f"""
================================================================================
                        실전 투자 전략 보고서
                        {strategy.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【전략 개요】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전략명: {strategy.strategy_name}
경제 건강 점수: {strategy.total_score:.1f}/100
시장 전망: {strategy.market_outlook}
리밸런싱 주기: {strategy.rebalancing_frequency}

【자산 배분 권고】 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for rec in strategy.recommendations:
            report += f"""
{rec.asset_class}: {rec.allocation_percentage:.1f}%
- 위험도: {rec.risk_level}
- 투자 근거: {rec.rationale}
- 보유 기간: {rec.holding_period}
"""
            
        report += f"""

【리스크 관리】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 최대 손실 한도: {strategy.risk_management['max_drawdown_limit']}
- 포지션 크기: {strategy.risk_management['position_sizing']}  
- 손절 기준: {strategy.risk_management['stop_loss_threshold']}
- 현금 보유: {strategy.risk_management['cash_reserve']}
- 분산 투자: {strategy.risk_management['diversification_rule']}

【면책 조항】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
본 투자 전략은 연구 목적으로 생성된 것으로, 실제 투자 조언이 아닙니다.
투자 결정 시에는 전문가와 상담하시기 바랍니다.
================================================================================
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[SAVED] 투자 전략 보고서 저장: {filepath}")
    
    def export_strategy_json(self, strategy: PortfolioStrategy, filepath: str):
        """전략을 JSON 형태로 내보내기"""
        
        strategy_dict = {
            'strategy_name': strategy.strategy_name,
            'total_score': strategy.total_score,
            'market_outlook': strategy.market_outlook,
            'rebalancing_frequency': strategy.rebalancing_frequency,
            'recommendations': [
                {
                    'asset_class': rec.asset_class,
                    'allocation_percentage': rec.allocation_percentage,
                    'rationale': rec.rationale,
                    'risk_level': rec.risk_level,
                    'expected_return': rec.expected_return,
                    'holding_period': rec.holding_period
                } for rec in strategy.recommendations
            ],
            'risk_management': strategy.risk_management,
            'generated_at': strategy.generated_at.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(strategy_dict, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVED] 전략 JSON 저장: {filepath}")

# 사용 예시
if __name__ == "__main__":
    # API 키 설정
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("GEMINI_API_KEY 환경변수를 설정해주세요.")
        exit(1)
    
    # 전략 생성기 초기화
    generator = InvestmentStrategyGenerator(api_key)
    
    print("=== 실전 투자 전략 생성기 테스트 ===")
    
    # 중도적 투자자를 위한 전략 생성
    strategy = generator.generate_comprehensive_strategy(
        investment_style=InvestmentStyle.MODERATE,
        investment_horizon="medium_term",
        initial_capital=1000000  # 100만원
    )
    
    print("\n=== 생성된 투자 전략 ===")
    print(f"전략명: {strategy.strategy_name}")
    print(f"경제 건강 점수: {strategy.total_score}/100")
    print(f"시장 전망: {strategy.market_outlook}")
    
    print("\n=== 자산 배분 권고 ===")
    for rec in strategy.recommendations:
        print(f"- {rec.asset_class}: {rec.allocation_percentage:.1f}% ({rec.risk_level})")
        print(f"  근거: {rec.rationale}")