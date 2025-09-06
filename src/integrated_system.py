"""
통합 경제병리학 AI 시스템 - 최종 버전
Integrated Economic Pathology AI System - Final Version
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
import json

# 프로젝트 모듈 import
from realtime_economic_pathology import RealTimePathologyDiagnoser
from gemini_ai_analyzer import GeminiEconomicAnalyzer, InvestmentStyle
from investment_strategy_generator import InvestmentStrategyGenerator
from data_pipeline_validator import DataPipelineValidator

class IntegratedEconomicSystem:
    """통합 경제 건강 관리 시스템"""
    
    def __init__(self, gemini_api_key: str, output_dir: str = "output"):
        """
        시스템 초기화
        
        Args:
            gemini_api_key: Gemini AI API 키
            output_dir: 출력 디렉토리
        """
        self.output_dir = output_dir
        
        # 핵심 모듈 초기화
        self.pathology_diagnoser = RealTimePathologyDiagnoser()
        self.gemini_analyzer = GeminiEconomicAnalyzer(gemini_api_key)
        self.strategy_generator = InvestmentStrategyGenerator(gemini_api_key)
        self.data_validator = DataPipelineValidator()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print("[SYSTEM] 통합 경제병리학 AI 시스템 초기화 완료")
    
    def run_comprehensive_analysis(self, investment_style: InvestmentStyle = InvestmentStyle.MODERATE) -> Dict[str, Any]:
        """
        종합적인 경제 분석 실행
        
        Args:
            investment_style: 투자 성향
            
        Returns:
            Dict: 전체 분석 결과
        """
        
        print("\n" + "="*80)
        print("         통합 경제병리학 AI 시스템 - 종합 분석 시작")
        print("="*80)
        
        analysis_results = {
            'timestamp': datetime.now(),
            'system_version': '2.0',
            'analysis_id': f"ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        try:
            # 1. 데이터 파이프라인 검증
            print("\n[1/6] 데이터 파이프라인 검증 중...")
            validation_results = self.data_validator.validate_entire_pipeline()
            analysis_results['data_validation'] = validation_results
            
            print(f"   [OK] 데이터 신뢰도: {validation_results['overall_reliability']:.1f}%")
            # pass_rate가 없는 경우를 대비한 안전 처리
            pass_rate = validation_results.get('pass_rate', 0)
            if pass_rate > 0:
                print(f"   [OK] 검증 통과율: {pass_rate:.1f}%")
            else:
                print(f"   [OK] 검증 단계: {len(validation_results.get('details', []))}개 완료")
            
            # 2. 실시간 경제 진단
            print("\n[2/6] 실시간 경제병리 진단 중...")
            pathology_diagnosis = self.pathology_diagnoser.diagnose_current_state()
            analysis_results['pathology_diagnosis'] = pathology_diagnosis
            
            health_score = pathology_diagnosis['overall_health_score']
            risk_level = pathology_diagnosis['risk_level']
            print(f"   [OK] 경제 건강 점수: {health_score}/100")
            print(f"   [OK] 위험 수준: {risk_level}")
            
            # 3. Gemini AI 경제 분석
            print("\n[3/6] Gemini AI 경제 분석 중...")
            current_indicators = pathology_diagnosis['current_indicators']
            
            ai_analysis = self.gemini_analyzer.analyze_economic_health(
                current_indicators=current_indicators,
                validation_report=validation_results
            )
            analysis_results['ai_analysis'] = {
                'diagnosis': ai_analysis.diagnosis,
                'prediction': ai_analysis.prediction,
                'risk_assessment': ai_analysis.risk_assessment,
                'confidence_score': ai_analysis.confidence_score
            }
            
            print(f"   [OK] AI 진단: {ai_analysis.diagnosis[:50]}...")
            print(f"   [OK] AI 신뢰도: {ai_analysis.confidence_score:.2f}")
            
            # 4. AI 기반 투자 전략 생성
            print("\n[4/6] AI 투자 전략 생성 중...")
            ai_investment_strategy = self.gemini_analyzer.generate_investment_strategy(
                current_indicators=current_indicators,
                health_score=health_score,
                investment_style=investment_style
            )
            analysis_results['ai_investment_strategy'] = ai_investment_strategy
            
            # 5. 실전 투자 전략 생성
            print("\n[5/6] 실전 투자 전략 생성 중...")
            practical_strategy = self.strategy_generator.generate_comprehensive_strategy(
                investment_style=investment_style,
                investment_horizon="medium_term"
            )
            analysis_results['practical_strategy'] = practical_strategy
            
            print(f"   [OK] 전략명: {practical_strategy.strategy_name}")
            print(f"   [OK] 권고 자산 수: {len(practical_strategy.recommendations)}개")
            
            # 6. 정책 권고안 생성 (정부/중앙은행용)
            print("\n[6/6] 정책 권고안 생성 중...")
            detected_pathologies = pathology_diagnosis.get('detected_pathologies', [])
            overall_health_score = pathology_diagnosis.get('overall_health_score', 79)
            risk_level = pathology_diagnosis.get('risk_level', 'ELEVATED')
            
            policy_recommendations = self.gemini_analyzer.generate_policy_recommendations(
                current_indicators=current_indicators,
                detected_pathologies=detected_pathologies,
                overall_health_score=overall_health_score,
                risk_level=risk_level,
                ai_analysis=ai_analysis
            )
            analysis_results['policy_recommendations'] = policy_recommendations
            
            print("   [OK] 정책 권고안 생성 완료")
            
            # 결과 저장
            self._save_comprehensive_results(analysis_results)
            
            print("\n" + "="*80)
            print("             종합 분석 완료 - 모든 보고서 생성됨")
            print("="*80)
            
            return analysis_results
            
        except Exception as e:
            print(f"\n[ERROR] 분석 중 오류 발생: {e}")
            analysis_results['error'] = str(e)
            return analysis_results
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """종합 결과 저장"""
        
        timestamp_str = results['timestamp'].strftime("%Y%m%d_%H%M%S")
        
        # 1. 실시간 진단 보고서 저장
        if 'pathology_diagnosis' in results:
            diagnosis_file = f"{self.output_dir}/realtime_diagnosis_report_{timestamp_str}.txt"
            self.pathology_diagnoser.save_diagnosis_report(
                results['pathology_diagnosis'], 
                diagnosis_file
            )
        
        # 2. 실시간 진단 대시보드 저장
        if 'pathology_diagnosis' in results:
            dashboard = self.pathology_diagnoser.create_realtime_dashboard(
                results['pathology_diagnosis']
            )
            dashboard_file = f"{self.output_dir}/realtime_dashboard_{timestamp_str}.png"
            dashboard.write_image(dashboard_file, width=1200, height=800)
            print(f"[SAVED] 실시간 대시보드: {dashboard_file}")
        
        # 3. 실전 투자 전략 보고서 저장
        if 'practical_strategy' in results:
            strategy_file = f"{self.output_dir}/investment_strategy_{timestamp_str}.txt"
            self.strategy_generator.save_strategy_report(
                results['practical_strategy'],
                strategy_file
            )
            
            # JSON 형태로도 저장
            strategy_json_file = f"{self.output_dir}/investment_strategy_{timestamp_str}.json"
            self.strategy_generator.export_strategy_json(
                results['practical_strategy'],
                strategy_json_file
            )
        
        # 4. 정책 권고안 저장
        if 'policy_recommendations' in results:
            policy_file = f"{self.output_dir}/policy_recommendations_{timestamp_str}.txt"
            with open(policy_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("정책 당국 권고안\n")
                f.write(f"{results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                f.write(results['policy_recommendations'])
            print(f"[SAVED] 정책 권고안: {policy_file}")
        
        # 5. 데이터 검증 보고서 저장
        if 'data_validation' in results:
            validation_file = f"{self.output_dir}/data_validation_{timestamp_str}.json"
            with open(validation_file, 'w', encoding='utf-8') as f:
                # datetime 객체를 문자열로 변환 (안전 처리)
                validation_data = results['data_validation'].copy()
                if 'timestamp' in validation_data:
                    timestamp_val = validation_data['timestamp']
                    if hasattr(timestamp_val, 'isoformat'):
                        validation_data['timestamp'] = timestamp_val.isoformat()
                    elif isinstance(timestamp_val, str):
                        validation_data['timestamp'] = timestamp_val
                    else:
                        validation_data['timestamp'] = str(timestamp_val)
                json.dump(validation_data, f, ensure_ascii=False, indent=2)
            print(f"[SAVED] 데이터 검증 보고서: {validation_file}")
        
        # 6. 종합 결과 요약 저장
        summary_file = f"{self.output_dir}/comprehensive_analysis_summary_{timestamp_str}.txt"
        self._create_executive_summary(results, summary_file)
        
        print(f"\n[SUCCESS] 모든 보고서가 {self.output_dir}/ 디렉토리에 저장되었습니다.")
    
    def _create_executive_summary(self, results: Dict[str, Any], filepath: str):
        """경영진/정책담당자용 요약 보고서"""
        
        # 주요 지표 추출
        health_score = results.get('pathology_diagnosis', {}).get('overall_health_score', 0)
        risk_level = results.get('pathology_diagnosis', {}).get('risk_level', 'UNKNOWN')
        data_reliability = results.get('data_validation', {}).get('overall_reliability', 0)
        
        ai_confidence = results.get('ai_analysis', {}).get('confidence_score', 0)
        detected_pathologies = results.get('pathology_diagnosis', {}).get('detected_pathologies', [])
        
        # 투자 전략 요약
        strategy = results.get('practical_strategy')
        top_allocations = []
        if strategy and hasattr(strategy, 'recommendations'):
            top_allocations = [(rec.asset_class, rec.allocation_percentage) 
                             for rec in strategy.recommendations[:3]]
        
        summary = f"""
================================================================================
                        경제병리학 AI 시스템 - 경영진 요약 보고서
                        Executive Summary - Economic Pathology AI System
================================================================================

분석 일시: {results['timestamp'].strftime('%Y년 %m월 %d일 %H:%M:%S')}
분석 ID: {results['analysis_id']}
시스템 버전: {results['system_version']}

【핵심 지표 현황】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
경제 건강 점수: {health_score:.1f}/100
위험 수준: {risk_level}
데이터 신뢰도: {data_reliability:.1f}%
AI 분석 신뢰도: {ai_confidence:.1f}

【진단 결과】 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        if detected_pathologies:
            summary += f"\n감지된 병리 상태: {len(detected_pathologies)}개"
            for pathology in detected_pathologies[:3]:  # 상위 3개만 표시
                name = pathology.get('pathology', 'Unknown').replace('_', ' ')
                severity = pathology.get('severity_score', 0)
                summary += f"\n- {name}: 심각도 {severity:.1f}/10"
        else:
            summary += "\n[OK] 특별한 병리 상태 감지되지 않음"

        summary += f"""

【투자 전략 권고】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        if top_allocations:
            summary += f"\n권고 자산 배분 (상위 3개):"
            for asset, percentage in top_allocations:
                summary += f"\n- {asset}: {percentage:.1f}%"
        else:
            summary += "\n투자 전략 데이터 없음"

        if 'ai_analysis' in results:
            ai_diagnosis = results['ai_analysis'].get('diagnosis', '')[:100]
            summary += f"""

【AI 분석 요약】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ai_diagnosis}{'...' if len(ai_diagnosis) >= 100 else ''}"""

        summary += f"""

【권고사항】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 데이터 품질: {'우수' if data_reliability > 90 else '보통' if data_reliability > 70 else '주의 필요'}
2. 경제 상태: {'양호' if health_score > 80 else '보통' if health_score > 60 else '주의 필요'}
3. 투자 포지션: {'적극적' if health_score > 75 else '중립적' if health_score > 50 else '방어적'}

【상세 보고서】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 실시간 진단: realtime_diagnosis_report_{results['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt
- 투자 전략: investment_strategy_{results['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt
- 정책 권고: policy_recommendations_{results['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt
- 데이터 검증: data_validation_{results['timestamp'].strftime('%Y%m%d_%H%M%S')}.json

【면책 조항】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
본 분석은 연구 목적으로 생성된 것으로, 실제 투자나 정책 결정에 직접 사용할 
수 없습니다. 모든 의사결정은 추가적인 전문가 검토를 거쳐야 합니다.

================================================================================
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"[SAVED] 경영진 요약 보고서: {filepath}")

# 메인 실행 함수
def main():
    """메인 실행 함수"""
    
    # Gemini API 키 설정
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY 환경변수를 설정해주세요.")
        sys.exit(1)
    
    # 시스템 초기화
    system = IntegratedEconomicSystem(
        gemini_api_key=GEMINI_API_KEY,
        output_dir="output"
    )
    
    print("통합 경제병리학 AI 시스템을 시작합니다...")
    
    # 종합 분석 실행
    results = system.run_comprehensive_analysis(
        investment_style=InvestmentStyle.MODERATE
    )
    
    if 'error' not in results:
        print("\n[SUCCESS] 분석이 성공적으로 완료되었습니다!")
        print(f"[FILES] 결과 파일들이 'output/' 디렉토리에 저장되었습니다.")
        
        # 주요 결과 출력
        health_score = results.get('pathology_diagnosis', {}).get('overall_health_score', 0)
        print(f"\n[SCORE] 경제 건강 점수: {health_score}/100")
        
        if 'practical_strategy' in results:
            strategy_name = results['practical_strategy'].strategy_name
            print(f"[STRATEGY] 생성된 투자 전략: {strategy_name}")
    else:
        print(f"\n[ERROR] 분석 중 오류가 발생했습니다: {results['error']}")

if __name__ == "__main__":
    main()