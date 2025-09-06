"""
경제병리학 연구 시스템 - 통합 메인 실행 파일
"""

import sys
import os
import threading
import time
from datetime import datetime

# 로컬 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_pathology_analysis():
    """병리학적 분석 실행"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 병리학적 분석 시작...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "src/realtime_economic_pathology.py"], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 병리학적 분석 완료")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 병리학적 분석 오류: {result.stderr}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 병리학적 분석 오류: {e}")

def run_integrated_analysis():
    """통합 시스템 분석 실행"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 통합 시스템 분석 시작...")
    try:
        from integrated_system import main as integrated_main
        integrated_main()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 통합 시스템 분석 완료")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 통합 시스템 분석 오류: {e}")

def run_unified_dashboard():
    """통합 대시보드 생성"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 통합 대시보드 생성 시작...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "src/unified_economic_pathology_research.py"], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 통합 대시보드 생성 완료")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 통합 대시보드 생성 오류: {result.stderr}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 통합 대시보드 생성 오류: {e}")

def main():
    """메인 실행 함수 - 모든 분석을 동시 실행"""
    print("=" * 80)
    print("경제병리학 연구 시스템 - 통합 분석")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 사용자 선택
    print("실행 옵션을 선택하세요:")
    print("1. 병리학적 분석만 실행")
    print("2. 통합 시스템 분석만 실행") 
    print("3. 통합 대시보드만 생성")
    print("4. 모든 분석 동시 실행 (권장)")
    print("5. 병리학적 분석 + 통합 대시보드")
    
    try:
        choice = input("\n선택하세요 (1-5, 기본값 4): ").strip() or "4"
    except KeyboardInterrupt:
        print("\n\n실행이 취소되었습니다.")
        return
    
    print()
    
    if choice == "1":
        run_pathology_analysis()
        
    elif choice == "2":
        run_integrated_analysis()
        
    elif choice == "3":
        run_unified_dashboard()
        
    elif choice == "4":
        # 모든 분석을 병렬로 실행
        threads = []
        
        # 각 분석을 별도 스레드에서 실행
        thread1 = threading.Thread(target=run_pathology_analysis, name="PathologyAnalysis")
        thread2 = threading.Thread(target=run_integrated_analysis, name="IntegratedAnalysis")
        thread3 = threading.Thread(target=run_unified_dashboard, name="UnifiedDashboard")
        
        threads.extend([thread1, thread2, thread3])
        
        # 모든 스레드 시작
        for thread in threads:
            thread.start()
            time.sleep(1)  # 약간의 간격으로 시작
        
        print("모든 분석이 병렬로 실행 중입니다...")
        print("각 분석의 진행 상황을 위에서 확인하세요.\n")
        
        # 모든 스레드가 완료될 때까지 대기
        for thread in threads:
            thread.join()
            
    elif choice == "5":
        # 병리학적 분석과 대시보드만 실행
        threads = []
        
        thread1 = threading.Thread(target=run_pathology_analysis, name="PathologyAnalysis")
        thread3 = threading.Thread(target=run_unified_dashboard, name="UnifiedDashboard")
        
        threads.extend([thread1, thread3])
        
        for thread in threads:
            thread.start()
            time.sleep(1)
        
        print("병리학적 분석과 대시보드 생성이 병렬로 실행 중입니다...")
        
        for thread in threads:
            thread.join()
            
    else:
        print("잘못된 선택입니다. 기본값(4)으로 실행합니다.")
        main()  # 재귀 호출로 다시 선택
        return
    
    print()
    print("=" * 80)
    print("모든 분석이 완료되었습니다!")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("생성된 파일들:")
    print("- output/economic_pathology_diagnosis.png")
    print("- output/integrated_analysis_dashboard.png") 
    print("- output/master_dashboard.png")
    print("- output/economic_health_report.txt")
    print("- output/validation_report.txt")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n시스템 오류: {e}")
        print("오류가 지속되면 API 키를 확인하거나 인터넷 연결을 점검하세요.")