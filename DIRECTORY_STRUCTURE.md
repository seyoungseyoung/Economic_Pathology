# 경제병리학 연구 시스템 - 디렉터리 구조

> **정리 완료일**: 2025년 9월 6일  
> **정리 내용**: 중복 파일 제거, 핵심 모듈만 유지, 통합 실행 시스템 구축

---

## 📁 전체 디렉터리 구조

```
body_economy/
├── 📄 main.py                                    # 🎯 통합 실행 스크립트 (NEW)
├── 📄 dashboard.py                               # 🌐 Streamlit 웹 대시보드
├── 📄 test_simple_ai.py                          # 🧪 Gemini AI 간단 테스트
├── 📄 test_gemini_models.py                      # 🧪 Gemini 모델 호환성 테스트
├── 📄 .env                                       # 🔐 API 키 설정 파일
├── 📄 README.md                                  # 📖 종합 문서 (통합됨)
├── 📄 DIRECTORY_STRUCTURE.md                     # 📋 본 문서
│
├── 📂 src/                                       # 💻 핵심 소스 코드 (정리됨)
│   ├── 📄 __init__.py                            # 📦 패키지 초기화
│   ├── 📄 integrated_system.py                   # 🤖 AI 통합 시스템 (MAIN)
│   ├── 📄 realtime_economic_pathology.py         # ⏱️ 실시간 진단 시스템 (MAIN)
│   ├── 📄 unified_economic_pathology_research.py # 📊 대시보드 생성 시스템 (MAIN)
│   ├── 📄 gemini_ai_analyzer.py                  # 🧠 Gemini AI 분석 엔진
│   ├── 📄 investment_strategy_generator.py       # 💰 투자 전략 생성기
│   └── 📄 data_pipeline_validator.py             # ✅ 데이터 검증 시스템
│
├── 📂 output/                                    # 📈 분석 결과 파일
│   ├── 📄 master_dashboard.png                   # 📊 통합 대시보드
│   ├── 📄 economic_pathology_diagnosis.png       # 🏥 실시간 진단 차트
│   ├── 📄 integrated_analysis_dashboard.png      # 🤖 AI 분석 대시보드
│   ├── 📄 economic_health_report.txt             # 📋 건강 상태 보고서
│   ├── 📄 investment_strategy.txt                # 💼 투자 전략 보고서
│   ├── 📄 validation_report.txt                  # ✅ 데이터 검증 보고서
│   └── 📄 comprehensive_analysis_summary.txt     # 📄 종합 분석 요약
│
├── 📂 archive/                                   # 🗃️ 아카이브 폴더
│   ├── 📂 old_tests/                             # 🧪 이전 테스트 파일들
│   ├── 📂 test_files/                            # 🧪 테스트 관련 파일들
│   └── 📂 reports/                               # 📄 이전 보고서들
│
└── 📂 useds_env/                                 # 🐍 Python 가상환경 (conda)
    └── 📂 Lib/site-packages/                     # 📦 설치된 패키지들
```

---

## 🎯 핵심 실행 파일

### 1. **main.py** - 통합 실행 스크립트 ⭐
```python
# 사용법
python main.py

# 선택 옵션:
# 1. 병리학적 분석만 실행
# 2. 통합 시스템 분석만 실행
# 3. 통합 대시보드만 생성
# 4. 모든 분석 동시 실행 (권장) ⭐⭐⭐
# 5. 병리학적 분석 + 통합 대시보드
```

### 2. **src/integrated_system.py** - AI 통합 시스템
- Gemini AI 연동
- 실시간 진단 + 투자 전략 통합
- 종합 분석 결과 생성

### 3. **src/realtime_economic_pathology.py** - 실시간 진단
- FRED, Yahoo Finance, IMF 데이터 연동
- 5개 병리 유형 실시간 진단
- 경제 건강 점수 산출

### 4. **src/unified_economic_pathology_research.py** - 대시보드 생성
- 96년 역사 분석 (1929-2025)
- 12개 위기 패턴 시각화
- 마스터 대시보드 생성

---

## 🗂️ 정리 작업 내용

### ✅ 제거된 파일들 (중복/미사용)

#### 실행 스크립트 (main.py로 통합)
- ❌ `run_analysis.py` - 복잡한 레거시 실행 스크립트
- ❌ `run_analysis_simple.py` - 간단한 실행 스크립트  
- ❌ `run_integrated_analysis.py` - 통합 분석 실행 스크립트

#### src/ 디렉터리 (17개 → 6개로 축소)
- ❌ `academic_economic_pathology.py` - 학술적 정의 (통합됨)
- ❌ `adaptive_thresholds.py` - 적응적 임계값 (미사용)
- ❌ `crisis_debug_logger.py` - 디버그 로거 (미사용)
- ❌ `crisis_pattern_database.py` - 위기 패턴 DB (통합됨)
- ❌ `data_fetcher.py` - 기본 데이터 수집 (대체됨)
- ❌ `diagnostic_system.py` - 기본 진단 (통합됨)
- ❌ `economic_health_analysis.py` - 건강 분석 (통합됨)
- ❌ `enhanced_data_fetcher.py` - 강화 데이터 수집 (통합됨)
- ❌ `enhanced_vital_signs.py` - 강화 바이탈 사인 (통합됨)
- ❌ `financial_data_preprocessor.py` - 데이터 전처리 (통합됨)
- ❌ `integrated_diagnostic_system.py` - 통합 진단 (대체됨)
- ❌ `market_regime_classifier.py` - 시장 체제 분류 (미사용)
- ❌ `medical_crisis_analysis.py` - 의료 위기 분석 (미사용)
- ❌ `medical_crisis_research.py` - 의료 위기 연구 (미사용)
- ❌ `score_normalization_system.py` - 점수 정규화 (통합됨)
- ❌ `validation_system.py` - 검증 시스템 (대체됨)
- ❌ `vital_signs.py` - 바이탈 사인 (통합됨)

#### 문서 파일 (README.md로 통합)
- ❌ `FINAL_RESEARCH_REPORT.md` - 최종 연구 보고서 (통합됨)
- ❌ `RESEARCH_SYSTEM_OVERVIEW.md` - 시스템 개요 (통합됨)

---

## 📊 현재 시스템 아키텍처

### 데이터 흐름
```
[외부 API] → [실시간 진단] → [AI 분석] → [투자 전략] → [대시보드]
    ↓              ↓            ↓           ↓           ↓
  FRED API    병리학적 분류   Gemini AI   자동 전략    시각화
  Yahoo Fin   건강 점수      신뢰도       리스크 관리   보고서
  IMF SDMX    위험 수준      권고사항     포트폴리오    파일 출력
```

### 모듈 간 관계
```
main.py
├── realtime_economic_pathology.py
│   ├── 📡 FRED/Yahoo/IMF API 연동
│   ├── 🏥 5개 병리 유형 진단
│   └── 📊 건강 점수 산출
│
├── integrated_system.py
│   ├── 🤖 Gemini AI 분석
│   ├── 💰 투자 전략 생성
│   └── 📋 종합 보고서 작성
│
└── unified_economic_pathology_research.py
    ├── 📈 역사적 패턴 분석
    ├── 🎨 대시보드 생성
    └── 📄 연구 보고서 출력
```

---

## 🔧 개발자 가이드

### 새 기능 추가 시 고려사항

#### 1. 모듈 추가
- `src/` 디렉터리에 추가
- `main.py`에서 호출 방식 결정
- 기존 3개 메인 모듈과의 통합성 고려

#### 2. 새 데이터 소스 추가
- `realtime_economic_pathology.py`에 API 연동
- `data_pipeline_validator.py`에 검증 로직 추가

#### 3. 새 병리 유형 추가
- 5개 기본 유형 외 추가시 전체 시스템 영향도 검토
- 색상 코딩 및 시각화 업데이트 필요

#### 4. AI 모델 교체/추가
- `gemini_ai_analyzer.py`에서 모델 변경
- API 키 관리 및 호환성 테스트 필수

---

## 📝 유지보수 체크리스트

### 정기 점검 사항
- [ ] **API 키 유효성** - FRED, Gemini 키 만료 확인
- [ ] **데이터 검증 결과** - 신뢰도 90% 이상 유지
- [ ] **출력 파일 정리** - output/ 폴더 용량 관리
- [ ] **아카이브 정리** - archive/ 폴더 불필요 파일 정리

### 성능 최적화 포인트
- [ ] **병렬 처리 활용** - main.py의 Threading 최적화
- [ ] **캐싱 시스템** - API 호출 결과 캐싱
- [ ] **메모리 관리** - 대용량 데이터 처리시 메모리 최적화

---

## 🚀 향후 개발 방향

### Phase 1: 안정화 (완료)
- ✅ 핵심 모듈 통합 및 중복 제거
- ✅ 데이터 신뢰도 92.1% 달성
- ✅ AI 통합 완료

### Phase 2: 확장성 (계획)
- 🔄 웹 기반 대시보드 구축
- 🔄 다국가 데이터 확장 (유럽, 아시아)
- 🔄 고빈도 데이터 실시간 처리

### Phase 3: 지능화 (계획)
- 🔄 다중 AI 모델 앙상블
- 🔄 자동 리밸런싱 시스템
- 🔄 사용자 맞춤형 알림 시스템

---

*본 문서는 시스템 정리 완료 후 디렉터리 구조를 문서화한 것입니다. 새로운 파일 추가나 구조 변경시 본 문서도 함께 업데이트하시기 바랍니다.*

---

**마지막 업데이트**: 2025년 9월 6일  
**정리 담당**: Claude Code AI Assistant  
**정리 결과**: 50+ 파일 → 20 핵심 파일, 모든 기능 유지