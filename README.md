# 경제병리학 AI 시스템 (Economic Pathology AI System)

## 개요

경제 위기를 의학적 질병으로 분석하여 AI 기반 진단, 투자전략, 정책권고를 제공하는 통합 시스템입니다. 2025년 병리학적 분포 분석을 통한 종합적 AI 처방으로 객관적이고 정확한 경제 분석을 제공합니다.

## 🚀 주요 특징

### 🏥 의학적 접근법
- **5대 병리 유형**: 순환계, 대사, 구조적, 면역, 신경 질환으로 경제 위기 분류
- **96년간 역사 분석**: 12개 주요 위기 사례 기반 패턴 학습
- **실시간 진단**: 현재 경제 상태의 병리학적 진단

### 🤖 Gemini AI 통합
- **종합적 분석**: 단일 병리가 아닌 2025년 전체 병리학적 분포 기반 처방
- **투자전략 생성**: 병리학적 진단에 따른 AI 투자 전략
- **정책권고**: 정부/중앙은행을 위한 구체적 정책 처방전

### 📊 고도화된 데이터 시스템
- **6단계 검증**: 99% 신뢰도의 데이터 파이프라인
- **실시간 연동**: FRED, Yahoo Finance, IMF 데이터 실시간 수집
- **종합 대시보드**: 통합 시각화 및 분석 결과

## 📋 시스템 요구사항

- Python 3.8 이상
- Windows/macOS/Linux 지원
- 인터넷 연결 (API 데이터 수집용)

## ⚙️ 설치 및 설정

### 1. 저장소 클론
```bash
git clone https://github.com/seyoungseyoung/Economic_Pathology.git
cd Economic_Pathology
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정
다음 환경변수를 설정하세요:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:FRED_API_KEY="your_fred_api_key_here"  # 선택사항
```

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your_gemini_api_key_here
set FRED_API_KEY=your_fred_api_key_here
```

**macOS/Linux:**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export FRED_API_KEY="your_fred_api_key_here"  # 선택사항
```

#### API 키 발급 방법:
- **Gemini API**: [Google AI Studio](https://aistudio.google.com/app/apikey)에서 무료 발급
- **FRED API**: [FRED 웹사이트](https://fred.stlouisfed.org/docs/api/)에서 무료 발급 (선택사항, 더 정확한 데이터를 위해 권장)

## 🎯 사용법

### 메인 시스템 실행
```bash
python main.py
```

실행하면 다음 옵션을 선택할 수 있습니다:

```
경제병리학 분석 시스템 - 메뉴 선택
================================================================================
현재 시간: 2025-09-07 00:32:44

원하는 옵션을 선택하세요:
1. 경제병리학 진단만 실행
2. 통합 시스템 진단만 실행  
3. 통합 대시보드만 실행
4. 전체 분석 실행 실행 (권장)
5. 경제병리학 분석 + 통합 대시보드

선택하세요 (1-5, 기본값 4):
```

**권장**: 옵션 4 (전체 분석 실행)를 선택하면 모든 기능을 체험할 수 있습니다.

### 대시보드 실행 (웹 인터페이스)
```bash
streamlit run dashboard.py
```
웹 브라우저에서 http://localhost:8501로 접속하여 대화형 대시보드를 이용할 수 있습니다.

## 📁 프로젝트 구조

```
Economic_Pathology/
├── main.py                              # 🎯 메인 실행 파일
├── dashboard.py                         # 🌐 Streamlit 웹 대시보드
├── requirements.txt                     # 📦 패키지 의존성
├── .gitignore                          # 🔒 Git 무시 파일
├── README.md                           # 📖 본 문서
├── src/                                # 💻 핵심 소스코드
│   ├── __init__.py
│   ├── integrated_system.py            # 🤖 통합 AI 시스템 (메인 엔진)
│   ├── gemini_ai_analyzer.py           # 🧠 Gemini AI 분석기
│   ├── realtime_economic_pathology.py  # ⚡ 실시간 경제 병리 진단
│   ├── investment_strategy_generator.py # 💰 투자 전략 생성기
│   ├── data_pipeline_validator.py      # ✅ 데이터 파이프라인 검증
│   └── unified_economic_pathology_research.py # 📊 통합 연구 시스템
└── output/                             # 📋 분석 결과물 저장소
    ├── *.png                          # 📈 차트 및 대시보드
    ├── *.txt                          # 📄 분석 보고서
    ├── *.json                         # 📊 구조화된 데이터
    └── *.xlsx                         # 📊 엑셀 보고서
```

## 🏥 병리학적 분류 체계

### 5대 경제 병리 유형

| 병리 유형 | 설명 | 주요 지표 | 역사적 사례 |
|---------|------|----------|------------|
| 🔵 **순환계 (CIRCULATORY)** | 유동성 경색, 신용 스프레드 확대 | VIX, SOFR-OIS | 1987 블랙먼데이, 2023 뱅킹 위기 |
| 🔴 **대사 (METABOLIC)** | 인플레이션/디플레이션 불균형 | CPI, 실업률 | 1970s 스태그플레이션, 2022 인플레이션 |
| 🟠 **구조적 (STRUCTURAL)** | 부채 과다, 자산 버블 | 부채/GDP, 자산가격 | 1929 대공황, 2008 금융위기 |
| 🟣 **면역 (IMMUNE)** | 시스템 리스크, 금융 전염 | 금융스트레스 지수 | 1997 아시아 위기, 2020 코로나 |
| 🟢 **신경 (NEURAL)** | 정책 불확실성, 의사결정 마비 | 정책불확실성 지수 | 1998 러시아 위기 |

### 96년간 위기 통계
- **구조적 위기**: 4회 | 평균 GDP 타격 -12.8% | 평균 회복 5.3년 | 완전회복률 62.5%
- **순환계 위기**: 2회 | 평균 GDP 타격 -3.8% | 평균 회복 1.6년 | 완전회복률 83%
- **대사 위기**: 3회 | 평균 GDP 타격 -6.2% | 평균 회복 2.0년 | 완전회복률 85%

## 🤖 AI 시스템 특징

### Gemini AI 통합 분석
- **종합적 진단**: 2025년 전체 병리학적 분포 기반 분석
- **역사적 맥락**: 96년간 12개 위기 사례와 현재 상황 비교
- **객관적 처방**: 단일 기준이 아닌 종합적 데이터 기반 권고

### 6단계 데이터 검증 시스템
1. **데이터 수집 검증** (Stage 1): API 연결, 데이터 완정성
2. **데이터 전처리** (Stage 2): Null 처리, 이상치 감지
3. **분류 체계 검증** (Stage 3): 병리 분류 정확도
4. **매핑 검증** (Stage 4): 지표-병리 매핑
5. **상관관계 검증** (Stage 5): 시스템 간 일관성
6. **출력 검증** (Stage 6): 최종 결과물 무결성

## 📊 출력물

실행 후 `output/` 폴더에 생성되는 주요 파일들:

### 📈 시각화
- `master_dashboard_YYYYMMDD_HHMMSS.png` - 종합 대시보드
- `realtime_dashboard_YYYYMMDD_HHMMSS.png` - 실시간 진단
- `integrated_analysis_dashboard_YYYYMMDD_HHMMSS.png` - 통합 분석

### 📄 보고서
- `policy_recommendations_YYYYMMDD_HHMMSS.txt` - **정책당국 권고안**
- `investment_strategy_YYYYMMDD_HHMMSS.txt` - **투자전략 보고서**
- `economic_health_report.txt` - 경제건강 진단서
- `comprehensive_analysis_summary_YYYYMMDD_HHMMSS.txt` - 종합분석 요약

### 📊 구조화된 데이터
- `investment_strategy_YYYYMMDD_HHMMSS.json` - 투자전략 JSON
- `data_validation_YYYYMMDD_HHMMSS.json` - 데이터 검증 결과

## 🔧 고급 설정

### 투자 성향 설정
코드에서 투자 성향을 변경할 수 있습니다:
```python
# src/integrated_system.py의 run_comprehensive_analysis 함수
investment_style = InvestmentStyle.CONSERVATIVE  # 보수적
investment_style = InvestmentStyle.MODERATE      # 중도적 (기본값)
investment_style = InvestmentStyle.AGGRESSIVE    # 공격적
```

### API 타임아웃 설정
환경변수로 API 타임아웃을 조정할 수 있습니다:
```bash
export API_TIMEOUT=30  # 30초 (기본값: 10초)
```

## ❗ 문제 해결

### 자주 발생하는 문제들

**1. API 키 오류**
```
[ERROR] GEMINI_API_KEY 환경변수를 설정해주세요.
```
→ 환경변수 설정 확인 (`echo $GEMINI_API_KEY`)

**2. 패키지 오류**
```bash
pip install --upgrade -r requirements.txt
```

**3. 유니코드 오류 (Windows)**
```bash
chcp 65001  # UTF-8 코드페이지 설정
```

**4. 출력 폴더 오류**
→ 프로젝트 루트 디렉토리에서 실행 확인

### 로그 및 디버깅
- 실행 중 오류 메시지를 자세히 확인하세요
- `output/` 폴더의 분석 결과를 통해 시스템 상태 점검
- API 키 유효성 및 인터넷 연결 상태 확인

## 🔒 보안

- ✅ API 키 하드코딩 제거됨 (환경변수 사용)
- ✅ `.gitignore`로 민감한 정보 보호
- ✅ 출력 파일들 Git 추적에서 제외

**중요**: API 키를 절대 코드에 직접 입력하지 마세요!

## 📜 면책사항

본 시스템은 연구 및 교육 목적으로 개발되었습니다. 실제 투자나 정책 결정에는 사용하지 마시고, 모든 결과는 참고용으로만 활용하시기 바랍니다.

## 🤝 기여

이슈 리포트나 개선 제안은 GitHub Issues를 통해 제출해주세요.

## 📞 지원

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **이메일**: 기술 지원 문의

---

**💡 팁**: 처음 사용하시는 분은 `python main.py`를 실행하고 옵션 4번(전체 분석)을 선택하시면 모든 기능을 체험할 수 있습니다!
