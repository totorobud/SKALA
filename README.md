 ┌───────────────────────────────┐
 │        Trend_Discovery_Agent  │
 │  ─ 주요 트렌드 탐색 (기사, 키워드) ─│
 └───────────────────────────────┘
                │
                ▼
 ┌───────────────────────────────┐
 │       Trend_Profiling_Agent   │
 │  ─ 논문 기반 기술 분석 및 핵심 요약 ─│
 └───────────────────────────────┘
                │
                ▼
 ┌───────────────────────────────┐
 │     Trend_Prediction_Agent    │
 │ ─ 기술 발전 방향 및 시장 예측 ─│
 └───────────────────────────────┘
                │
                ▼
 ┌───────────────────────────────┐
 │    Risk_Opportunity_Agent     │
 │ ─ 리스크·기회 분석 (RAG 기반) ─│
 └───────────────────────────────┘
                │
                ▼
 ┌───────────────────────────────┐
 │   Report_Generation_Agent     │
 │ ─ 통합 보고서 생성 및 시각화 ─│
 └───────────────────────────────┘




## 1️⃣ Trend Discovery Agent  
> **Bio + AI 트렌드 자동 수집 및 정량 평가 에이전트**

### 🧭 개요  
`Trend_Discovery_Agent`는 **Bio + AI 트렌드의 최신 동향을 자동으로 수집하고 정량적으로 평가**하는  
파이프라인의 첫 번째 단계입니다.  
`Labiotech`, `FierceBiotech` 등 주요 미디어에서 최근 3년간의 기사를 Tavily API로 수집해,  
트렌드 키워드를 식별하고 지표 기반으로 랭킹화합니다.

---

### ⚙️ 작동 로직  

#### 1. 기사 수집 (Tavily API)
- Tavily를 사용해 아래 쿼리로 **AI + Biotechnology 관련 기사**를 검색합니다.  
  예시 쿼리:
"artificial intelligence drug discovery"
"machine learning genomics"
"AI protein design"
"computational biology"
"AI biotech startups"


- `Labiotech` 및 `FierceBiotech` 도메인으로 한정하여 최근 3년치 기사만 수집합니다.
- 각 기사에서 **제목, 본문, URL, 게시일**을 추출하고 중복 제거합니다.
- `.env` 파일의 `TAVILY_API_KEY`가 필요합니다.

#### 2. 키워드 추출
- Bio+AI 전문 키워드 세트를 기반으로 각 기사에서 등장 빈도를 집계합니다.  
- `AI drug discovery`, `protein folding`, `synthetic biology`, `AI clinical trial` 등  
총 40여 개의 세부 키워드를 탐색합니다.
- 3회 이상 등장한 키워드만 필터링하고, 상위 30개를 최종 후보로 선정합니다.

#### 3. 핵심 지표 계산
각 키워드별로 다음 3개의 지표를 산출합니다:
- **Article Count**: 관련 기사 수  
- **Company Mentions**: 제약사 및 AI 기업(예: Pfizer, Recursion, NVIDIA 등) 언급 횟수  
- **Time Momentum**: 최근 12개월 기사 비중 (전체 기사 중 최근 기사 비율)

#### 4. Presence 점수 산출
각 지표를 0~1로 정규화한 후 가중 평균으로 통합:
```python
presence = 0.5 * article_norm + 0.3 * company_norm + 0.2 * momentum_norm

#### 4. Presence 점수 산출

Presence 점수를 기준으로 상위 N개 트렌드를 선정하고,
TrendItem 객체 형태로 state.discovered에 저장합니다.

TrendItem(
    title="AI Protein Design",
    summary="Presence=0.82 | Articles=34 | CompanyMentions=12 | Momentum=0.75",
    score=0.82,
    sources=[...],
    tags=["discovery", "bio-ai"]
)