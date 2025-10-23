# bio_agent_code/agents/Trend_Discovery_Agent.py
from __future__ import annotations
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

print("[INIT] Loading environment variables...")
print(f"[INIT] Python environment check...")

# --- 외부 의존 ---
print("[INIT] Checking Tavily library...")
try:
    from tavily import TavilyClient
    _HAS_TAVILY = True
    print("[INIT] ✓ Tavily library imported successfully")
except ImportError as e:
    _HAS_TAVILY = False
    print(f"[INIT] ✗ Tavily import failed: {e}")
    print("[INIT] Run: pip install tavily-python")
except Exception as e:
    _HAS_TAVILY = False
    print(f"[INIT] ✗ Tavily unexpected error: {e}")

# --- 내부 의존 ---
print("[INIT] Importing internal modules...")
try:
    from ..state import PipelineState, TrendItem
    print("[INIT] ✓ Internal modules imported")
except Exception as e:
    print(f"[INIT] ✗ Internal import failed: {e}")

# ---------------------------------------------------------
# 1) 대상 매체: Labiotech, Fierce Biotech (최근 3년 고정)
# ---------------------------------------------------------
NEWS_SITES = [
    "labiotech.eu",
    "fiercebiotech.com",
]
DAYS_3Y = 365 * 3

# 기업 사전(Company Mention 계산에 사용)
COMPANIES = [
    # Big Pharma
    "Pfizer","Roche","Novartis","AstraZeneca","Sanofi","Merck","GSK","Johnson & Johnson","J&J","Bayer",
    # AI Bio / Platform
    "Recursion","Insilico Medicine","Exscientia","BenevolentAI","Isomorphic Labs","DeepMind","Schrodinger",
    # Tech infra
    "NVIDIA","Google","Microsoft","AWS","Databricks","Snowflake",
    # BioTech
    "Moderna","BioNTech","Illumina","Tempus","Ginkgo Bioworks","Zymergen","Absci","Generate Biomedicines"
]

# Presence 가중치 (요구된 3요소 기반)
W_ARTICLE  = 0.5
W_COMPANY  = 0.3
W_MOMENTUM = 0.2

# ------------------ 유틸 ------------------
def _norm(x: str) -> str:
    return (x or "").strip()

def _count_company_mentions(text: str, companies: List[str]) -> int:
    t = " " + text.lower() + " "
    cnt = 0
    for c in companies:
        # 단어경계 체크
        pattern = rf"(?<![a-z0-9]){re.escape(c.lower())}(?![a-z0-9])"
        if re.search(pattern, t):
            cnt += 1
    return cnt

def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        return None

def _minmax(arr: List[float]) -> List[float]:
    if not arr:
        return []
    lo, hi = min(arr), max(arr)
    if hi == lo:
        return [0.0 for _ in arr]
    return [(v - lo) / (hi - lo) for v in arr]

def _presence(a_norm: float, c_norm: float, m_norm: float) -> float:
    return round(W_ARTICLE*a_norm + W_COMPANY*c_norm + W_MOMENTUM*m_norm, 4)

# ------------------ Tavily 래퍼 ------------------
def _tavily_search(query: str, days: int = DAYS_3Y, max_results: int = 50) -> List[Dict]:
    """
    Tavily로 검색 (도메인 필터는 include_domains 사용)
    """
    print(f"\n[TAVILY] Starting search for: '{query}'")
    
    if not _HAS_TAVILY:
        print("[TAVILY] ✗ Tavily library not available")
        print("[TAVILY] Install with: pip install tavily-python")
        return []

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[TAVILY] ✗ TAVILY_API_KEY not found in environment")
        print("[TAVILY] Check your .env file")
        return []
    
    print(f"[TAVILY] ✓ API Key found: {api_key[:10]}...")
    print(f"[TAVILY] Search parameters:")
    print(f"  - Query: {query}")
    print(f"  - Domains: {NEWS_SITES}")
    print(f"  - Days: {days}")
    print(f"  - Max results: {max_results}")
    
    try:
        print("[TAVILY] Creating TavilyClient...")
        client = TavilyClient(api_key=api_key)
        print("[TAVILY] ✓ Client created")
        
        print("[TAVILY] Executing search...")
        res = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
            include_domains=NEWS_SITES,
            days=days
        )
        print("[TAVILY] ✓ Search completed")
        
        hits = res.get("results", [])
        print(f"[TAVILY] ✓ Found {len(hits)} results")
        
        if len(hits) > 0:
            print(f"[TAVILY] Sample result URLs:")
            for i, h in enumerate(hits[:3], 1):
                print(f"  {i}. {h.get('url', 'N/A')}")
        
        out = []
        for h in hits:
            out.append({
                "title": _norm(h.get("title")),
                "content": _norm(h.get("content")),
                "url": _norm(h.get("url")),
                "published_date": _norm(h.get("published_date") or h.get("date")),
            })
        
        print(f"[TAVILY] ✓ Processed {len(out)} articles")
        return out
        
    except Exception as e:
        print(f"[TAVILY] ✗ Search failed: {type(e).__name__}")
        print(f"[TAVILY] Error details: {str(e)}")
        import traceback
        print("[TAVILY] Traceback:")
        traceback.print_exc()
        return []

def _gather_all_articles() -> List[Dict]:
    """
    Bio + AI 관련 기사를 수집
    """
    print("\n[GATHER] Starting article collection...")
    
    queries = [
        "artificial intelligence drug discovery",
        "AI biotechnology",
        "machine learning genomics",
        "AI protein design",
        "computational biology",
        "AI biotech startups"
    ]
    
    print(f"[GATHER] Will search {len(queries)} queries")
    
    merged: Dict[str, Dict] = {}
    for idx, query in enumerate(queries, 1):
        print(f"\n[GATHER] Query {idx}/{len(queries)}: {query}")
        results = _tavily_search(query)
        print(f"[GATHER] Got {len(results)} results from this query")
        
        for r in results:
            if r["url"]:  # URL이 있는 경우만
                merged[r["url"]] = r
        
        print(f"[GATHER] Total unique articles so far: {len(merged)}")
    
    final_articles = list(merged.values())
    print(f"\n[GATHER] ✓ Collection complete: {len(final_articles)} unique articles")
    return final_articles

def _extract_keywords(articles: List[Dict], min_mentions: int = 3) -> List[str]:
    """
    기사 제목 + 내용에서 Bio+AI 관련 키워드 추출 (개선 버전)
    """
    print(f"\n[KEYWORDS] Extracting keywords from {len(articles)} articles...")
    print(f"[KEYWORDS] Minimum mentions threshold: {min_mentions}")
    
    # 더 구체적인 Bio+AI 기술 키워드만 추출
    specific_keywords = [
        # AI Drug Discovery 관련
        "AI drug discovery", "AI-driven drug discovery", "AI drug design",
        "generative AI drug", "deep learning drug discovery", "machine learning drug discovery",
        
        # Protein/Biology AI
        "AI protein folding", "protein structure prediction", "AlphaFold",
        "AI protein design", "computational protein design", "protein design AI",
        
        # Genomics AI
        "AI genomics", "machine learning genomics", "AI gene editing",
        "CRISPR AI", "AI genetic analysis", "genomic AI",
        
        # Bio Manufacturing
        "AI biomanufacturing", "synthetic biology AI", "AI bioprocess",
        "synthetic biology", "AI manufacturing",
        
        # Clinical/Biomarker
        "AI biomarker discovery", "AI clinical trial", "AI drug screening",
        "AI target identification", "AI lead optimization", "biomarker AI",
        
        # Antibody/Biologics
        "AI antibody discovery", "AI antibody design", "generative AI antibody",
        "AI biologics", "antibody discovery", "antibody design AI",
        
        # Platform/Technology
        "digital twin biology", "multi-omics AI", "AI cell therapy", 
        "AI vaccine design", "AI immunotherapy", "cell therapy AI",
        
        # General Bio-AI (더 구체적)
        "machine learning biology", "deep learning biology", 
        "neural network drug", "AI biotech platform",
        "computational drug design", "AI molecular design"
    ]
    
    keyword_counter = Counter()
    
    # 각 기사에서 키워드 매칭
    for idx, article in enumerate(articles, 1):
        text = f"{article['title']} {article['content']}".lower()
        
        for keyword in specific_keywords:
            # 대소문자 무시하고 키워드 포함 여부 확인
            if keyword.lower() in text:
                # 정규화된 형태로 저장 (첫 글자 대문자)
                normalized_kw = ' '.join(word.capitalize() for word in keyword.split())
                keyword_counter[normalized_kw] += 1
        
        if idx % 20 == 0:
            print(f"[KEYWORDS] Processed {idx}/{len(articles)} articles...")
    
    print(f"[KEYWORDS] Found {len(keyword_counter)} keyword matches")
    
    # 빈도 기준 필터링
    filtered = [(kw, cnt) for kw, cnt in keyword_counter.items() if cnt >= min_mentions]
    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
    
    print(f"[KEYWORDS] After filtering (>= {min_mentions} mentions): {len(filtered_sorted)} keywords")
    
    if len(filtered_sorted) > 0:
        print(f"[KEYWORDS] Top keywords found:")
        for i, (kw, count) in enumerate(filtered_sorted[:15], 1):
            print(f"  {i}. '{kw}' ({count} mentions)")
    
    # 상위 30개 키워드 반환
    keywords = [kw for kw, _ in filtered_sorted[:30]]
    
    return keywords

# ------------------ 코어 지표 계산 ------------------
def _analyze_keyword(keyword: str, all_articles: List[Dict]) -> Tuple[Dict, List[str]]:
    """
    특정 키워드에 대한 지표 계산
    """
    keyword_lower = keyword.lower()
    matched_articles = []
    
    for article in all_articles:
        text = f"{article['title']} {article['content']}".lower()
        if keyword_lower in text:
            matched_articles.append(article)
    
    total_count = len(matched_articles)
    
    if total_count == 0:
        return (
            dict(article_count=0, company_mentions=0, time_momentum=0.0, recent_count=0, total_count=0),
            []
        )
    
    # Company Mention 계산
    comp = 0
    for article in matched_articles:
        comp += _count_company_mentions(f"{article['title']} {article['content']}", COMPANIES)
    
    # Time Momentum 계산
    now = datetime.utcnow()
    cutoff_12m = now - timedelta(days=365)
    
    dated = 0
    recent = 0
    for article in matched_articles:
        dt = _parse_dt(article.get("published_date"))
        if dt:
            dated += 1
            if dt >= cutoff_12m:
                recent += 1
    
    if dated == 0:
        dated = total_count
        recent = int(round(total_count / 3))
    
    share_12m = recent / max(1, dated)
    time_momentum = min(1.0, max(0.0, share_12m / (1/3)))
    
    metrics = dict(
        article_count=total_count,
        company_mentions=comp,
        time_momentum=time_momentum,
        recent_count=recent,
        total_count=total_count
    )
    sources = [article["url"] for article in matched_articles[:5]]
    
    return metrics, sources

# ------------------ 메인 실행 ------------------
def run(state: PipelineState) -> PipelineState:
    """
    요구사항:
    1) Labiotech/FierceBiotech에서 Bio+AI 관련 기사 수집 (최근 3년)
    2) 기사에서 Bio+AI 트렌드 키워드 자동 추출
    3) Article Count / Company Mention / Time Momentum 기반 Trend Presence 계산
    4) Presence 상위 10개를 state.discovered에 저장하여 다음 agent로 전달
    """
    print("\n" + "="*80)
    print("[RUN] Starting Trend Discovery Agent")
    print("="*80)
    
    state.log("[Discovery] start - collecting articles from Labiotech & FierceBiotech")
    target_n = max(1, state.target_count)
    
    # 1단계: 모든 Bio+AI 기사 수집
    state.log("[Discovery] gathering all bio+AI articles...")
    all_articles = _gather_all_articles()
    state.log(f"[Discovery] collected {len(all_articles)} articles")
    
    if len(all_articles) == 0:
        print("\n[RUN] ✗ No articles collected - stopping")
        state.log("[Discovery] WARNING: No articles found. Check TAVILY_API_KEY or network.")
        state.discovered = []
        return state
    
    # 2단계: 키워드 추출
    state.log("[Discovery] extracting trend keywords...")
    keywords = _extract_keywords(all_articles, min_mentions=3)
    state.log(f"[Discovery] extracted {len(keywords)} potential trend keywords")
    
    if len(keywords) == 0:
        print("\n[RUN] ✗ No keywords extracted - stopping")
        state.log("[Discovery] WARNING: No keywords extracted")
        state.discovered = []
        return state
    
    # 3단계: 각 키워드별 지표 계산
    print("\n[RUN] Analyzing keywords...")
    rows: List[Dict] = []
    for idx, kw in enumerate(keywords, 1):
        print(f"[RUN] Analyzing {idx}/{len(keywords)}: {kw}")
        m, src = _analyze_keyword(kw, all_articles)
        rows.append({"keyword": kw, **m, "sources": src})
        state.log(f"[Discovery] {kw} | A={m['article_count']} C={m['company_mentions']} M={m['time_momentum']:.2f}")
    
    # 4단계: 정규화 및 Presence 계산
    print("\n[RUN] Calculating Presence scores...")
    arts = [r["article_count"] for r in rows]
    comps = [r["company_mentions"] for r in rows]
    moms = [r["time_momentum"] for r in rows]
    
    arts_n = _minmax(arts)
    comps_n = _minmax(comps)
    moms_n = _minmax(moms)
    
    for i, r in enumerate(rows):
        r["article_norm"] = arts_n[i]
        r["company_norm"] = comps_n[i]
        r["momentum_norm"] = moms_n[i]
        r["presence"] = _presence(arts_n[i], comps_n[i], moms_n[i])
    
    # 5단계: Presence 기준 상위 N개 선별
    ranked = sorted(rows, key=lambda x: x["presence"], reverse=True)[:target_n]
    print(f"\n[RUN] ✓ Selected top {len(ranked)} trends by Presence score")
    
    # 6단계: TrendItem으로 변환하여 state.discovered에 저장 (다음 agent로 전달 준비)
    items: List[TrendItem] = []
    for rank, r in enumerate(ranked, 1):
        summary = (
            f"Rank #{rank} | "
            f"Presence={r['presence']:.4f} | "
            f"Articles={r['article_count']} | "
            f"CompanyMentions={r['company_mentions']} | "
            f"Momentum={r['time_momentum']:.2f}"
        )
        items.append(
            TrendItem(
                title=r["keyword"],
                summary=summary,
                sources=r["sources"],
                tags=["discovery","bio-ai","labiotech","fiercebiotech"],
                score=r["presence"]  # Presence 점수를 score로 저장
            )
        )
    
    # state.discovered에 저장 (다음 agent가 이 데이터를 사용)
    state.discovered = items
    state.log(f"[Discovery] ranked_top={len(items)} trends (target={target_n})")
    state.log(f"[Discovery] Top 3 trends: {', '.join([item.title for item in items[:3]])}")
    state.log("[Discovery] Results saved to state.discovered - ready for next agent")
    state.log("[Discovery] done")
    
    print(f"\n[RUN] ✓ Discovery complete - {len(items)} trends ready for next agent")
    print("="*80)
    return state


# ------------------ 테스트 실행 코드 ------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Trend Discovery Agent 실행 중...")
    print("=" * 80)
    
    # Tavily API 키 확인
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("\n[ERROR] TAVILY_API_KEY가 설정되지 않았습니다!")
        print("다음 명령어로 .env 파일에 API 키를 추가하세요:")
        print("echo 'TAVILY_API_KEY=your_api_key_here' >> .env")
        exit(1)
    else:
        print(f"\n✓ TAVILY_API_KEY 확인됨: {api_key[:10]}...")
    
    # Tavily 라이브러리 확인
    if not _HAS_TAVILY:
        print("\n[ERROR] Tavily 라이브러리가 설치되지 않았습니다!")
        print("다음 명령어로 설치하세요:")
        print("pip install tavily-python")
        exit(1)
    
    # State 초기화
    test_state = PipelineState(target_count=10)
    
    # 에이전트 실행
    result_state = run(test_state)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print(f"총 {len(result_state.discovered)}개 트렌드 발견")
    print("=" * 80)
    
    if len(result_state.discovered) > 0:
        print("\n[ 트렌드 랭킹 - Presence 기준 상위 10개 ]")
        print("-" * 80)
        for idx, trend in enumerate(result_state.discovered, 1):
            print(f"\n{idx}. {trend.title}")
            print(f"   Presence 점수: {trend.score:.4f}")
            print(f"   {trend.summary}")
            print(f"   소스 개수: {len(trend.sources)}")
            if trend.sources:
                print(f"   대표 URL: {trend.sources[0]}")
        
        # 다음 agent로 전달 준비 완료 확인
        print("\n" + "=" * 80)
        print("[ State.discovered 내용 확인 ]")
        print("=" * 80)
        print(f"✓ state.discovered에 {len(result_state.discovered)}개 TrendItem 저장 완료")
        print(f"✓ 다음 agent에서 사용 가능:")
        print(f"   - next_state = next_agent.run(result_state)")
        print(f"   - next_state.discovered[0].title = '{result_state.discovered[0].title}'")
        print(f"   - next_state.discovered[0].score = {result_state.discovered[0].score:.4f}")
    else:
        print("\n[경고] 발견된 트렌드가 없습니다.")
    
    print("\n" + "=" * 80)
    print("[ 실행 로그 ]")
    print("=" * 80)
    if hasattr(result_state, '_logs'):
        for log in result_state._logs:
            print(log)
    elif hasattr(result_state, 'logs'):
        for log in result_state.logs:
            print(log)
    
    print("\n" + "=" * 80)
    print("✓ 실행 완료 - 다음 agent로 전달 준비 완료")
    print("=" * 80)