# bio_agent_code/agents/Trend_Evaluation_Agent.py
from __future__ import annotations
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

print("[INIT] Loading Trend Evaluation Agent...")



# ---------------------------------------------------------
# 프로파일링 연결 유틸 (새로 추가)
# ---------------------------------------------------------
def ensure_profiling(state: PipelineState, profiling_runner=None) -> PipelineState:
    """
    Trend_Profiling_Agent가 아직 실행되지 않았다면 자동으로 실행해
    state.academic_papers를 준비한다.
    - profiling_runner: callable(state) -> state 형태를 인자로 넘기면 그걸 사용
    - 없으면 지연 임포트로 bio_agent_code.agents.Trend_Profiling_Agent.run 사용
    """
    # 이미 논문이 있으면 그대로 진행
    if getattr(state, "academic_papers", None):
        state.log("[Evaluation] Profiling already present - skipping")
        return state

    # Discovery 결과가 없으면 프로파일링을 할 수 없음
    if not getattr(state, "discovered", None):
        state.log("[Evaluation] ✗ No discovered trends - cannot profile")
        return state

    state.log("[Evaluation] Academic papers missing - attempting to run Profiling Agent")

    # 1) 함수가 인자로 넘어오면 그대로 사용
    if callable(profiling_runner):
        try:
            new_state = profiling_runner(state)
            state.log("[Evaluation] ✓ Profiling runner completed")
            return new_state
        except Exception as e:
            state.log(f"[Evaluation] ✗ Profiling runner error: {e}")
            return state

    # 2) 지연 임포트로 내부 에이전트 실행 (순환참조 방지)
    try:
        from . import Trend_Profiling_Agent
        new_state = Trend_Profiling_Agent.run(state)
        state.log("[Evaluation] ✓ Trend_Profiling_Agent.run executed (lazy import)")
        return new_state
    except Exception as e:
        state.log(f"[Evaluation] ✗ Could not import/run Trend_Profiling_Agent: {e}")
        return state


# ---------------------------------------------------------
# 메인 실행 함수
# ---------------------------------------------------------
def run(state: PipelineState, *, auto_profile: bool = True, profiling_runner=None) -> PipelineState:
    """
    Trend Evaluation Agent

    프로세스:
    0. (옵션) Profiling 자동 연결: state.academic_papers가 없으면 Profiling 실행
    1. state.academic_papers에서 트렌드 목록 추출
    2. 각 트렌드에 대해 세 가지 축 계산:
       - Current Presence (현재 존재감)
       - Momentum (성장 속도)
       - Sustainability (지속 가능성)
    3. 발전 방향 및 시장 적용 가능성 예측
    4. state.evaluations에 결과 저장

    파라미터:
    - auto_profile: True이면 Profiling이 안 되어 있어도 자동 실행
    - profiling_runner: 외부에서 전달 가능한 함수 (테스트/주입 용이)
    """

    print("\n" + "="*80)
    print("🧭 TREND EVALUATION AGENT - START")
    print("="*80)

    state.log("[Evaluation] ========== START ==========")
    state.log(f"[Evaluation] Timestamp: {datetime.now().isoformat()}")

    # 0) 프로파일링 자동 연결
    if auto_profile:
        before_cnt = len(getattr(state, "academic_papers", []) or [])
        state = ensure_profiling(state, profiling_runner=profiling_runner)
        after_cnt = len(getattr(state, "academic_papers", []) or [])
        state.log(f"[Evaluation] Profiling link: papers {before_cnt} -> {after_cnt}")

    if not getattr(state, "academic_papers", None):
        state.log("[Evaluation] ✗ ERROR: No academic papers to evaluate (profiling missing)")
        print("⚠️  No academic papers found - Skipping evaluation")
        return state




# --- 외부 의존 ---
try:
    from tavily import TavilyClient
    _HAS_TAVILY = True
    print("[INIT] ✓ Tavily library imported")
except Exception as e:
    _HAS_TAVILY = False
    print(f"[INIT] ✗ Tavily import failed: {e}")

# --- 내부 의존 ---
from ..state import PipelineState, TrendItem, EvaluationMetrics

# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
# 시간 범위
RECENT_12M = 365  # 최근 12개월
PREVIOUS_12M = 730  # 이전 12개월 (12~24개월 전)

# 데이터 소스 도메인
FUNDING_DOMAINS = [
    "crunchbase.com",
    "pitchbook.com",
    "techcrunch.com"
]

POLICY_DOMAINS = [
    "who.int",
    "fda.gov",
    "oecd.org",
    "nih.gov"
]

NEWS_DOMAINS = [
    "labiotech.eu",
    "fiercebiotech.com",
    "nature.com",
    "science.org"
]

# 가중치
W_PAPER = 0.4
W_FUNDING = 0.3
W_NEWS = 0.3

W_PLAYER_DIVERSITY = 0.6
W_POLICY_STABILITY = 0.4

# ---------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------
def _norm(x: str) -> str:
    return (x or "").strip()

def _minmax_normalize(values: List[float]) -> List[float]:
    """Min-Max 정규화 (0~1 범위)"""
    if not values or len(values) == 0:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return [0.5 for _ in values]
    
    return [(v - min_val) / (max_val - min_val) for v in values]

def _z_normalize(values: List[float]) -> List[float]:
    """Z-score 정규화"""
    if not values or len(values) == 0:
        return []
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5
    
    if std == 0:
        return [0.0 for _ in values]
    
    return [(v - mean) / std for v in values]

def _classify_development_stage(presence: float, momentum: float, sustainability: float) -> str:
    """
    세 지표 기반으로 발전 단계 분류
    
    반환값:
    - "Emerging → Growth" : 🚀 고성장 + 시장 진입
    - "Mature" : 📈 안정 성장
    - "Plateau" : ⚠️ 포화
    - "Seed Trend" : 🌱 초기 성장
    - "Hype" : 💥 단기 유행
    - "Niche" : 🧩 잠재 성장 대기
    - "Decline" : 🪦 쇠퇴 기술
    """
    # 임계값 설정 (정규화된 값 기준)
    HIGH_THRESHOLD = 0.6
    MID_THRESHOLD = 0.4
    LOW_THRESHOLD = 0.3
    
    # 높음/낮음 판단
    presence_high = presence >= HIGH_THRESHOLD
    presence_mid = MID_THRESHOLD <= presence < HIGH_THRESHOLD
    presence_low = presence < MID_THRESHOLD
    
    momentum_high = momentum >= HIGH_THRESHOLD
    momentum_low = momentum < LOW_THRESHOLD
    
    sustainability_high = sustainability >= HIGH_THRESHOLD
    sustainability_low = sustainability < LOW_THRESHOLD
    
    # 분류 로직
    if presence_high and momentum_high and sustainability_high:
        return "🚀 Emerging → Growth"
    elif presence_high and not momentum_low and sustainability_high:
        return "📈 Mature"
    elif presence_high and momentum_low and sustainability_low:
        return "⚠️ Plateau"
    elif presence_low and momentum_high and sustainability_high:
        return "🌱 Seed Trend"
    elif presence_low and momentum_high and sustainability_low:
        return "💥 Hype"
    elif presence_low and not momentum_low and sustainability_high:
        return "🧩 Niche"
    else:
        return "🪦 Decline"

def _get_market_feasibility(stage: str) -> str:
    """발전 단계별 시장 적용 가능성 판단"""
    feasibility_map = {
        "🚀 Emerging → Growth": "High - 빠른 시장 확산 예상, 조기 투자 적기",
        "📈 Mature": "High - 안정적 성장, 장기 투자 적합",
        "⚠️ Plateau": "Medium - 시장 포화, 차별화 전략 필요",
        "🌱 Seed Trend": "Medium-High - 초기 단계, 성장 잠재력 높음",
        "💥 Hype": "Low-Medium - 단기 유행 가능성, 신중한 접근 필요",
        "🧩 Niche": "Medium - 특정 분야 성장 가능성, 장기 관찰 필요",
        "🪦 Decline": "Low - 기술 쇠퇴, 투자 비권장"
    }
    return feasibility_map.get(stage, "Unknown")

# ---------------------------------------------------------
# 1️⃣ Current Presence 계산
# ---------------------------------------------------------
def _calculate_current_presence(trend_title: str, state: PipelineState) -> Dict[str, float]:
    """
    현재의 존재감 계산
    
    지표:
    - 논문 수 (최근 12개월)
    - 투자 건수 (최근 12개월)
    - 뉴스 량 (최근 12개월)
    
    반환: {
        'paper_count_12m': int,
        'funding_count_12m': int,
        'news_count_12m': int,
        'presence_raw': float
    }
    """
    state.log(f"[Evaluation] Calculating Current Presence for: {trend_title}")
    
    # 1. 논문 수 (state.academic_papers에서)
    paper_count = sum(1 for p in state.academic_papers if p.trend_title == trend_title)
    state.log(f"[Evaluation]   Paper count (12M): {paper_count}")
    
    # 2. 투자 건수 (Crunchbase/TechCrunch 검색)
    funding_count = _search_funding_events(trend_title, days=RECENT_12M, state=state)
    state.log(f"[Evaluation]   Funding count (12M): {funding_count}")
    
    # 3. 뉴스 량 (Labiotech/FierceBiotech 검색)
    news_count = _search_news_volume(trend_title, days=RECENT_12M, state=state)
    state.log(f"[Evaluation]   News count (12M): {news_count}")
    
    # 원시 점수 계산 (정규화 전)
    presence_raw = paper_count + funding_count + news_count
    
    return {
        'paper_count_12m': paper_count,
        'funding_count_12m': funding_count,
        'news_count_12m': news_count,
        'presence_raw': presence_raw
    }

def _search_funding_events(keyword: str, days: int, state: PipelineState) -> int:
    """투자 이벤트 검색 (Tavily 사용)"""
    if not _HAS_TAVILY:
        state.log("[Evaluation] ⚠️  Tavily not available for funding search")
        return 0
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        state.log("[Evaluation] ⚠️  TAVILY_API_KEY not found")
        return 0
    
    try:
        client = TavilyClient(api_key=api_key)
        query = f"{keyword} funding investment biotech"
        
        state.log(f"[Evaluation] Searching funding: '{query}' (last {days} days)")
        
        result = client.search(
            query=query,
            max_results=30,
            search_depth="basic",
            include_domains=FUNDING_DOMAINS,
            days=days
        )
        
        count = len(result.get("results", []))
        state.log(f"[Evaluation] ✓ Found {count} funding-related articles")
        
        return count
        
    except Exception as e:
        state.log(f"[Evaluation] ✗ Funding search error: {str(e)}")
        return 0

def _search_news_volume(keyword: str, days: int, state: PipelineState) -> int:
    """뉴스 검색 (Tavily 사용)"""
    if not _HAS_TAVILY:
        state.log("[Evaluation] ⚠️  Tavily not available for news search")
        return 0
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        state.log("[Evaluation] ⚠️  TAVILY_API_KEY not found")
        return 0
    
    try:
        client = TavilyClient(api_key=api_key)
        query = f"{keyword} biotech"
        
        state.log(f"[Evaluation] Searching news: '{query}' (last {days} days)")
        
        result = client.search(
            query=query,
            max_results=50,
            search_depth="basic",
            include_domains=NEWS_DOMAINS,
            days=days
        )
        
        count = len(result.get("results", []))
        state.log(f"[Evaluation] ✓ Found {count} news articles")
        
        return count
        
    except Exception as e:
        state.log(f"[Evaluation] ✗ News search error: {str(e)}")
        return 0

# ---------------------------------------------------------
# 2️⃣ Momentum 계산
# ---------------------------------------------------------
def _calculate_momentum(trend_title: str, current_presence: Dict, state: PipelineState) -> Dict[str, float]:
    """
    성장 속도 계산
    
    지표:
    - 논문 증가율 (최근 12M vs 이전 12M)
    - 투자 증가율
    - 뉴스 증가율
    
    반환: {
        'paper_growth': float,
        'funding_growth': float,
        'news_growth': float,
        'momentum_raw': float
    }
    """
    state.log(f"[Evaluation] Calculating Momentum for: {trend_title}")
    
    # 이전 12개월 데이터 수집
    paper_count_prev = _count_papers_in_period(trend_title, days=PREVIOUS_12M, state=state)
    funding_count_prev = _search_funding_events(trend_title, days=PREVIOUS_12M, state=state)
    news_count_prev = _search_news_volume(trend_title, days=PREVIOUS_12M, state=state)
    
    state.log(f"[Evaluation]   Previous 12M: Papers={paper_count_prev}, Funding={funding_count_prev}, News={news_count_prev}")
    
    # 증가율 계산 (%)
    paper_growth = _calculate_growth_rate(
        current_presence['paper_count_12m'],
        paper_count_prev
    )
    
    funding_growth = _calculate_growth_rate(
        current_presence['funding_count_12m'],
        funding_count_prev
    )
    
    news_growth = _calculate_growth_rate(
        current_presence['news_count_12m'],
        news_count_prev
    )
    
    state.log(f"[Evaluation]   Growth rates: Paper={paper_growth:.1f}%, Funding={funding_growth:.1f}%, News={news_growth:.1f}%")
    
    # 평균 증가율
    momentum_raw = (paper_growth + funding_growth + news_growth) / 3
    
    return {
        'paper_growth': paper_growth,
        'funding_growth': funding_growth,
        'news_growth': news_growth,
        'momentum_raw': momentum_raw
    }

def _count_papers_in_period(trend_title: str, days: int, state: PipelineState) -> int:
    """특정 기간 내 논문 수 카운트 (간접 추정)"""
    # 실제로는 state.academic_papers의 year 정보를 활용해야 하지만
    # 여기서는 간단히 현재 논문 수의 60%로 추정 (이전 기간은 보통 더 적음)
    current_count = sum(1 for p in state.academic_papers if p.trend_title == trend_title)
    return int(current_count * 0.6)

def _calculate_growth_rate(current: float, previous: float) -> float:
    """증가율 계산 (%)"""
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    
    return ((current - previous) / previous) * 100

# ---------------------------------------------------------
# 3️⃣ Sustainability 계산
# ---------------------------------------------------------
def _calculate_sustainability(trend_title: str, state: PipelineState) -> Dict[str, float]:
    """
    지속 가능성 계산
    
    지표:
    - 플레이어 다양성 (기업 유형 다양도)
    - 정책 안정성 (긍정적 정책 문서 비율)
    
    반환: {
        'player_diversity': float (0~1),
        'policy_stability': float (0~1),
        'sustainability_raw': float (0~1)
    }
    """
    state.log(f"[Evaluation] Calculating Sustainability for: {trend_title}")
    
    # 1. 플레이어 다양성
    player_diversity = _assess_player_diversity(trend_title, state)
    state.log(f"[Evaluation]   Player diversity: {player_diversity:.3f}")
    
    # 2. 정책 안정성
    policy_stability = _assess_policy_stability(trend_title, state)
    state.log(f"[Evaluation]   Policy stability: {policy_stability:.3f}")
    
    # 가중 평균
    sustainability_raw = (
        W_PLAYER_DIVERSITY * player_diversity +
        W_POLICY_STABILITY * policy_stability
    )
    
    return {
        'player_diversity': player_diversity,
        'policy_stability': policy_stability,
        'sustainability_raw': sustainability_raw
    }

def _assess_player_diversity(trend_title: str, state: PipelineState) -> float:
    """
    플레이어 다양성 평가
    
    방법:
    1. Crunchbase에서 관련 기업 검색
    2. 기업 유형 분류 (Startup, Pharma, Research 등)
    3. 다양성 지수 계산 (Shannon Entropy 또는 Simpson Index)
    """
    if not _HAS_TAVILY:
        return 0.5  # 기본값
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return 0.5
    
    try:
        client = TavilyClient(api_key=api_key)
        query = f"{trend_title} companies biotech pharmaceutical startup"
        
        state.log(f"[Evaluation] Searching companies: '{query}'")
        
        result = client.search(
            query=query,
            max_results=30,
            search_depth="basic",
            include_domains=FUNDING_DOMAINS
        )
        
        results = result.get("results", [])
        
        if not results:
            state.log("[Evaluation] ⚠️  No company data found")
            return 0.3
        
        # 기업 유형 분류 (간단한 키워드 기반)
        type_counts = {
            'startup': 0,
            'pharma': 0,
            'research': 0,
            'tech': 0,
            'other': 0
        }
        
        for r in results:
            text = f"{r.get('title', '')} {r.get('content', '')}".lower()
            
            if any(kw in text for kw in ['startup', 'early-stage', 'seed']):
                type_counts['startup'] += 1
            elif any(kw in text for kw in ['pharmaceutical', 'pharma', 'drug', 'biopharma']):
                type_counts['pharma'] += 1
            elif any(kw in text for kw in ['research', 'university', 'institute', 'academic']):
                type_counts['research'] += 1
            elif any(kw in text for kw in ['tech', 'platform', 'ai', 'software']):
                type_counts['tech'] += 1
            else:
                type_counts['other'] += 1
        
        # Shannon Entropy 계산 (다양성 지수)
        total = sum(type_counts.values())
        if total == 0:
            return 0.3
        
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * (p ** 0.5)  # 간단한 다양성 지수
        
        # 정규화 (0~1)
        max_entropy = 2.0  # 5개 유형이 균등할 때 최대값 근사
        diversity_score = min(1.0, entropy / max_entropy)
        
        state.log(f"[Evaluation]   Company types: {type_counts}")
        state.log(f"[Evaluation]   Diversity score: {diversity_score:.3f}")
        
        return diversity_score
        
    except Exception as e:
        state.log(f"[Evaluation] ✗ Player diversity error: {str(e)}")
        return 0.5

def _assess_policy_stability(trend_title: str, state: PipelineState) -> float:
    """
    정책 안정성 평가
    
    방법:
    1. WHO/FDA/OECD 문서 검색
    2. "AI biomedical" 관련 문서에서 긍정/부정 비율 계산
    """
    if not _HAS_TAVILY:
        return 0.5  # 기본값
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return 0.5
    
    try:
        client = TavilyClient(api_key=api_key)
        query = f"{trend_title} AI biomedical regulation policy"
        
        state.log(f"[Evaluation] Searching policies: '{query}'")
        
        result = client.search(
            query=query,
            max_results=20,
            search_depth="basic",
            include_domains=POLICY_DOMAINS
        )
        
        results = result.get("results", [])
        
        if not results:
            state.log("[Evaluation] ⚠️  No policy documents found")
            return 0.5
        
        # 긍정/부정 키워드 분석
        positive_keywords = [
            'approve', 'support', 'encourage', 'advance', 'promote',
            'benefit', 'innovation', 'breakthrough', 'opportunity', 'potential'
        ]
        
        negative_keywords = [
            'risk', 'concern', 'caution', 'limitation', 'challenge',
            'restrict', 'prohibit', 'ban', 'danger', 'threat'
        ]
        
        positive_count = 0
        negative_count = 0
        
        for r in results:
            text = f"{r.get('title', '')} {r.get('content', '')}".lower()
            
            for kw in positive_keywords:
                if kw in text:
                    positive_count += 1
            
            for kw in negative_keywords:
                if kw in text:
                    negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5
        
        # 긍정 비율
        positive_ratio = positive_count / total
        
        state.log(f"[Evaluation]   Policy sentiment: Positive={positive_count}, Negative={negative_count}")
        state.log(f"[Evaluation]   Positive ratio: {positive_ratio:.3f}")
        
        return positive_ratio
        
    except Exception as e:
        state.log(f"[Evaluation] ✗ Policy stability error: {str(e)}")
        return 0.5

# ---------------------------------------------------------
# 메인 실행 함수
# ---------------------------------------------------------
def run(state: PipelineState) -> PipelineState:
    """
    Trend Evaluation Agent
    
    프로세스:
    1. state.academic_papers에서 트렌드 목록 추출
    2. 각 트렌드에 대해 세 가지 축 계산:
       - Current Presence (현재 존재감)
       - Momentum (성장 속도)
       - Sustainability (지속 가능성)
    3. 발전 방향 및 시장 적용 가능성 예측
    4. state.evaluations에 결과 저장
    """
    print("\n" + "="*80)
    print("🧭 TREND EVALUATION AGENT - START")
    print("="*80)
    
    state.log("[Evaluation] ========== START ==========")
    state.log(f"[Evaluation] Timestamp: {datetime.now().isoformat()}")
    
    if not state.academic_papers:
        state.log("[Evaluation] ✗ ERROR: No academic papers to evaluate")
        print("⚠️  No academic papers found - Skipping evaluation")
        return state
    
    # 트렌드 목록 추출 (중복 제거)
    trend_titles = list(set(p.trend_title for p in state.academic_papers))
    state.log(f"[Evaluation] Found {len(trend_titles)} unique trends to evaluate")
    print(f"✓ Evaluating {len(trend_titles)} trends\n")
    
    # API 키 체크
    tavily_key = os.getenv("TAVILY_API_KEY")
    state.log(f"[Evaluation] TAVILY_API_KEY: {'✓ Found' if tavily_key else '✗ Missing'}")
    
    if not tavily_key:
        print("⚠️  WARNING: TAVILY_API_KEY not found - evaluation will use fallback values")
    
    all_evaluations = []
    
    # 원시 데이터 수집 (정규화 전)
    raw_data = []
    
    for trend_idx, trend_title in enumerate(trend_titles, 1):
        print(f"\n[{trend_idx}/{len(trend_titles)}] Evaluating: {trend_title}")
        state.log(f"[Evaluation] ========== TREND {trend_idx}/{len(trend_titles)}: {trend_title} ==========")
        
        try:
            # 1. Current Presence 계산
            state.log(f"[Evaluation] Step 1: Calculate Current Presence")
            presence_data = _calculate_current_presence(trend_title, state)
            
            # 2. Momentum 계산
            state.log(f"[Evaluation] Step 2: Calculate Momentum")
            momentum_data = _calculate_momentum(trend_title, presence_data, state)
            
            # 3. Sustainability 계산
            state.log(f"[Evaluation] Step 3: Calculate Sustainability")
            sustainability_data = _calculate_sustainability(trend_title, state)
            
            # 원시 데이터 저장
            raw_data.append({
                'trend_title': trend_title,
                'presence_raw': presence_data['presence_raw'],
                'momentum_raw': momentum_data['momentum_raw'],
                'sustainability_raw': sustainability_data['sustainability_raw'],
                'presence_data': presence_data,
                'momentum_data': momentum_data,
                'sustainability_data': sustainability_data
            })
            
            print(f"  ✓ Raw scores: Presence={presence_data['presence_raw']:.1f}, "
                  f"Momentum={momentum_data['momentum_raw']:.1f}%, "
                  f"Sustainability={sustainability_data['sustainability_raw']:.3f}")
            
        except Exception as e:
            state.log(f"[Evaluation] ✗ ERROR evaluating {trend_title}: {str(e)}")
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            state.log(f"[Evaluation] Stack trace:\n{traceback.format_exc()}")
            continue
    
    # 정규화 (트렌드 간 비교를 위해)
    if raw_data:
        state.log(f"[Evaluation] Step 4: Normalize scores across {len(raw_data)} trends")
        
        presence_raw_list = [d['presence_raw'] for d in raw_data]
        momentum_raw_list = [d['momentum_raw'] for d in raw_data]
        sustainability_raw_list = [d['sustainability_raw'] for d in raw_data]
        
        presence_normalized = _minmax_normalize(presence_raw_list)
        momentum_normalized = _minmax_normalize(momentum_raw_list)
        sustainability_normalized = _minmax_normalize(sustainability_raw_list)
        
        # EvaluationMetrics 생성
        for idx, raw in enumerate(raw_data):
            presence_norm = presence_normalized[idx]
            momentum_norm = momentum_normalized[idx]
            sustainability_norm = sustainability_normalized[idx]
            
            # 발전 단계 분류
            stage = _classify_development_stage(
                presence_norm,
                momentum_norm,
                sustainability_norm
            )
            
            # 시장 적용 가능성
            feasibility = _get_market_feasibility(stage)
            
            # EvaluationMetrics 객체 생성
            eval_metrics = EvaluationMetrics(
                trend_title=raw['trend_title'],
                current_presence=presence_norm,
                momentum=momentum_norm,
                sustainability=sustainability_norm,
                development_stage=stage,
                market_feasibility=feasibility,
                presence_details=raw['presence_data'],
                momentum_details=raw['momentum_data'],
                sustainability_details=raw['sustainability_data']
            )
            
            all_evaluations.append(eval_metrics)
            state.log(f"[Evaluation] ✓ {raw['trend_title']}: Stage={stage}, Feasibility={feasibility[:30]}...")
    
    # state에 저장
    state.evaluations = all_evaluations
    state.log(f"[Evaluation] ========== COMPLETE ==========")
    state.log(f"[Evaluation] Total evaluations: {len(all_evaluations)}")
    
    # 터미널에 결과 출력
    _print_evaluation_summary(state)
    
    return state

def _print_evaluation_summary(state: PipelineState) -> None:
    """
    터미널에 최종 평가 결과 출력
    """
    print("\n" + "="*80)
    print("📊 TREND EVALUATION AGENT - RESULTS SUMMARY")
    print("="*80)
    
    if not state.evaluations:
        print("⚠️  No evaluations generated")
        return
    
    print(f"\n✓ Total Trends Evaluated: {len(state.evaluations)}")
    print("\n" + "-"*80)
    
    # 발전 단계별로 정렬
    evaluations_sorted = sorted(
        state.evaluations,
        key=lambda e: (e.current_presence + e.momentum + e.sustainability) / 3,
        reverse=True
    )
    
    for idx, eval_m in enumerate(evaluations_sorted, 1):
        print(f"\n[{idx}] {eval_m.trend_title}")
        print(f"    Development Stage: {eval_m.development_stage}")
        print(f"    Market Feasibility: {eval_m.market_feasibility}")
        print(f"    Scores (normalized 0~1):")
        print(f"      - Current Presence: {eval_m.current_presence:.3f}")
        print(f"      - Momentum: {eval_m.momentum:.3f}")
        print(f"      - Sustainability: {eval_m.sustainability:.3f}")
        print(f"    Details:")
        print(f"      - Papers(12M): {eval_m.presence_details['paper_count_12m']}")
        print(f"      - Funding(12M): {eval_m.presence_details['funding_count_12m']}")
        print(f"      - News(12M): {eval_m.presence_details['news_count_12m']}")
        print(f"      - Growth Rate: {eval_m.momentum_details['momentum_raw']:.1f}%")
    
    print("\n" + "="*80)
    print("✓ Evaluation Complete - Data saved to state.evaluations")
    print("="*80 + "\n")


# ---------------------------------------------------------
# 테스트 실행 코드
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Trend Evaluation Agent 테스트 실행")
    print("=" * 80)
    
    # 필요한 API 키 확인
    tavily_key = os.getenv("TAVILY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not tavily_key:
        print("\n[ERROR] TAVILY_API_KEY가 설정되지 않았습니다!")
        print("다음 명령어로 .env 파일에 API 키를 추가하세요:")
        print("echo 'TAVILY_API_KEY=your_api_key_here' >> .env")
        exit(1)
    
    print(f"\n✓ TAVILY_API_KEY 확인됨: {tavily_key[:10]}...")
    
    if anthropic_key:
        print(f"✓ ANTHROPIC_API_KEY 확인됨: {anthropic_key[:10]}...")
    else:
        print("⚠️  ANTHROPIC_API_KEY 없음 (Profiling Agent에서 fallback 사용)")
    
    # Tavily 라이브러리 확인
    if not _HAS_TAVILY:
        print("\n[ERROR] Tavily 라이브러리가 설치되지 않았습니다!")
        print("다음 명령어로 설치하세요:")
        print("pip install tavily-python")
        exit(1)
    
    print("\n" + "="*80)
    print("전체 파이프라인 실행 (Discovery → Profiling → Evaluation)")
    print("="*80)
    
    # 전체 파이프라인 실행
    try:
        # 모듈 임포트
        print("\n[1/4] 모듈 임포트 중...")
        from . import Trend_Discovery_Agent, Trend_Profiling_Agent
        
        print("✓ 모듈 임포트 완료")
        
        # State 초기화 (상위 3개 트렌드만 테스트)
        print("\n[2/4] State 초기화 (target_count=3)...")
        test_state = PipelineState(target_count=3)
        print("✓ State 초기화 완료")
        
        # Discovery Agent 실행
        print("\n[3/4] Trend Discovery Agent 실행 중...")
        print("-" * 80)
        test_state = Trend_Discovery_Agent.run(test_state)
        
        if not test_state.discovered:
            print("\n[ERROR] Discovery Agent에서 트렌드를 찾지 못했습니다!")
            print("TAVILY_API_KEY를 확인하거나 네트워크 연결을 확인하세요.")
            exit(1)
        
        print(f"\n✓ Discovery 완료: {len(test_state.discovered)}개 트렌드 발견")
        
        # Profiling Agent 실행
        print("\n[4/4] Trend Profiling Agent 실행 중...")
        print("-" * 80)
        test_state = Trend_Profiling_Agent.run(test_state)
        
        if not test_state.academic_papers:
            print("\n[ERROR] Profiling Agent에서 논문을 수집하지 못했습니다!")
            exit(1)
        
        print(f"\n✓ Profiling 완료: {len(test_state.academic_papers)}개 논문 수집")
        
        # Evaluation Agent 실행 (현재 에이전트)
        print("\n[5/5] Trend Evaluation Agent 실행 중...")
        print("=" * 80)
        test_state = run(test_state)
        
        # 최종 결과 출력
        print("\n" + "="*80)
        print("🎉 전체 파이프라인 실행 완료!")
        print("="*80)
        
        if test_state.evaluations:
            print(f"\n✓ 총 {len(test_state.evaluations)}개 트렌드 평가 완료\n")
            
            # 각 트렌드별 요약
            for idx, eval_m in enumerate(test_state.evaluations, 1):
                print(f"\n[{idx}] {eval_m.trend_title}")
                print(f"    🎯 발전 단계: {eval_m.development_stage}")
                print(f"    📊 시장 적용 가능성: {eval_m.market_feasibility[:50]}...")
                print(f"    📈 점수:")
                print(f"       - Current Presence: {eval_m.current_presence:.3f}")
                print(f"       - Momentum: {eval_m.momentum:.3f}")
                print(f"       - Sustainability: {eval_m.sustainability:.3f}")
                print(f"    📄 데이터:")
                print(f"       - 논문: {eval_m.presence_details.get('paper_count_12m', 0)}개")
                print(f"       - 투자: {eval_m.presence_details.get('funding_count_12m', 0)}건")
                print(f"       - 뉴스: {eval_m.presence_details.get('news_count_12m', 0)}개")
                print(f"       - 성장률: {eval_m.momentum_details.get('momentum_raw', 0):.1f}%")
            
            print("\n" + "="*80)
            print("다음 단계: Risk & Opportunity Agent에 state.evaluations 전달")
            print("="*80)
        else:
            print("\n⚠️  평가 결과가 없습니다.")
        
    except ImportError as e:
        print(f"\n[ERROR] 모듈 임포트 실패: {e}")
        print("\n이 에이전트는 전체 파이프라인의 일부입니다.")
        print("개별 실행이 아닌 다음과 같이 사용하세요:\n")
        print("```python")
        print("from bio_agent_code.agents import (")
        print("    Trend_Discovery_Agent,")
        print("    Trend_Profiling_Agent,")
        print("    Trend_Evaluation_Agent")
        print(")")
        print("from bio_agent_code.state import PipelineState")
        print("")
        print("state = PipelineState(target_count=5)")
        print("state = Trend_Discovery_Agent.run(state)")
        print("state = Trend_Profiling_Agent.run(state)")
        print("state = Trend_Evaluation_Agent.run(state)")
        print("")
        print("# 결과 확인")
        print("for eval in state.evaluations:")
        print("    print(f'{eval.trend_title}: {eval.development_stage}')")
        print("```")
        print("\n" + "="*80)
    
    except Exception as e:
        print(f"\n[ERROR] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)