from __future__ import annotations
import os
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# --- 외부 의존 ---
try:
    from tavily import TavilyClient
    _HAS_TAVILY = True
except Exception as e:
    _HAS_TAVILY = False
    print(f"[WARNING] Tavily import failed: {e}")

try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except Exception as e:
    _HAS_ANTHROPIC = False
    print(f"[WARNING] Anthropic import failed: {e}")

# --- 내부 의존 ---
from ..state import PipelineState, TrendItem, PaperItem

# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
PAPERS_PER_TREND = 10  # 각 트렌드당 최근 논문 10개
SCHOLAR_SITE = "site:scholar.google.com"
RECENT_YEARS = 3  # 최근 3년 논문 우선

# ---------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------
def _norm(x: str) -> str:
    return (x or "").strip()

def _extract_year(text: str) -> Optional[int]:
    """텍스트에서 연도 추출 (2020-2025 범위)"""
    matches = re.findall(r'\b(202[0-5])\b', text)
    if matches:
        return int(matches[0])
    return None

def _parse_scholar_results(results: List[Dict], state: PipelineState) -> List[Dict]:
    """
    웹 검색 결과에서 논문 정보 파싱
    
    반환 형식:
    {
        'title': str,
        'authors': str,
        'year': int,
        'abstract': str,
        'citation': str,
        'url': str
    }
    """
    state.log(f"[Profiling] Parsing {len(results)} search results...")
    
    papers = []
    current_year = datetime.now().year
    cutoff_year = current_year - RECENT_YEARS
    
    for idx, r in enumerate(results, 1):
        try:
            title = _norm(r.get("title", ""))
            content = _norm(r.get("content", ""))
            url = _norm(r.get("url", ""))
            
            if not title or not url:
                state.log(f"[Profiling] Skipping result {idx}: missing title or URL")
                continue
            
            # 연도 추출 (제목 또는 내용에서)
            year = _extract_year(f"{title} {content}")
            
            # 최근 논문만 필터링
            if year and year < cutoff_year:
                state.log(f"[Profiling] Skipping result {idx}: year {year} < cutoff {cutoff_year}")
                continue
            
            # 저자 정보 추출 (간단한 휴리스틱)
            authors = ""
            if " - " in content:
                parts = content.split(" - ")
                if len(parts) > 1:
                    authors = parts[0][:200]  # 저자 부분 제한
            
            # Abstract는 content의 일부 (처음 500자)
            abstract = content[:500] if content else ""
            
            # 인용 구문 생성 (APA 스타일 근사)
            citation = _generate_citation(title, authors, year)
            
            papers.append({
                'title': title,
                'authors': authors,
                'year': year or current_year,
                'abstract': abstract,
                'citation': citation,
                'url': url
            })
            
            state.log(f"[Profiling] ✓ Parsed paper {idx}: {title[:50]}... ({year or current_year})")
            
        except Exception as e:
            state.log(f"[Profiling] ✗ Error parsing result {idx}: {str(e)}")
            continue
    
    # 연도 기준 내림차순 정렬 (최신순)
    papers.sort(key=lambda x: x['year'], reverse=True)
    
    state.log(f"[Profiling] Successfully parsed {len(papers)} papers from {len(results)} results")
    
    return papers[:PAPERS_PER_TREND]

def _generate_citation(title: str, authors: str, year: Optional[int]) -> str:
    """간단한 인용 구문 생성"""
    year_str = str(year) if year else "n.d."
    authors_short = authors[:100] + "..." if len(authors) > 100 else authors
    
    if authors_short:
        return f"{authors_short} ({year_str}). {title}"
    else:
        return f"({year_str}). {title}"

def _web_search_scholar(query: str, state: PipelineState, max_results: int = 20) -> List[Dict]:
    """
    Google Scholar에서 논문 검색 (Tavily API 사용)
    """
    state.log(f"[Profiling] Starting web search for: '{query}'")
    
    if not _HAS_TAVILY:
        state.log("[Profiling] ✗ ERROR: Tavily library not available")
        return []
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        state.log("[Profiling] ✗ ERROR: TAVILY_API_KEY not found in environment")
        return []
    
    try:
        client = TavilyClient(api_key=api_key)
        
        # Google Scholar 사이트 제한 검색
        search_query = f"{query} {SCHOLAR_SITE}"
        state.log(f"[Profiling] Search query: '{search_query}'")
        
        result = client.search(
            query=search_query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False
        )
        
        results = result.get("results", [])
        state.log(f"[Profiling] ✓ Tavily returned {len(results)} results")
        
        formatted = []
        for r in results:
            formatted.append({
                "title": _norm(r.get("title", "")),
                "content": _norm(r.get("content", "")),
                "url": _norm(r.get("url", "")),
            })
        
        return formatted
        
    except Exception as e:
        state.log(f"[Profiling] ✗ ERROR during web search: {str(e)}")
        state.log(f"[Profiling] Stack trace: {traceback.format_exc()}")
        return []

def _summarize_technology(trend_title: str, papers: List[Dict], state: PipelineState) -> str:
    """
    Claude API를 사용하여 논문들을 기반으로 기술 요약 생성
    """
    state.log(f"[Profiling] Generating technology summary for: {trend_title}")
    
    if not _HAS_ANTHROPIC:
        state.log("[Profiling] WARNING: Anthropic library not available, using fallback")
        return _fallback_summary(trend_title, papers, state)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        state.log("[Profiling] WARNING: ANTHROPIC_API_KEY not found, using fallback")
        return _fallback_summary(trend_title, papers, state)
    
    try:
        client = Anthropic(api_key=api_key)
        
        # 논문 정보를 프롬프트에 포함
        papers_context = "\n\n".join([
            f"논문 {i+1}:\n제목: {p['title']}\n저자: {p['authors']}\n연도: {p['year']}\n초록: {p['abstract']}"
            for i, p in enumerate(papers[:5])  # 상위 5개만 사용
        ])
        
        state.log(f"[Profiling] Using top {min(5, len(papers))} papers for summary")
        
        prompt = f"""다음은 "{trend_title}" 트렌드와 관련된 최근 학술 논문들입니다.

{papers_context}

위 논문들을 바탕으로 "{trend_title}" 기술에 대해 다음 형식으로 요약해주세요:

1. 기술 개요 (2-3문장)
2. 주요 특징 및 장점 (3-4개 포인트)
3. 현재 연구 동향 (2-3문장)
4. 향후 전망 (1-2문장)

전문적이면서도 이해하기 쉽게 작성해주세요."""

        state.log("[Profiling] Calling Claude API...")
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = message.content[0].text
        state.log(f"[Profiling] ✓ Claude API returned summary ({len(summary)} chars)")
        
        return summary
        
    except Exception as e:
        state.log(f"[Profiling] ✗ ERROR calling Claude API: {str(e)}")
        state.log(f"[Profiling] Stack trace: {traceback.format_exc()}")
        return _fallback_summary(trend_title, papers, state)

def _fallback_summary(trend_title: str, papers: List[Dict], state: PipelineState) -> str:
    """API 실패 시 기본 요약 생성"""
    state.log("[Profiling] Using fallback summary generation")
    
    paper_count = len(papers)
    recent_years = sorted(list(set([p['year'] for p in papers])), reverse=True)[:3]
    
    summary = f"""
{trend_title} 기술 요약

최근 {RECENT_YEARS}년간 {paper_count}개의 주요 논문이 발표되었습니다.
연구가 활발한 연도: {', '.join(map(str, recent_years))}

주요 연구 분야:
"""
    
    # 상위 3개 논문 제목 나열
    for i, p in enumerate(papers[:3], 1):
        summary += f"\n{i}. {p['title']} ({p['year']})"
    
    summary += f"\n\n이 기술은 생명과학 분야에서 AI를 활용하여 혁신을 이루고 있으며, 지속적인 연구 개발이 진행 중입니다."
    
    return summary

def _print_results_summary(state: PipelineState) -> None:
    """
    터미널에 최종 결과 출력
    """
    print("\n" + "="*80)
    print("📊 TREND PROFILING AGENT - RESULTS SUMMARY")
    print("="*80)
    
    if not state.academic_papers:
        print("⚠️  No academic papers collected")
        return
    
    # 트렌드별로 그룹화
    trends_data = {}
    for paper in state.academic_papers:
        if paper.trend_title not in trends_data:
            trends_data[paper.trend_title] = []
        trends_data[paper.trend_title].append(paper)
    
    print(f"\n✓ Total Trends Analyzed: {len(trends_data)}")
    print(f"✓ Total Papers Collected: {len(state.academic_papers)}")
    print("\n" + "-"*80)
    
    # 각 트렌드별 상세 정보
    for idx, (trend_title, papers) in enumerate(trends_data.items(), 1):
        print(f"\n[{idx}] TREND: {trend_title}")
        print(f"    Papers Found: {len(papers)}")
        
        # 연도 분포
        years = [p.year for p in papers]
        year_counts = {}
        for y in years:
            year_counts[y] = year_counts.get(y, 0) + 1
        print(f"    Year Distribution: {dict(sorted(year_counts.items(), reverse=True))}")
        
        # 기술 요약 미리보기
        if papers and papers[0].tech_summary:
            summary_preview = papers[0].tech_summary[:150].replace('\n', ' ')
            print(f"    Tech Summary: {summary_preview}...")
        
        # 상위 3개 논문
        print(f"    Top Papers:")
        for i, paper in enumerate(papers[:3], 1):
            print(f"      {i}. {paper.paper_title[:60]}... ({paper.year})")
            print(f"         Authors: {paper.authors[:50] if paper.authors else 'N/A'}...")
            print(f"         URL: {paper.url}")
    
    print("\n" + "="*80)
    print("✓ Profiling Complete - Data saved to state.academic_papers")
    print("="*80 + "\n")

# ---------------------------------------------------------
# 메인 실행 함수
# ---------------------------------------------------------
def run(state: PipelineState) -> PipelineState:
    """
    Trend Profiling Agent
    
    프로세스:
    1. state.discovered에서 트렌드 목록 가져오기
    2. 각 트렌드에 대해 Google Scholar 검색
    3. 최근 논문 10개 추출 (제목, abstract, 인용구문)
    4. 논문 기반 기술 요약 생성
    5. state.academic_papers에 결과 저장
    6. 터미널에 결과 출력
    """
    print("\n" + "="*80)
    print("🔬 TREND PROFILING AGENT - START")
    print("="*80)
    
    state.log("[Profiling] ========== START ==========")
    state.log(f"[Profiling] Timestamp: {datetime.now().isoformat()}")
    
    if not state.discovered:
        state.log("[Profiling] ✗ ERROR: No trends to analyze (state.discovered is empty)")
        print("⚠️  No trends found in state.discovered - Skipping profiling")
        return state
    
    state.log(f"[Profiling] Found {len(state.discovered)} trends to analyze")
    print(f"✓ Processing {len(state.discovered)} trends from Discovery Agent\n")
    
    # API 키 체크
    tavily_key = os.getenv("TAVILY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    state.log(f"[Profiling] TAVILY_API_KEY: {'✓ Found' if tavily_key else '✗ Missing'}")
    state.log(f"[Profiling] ANTHROPIC_API_KEY: {'✓ Found' if anthropic_key else '✗ Missing'}")
    
    if not tavily_key:
        print("⚠️  WARNING: TAVILY_API_KEY not found - search will fail")
    if not anthropic_key:
        print("⚠️  WARNING: ANTHROPIC_API_KEY not found - using fallback summaries")
    
    all_papers: List[PaperItem] = []
    
    # 각 트렌드 처리
    for trend_idx, trend in enumerate(state.discovered, 1):
        trend_title = trend.title
        
        print(f"\n[{trend_idx}/{len(state.discovered)}] Processing: {trend_title}")
        state.log(f"[Profiling] ========== TREND {trend_idx}/{len(state.discovered)}: {trend_title} ==========")
        
        try:
            # Google Scholar 검색 수행
            state.log(f"[Profiling] Step 1: Web search")
            results = _web_search_scholar(trend_title, state, max_results=20)
            
            if not results:
                state.log(f"[Profiling] ✗ No search results for: {trend_title}")
                print(f"  ⚠️  No search results found - skipping")
                continue
            
            # 논문 파싱
            state.log(f"[Profiling] Step 2: Parse papers")
            papers = _parse_scholar_results(results, state)
            
            if not papers:
                state.log(f"[Profiling] ✗ No valid papers after parsing for: {trend_title}")
                print(f"  ⚠️  No valid papers found after parsing - skipping")
                continue
            
            print(f"  ✓ Found {len(papers)} papers")
            
            # 기술 요약 생성
            state.log(f"[Profiling] Step 3: Generate tech summary")
            tech_summary = _summarize_technology(trend_title, papers, state)
            
            # PaperItem 생성
            state.log(f"[Profiling] Step 4: Create PaperItem objects")
            for paper_idx, paper in enumerate(papers, 1):
                paper_item = PaperItem(
                    trend_title=trend_title,
                    paper_title=paper['title'],
                    authors=paper['authors'],
                    year=paper['year'],
                    abstract=paper['abstract'],
                    citation=paper['citation'],
                    url=paper['url'],
                    tech_summary=tech_summary
                )
                all_papers.append(paper_item)
                state.log(f"[Profiling] Created PaperItem {paper_idx}/{len(papers)}")
            
            print(f"  ✓ Generated tech summary ({len(tech_summary)} chars)")
            state.log(f"[Profiling] ✓ Successfully processed trend: {trend_title}")
            
        except Exception as e:
            state.log(f"[Profiling] ✗ CRITICAL ERROR processing {trend_title}: {str(e)}")
            state.log(f"[Profiling] Stack trace:\n{traceback.format_exc()}")
            print(f"  ✗ ERROR: {str(e)}")
            continue
    
    # 상태에 저장
    state.academic_papers = all_papers
    state.log(f"[Profiling] ========== COMPLETE ==========")
    state.log(f"[Profiling] Total papers collected: {len(all_papers)}")
    state.log(f"[Profiling] Trends processed: {len(set(p.trend_title for p in all_papers))}")
    
    # 터미널에 결과 출력
    _print_results_summary(state)
    
    return state


# ---------------------------------------------------------
# 테스트 실행용 메인 블록
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    print("\n" + "="*80)
    print("🧪 TREND PROFILING AGENT - TEST MODE")
    print("="*80)
    
    from bio_agent_code.state import PipelineState, TrendItem
    
    # 샘플 트렌드 데이터 생성 (Discovery Agent 결과 시뮬레이션)
    sample_trends = [
        TrendItem(
            title="AI Drug Discovery",
            summary="AI를 활용한 신약 개발 트렌드",
            sources=["https://labiotech.eu", "https://fiercebiotech.com"],
            tags=["discovery", "news"],
            score=0.95
        ),
        TrendItem(
            title="Protein Folding AI",
            summary="AlphaFold 등 단백질 구조 예측 AI",
            sources=["https://labiotech.eu"],
            tags=["discovery", "news"],
            score=0.90
        ),
    ]
    
    state = PipelineState(target_count=2)
    state.discovered = sample_trends
    
    print(f"\n✓ Created test state with {len(sample_trends)} sample trends:")
    for i, trend in enumerate(sample_trends, 1):
        print(f"   {i}. {trend.title} (score: {trend.score})")
    
    print("\n🚀 Starting Trend Profiling Agent...\n")
    
    # Agent 실행
    try:
        result_state = run(state)
        
        print(f"\n" + "="*80)
        print("✅ TEST COMPLETE!")
        print("="*80)
        print(f"Papers collected: {len(result_state.academic_papers)}")
        print(f"Logs generated: {len(result_state._logs)}")
        
        if result_state.academic_papers:
            print(f"\nFirst paper example:")
            paper = result_state.academic_papers[0]
            print(f"  Trend: {paper.trend_title}")
            print(f"  Title: {paper.paper_title[:60]}...")
            print(f"  Year: {paper.year}")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED!")
        print(f"Error: {str(e)}")
        print(f"\nStack trace:")
        traceback.print_exc()
        
        
        
        # 또는 코드에서
from bio_agent_code.agents.Trend_Profiling_Agent import run
result_state = run(state)

# 기술 요약 확인
for paper in result_state.academic_papers[:1]:  # 첫 번째 트렌드
    print(paper.tech_summary)