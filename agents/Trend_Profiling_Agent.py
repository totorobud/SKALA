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

def _fetch_paper_content(url: str, state: PipelineState) -> str:
    """
    논문 URL에서 전체 내용 가져오기
    """
    if not _HAS_TAVILY:
        return ""
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ""
    
    try:
        client = TavilyClient(api_key=api_key)
        state.log(f"[Profiling] Fetching full content from: {url[:50]}...")
        
        # Tavily extract를 사용하여 전체 내용 추출
        result = client.extract(urls=[url])
        
        if result and 'results' in result and len(result['results']) > 0:
            full_content = result['results'][0].get('raw_content', '')
            state.log(f"[Profiling] ✓ Fetched {len(full_content)} chars from paper")
            return full_content[:5000]  # 최대 5000자로 제한
        
        return ""
        
    except Exception as e:
        state.log(f"[Profiling] ⚠️ Could not fetch full content: {str(e)}")
        return ""

def _summarize_technology(trend_title: str, papers: List[Dict], state: PipelineState) -> str:
    """
    Claude API를 사용하여 논문들을 기반으로 기술 요약 생성
    - 논문 전체 내용을 가져와서 분석
    - 기술 중심의 구체적 설명
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
        
        # 상위 3개 논문의 전체 내용 가져오기
        papers_with_content = []
        for i, paper in enumerate(papers[:3], 1):
            state.log(f"[Profiling] Fetching content for paper {i}/3...")
            full_content = _fetch_paper_content(paper['url'], state)
            
            papers_with_content.append({
                'title': paper['title'],
                'authors': paper['authors'],
                'year': paper['year'],
                'abstract': paper['abstract'],
                'full_content': full_content if full_content else paper['abstract']
            })
        
        # 논문 정보를 프롬프트에 포함
        papers_context = "\n\n" + "="*80 + "\n\n"
        papers_context += "\n\n".join([
            f"[논문 {i+1}]\n제목: {p['title']}\n저자: {p['authors']}\n연도: {p['year']}\n\n내용:\n{p['full_content']}"
            for i, p in enumerate(papers_with_content)
        ])
        
        state.log(f"[Profiling] Prepared {len(papers_with_content)} papers with full content")
        
        prompt = f"""당신은 생명과학 분야 기술 분석 전문가입니다.

다음은 "{trend_title}" 기술과 관련된 최근 학술 논문들입니다:

{papers_context}

위 논문들의 내용을 바탕으로 "{trend_title}" 기술에 대해 다음 형식으로 기술적 설명을 작성하세요:

## 기술 개요
(이 기술이 무엇인지, 어떤 문제를 해결하는지 2-3줄로 기술적으로 설명)

## 핵심 메커니즘
(이 기술이 작동하는 원리나 핵심 접근 방식을 구체적으로 설명. 알고리즘, 모델 구조, 데이터 처리 방식 등 기술적 세부사항 포함)

## 주요 적용 분야
(이 기술이 적용되는 생명과학의 구체적 분야들을 나열)

## 기술적 특징
(논문에서 언급된 이 기술의 고유한 특성이나 장점을 기술적 관점에서 설명)

**작성 원칙:**
1. 마케팅 문구, 일반론적 표현 금지
2. 논문에서 언급된 구체적 기술, 방법론, 결과만 포함
3. "혁신", "획기적", "지속적 연구 개발" 같은 추상적 표현 사용 금지
4. 기술 자체의 작동 방식과 과학적 원리에 집중
5. 논문 내용을 근거로 사실만 서술"""

        state.log("[Profiling] Calling Claude API for technical summary...")
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = message.content[0].text
        state.log(f"[Profiling] ✓ Claude API returned technical summary ({len(summary)} chars)")
        
        return summary
        
    except Exception as e:
        state.log(f"[Profiling] ✗ ERROR calling Claude API: {str(e)}")
        state.log(f"[Profiling] Stack trace: {traceback.format_exc()}")
        return _fallback_summary(trend_title, papers, state)

def _fallback_summary(trend_title: str, papers: List[Dict], state: PipelineState) -> str:
    """API 실패 시 논문 초록 기반 기술 요약 생성"""
    state.log("[Profiling] Using fallback summary generation (abstract-based)")
    
    if not papers:
        return f"{trend_title}: 논문 데이터 없음"
    
    # 논문 초록들에서 기술적 키워드와 패턴 추출
    abstracts = [p['abstract'] for p in papers[:5] if p['abstract']]
    
    summary = f"## {trend_title}\n\n"
    summary += f"**데이터 기반:** {len(papers)}개 논문 분석 ({', '.join(str(p['year']) for p in papers[:5])})\n\n"
    
    # 상위 논문 제목과 초록 요약
    summary += "**주요 연구 내용:**\n"
    for i, paper in enumerate(papers[:3], 1):
        summary += f"{i}. {paper['title']}\n"
        if paper['abstract']:
            # 초록의 핵심 문장 추출 (첫 150자)
            core_abstract = paper['abstract'][:150].strip()
            summary += f"   - {core_abstract}...\n"
    
    summary += f"\n**기술 분류:** {trend_title} 분야\n"
    summary += f"**활성 연구 기간:** 최근 {RECENT_YEARS}년\n"
    
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
