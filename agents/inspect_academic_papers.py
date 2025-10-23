#!/usr/bin/env python3
"""
academic_papers 데이터 구조 확인 스크립트

실행 방법:
    python inspect_academic_papers.py
"""

import os
import sys
import json
from collections import defaultdict, Counter

# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bio_agent_code.state import PipelineState, TrendItem
from bio_agent_code.agents.Trend_Profiling_Agent import run


def inspect_paper_item(paper, index=0):
    """PaperItem 객체의 상세 정보 출력"""
    print(f"\n{'='*80}")
    print(f"📄 Paper #{index + 1}")
    print(f"{'='*80}")
    print(f"Trend Title:    {paper.trend_title}")
    print(f"Paper Title:    {paper.paper_title}")
    print(f"Authors:        {paper.authors or '(없음)'}")
    print(f"Year:           {paper.year}")
    print(f"URL:            {paper.url}")
    print(f"\nAbstract (처음 200자):")
    print(f"  {paper.abstract[:200]}...")
    print(f"\nCitation:")
    print(f"  {paper.citation}")
    print(f"\nTech Summary (처음 300자):")
    print(f"  {paper.tech_summary[:300]}...")
    print(f"\n필드 길이:")
    print(f"  - abstract: {len(paper.abstract)} 문자")
    print(f"  - tech_summary: {len(paper.tech_summary)} 문자")


def analyze_academic_papers(state):
    """academic_papers 데이터 전체 분석"""
    
    papers = state.academic_papers
    
    if not papers:
        print("⚠️  academic_papers가 비어있습니다!")
        return
    
    print("\n" + "="*80)
    print("📊 ACADEMIC PAPERS 데이터 분석")
    print("="*80)
    
    # 1. 기본 통계
    print(f"\n1️⃣ 기본 통계:")
    print(f"   총 논문 수: {len(papers)}개")
    print(f"   트렌드 수: {len(set(p.trend_title for p in papers))}개")
    
    # 2. 트렌드별 분포
    print(f"\n2️⃣ 트렌드별 논문 수:")
    trend_counts = Counter(p.trend_title for p in papers)
    for trend, count in trend_counts.most_common():
        print(f"   • {trend}: {count}개")
    
    # 3. 연도별 분포
    print(f"\n3️⃣ 연도별 논문 수:")
    year_counts = Counter(p.year for p in papers)
    for year, count in sorted(year_counts.items(), reverse=True):
        print(f"   • {year}: {count}개")
    
    # 4. 트렌드별 그룹화
    print(f"\n4️⃣ 트렌드별 상세 정보:")
    trends_dict = defaultdict(list)
    for paper in papers:
        trends_dict[paper.trend_title].append(paper)
    
    for trend_name, trend_papers in trends_dict.items():
        print(f"\n   [{trend_name}]")
        print(f"   논문 수: {len(trend_papers)}개")
        
        # 연도 분포
        years = Counter(p.year for p in trend_papers)
        print(f"   연도 분포: {dict(years)}")
        
        # 저자 정보 통계
        has_authors = sum(1 for p in trend_papers if p.authors)
        print(f"   저자 정보 있음: {has_authors}개 / {len(trend_papers)}개")
        
        # 상위 3개 논문
        print(f"   상위 3개 논문:")
        for i, paper in enumerate(trend_papers[:3], 1):
            print(f"      {i}. {paper.paper_title[:50]}... ({paper.year})")
    
    # 5. 필드별 통계
    print(f"\n5️⃣ 필드별 통계:")
    
    # abstract 길이
    abstract_lengths = [len(p.abstract) for p in papers]
    print(f"   Abstract 길이:")
    print(f"      평균: {sum(abstract_lengths) / len(abstract_lengths):.1f}자")
    print(f"      최소: {min(abstract_lengths)}자")
    print(f"      최대: {max(abstract_lengths)}자")
    
    # tech_summary 길이
    tech_summary_lengths = [len(p.tech_summary) for p in papers]
    print(f"   Tech Summary 길이:")
    print(f"      평균: {sum(tech_summary_lengths) / len(tech_summary_lengths):.1f}자")
    print(f"      최소: {min(tech_summary_lengths)}자")
    print(f"      최대: {max(tech_summary_lengths)}자")
    
    # 고유한 tech_summary 개수
    unique_summaries = len(set(p.tech_summary for p in papers))
    print(f"   고유한 Tech Summary: {unique_summaries}개")
    
    # 6. 데이터 품질 체크
    print(f"\n6️⃣ 데이터 품질 체크:")
    
    empty_authors = sum(1 for p in papers if not p.authors)
    print(f"   • 저자 정보 없음: {empty_authors}개 ({empty_authors/len(papers)*100:.1f}%)")
    
    empty_abstracts = sum(1 for p in papers if not p.abstract)
    print(f"   • Abstract 없음: {empty_abstracts}개 ({empty_abstracts/len(papers)*100:.1f}%)")
    
    empty_urls = sum(1 for p in papers if not p.url)
    print(f"   • URL 없음: {empty_urls}개 ({empty_urls/len(papers)*100:.1f}%)")
    
    missing_summaries = sum(1 for p in papers if not p.tech_summary)
    print(f"   • Tech Summary 없음: {missing_summaries}개 ({missing_summaries/len(papers)*100:.1f}%)")


def export_to_json(state, filename="academic_papers.json"):
    """academic_papers를 JSON 파일로 내보내기"""
    papers_data = []
    
    for paper in state.academic_papers:
        papers_data.append({
            "trend_title": paper.trend_title,
            "paper_title": paper.paper_title,
            "authors": paper.authors,
            "year": paper.year,
            "abstract": paper.abstract,
            "citation": paper.citation,
            "url": paper.url,
            "tech_summary": paper.tech_summary
        })
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 데이터 내보내기 완료: {filename}")
    print(f"   파일 크기: {os.path.getsize(filename):,} bytes")


def main():
    print("\n" + "="*80)
    print("🔍 ACADEMIC PAPERS 데이터 구조 검사")
    print("="*80)
    
    # 샘플 데이터로 Agent 실행
    print("\n📊 샘플 데이터로 Agent 실행 중...")
    
    sample_trends = [
        TrendItem(
            title="AI Drug Discovery",
            summary="AI 신약 개발",
            sources=["https://example.com"],
            tags=["ai"],
            score=0.95
        ),
    ]
    
    state = PipelineState(target_count=1)
    state.discovered = sample_trends
    
    print("⚠️  주의: Tavily API를 사용하므로 실제 검색이 수행됩니다.\n")
    
    try:
        result_state = run(state)
        
        if not result_state.academic_papers:
            print("⚠️  논문이 수집되지 않았습니다. API 키를 확인하세요.")
            return
        
        # 전체 분석
        analyze_academic_papers(result_state)
        
        # 첫 번째 논문 상세 정보
        print("\n" + "="*80)
        print("📋 샘플 논문 상세 정보")
        print("="*80)
        inspect_paper_item(result_state.academic_papers[0], 0)
        
        # JSON 내보내기
        export_to_json(result_state, "academic_papers_sample.json")
        
        # 사용 예시
        print("\n" + "="*80)
        print("💡 다음 Agent에서 사용하는 방법")
        print("="*80)
        print("""
# 트렌드별로 그룹화
from collections import defaultdict

trends = defaultdict(list)
for paper in state.academic_papers:
    trends[paper.trend_title].append(paper)

# 각 트렌드의 논문 순회
for trend_name, papers in trends.items():
    print(f"\\n=== {trend_name} ===")
    print(f"논문 수: {len(papers)}")
    print(f"기술 요약: {papers[0].tech_summary[:100]}...")
    
    for paper in papers:
        print(f"  - {paper.paper_title} ({paper.year})")
        print(f"    {paper.url}")
        """)
        
        print("\n✅ 검사 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()