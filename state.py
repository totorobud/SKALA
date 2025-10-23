# --- state.py (lean & agent-safe) ---
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import sys

@dataclass
class TrendItem:
    title: str
    summary: str
    sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    score: float = 0.0

@dataclass
class PaperItem:
    """학술 논문 정보 - Trend_Profiling_Agent용"""
    trend_title: str          # 관련 트렌드 제목
    paper_title: str          # 논문 제목
    authors: str              # 저자
    year: int                 # 출판 연도
    abstract: str             # 초록
    citation: str             # 인용 구문
    url: str                  # 논문 URL
    tech_summary: str = ""    # 기술 요약 (해당 트렌드 전체에 대한)

@dataclass
class PipelineState:
    target_count: int = 10
    discovered: List[TrendItem] = field(default_factory=list)
    academic_papers: List[PaperItem] = field(default_factory=list)  # Trend_Profiling_Agent 결과 저장
    _logs: List[str] = field(default_factory=list, repr=False)

    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr)
        self._logs.append(msg)

__all__ = ["PipelineState", "TrendItem", "PaperItem"]


# state.py에 추가할 EvaluationMetrics 클래스

from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class EvaluationMetrics:
    """
    트렌드 평가 메트릭스
    
    세 가지 축:
    - Current Presence: 현재의 존재감 (0~1)
    - Momentum: 성장 속도 (0~1)
    - Sustainability: 지속 가능성 (0~1)
    """
    trend_title: str
    
    # 정규화된 점수 (0~1)
    current_presence: float  # 현재 존재감
    momentum: float          # 성장 속도
    sustainability: float    # 지속 가능성
    
    # 분류 결과
    development_stage: str   # 발전 단계 (e.g., "🚀 Emerging → Growth")
    market_feasibility: str  # 시장 적용 가능성
    
    # 상세 데이터
    presence_details: Dict = field(default_factory=dict)
    momentum_details: Dict = field(default_factory=dict)
    sustainability_details: Dict = field(default_factory=dict)


# PipelineState에 추가할 필드:
# evaluations: List[EvaluationMetrics] = field(default_factory=list)
