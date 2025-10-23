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