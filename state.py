# --- state.py (멀티 에이전트 파이프라인 연결 완성본) ---
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import sys

# =========================================================
# ① Trend Discovery 결과 (기사·키워드 기반)
# =========================================================
@dataclass
class TrendItem:
    title: str
    summary: str
    sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    score: float = 0.0  # Presence-like score

# =========================================================
# ② Trend Profiling 결과 (학술 논문)
# =========================================================
@dataclass
class PaperItem:
    trend_title: str       # 연결될 트렌드명
    paper_title: str       # 논문 제목
    authors: str           # 저자
    year: int              # 출판 연도
    abstract: str          # 초록
    citation: str          # 인용 구문
    url: str               # 링크
    tech_summary: str = "" # 트렌드 기술 요약(해당 트렌드 공통 요약)

# =========================================================
# ③ Trend Evaluation 결과 (정규화 점수 및 분류)
#    Trend_Evaluation_Agent가 요구하는 스키마
# =========================================================
@dataclass
class EvaluationMetrics:
    trend_title: str

    # 정규화 점수 (0~1)
    current_presence: float
    momentum: float
    sustainability: float

    # 분류 결과
    development_stage: str     # 예: "🚀 Emerging → Growth"
    market_feasibility: str    # 예: "High - ..."

    # 상세 원시 데이터 (리포트/디버깅용)
    presence_details: Dict[str, Any] = field(default_factory=dict)
    momentum_details: Dict[str, Any] = field(default_factory=dict)
    sustainability_details: Dict[str, Any] = field(default_factory=dict)

# =========================================================
# ④ Risk & Opportunity 결과 (RAG) — Report_Generation과 호환
# =========================================================
@dataclass
class Evidence:
    doc_name: str
    page: int
    snippet: str
    score: float

@dataclass
class RiskItem:
    title: str
    description: str
    severity: str                 # Low/Medium/High/Critical
    likelihood: str               # Low/Medium/High
    impact: str
    timeframe: str                # near/mid/long
    stakeholders: List[str]
    mitigations: List[str]
    confidence: str
    evidences: List[Evidence] = field(default_factory=list)

@dataclass
class OpportunityItem:
    title: str
    description: str
    potential_value: str
    feasibility: str              # Low/Medium/High
    impact_scope: str             # patient/provider/system/market/global
    timeframe: str
    stakeholders: List[str]
    enablers: List[str]
    confidence: str
    evidences: List[Evidence] = field(default_factory=list)

@dataclass
class RiskOpportunityResult:
    query: str
    risks: List[RiskItem] = field(default_factory=list)
    opportunities: List[OpportunityItem] = field(default_factory=list)
    overall_confidence: str = "Medium"
    notes: str = ""

# =========================================================
# ⑤ 파이프라인 상태 (모든 에이전트 공용 상태 컨테이너)
# =========================================================
@dataclass
class PipelineState:
    target_count: int = 10

    # 단계별 산출물
    discovered: List[TrendItem] = field(default_factory=list)          # Discovery
    academic_papers: List[PaperItem] = field(default_factory=list)     # Profiling
    evaluations: List[EvaluationMetrics] = field(default_factory=list) # Evaluation
    risk_opportunity: Optional[RiskOpportunityResult] = None           # Risk & Opportunity

    # 메타/로그
    meta: Dict[str, Any] = field(default_factory=dict)
    _logs: List[str] = field(default_factory=list, repr=False)

    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr)
        self._logs.append(msg)

    def summary(self) -> str:
        return (
            f"[State Summary]\n"
            f"- Discovered: {len(self.discovered)}\n"
            f"- Academic Papers: {len(self.academic_papers)}\n"
            f"- Evaluations: {len(self.evaluations)}\n"
            f"- Risk/Opportunity: {'set' if self.risk_opportunity else 'none'}\n"
        )

# =========================================================
# ⑥ Risk_Opportunity_Agent 결과를 state에 장착하는 유틸
#    - AnalysisResult dataclass 또는 dict, 둘 다 수용
# =========================================================
def _to_evidence_list(items: List[Dict[str, Any]]) -> List[Evidence]:
    evs: List[Evidence] = []
    for it in items or []:
        evs.append(Evidence(
            doc_name=str(it.get("doc_name", "")),
            page=int(it.get("page", -1)),
            snippet=str(it.get("snippet", ""))[:1000],
            score=float(it.get("score", 0.0)),
        ))
    return evs

def attach_risk_opportunity(
    state: PipelineState,
    result: Union[RiskOpportunityResult, Dict[str, Any], Any]
) -> PipelineState:
    """
    Risk_Opportunity_Agent.analyze(...) 반환값을
    PipelineState.risk_opportunity에 장착한다.
    """
    if isinstance(result, RiskOpportunityResult):
        state.risk_opportunity = result
        state.log("[State] risk_opportunity attached (dataclass)")
        return state

    # dict 또는 dataclass(asdict 가능) 처리
    try:
        data: Dict[str, Any] = asdict(result)
    except Exception:
        data = dict(result) if isinstance(result, dict) else {}

    ro = RiskOpportunityResult(
        query=data.get("query", ""),
        overall_confidence=data.get("overall_confidence", "Medium"),
        notes=data.get("notes", "")
    )

    for r in data.get("risks", []) or []:
        ro.risks.append(RiskItem(
            title=r.get("title", ""),
            description=r.get("description", ""),
            severity=r.get("severity", "Medium"),
            likelihood=r.get("likelihood", "Medium"),
            impact=r.get("impact", ""),
            timeframe=r.get("timeframe", "mid"),
            stakeholders=list(r.get("stakeholders", [])),
            mitigations=list(r.get("mitigations", [])),
            confidence=r.get("confidence", "Medium"),
            evidences=_to_evidence_list(r.get("evidences", []))
        ))

    for o in data.get("opportunities", []) or []:
        ro.opportunities.append(OpportunityItem(
            title=o.get("title", ""),
            description=o.get("description", ""),
            potential_value=o.get("potential_value", ""),
            feasibility=o.get("feasibility", "Medium"),
            impact_scope=o.get("impact_scope", "system"),
            timeframe=o.get("timeframe", "mid"),
            stakeholders=list(o.get("stakeholders", [])),
            enablers=list(o.get("enablers", [])),
            confidence=o.get("confidence", "Medium"),
            evidences=_to_evidence_list(o.get("evidences", []))
        ))

    state.risk_opportunity = ro
    state.log("[State] risk_opportunity attached (coerced)")
    return state

__all__ = [
    "PipelineState",
    "TrendItem", "PaperItem", "EvaluationMetrics",
    "Evidence", "RiskItem", "OpportunityItem", "RiskOpportunityResult",
    "attach_risk_opportunity",
]
