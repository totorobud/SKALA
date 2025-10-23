#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk_Opportunity_Agent.py
RAG 기반 'AI in Health' 리스크·기회 분석 에이전트

기능 요약
- 두 OECD PDF를 페이지 단위로 파싱하고, 의미 단위로 청크화하여 메타데이터(문서명, 페이지)를 보존
- HuggingFace 임베딩 + FAISS 로컬 벡터스토어에 인덱싱/영속화
- 상위 k 증거 청크를 기반으로 LLM이 구조화된 리스크/기회 JSON을 생성(신뢰도/근거 페이지 포함)
- LLM 백엔드: OpenAI(기본, 환경변수 OPENAI_API_KEY 필요) 또는 Ollama(로컬 모델) 선택 가능
- CLI 지원: 인덱스 빌드/업데이트, 질의, 원문 근거 미리보기

의존 패키지(예시)
pip install pypdf sentence-transformers faiss-cpu langchain langchain-community langchain-openai tiktoken
# (Ollama 사용 시) 로컬에 ollama 서버 및 모델 설치 필요: https://ollama.ai
"""

from __future__ import annotations
import os
import re
import json
import argparse
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# PDF 파싱 & 청크
from pypdf import PdfReader

# Vector store & embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM 백엔드
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OpenAI or Ollama 선택적 로딩
LLM_BACKEND = os.getenv("RO_AGENT_LLM_BACKEND", "openai").lower()  # "openai" | "ollama"
OPENAI_MODEL = os.getenv("RO_AGENT_OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("RO_AGENT_OLLAMA_MODEL", "llama3.1")

# 경로 기본값(누나가 준 경로 그대로)
DEFAULT_PDFS = [
    "/Users/songsua/Desktop/SKALA/21.bio_ai_agent/bio_agent_code/data/OECD_2024 – AI in Health- Huge Potential, Huge Risks (2024) .pdf",
    "/Users/songsua/Desktop/SKALA/21.bio_ai_agent/bio_agent_code/data/OECD_2024_FRAMEWORK FOR ANTICIPATORY GOVERNANCE OF EMERGING TECHNOLOGIES.pdf",
]
DEFAULT_INDEX_DIR = "./ro_agent_faiss_index"

# =========================
# 데이터 스키마
# =========================

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
    severity: str        # e.g., Low/Medium/High/Critical
    likelihood: str      # e.g., Low/Medium/High
    impact: str          # qualitative summary
    timeframe: str       # e.g., near(0-1y)/mid(1-3y)/long(3y+)
    stakeholders: List[str]
    mitigations: List[str]
    confidence: str      # Low/Medium/High
    evidences: List[Evidence]

@dataclass
class OpportunityItem:
    title: str
    description: str
    potential_value: str   # qualitative/quantitative where possible
    feasibility: str       # Low/Medium/High
    impact_scope: str      # patient/provider/system/market
    timeframe: str
    stakeholders: List[str]
    enablers: List[str]
    confidence: str
    evidences: List[Evidence]

@dataclass
class AnalysisResult:
    query: str
    risks: List[RiskItem]
    opportunities: List[OpportunityItem]
    overall_confidence: str
    notes: str

# =========================
# 유틸: PDF 파서 & 청크러
# =========================

def read_pdf_to_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """PDF를 페이지별 텍스트 리스트로 반환"""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # 공백 정리
        txt = re.sub(r"[ \t]+", " ", txt).strip()
        pages.append((i+1, txt))  # 1-based page no.
    return pages

def make_documents_from_pdfs(pdf_paths: List[str]) -> List[Document]:
    """각 페이지를 Document로 만들고, 의미 단위로 재-청크"""
    docs: List[Document] = []
    raw_docs: List[Document] = []

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
        basename = os.path.basename(path)
        pages = read_pdf_to_pages(path)
        for page_no, text in pages:
            if not text:
                continue
            raw_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "doc_name": basename,
                        "page": page_no
                    }
                )
            )

    # 의미 기반 청크: 긴 페이지를 섹션/문장 기준으로 분절
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", "•", "-", "—", " "]
    )
    for d in raw_docs:
        for chunk in splitter.split_text(d.page_content):
            if not chunk.strip():
                continue
            docs.append(
                Document(
                    page_content=chunk.strip(),
                    metadata=d.metadata.copy()
                )
            )

    return docs

# =========================
# 인덱싱/저장
# =========================

def build_or_update_index(
    pdf_paths: List[str],
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> FAISS:
    """PDF를 읽어 벡터스토어를 생성하거나 업데이트"""
    os.makedirs(index_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    docs = make_documents_from_pdfs(pdf_paths)
    if not docs:
        raise RuntimeError("No documents parsed from given PDFs.")

    index_path = os.path.join(index_dir, "faiss.index")
    store_path = os.path.join(index_dir, "index.pkl")

    if os.path.exists(index_path) and os.path.exists(store_path):
        print("[INFO] Existing FAISS index found. Updating...")
        db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
        db.save_local(index_dir)
    else:
        print("[INFO] Building new FAISS index...")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(index_dir)

    print(f"[INFO] Index ready at: {index_dir} (docs={len(docs)})")
    return db

def load_index(
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db

# =========================
# LLM 백엔드
# =========================

def get_llm():
    """
    OpenAI 또는 Ollama LLM을 반환.
    반환 객체는 LangChain Runnable 호환( .invoke(inputs) 로 텍스트)하게 구성
    """
    if LLM_BACKEND == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception as e:
            raise RuntimeError("Install langchain-ollama to use OLLAMA backend: pip install langchain-ollama") from e
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
        return llm | StrOutputParser()

    # default: openai
    try:
        from langchain_openai import ChatOpenAI
    except Exception as e:
        raise RuntimeError("Install langchain-openai: pip install langchain-openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    return llm | StrOutputParser()

# =========================
# 프롬프트
# =========================

SYSTEM_RULES = """You are a rigorous analyst specializing in AI in healthcare policy, safety, and governance.
Follow strictly:
- Use only the provided EVIDENCE. Do not invent facts.
- When uncertain, say so; lower confidence accordingly.
- Output MUST be valid JSON following the requested schema.
- Concise, decision-useful writing. No fluff.
- Reflect OECD language where appropriate (risk governance, transparency, accountability, equity, safety, data quality).
"""

USER_TASK_TEMPLATE = """
GOAL: Analyze AI in health using provided evidence and answer the user query:
QUERY: "{query}"

Return a JSON with:
{{
  "query": "...",
  "risks": [
    {{
      "title": "...",
      "description": "...",
      "severity": "Low|Medium|High|Critical",
      "likelihood": "Low|Medium|High",
      "impact": "...",
      "timeframe": "near(0-1y)|mid(1-3y)|long(3y+)",
      "stakeholders": ["patients","providers","regulators","payers","developers", "..."],
      "mitigations": ["...","..."],
      "confidence": "Low|Medium|High",
      "evidences": [{{"doc_name":"...","page":1,"snippet":"...","score":0.0}}]
    }}
  ],
  "opportunities": [
    {{
      "title": "...",
      "description": "...",
      "potential_value": "...",
      "feasibility": "Low|Medium|High",
      "impact_scope": "patient|provider|system|market|global",
      "timeframe": "near|mid|long",
      "stakeholders": ["..."],
      "enablers": ["..."],
      "confidence": "Low|Medium|High",
      "evidences": [{{"doc_name":"...","page":1,"snippet":"...","score":0.0}}]
    }}
  ],
  "overall_confidence": "Low|Medium|High",
  "notes": "short synthesis/assumptions"
}}

EVIDENCE (top-{k} chunks):
{evidence_block}

Important:
- Map mitigations/enablers to OECD governance levers where possible (e.g., transparency, oversight mechanisms, data governance, evaluation/monitoring, procurement standards, workforce training).
- Do NOT include any text outside the JSON.
"""

PROMPT = PromptTemplate.from_template(SYSTEM_RULES + USER_TASK_TEMPLATE)

# =========================
# 분석 실행
# =========================

def retrieve_evidence(
    db: FAISS,
    query: str,
    k: int = 8
) -> Tuple[str, List[Dict[str, Any]]]:
    """RAG 검색: 상위 k개 청크를 문자열 블록과 메타로 반환"""
    results = db.similarity_search_with_score(query, k=k)
    blocks = []
    items: List[Dict[str, Any]] = []
    for doc, score in results:
        meta = doc.metadata or {}
        doc_name = meta.get("doc_name", os.path.basename(meta.get("source", "")))
        page = int(meta.get("page", -1))
        snippet = doc.page_content[:800].replace("\n", " ").strip()
        blocks.append(f"[{doc_name} | p.{page} | score={score:.3f}] {snippet}")
        items.append({
            "doc_name": doc_name,
            "page": page,
            "snippet": snippet,
            "score": float(score)
        })
    return "\n\n".join(blocks), items

def coerce_schema(json_text: str) -> AnalysisResult:
    """LLM 출력을 스키마로 변환(유효성 보정 포함)"""
    # JSON만 추출
    m = re.search(r"\{.*\}\s*$", json_text, flags=re.S)
    data = json.loads(m.group(0) if m else json_text)

    def _evidences(raw_list):
        evs: List[Evidence] = []
        for r in raw_list or []:
            evs.append(Evidence(
                doc_name=r.get("doc_name",""),
                page=int(r.get("page", -1)),
                snippet=r.get("snippet","")[:500],
                score=float(r.get("score", 0.0))
            ))
        return evs

    risks: List[RiskItem] = []
    for r in data.get("risks", []):
        risks.append(RiskItem(
            title=r.get("title","").strip(),
            description=r.get("description","").strip(),
            severity=r.get("severity","Medium"),
            likelihood=r.get("likelihood","Medium"),
            impact=r.get("impact","").strip(),
            timeframe=r.get("timeframe","mid"),
            stakeholders=r.get("stakeholders",[]),
            mitigations=r.get("mitigations",[]),
            confidence=r.get("confidence","Medium"),
            evidences=_evidences(r.get("evidences",[]))
        ))

    opps: List[OpportunityItem] = []
    for o in data.get("opportunities", []):
        opps.append(OpportunityItem(
            title=o.get("title","").strip(),
            description=o.get("description","").strip(),
            potential_value=o.get("potential_value","").strip(),
            feasibility=o.get("feasibility","Medium"),
            impact_scope=o.get("impact_scope","system"),
            timeframe=o.get("timeframe","mid"),
            stakeholders=o.get("stakeholders",[]),
            enablers=o.get("enablers",[]),
            confidence=o.get("confidence","Medium"),
            evidences=_evidences(o.get("evidences",[]))
        ))

    result = AnalysisResult(
        query=data.get("query",""),
        risks=risks,
        opportunities=opps,
        overall_confidence=data.get("overall_confidence","Medium"),
        notes=data.get("notes","").strip()
    )
    return result

def analyze(
    query: str,
    db: Optional[FAISS] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    k: int = 8
) -> AnalysisResult:
    """주요 함수: 질의 → 증거 검색 → LLM 구조화 분석 → 스키마 반환"""
    if db is None:
        db = load_index(index_dir=index_dir)

    evidence_block, evidence_items = retrieve_evidence(db, query, k=k)

    llm = get_llm()
    prompt = PROMPT.format(query=query, k=k, evidence_block=evidence_block)

    raw = llm.invoke({"input": prompt}) if callable(getattr(llm, "invoke", None)) else llm(prompt)
    result = coerce_schema(raw)

    # LLM이 넣은 evidence에 검색 점수 병합(문서명+페이지로 매칭)
    score_map = {(e["doc_name"], e["page"]): e["score"] for e in evidence_items}
    def _merge_scores(evs: List[Evidence]):
        for e in evs:
            key = (e.doc_name, e.page)
            if key in score_map:
                e.score = float(score_map[key])

    for r in result.risks:
        _merge_scores(r.evidences)
    for o in result.opportunities:
        _merge_scores(o.evidences)

    # 질의 저장
    result.query = query
    return result

# =========================
# 출력/미리보기
# =========================

def print_result(result: AnalysisResult, max_items: int = 5):
    print("\n=== RISK & OPPORTUNITY ANALYSIS ===")
    print(f"Query: {result.query}")
    print(f"Overall Confidence: {result.overall_confidence}")
    if result.notes:
        print(f"Notes: {result.notes}")

    def _fmt_ev(e: Evidence) -> str:
        return f"- [{e.doc_name} p.{e.page} | score={e.score:.3f}] {e.snippet[:160]}..."

    print("\n--- Risks ---")
    for i, r in enumerate(result.risks[:max_items], 1):
        print(f"{i}. {r.title} | Sev:{r.severity} Lik:{r.likelihood} Conf:{r.confidence}")
        print(f"   Impact: {r.impact}")
        print(f"   Timeframe: {r.timeframe} | Stakeholders: {', '.join(r.stakeholders[:6])}")
        print(f"   Mitigations: " + "; ".join(r.mitigations[:4]))
        if r.evidences:
            print("   Evidence:")
            for ev in r.evidences[:3]:
                print("   " + _fmt_ev(ev))

    print("\n--- Opportunities ---")
    for i, o in enumerate(result.opportunities[:max_items], 1):
        print(f"{i}. {o.title} | Feasibility:{o.feasibility} Scope:{o.impact_scope} Conf:{o.confidence}")
        print(f"   Value: {o.potential_value}")
        print(f"   Timeframe: {o.timeframe} | Stakeholders: {', '.join(o.stakeholders[:6])}")
        print(f"   Enablers: " + "; ".join(o.enablers[:4]))
        if o.evidences:
            print("   Evidence:")
            for ev in o.evidences[:3]:
                print("   " + _fmt_ev(ev))

def preview_sources(db: FAISS, query: str, k: int = 5):
    """상위 증거 청크 간단 미리보기"""
    block, _ = retrieve_evidence(db, query, k=k)
    print("\n=== Evidence Preview ===")
    print(block)

# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="RAG-based Risk & Opportunity Agent for AI in Health (OECD PDFs)"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="Build/Update FAISS index from PDFs")
    p_build.add_argument("--pdfs", nargs="*", default=DEFAULT_PDFS, help="List of PDF paths")
    p_build.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p_build.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")

    p_query = sub.add_parser("query", help="Run analysis on a query")
    p_query.add_argument("--q", required=True, help="Analysis question/prompt")
    p_query.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p_query.add_argument("--k", type=int, default=8)
    p_query.add_argument("--preview", action="store_true", help="Show evidence preview before analysis")
    p_query.add_argument("--json", action="store_true", help="Print full JSON result")

    args = parser.parse_args()

    if args.cmd == "build":
        build_or_update_index(args.pdfs, args.index_dir, args.embedding_model)

    elif args.cmd == "query":
        db = load_index(args.index_dir)
        if args.preview:
            preview_sources(db, args.q, k=args.k)
        result = analyze(args.q, db=db, index_dir=args.index_dir, k=args.k)
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print_result(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
