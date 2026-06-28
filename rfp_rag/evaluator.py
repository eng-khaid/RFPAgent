from __future__ import annotations

import re
from pathlib import Path

from main.hybrid_search import HybridSearchRetriever
from parsing_pipeline.ingestion import SCRAPED_DIR
from rfp_rag.document_loader import load_rfp_text
from rfp_rag.llm_judge import DeepSeekJudge
from rfp_rag.prompts import build_feasibility_prompt
from rfp_rag.schemas import EvaluationResult, EvidenceChunk, Requirement

# "| Legal entity | Acme Corp, Inc. (Delaware C-Corp) |" -> "Acme Corp, Inc."
_LEGAL_ENTITY_RE = re.compile(r"\|\s*legal entity\s*\|\s*([^|]+?)\s*\|", re.IGNORECASE)
# Fallback: the "<Name> is a ..." opening sentence of an overview document.
_OVERVIEW_NAME_RE = re.compile(r"\b([A-Z][\w&.,'-]*(?:\s+[A-Z0-9][\w&.,'-]*){0,5})\s+is\s+a\b")


def detect_company_name(scraped_dir: Path = SCRAPED_DIR) -> str:
    """Best-effort company name read from the knowledge base.

    Looks for an explicit '| Legal entity | <name> |' row first, then falls back
    to the '<Name> is a ...' opening sentence of the company overview. Returns ''
    when nothing matches, so the prompt stays generic instead of guessing a name.
    """
    # Scan overview/company files first; the legal-entity row lives there.
    files = sorted(
        scraped_dir.glob("*.md"),
        key=lambda p: not any(k in p.stem.lower() for k in ("overview", "company")),
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        match = _LEGAL_ENTITY_RE.search(text)
        if match:
            name = re.sub(r"\s*\(.*?\)\s*$", "", match.group(1)).strip()
            if name:
                return name
    for path in files:
        match = _OVERVIEW_NAME_RE.search(path.read_text(encoding="utf-8"))
        if match:
            return match.group(1).strip()
    return ""


class RfpRagEvaluator:
    def __init__(
        self,
        top_k: int = 6,
        use_vector: bool = True,
        use_llm: bool = True,
        judge: DeepSeekJudge | None = None,
        max_docs: int = 5,
        company_name: str | None = None,
    ) -> None:
        self.use_vector = use_vector
        self.use_llm = use_llm
        self.max_docs = max_docs
        self.judge = judge
        # Read the company name from the knowledge base unless one is passed in.
        self.company_name = detect_company_name() if company_name is None else company_name
        self.retriever = HybridSearchRetriever(top_k=top_k, use_vector=use_vector)

    def evaluate_file(self, path: str, language: str = "ar") -> EvaluationResult:
        rfp_text = load_rfp_text(path)
        evidence = self.retrieve_evidence(rfp_text)

        if not self.use_llm:
            return self._evaluate_without_llm(rfp_text, evidence, language)

        judge = self.judge or DeepSeekJudge(company_name=self.company_name)
        return judge.judge(rfp_text=rfp_text, evidence=evidence, language=language)

    def _evaluate_without_llm(
        self,
        rfp_text: str,
        evidence: list[EvidenceChunk],
        language: str = "ar",
    ) -> EvaluationResult:
        score = self._estimate_evidence_score(evidence)

        proposal_markdown = self._build_draft_report(rfp_text, evidence, language)
        return EvaluationResult(
            feasibility_score=score,
            decision=self._evidence_decision(score),
            requirements=[
                Requirement(
                    category="Analysis",
                    item="RFP text embedded once and matched against the knowledge base",
                    value=f"{len(evidence)} source documents retrieved",
                    priority="high",
                )
            ],
            evidence=evidence,
            proposal_markdown=proposal_markdown,
        )

    def retrieve_evidence(self, rfp_text: str) -> list[EvidenceChunk]:
        # Per-chunk (multi-query) whole-document retrieval. The RFP is split into
        # small query windows so each stays inside the embedding model's
        # ~256-token budget -- the whole RFP participates in vector search, not
        # just its opening -- and each requirement finds its own best evidence
        # instead of being blended into one averaged query vector. Documents are
        # ranked by their best match to any requirement, then fed in full so a
        # terse fact (e.g. the "SOC 1 Type II" row) is never dropped with its
        # low-ranking chunk.
        queries = self._split_query(rfp_text)
        sources = self.retriever.retrieve_sources_multi(queries, max_docs=self.max_docs)
        return [self._source_to_evidence(source) for source in sources]

    def build_prompt(
        self, rfp_text: str, evidence: list[EvidenceChunk], language: str = "ar"
    ) -> str:
        return build_feasibility_prompt(
            rfp_text=rfp_text,
            evidence=evidence,
            company_name=self.company_name,
            language=language,
        )

    @staticmethod
    def _split_query(rfp_text: str, max_chars: int = 700) -> list[str]:
        """Split the RFP into small query windows for per-chunk retrieval.

        Lightweight on purpose -- paragraph packing, no semantic model -- because
        this runs on every upload. Paragraphs are packed into windows under
        ``max_chars`` (oversized paragraphs are hard-split) so each query stays
        within the embedding model's ~256-token budget.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", rfp_text) if p.strip()]
        windows: list[str] = []
        buffer = ""
        for para in paragraphs:
            if len(para) > max_chars:
                if buffer:
                    windows.append(buffer)
                    buffer = ""
                for start in range(0, len(para), max_chars):
                    windows.append(para[start:start + max_chars])
                continue
            if buffer and len(buffer) + 1 + len(para) > max_chars:
                windows.append(buffer)
                buffer = para
            else:
                buffer = f"{buffer}\n{para}" if buffer else para
        if buffer:
            windows.append(buffer)
        return windows

    @staticmethod
    def _estimate_evidence_score(evidence: list[EvidenceChunk]) -> int:
        if not evidence:
            return 15
        return min(75, 25 + (len(evidence) * 8))

    @staticmethod
    def _evidence_decision(score: int) -> str:
        if score >= 65:
            return "Preliminary fit - needs LLM judgment"
        if score >= 45:
            return "Partial fit - needs LLM judgment"
        return "Weak evidence match - needs LLM judgment"

    def _source_to_evidence(self, source: str) -> EvidenceChunk:
        return EvidenceChunk(
            source=source,
            chunk="full",
            content=self._load_source_text(source),
        )

    def _load_source_text(self, source: str) -> str:
        """Full Markdown for a source document, read from disk when available."""
        path = SCRAPED_DIR / source
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        # Fallback: stitch the stored chunks if the source file moved/renamed.
        return self.retriever.document_text(source)

    def _build_draft_report(
        self,
        rfp_text: str,
        evidence: list[EvidenceChunk],
        language: str = "ar",
    ) -> str:
        prompt = self.build_prompt(rfp_text, evidence, language)
        evidence_list = "\n".join(
            f"- {chunk.label()}: {chunk.content[:180].strip()}..."
            for chunk in evidence
        )

        if language == "en":
            company = self.company_name or "the knowledge base"
            return f"""# Preliminary RFP feasibility assessment

Read the RFP file and retrieved the closest evidence from {company}.

## Status
Ready to send to the language model for the final feasibility score.

## Retrieved evidence
{evidence_list or "- No matching evidence found."}

## Next step
Use the following text as the language-model prompt:

```text
{prompt}
```
"""

        kb_label = f"قاعدة معرفة {self.company_name}" if self.company_name else "قاعدة المعرفة"
        return f"""# تقييم أولي لملاءمة المناقصة

تمت قراءة ملف RFP واسترجاع الأدلة الأقرب من {kb_label}.

## الحالة
جاهز لإرساله إلى نموذج اللغة لإصدار درجة الجدوى النهائية.

## الأدلة المسترجعة
{evidence_list or "- لم يتم العثور على أدلة مطابقة."}

## الخطوة التالية
استخدم النص التالي كـ prompt لنموذج اللغة:

```text
{prompt}
```
"""
