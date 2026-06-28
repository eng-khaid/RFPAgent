from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Requirement:
    category: str
    item: str
    value: str = ""
    priority: str = "medium"

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "item": self.item,
            "value": self.value,
            "priority": self.priority,
        }


@dataclass
class EvidenceChunk:
    source: str
    chunk: int | str
    content: str

    def label(self) -> str:
        # Whole-document evidence has no meaningful chunk index; cite by source
        # only so the label matches the citation format asked of the judge.
        if self.chunk in (None, "", "full"):
            return self.source
        return f"{self.source} #{self.chunk}"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "chunk": self.chunk,
            "content": self.content,
        }


@dataclass
class EvaluationResult:
    title: str = "RFP feasibility review"
    issuer: str = "Unknown"
    deadline: str = "Unknown"
    duration: str = "Unknown"
    budget: str = "Unknown"
    ref: str = ""
    feasibility_score: int = 0
    decision: str = "Needs review"
    requirements: list[Requirement] = field(default_factory=list)
    evidence: list[EvidenceChunk] = field(default_factory=list)
    proposal_markdown: str = ""

    def to_frontend_dict(self) -> dict:
        return {
            "meta": {
                "title": self.title,
                "issuer": self.issuer,
                "deadline": self.deadline,
                "duration": self.duration,
                "budget": self.budget,
                "ref": self.ref,
                "rating": f"{self.feasibility_score}/100",
                "decision": self.decision,
            },
            "requirements": [req.to_dict() for req in self.requirements],
            "proposal_markdown": self.proposal_markdown,
            "evidence": [chunk.to_dict() for chunk in self.evidence],
        }
