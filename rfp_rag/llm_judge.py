from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

from rfp_rag.prompts import build_feasibility_prompt
from rfp_rag.schemas import EvaluationResult, EvidenceChunk, Requirement

load_dotenv()

DEEPSEEK_MODEL = "deepseek-chat"


def build_system_prompt(company_name: str = "") -> str:
    company = company_name.strip() or "the company described in the knowledge base"
    return f"""You are an expert bid/no-bid evaluator.
You judge RFP feasibility for {company} using retrieved evidence.
Return strict JSON only.
Every claim about {company}'s capability must be grounded in the supplied evidence.
"""


class DeepSeekJudgeError(RuntimeError):
    """Raised when the DeepSeek judge cannot return a usable result."""


class DeepSeekJudge:
    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        timeout_seconds: int = 90,
        company_name: str = "",
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.company_name = company_name

    def judge(
        self,
        rfp_text: str,
        evidence: list[EvidenceChunk],
        language: str = "ar",
    ) -> EvaluationResult:
        prompt = build_feasibility_prompt(
            rfp_text=rfp_text,
            evidence=evidence,
            company_name=self.company_name,
            language=language,
        )
        content = self._chat_completion(prompt)
        payload = self._parse_json_content(content)
        return self._payload_to_result(payload, evidence)

    def _chat_completion(self, prompt: str) -> str:
        api_key = get_deepseek_api_key()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_system_prompt(self.company_name)},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": False,
        }

        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise DeepSeekJudgeError(
                f"DeepSeek API returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekJudgeError("DeepSeek response did not include message content.") from exc

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise DeepSeekJudgeError("DeepSeek returned invalid JSON.") from exc

        if not isinstance(parsed, dict):
            raise DeepSeekJudgeError("DeepSeek JSON response must be an object.")
        return parsed

    @staticmethod
    def _payload_to_result(
        payload: dict[str, Any],
        evidence: list[EvidenceChunk],
    ) -> EvaluationResult:
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        score = _parse_score(
            payload.get("feasibility_score")
            or payload.get("score")
            or meta.get("rating")
            or meta.get("score")
        )

        requirements = []
        for item in payload.get("requirements", []):
            if not isinstance(item, dict):
                continue
            requirements.append(
                Requirement(
                    category=str(item.get("category") or "Other"),
                    item=str(item.get("item") or ""),
                    value=str(item.get("value") or ""),
                    priority=_normalize_priority(item.get("priority")),
                )
            )

        proposal = (
            payload.get("proposal_markdown")
            or payload.get("proposal_md")
            or payload.get("proposal")
            or ""
        )

        return EvaluationResult(
            title=str(meta.get("title") or "RFP feasibility review"),
            issuer=str(meta.get("issuer") or "Unknown"),
            deadline=str(meta.get("deadline") or "Unknown"),
            duration=str(meta.get("duration") or "Unknown"),
            budget=str(meta.get("budget") or "Unknown"),
            ref=str(meta.get("ref") or ""),
            feasibility_score=score,
            decision=str(meta.get("decision") or _decision_from_score(score)),
            requirements=requirements,
            evidence=evidence,
            proposal_markdown=str(proposal),
        )


def get_deepseek_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise DeepSeekJudgeError(
            "Missing DeepSeek API key. Add DEEPSEEK_API_KEY to .env."
        )
    return api_key


def _parse_score(value: Any) -> int:
    if isinstance(value, (int, float)):
        return max(0, min(100, int(round(value))))

    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return 0
    return max(0, min(100, int(round(float(match.group(0))))))


def _normalize_priority(value: Any) -> str:
    priority = str(value or "medium").lower().strip()
    if priority in {"high", "medium", "low"}:
        return priority
    return "medium"


def _decision_from_score(score: int) -> str:
    if score >= 85:
        return "Strong Fit"
    if score >= 70:
        return "Possible"
    if score >= 50:
        return "Partial Fit"
    if score >= 30:
        return "Weak Fit"
    return "Not Recommended"
