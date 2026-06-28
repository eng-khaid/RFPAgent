from rfp_rag.schemas import EvidenceChunk

LANGUAGE_NAMES = {"ar": "Arabic", "en": "English"}


def build_feasibility_prompt(
    rfp_text: str,
    evidence: list[EvidenceChunk],
    company_name: str = "",
    language: str = "ar",
) -> str:
    company = company_name.strip() or "the company described in the knowledge base"
    lang_name = LANGUAGE_NAMES.get(language, "Arabic")
    evidence_text = "\n\n".join(
        f"[{idx}. {chunk.label()}]\n{chunk.content}"
        for idx, chunk in enumerate(evidence, start=1)
    )

    return f"""You are evaluating whether {company} can deliver what this RFP requires.

Use only the RFP text and the retrieved knowledge-base evidence.
Do not invent capabilities that are not supported by evidence.

OUTPUT LANGUAGE: Write all human-readable text in {lang_name} -- this includes
proposal_markdown, the decision, and every requirement's category, item, and
value. Keep the JSON keys and the priority field (high/medium/low) in English.
Evidence citations like [01_Certifications_Compliance_Registrations.md] keep
their original filenames.

Scoring criteria:
- Capability Match: 40 points
- Relevant Experience: 25 points
- Technology / Vendor Fit: 20 points
- Delivery Risk: 10 points
- Missing Information: 5 points

Return only valid JSON. Do not wrap it in Markdown.

Expected JSON shape:
{{
  "meta": {{
    "title": "RFP title",
    "issuer": "issuing organization or Unknown",
    "deadline": "submission deadline or Unknown",
    "duration": "project duration or Unknown",
    "budget": "budget or Unknown",
    "ref": "reference number or empty string",
    "rating": "0-100/100",
    "decision": "Strong Fit | Possible | Partial Fit | Weak Fit | Not Recommended"
  }},
  "requirements": [
    {{
      "category": "Infrastructure | Security | ELV | Data Center | Network | Cloud | Support | Other",
      "item": "requirement from the RFP",
      "value": "short feasibility note grounded in evidence",
      "priority": "high | medium | low"
    }}
  ],
  "proposal_markdown": "{lang_name} Markdown feasibility report with evidence citations like [01_Certifications_Compliance_Registrations.md]"
}}

RFP TEXT:
{rfp_text}

KNOWLEDGE BASE EVIDENCE:
{evidence_text}
"""
