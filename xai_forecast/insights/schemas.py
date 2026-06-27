"""Data contracts for the insights pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateFinding:
    finding_id:   str
    finding_type: str   # over_forecast_bias | dominant_driver | demand_cliff |
                        # external_coincidence | counterfactual_material | contrastive_gap
    score:        float  # 0.0 – 1.0, higher = more evidence
    summary:      str    # one-line human-readable trigger description
    evidence:     dict[str, Any] = field(default_factory=dict)  # raw evidence rows


@dataclass
class EvidencePack:
    finding_id:   str
    finding_type: str
    summary:      str
    data:         dict[str, Any]     # structured evidence for LLM consumption
    raw_refs:     list[str]          # traceability: table/key refs (e.g. "xai_results:2013-03-02")
    data_confidence: str = 'low'     # high | medium | low


@dataclass
class Hypothesis:
    finding_id:  str
    headline:    str
    explanation: str   # for the target audience (ds or business)
    evidence_refs: list[str]   # which evidence fields support this
    confidence:  str = 'low'   # high | medium | low


@dataclass
class Critique:
    finding_id:    str
    status:        str   # accepted | rejected | needs_review
    confidence:    str   # high | medium | low (may downgrade from hypothesis)
    notes:         str   # critic reasoning (always populated)
    overclaim:     bool = False
    causal_external: bool = False  # True if the hypothesis wrongly claimed external causation


@dataclass
class LedgerRow:
    finding_id:   str
    finding_type: str
    status:       str
    confidence:   str
    evidence:     dict[str, Any]
    hypothesis:   dict[str, Any] | None
    critic_notes: str
