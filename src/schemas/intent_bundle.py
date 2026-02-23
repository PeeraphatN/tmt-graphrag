"""
Intent Bundle schema for hierarchical intent routing.

This module intentionally does not wire into the current pipeline yet.
It provides a stable contract for the next implementation phase.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionIntent(str, Enum):
    LOOKUP = "lookup"
    VERIFY = "verify"
    LIST = "list"
    COUNT = "count"
    COMPARE = "compare"
    UNKNOWN = "unknown"


class FacetIntent(str, Enum):
    MANUFACTURER = "manufacturer"
    INGREDIENT = "ingredient"
    NLEM = "nlem"
    FORMULA = "formula"
    HIERARCHY = "hierarchy"
    ID_LOOKUP = "id_lookup"
    GENERAL = "general"


class SlotSource(str, Enum):
    RULE = "rule"
    NER = "ner"
    VECTOR = "vector"
    MANUAL = "manual"


class SlotValue(BaseModel):
    name: str = Field(description="Slot name, e.g. tmtid or manufacturer.")
    value: str = Field(description="Extracted slot value.")
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the extracted slot value.",
    )
    source: SlotSource = Field(
        SlotSource.RULE,
        description="Source used to extract the slot value.",
    )


class IntentControlFeatures(BaseModel):
    token_count: int = Field(0, ge=0)
    entity_token_count: int = Field(0, ge=0)
    entity_ratio: float = Field(0.0, ge=0.0, le=1.0)
    has_exact_id: float = Field(0.0, ge=0.0, le=1.0)
    has_dose_unit: float = Field(0.0, ge=0.0, le=1.0)
    ambiguity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Optional uncertainty score from classifier heads.",
    )


class RetrievalPlan(BaseModel):
    retrieval_mode: str = Field(
        "balanced",
        description="High-level retrieval mode (e.g. balanced/vector_heavy/fulltext_heavy).",
    )
    vector_weight: float = Field(0.5, ge=0.0, le=1.0)
    fulltext_weight: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(10, ge=1)
    depth: int = Field(2, ge=0)
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured filters forwarded to search.",
    )
    must_match: list[str] = Field(
        default_factory=list,
        description="Terms or ids that should be enforced as exact constraints.",
    )


class IntentBundle(BaseModel):
    query: str = Field(description="Original user query.")

    action_intent: ActionIntent = Field(
        ActionIntent.UNKNOWN,
        description="Primary action intent (single-label).",
    )
    facet_intents: list[FacetIntent] = Field(
        default_factory=list,
        description="Facet intents (multi-label).",
    )
    slots: list[SlotValue] = Field(
        default_factory=list,
        description="Extracted structured slots from rule/NER/vector paths.",
    )

    action_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Optional score map from action classifier head.",
    )
    facet_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Optional score map from facet classifier head.",
    )

    control_features: IntentControlFeatures = Field(default_factory=IntentControlFeatures)
    adaptive_retrieval_weights: RetrievalPlan = Field(default_factory=RetrievalPlan)

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Debug metadata for observability and offline analysis.",
    )

    def has_facet(self, facet: FacetIntent | str) -> bool:
        facet_value = facet.value if isinstance(facet, FacetIntent) else str(facet)
        return any(f.value == facet_value for f in self.facet_intents)

