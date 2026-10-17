"""Pydantic schemas used to validate classical and LLM outputs before anything is stored."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ActionItemModel(BaseModel):
    task: str = Field(min_length=3, max_length=300)
    owner: Optional[str] = Field(default=None, max_length=80)
    deadline_text: Optional[str] = Field(default=None, max_length=80)
    deadline_date: Optional[date] = None
    evidence_idx: List[int] = Field(min_length=1)

    @field_validator("owner", "deadline_text", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return v.strip() or None if isinstance(v, str) else v


class DecisionModel(BaseModel):
    text: str = Field(min_length=3, max_length=400)
    evidence_idx: List[int] = Field(min_length=1)


class LLMCandidate(BaseModel):
    """What the fallback LLM must return for one uncertain sentence."""
    is_action_item: bool = False
    is_decision: bool = False
    task: Optional[str] = None
    owner: Optional[str] = None
    deadline_text: Optional[str] = None
    evidence_quote: str = ""
    uncertain_fields: List[str] = Field(default_factory=list)


class QAAnswer(BaseModel):
    answer: str
    supported: bool = False
    citations: List[int] = Field(default_factory=list)  # chunk ids actually used
