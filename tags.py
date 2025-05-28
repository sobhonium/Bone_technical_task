"""
Author: Sobhan
Date: 2025-05-28
Project: Bones Ltd. - Technical Challenge
File: tags.py

Description:
    Allows the use of customizable tags and sentiments for natural language analysis.
    This is intended to bypass limitations of predefined tag systems (e.g., spaCy, text2emotion).
    
    Note:
    Use caution when relying on LLMs to fill custom tags. Results may vary in reliability,
    especially with large tag sets. For instance, Groq's performance degrades with too many tags.
"""

from typing import Literal
from pydantic import BaseModel


class TagOutput(BaseModel):
    """
    A schema for tagging linguistic, affective, and contextual properties of a message.
    Useful for guiding LLMs with custom tag sets beyond standard NLP libraries.
    """

    # Basic Metadata
    language: str
    context: str

    # Linguistic/Discourse
    formality: Literal["very informal", "informal", "neutral", "formal", "very formal"]
    complexity: Literal["very simple", "simple", "moderate", "complex", "very complex"]
    coherence: Literal["incoherent", "somewhat coherent", "coherent", "very coherent"]
    verbosity: Literal["terse", "brief", "balanced", "verbose", "very verbose"]
    fluency: Literal["disfluent", "somewhat fluent", "fluent", "very fluent"]
    clarity: Literal["unclear", "somewhat clear", "clear", "very clear"]
    repetition: Literal["none", "low", "moderate", "high", "very high"]
    disfluency: Literal["none", "low", "moderate", "high", "very high"]

    # Emotional/Affective
    sentiment: Literal["very negative", "negative", "neutral", "positive", "very positive"]
    politeness: Literal["very impolite", "impolite", "neutral", "polite", "very polite"]
    empathy: Literal["none", "low", "moderate", "high", "very high"]
    aggressiveness: Literal["none", "low", "moderate", "high", "very high"]
    sarcasm: Literal["none", "low", "moderate", "high", "very high"]
    humor: Literal["none", "low", "moderate", "high", "very high"]
    hostility: Literal["none", "low", "moderate", "high", "very high"]
    emotional_intensity: Literal["none", "low", "moderate", "high", "very high"]

    # Optional / Extended Features
    # Uncomment and use as needed

    # # Cognitive/Reasoning
    # confidence: Literal["none", "low", "moderate", "high", "very high"]
    # certainty: Literal["very uncertain", "uncertain", "neutral", "certain", "very certain"]
    # curiosity: Literal["none", "low", "moderate", "high", "very high"]
    # confusion: Literal["none", "low", "moderate", "high", "very high"]
    # deception: Literal["none", "low", "moderate", "high", "very high"]
    # rationality: Literal["emotional", "mostly emotional", "balanced", "mostly logical", "logical"]

    # # Intent/Pragmatics
    # goal: str
    # intent_type: Literal["question", "inform", "persuade", "command", "express emotion", "other"]
    # persuasiveness: Literal["none", "low", "moderate", "high", "very high"]
    # manipulation: Literal["none", "low", "moderate", "high", "very high"]
    # cooperation_level: Literal["none", "low", "moderate", "high", "very high"]
    # topic_shift: Literal["none", "low", "moderate", "high"]
    # reference_resolution: Literal["poor", "fair", "good", "very good"]

    # # Relational/Interactional
    # dominance: Literal["submissive", "balanced", "dominant"]
    # supportiveness: Literal["none", "low", "moderate", "high", "very high"]
    # interruption_frequency: Literal["none", "low", "moderate", "high", "very high"]
    # turn_taking_balance: Literal["speaker-heavy", "balanced", "listener-heavy"]
    # alignment: Literal["disagree", "neutral", "agree"]
    # relationship_type: Literal["peer", "superior", "subordinate", "customer", "service provider", "other"]

    # # Meta/Structural
    # flow: Literal["poor", "fair", "good", "very good"]
    # engagement_level: Literal["none", "low", "moderate", "high", "very high"]
    # topic_relevance: Literal["irrelevant", "somewhat relevant", "relevant", "very relevant"]
    # new_information_ratio: Literal["none", "low", "moderate", "high", "very high"]
    # self_disclosure_level: Literal["none", "low", "moderate", "high", "very high"]
    # question_density: Literal["none", "low", "moderate", "high", "very high"]
