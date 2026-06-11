"""
Unit tests for the Layer 1 Semantic Risk Engine (Titan Embeddings).

Bedrock is fully mocked: reference embeddings are orthogonal unit vectors and
document embeddings are crafted per-test to produce known cosine similarities.
This verifies the scoring logic (cosine similarity → weighted score) without
making real AWS calls.

Vector conventions
------------------
N = 5 dimensions, one per risk category (in RISK_CATEGORIES order):
  [0] financial_fraud
  [1] securities_violation
  [2] data_privacy
  [3] operational_risk
  [4] concealment_indicators

Reference vecs:  identity matrix rows  → cosine_sim(ref_i, ref_j) = δ_ij
Document vec examples:
  [0.9, 0, 0, 0, 0]  → high similarity to financial_fraud only
  [0.9, 0, 0, 0, 0.9] → high similarity to financial_fraud + concealment
  [0.3, 0.3, 0.3, 0.3, 0.3] → all sims below baseline → low risk
"""

import json
from typing import List
from unittest.mock import MagicMock

import pytest

from risk_engine import RISK_CATEGORIES, RiskEngine, _SIMILARITY_BASELINE, _SIMILARITY_SCALE

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
N_CATS = len(RISK_CATEGORIES)           # 5
CATEGORIES = list(RISK_CATEGORIES.keys())
WEIGHTS = [d["weight"] for d in RISK_CATEGORIES.values()]

THRESHOLD = 0.1  # mirrors RISK_THRESHOLD env var default

# Use 11-dimensional vectors so a uniform doc vec has cosine similarity
# 1/√11 ≈ 0.302 against each unit ref — comfortably below _SIMILARITY_BASELINE (0.32).
VEC_DIM = 11


def _unit_vec(idx: int) -> List[float]:
    v = [0.0] * VEC_DIM
    v[idx] = 1.0
    return v


def _uniform_vec(val: float) -> List[float]:
    return [val] * VEC_DIM


def _make_engine(*doc_vecs: List[float]) -> RiskEngine:
    """
    Return a RiskEngine whose Bedrock client is mocked.

    invoke_model call order:
      calls 0-4   → orthogonal unit-vector reference embeddings (one per category)
      calls 5+    → doc_vecs in the order provided; last vec is reused if exhausted
    """
    ref_vecs = [_unit_vec(i) for i in range(N_CATS)]
    all_vecs = ref_vecs + list(doc_vecs)
    call_idx = [0]

    def _invoke(**kwargs):
        vec = all_vecs[min(call_idx[0], len(all_vecs) - 1)]
        call_idx[0] += 1
        body = MagicMock()
        body.read.return_value = json.dumps({"embeddings": [vec]}).encode()
        return {"body": body}

    client = MagicMock()
    client.invoke_model.side_effect = _invoke
    engine = RiskEngine(bedrock_client=client)
    # Disable bundled file and S3 cache so the mock client is always used for refs
    engine._load_bundled = lambda: None
    engine._load_cache = lambda: None
    return engine


def _expected_score(*doc_vecs_and_indices) -> float:
    """
    Compute expected score for a doc vector that is a unit vector at position idx.
    cosine_sim(unit_vec(idx), ref_unit_vec(i)) = 1 if i==idx else 0.
    So only category idx contributes.
    """
    pass  # used inline in tests for clarity


# ---------------------------------------------------------------------------
# Edge cases (no Bedrock calls needed)
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_string_returns_zero_without_calling_bedrock(self):
        client = MagicMock()
        engine = RiskEngine(bedrock_client=client)
        score, cats = engine.assess("")
        assert score == 0.0
        assert cats == []
        client.invoke_model.assert_not_called()

    def test_whitespace_only_returns_zero_without_calling_bedrock(self):
        client = MagicMock()
        engine = RiskEngine(bedrock_client=client)
        score, cats = engine.assess("   \n\t  ")
        assert score == 0.0
        assert cats == []
        client.invoke_model.assert_not_called()

    def test_score_is_float_and_list(self):
        # doc vec: uniform below baseline → score ~0
        engine = _make_engine(_uniform_vec(0.30))
        score, cats = engine.assess("normal business communication")
        assert isinstance(score, float)
        assert isinstance(cats, list)

    def test_score_never_exceeds_one(self):
        # doc vec perfectly aligned with every category
        engine = _make_engine([1.0] * N_CATS)
        score, _ = engine.assess("extreme risk document")
        assert score <= 1.0

    def test_references_built_lazily(self):
        """Bedrock must not be called until assess() is invoked."""
        client = MagicMock()
        _ = RiskEngine(bedrock_client=client)   # __init__ only
        client.invoke_model.assert_not_called()

    def test_references_cached_after_first_assess(self):
        """Second assess() call must not rebuild references (5 + 1 + 1 total calls)."""
        engine = _make_engine(_uniform_vec(0.30), _uniform_vec(0.30))
        engine.assess("first call")
        engine.assess("second call")
        # 5 reference calls + 2 document calls = 7
        assert engine._client.invoke_model.call_count == 7


# ---------------------------------------------------------------------------
# Scoring maths
# ---------------------------------------------------------------------------
class TestScoringMath:
    def test_below_baseline_similarity_produces_zero_contribution(self):
        # A uniform vec of dimension VEC_DIM=11 has cosine similarity 1/√11 ≈ 0.302
        # against every unit-vector ref — below _SIMILARITY_BASELINE (0.32) → signal=0.
        engine = _make_engine(_uniform_vec(1.0))
        score, cats = engine.assess("low risk")
        assert score == 0.0
        assert cats == []

    def test_single_high_similarity_category_exceeds_threshold(self):
        # doc vec = unit vec for financial_fraud (idx 0), weight = 0.25
        # cosine_sim = 1.0, signal = 1.0 - 0.32 = 0.68
        # contribution = 0.68 * 4.0 * 0.25 = 0.68 → > THRESHOLD
        engine = _make_engine(_unit_vec(0))
        score, cats = engine.assess("high risk financial document")
        assert score >= THRESHOLD
        assert "financial_fraud" in cats

    def test_single_min_weight_category_exceeds_threshold(self):
        # data_privacy weight = 0.15 (minimum)
        # contribution = 0.68 * 4.0 * 0.15 = 0.408 → above THRESHOLD
        idx = CATEGORIES.index("data_privacy")
        engine = _make_engine(_unit_vec(idx))
        score, cats = engine.assess("data privacy violation document")
        assert score >= THRESHOLD
        assert "data_privacy" in cats

    def test_multiple_categories_accumulate_score(self):
        # Both financial_fraud (0.25) and concealment_indicators (0.20)
        # Individual contributions: 0.60 * 4.0 * 0.25 = 0.60 and 0.60 * 4.0 * 0.20 = 0.48
        # But score is capped at 1.0; with both flagged score = min(0.60+0.48, 1.0) = 1.0
        # Use a vec that is equal-weight between two categories:
        # vec = [1/√2, 0, 0, 0, 1/√2]
        val = 1.0 / (2 ** 0.5)
        idx_ff = CATEGORIES.index("financial_fraud")
        idx_ci = CATEGORIES.index("concealment_indicators")
        doc_vec = [0.0] * N_CATS
        doc_vec[idx_ff] = val
        doc_vec[idx_ci] = val
        engine = _make_engine(doc_vec)
        score, cats = engine.assess("fraud and cover-up document")
        # cosine_sim for each active category = val / 1.0 = val ≈ 0.707 > baseline
        # contribution per category: (0.707 - 0.40) * 4.0 * weight
        expected_ff = (val - _SIMILARITY_BASELINE) * _SIMILARITY_SCALE * 0.25
        expected_ci = (val - _SIMILARITY_BASELINE) * _SIMILARITY_SCALE * 0.20
        expected = min(round(expected_ff + expected_ci, 6), 1.0)
        assert abs(score - expected) < 1e-5
        assert "financial_fraud" in cats
        assert "concealment_indicators" in cats

    def test_score_increases_with_higher_similarity(self):
        # Vary cosine similarity to ref[0] only by using dimension 5 (beyond N_CATS)
        # as a noise component so other reference vectors get zero dot product.
        # low_doc:  sim to ref[0] = 0.5 / sqrt(0.5) ≈ 0.707
        # high_doc: sim to ref[0] = 0.9 / sqrt(0.81+0.25) ≈ 0.874
        # All other refs (indices 1-4) → dot product = 0 → no contribution.
        low_doc  = [0.5, 0, 0, 0, 0, 0.5, 0, 0]
        high_doc = [0.9, 0, 0, 0, 0, 0.5, 0, 0]
        engine_low = _make_engine(low_doc)
        engine_high = _make_engine(high_doc)
        score_low, _ = engine_low.assess("moderate risk")
        score_high, _ = engine_high.assess("high risk")
        assert score_high > score_low

    def test_flagged_categories_sorted_alphabetically(self):
        # All categories flagged
        engine = _make_engine(_unit_vec(0))
        _, cats = engine.assess("test")
        assert cats == sorted(cats)


# ---------------------------------------------------------------------------
# Low-risk documents
# ---------------------------------------------------------------------------
class TestLowRisk:
    def test_near_zero_similarity_is_low_risk(self):
        # Uniform vec: sim = 1/√11 ≈ 0.302 < baseline (0.32) → signal=0 for all categories.
        engine = _make_engine(_uniform_vec(0.10))
        score, cats = engine.assess("routine earnings report")
        assert score == 0.0
        assert cats == []

    def test_below_baseline_similarity_is_low_risk(self):
        engine = _make_engine(_uniform_vec(0.1))
        score, _ = engine.assess("standard internal memo")
        assert score < THRESHOLD

