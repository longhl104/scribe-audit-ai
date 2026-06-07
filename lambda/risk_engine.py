"""
Layer 1 — Semantic Risk Engine (Amazon Titan Embeddings).

Replaces regex keyword matching with cosine-similarity scoring against
pre-computed reference embeddings for each risk category.  This approach
captures paraphrased and semantically equivalent language that exact-match
patterns would miss.

Dependencies: boto3 (pre-installed in the Lambda runtime).
"""

import json
import logging
import os
from typing import List, Tuple

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBEDDING_MODEL_ID: str = os.environ.get(
    "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

# ---------------------------------------------------------------------------
# Risk category definitions
# Each category has a calibrated weight and a rich reference description.
# Titan encodes the semantic meaning of the reference; the document is then
# scored by how similar its embedding is to each reference embedding.
# Combined weights sum to 1.0.
# ---------------------------------------------------------------------------
RISK_CATEGORIES: dict = {
    "financial_fraud": {
        "weight": 0.25,
        "reference": (
            "Financial fraud involving money laundering, embezzlement, bribery, "
            "corruption, misappropriation of funds, fictitious transactions, shell "
            "companies, hidden accounts, unauthorized fund transfers, kickbacks, "
            "and falsification of financial records or statements."
        ),
    },
    "securities_violation": {
        "weight": 0.25,
        "reference": (
            "Securities violations including insider trading on material non-public "
            "information, market manipulation, pump and dump schemes, front running, "
            "wash trading, naked short selling, spoofing, misrepresentation of "
            "investment securities, and failure to disclose material information."
        ),
    },
    "data_privacy": {
        "weight": 0.15,
        "reference": (
            "Data privacy violations including GDPR non-compliance, personal data "
            "breaches, unauthorized access to sensitive information, PII exposure, "
            "HIPAA violations, CCPA non-compliance, unencrypted data transmission, "
            "and unauthorized data exfiltration or sharing."
        ),
    },
    "operational_risk": {
        "weight": 0.15,
        "reference": (
            "Operational compliance failures including regulatory breaches, audit "
            "failures, internal control failures, policy violations, sanctions "
            "violations, anti-money laundering failures, know-your-customer failures, "
            "suspicious activity, unusual transactions, and transaction structuring "
            "to avoid regulatory reporting requirements."
        ),
    },
    "concealment_indicators": {
        "weight": 0.20,
        "reference": (
            "Concealment of misconduct including requests for off-the-record "
            "communications, instructions to delete messages or destroy evidence, "
            "covering up regulatory violations, bypassing internal controls, "
            "circumventing audit processes, avoiding documentation, undocumented "
            "payments, and informal arrangements to hide improper activities."
        ),
    },
}

# ---------------------------------------------------------------------------
# Scoring constants
#
# Titan Embeddings v2 produces cosine similarities in roughly [0.30, 0.95]
# for compliance text.  _SIMILARITY_BASELINE subtracts the expected background
# level so that unrelated documents contribute ~0 to the score.
# _SIMILARITY_SCALE amplifies the remaining signal so that a single category
# with high similarity (≥ 0.80) can breach the default RISK_THRESHOLD of 0.35.
#
# Tuning guide:
#   • Raise _SIMILARITY_BASELINE → fewer false positives, more false negatives.
#   • Lower _SIMILARITY_BASELINE → opposite.
#   • RISK_THRESHOLD (handler env var, default 0.35) is the final gate.
# ---------------------------------------------------------------------------
_SIMILARITY_BASELINE: float = 0.40
_SIMILARITY_SCALE: float = 4.0   # (1.0 - 0.40) * 4.0 * min_weight(0.15) ≈ 0.36 > 0.35


class RiskEngine:
    """
    Score a document for compliance risk using semantic embeddings.

    Reference embeddings are built lazily on the first call to assess() and
    cached for the lifetime of the container (warm Lambda reuse).  This avoids
    making Bedrock calls at module-import time, which simplifies testing and
    keeps cold-start module load fast.
    """

    def __init__(self, bedrock_client=None) -> None:
        self._client = bedrock_client or boto3.client(
            "bedrock-runtime", region_name=_REGION
        )
        self._references: dict | None = None  # built on first assess() call

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Invoke Titan Embeddings and return the embedding vector."""
        response = self._client.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"]

    def _build_references(self) -> dict:
        """Embed each category reference text once and cache the results."""
        logger.info("Building category reference embeddings (first invocation)...")
        refs = {}
        for category, data in RISK_CATEGORIES.items():
            refs[category] = {
                "weight": data["weight"],
                "vector": self._embed(data["reference"]),
            }
        logger.info("Reference embeddings ready for %d categories.", len(refs))
        return refs

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors; returns 0.0 for zero-norm inputs."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Public API  (same interface as the previous keyword engine)
    # ------------------------------------------------------------------

    def assess(self, text: str) -> Tuple[float, List[str]]:
        """
        Score a document for compliance risk.

        Returns
        -------
        score : float in [0.0, 1.0]
        flagged_categories : list[str]
            Categories whose signal (similarity − baseline) was positive,
            sorted alphabetically.

        Scoring model
        -------------
        For each risk category:

            signal       = max(0, cosine_similarity(doc, ref) − _SIMILARITY_BASELINE)
            contribution = signal × _SIMILARITY_SCALE × category_weight

        The baseline removes background semantic similarity that any compliance
        text would naturally share.  _SIMILARITY_SCALE (4.0) is calibrated so
        that a single category with similarity ≥ 0.80 produces a contribution
        above the default RISK_THRESHOLD of 0.35.  The final score is capped
        at 1.0.

        Titan Embeddings v2 token limit is 8,192; text is truncated to 8,000
        characters to stay safely within the limit.
        """
        if not text or not text.strip():
            return 0.0, []

        if self._references is None:
            self._references = self._build_references()

        # Truncate conservatively to stay within Titan's token limit
        doc_vector = self._embed(text[:8000])

        total_score = 0.0
        flagged: list = []

        for category, data in self._references.items():
            similarity = self._cosine_similarity(doc_vector, data["vector"])
            signal = max(0.0, similarity - _SIMILARITY_BASELINE)
            contribution = signal * _SIMILARITY_SCALE * data["weight"]
            total_score += contribution

            if signal > 0.0:
                flagged.append(category)

        return min(round(total_score, 6), 1.0), sorted(flagged)
