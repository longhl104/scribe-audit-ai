"""
ScribeAudit AI — Lambda Handler
================================
Triggered by S3 OBJECT_CREATED events. Orchestrates the two-layer AI pipeline:

  Layer 1 → Classical NLP risk scoring (RiskEngine)
  Layer 2 → Amazon Bedrock compliance agent (only for HIGH-RISK docs)

Results are persisted to DynamoDB.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3

from risk_engine import RiskEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Module-level clients — initialised once per Lambda container (warm reuse)
# ---------------------------------------------------------------------------
_REGION = os.environ.get("AWS_REGION", "us-east-1")
s3_client = boto3.client("s3", region_name=_REGION)
dynamodb = boto3.resource("dynamodb", region_name=_REGION)
bedrock_client = boto3.client("bedrock-runtime", region_name=_REGION)

# ---------------------------------------------------------------------------
# Configuration (injected via Lambda environment variables)
# ---------------------------------------------------------------------------
TABLE_NAME: str = os.environ["DYNAMODB_TABLE"]
RISK_THRESHOLD: float = float(os.environ.get("RISK_THRESHOLD", "0.35"))
BEDROCK_MODEL_ID: str = os.environ.get(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)
MAX_BEDROCK_CHARS: int = int(os.environ.get("MAX_BEDROCK_CHARS", "4000"))

_risk_engine = RiskEngine()


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
def lambda_handler(event: dict, context) -> dict:
    table = dynamodb.Table(TABLE_NAME)
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        logger.info("Processing s3://%s/%s", bucket, key)

        try:
            result = _process_document(table, bucket, key)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process s3://%s/%s: %s", bucket, key, exc)
            results.append({"bucket": bucket, "key": key, "error": str(exc)})

    return {"statusCode": 200, "body": json.dumps(results)}


# ---------------------------------------------------------------------------
# Core processing logic (separated for testability)
# ---------------------------------------------------------------------------
def _process_document(table, bucket: str, key: str) -> dict:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    text: str = response["Body"].read().decode("utf-8")

    # --- Layer 1 ---
    risk_score, flagged_terms = _risk_engine.assess(text)
    document_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Layer 2 (conditional) ---
    if risk_score >= RISK_THRESHOLD:
        audit_summary = invoke_bedrock_agent(text, flagged_terms)
        risk_level = "HIGH"
    else:
        audit_summary = {
            "verdict": "LOW_RISK",
            "details": "No significant compliance indicators detected by Layer 1 engine.",
        }
        risk_level = "LOW"

    logger.info(
        "document_id=%s risk_level=%s score=%.4f flagged=%s",
        document_id,
        risk_level,
        risk_score,
        flagged_terms,
    )

    # --- Persist ---
    item = {
        "document_id": document_id,
        "timestamp": timestamp,
        "s3_bucket": bucket,
        "s3_key": key,
        "risk_level": risk_level,
        "risk_score": Decimal(str(round(risk_score, 6))),
        "flagged_terms": flagged_terms,
        "audit_summary": json.dumps(audit_summary),
    }
    table.put_item(Item=item)

    return {
        "document_id": document_id,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "flagged_terms": flagged_terms,
    }


# ---------------------------------------------------------------------------
# Bedrock compliance agent
# ---------------------------------------------------------------------------
def invoke_bedrock_agent(text: str, flagged_terms: list) -> dict:
    """
    Send the document to Amazon Bedrock and request a structured JSON audit report.
    Uses Claude 3 Haiku by default for cost efficiency.
    """
    prompt = (
        "You are a compliance auditor specialising in financial regulation. "
        "Analyse the following business transcript for regulatory violations.\n\n"
        f"Pre-detected risk indicators: {', '.join(flagged_terms) if flagged_terms else 'none'}\n\n"
        f"Document (truncated to {MAX_BEDROCK_CHARS} chars):\n"
        f"{text[:MAX_BEDROCK_CHARS]}\n\n"
        "Respond ONLY with a valid JSON object matching this exact schema — no markdown fences:\n"
        "{\n"
        '  "verdict": "HIGH_RISK | MEDIUM_RISK | LOW_RISK",\n'
        '  "regulatory_framework": ["applicable regulation names"],\n'
        '  "violations": [\n'
        '    {\n'
        '      "type": "violation category",\n'
        '      "description": "concise description",\n'
        '      "severity": "CRITICAL | HIGH | MEDIUM",\n'
        '      "excerpt": "verbatim quote from document"\n'
        '    }\n'
        '  ],\n'
        '  "recommended_actions": ["action items"],\n'
        '  "summary": "one-paragraph executive summary"\n'
        "}"
    )

    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
    )

    response = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=request_body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    raw_text: str = response_body["content"][0]["text"]

    # Extract the JSON object from the response
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw_text[start:end])
        except json.JSONDecodeError:
            logger.warning("Bedrock returned malformed JSON; storing raw response.")

    return {"verdict": "PARSE_ERROR", "raw_response": raw_text}
