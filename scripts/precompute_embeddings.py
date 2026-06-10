#!/usr/bin/env python3
"""
Pre-compute reference embeddings for the ScribeAudit risk categories.

Run this script once (locally or in CI) whenever RISK_CATEGORIES change.
Outputs lambda/reference_embeddings.json, which is bundled with the Lambda
so cold starts never need to call Titan for reference vectors.

Usage:
    python scripts/precompute_embeddings.py [--profile <aws-profile>] [--region <region>]

Example:
    python scripts/precompute_embeddings.py --profile longhl104 --region ap-southeast-2
"""

import argparse
import json
import os
import sys
import time

# Allow importing from the lambda package without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda"))

import boto3
from risk_engine import RISK_CATEGORIES, EMBEDDING_MODEL_ID

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "lambda", "reference_embeddings.json"
)
DELAY_BETWEEN_CALLS = 3.0   # seconds — stay within Bedrock TPS on first run
MAX_RETRIES = 8
BASE_DELAY = 2.0


def embed_with_backoff(client, text: str, label: str) -> list:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=json.dumps({"texts": [text], "input_type": "search_document"}),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())["embeddings"][0]
        except Exception as exc:
            resp = getattr(exc, "response", {}) or {}
            err = resp.get("Error", {})
            code = err.get("Code", type(exc).__name__)
            message = err.get("Message", str(exc))
            if attempt == 0:
                print(f"  [{label}] Error code: {code!r}")
                print(f"  [{label}] Error message: {message!r}")
            if code in ("ThrottlingException", "ServiceUnavailableException") and attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + 1.0
                print(f"  [{label}] Throttled (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Titan reference embeddings.")
    parser.add_argument("--profile", default=None, help="AWS profile name")
    parser.add_argument("--region", default="ap-southeast-2", help="AWS region")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("bedrock-runtime")

    print(f"Using model: {EMBEDDING_MODEL_ID}")
    print(f"Region:      {args.region}")
    print(f"Output:      {os.path.abspath(OUTPUT_PATH)}\n")

    result = {}
    for i, (category, data) in enumerate(RISK_CATEGORIES.items()):
        print(f"[{i+1}/{len(RISK_CATEGORIES)}] Embedding '{category}'...")
        vector = embed_with_backoff(client, data["reference"], category)
        result[category] = {
            "weight": data["weight"],
            "vector": vector,
        }
        print(f"  ✓ {len(vector)}-dimensional vector")
        if i < len(RISK_CATEGORIES) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"model_id": EMBEDDING_MODEL_ID, "categories": result}, f)

    print(f"\n✅ Saved {len(result)} reference embeddings to {OUTPUT_PATH}")
    print("   Commit this file and redeploy to eliminate cold-start Titan calls.")


if __name__ == "__main__":
    main()
