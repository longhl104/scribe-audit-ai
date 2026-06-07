"""
Integration-style tests for the Lambda handler.
AWS service calls are patched with unittest.mock — no real AWS account required.
"""

import json
from decimal import Decimal
from unittest.mock import MagicMock, call, patch

import pytest

import handler  # conftest.py already added lambda/ to sys.path

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------
LOW_RISK_TEXT = (
    "The quarterly earnings report confirms 5% year-over-year revenue growth. "
    "All financial statements have been independently audited and certified. "
    "The board approved the dividend reinvestment plan at the annual general meeting."
)

HIGH_RISK_TEXT = (
    "This is strictly off the record. We moved funds through a shell company to hide "
    "the unauthorised transfer. Make sure there is no paper trail — delete this once read. "
    "The money laundering operation must not appear in any audit trail."
)

S3_EVENT_FACTORY = lambda key="doc.txt": {
    "Records": [
        {
            "s3": {
                "bucket": {"name": "test-ingestion-bucket"},
                "object": {"key": key},
            }
        }
    ]
}


def _s3_body_mock(text: str):
    """Return a mock that mimics boto3 StreamingBody."""
    body = MagicMock()
    body.read.return_value = text.encode("utf-8")
    return {"Body": body}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLambdaHandler:
    def test_low_risk_document_stored_with_low_level(self):
        mock_table = MagicMock()

        with (
            patch("handler.s3_client") as mock_s3,
            patch("handler.dynamodb") as mock_ddb,
            patch("handler._risk_engine.assess", return_value=(0.10, [])),
        ):
            mock_s3.get_object.return_value = _s3_body_mock(LOW_RISK_TEXT)
            mock_ddb.Table.return_value = mock_table

            result = handler.lambda_handler(S3_EVENT_FACTORY(), {})

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert len(body) == 1
        assert body[0]["risk_level"] == "LOW"
        assert body[0]["risk_score"] < 0.35

        # Verify DynamoDB put_item was called once with the correct risk_level
        mock_table.put_item.assert_called_once()
        stored_item = mock_table.put_item.call_args[1]["Item"]
        assert stored_item["risk_level"] == "LOW"
        assert stored_item["s3_bucket"] == "test-ingestion-bucket"
        assert stored_item["s3_key"] == "doc.txt"
        assert isinstance(stored_item["risk_score"], Decimal)

    def test_high_risk_document_triggers_bedrock(self):
        mock_table = MagicMock()
        bedrock_response = {
            "verdict": "HIGH_RISK",
            "regulatory_framework": ["BSA", "AML"],
            "violations": [
                {
                    "type": "Money Laundering",
                    "description": "Funds moved through shell company",
                    "severity": "CRITICAL",
                    "excerpt": "shell company",
                }
            ],
            "recommended_actions": ["File SAR immediately", "Freeze account"],
            "summary": "Multiple critical AML violations detected.",
        }

        with (
            patch("handler.s3_client") as mock_s3,
            patch("handler.dynamodb") as mock_ddb,
            patch("handler._risk_engine.assess", return_value=(0.72, ["financial_fraud", "concealment_indicators"])),
            patch("handler.invoke_bedrock_agent", return_value=bedrock_response) as mock_bedrock,
        ):
            mock_s3.get_object.return_value = _s3_body_mock(HIGH_RISK_TEXT)
            mock_ddb.Table.return_value = mock_table

            result = handler.lambda_handler(S3_EVENT_FACTORY(), {})

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body[0]["risk_level"] == "HIGH"

        # Bedrock must have been called exactly once
        mock_bedrock.assert_called_once()
        call_text, call_terms = mock_bedrock.call_args[0]
        assert isinstance(call_text, str)
        assert isinstance(call_terms, list)

        # DynamoDB item must record HIGH risk
        stored_item = mock_table.put_item.call_args[1]["Item"]
        assert stored_item["risk_level"] == "HIGH"
        audit = json.loads(stored_item["audit_summary"])
        assert audit["verdict"] == "HIGH_RISK"

    def test_bedrock_not_called_for_low_risk(self):
        mock_table = MagicMock()

        with (
            patch("handler.s3_client") as mock_s3,
            patch("handler.dynamodb") as mock_ddb,
            patch("handler._risk_engine.assess", return_value=(0.10, [])),
            patch("handler.invoke_bedrock_agent") as mock_bedrock,
        ):
            mock_s3.get_object.return_value = _s3_body_mock(LOW_RISK_TEXT)
            mock_ddb.Table.return_value = mock_table

            handler.lambda_handler(S3_EVENT_FACTORY(), {})

        mock_bedrock.assert_not_called()

    def test_multiple_records_processed(self):
        mock_table = MagicMock()
        multi_event = {
            "Records": [
                {"s3": {"bucket": {"name": "bucket"}, "object": {"key": "a.txt"}}},
                {"s3": {"bucket": {"name": "bucket"}, "object": {"key": "b.txt"}}},
            ]
        }

        with (
            patch("handler.s3_client") as mock_s3,
            patch("handler.dynamodb") as mock_ddb,
            patch("handler._risk_engine.assess", return_value=(0.10, [])),
            patch("handler.invoke_bedrock_agent", return_value={"verdict": "HIGH_RISK"}),
        ):
            mock_s3.get_object.return_value = _s3_body_mock(LOW_RISK_TEXT)
            mock_ddb.Table.return_value = mock_table

            result = handler.lambda_handler(multi_event, {})

        body = json.loads(result["body"])
        assert len(body) == 2
        assert mock_table.put_item.call_count == 2

    def test_handler_returns_200_on_s3_error(self):
        """A per-document error should not crash the entire invocation."""
        mock_table = MagicMock()

        with (
            patch("handler.s3_client") as mock_s3,
            patch("handler.dynamodb") as mock_ddb,
        ):
            mock_s3.get_object.side_effect = Exception("S3 object not found")
            mock_ddb.Table.return_value = mock_table

            result = handler.lambda_handler(S3_EVENT_FACTORY(), {})

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "error" in body[0]

    def test_document_id_is_unique_per_invocation(self):
        mock_table = MagicMock()

        with (
            patch("handler.s3_client") as mock_s3,
            patch("handler.dynamodb") as mock_ddb,
            patch("handler._risk_engine.assess", return_value=(0.10, [])),
        ):
            mock_s3.get_object.return_value = _s3_body_mock(LOW_RISK_TEXT)
            mock_ddb.Table.return_value = mock_table

            result1 = handler.lambda_handler(S3_EVENT_FACTORY("x.txt"), {})
            result2 = handler.lambda_handler(S3_EVENT_FACTORY("y.txt"), {})

        id1 = json.loads(result1["body"])[0]["document_id"]
        id2 = json.loads(result2["body"])[0]["document_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# invoke_bedrock_agent unit tests (mock boto3 bedrock client directly)
# ---------------------------------------------------------------------------
class TestInvokeBedrockAgent:
    def _make_bedrock_response(self, json_text: str):
        body_mock = MagicMock()
        body_mock.read.return_value = json.dumps(
            {"content": [{"text": json_text}]}
        ).encode()
        return {"body": body_mock}

    def test_returns_parsed_json_on_clean_response(self):
        payload = json.dumps(
            {
                "verdict": "HIGH_RISK",
                "regulatory_framework": ["GDPR"],
                "violations": [],
                "recommended_actions": [],
                "summary": "Test summary.",
            }
        )

        with patch("handler.bedrock_client") as mock_client:
            mock_client.invoke_model.return_value = self._make_bedrock_response(payload)
            result = handler.invoke_bedrock_agent("some text", ["fraud"])

        assert result["verdict"] == "HIGH_RISK"

    def test_returns_error_dict_on_malformed_json(self):
        with patch("handler.bedrock_client") as mock_client:
            mock_client.invoke_model.return_value = self._make_bedrock_response(
                "Sorry, I cannot produce JSON right now."
            )
            result = handler.invoke_bedrock_agent("some text", [])

        assert result["verdict"] == "PARSE_ERROR"

    def test_json_extracted_from_prose_wrapper(self):
        """Bedrock sometimes wraps JSON in explanatory prose."""
        inner = json.dumps({"verdict": "MEDIUM_RISK", "regulatory_framework": []})
        prose_wrapped = f"Here is the analysis:\n{inner}\nEnd of analysis."

        with patch("handler.bedrock_client") as mock_client:
            mock_client.invoke_model.return_value = self._make_bedrock_response(prose_wrapped)
            result = handler.invoke_bedrock_agent("some text", [])

        assert result["verdict"] == "MEDIUM_RISK"
