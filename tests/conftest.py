import os
import sys

# Make the lambda package importable from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda"))

# Provide dummy AWS credentials so boto3 does not raise NoCredentialsError
# when tests instantiate clients (moto intercepts the actual calls).
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("DYNAMODB_TABLE", "test-audit-table")
os.environ.setdefault("RISK_THRESHOLD", "0.1")
os.environ.setdefault("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
os.environ.setdefault("EMBEDDING_MODEL_ID", "cohere.embed-english-v3")
