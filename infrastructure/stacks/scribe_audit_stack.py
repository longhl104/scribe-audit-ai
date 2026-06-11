"""
ScribeAudit AI — CDK Stack Definition
======================================
Provisions:
  - S3 bucket         (document ingestion)
  - Lambda function   (two-layer AI pipeline)
  - DynamoDB table    (audit results store)
  - IAM permissions   (least-privilege)
"""

import os

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
)
from constructs import Construct

# Path to the Lambda source directory (two levels up from this file)
_LAMBDA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "lambda"
)


class ScribeAuditStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # DynamoDB — audit results store
        # Free tier: 25 GB storage, 25 RCU/WCU provisioned (PAY_PER_REQUEST
        # is even cheaper for sporadic workloads).
        # ------------------------------------------------------------------
        audit_table = dynamodb.Table(
            self,
            "AuditResultsTable",
            table_name="scribe-audit-results",
            partition_key=dynamodb.Attribute(
                name="document_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.RETAIN,  # keep data on stack destroy
        )

        # GSI — query by risk_level (e.g. fetch all HIGH-RISK documents)
        audit_table.add_global_secondary_index(
            index_name="RiskLevelIndex",
            partition_key=dynamodb.Attribute(
                name="risk_level", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # ------------------------------------------------------------------
        # Lambda — two-layer AI pipeline
        # ------------------------------------------------------------------
        audit_fn = lambda_.Function(
            self,
            "ScribeAuditFunction",
            function_name="scribe-audit-processor",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(_LAMBDA_DIR),
            timeout=Duration.seconds(60),
            memory_size=256,
            environment={
                "DYNAMODB_TABLE": audit_table.table_name,
                "RISK_THRESHOLD": "0.1",
                "BEDROCK_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
                "MAX_BEDROCK_CHARS": "4000",
                "EMBEDDING_MODEL_ID": "cohere.embed-english-v3",
                "REFERENCE_CACHE_BUCKET": f"scribe-audit-ingestion-{self.account}",
            },
            description="ScribeAudit AI: hybrid NLP + Bedrock compliance risk processor",
        )

        # ------------------------------------------------------------------
        # S3 — document ingestion bucket
        # ------------------------------------------------------------------
        ingestion_bucket = s3.Bucket(
            self,
            "DocumentIngestionBucket",
            # Append account ID to guarantee global uniqueness
            bucket_name=f"scribe-audit-ingestion-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,  # clean teardown in dev/staging
        )

        # Trigger Lambda on .txt and .md uploads
        for suffix in (".txt", ".md"):
            ingestion_bucket.add_event_notification(
                s3.EventType.OBJECT_CREATED,
                s3n.LambdaDestination(audit_fn), # type: ignore
                s3.NotificationKeyFilter(suffix=suffix),
            )

        # ------------------------------------------------------------------
        # IAM — least-privilege permissions
        # ------------------------------------------------------------------
        # S3: read uploaded documents + write reference embedding cache
        ingestion_bucket.grant_read_write(audit_fn)

        # DynamoDB: write audit results only
        audit_table.grant_write_data(audit_fn)

        # Bedrock: invoke the configured model (scoped to inference profiles)
        audit_fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="AllowBedrockInference",
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                    f"arn:aws:bedrock:{self.region}::foundation-model/meta.llama3-8b-instruct-v1:0",
                    f"arn:aws:bedrock:{self.region}::foundation-model/cohere.embed-english-v3",
                ],
            )
        )

        # ------------------------------------------------------------------
        # CloudFormation outputs
        # ------------------------------------------------------------------
        cdk.CfnOutput(
            self,
            "IngestionBucketName",
            value=ingestion_bucket.bucket_name,
            description="Drop .txt or .md documents here to trigger the pipeline",
        )
        cdk.CfnOutput(
            self,
            "AuditTableName",
            value=audit_table.table_name,
            description="DynamoDB table storing structured audit results",
        )
        cdk.CfnOutput(
            self,
            "LambdaFunctionName",
            value=audit_fn.function_name,
            description="Lambda processor function name",
        )
