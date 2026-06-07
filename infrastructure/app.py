#!/usr/bin/env python3
"""AWS CDK app entry point for the ScribeAudit AI stack."""

import os

import aws_cdk as cdk

from stacks.scribe_audit_stack import ScribeAuditStack

app = cdk.App()

ScribeAuditStack(
    app,
    "ScribeAuditStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
    description="ScribeAudit AI — serverless document intelligence and compliance risk router",
)

cdk.Tags.of(app).add("Project", "ScribeAuditAI")
cdk.Tags.of(app).add("ManagedBy", "CDK")

app.synth()
