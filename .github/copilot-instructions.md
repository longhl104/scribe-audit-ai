---
name: ScribeAudit AI Project Instructions
description: Instructions for the ScribeAudit AI project, a serverless compliance risk scoring pipeline.
applyTo: "**/*"
---

# ScribeAudit AI Project Overview

This project implements a lightweight, serverless AI pipeline for compliance risk scoring of business transcripts.

## Purpose
The primary goal is to ingest raw business transcripts, assess them for compliance risk, and generate structured audit reports for high-risk documents.

## Architecture
The system employs a two-layer AI strategy:
-   **Layer 1: Semantic Risk Engine**
    -   Technology: Amazon Titan Embeddings v2 (cosine similarity)
    -   Function: Scores every document for risk by comparing its embedding against pre-built reference embeddings for various risk categories.
    -   Cost-efficiency: Acts as a cheap pre-filter, processing all documents.

-   **Layer 2: Bedrock Compliance Agent**
    -   Technology: Amazon Bedrock (Claude 3 Haiku)
    -   Function: Invoked only for documents identified as high-risk by Layer 1. Generates a structured JSON audit report.
    -   Cost-efficiency: Only runs on documents that warrant deeper inspection, saving costs.

## Key Technologies
-   AWS Lambda (Python 3.14) for core processing.
-   Amazon S3 for document ingestion.
-   Amazon DynamoDB for storing structured audit results.
-   Amazon Bedrock for AI model inference (Titan Embeddings v2 and Claude 3 Haiku).
-   AWS CDK for infrastructure provisioning.

## Project Structure Highlights
-   `lambda/`: Contains the Lambda handler, risk engine logic, and dependencies.
-   `infrastructure/`: AWS CDK application for deploying the AWS resources.
-   `tests/`: Unit and integration tests.
-   `sample_docs/`: Sample high-risk and low-risk documents.

## Important Notes
-   The two-layer AI strategy is crucial for cost optimization, processing most documents cheaply and only using the more expensive LLM for high-risk cases.
-   Ensure that Bedrock model access is enabled for `Anthropic Claude 3 Haiku` and `Amazon Titan Embeddings v2` in your AWS account for the project to function correctly.
