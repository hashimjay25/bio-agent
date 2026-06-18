"""
deploy.py — Deploy the registered cardiovascular CHD-risk model as an Azure ML
Managed Online Endpoint.

Prerequisites:
  pip install azure-ai-ml azure-identity python-dotenv

Usage:
  python deploy.py [--model-version VERSION]

After completion, the script prints:
    - Endpoint scoring URI  → set as ML_ENDPOINT_URL in .env
    - Auth mode reminder    → set ML_AUTH_MODE=aad in .env
"""

import argparse
import os
from pathlib import Path
from time import sleep

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Prefer workspace settings from .env over stale process/session variables.
load_dotenv(override=True)

SUBSCRIPTION_ID = os.environ["AZURE_ML_SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ["AZURE_ML_RESOURCE_GROUP"]
WORKSPACE_NAME = os.environ["AZURE_ML_WORKSPACE_NAME"]

ENDPOINT_NAME = "bio-risk-endpoint"
DEPLOYMENT_NAME = "blue"
MODEL_NAME = "bio-risk-classifier"


def main(model_version: str = "latest") -> None:
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # ── 1. Create or update the endpoint ──────────────────────────────────
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Cardiovascular 10-year CHD risk inference endpoint (XGBoost)",
        auth_mode="AADToken",
    )
    print(f"Creating/updating endpoint: {ENDPOINT_NAME} …")
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    poller.result()  # wait
    print("Endpoint ready.")

    # ── 2. Resolve model version ──────────────────────────────────────────
    if model_version == "latest":
        models = list(ml_client.models.list(name=MODEL_NAME))
        if not models:
            raise RuntimeError(
                f"No registered model named '{MODEL_NAME}' found. "
                "Run submit_training_job.py first."
            )
        # list() returns newest-first when sorted by version
        latest = sorted(models, key=lambda m: int(m.version), reverse=True)[0]
        model_ref = f"azureml:{MODEL_NAME}:{latest.version}"
        print(f"Using model version: {latest.version}")
    else:
        model_ref = f"azureml:{MODEL_NAME}:{model_version}"
        print(f"Using model version: {model_version}")

    # ── 3. Scoring environment ────────────────────────────────────────────
    scoring_env = Environment(
        name="bio-risk-score-env",
        conda_file=str(Path(__file__).parent / "conda.yaml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    # ── 4. Create deployment ──────────────────────────────────────────────
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model_ref,
        environment=scoring_env,
        code_configuration=CodeConfiguration(
            code=str(Path(__file__).parent),
            scoring_script="score.py",
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    print(f"Creating deployment '{DEPLOYMENT_NAME}' (may take 5–15 min) …")
    poller = ml_client.online_deployments.begin_create_or_update(deployment)
    poller.result()
    print("Deployment ready.")

    # ── 5. Route 100 % of traffic to the blue deployment ─────────────────
    endpoint_obj = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint_obj.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint_obj).result()

    # ── 6. Print connection details ───────────────────────────────────────
    endpoint_obj = ml_client.online_endpoints.get(ENDPOINT_NAME)
    print("\n" + "=" * 60)
    print("ENDPOINT READY — copy these into your .env file:")
    print(f"  ML_ENDPOINT_URL={endpoint_obj.scoring_uri}")
    print("  ML_AUTH_MODE=aad")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-version",
        default="latest",
        help="Registered model version to deploy (default: latest)",
    )
    args = parser.parse_args()
    main(model_version=args.model_version)
