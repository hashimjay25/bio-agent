"""
deploy_agent.py — Build the Docker image, push to ACR, and register the
bio-risk agent as a Hosted Agent on Azure AI Foundry.

Prerequisites:
  - Docker Desktop running locally
  - az CLI logged in: az login
  - An Azure Container Registry linked to your Foundry project
  - Agent + model already running (run azure_ml/deploy.py first)

Usage:
  python deploy_agent.py --acr <acr-login-server> [--image-tag v1]

Example:
  python deploy_agent.py --acr myregistry.azurecr.io --image-tag v1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv

# Repo root holds the Dockerfile and the agent/ folder; build context must be there.
REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env", override=True)

AGENT_NAME = "bio-risk-agent"
IMAGE_NAME = "bio-risk-agent"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    # shell=True is required on Windows so .cmd wrappers like az.cmd are found
    result = subprocess.run(cmd, capture_output=False, check=check, shell=(sys.platform == "win32"))
    return result


def main(acr_server: str, image_tag: str) -> None:
    image_ref = f"{acr_server}/{IMAGE_NAME}:{image_tag}"

    # Always run docker against the repo root so the Dockerfile + agent/ are visible.
    os.chdir(REPO_ROOT)

    # ── 1. Build Docker image ──────────────────────────────────────────────
    print(f"\n[1/4] Building Docker image: {image_ref}")
    run(["docker", "build", "-t", image_ref, "."])

    # ── 2. Push to ACR ─────────────────────────────────────────────────────
    print(f"\n[2/4] Pushing image to ACR …")
    run(["az", "acr", "login", "--name", acr_server.split(".")[0]])
    run(["docker", "push", image_ref])

    # ── 3. Register Hosted Agent in Foundry ───────────────────────────────
    print("\n[3/4] Registering Hosted Agent in Azure AI Foundry …")

    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    ml_endpoint_url = os.environ["ML_ENDPOINT_URL"]
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(endpoint=endpoint, credential=credential, allow_preview=True)

    from azure.ai.projects.models import HostedAgentDefinition, ProtocolVersionRecord

    # Delete any existing agent with the same name (e.g. wrong kind from a previous attempt)
    try:
        project_client.agents.delete(AGENT_NAME)
        print(f"  Deleted existing agent '{AGENT_NAME}' to allow recreation as hosted type.")
    except Exception:
        pass  # Agent doesn't exist yet — that's fine

    definition = HostedAgentDefinition(
        cpu="1",
        memory="2Gi",
        image=image_ref,
        container_protocol_versions=[
            ProtocolVersionRecord(protocol="responses", version="1.0.0"),
        ],
        environment_variables={
            "ML_ENDPOINT_URL": ml_endpoint_url,
            "ML_AUTH_MODE": "aad",
            "ML_AAD_SCOPE": os.environ.get("ML_AAD_SCOPE", "https://ml.azure.com/.default"),
            # FOUNDRY_* and AGENT_* env vars are reserved; use a custom name for the model.
            "MODEL_DEPLOYMENT_NAME": os.environ.get("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-4.1"),
        },
    )

    # Debug: print the payload being sent
    try:
        import json
        from azure.ai.projects._utils.model_base import SdkJSONEncoder
        print("\n[DEBUG] HostedAgentDefinition payload:")
        print(json.dumps(definition, cls=SdkJSONEncoder, exclude_readonly=True, indent=2))
    except Exception as e:
        print(f"[DEBUG] Could not print payload: {e}")

    try:
        result = project_client.agents.create_version(
            agent_name=AGENT_NAME,
            definition=definition,
            description="Analyses clinical biomarker CSVs via Azure ML to predict patient disease risk.",
        )
        print(f"\n[4/4] Hosted Agent deployed successfully!")
        print(f"  Agent name : {AGENT_NAME}")
        print(f"  Version    : {getattr(result, 'version', 'N/A')}")
        print(f"  Status     : {getattr(result, 'status', 'N/A')}")
        print(
            "\nOpen Azure AI Foundry → Agents → Bio-Risk Analysis Agent → "
            "Playground to start chatting."
        )
    except Exception as exc:
        print(f"\n[4/4] Deployment failed: {exc}")
        if isinstance(exc, HttpResponseError):
            try:
                print("[DEBUG] Service response:")
                print(exc.response.text())
            except Exception:
                pass
        if "Unsupported region for Foundry Hosted Agents" in str(exc):
            print(
                "\nThis Azure AI Foundry project region does not currently support Hosted Agents. "
                "Create/select a Foundry project in a supported region, then rerun this script."
            )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Bio-Risk Agent to Azure AI Foundry")
    parser.add_argument(
        "--acr",
        required=True,
        help="ACR login server, e.g. myregistry.azurecr.io",
    )
    parser.add_argument(
        "--image-tag",
        default="latest",
        help="Docker image tag (default: latest)",
    )
    args = parser.parse_args()
    main(acr_server=args.acr, image_tag=args.image_tag)
