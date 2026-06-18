"""
main.py — Bio-Risk Analysis Agent entry point.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  Azure AI Foundry Chat UI                                   │
  │  User uploads CSV + sends message                           │
  └─────────────────────────────┬───────────────────────────────┘
                                │ HTTP (agentserver)
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  THIS CONTAINER  (bio-risk-agent)                           │
  │                                                             │
  │  ChatAgent ──► score_session_csv(filename)      ──► [AI]   │
  │              (chains prep_session_csv + run_bio_inference) │
  │           ──► natural-language analysis to user            │
  └─────────────────────────────────────────────────────────────┘

Environment variables required (see .env.template):
  FOUNDRY_PROJECT_ENDPOINT          — Foundry project URL
  FOUNDRY_MODEL_DEPLOYMENT_NAME     — e.g. gpt-4o
    ML_ENDPOINT_URL                   — Azure ML scoring URI
    ML_AUTH_MODE                      — Use "aad" to authenticate with Azure AD
    ML_AAD_SCOPE                      — Optional token scope (default: https://ml.azure.com/.default)
"""

import argparse
import asyncio
import json
import logging
import os

from agent_framework import Agent
from agent_framework.exceptions import ChatClientException
from agent_framework.foundry import FoundryChatClient
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load .env FIRST; allow Foundry runtime env vars to override (override=False)
load_dotenv(override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a cardiovascular risk analysis assistant integrated with Azure ML.
The deployed model is an XGBoost classifier that predicts each patient's
10-year risk of coronary heart disease (CHD) from clinical features such as
age, gender, blood pressure, cholesterol, smoking status, BMI, glucose, and
diabetes/hypertension/stroke history.

When the user uploads a CSV file and asks you to analyse it:

1. Call score_session_csv (pass an empty string for `filename`, or the exact
   filename if the user mentions it). This single tool prepares the CSV and
   scores it against the Azure ML endpoint in one step. Do NOT call any other
   tools for scoring. Do NOT try to construct or pass feature data yourself.

2. If the tool returns an object with an "error" key, report the error verbatim
   to the user and STOP. Do not retry, do not guess, do not fabricate results.

3. Otherwise, present the results using ONLY the fields the tool returned:
   - State `row_count` as the number of patients/samples scored.
   - Report `summary.high_risk_count`, `summary.low_risk_count`,
     `summary.high_risk_pct` (interpreted as predicted 10-year CHD risk).
   - List each index in `top_high_risk_rows` together with its high-risk
     probability `probabilities[i][1]`, rounded to 4 decimals.
   - Mention any `warnings` from the tool.
   - Add a brief clinical-interpretation reminder that this is a model
     prediction and should be reviewed by a qualified clinician.

Never invent numbers — every figure must come from the tool's response.
Be concise, structured, and use plain English with bullet points where helpful.
""".strip()


def build_agent(project_client: AIProjectClient) -> Agent:
    """
    Construct the ChatAgent with its two tools.

    The tools are plain async callables — agent_framework inspects their type
    annotations and docstrings to build the function schemas automatically.
    """
    from agent_framework.foundry import FoundryChatClient
    from azure.identity import DefaultAzureCredential

    # Import tools — pass project_client into data_prep via closure
    from tools.ml_inference import score_session_csv

    credential = DefaultAzureCredential()
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=(
            os.environ.get("MODEL_DEPLOYMENT_NAME")
            or os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME")
            or os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"]
        ),
        credential=credential,
    )

    agent = Agent(
        client=client,
        name="BioRiskAgent",
        instructions=SYSTEM_PROMPT,
        tools=[score_session_csv],
    )
    return agent


def run_http_server() -> None:
    """Start the agent as a hosted HTTP server (Foundry Responses protocol)."""
    try:
        from agent_framework_foundry_hosting import ResponsesHostServer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "HTTP server mode requires agent-framework + agent-framework-foundry-hosting. "
            "Use '--cli' for local testing, or install the hosted-agent dependencies."
        ) from exc

    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=credential,
    )
    agent = build_agent(project_client)
    # History is managed by the Foundry hosting infrastructure.
    # Merge into existing default_options (which contains tools + tool_choice).
    try:
        existing = dict(agent.default_options or {})
        existing["store"] = False
        agent.default_options = existing
    except Exception:
        pass

    try:
        opts = agent.default_options or {}
        tool_objs = opts.get("tools", []) or []
        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", repr(t)) for t in tool_objs]
        logger.info("Agent registered tools: %s", tool_names)
        logger.info("Agent default_options keys: %s", list(opts.keys()))
    except Exception as exc:
        logger.warning("could not introspect agent tools: %s", exc)

    logger.info("Starting Bio-Risk Agent HTTP server (Responses protocol)…")
    ResponsesHostServer(agent).run()


async def run_cli() -> None:
    """Interactive CLI mode for local testing without the HTTP server."""
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=credential,
    )

    agent = build_agent(project_client)
    from tools.data_prep import prep_local_csv
    from tools.ml_inference import run_bio_inference

    print("Bio-Risk Agent — CLI mode. Type 'exit' to quit.")
    print("Use 'load <csv-path>' to score a local CSV file.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        if user_input.lower().startswith("load "):
            csv_path = user_input[5:].strip().strip('"')
            prep_result = prep_local_csv(csv_path)
            prep_json = json.loads(prep_result)

            if "error" in prep_json:
                print(f"\nAgent: {prep_json['error']}\n")
                continue

            inference_result = run_bio_inference(prep_result)
            inference_json = json.loads(inference_result)
            if "error" in inference_json:
                print(f"\nAgent: {inference_json['error']}\n")
                continue

            top_rows = [
                {
                    "row_index": index,
                    "high_risk_probability": round(inference_json["probabilities"][index][1], 3),
                }
                for index in inference_json["top_high_risk_rows"]
            ]
            print("\nAgent: Local CSV scored successfully.")
            print(f"Rows scored: {inference_json['row_count']}")
            print(f"Summary: {json.dumps(inference_json['summary'])}")
            if prep_json["warnings"]:
                print(f"Warnings: {prep_json['warnings']}")
            print(f"Top high-risk rows: {json.dumps(top_rows)}\n")
            continue

        try:
            response = await agent.run(user_input)
            print(f"\nAgent: {response.text}\n")
        except ChatClientException as exc:
            msg = str(exc)
            if "DeploymentNotFound" in msg:
                print(
                    "\nAgent: Foundry model deployment was not found. "
                    "Check FOUNDRY_MODEL_DEPLOYMENT_NAME in .env and ensure the deployment exists in your "
                    "Azure AI Foundry resource.\n"
                )
                continue
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Bio-Risk Analysis Agent")
    parser.add_argument(
        "--server",
        action="store_true",
        default=False,
        help="Run as HTTP server",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        default=False,
        help="Run in interactive CLI mode for local testing",
    )
    args = parser.parse_args()

    if args.server:
        run_http_server()
    else:
        asyncio.run(run_cli())


if __name__ == "__main__":
    main()
