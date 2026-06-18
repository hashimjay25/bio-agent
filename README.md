# Bio-Risk Agent

End-to-end **cardiovascular risk analysis** demo built on Microsoft Foundry,
Azure ML, and Streamlit.

A user uploads a CSV of patient records → a Foundry-hosted chat agent calls a
custom XGBoost model deployed as an Azure ML managed online endpoint → the
agent narrates the results in natural language.

> **Disclaimer:** Trained on the public
> [Cardio Vascular Dataset](https://www.kaggle.com/datasets/rahuljayakody/cardio-vascular-dataset)
> (Framingham-style, published on Kaggle by Rahul Jayakody) for demonstration
> only. Not a medical device. Predictions must not be used for clinical
> decisions.

---

## TL;DR

```powershell
git clone <your-fork-url> bio-agent
cd bio-agent
python -m venv .venv; . .\.venv\Scripts\Activate.ps1
pip install -r agent/requirements.txt -r streamlit_requirements.txt
copy .env.template .env                          # then fill in values
# Drop the Kaggle CSV into data/cardio vascular_dataset.csv, then:
python scripts/split_cardio_dataset.py           # creates train + inference splits
cd azure_ml; python submit_training_job.py; python deploy.py; cd ..
python scripts/deploy_agent.py --acr <your-acr>.azurecr.io
streamlit run streamlit_app.py
```

---

## Architecture

```
┌──────────────────┐       ┌──────────────────────────┐       ┌────────────────────────┐
│  Streamlit UI    │  ───► │  Foundry Hosted Agent    │  ───► │  Azure ML Endpoint     │
│  (CSV upload +   │       │  bio-risk-agent (Docker) │       │  bio-risk-endpoint     │
│   chat)          │       │  agent_framework + GPT   │       │  XGBoost classifier    │
└──────────────────┘       └──────────────────────────┘       └────────────────────────┘
        ▲                              ▲                                 ▲
        │                              │                                 │
        │ AAD token                    │ AAD token                       │ AAD token
        └──────────────────────────────┴─────────────────────────────────┘
```

| Component        | Tech                                                        | Code                                            |
| ---------------- | ----------------------------------------------------------- | ----------------------------------------------- |
| Frontend         | Streamlit                                                   | [streamlit_app.py](streamlit_app.py)               |
| Hosted agent     | `agent_framework` + Foundry Responses protocol, Docker    | [agent/](agent/)                                   |
| Agent deployer   | Builds + pushes image, registers Foundry agent version      | [scripts/deploy_agent.py](scripts/deploy_agent.py) |
| Model training   | XGBoost in scikit-learn pipeline, on Azure ML compute       | [azure_ml/train.py](azure_ml/train.py)             |
| Model serving    | Azure ML managed online endpoint, AAD auth                  | [azure_ml/score.py](azure_ml/score.py)             |
| Infra (optional) | Bicep /`azd` templates for Foundry project + dependencies | [infra/](infra/)                                   |

The model predicts the **10-year risk of coronary heart disease (`TenYearCHD`)**
from 15 features: gender, age, education, currentSmoker, cigsPerDay, BPMeds,
prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate,
glucose.

---

## Prerequisites

- Python 3.11
- An Azure subscription with permission to create:
  - Azure Machine Learning workspace + a compute cluster
  - Azure AI Foundry project (with hosted-agents enabled)
  - Azure Container Registry
- The Azure CLI logged in (`az login`)
- Docker (only needed to deploy the hosted agent)

---

## Quick start

### 1. Clone and set up a virtual environment

```powershell
git clone <your-fork-url> bio-agent
cd bio-agent
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r agent/requirements.txt
pip install -r streamlit_requirements.txt
```

### 2. Configure secrets

```powershell
copy .env.template .env
# then edit .env with your own values — see comments inside the template
```

`.env` is gitignored. Never commit it.

### 3. Get the dataset

The source dataset is **not** committed to this repo — download it from Kaggle
and drop it into `data/`:

1. Grab the
   [Cardio Vascular Dataset](https://www.kaggle.com/datasets/rahuljayakody/cardio-vascular-dataset)
   (Kaggle account required).
2. Place the file at `data/cardio vascular_dataset.csv` (filename includes the
   space — leave it as-is).
3. Generate the train/inference split:

```powershell
python scripts/split_cardio_dataset.py
```

This produces `data/cardio_train.csv` (≈8.5k rows, with labels) and
`data/cardio_inference.csv` (≈2.1k rows, no labels). All three CSVs are
gitignored so they never get pushed.

### 4. Train and deploy the model

```powershell
cd azure_ml
# 4a. Submit the training job, wait for it to finish, register the model.
python submit_training_job.py
# 4b. Create / update the managed online endpoint.
python deploy.py
```

Copy the resulting scoring URI into `ML_ENDPOINT_URL` in your `.env`.

### 5. Build and deploy the hosted agent

```powershell
python scripts/deploy_agent.py --acr <your-acr-name>.azurecr.io
```

This builds the Docker image (build context is the repo root), pushes it to
ACR, and registers a new version of the `bio-risk-agent` hosted agent in your
Foundry project.

### 6. Run the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

Upload a CSV (e.g. `data/cardio_inference.csv`) and ask the agent to analyse it.

---

## Local CLI mode (no hosting required)

For quick iteration without deploying the agent:

```powershell
cd agent
python main.py --cli
# then at the prompt:
# load ../data/cardio_inference.csv
```

The CLI calls the Azure ML endpoint directly using your `.env` config.

---

## Repo layout

```
.
├── agent/                       # Hosted-agent source (Foundry container)
│   ├── main.py                  # ChatAgent + HTTP server / CLI entry point
│   ├── tools/
│   │   ├── data_prep.py         # CSV ingestion, feature validation, encoding
│   │   └── ml_inference.py      # Calls Azure ML endpoint
│   └── requirements.txt
├── azure_ml/                    # Azure ML training + deployment scripts
│   ├── train.py                 # XGBoost training script
│   ├── score.py                 # Online endpoint scoring entry point
│   ├── submit_training_job.py
│   ├── deploy.py
│   └── conda.yaml
├── scripts/                     # One-shot helper scripts
│   ├── split_cardio_dataset.py  # Train/inference split helper
│   └── deploy_agent.py          # Build, push, register Foundry agent version
├── data/                        # (gitignored) Datasets — download from Kaggle
│   ├── cardio vascular_dataset.csv  # Source dataset (you provide)
│   ├── cardio_train.csv         # Generated by split_cardio_dataset.py
│   └── cardio_inference.csv     # Generated by split_cardio_dataset.py
├── Dockerfile                   # Hosted-agent image
├── streamlit_app.py             # Browser frontend
├── streamlit_requirements.txt
├── .env.template                # Copy to .env and fill in
├── LICENSE
└── README.md
```

---

## Environment variables

See [.env.template](.env.template) for the full list. Required for normal use:

| Variable                              | Purpose                                                 |
| ------------------------------------- | ------------------------------------------------------- |
| `FOUNDRY_PROJECT_ENDPOINT`          | Foundry project URL                                     |
| `FOUNDRY_MODEL_DEPLOYMENT_NAME`     | Chat model deployment (e.g.`gpt-4.1`)                 |
| `ML_ENDPOINT_URL`                   | Azure ML scoring URI from `azure_ml/deploy.py`        |
| `ML_AUTH_MODE`                      | `aad` (the only mode currently supported)             |
| `ML_AAD_SCOPE`                      | Optional, defaults to `https://ml.azure.com/.default` |
| `AZURE_ML_SUBSCRIPTION_ID`          | Used by training + deploy scripts                       |
| `AZURE_ML_RESOURCE_GROUP`           | "                                                       |
| `AZURE_ML_WORKSPACE_NAME`           | "                                                       |
| `AZURE_ML_COMPUTE_NAME`             | Compute cluster used to run the training job            |
| `FOUNDRY_AGENT_NAME` *(optional)* | Defaults to `bio-risk-agent`                          |

---

## Authentication

Everything authenticates with `DefaultAzureCredential`. Locally that means
`az login` is enough; in cloud environments it picks up the assigned managed
identity. No secrets or keys are stored in code.

---

## Dataset

The input file `data/cardio vascular_dataset.csv` is the
[Cardio Vascular Dataset](https://www.kaggle.com/datasets/rahuljayakody/cardio-vascular-dataset)
published on Kaggle by Rahul Jayakody. It is a Framingham Heart Study-style
table of cardiovascular risk factors and a `TenYearCHD` label. It is **not**
included in this repository — download it directly from Kaggle. Please consult
the Kaggle page for the dataset's own license and usage terms before
redistributing it.

---

## License

[MIT](LICENSE) — see the LICENSE file. Note that the MIT license covers the
code in this repository only; the dataset retains its original Kaggle
licensing terms (see the Dataset section above).
