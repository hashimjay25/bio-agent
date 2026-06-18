"""
Streamlit frontend for the Foundry-hosted `bio-risk-agent`.

The agent itself remains hosted in Microsoft Foundry — this app just
provides a browser UI for uploading CSVs and chatting with it. All calls
go through the Foundry project endpoint using your Azure credentials.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

load_dotenv(override=True)

AGENT_NAME = os.environ.get("FOUNDRY_AGENT_NAME", "bio-risk-agent")
ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").rstrip("/")

st.set_page_config(page_title="Bio-Risk Agent", layout="wide")
st.title("Bio-Risk Agent")
st.caption(f"Hosted in Foundry · Agent: **{AGENT_NAME}**")

if not ENDPOINT:
    st.error("FOUNDRY_PROJECT_ENDPOINT is not set. Put it in a .env file or your environment.")
    st.stop()


# ---------------------------------------------------------------------------
# Cached Foundry client + credential
# ---------------------------------------------------------------------------
@st.cache_resource
def get_clients():
    cred = DefaultAzureCredential()
    proj = AIProjectClient(endpoint=ENDPOINT, credential=cred, allow_preview=True)
    oai = proj.get_openai_client(agent_name=AGENT_NAME, timeout=300)
    return cred, oai


cred, oai = get_clients()


def ensure_session() -> str:
    """Open a Foundry agent session if we don't have one yet, return its id."""
    if st.session_state.get("session_id"):
        return st.session_state["session_id"]
    intro = oai.responses.create(
        input="Session starting. The user may upload a CSV at any time."
    )
    st.session_state["session_id"] = intro.agent_session_id
    st.session_state["last_response_id"] = intro.id
    return intro.agent_session_id


def upload_csv(file_name: str, data: bytes) -> tuple[int, str]:
    """PUT a CSV into the active session sandbox."""
    sid = ensure_session()
    token = cred.get_token("https://ai.azure.com/.default").token
    url = (
        f"{ENDPOINT}/agents/{AGENT_NAME}/endpoint/sessions/{sid}/files/content"
        f"?api-version=v1&path={file_name}"
    )
    r = requests.put(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Foundry-Features": "HostedAgents=V1Preview",
            "Content-Type": "application/octet-stream",
        },
        data=data,
        timeout=120,
    )
    return r.status_code, r.text


def ask(prompt: str) -> str:
    """Send a follow-up turn bound to the current session."""
    sid = ensure_session()
    kwargs = {
        "input": prompt,
        "extra_body": {"agent_session_id": sid},
    }
    if st.session_state.get("last_response_id"):
        kwargs["previous_response_id"] = st.session_state["last_response_id"]
    reply = oai.responses.create(**kwargs)
    st.session_state["last_response_id"] = reply.id
    return reply.output_text or "(no text)"


# ---------------------------------------------------------------------------
# Initialise chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list[{role,content}]
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = []  # list[filename]


# ---------------------------------------------------------------------------
# Sidebar — file uploader + session controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload CSV")
    uploaded = st.file_uploader(
        "Pick a biomarker CSV", type=["csv"], accept_multiple_files=False
    )
    if uploaded is not None and uploaded.name not in st.session_state["uploaded"]:
        with st.spinner(f"Uploading {uploaded.name} to the session…"):
            code, body = upload_csv(uploaded.name, uploaded.getvalue())
        if code < 300:
            st.success(f"Uploaded {uploaded.name} ({len(uploaded.getvalue())} bytes)")
            st.session_state["uploaded"].append(uploaded.name)
            note = (
                f"_Uploaded `{uploaded.name}` to the session. "
                "Ask the agent to analyse it whenever you're ready._"
            )
            st.session_state["messages"].append({"role": "system", "content": note})
        else:
            st.error(f"Upload failed: HTTP {code}\n{body[:400]}")

    st.divider()
    st.subheader("Session")
    st.code(st.session_state.get("session_id") or "(not opened yet)", language=None)
    if st.session_state["uploaded"]:
        st.write("**Files in session:**")
        for n in st.session_state["uploaded"]:
            st.write(f"• `{n}`")
    if st.button("New session"):
        for k in ("session_id", "last_response_id", "messages", "uploaded"):
            st.session_state.pop(k, None)
        st.rerun()


# ---------------------------------------------------------------------------
# Chat history + input
# ---------------------------------------------------------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"] if msg["role"] != "system" else "assistant"):
        st.markdown(msg["content"])

placeholder = "Ask the agent to analyse the CSV you uploaded…"
if not st.session_state["uploaded"]:
    placeholder = "Say hello, or upload a CSV from the sidebar first."

if prompt := st.chat_input(placeholder):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = ask(prompt)
            except Exception as exc:
                answer = f"**Error from Foundry:** `{exc}`"
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
