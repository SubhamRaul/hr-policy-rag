import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HR Policy Assistant", page_icon="📚")

st.title("📚 Internal HR Policy Assistant")
st.caption("Answers are generated only from policies indexed by the backend.")

with st.sidebar:
    st.header("Admin")
    uploaded = st.file_uploader(
        "Upload a policy",
        type=["md", "txt", "pdf"],
        help="Authentication is intentionally out of scope for this prototype.",
    )

    if uploaded is not None and st.button("Index policy"):
        try:
            response = requests.post(
                f"{API_URL}/admin/upload",
                files={
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                },
                timeout=120,
            )
            if response.ok:
                payload = response.json()
                st.success(
                    f"Indexed {payload['chunks_indexed']} chunks from "
                    f"{payload['document']}."
                )
            else:
                st.error(response.json().get("detail", response.text))
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")

question = st.text_input(
    "Ask an HR policy question",
    placeholder="e.g. How many casual leave days can I carry forward?",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=120,
            )

            if not response.ok:
                st.error(response.text)
            else:
                payload = response.json()

                if payload["refused"]:
                    st.warning(payload["answer"])
                    st.caption(payload.get("reason", ""))
                else:
                    st.success(payload["answer"])
                    st.subheader("Citations")
                    for citation in payload["citations"]:
                        st.markdown(
                            f"- **{citation['document']}** — "
                            f"section: **{citation['section']}**"
                        )
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")
