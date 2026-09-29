from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import streamlit as st

from config import SYSTEMS
from ui.components import inject_styles, render_query_processing, render_results


BACKEND = os.environ.get("STEP21_BACKEND_URL", "http://127.0.0.1:8765").rstrip("/")


def api_get(path):
    with urllib.request.urlopen(BACKEND + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path, payload, timeout=1800):
    request = urllib.request.Request(
        BACKEND + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = json.loads(exc.read().decode("utf-8"))
        raise RuntimeError(detail.get("error", str(exc))) from exc
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


@st.cache_data(show_spinner=False)
def question_list():
    return api_get("/questions")["questions"]


def main():
    st.set_page_config(page_title="Qur’anic Passage Retrieval — Thesis Demonstration", page_icon="📖", layout="wide")
    inject_styles()
    st.markdown(
        """
        <div class="hero">
          <h1>Qur’anic Passage Retrieval — Thesis Demonstration</h1>
          <p><b>Saad Moh'd Saad Abu Morri</b><br>MSc Data Science · Ala-Too International University</p>
          <p>Interactive demonstration of the retrieval systems evaluated in the MSc thesis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Systems")
        st.markdown("**BM25** — Lexical word-based retrieval.")
        st.markdown("**Dense E5** — Pretrained multilingual semantic retrieval using embeddings.")
        st.markdown("**Fixed RALSR** — The thesis’s root-aware lexical-semantic retrieval framework.")
        st.markdown("**CrossEncoder** — A pretrained model that reranks RALSR candidates by reading the question and passage together.")
        st.divider()
        st.caption("This application is a demonstration interface for the retrieval systems evaluated in the thesis. It does not alter the thesis methodology or reported experimental results.")

    mode = st.radio(
        "Mode",
        ["QuranQA Benchmark Demonstration", "Free Arabic Question"],
        horizontal=True,
    )
    top_n = st.selectbox("Number of passages", [5, 10], index=0, format_func=lambda n: f"Top {n}")

    if mode == "QuranQA Benchmark Demonstration":
        st.markdown("<div class='frozen'><b>Frozen benchmark mode.</b> These are the same frozen retrieval rankings evaluated in the thesis.</div>", unsafe_allow_html=True)
        questions = question_list()
        labels = {f"{item['qid']} — {item['question']}": item for item in questions}
        selected_label = st.selectbox("Official QuranQA question", list(labels), index=0)
        selected = labels[selected_label]
        st.markdown(f"<div class='arabic-question'><b>QID {selected['qid']}</b> — {selected['question']}</div>", unsafe_allow_html=True)
        compare = st.checkbox("Compare all four systems", value=False)
        systems = list(SYSTEMS) if compare else [st.selectbox("Retrieval system", SYSTEMS)]
        if st.button("Show frozen ranking", type="primary"):
            for system in systems:
                with st.spinner(f"Loading frozen {system} ranking…"):
                    payload = api_post("/benchmark", {"qid": selected["qid"], "system": system, "top_n": top_n})
                st.subheader(system)
                if payload.get("unjudged_note"):
                    st.info(payload["unjudged_note"])
                render_results(payload["results"], system)
    else:
        st.markdown("<div class='live'><b>Live demonstration — not part of the frozen thesis evaluation.</b> No MAP, MRR, AP, accuracy, or official relevance is calculated.</div>", unsafe_allow_html=True)
        question = st.text_area("Arabic question", placeholder="اكتب سؤالاً باللغة العربية", height=100)
        system = st.selectbox("Retrieval system", SYSTEMS, key="free_system")
        if st.button("Retrieve Qur’anic passages", type="primary"):
            if not question.strip():
                st.error("Please enter an Arabic question.")
            elif not re.search(r"[\u0600-\u06FF]", question) or re.search(r"[A-Za-z]", question):
                st.error("This demonstration accepts Arabic questions only.")
            else:
                with st.spinner("Running the validated thesis pipeline…"):
                    payload = api_post("/free", {"question": question.strip(), "system": system, "top_n": top_n})
                st.markdown(f"<div class='arabic-question'>{question}</div>", unsafe_allow_html=True)
                if system in {"Fixed RALSR", "RALSR + CrossEncoder"}:
                    with st.expander("Query roots and Maqāyīs evidence", expanded=True):
                        render_query_processing(payload.get("query_processing"))
                    if payload.get("candidate_count") is not None:
                        st.caption(f"Root-sharing candidate passages: {payload['candidate_count']}")
                if payload.get("zero_candidate"):
                    st.warning(payload["zero_candidate_message"])
                    st.caption("Under the thesis method this structurally produces Passage_ID = -1; it is not a learned no-answer prediction.")
                else:
                    render_results(payload["results"], system)

    st.divider()
    st.caption("Benchmark results shown in the thesis are frozen experimental results. Free-query retrieval is an interactive demonstration and is not included in the thesis evaluation metrics.")


if __name__ == "__main__":
    main()
