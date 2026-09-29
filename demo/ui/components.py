from __future__ import annotations

import html

import streamlit as st


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8f5; }
        .block-container { max-width: 1180px; padding-top: 1.7rem; }
        .hero { border-right: 7px solid #1f6f4a; background: white; padding: 1.2rem 1.5rem;
                border-radius: 10px; box-shadow: 0 3px 14px rgba(0,0,0,.06); }
        .arabic-question { direction: rtl; text-align: right; font-size: 1.35rem; line-height: 2; }
        .result-card { background: white; border: 1px solid #dfe5df; border-radius: 10px;
                       padding: 1rem 1.2rem; margin: .7rem 0; }
        .passage { direction: rtl; text-align: right; font-size: 1.22rem; line-height: 2.05; }
        .meta { color: #4b5d52; font-size: .92rem; }
        .frozen { background: #e8f3ed; border: 1px solid #99c4aa; padding: .75rem 1rem; border-radius: 8px; }
        .live { background: #fff4df; border: 1px solid #e5b85c; padding: .75rem 1rem; border-radius: 8px; }
        [data-testid="stSidebar"] { background: #eef3ef; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_query_processing(payload):
    if not payload:
        return
    st.markdown("**Query processing**")
    st.write("Selected terms:", "، ".join(payload.get("selected_terms", [])) or "—")
    st.write("Accepted roots:", "، ".join(payload.get("accepted_roots", [])) or "—")
    st.write("Unresolved terms:", "، ".join(payload.get("unresolved_terms", [])) or "None")
    records = payload.get("term_records", [])
    if records:
        st.dataframe(
            [
                {
                    "Term": row.get("term"),
                    "Root": row.get("root") or "—",
                    "Source": row.get("source") or "Unresolved",
                    "Maqāyīs": "Found" if row.get("maqayis_found") else "Not found",
                }
                for row in records
            ],
            hide_index=True,
            use_container_width=True,
        )


def render_explanation(explanation):
    if not explanation:
        return
    shared = explanation.get("shared_roots", [])
    st.write("Shared roots:", "، ".join(shared) or "—")
    sources = explanation.get("query_root_sources", [])
    if sources:
        st.write(
            "Query roots:",
            "، ".join(
                f"{item.get('Query_Term')} → {item.get('Root_AR')} ({item.get('Root_Source')})"
                for item in sources if item.get("Root_AR")
            ) or "—",
        )
    features = explanation.get("features", {})
    if features:
        st.dataframe(
            [{"Feature": key.replace("_", " "), "Value": value} for key, value in features.items()],
            hide_index=True,
            use_container_width=True,
        )
    if explanation.get("ralsr_score") is not None:
        st.caption(f"Fixed RALSR score: {explanation['ralsr_score']:.6f}")


def render_results(results, system):
    for row in results:
        if row.get("structural_no_answer"):
            st.warning("Structural RALSR output: -1 (zero valid candidates). This is not a learned no-answer prediction.")
            continue
        pid = html.escape(str(row["passage_id"]))
        passage = html.escape(str(row["passage_text"]))
        score = float(row["score"])
        score_label = "CrossEncoder score" if system == "RALSR + CrossEncoder" else "Score"
        st.markdown(
            f"<div class='result-card'><div class='meta'><b>Rank {row['rank']}</b> &nbsp; | &nbsp; "
            f"Passage_ID: <b>{pid}</b> &nbsp; | &nbsp; {score_label}: <b>{score:.6f}</b></div>"
            f"<div class='passage'>{passage}</div></div>",
            unsafe_allow_html=True,
        )
        explanation = row.get("explanation")
        if explanation:
            with st.expander("Why was this passage retrieved?"):
                render_explanation(explanation)
        elif system == "Fixed RALSR" and row.get("features"):
            with st.expander("Why was this passage retrieved?"):
                render_explanation({"shared_roots": row.get("shared_roots", []), "features": row.get("features", {}), "ralsr_score": row.get("score")})
        elif system == "RALSR + CrossEncoder" and row.get("features"):
            with st.expander("RALSR candidate evidence and reranking"):
                render_explanation({"shared_roots": row.get("shared_roots", []), "features": row.get("features", {}), "ralsr_score": row.get("original_ralsr_score")})
                st.caption(f"Original fixed-RALSR rank: {row.get('original_ralsr_rank')}")
