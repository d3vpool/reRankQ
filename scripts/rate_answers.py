#!/usr/bin/env python
"""
Human rating of generated RAG answers.

Separate from query review and chunk judging. Scores are stored as human
judgments in data/eval/answer_ratings.jsonl; no LLM score can enter this file.

Run:
    streamlit run scripts/rate_answers.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.evaluation.eval_dataset import load_jsonl  # noqa: E402
from src.evaluation.rag_eval import HumanAnswerEvalStore  # noqa: E402

EVAL_DIR = Path("data/eval")
SUMMARY_FIELDS = ("faithfulness", "answer_relevance", "citation_correctness", "completeness")

st.set_page_config(page_title="reRankQ answer rating", layout="wide")


@st.cache_data(show_spinner=False)
def load_answers(directory: str) -> list[dict]:
    return load_jsonl(Path(directory) / "answers.jsonl")


def main() -> None:
    st.title("reRankQ — answer rating")
    st.caption(
        "Rate generated answers against the exact retrieved context shown. "
        "These are human scores; nothing here is model-generated."
    )

    runs = sorted(Path("results/generation").glob("*/answers.jsonl"))
    if not runs:
        st.info(
            "No generated answers found. Run scripts/run_generation_eval.py once "
            "queries are approved and a provider is configured."
        )
        return

    run_path = st.sidebar.selectbox(
        "Generation run", options=runs, format_func=lambda path: path.parent.name
    )
    answers = load_answers(str(run_path.parent))
    if not answers:
        st.warning("This run has no answer records.")
        return

    store = HumanAnswerEvalStore(EVAL_DIR / "answer_ratings.jsonl")
    rated = store.latest()

    with st.sidebar:
        reviewer = st.text_input("Reviewer ID", value="reviewer-1")
        st.metric("Rated", f"{len(rated)} / {len(answers)}")

    options = [answer["answer_id"] for answer in answers]
    selected = st.selectbox("Answer", options=options)
    answer = next(item for item in answers if item["answer_id"] == selected)

    st.markdown(f"**Question:** {answer['query']}")
    if not answer.get("answer"):
        st.warning(
            f"No answer was generated for this record: {answer.get('generation_status')} "
            f"({answer.get('generation_reason', 'no reason recorded')})"
        )
        return

    st.markdown("**Answer**")
    st.write(answer["answer"])

    st.markdown("**Retrieved context used**")
    for chunk in answer["retrieved_context"]:
        with st.expander(f"[{chunk['rank']}] {chunk.get('doc_id')} · {chunk.get('url', '')}"):
            st.write(chunk.get("content") or "")

    existing = rated.get(answer["answer_id"], {})
    scores = {}
    for field in SUMMARY_FIELDS:
        scores[field] = st.radio(
            field.replace("_", " ").title(),
            options=[1, 2, 3, 4, 5],
            index=(existing.get("scores", {}).get(field, 3) - 1),
            horizontal=True,
            key=f"{field}_{answer['answer_id']}",
        )
    note = st.text_area("Note", value="", height=80)

    if st.button("Save rating", disabled=not reviewer.strip()):
        try:
            store.record(
                answer_id=answer["answer_id"],
                reviewer_id=reviewer,
                scores={field: int(value) for field, value in scores.items()},
                note=note,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("Saved.")
            st.rerun()


main()
