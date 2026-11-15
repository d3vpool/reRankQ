#!/usr/bin/env python
"""
Local relevance-judgment interface (Streamlit).

Shows one query at a time with its pooled candidate chunks, lets a judge assign
a graded relevance label (0-3), and appends every judgment to
``data/eval/judgments.jsonl``. Retrieval systems and ranks are not shown, and
the source passage used to generate a query is never displayed, to avoid
ranking and provenance bias.

Run:
    streamlit run scripts/judge_relevance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.evaluation.eval_dataset import RELEVANCE_GRADES, load_jsonl  # noqa: E402
from src.evaluation.judgments import JudgmentStore, next_unjudged_query  # noqa: E402

EVAL_DIR = Path("data/eval")
GRADE_OPTIONS = [0, 1, 2, 3]

st.set_page_config(page_title="reRankQ relevance judgments", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(eval_dir: str):
    base = Path(eval_dir)
    return (
        load_jsonl(base / "queries.jsonl"),
        load_jsonl(base / "candidates.jsonl"),
    )


def main() -> None:
    st.title("reRankQ — relevance judgments")

    if not (EVAL_DIR / "queries.jsonl").exists():
        st.error(
            "No dataset found. Run `python scripts/build_eval_dataset.py` first."
        )
        return

    queries, candidates = load_dataset(str(EVAL_DIR))
    if not queries or not candidates:
        st.error("Dataset is empty. Rebuild it with scripts/build_eval_dataset.py.")
        return

    store = JudgmentStore(EVAL_DIR / "judgments.jsonl")
    queries_by_id = {query["query_id"]: query for query in queries}
    progress = store.progress(candidates)

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Session")
        judge_id = st.text_input("Judge ID", value="judge-1")
        strategy = st.selectbox(
            "Chunking strategy", sorted({c["chunking_strategy"] for c in candidates})
        )
        split = st.selectbox("Split", ["all", "dev", "test"])

        st.divider()
        st.header("Progress")
        scoped = [c for c in candidates if c["chunking_strategy"] == strategy]
        scoped_progress = store.progress(scoped)
        st.metric(
            "Judged",
            f"{scoped_progress['judged']} / {scoped_progress['total']}",
            f"{scoped_progress['percent_judged']:.1f}%",
        )
        st.caption(f"{scoped_progress['pending']} candidates still pending")
        st.caption(f"Whole dataset: {progress['judged']} / {progress['total']} judged")

        st.divider()
        st.header("Export")
        if st.button("Export judgments (JSONL + CSV)"):
            out_dir = EVAL_DIR / "exports"
            jsonl_path = store.export_jsonl(out_dir / "judgments.jsonl")
            csv_path = store.export_csv(out_dir / "judgments.csv")
            st.success(f"Wrote {jsonl_path} and {csv_path}")

        st.caption(
            "Judgments are appended to data/eval/judgments.jsonl. "
            "Re-recording a candidate supersedes the earlier grade; nothing is deleted."
        )

    # ------------------------------------------------------------- query pick
    st.subheader("Query")

    if "selected_query" not in st.session_state:
        st.session_state.selected_query = next_unjudged_query(
            queries, candidates, store, strategy=strategy
        ) or queries[0]["query_id"]

    query_ids = [q["query_id"] for q in queries]
    if st.session_state.selected_query not in queries_by_id:
        st.session_state.selected_query = query_ids[0]

    selected_index = query_ids.index(st.session_state.selected_query)
    selected_id = st.selectbox(
        "Select query",
        options=query_ids,
        index=selected_index,
        format_func=lambda qid: (
            f"{qid} — {queries_by_id[qid]['query']}"
            f" ({queries_by_id[qid]['split']})"
        ),
    )
    st.session_state.selected_query = selected_id
    query = queries_by_id[selected_id]

    st.markdown(f"### {query['query']}")
    st.caption(
        f"query {query['query_id']} · split: {query['split']} · "
        f"status: {query['validation_status']}"
    )

    query_candidates = [
        c
        for c in candidates
        if c["query_id"] == selected_id and c["chunking_strategy"] == strategy
    ]
    if split != "all" and query["split"] != split:
        st.info(f"This query belongs to the {query['split']} split; switch the split filter to judge it.")
    if not query_candidates:
        st.warning("No pooled candidates for this query and strategy.")
        return

    judged_ids = store.judged_candidate_ids()
    remaining = sum(1 for c in query_candidates if c["candidate_id"] not in judged_ids)
    st.caption(f"{len(query_candidates)} candidates · {remaining} pending")

    st.info(
        "Grade only from the chunk text itself. 0 = not relevant, 1 = marginal, "
        "2 = relevant, 3 = highly relevant. Pending is not the same as 0."
    )

    for position, candidate in enumerate(query_candidates, start=1):
        existing = store.get(candidate["candidate_id"])
        default_grade = existing["relevance_grade"] if existing else None
        default_rationale = existing.get("rationale", "") if existing else ""

        label = f"Candidate {position} · {candidate['candidate_id']}"
        if existing and existing.get("judgment_status") == "judged":
            label += f" · judged {existing['relevance_grade']}"

        with st.expander(label, expanded=existing is None and position == 1):
            if candidate.get("heading_context"):
                st.caption(f"Section: {candidate['heading_context']}")
            elif candidate.get("heading_path"):
                st.caption(f"Section: {' > '.join(candidate['heading_path'])}")
            if candidate.get("url"):
                st.caption(f"Source: {candidate['url']}")

            st.text_area(
                "Chunk text",
                value=candidate["content"],
                height=220,
                disabled=True,
                key=f"chunk_{candidate['candidate_id']}",
            )

            grade = st.radio(
                "Relevance grade",
                options=GRADE_OPTIONS,
                index=GRADE_OPTIONS.index(default_grade) if default_grade is not None else None,
                horizontal=True,
                format_func=lambda g: f"{g} — {RELEVANCE_GRADES[g].split('—')[0].strip()}",
                key=f"grade_{candidate['candidate_id']}",
            )
            rationale = st.text_input(
                "Rationale (optional)",
                value=default_rationale,
                key=f"rationale_{candidate['candidate_id']}",
            )

            if st.button("Save judgment", key=f"save_{candidate['candidate_id']}"):
                if grade is None:
                    st.error("Pick a grade before saving.")
                elif not judge_id.strip():
                    st.error("Set a judge ID in the sidebar.")
                else:
                    store.record(
                        query_id=query["query_id"],
                        candidate_id=candidate["candidate_id"],
                        chunk_id=candidate["chunk_id"],
                        doc_id=candidate["doc_id"],
                        chunking_strategy=candidate["chunking_strategy"],
                        relevance_grade=int(grade),
                        judge_id=judge_id,
                        rationale=rationale,
                        judgment_status="judged",
                    )
                    st.success("Saved.")
                    st.rerun()

    st.divider()
    if st.button("Next pending query"):
        pending_id = next_unjudged_query(
            queries,
            candidates,
            store,
            strategy=strategy,
            split=None if split == "all" else split,
        )
        if pending_id is None:
            st.success("No pending candidates left for this filter.")
        else:
            st.session_state.selected_query = pending_id
            st.rerun()


main()
