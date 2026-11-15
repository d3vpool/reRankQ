#!/usr/bin/env python
"""
Local query-review interface (Streamlit).

Reviewing a query is separate from judging chunk relevance: this app never
shows candidate chunks, retrieval systems or ranks. A reviewer reads the query,
its provenance and the grounding passage, then accepts it, rejects it, marks it
as needing revision, and optionally edits the wording.

Every decision is appended to ``data/eval/query_reviews.jsonl`` with the
original wording and a revision history.

Run:
    streamlit run scripts/review_queries.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.evaluation.eval_dataset import CATEGORIES, load_jsonl  # noqa: E402
from src.evaluation.query_review import (  # noqa: E402
    POOL_BASELINE_FILENAME,
    ReviewStore,
    build_approved_dataset,
    candidate_pool_status,
    effective_queries,
    load_pool_baseline,
    next_unreviewed_query,
    review_progress,
    write_pool_baseline,
)

EVAL_DIR = Path("data/eval")
STATUS_LABELS = {
    "candidate": "unreviewed",
    "accepted": "accepted",
    "rejected": "rejected",
    "needs_revision": "needs revision",
}

st.set_page_config(page_title="reRankQ query review", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(eval_dir: str):
    base = Path(eval_dir)
    return load_jsonl(base / "queries.jsonl"), load_jsonl(base / "approved_queries.jsonl")


def main() -> None:
    st.title("reRankQ — query review")
    st.caption(
        "Query acceptance only. Relevance judgments happen in a separate app, "
        "and this screen does not show candidate pools or rankings."
    )

    if not (EVAL_DIR / "queries.jsonl").exists():
        st.error("No dataset found. Run `python scripts/build_eval_dataset.py` first.")
        return

    queries, _ = load_dataset(str(EVAL_DIR))
    if not queries:
        st.error("Dataset is empty. Rebuild it with scripts/build_eval_dataset.py.")
        return

    store = ReviewStore(EVAL_DIR / "query_reviews.jsonl")
    doc_titles = {}
    manifest_path = EVAL_DIR / "corpus_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        doc_titles = {doc["doc_id"]: doc["title"] for doc in manifest.get("documents", [])}

    baseline = None
    if (EVAL_DIR / POOL_BASELINE_FILENAME).exists():
        baseline = load_pool_baseline(EVAL_DIR / POOL_BASELINE_FILENAME)
    else:
        write_pool_baseline(queries, EVAL_DIR / POOL_BASELINE_FILENAME, source="existing pool build")
        baseline = load_pool_baseline(EVAL_DIR / POOL_BASELINE_FILENAME)

    merged = {query["query_id"]: query for query in effective_queries(queries, store)}
    progress = review_progress(queries, store)
    pool_status = candidate_pool_status(queries, baseline, store)

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Reviewer")
        reviewer_id = st.text_input("Reviewer ID", value="reviewer-1")

        st.divider()
        st.header("Progress")
        st.metric(
            "Reviewed",
            f"{progress['reviewed']} / {progress['total']}",
            f"{progress['percent_reviewed']:.0f}%",
        )
        st.caption(
            " · ".join(
                f"{STATUS_LABELS[status]}: {count}"
                for status, count in progress["by_status"].items()
            )
        )
        st.caption(f"Accepted so far: {progress['accepted']}")

        st.divider()
        st.header("Filter")
        category = st.selectbox("Category", ["all", *CATEGORIES])
        status_filter = st.selectbox(
            "Status", ["all", "candidate", "accepted", "rejected", "needs_revision"]
        )

        st.divider()
        if st.button("Export reviews + approved set (CSV/JSONL)"):
            out_dir = EVAL_DIR / "exports"
            store.export_jsonl(out_dir / "query_reviews.jsonl")
            store.export_csv(out_dir / "query_reviews.csv")
            approved, meta = build_approved_dataset(
                queries, store, pool_status=pool_status
            )
            (out_dir / "approved_queries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in approved)
            )
            st.success(
                f"Wrote reviews and {meta['num_queries']} approved queries to {out_dir}"
            )

    # ------------------------------------------------------------- query list
    visible = [
        query
        for query in queries
        if (category == "all" or query["category"] == category)
        and (
            status_filter == "all"
            or merged[query["query_id"]]["validation_status"] == status_filter
        )
    ]
    if not visible:
        st.info("No queries match the current filters.")
        return

    if "selected_query" not in st.session_state:
        pending = next_unreviewed_query(
            queries, store, category=None if category == "all" else category
        )
        st.session_state.selected_query = pending or visible[0]["query_id"]
    visible_ids = [query["query_id"] for query in visible]
    if st.session_state.selected_query not in visible_ids:
        st.session_state.selected_query = visible_ids[0]

    selected_id = st.selectbox(
        "Select query",
        options=visible_ids,
        index=visible_ids.index(st.session_state.selected_query),
        format_func=lambda qid: (
            f"{qid} · {STATUS_LABELS[merged[qid]['validation_status']]} · "
            f"{merged[qid]['query'][:90]}"
        ),
    )
    st.session_state.selected_query = selected_id

    query = next(item for item in queries if item["query_id"] == selected_id)
    review = store.get(selected_id)
    current = merged[selected_id]
    stale = next(
        (entry for entry in pool_status["entries"] if entry["query_id"] == selected_id),
        None,
    )

    badge = STATUS_LABELS[current["validation_status"]]
    st.markdown(f"### {current['query']}")
    st.caption(
        f"query {selected_id} · category: {current['category']} · status: {badge} · "
        f"split: {query.get('split')} · pool: {stale['status'] if stale else 'unknown'}"
    )

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Provenance**")
        docs = query.get("source_doc_ids") or []
        st.write(
            ", ".join(f"{doc} ({doc_titles.get(doc, 'unknown title')})" for doc in docs)
            or "— none recorded —"
        )
        st.caption(f"section: {query.get('source_section') or '—'}")
        st.caption(f"corpus: {query.get('corpus_version') or '—'}")
        st.markdown("**Grounding passage (review only)**")
        st.text_area(
            "Passage",
            value=query.get("source_passage") or "",
            height=200,
            disabled=True,
            key=f"passage_{selected_id}",
        )

        if review and review["revision_history"]:
            st.markdown("**Revision history**")
            for revision in review["revision_history"]:
                st.caption(
                    f"{revision['timestamp']} · {revision['reviewer_id']}: "
                    f"“{revision['from_query']}” → “{revision['to_query']}” "
                    f"({revision['from_category']} → {revision['to_category']})"
                )
        if stale and stale["status"] == "stale":
            st.warning(
                "The candidate pool was built from the earlier wording "
                f"“{stale['pool_query']}”. It is stale until the pool is rebuilt."
            )

    with right:
        st.markdown("**Review decision**")
        edited_query = st.text_area(
            "Query wording", value=current["query"], height=110, key=f"text_{selected_id}"
        )
        edited_category = st.selectbox(
            "Category",
            options=list(CATEGORIES),
            index=list(CATEGORIES).index(current["category"]),
            key=f"cat_{selected_id}",
        )
        note = st.text_area(
            "Review note (why this decision)", value="", height=90, key=f"note_{selected_id}"
        )

        if not reviewer_id.strip():
            st.error("Set a reviewer ID in the sidebar.")

        columns = st.columns(3)
        decisions = [
            ("Accept", "accepted"),
            ("Reject", "rejected"),
            ("Needs revision", "needs_revision"),
        ]
        for column, (label, status) in zip(columns, decisions):
            if column.button(label, key=f"{status}_{selected_id}", disabled=not reviewer_id.strip()):
                try:
                    store.record(
                        query=query,
                        review_status=status,
                        reviewer_id=reviewer_id,
                        review_note=note,
                        query_text=edited_query,
                        category=edited_category,
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success(f"Recorded: {STATUS_LABELS[status]}")
                    st.rerun()

    st.divider()
    if st.button("Next unreviewed query"):
        target = next_unreviewed_query(
            queries,
            store,
            category=None if category == "all" else category,
            only_status=None if status_filter == "all" else status_filter,
        )
        if target is None:
            st.success("No unreviewed queries match this filter.")
        else:
            st.session_state.selected_query = target
            st.rerun()


main()
