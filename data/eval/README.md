# reRankQ Evaluation Dataset

This directory holds a **candidate** evaluation dataset, a human query-review
workflow, a human relevance-judgment workflow, and a frozen group-aware split.

Nothing here is human-validated yet: **0 of 50 queries have been reviewed**, so
`approved_queries.jsonl` is empty, and all 598 candidate chunks are still
`pending`. Query acceptance and relevance judgment are two separate human tasks.

## Current snapshot

| Item | Value |
|---|---|
| Corpus version | `corpus-fde4c9c42bf3` |
| Dataset version | `ds-0d0ddd1f3776` |
| Split method | `group-doc-section-neardup` |
| Split version | `split-f4a04d80` (seed 13) |
| Split | 35 dev / 15 test, 12 query groups |
| Documents | 30 (Stripe docs, seeded at `/payments/payment-intents`) |
| Chunking strategies | `fixed_size` (144 chunks), `structure_aware` (222 chunks) |
| Candidate queries | 50 |
| Review status | 50 candidate, 0 accepted, 0 rejected, 0 needs_revision |
| Approved queries | 0 — `approved_queries.jsonl` is empty |
| Pooled candidates | 598 (301 fixed-size, 297 structure-aware) |
| Judged candidates | 0 — all pending |
| Stale candidate pools | 0 |
| Embedding model | `all-MiniLM-L6-v2` (384-d) |
| Tokenizer | `tiktoken` `cl100k_base` |

## 1. Corpus snapshot methodology

`scripts/build_eval_dataset.py` writes `corpus_manifest.json` with one record
per document (`doc_id`, canonical URL, title, section, character count,
SHA-256 of the exact content), the chunking configuration, chunk counts and a
chunk-id hash per strategy, the embedding model, the tokenizer, and library
versions.

`corpus_version` is a SHA-256 over the sorted `doc_id + url + content-hash`
rows. Reordering documents does not change it; changing any byte does. The
builder refuses to overwrite a snapshot unless `--new-version` is passed, and
`validate_eval_dataset.py` re-verifies the raw corpus against the manifest.

Each index directory records `corpus_version` and its strategy in
`index_manifest.json`, so loading an index from another corpus or strategy
raises rather than returning foreign chunks. The raw corpus is never re-crawled
by any evaluation script.

## 2. Query-generation methodology

`generate_candidate_queries()` turns fixed blueprints
`(question, category, topic, keyword alternatives)` into queries only when the
frozen corpus actually contains one of the keywords. The grounding chunk
supplies `source_doc_ids`, `source_section` (heading breadcrumb),
`source_passage`, `source_chunk_id`; `generation_method` is recorded as
`rule_blueprint+corpus_grounding`. Query text is normalised and de-duplicated,
at most three queries per topic, and ungrounded blueprints are reported as
skipped. Every generated query starts as `validation_status: candidate`.

## 3. Query categories

terminology 11 · conceptual 10 · api_usage 10 · procedural 7 · troubleshooting 6
· contextual 6.

## 4. Candidate pooling methodology

Per query **and per strategy** (never mixed), top 3 from BM25, dense, hybrid
RRF, and hybrid RRF + cross-encoder rerank (rerank draws from the hybrid top 6).
Deduplicated by `(query, strategy, chunk)`; only chunks present in that
strategy's frozen chunk set are pooled.

Judge-facing `candidates.jsonl` carries only `candidate_id`, `query_id`,
`chunking_strategy`, `chunk_id`, `doc_id`, content, heading context, URL, title,
`corpus_version` — no system names, ranks or scores.
`candidate_provenance.jsonl` (analysis only) holds the contributing systems,
ranks, scores, `num_systems` and `best_rank`.

## 5. Query review and acceptance

`query_reviews.jsonl` is an append-only store of review decisions; the latest
record per `query_id` is authoritative. Nothing is deleted, so a decision can be
audited or reversed later.

Statuses:

| Status | Meaning |
|---|---|
| `candidate` | generated, not yet reviewed |
| `accepted` | reviewed and approved for evaluation |
| `rejected` | unsuitable for evaluation |
| `needs_revision` | potentially useful, must be edited before acceptance |

Approval criteria: the question must be answerable from the frozen corpus, be
well-posed and unambiguous, test one recognisable retrieval challenge, not copy
a heading, not contain its own answer, and be worth judging. A reviewer also
confirms that the grounding document and section are the right provenance.

Each record stores: `query_id`, `original_query`, `current_query`,
`original_category`, `current_category`, `source_doc_ids`, `source_section`,
`source_passage`, `source_chunk_id`, `corpus_version`, `review_status`,
`reviewer_id`, `review_timestamp`, `review_note`, `revision_history`.

Editing never overwrites the original wording. The new wording goes in
`current_query`, and `revision_history` gains an entry with timestamp, reviewer,
`from_query`/`to_query`, `from_category`/`to_category` and the note.

Interface:

```bash
.venv/bin/streamlit run scripts/review_queries.py
```

One query at a time with category, source document and section, the grounding
passage (review only), an editable wording field, an editable category, a review
note, and **Accept / Reject / Needs revision**. Progress by status is shown in
the sidebar, filters exist for category and status, "Next unreviewed query"
resumes where you stopped, and every decision is appended immediately, so
closing the app loses nothing. The screen shows no candidate chunks, retrieval
systems, ranks or grades.

## 6. Relevance grading guidelines

| Grade | Meaning |
|---|---|
| 0 | Not relevant — the chunk does not help answer the query. |
| 1 | Marginally relevant — related terminology or background, no real answer. |
| 2 | Relevant — answers a meaningful part of the query. |
| 3 | Highly relevant — directly and sufficiently answers the query. |

Judge the chunk text, not the title, URL or heading alone. Grade 0 means "I read
it and it does not help". An unread candidate is `pending`, which is **not** 0.

Judging is a different task from reviewing queries:

```bash
.venv/bin/streamlit run scripts/judge_relevance.py
```

Judgments append to `judgments.jsonl` (append-only, re-grading supersedes), with
progress, resume and JSONL/CSV export from the sidebar.

## 7. Query-quality checks (advisory)

`query_review_report()` and `validate_eval_dataset.py` produce warnings; they
never accept, reject or edit a query. Current warnings: `short_query` 17,
`passage_markup` 26, `near_duplicate` 2 — 34 queries have at least one warning.

Heuristics: empty or short queries; identical wording; near-duplicate wording
(content-token Jaccard ≥ 0.6); query equal to a heading, section or document
title; query text appearing verbatim inside its grounding passage; unknown
tokens relative to the corpus vocabulary; missing `source_doc_ids`,
`source_section` or `source_passage`; unknown source document; source chunk id
not present in either strategy; grounding passage containing `##` markup or
newlines; invalid category; corpus-version mismatch.

`passage_markup` is a real finding about Task 3's passage extraction: 26
passages begin with heading text and include markdown. It does not invalidate a
query, but reviewers should read the section rather than trust the passage.

Structural validation (schema, provenance, versions) is reported separately from
human approval: a dataset can be structurally valid with 0 accepted queries.

## 8. Split audit and group-aware split

The original split (`stratified-category`, seed 13, 37 dev / 13 test) was
audited by `scripts/audit_split_leakage.py` **before** any change
(`split_audit_legacy_split.json`):

- 6 of 12 document-connected query groups appeared in **both** splits.
- 6 source documents were shared across splits.
- 7 source sections were shared across splits (e.g. "Tour of the API > Core
  concepts": dev `q_021`, test `q_005`).
- 0 exact duplicate queries crossed the split; 1 near-duplicate pair did
  (`q_005` / `q_021`, similarity 0.67).

The leak was therefore real and substantial, so a group-aware split replaced it.

Grouping (`build_query_groups`) is the connected-component closure of: queries
sharing any `source_doc_ids` entry (the whole set is used, so multi-document
queries bridge their documents), queries sharing a normalised `source_section`,
and near-duplicate query pairs (Jaccard ≥ 0.8). This yields 12 groups with sizes
12, 10, 7, 6, 3, 3, 2, 2, 2, 1, 1, 1.

`assign_group_splits` chooses the group subset for `test` exhaustively
(deterministically, fixed seed 13) maximising category coverage, then the number
of distinct groups, then closeness to a 25% target. The result:

- **dev 35 / test 15** (30%, inside the 20–30% window).
- test = 7 groups over 7 documents: 6 + 2 + 2 + 2 + 1 + 1 + 1.
- test covers all six categories (terminology 4, conceptual 2, api_usage 2,
  procedural 3, troubleshooting 2, contextual 2).
- no group, document, section, duplicate or near-duplicate crosses the split
  (`split_audit.json`).

The balanced 37/13 split is no longer reachable: the largest group alone holds
12 queries, and 15 was the closest group-safe assignment that preserved full
category coverage. The split is versioned as `split-f4a04d80` and stored in
`split_assignment.json` (groups, assignment, per-split counts, warnings).
Recomputing it is deterministic; the fields in `queries.jsonl` are written from
that file.

Limitations: categories are not perfectly balanced (terminology 4 vs conceptual
2 in test); test is a small 15-query sample, so per-category estimates will be
noisy; grouping uses source provenance and wording only, not semantic topic
modelling, so two genuinely related queries from different documents can still
land in different splits.

## 9. Approved evaluation set

`approved_queries.jsonl` contains **only** queries whose latest review status is
`accepted`, sorted by `query_id`, with approved wording and category, full
provenance, `corpus_version`, final `split` and `split_group`, the review
metadata and revision history, and pool state (`query_wording`, `status`,
`is_stale`).

`approved_version` is a SHA-256 over each row's
`query_id | query | category | split | corpus_version`, so editing wording,
changing a category, moving a split or accepting another query changes the
version. `approved_dataset_meta.json` records the version, counts by split and
category, stale-pool count and any problems.

Rejected and needs-revision queries are never included, but their review
records remain in `query_reviews.jsonl` for auditability.

**Current state: 0 accepted queries, so the approved file is empty and its
version (`approved-e3b0c44298fc`) is the hash of an empty set. No query has been
human-approved.**

## 10. Candidate-pool consistency

`candidate_pool_baseline.json` records the wording each pool was retrieved with
(reconstructed from the Task 3 build state; future builds write it directly).

If a review edits a query, its pool becomes **stale**: those candidates were
retrieved for the old wording and cannot be treated as the pool for the new one.
`build_approved_dataset.py` reports stale pools and flags them in the approved
rows; `validate_eval_dataset.py` shows `pool consistency`.

Rebuild only the affected pools:

```bash
.venv/bin/python scripts/rebuild_candidate_pools.py --stale
.venv/bin/python scripts/rebuild_candidate_pools.py --query-ids q_003,q_017
```

That script archives the previous rows under `pool_archive/<timestamp>/`,
re-pools the affected queries with the current wording, leaves other queries and
all judgments untouched, and warns if existing judgments referenced candidates
that no longer exist. Nothing is rebuilt automatically: currently 0 pools are
stale.

## 11. Known limitations

- Queries are template-based, so phrasing variety is limited; they remain
  candidates until a human reviews them.
- 17 queries are short (few content tokens) and 26 grounding passages contain
  markdown; both are advisory warnings, not defects to be fixed automatically.
- Rebuilding the dataset regenerates queries from the blueprint list, so
  hand-edited wording lives in the review store, not in `queries.jsonl`.
- Pools come from four systems at depth 3, so a chunk none of them retrieves can
  never be judged relevant (pool bias); comparisons are limited to those systems.
- One judge per candidate; no inter-annotator agreement or adjudication.
- The corpus is 30 pages and has no dedicated webhooks page.
- Grouping is heuristic (documents, sections, wording), not a topic model.

## 12. Adding new queries

Add a blueprint to `QUERY_BLUEPRINTS` in `src/evaluation/eval_dataset.py`, run
`scripts/build_eval_dataset.py`, then review the new query in the review app.
New queries start as `candidate` and are not part of the approved set until
accepted.

## 13. Commands

```bash
# review queries (accept / reject / needs revision / edit)
.venv/bin/streamlit run scripts/review_queries.py

# judge relevance (separate task)
.venv/bin/streamlit run scripts/judge_relevance.py

# audit split leakage
.venv/bin/python scripts/audit_split_leakage.py [--json] [--write]

# freeze the split and rebuild the approved set
.venv/bin/python scripts/build_approved_dataset.py [--method existing] [--dry-run]

# validate everything (queries, review, split, pools, judgments)
.venv/bin/python scripts/validate_eval_dataset.py [--json]

# rebuild pools after edits
.venv/bin/python scripts/rebuild_candidate_pools.py --stale

# rebuild the candidate dataset (corpus change)
.venv/bin/python scripts/build_eval_dataset.py --new-version
```
