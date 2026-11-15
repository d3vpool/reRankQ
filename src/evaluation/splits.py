"""
Leakage auditing and group-aware development/test splits.

The original split assigned queries independently, stratified by category. That
lets two queries grounded in the same document (or the same section, or near
duplicates of each other) land on opposite sides of the split, which leaks test
information into development.

This module groups queries that cannot be separated without leaking (shared
source document, shared section, near-duplicate wording), audits a split for
those overlaps, and assigns whole groups to a split.
"""

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

SPLIT_METHOD = "group-doc-section-neardup"
GROUP_SPLIT_SEED = 13
GROUP_TEST_FRACTION = 0.25
GROUP_NEAR_DUP_THRESHOLD = 0.8
AUDIT_NEAR_DUP_THRESHOLD = 0.6
MAX_ENUMERATED_GROUPS = 14

_STOPWORDS = frozenset(
    """
    a an the is are was were be been being do does did doing how what why when
    which who whom whose i my me we our you your it its this that these those
    of to in on for and or but if then than as at by with from about into over
    can could should would might may must will shall use used using
    """.split()
)


def normalise_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def query_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", (text or "").lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def near_duplicate_pairs(
    queries: Iterable[dict], threshold: float = AUDIT_NEAR_DUP_THRESHOLD
) -> list[dict]:
    """Pairs of queries whose content-token Jaccard similarity reaches threshold."""
    queries = sorted(queries, key=lambda q: q["query_id"])
    pairs = []
    for first, second in itertools.combinations(queries, 2):
        similarity = jaccard(query_tokens(first["query"]), query_tokens(second["query"]))
        if similarity >= threshold:
            pairs.append(
                {
                    "query_ids": [first["query_id"], second["query_id"]],
                    "queries": [first["query"], second["query"]],
                    "similarity": round(similarity, 3),
                    "splits": [first.get("split"), second.get("split")],
                }
            )
    pairs.sort(key=lambda p: (-p["similarity"], p["query_ids"]))
    return pairs


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self._parent[item]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        while item != parent:
            self._parent[item], item = parent, self._parent[item]
        return parent

    def union(self, first: str, second: str) -> None:
        self._parent[self.find(first)] = self.find(second)


def _group_id(query_ids: Iterable[str]) -> str:
    digest = hashlib.sha1("|".join(sorted(query_ids)).encode()).hexdigest()
    return f"grp_{digest[:8]}"


def build_query_groups(
    queries: Iterable[dict], threshold: float = GROUP_NEAR_DUP_THRESHOLD
) -> list[dict]:
    """
    Connected groups of queries that must share a split.

    Edges: shared source document (any of ``source_doc_ids``), shared normalised
    ``source_section``, and near-duplicate wording at ``threshold``. Groups are
    a connected-component closure, so transitive relationships are respected.
    """
    queries = list(queries)
    if not queries:
        return []
    by_id = {query["query_id"]: query for query in queries}
    uf = _UnionFind(by_id)

    def merge_bucket(bucket: list[str]) -> None:
        for query_id in bucket[1:]:
            uf.union(bucket[0], query_id)

    docs: dict[str, list[str]] = defaultdict(list)
    sections: dict[str, list[str]] = defaultdict(list)
    for query in queries:
        for doc_id in query.get("source_doc_ids") or []:
            if doc_id:
                docs[doc_id].append(query["query_id"])
        section = normalise_text(query.get("source_section") or "")
        if section:
            sections[section].append(query["query_id"])

    for bucket in docs.values():
        merge_bucket(bucket)
    for bucket in sections.values():
        merge_bucket(bucket)
    for pair in near_duplicate_pairs(queries, threshold):
        uf.union(*pair["query_ids"])

    components: dict[str, list[str]] = defaultdict(list)
    for query_id in by_id:
        components[uf.find(query_id)].append(query_id)

    groups = []
    for query_ids in components.values():
        query_ids = sorted(query_ids)
        doc_ids = sorted(
            {doc for qid in query_ids for doc in by_id[qid].get("source_doc_ids") or []}
        )
        groups.append(
            {
                "group_id": _group_id(query_ids),
                "query_ids": query_ids,
                "doc_ids": doc_ids,
                "sections": sorted(
                    {
                        by_id[qid].get("source_section") or ""
                        for qid in query_ids
                        if by_id[qid].get("source_section")
                    }
                ),
                "size": len(query_ids),
            }
        )
    groups.sort(key=lambda g: (-g["size"], g["group_id"]))
    return groups


def _split_counts(queries: list[dict]) -> dict:
    return dict(Counter(query.get("split") for query in queries))


def audit_split_leakage(
    queries: Iterable[dict], threshold: float = AUDIT_NEAR_DUP_THRESHOLD
) -> dict:
    """
    Report every way the current dev/test assignment shares information.

    Looks at shared source documents, shared sections, duplicate wording and
    near-duplicate wording across splits, plus group-level overlap and category
    distribution. Reports findings; never changes the split.
    """
    queries = list(queries)
    groups = build_query_groups(queries)
    by_id = {query["query_id"]: query for query in queries}

    def shared(attribute: str, key_fn) -> list[dict]:
        buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"dev": [], "test": []})
        for query in queries:
            for key in key_fn(query):
                if key:
                    buckets[key][query.get("split")].append(query["query_id"])
        return [
            {
                attribute: key,
                "dev": sorted(value["dev"]),
                "test": sorted(value["test"]),
                "dev_count": len(value["dev"]),
                "test_count": len(value["test"]),
            }
            for key, value in sorted(buckets.items())
            if value["dev"] and value["test"]
        ]

    shared_docs = shared("doc_id", lambda q: q.get("source_doc_ids") or [])
    shared_sections = shared("section", lambda q: [q.get("source_section") or ""])

    def duplicates_across_splits(exact: bool) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for query in queries:
            buckets[normalise_text(query["query"])].append(query)
        pairs = []
        for bucket in buckets.values():
            splits = {query.get("split") for query in bucket}
            if len(bucket) > 1 and len(splits) > 1:
                pairs.append(
                    {
                        "query_ids": sorted(query["query_id"] for query in bucket),
                        "queries": [query["query"] for query in bucket],
                        "splits": [query.get("split") for query in bucket],
                    }
                )
        return pairs if exact else []

    exact_duplicates = duplicates_across_splits(exact=True)
    near_duplicates = [
        pair
        for pair in near_duplicate_pairs(queries, threshold)
        if pair["splits"][0] != pair["splits"][1]
        and pair["query_ids"] not in [p["query_ids"] for p in exact_duplicates]
    ]

    groups_spanning = [
        {
            "group_id": group["group_id"],
            "size": group["size"],
            "query_ids": group["query_ids"],
            "dev": sorted(
                qid for qid in group["query_ids"] if by_id[qid].get("split") == "dev"
            ),
            "test": sorted(
                qid for qid in group["query_ids"] if by_id[qid].get("split") == "test"
            ),
        }
        for group in groups
        if len({by_id[qid].get("split") for qid in group["query_ids"]}) > 1
    ]

    leakage_found = bool(
        shared_docs or shared_sections or exact_duplicates or near_duplicates or groups_spanning
    )
    return {
        "split_method_in_data": "stratified-category" if not groups_spanning else "leaky",
        "num_queries": len(queries),
        "by_split": _split_counts(queries),
        "by_category_split": {
            split: dict(
                Counter(query["category"] for query in queries if query.get("split") == split)
            )
            for split in ("dev", "test")
        },
        "num_groups": len(groups),
        "group_sizes": [group["size"] for group in groups],
        "groups_spanning_splits": groups_spanning,
        "shared_documents": shared_docs,
        "shared_sections": shared_sections,
        "exact_duplicates_across_splits": exact_duplicates,
        "near_duplicates_across_splits": near_duplicates,
        "near_duplicate_threshold": threshold,
        "leakage_found": leakage_found,
        "summary": (
            f"{len(groups_spanning)} of {len(groups)} query groups span both splits; "
            f"{len(shared_docs)} documents and {len(shared_sections)} sections are shared; "
            f"{len(exact_duplicates)} duplicate and {len(near_duplicates)} near-duplicate "
            "query pairs cross the split."
            if leakage_found
            else "No document, section, duplicate or near-duplicate overlap across splits."
        ),
    }


def split_version(assignment: dict[str, str], groups: list[dict], seed: int) -> str:
    """Version of a split: assignment plus group composition and seed."""
    rows = [f"{qid}:{split}" for qid, split in sorted(assignment.items())]
    if groups:
        rows += [f"{group['group_id']}:{assignment.get(group['query_ids'][0], '')}" for group in groups]
    rows += [SPLIT_METHOD, f"seed={seed}"]
    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return f"split-{digest[:8]}"


def _enumerate_subsets(groups: list[dict], low: int, high: int, target: int) -> list[dict]:
    candidates = []
    count = len(groups)
    for mask in range(1, (1 << count) - 1):
        selected = [groups[index] for index in range(count) if mask & (1 << index)]
        size = sum(group["size"] for group in selected)
        candidates.append({"groups": selected, "size": size})
    in_range = [candidate for candidate in candidates if low <= candidate["size"] <= high]
    if in_range:
        return in_range
    # No subset meets the fraction window: keep group isolation and report it.
    return sorted(candidates, key=lambda candidate: (abs(candidate["size"] - target), candidate["size"]))[:1]


def _greedy_subset(groups: list[dict], low: int, high: int, target: int, seed: int) -> list[dict]:
    import random

    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    selected: list[dict] = []
    size = 0
    for group in ordered:
        if size + group["size"] <= high:
            selected.append(group)
            size += group["size"]
    if not selected:
        selected = [min(groups, key=lambda group: group["size"])]
    return [{"groups": selected, "size": sum(g["size"] for g in selected)}]


def assign_group_splits(
    queries: list[dict],
    seed: int = GROUP_SPLIT_SEED,
    test_fraction: float = GROUP_TEST_FRACTION,
    threshold: float = GROUP_NEAR_DUP_THRESHOLD,
) -> dict:
    """
    Assign whole query groups to ``dev`` or ``test``.

    Chooses the group subset for ``test`` that covers the most categories, then
    the most distinct groups, then lands closest to ``test_fraction``. Groups
    are never split, so a perfectly balanced split may be impossible.
    """
    queries = list(queries)
    if not queries:
        raise ValueError("Cannot split an empty query set")

    groups = build_query_groups(queries, threshold)
    if len(groups) == 1 and groups[0]["size"] == len(queries):
        raise ValueError(
            "All queries form one connected group (shared documents, sections or "
            "near-duplicate wording); no non-leaking split exists"
        )

    num_queries = len(queries)
    target = max(1, round(num_queries * test_fraction))
    low = max(1, round(num_queries * 0.2))
    high = max(low, round(num_queries * 0.3))

    if len(groups) <= MAX_ENUMERATED_GROUPS:
        candidates = _enumerate_subsets(groups, low, high, target)
    else:
        candidates = _greedy_subset(groups, low, high, target, seed)

    by_id = {query["query_id"]: query for query in queries}

    def score(candidate: dict) -> tuple:
        selected_ids = {qid for group in candidate["groups"] for qid in group["query_ids"]}
        categories = len({by_id[qid]["category"] for qid in selected_ids})
        doc_ids = {doc for group in candidate["groups"] for doc in group["doc_ids"]}
        return (
            categories,
            len(candidate["groups"]),
            -abs(candidate["size"] - target),
            len(doc_ids),
            tuple(sorted(selected_ids)),
        )

    best = max(candidates, key=score)
    test_ids = {qid for group in best["groups"] for qid in group["query_ids"]}
    assignment = {
        query["query_id"]: "test" if query["query_id"] in test_ids else "dev"
        for query in queries
    }

    for group in groups:
        group["split"] = "test" if group["query_ids"][0] in test_ids else "dev"
    for query in queries:
        query["split"] = assignment[query["query_id"]]

    version = split_version(assignment, groups, seed)
    test_queries = [by_id[qid] for qid in sorted(test_ids)]
    warnings = []
    largest = groups[0]
    if largest["size"] >= 0.6 * num_queries:
        warnings.append(
            f"largest group {largest['group_id']} holds {largest['size']} of {num_queries} "
            "queries; a balanced split is not possible without leakage"
        )
    if not (low <= len(test_queries) <= high):
        warnings.append(
            f"test split has {len(test_queries)} queries, outside the {low}-{high} window"
        )
    missing = sorted(
        {query["category"] for query in queries}
        - {query["category"] for query in test_queries}
    )
    if missing:
        warnings.append(f"categories missing from test split: {missing}")
    if len(best["groups"]) == 1:
        warnings.append(
            "test split is a single query group, so it covers one connected topic"
        )

    return {
        "split_method": SPLIT_METHOD,
        "split_version": version,
        "split_seed": seed,
        "test_fraction_target": test_fraction,
        "near_dup_threshold": threshold,
        "num_queries": num_queries,
        "num_groups": len(groups),
        "group_sizes": [group["size"] for group in groups],
        "groups": groups,
        "assignment": assignment,
        "by_split": dict(Counter(assignment.values())),
        "by_category_split": {
            split: dict(
                Counter(
                    query["category"]
                    for query in queries
                    if assignment[query["query_id"]] == split
                )
            )
            for split in ("dev", "test")
        },
        "warnings": warnings,
    }


def apply_split_assignment(queries: Iterable[dict], assignment: dict[str, str]) -> None:
    for query in queries:
        if query["query_id"] not in assignment:
            raise KeyError(f"split assignment missing {query['query_id']}")
        query["split"] = assignment[query["query_id"]]


def write_split_assignment(meta: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_split_assignment(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No split assignment at {path}")
    return json.loads(path.read_text())
