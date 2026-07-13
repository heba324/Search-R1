"""Translate stored Lucene documents into Search-R1 retrieval documents."""

import json
from typing import Callable, Iterable, Optional


def document_from_raw(raw: str):
    return document_from_record(json.loads(raw))


def document_from_record(record):
    contents = record["contents"]
    lines = contents.split("\n")
    return {
        "title": lines[0].strip('"'),
        "text": "\n".join(lines[1:]),
        "contents": contents,
    }


def build_search_batches(
    queries: Iterable[str],
    requested: int,
    return_scores: bool,
    search: Callable,
    document_for_hit: Callable,
    on_query_error: Optional[Callable[[str, Exception], None]] = None,
):
    """Return one valid result list per query, including empty or malformed queries."""
    batches = []
    for raw_query in queries:
        query = raw_query.strip()
        hits = []
        if query:
            try:
                hits = list(search(query, requested))
            except Exception as error:
                if on_query_error is not None:
                    on_query_error(query, error)
        batch = []
        for hit in hits[:requested]:
            document = document_for_hit(hit)
            batch.append(
                {"document": document, "score": hit.score}
                if return_scores
                else document
            )
        batches.append(batch)
    return {"result": batches}
