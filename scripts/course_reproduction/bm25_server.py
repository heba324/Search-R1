#!/usr/bin/env python3
"""Serve the official Search-R1 Wikipedia BM25 index without using a GPU."""

from __future__ import annotations

import argparse
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset

from bm25_schema import document_from_raw, document_from_record


class BM25Service:
    def __init__(self, index_path: str, corpus_path: str, topk: int):
        self.searcher = LuceneSearcher(index_path)
        self.topk = topk
        first_document = self.searcher.doc(0)
        self.index_contains_text = first_document is not None and first_document.raw() is not None
        self.corpus = None
        if not self.index_contains_text:
            self.corpus = load_dataset("json", data_files=corpus_path, split="train", num_proc=4)

    def document_for_hit(self, hit):
        if self.index_contains_text:
            return document_from_raw(self.searcher.doc(hit.docid).raw())
        return document_from_record(self.corpus[int(hit.docid)])

    def batch_search(self, queries: List[str], topk: Optional[int], return_scores: bool):
        requested = topk or self.topk
        batches = []
        for query in queries:
            hits = self.searcher.search(query, requested)
            if len(hits) < requested:
                raise RuntimeError(f"BM25 returned only {len(hits)} documents; expected {requested}.")
            batch = []
            for hit in hits[:requested]:
                document = self.document_for_hit(hit)
                batch.append({"document": document, "score": hit.score} if return_scores else document)
            batches.append(batch)
        return {"result": batches}


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
service: Optional[BM25Service] = None


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    if service is None:
        raise RuntimeError("BM25 service was not initialized.")
    return service.batch_search(request.queries, request.topk, request.return_scores)


def main() -> None:
    global service
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    service = BM25Service(args.index_path, args.corpus_path, args.topk)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
