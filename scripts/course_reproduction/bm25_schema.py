"""Translate stored Lucene documents into Search-R1 retrieval documents."""

import json


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
