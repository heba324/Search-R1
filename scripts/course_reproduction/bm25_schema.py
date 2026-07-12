"""Translate stored Lucene documents into Search-R1 retrieval documents."""

import json


def document_from_raw(raw: str):
    contents = json.loads(raw)["contents"]
    lines = contents.split("\n")
    return {
        "title": lines[0].strip('"'),
        "text": "\n".join(lines[1:]),
        "contents": contents,
    }
