"""Disk-backed lexical retrieval over a complete, task-independent JSONL corpus."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path

from core.provenance import file_sha


class KnowledgeBase:
    def __init__(self, path: Path, cache_dir: Path):
        self.sha256 = file_sha(path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / (self.sha256 + ".sqlite")
        if not target.is_file():
            fd, temporary = tempfile.mkstemp(prefix="kb_", suffix=".sqlite", dir=cache_dir)
            os.close(fd)
            connection = sqlite3.connect(temporary)
            try:
                connection.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, url TEXT, title TEXT, text TEXT)")
                connection.execute("CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED, title, text)")
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row.get("text"), str) or not row["text"].strip():
                            raise ValueError("Every KB document needs nonempty text")
                        key = "kb:" + str(row["id"])
                        title = str(row.get("title", ""))
                        connection.execute("INSERT INTO documents VALUES (?,?,?,?)",
                                           (key, str(row.get("url", key)), title, row["text"]))
                        connection.execute("INSERT INTO search VALUES (?,?,?)", (key, title, row["text"]))
                if not connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]:
                    raise ValueError("The KB is empty")
                connection.commit()
                connection.close()
                os.replace(temporary, target)
            except Exception:
                connection.close()
                Path(temporary).unlink(missing_ok=True)
                raise
        self.connection = sqlite3.connect(target)
        self.connection.row_factory = sqlite3.Row

    def search(self, query: str):
        words = list(dict.fromkeys(re.findall(r"\w+", query.casefold())))[:40]
        if not words:
            return []
        match = " OR ".join('"' + word + '"' for word in words)
        rows = self.connection.execute("SELECT d.id,d.url,d.title,substr(d.text,1,500) AS snippet "
            "FROM search s JOIN documents d ON d.id=s.id WHERE search MATCH ? ORDER BY bm25(search), d.id LIMIT 5", (match,))
        return [{"url": row["id"], "source_url": row["url"], "title": row["title"], "snippet": row["snippet"]} for row in rows]

    def document(self, key: str):
        row = self.connection.execute("SELECT url,title,text FROM documents WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError("Unknown KB document")
        return dict(row)

    def close(self):
        self.connection.close()
