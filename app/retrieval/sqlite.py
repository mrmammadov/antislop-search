"""SQLite: FTS5 (bm25()) for text + sqlite-vec (vec0, brute force) for vectors. In-process, in-memory.

Text side
- FTS5 table with columns (title, body, extra); tokenizer `porter unicode61 remove_diacritics 2`
  (Porter stemming over Unicode word splitting). FTS5 has no stopword list.
- Ranking: FTS5's built-in bm25() (k1=1.2, b=0.75, all column weights 1.0). bm25() returns
  *lower is better* (negative numbers), so we negate it.
- Query: words of the query, each double-quoted (so FTS5 operators/syntax can't break it),
  joined with OR. FTS5's implicit operator is AND, which would make long queries match nothing;
  OR ranks partial matches instead.

Vector side
- sqlite-vec vec0 virtual table, float[dim], distance_metric=cosine. vec0 KNN is an exact
  brute-force scan (no ANN index): exact, and fast at this corpus size. score = 1 - cosine distance.
"""

import re
import sqlite3

import numpy as np
import sqlite_vec

from app.retrieval.types import Hit, Passage

_WORD = re.compile(r"\w+", re.UNICODE)


def words(query: str) -> list[str]:
    """Plain word tokens of a query, so FTS5 query syntax/operators in it can't break MATCH."""
    return _WORD.findall(query)


def extra_text(fields: dict, text_fields: tuple[str, ...]) -> str:
    """Passage.fields selected by Config.text_fields, flattened to one string."""
    return " ".join(str(fields.get(f) or "") for f in text_fields)


class SqliteEngine:
    def __init__(self, namespace: str = "default"):
        self.ns = namespace
        self.db = sqlite3.connect(":memory:")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._ids: list[str] = []    # FTS rowid-1 -> passage id
        self._vids: list[str] = []   # vec rowid-1 -> passage id

    def build_text(self, passages: list[Passage], text_fields: tuple[str, ...] = ()) -> None:
        t = f"{self.ns}_fts"
        self.db.execute(f"DROP TABLE IF EXISTS {t}")
        self.db.execute(f"CREATE VIRTUAL TABLE {t} USING fts5(title, body, extra, "
                        "tokenize = 'porter unicode61 remove_diacritics 2')")
        self._ids = [p.id for p in passages]
        self.db.executemany(f"INSERT INTO {t}(rowid, title, body, extra) VALUES (?, ?, ?, ?)",
                            [(i + 1, p.title, p.text, extra_text(p.fields, text_fields))
                             for i, p in enumerate(passages)])
        self.db.commit()

    def search_text(self, query: str, k: int) -> list[Hit]:
        terms = words(query)
        if not terms:
            return []
        match = " OR ".join('"' + w.replace('"', '""') + '"' for w in terms)
        t = f"{self.ns}_fts"
        rows = self.db.execute(f"SELECT rowid, bm25({t}) AS s FROM {t} WHERE {t} MATCH ? "
                               "ORDER BY s LIMIT ?", (match, k)).fetchall()
        return [Hit(self._ids[r - 1], -s) for r, s in rows]

    def build_vectors(self, passages: list[Passage], vectors: np.ndarray) -> None:
        t = f"{self.ns}_vec"
        dim = vectors.shape[1]
        self.db.execute(f"DROP TABLE IF EXISTS {t}")
        self.db.execute(f"CREATE VIRTUAL TABLE {t} USING vec0("
                        f"embedding float[{dim}] distance_metric=cosine)")
        self._vids = [p.id for p in passages]
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self.db.executemany(f"INSERT INTO {t}(rowid, embedding) VALUES (?, ?)",
                            [(i + 1, v.tobytes()) for i, v in enumerate(vecs)])
        self.db.commit()

    def search_vectors(self, qvec: np.ndarray, k: int) -> list[Hit]:
        t = f"{self.ns}_vec"
        q = np.ascontiguousarray(qvec, dtype=np.float32).tobytes()
        rows = self.db.execute(f"SELECT rowid, distance FROM {t} WHERE embedding MATCH ? AND k = ? "
                               "ORDER BY distance", (q, min(k, len(self._vids)))).fetchall()
        return [Hit(self._vids[r - 1], 1.0 - d) for r, d in rows]

    def close(self) -> None:
        self.db.close()
