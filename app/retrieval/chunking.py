"""Chunkers: Doc -> Passages. Register new ones in CHUNKERS.

Token counts use the embedder's own tokenizer (bge-small's WordPiece), so a chunk sized N tokens
really is N tokens to the model. Every chunk is budgeted so `embed_text()` (title + "\\n\\n" + chunk)
fits the 512-token window: chunk tokens <= 510 - title tokens ([CLS]/[SEP] take the other 2).
"""

import re
from collections.abc import Callable
from functools import cache, partial

from app.retrieval.types import Passage

META_FIELDS = ("author", "type", "date", "summary", "key_highlight")
TOKENIZER = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 510  # 512 minus [CLS] and [SEP]


def _meta(doc: dict) -> dict:
    m = doc["metadata"]
    return {"author": doc["author"], **{k: m.get(k) for k in META_FIELDS if k != "author"}}


def whole_doc(doc: dict) -> list[Passage]:
    """Baseline: one passage per doc, full text (embedders will truncate long ones — that's a finding)."""
    return [Passage(id=f"{doc['id']}#0", doc_id=doc["id"], title=doc["title"],
                    text=doc["text"], fields=_meta(doc))]


# --- token helpers ---------------------------------------------------------------------------

@cache
def _tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(TOKENIZER)
    tok.no_truncation()
    tok.no_padding()
    return tok


def _offsets(text: str) -> list[tuple[int, int]]:
    """Char span of each model token in text (no special tokens)."""
    return _tokenizer().encode(text, add_special_tokens=False).offsets


def _ntok(text: str) -> int:
    return len(_offsets(text)) if text else 0


def _budget(doc: dict, size: int) -> int:
    """Chunk token budget: `size`, capped so title + separator + chunk fits the model window.
    2 tokens of slack: re-tokenizing a slice can split its edge words differently."""
    return max(32, min(size, MAX_TOKENS - 2 - _ntok(doc["title"] + "\n\n")))


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Slide a `size`-token window with `overlap` tokens of overlap; slice the original text by offsets."""
    offs = _offsets(text)
    if not offs:
        return []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(offs), step):
        end = min(start + size, len(offs))
        out.append(text[offs[start][0]:offs[end - 1][1]])
        if end == len(offs):
            break
    return out


def _passages(doc: dict, texts: list[str]) -> list[Passage]:
    """Wrap chunk texts as passages; a doc with no text still yields one (title-only) passage."""
    texts = [t.strip() for t in texts if t.strip()] or [""]
    f = _meta(doc)
    return [Passage(id=f"{doc['id']}#{i}", doc_id=doc["id"], title=doc["title"], text=t, fields=f)
            for i, t in enumerate(texts)]


# --- chunkers --------------------------------------------------------------------------------

def token_windows(doc: dict, size: int, overlap: int) -> list[Passage]:
    """Fixed-size token windows (bge-small tokenizer), blind to structure."""
    b = _budget(doc, size)
    return _passages(doc, _windows(doc["text"], b, min(overlap, b // 2)))


_SENT = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")


def _units(text: str, budget: int) -> list[tuple[str, int]]:
    """Break text into (piece, n_tokens) no larger than budget: paragraphs, then sentences,
    then hard token windows (unpunctuated auto-captions are one giant 'sentence')."""
    out = []
    for para in text.split("\n"):  # cleaned text: one paragraph per line (no blank lines)
        para = para.strip()
        if not para:
            continue
        if (n := _ntok(para)) <= budget:
            out.append((para, n))
            continue
        for sent in _SENT.split(para):
            if (n := _ntok(sent)) <= budget:
                out.append((sent, n))
            else:
                out.extend((w, _ntok(w)) for w in _windows(sent, budget, 0))
    return out


def paragraph_pack(doc: dict, size: int) -> list[Passage]:
    """Structure-aware: greedily pack whole paragraphs (or sentences of an oversized one) up to
    `size` tokens. Never cuts mid-paragraph unless the paragraph alone exceeds the budget."""
    b = _budget(doc, size)
    chunks, cur, n_cur = [], [], 0
    for piece, n in _units(doc["text"], b):
        # joining newlines add no WordPiece tokens, so piece counts sum exactly
        if cur and n_cur + n > b:
            chunks.append(cur)
            cur, n_cur = [], 0
        cur.append(piece)
        n_cur += n
    if cur:
        chunks.append(cur)
    return _passages(doc, ["\n".join(c) for c in chunks])


CHUNKERS: dict[str, Callable[[dict], list[Passage]]] = {
    "doc": whole_doc,
    **{f"tok{s}o{o}": partial(token_windows, size=s, overlap=o)
       for s, o in [(128, 0), (128, 20), (256, 0), (256, 40), (400, 0), (400, 60)]},
    "tokmax": partial(token_windows, size=MAX_TOKENS, overlap=0),  # fill the window minus title
    **{f"para{s}": partial(paragraph_pack, size=s) for s in (128, 256, 400)},
    "paramax": partial(paragraph_pack, size=MAX_TOKENS),
}


def chunk(docs: list[dict], chunker: str) -> list[Passage]:
    fn = CHUNKERS[chunker]
    return [p for d in docs for p in fn(d)]


def embed_text(p: Passage) -> str:
    """What gets embedded for a passage. Title gives chunks context; keep in sync across embedders."""
    return f"{p.title}\n\n{p.text}" if p.text else p.title
