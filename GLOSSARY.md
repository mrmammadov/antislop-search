# Glossary

Terms used in this project, in plain words, with this project's own numbers where they help.
Numbers are nDCG@10 unless noted ("smoke" = the benchmark's 157-query smoke eval, "v1" = `data/eval/v1`).

## Search basics

| Term | Meaning |
|---|---|
| **Corpus** | Everything you search over. Here: 177 antislop resources → 170 searchable (`data/clean/docs.jsonl`). |
| **Document (doc)** | One resource: an essay, talk or video transcript. Relevance is judged per doc. |
| **Passage / chunk** | A piece of a doc that gets indexed on its own. `para256` turns 170 docs into ~3,400 passages. |
| **Chunking / chunker** | The rule that cuts docs into passages. `para256` = pack whole paragraphs up to 256 tokens; `tok256o40` = blind 256-token windows overlapping by 40. |
| **Index** | A data structure built ahead of time so search doesn't scan raw text, like a book's index. |
| **Query** | What the user types. |
| **Retrieval** | Finding candidate passages for a query. Modes here: `bm25`, `dense`, `hybrid`. |
| **Ranking** | Putting the candidates in order. Finding is easy here (Recall@50 ≈ 0.97); ordering is the weak spot. |
| **Known-item vs topical query** | Known-item: you're looking for one specific doc ("the Sivers essay about dying empty"). Topical: anything on a subject ("how to deal with fear"). BM25 is strong on known-item, dense helps topical. |

## Keyword search (sparse)

| Term | Meaning |
|---|---|
| **Inverted index** | Word → list of passages containing it. The core of keyword search. |
| **Token / tokenizer** | The units text is split into. For BM25, roughly words. For the embedding model, WordPiece sub-words (`bge-small`'s tokenizer), which is how chunk sizes are measured. |
| **Stemming (Porter)** | Reducing words to a root so "running" matches "run". SQLite FTS5 uses Porter here. |
| **Stopwords** | Very common words ("the", "of") that are often ignored. FTS5 has no stopword list; IDF down-weights them instead. |
| **TF / IDF** | Term frequency (how often a word is in this passage) / inverse document frequency (how rare the word is overall). Rare words count more. |
| **BM25** | The standard keyword ranking formula: TF × IDF, with diminishing returns for repeats (k1=1.2) and a correction for long passages (b=0.75). Smoke: 0.736 whole-doc. |
| **FTS5** | SQLite's built-in full-text search: inverted index + `bm25()`. Our keyword side. |

## Vector search (dense)

| Term | Meaning |
|---|---|
| **Embedding** | A list of numbers (384 for `bge-small`) representing a text's meaning. Similar meanings → nearby vectors, even with no shared words. |
| **Embedding model / embedder** | The model that makes embeddings. Ours: `bge-small` (BAAI/bge-small-en-v1.5), run on CPU via fastembed + ONNX. |
| **Dense retrieval** | Search by comparing the query's embedding to every passage's embedding. Called "dense" because every number in the vector is used (vs. sparse word lists). |
| **Cosine similarity** | How close two vectors point, from -1 to 1. With **L2-normalized** vectors (length 1) it's just a dot product. |
| **Context window / truncation** | `bge-small` reads at most 512 tokens; the rest is silently dropped. Embedding whole docs (avg ~3.7k words) threw most of each essay away: dense 0.500 whole-doc → 0.653 chunked. |
| **kNN / brute force** | Find the k nearest vectors by checking all of them. Exact. Fine at ~3,400 passages. |
| **ANN / HNSW** | Approximate nearest neighbours: a graph index (HNSW) that skips most vectors. Needed at millions of vectors, trades a little recall for speed. Not used here. |
| **sqlite-vec** | SQLite extension for vectors (`vec0` table, brute-force cosine). Our dense side. |
| **Embedding cache** | `data/embeddings/`: vectors saved to disk, keyed by model + text hash, so restarts don't re-embed. |

## Combining and ranking

| Term | Meaning |
|---|---|
| **Hybrid search** | Run keyword and dense search, merge the two result lists. Ours: best of the three modes (v1: 0.721). |
| **Fusion / RRF** | Reciprocal Rank Fusion: each list gives a passage 1/(60 + rank); add them up. Uses ranks, not scores, so BM25 and cosine scales don't need matching. |
| **Doc aggregation** | Turning passage hits into a doc ranking. Ours: a doc's score = its best passage's score (max). |
| **Reranker / cross-encoder** | A slower, smarter model that re-reads the top ~100 (query, passage) pairs together and re-orders them. The next lever (ticket 009), since ordering is the weak spot. |
| **Snippet** | The text shown under a result: the window of the best passage with the most query words. |

## Measuring quality

| Term | Meaning |
|---|---|
| **Eval set / test collection** | Queries + **qrels** (relevance judgments: query, doc, grade 0–3). Ours: `data/eval/v1`, 33 queries. |
| **Smoke eval** | A quick, free eval from structure already in the data: each antislop summary is a query whose answer is its own doc. Fast, but favours BM25. |
| **LLM-judged** | Relevance grades written by an LLM instead of a person. Cheap; needs spot-checking (ticket 011). |
| **nDCG@10** | Main score, 0–1: are the most relevant docs at the very top of the first 10? Rewards higher grades and higher positions. |
| **Recall@k** | Share of all relevant docs that appear in the top k. "Did we find them at all?" |
| **MRR** | Mean Reciprocal Rank: 1 / position of the first relevant result. "How far down is the first good one?" |
| **judged@10** | Share of the top 10 that has any judgment. Low = scores are unreliable (unjudged docs count as irrelevant). |
| **Pooling / pooling bias** | Judges only grade docs that some system returned (the "pool"). Systems that weren't in the pool surface unjudged docs and get under-scored. v1 was pooled from whole-doc runs, so chunked setups look worse than they are (ticket 012). |

## App and tooling

| Term | Meaning |
|---|---|
| **Pipeline / Config** | `app/retrieval/pipeline.py`: chunker → SQLite (keyword and/or vector) → fusion → doc ranking. A `Config` names one setup, e.g. `hybrid__para256__bge-small__rrf`. |
| **Knob** | A setting you can switch in the UI. Only the retrieval mode is left. |
| **URL as state** | Putting the query and settings in the URL (`?q=…&retrieval=bm25`), so reload, back/forward, bookmarks and sharing all work. Same as Google or Amazon. |
| **FastAPI / uvicorn** | The Python web framework for the API / the server program that runs it. |
| **uv / venv** | uv manages Python, dependencies and the virtual environment (`.venv`, an isolated set of packages for this project). Always `uv run`. |
| **Dependency group** | Optional set of dependencies. `extract` (scraping libs) is installed locally but skipped on the server: `uv sync --no-default-groups`. |
| **Editable path dependency** | Using another local folder as a package, live. The app used `../search_experiments` this way until 2026-10-09. |
| **PID / `lsof`** | Process ID / "list open files", e.g. `lsof -i :8000` shows what's using port 8000. |
| **pdb / breakpoint** | Python's debugger. Stop at a line, inspect values (`p`), step (`n`, `s`), see return values (`r`). |
| **gh / OAuth token scopes** | GitHub's CLI / what the login token may do (`repo` = full access to your repos). |
