# Agentic Crawler

Autonomous information retrieval agent that plans sub-queries, crawls the live web (Wikipedia-first, with general web search as a supplement), indexes content into ChromaDB, answers via multi-query RAG with cross-encoder reranking, and validates claims with NLI entailment checking against the full retrieved context. Built as a **LangGraph** `StateGraph` implementing an **Observe → Plan → Act → Validate** loop, running entirely on locally-hosted infrastructure.

## Architecture

```
User Query
    │
    ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐
│ Observe │───▶│  Plan   │───▶│   Act   │───▶│ Validate  │
└─────────┘    └─────────┘    └─────────┘    └─────┬─────┘
     ▲                            │ crawl/index      │
     │                            ▼                  │
     └──────── refine & re-crawl ◀──── continue ─────┘
                                           │
                                accept / budget exhausted
                                           ▼
                                      Finalize
```

| Component | Responsibility |
|-----------|----------------|
| **QueryPlanner** | LLM JSON planning, compound-question splitting, disambiguation-aware search terms |
| **CrawlingAgent** | Wikipedia-first search, robots.txt, rate limits, cosine-similarity link scoring, frontier |
| **KnowledgeIndexer** | clean → junk-filter → chunk → embed → ChromaDB (incremental, per-page) |
| **RagEngine** | multi-query retrieval, cross-encoder reranking, sequential citations, cited answer generation |
| **Validator** | claim decomposition, NLI entailment against full retrieved context, confidence scoring |

## Requirements

- Python 3.13+
- [Ollama](https://ollama.com) installed locally, with a model pulled (default: `qwen2.5:7b`)
- No paid API key required — the system runs entirely on local infrastructure by default

Optional alternative LLM backends (`llm/groq_client.py`, `llm/openrouter_client.py`) are also implemented if you prefer a cloud provider, but Ollama is the default and recommended path.

## Setup

```bash
cd agentic-crawler
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
playwright install chromium   # optional, for JS-rendered pages
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
```

Then pull the local model if you haven't already:

```bash
ollama pull qwen2.5:7b
```

Configure tunables in `config/settings.yaml` (this is the effective source of truth — values here override the Python class defaults in `config/settings.py`). Environment variables in `.env` override both.

## Run a Query

```bash
python main.py --query "Who created the Python programming language?"
```

Output is JSON with answer, confidence, sources, pages crawled, iterations used, and a full execution trace.

## Run Evaluation

Benchmark against a real, seeded random sample of TriviaQA (EM/F1, confidence, hallucination rate, pages crawled, latency):

```bash
python main.py --evaluate --n-samples 20 --seed 42 --output data/processed/benchmark_results.json
```

Start with a small `--n-samples` (e.g. 5) to sanity-check before committing to a larger run — each question takes roughly 2-4 minutes on local hardware.

## Run Tests

```bash
pytest tests/unit -q
pytest tests/integration -q
pytest tests/unit/test_embedder.py -q -m slow   # downloads embedding model on first run
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_pages_per_query` | 35 | Max pages crawled per run |
| `max_crawl_depth` | 3 | Max link depth from seeds |
| `max_iterations` | 5 | Max Observe-Plan-Act-Validate cycles |
| `confidence_threshold` | 0.70 | Validation confidence to accept answer |
| `top_k` | 10 | Retrieved chunks for RAG (post-reranking) |
| `chunk_size_tokens` | 120 | Chunk size (word-based estimate) |
| `chunk_overlap_tokens` | 40 | Chunk overlap |
| `rerank_enabled` | true | Enables cross-encoder reranking of retrieved chunks |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `retrieve_multiplier` | 3 | Candidates fetched pre-rerank, relative to `top_k` |
| `requests_per_second_per_domain` | 1.0 | Per-domain rate limit |
| `min_content_length` | 200 | Minimum cleaned text length |
| `nli_model` | `cross-encoder/nli-deberta-v3-base` | Entailment-checking model used by the Validation Module |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Local LLM served via Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server address |
| `CHROMA_PERSIST_DIR` | `./data/vectorstore` | Vector store path |
| `USER_AGENT` | project-identifying string with contact email | Required by Wikimedia's bot policy for reliable API access |

## Known Limitations

- **Retrieval specificity on ambiguous topics.** The crawler reliably finds real, topically-relevant pages, but on questions with an ambiguous or broad topic (e.g. asking about one specific fact within a broadly-related subject), it can retrieve a genuinely real but insufficiently specific source — producing a confidently-stated but incorrect answer that the Validation Module correctly judges as grounded in the (wrong-but-real) retrieved content.
- **Small local model ceiling.** Using a 7-billion-parameter local model keeps the system free and self-hosted, but caps achievable accuracy relative to larger/cloud-hosted models on genuinely obscure trivia.
- **Search seeding** queries Wikipedia's full-text search first, falling back to DuckDuckGo HTML results only when Wikipedia's coverage is insufficient; results still depend on network conditions and each service's own availability.
- **Token counting** in the chunker uses a word-count approximation, not a true tokenizer.
- **Single-hop QA** benchmark (TriviaQA `rc.nocontext`) does not evaluate multi-hop reasoning across multiple documents.
- **Playwright** is only invoked when static HTML falls below `min_content_length`, so JavaScript-rendered pages with substantial static fallback content may not trigger full rendering.
- **Per-query runtime** averages 2-4 minutes on local hardware, making full-benchmark-scale evaluation impractical; evaluation is run on a fixed, seeded random sample instead (see report for methodology).

## Project Layout

See the repository tree in the project specification. Cross-package imports are restricted to `agents/agent_graph.py` and `app/cli.py` for testability — no module in `crawler/`, `indexer/`, `retrieval/`, `validation/`, or `llm/` imports from another top-level package directly.

## License

MIT