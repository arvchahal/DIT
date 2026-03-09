# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DIT (Distributed Inference Tables) is a framework for routing inference requests to specialized "expert" models across a distributed network. Inspired by Distributed Hash Tables, it partitions inference across small domain-specific models, routes queries dynamically, and scales horizontally via NATS pub/sub messaging.

## Commands

**Run all tests:**
```bash
pytest
```

**Run a single test file:**
```bash
pytest tst/routers/test_domain_router.py
```

**Run a specific test:**
```bash
pytest tst/routers/test_domain_router.py::test_route_single_domain_word
```

**Run local experiments** (from repo root):
```bash
PYTHONPATH=src python src/experiments/local/exp0.py
```

**Run distributed experiments** (requires NATS server running on `nats://127.0.0.1:4222`):
```bash
# Start NATS
nats-server -m 8222

# Start subscriber workers (one per expert)
python src/experiments/distributed/exp1_router_comparison/sub.py \
  --model-id flan-t5 --task text2text-generation --model-name google/flan-t5-base
python src/experiments/distributed/exp1_router_comparison/sub.py \
  --model-id biomedlm --task text-generation --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
python src/experiments/distributed/exp1_router_comparison/sub.py \
  --model-id tinyllama --task text-generation --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
python src/experiments/distributed/exp1_router_comparison/sub.py \
  --model-id law-llm --task text-generation --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Run Exp1 publisher (router comparison)
python src/experiments/distributed/exp1_router_comparison/pub.py \
  --experts flan-t5 biomedlm tinyllama law-llm

# Run Exp2 publisher (monolith baseline — needs OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python src/experiments/distributed/exp2/pub.py --model gpt-4o-mini

# Run Exp3 publisher (load-aware / rate-limited routing)
python src/experiments/distributed/exp3/pub.py \
  --experts flan-t5 biomedlm tinyllama law-llm

# Echo mode (smoke test — no NATS, no models needed)
python src/experiments/distributed/exp1_router_comparison/pub.py \
  --experts flan-t5 biomedlm tinyllama law-llm --echo --cycles 1
```

**Analyze results:**
```bash
export OPENAI_API_KEY=sk-...
python src/experiments/analysis/analyze.py \
  --exp1-csv data/exp1_router_comparison/*/results.csv --out-dir ./plots
# Skip LLM scoring (just plot latency from pre-scored CSV):
python src/experiments/analysis/analyze.py \
  --exp1-csv scored.csv --skip-scoring --out-dir ./plots
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components (`src/dit_components/`)

- **`DIT`** — Central orchestrator. Holds an expert table (`Dict[str, DitExpert]`) and a `Router`. Calling `dit.exec(query)` routes the query to the best expert and runs inference, returning `{"response": ..., "expert": key}`. The router can be swapped at runtime via the `router` property setter.
- **`DitExpert`** — Thin wrapper around any callable model or HuggingFace pipeline. In local mode, wraps a HF `pipeline()`. In distributed mode, wraps a remote callable that sends requests over NATS. Supports `quantize="8bit"|"4bit"` via `load_model()`.
- **`DitRouter`** — Adapter that connects the expert table to a `Router` implementation. Used internally by `DIT`.
- **`DITOrchestrator`** — Batch runner that executes queries through a DIT instance with a time budget.

### Routers (`src/routers/`)

All routers extend the abstract `Router` base class which requires implementing `route(query: str) -> str` (returns an expert key). The base class provides a `fallback()` method that picks a random expert.

- **`SimpleRouter`** — Round-robin across experts.
- **`DomainRouter`** — Keyword-matching: tallies domain descriptor hits per expert, picks the highest. Requires `mapping_expert_to_descriptors`.
- **`DomainSimplifiedRouter`** — First-match variant of DomainRouter (returns on first keyword hit).
- **`EmbeddingRouter`** — Cosine similarity between query embedding and expert label embeddings using `paraphrase-MiniLM-L3-v2`. Maintains an MRU cache for tie-breaking.
- **`LoadAwareRouter`** — Wraps any base router and dynamically shifts traffic away from rate-limited or slow experts. Uses `ExpertStatsTracker` (EMA latency, sliding-window RPS, error rate). Used in Exp3.
- **`GNNRouter`** — Placeholder (empty file).

### Distributed Messaging (`src/microservice/`)

Uses NATS for pub/sub communication with protobuf serialization. Subject pattern: `models.<model_id>`, queue group: `ditq.<model_id>`.

- **`Publisher`** — Sends protobuf `Request` messages over NATS with timeout/retry. Runs a background asyncio event loop in a daemon thread, exposing `ask_sync()` for synchronous callers.
- **`make_remote_callable(publisher, model_id)`** — Returns a `callable(query) -> str` pluggable into `DitExpert(model=...)` for transparent remote calls.
- **`make_tracked_remote_callable(publisher, model_id, stats_tracker)`** — Same as above but also records latency/errors into `ExpertStatsTracker` for `LoadAwareRouter`.
- **`Subscriber`** — Listens on `models.<model_id>`, runs `expert.run_model()`, always replies with a protobuf `Response` (even on errors). Concurrency bounded by `asyncio.Semaphore`.
- **`ditsub.py`** — Alternative implementation (`RouterPublisher`, `DitSubscriber`, `RemoteDIT`) with async-first API.

### Protobuf (`src/protos/` and `src/protoc/`)

Proto definitions are in `src/protos/`. Generated Python stubs are in `src/protoc/` (protobuf v4.25.3). Import via `from protoc import Request, Response, Status`.

### Other Modules

- **`src/evaluator/`** — Accuracy metrics: BERTScore, BLEU, ROUGE.
- **`src/query_selector/`** — Iterator that loads CSVs, shuffles, and yields queries across multiple cycles.

### Experiments (`src/experiments/`)

- **`local/`** — Runs routers in-process (no NATS). `exp0.py` benchmarks routing latency across all router types.
- **`distributed/exp1_router_comparison/`** — Compares 4 routers (round_robin, embedding, domain, domain_simplified) against 4 remote expert workers via NATS. Reads `data/queries/combined.csv` (2,002 queries). Output: `data/exp1_router_comparison/<timestamp>/results.csv`.
- **`distributed/exp2/`** — Monolith baseline: sends all 2,002 queries to a single OpenAI model (GPT-4o-mini or GPT-4o). Requires `OPENAI_API_KEY`. Output: `data/exp2_monolith/<timestamp>/results.csv`.
- **`distributed/exp3/`** — Rate-limited routing: compares embedding, domain, and `LoadAwareRouter` under simulated throttling of one expert. Output: `data/exp3_rate_limited/<timestamp>/results.csv`.
- **`analysis/analyze.py`** — LLM-as-judge scoring (via OpenAI) + latency/accuracy plots from Exp1 and Exp2 CSVs.

## Key Patterns

- **Python path setup**: Tests use `tst/conftest.py` to add `src/` to `sys.path`. Experiment scripts do the same inline. When running, either set `PYTHONPATH=src` or run from a context where conftest handles it.
- **Router constructor convention**: All routers use keyword-only args. `experts` is always required. Domain routers also require `mapping_expert_to_descriptors`. Routers may accept extra `**kwargs` for compatibility.
- **Protobuf pinned to v4.25.3**: Generated stubs require this version. Mixing with protobuf v5 causes `MessageFactory.GetPrototype` errors.
- **Publisher is synchronous and sequential**: `ask_sync()` blocks until NATS reply arrives. The pub scripts process one query at a time — total runtime scales linearly with query count × inference time.
- **Quantization**: `sub.py` supports `--quantize 4bit|8bit` (requires `bitsandbytes`). Use for 7B+ models on Mac to reduce RAM from ~14 GB to ~4-5 GB per model.
- **Prompt format for base models**: When no chat template is available, subscribers fall back to `Q: <query>\nA:` format for text-generation tasks.

## Running on a Single Mac (Resource Guide)

The publisher sends queries **sequentially** (one at a time), so total runtime = `num_queries × num_routers × inference_time_per_query`.

**Exp1 scale**: 2,002 queries × 4 routers = ~8,000 rows. **Exp3 scale**: 2,002 × 3 routers = ~6,000 rows.

| Model class | RAM/model | 3 experts total | Runtime (Exp1) |
|---|---|---|---|
| Classifiers (distilbert, etc.) | ~250 MB | ~750 MB | 5–15 min |
| Flan-T5-base (text2text) | ~900 MB | ~2.7 GB | 15–45 min |
| TinyLlama 1.1B | ~2.2 GB fp16 | ~6.6 GB | 1–3.5 hours |
| 7B–8B with `--quantize 4bit` | ~4–5 GB | ~12–15 GB | 4–10 hours |
| 7B–8B fp16 | ~14–16 GB | **~45 GB — OOM** | Not feasible |

**Recommendation for local Mac runs**: Use 1B-class models (TinyLlama, Flan-T5-base) or `--quantize 4bit` for larger models. 3 experts is workable (4 is preferred for full domain coverage). Use `--cycles 1` and `--routers embedding domain` to cut run time roughly in half.
