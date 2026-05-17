# gitsight

**Semantic git history analysis — clusters commits into themes with AI labels.**

---

## What's New in v1.1.0

### 1. Config File Support (`~/.gitsight.toml`)

Set persistent defaults in `~/.gitsight.toml` (global) or `.gitsight.toml` in any project directory. CLI flags always take precedence.

```toml
# ~/.gitsight.toml  or  ./.gitsight.toml
model            = "claude-haiku-4-5-20251001"
threshold        = 0.25
min_cluster_size = 3
concurrency      = 5
max_commits      = 500
no_ai            = false
timeline         = true
author_stats     = true
```

Supported keys: `model`, `threshold`, `min_cluster_size`, `concurrency`, `max_commits`, `no_ai`, `branch`, `export_format`, `timeline`, `author_stats`.

### 2. Commit Timeline Chart (`--timeline`)

Print a weekly commit-frequency bar chart before the clustering results. Each row is one calendar week; bar width is proportional to the busiest week. Shows up to 26 weeks of history.

```bash
gitsight . --timeline
```

```
──────────────────── Commit Timeline  (weekly) ────────────────────
 2026-03-30  ████████████████░░░░░░░░░░░░░░░░  12
 2026-04-06  ████████████████████░░░░░░░░░░░░  15
 2026-04-13  ████████████████████████░░░░░░░░  18
 2026-04-20  ████████████████████████████████  24
 2026-04-27  ████████████░░░░░░░░░░░░░░░░░░░░   9
 2026-05-04  ████████████████░░░░░░░░░░░░░░░░  12
 2026-05-11  ████████░░░░░░░░░░░░░░░░░░░░░░░░   6
```

### 3. Author Velocity Table (`--author-stats`)

Print a Rich table showing each contributor's commit count, active date range, span in days, and average commits per week. Sorted by total commits.

```bash
gitsight . --author-stats
```

```
─────────────────────── Author Velocity ───────────────────────────
╭──────────────────┬─────────┬─────────────┬─────────────┬──────────┬──────────╮
│ Author           │ Commits │ First commit │ Last commit │ Span (d) │ Avg / wk │
├──────────────────┼─────────┼─────────────┼─────────────┼──────────┼──────────┤
│ Alice            │      89 │  2025-11-01 │  2026-05-10 │      190 │     3.3  │
│ Bob              │      71 │  2025-11-15 │  2026-05-08 │      174 │     2.9  │
│ Carol            │      47 │  2026-01-03 │  2026-05-01 │      118 │     2.8  │
│ Dave             │      40 │  2025-12-01 │  2026-04-20 │      140 │     2.0  │
╰──────────────────┴─────────┴─────────────┴─────────────┴──────────┴──────────╯
```

---

gitsight reads your git log, groups commits by semantic similarity using TF-IDF vectors and cosine clustering, then streams Claude labels for each group to tell you *what* your codebase has been doing and *why* it matters.

```
  247 commits loaded
  Top authors: Alice 89  Bob 71  Carol 47  Dave 40
  Date range: 2025-11-01 → 2026-05-12

  5 clusters found  (threshold=0.20, min-size=2)

──────────────── 1. Authentication Refactoring  42 commits ─────────────────
  Migrated session management from Redis to JWT tokens across all services,
  including middleware updates and token refresh logic.
  ► Action: Audit token expiry settings before the next security review.
  Keywords: token, jwt, session, refresh, middleware, auth

  abc12345  2026-03-14  Alice    feat: implement JWT refresh endpoint
  cd789abc  2026-03-15  Bob      fix: token expiry off-by-one in auth middleware
  ...

──────────────────── 2. Database Migration Sprint  38 commits ──────────────
  Ported the user and order tables from Postgres 13 to 15, with schema
  migrations and query plan optimisations for the reporting queries.
  ► Action: Monitor slow-query logs for 2 weeks post-migration.
  ...
```

## Breakthrough Techniques

| Technique | Where |
|---|---|
| **LLM integration** | Claude Haiku streams theme labels for each commit cluster in real-time; concurrent API calls via `asyncio.TaskGroup` |
| **Full async architecture** | `asyncio.create_subprocess_exec` for git log, `asyncio.TaskGroup` for concurrent Claude calls with semaphore-bounded concurrency |
| **Semantic vector search** | Pure-numpy TF-IDF + cosine similarity clustering — no sentence-transformers, no FAISS, zero native extensions |
| **Live Rich UI** | Rich `Console` with `Rule`, `Table`, and streaming output printed as Claude tokens arrive |

## Install

```bash
pip install gitsight                     # clustering only (no AI)
pip install "gitsight[ai]"               # + anthropic for Claude labels
pip install "gitsight[dev]"              # + test dependencies
```

## Quick Start

```bash
# Basic analysis (current repo, last 200 commits)
gitsight .

# Last 3 months of a specific repo
gitsight /path/to/repo --since "3 months ago"

# Filter by author
gitsight . --author "Alice"

# Export a Markdown report
gitsight . --output report.md

# Clustering only — no API key needed
gitsight . --no-ai
```

Set your Anthropic API key for AI labeling:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Sample Output

```
  247 commits loaded
  Top authors: Alice 89  Bob 71  Carol 47
  Date range:  2025-11-01 → 2026-05-12

  5 clusters found  (threshold=0.20, min-size=2)

  Labeling 5 clusters with Claude (claude-haiku-4-5-20251001)…

──── 1. Authentication Refactoring  42 commits ────────────────────────────
  Migrated session management from Redis to JWT tokens across all services,
  including middleware updates and token refresh logic.
  ► Action: Audit token expiry settings before the next security review.
  Keywords: token, jwt, session, refresh, middleware

  abc12345  2026-03-14  Alice    feat: implement JWT refresh endpoint
  cd789abc  2026-03-15  Bob      fix: token expiry off-by-one
  ef012345  2026-03-17  Alice    test: add JWT integration tests
  ...

──── 2. Database Migration Sprint  38 commits ────────────────────────────
  ...

──────────────────── Executive Summary ────────────────────────────────────
The repository has been in a major infrastructure modernisation phase.
Authentication and database layers saw the heaviest investment. The high
volume of hotfixes in the payments cluster suggests that area needs more
test coverage before the next release cycle.
```

## All Options

```
usage: gitsight [-h] [--version] [--max-commits N] [--since DATE]
                [--until DATE] [--author PATTERN] [--branch REF]
                [--threshold FLOAT] [--min-cluster-size N]
                [--no-ai] [--model MODEL] [--concurrency N]
                [--output FILE] [--export-format {json,markdown}]
                [repo]

positional arguments:
  repo                    Path to git repository (default: .)

options:
  --max-commits N         Maximum commits to analyze (default: 200)
  --since DATE            Only commits after this date, e.g. "3 months ago"
  --until DATE            Only commits before this date
  --author PATTERN        Filter by author name or email
  --branch REF            Branch or ref to analyze (default: HEAD)
  --threshold FLOAT       Cosine similarity threshold 0.0–1.0 (default: 0.20)
  --min-cluster-size N    Min commits per cluster (default: 2)
  --no-ai                 Keyword labels only, skip Claude API
  --model MODEL           Claude model ID (default: claude-haiku-4-5-20251001)
  --concurrency N         Max concurrent Claude calls (default: 3)
  --output FILE           Write report to FILE (.md or .json)
  --export-format         Force json or markdown (inferred from extension)
```

## How It Works

```
git log  →  Commit list
               │
               ▼
        TF-IDF vectors           ← numpy, zero ML deps
        (bag-of-words, IDF-weighted,
         L2-normalised per commit)
               │
               ▼
        Cosine similarity        ← dot product on unit vectors
        greedy clustering
               │
        ┌──────┴──────┐
        │  Cluster 1  │  Cluster 2  │  ...
        └──────┬──────┘
               │
               ▼
        asyncio.TaskGroup        ← concurrent Claude calls
        (semaphore-bounded)
               │
               ▼
        Streaming labels         ← claude-haiku-4-5-20251001
        (title + summary + action)
               │
               ▼
        Rich terminal output
        + optional JSON/Markdown export
```

### Clustering Algorithm

1. Tokenise each commit message: lowercase, strip punctuation, remove stop words and common git keywords (`feat`, `fix`, `chore`, etc.), split CamelCase.
2. Build TF-IDF matrix: term frequency × IDF (smooth sklearn-style), L2-normalised so cosine similarity = dot product.
3. Greedy single-pass: each commit either joins the nearest existing cluster (if similarity > threshold) or seeds a new one. Centroids are updated as running means and re-normalised.
4. Clusters smaller than `--min-cluster-size` are discarded.

### Concurrency Model

- `asyncio.create_subprocess_exec` reads git log without blocking.
- `asyncio.TaskGroup` launches one labeling task per cluster, all running concurrently.
- An `asyncio.Semaphore(max_concurrent)` caps simultaneous Claude API calls to avoid rate-limit errors.
- Each Claude call runs in a thread via `asyncio.to_thread` since the Anthropic SDK uses a synchronous streaming interface.

## Programmatic API

```python
import asyncio
from pathlib import Path
from gitsight.git import read_commits_async
from gitsight.vectors import embed_messages, greedy_cluster, top_keywords
from gitsight.export import to_json

async def analyze(repo: Path) -> str:
    commits = await read_commits_async(repo, max_count=500, since="6 months ago")
    matrix, vocab = embed_messages([c.message for c in commits])
    clusters = greedy_cluster(matrix, threshold=0.20)
    keywords = [top_keywords(g, matrix, vocab) for g in clusters]
    # ... label with Claude ...
    return to_json(commits, clusters, labels, repo)

asyncio.run(analyze(Path(".")))
```

## Running Tests

```bash
pip install "gitsight[dev]"
pytest tests/ -v
```

## License

MIT
