# Parallel and resumable API extraction

## Input and dependencies

Use Python 3.9 or newer and `jsonschema` from [../scripts/requirements.txt](../scripts/requirements.txt). The runner uses the standard library for HTTP, threads, rate limiting, SQLite and CSV exports. It supports an OpenAI-compatible `/chat/completions` endpoint with JSON output. The configured model must accept temperature 0 and the supplied token limit.

Input can be CSV, a JSON array, a single JSON article, or JSONL. Each record requires a unique `id` or `pmid`, `title` and `abstract`. Additional metadata is preserved in JSON results. If the same PMID occurs in separate relation records, supply a unique composite `id`; the runner will reject duplicate identifiers. A run has one research-question object. For multiple relations, use one run/output directory per question and merge results with the relation identifier afterward.

Set `EVIDENCE_API_KEY` securely in the environment, or specify the name of an already configured variable with `--api-key-env`. Keys never appear in command arguments or run manifests. Keep input files and run outputs outside the skill folder and public repository.

## Commands

Run from the skill folder. Replace the provider URL and model with the user's configured service. The base URL is the path immediately before `/chat/completions`, including `/v1` when that provider requires it.

```sh
python3 -m pip install -r scripts/requirements.txt
python3 scripts/run_batch.py \
  --input /path/to/abstracts.jsonl \
  --question /path/to/question.json \
  --output /path/to/run-output \
  --base-url https://api.deepseek.com \
  --model deepseek-flash \
  --disable-thinking \
  --workers 4 \
  --requests-per-minute 60
```

The provider/model above are an execution example. They do not set a research-question default. `--disable-thinking` sends the provider-specific `thinking.type=disabled` setting used in the original DeepSeek workflow. Use it only with providers supporting that option. Reasoning-enabled models may consume the output budget before returning JSON; configure the mode and token budget deliberately. Each study normally uses two model calls, plus any retries. `--requests-per-minute` limits total request starts across both stages and all worker threads in this process. It is not a token-per-minute limiter; lower the rate for long abstracts or provider token limits. Separate output directories do not share a rate limiter, so divide the account's allowance among concurrent runs.

The default stage is `both` (evidence extraction then entity recognition). Use `--stage extract` for evidence only. Use `--index /path/to/verified-index.json` to attach verified MeSH/ICD/UMLS codes. Without an index, entity mentions and roles are still extracted, and codes remain null pending terminology lookup.

For an interrupted run, repeat the same command with `--resume`. It processes pending/interrupted studies and skips successful stages. For recorded failed studies, repeat it with `--retry-failed`. A mapping failure retries mapping while retaining the saved extraction. The two options are mutually exclusive; process pending records and recorded failures in separate invocations when both exist.

## Persistence and retries

- `run.sqlite3` is the durable source of truth. Each successful extraction is committed before mapping begins, and completed mapping is committed before the terminal status. A crash between commits can leave an in-flight API request repeated on restart; exactly-once billing is not guaranteed.
- `owner.sqlite3` holds a process-wide write lock. A second coordinator cannot process the same output directory concurrently. The operating system releases the lock when the process exits. Use a local filesystem with SQLite locking support, rather than a cloud-synced or network share.
- Only `workers` tasks are submitted at once. Worker count is configurable from 1 to 64 and defaults to 4. Inputs are loaded into memory, so partition unusually large collections into stable shards that fit available memory.
- Every request attempt consumes the shared rate limiter. HTTP 408, 409, 429, server errors and network/response failures receive bounded retries. Respect `Retry-After` up to 300 seconds; otherwise use exponential delay with jitter. Repeated rate limits require lowering the run's rate. Validation failures receive the prior response plus the concrete correction message.
- `--max-attempts` is 1–3 attempts per stage per invocation, default 3. `--timeout` is the per-request socket timeout, default 90 seconds. HTTP 401/403 stops new scheduling and leaves unfinished work recoverable after credentials are corrected. A manual retry invocation grants a fresh per-stage attempt budget.
- On Ctrl-C, the coordinator stops starting new requests, waits for active requests to settle, and exports the committed state. Force termination still leaves committed SQLite stages available for `--resume`.
- Resume requires identical question, inputs, provider/model, prompts, index, output token settings and runner implementation. A changed question or source cannot reuse previous results silently. Worker count, request rate, timeout and retry budget can be adjusted on recovery. Use a new output directory for incompatible changes.

The runner preserves literal source spans and checks JSON Schema, selected metric/direction/CI consistency, entity quote containment and index identifiers. Semantic eligibility, numeric attribution to the correct model, and ambiguous code meanings still require review as described in [workflow.md](workflow.md).

## Outputs and verification

- `results.jsonl` contains each original article, question, extraction, entities, status and error. Source quotes and raw extracted fields remain available.
- `evidence.csv` presents the unified evidence table, including status, model/timepoint context and entity mappings. Numeric columns remain numeric. Text that could execute as a spreadsheet formula is escaped only in CSV; JSON retains the original text.
- `summary.json` records aggregate counts, requested model, manifest fingerprints, mapping scope and validation scope.
- The SQLite `requests` table records attempted calls, duration, provider-reported model, token usage when returned, and received model content. The API key and error response bodies are not logged.

CSV/JSONL/summary are atomic snapshots generated when a run starts or stops. During execution the SQLite state advances after every stage; rerun with `--resume` to regenerate exports after a forced termination. A status of `succeeded` means configured stages completed and automated checks passed, including any explicit unresolved terminology. `no_eligible_estimate`, `failed`, `mapping_failed` and `pending` remain distinct.

Exit code 0 means all studies reached `succeeded` or `no_eligible_estimate`; exit code 1 means unfinished/failed studies remain, and 2 indicates invalid configuration or inputs. `--stage both` can complete with unindexed concepts, which are explicitly marked `mapping_status=unresolved`.

Run the isolated local regression suite without model credentials or provider charges.

```sh
python3 -m unittest discover -s tests -v
```

The suite uses synthetic abstracts and a local fake service to check bounded parallelism, 429 handling, pacing, completed-work skipping, mapping-only retries, interrupted-stage recovery, output ownership, changed-question rejection, duplicate IDs and source validation. Its mock outputs are software fixtures, not research evidence.

## Relation to the original NC code

The project's `notebooks/nc_revision (1).ipynb` implements extraction workers with `threading.Thread`, `queue.Queue`, and eight threads. Its results are collected from an in-memory output queue after `input_queue.join()`. The current batch runner adds persistent stage checkpoints, configuration fingerprints and controlled recovery for this skill. These added controls should not be attributed to the original article's implementation.

Article and associated repository are linked in [Nature Communications](https://doi.org/10.1038/s41467-025-62783-x) and [the project notebook](https://github.com/xuanyshi/llm-evidence-triangulation/blob/main/notebooks/nc_revision%20%281%29.ipynb).
