# CLAUDE.md - modal-eval

## Overview
Unified LLM evaluation framework on Modal with vLLM serving. Uses `verifiers` (vf-eval) from PrimeIntellect for evaluation.

## Development Environment

**IMPORTANT: This project uses `uv` for package management, NOT conda/pip directly.**

```bash
# Correct way to run commands
uv run modal-eval run -m org/model -b benchmark
uv run python -m modal_eval.cli ...

# Check installed packages (NOT `pip show`)
uv pip show <package>
# Or inspect: .venv/lib/python3.x/site-packages/

# Install dependencies
uv sync
uv sync --all-extras  # Include dev/hub/vf-eval extras
```

The `.venv/` directory contains project dependencies. Plain `python` or `pip` may point to conda and give incorrect results.

## Entry Points

```bash
uv run modal-eval run -m allenai/Olmo-3-7B-Think -b livecodebench  # Run evaluation
uv run modal-eval resume -m model -b benchmark                      # Resume interrupted
uv run modal-eval deploy -m model -b benchmark                      # Deploy only
uv run modal-eval stop --all                                        # Stop Modal apps
uv run modal-eval status                                            # Show deployed
uv run modal-eval runs                                              # List recent runs
uv run modal-eval benchmarks                                        # List benchmarks
```

CLI with overrides:
```bash
uv run modal-eval run -m model -b benchmark \
    --infra.gpu="H100:2" \
    --vllm.max_model_len=81920 \
    --sampling.temperature=0.5 \
    --eval.rollouts=8
```

## Architecture

```
modal_eval/
├── cli.py              # Click CLI entry point
├── runner.py           # EvalRunner + ProxyManager orchestration
├── serve.py            # Modal vLLM server definition
├── proxy.py            # FastAPI proxy (handles Modal 150s timeout)
├── models.py           # Pydantic configs: RunConfig, InfraConfig, VLLMConfig, etc.
├── registry.py         # Benchmark YAML loading & merging
├── overrides.py        # CLI override parsing (--namespace.field=value)
├── checkpoint.py       # Resume/checkpoint logic
├── modal_lifecycle.py  # Modal CLI wrapper (deploy/stop/status)
├── push_to_hub.py      # Export results to HuggingFace Hub
├── config/             # Benchmark YAML configs (livecodebench.yaml, etc.)
└── templates/          # Jinja2 chat templates (code.jinja, math.jinja, etc.)
```

## Configuration System

Three layers (later overrides earlier):
1. **Benchmark YAML** (`config/*.yaml`) - Base defaults per benchmark
2. **CLI overrides** (`--vllm.max_model_len=81920`) - Runtime overrides
3. **RunConfig** - Final merged configuration saved to `run_config.json`

## Output Directory Structure

```
outputs/evals/
└── {env_id}--{org}--{model}/           # e.g., livecodebench-modal--allenai--Olmo-3-7B-Think
    └── {run_id}/                       # 8-char hex, e.g., a71da48a
        ├── run_config.json             # Configuration used for this run
        ├── checkpoint.jsonl            # Raw checkpoint data (during eval)
        ├── checkpoint_meta.json        # Checkpoint state for resume
        ├── results.jsonl               # Final results (after completion)
        ├── metadata.json               # Summary metrics (after completion)
        └── logs/
            ├── vf_eval.log             # Evaluation progress, errors
            ├── proxy.log               # Local proxy server logs
            └── modal_vllm.log          # Modal vLLM server logs
```

## Output File Schemas

### run_config.json
```json
{
  "model_id": "org/model-name",
  "model_slug": "org-model-name",
  "benchmark": { "name": "...", "env_id": "...", "sampling": {...} },
  "server_profile": { "gpu": "A100-80GB:2", "max_model_len": 40960, ... },
  "sampling": { "temperature": 0.8, "top_p": 0.95, "max_tokens": 32768 },
  "rollouts": 4,
  "started_at": "2026-01-03T16:58:45.982168+00:00",
  "checkpoint_dir": "outputs/evals/.../run_id",
  "app_name": "mk-org-model-benchmark",
  "modal_url": "https://...modal.run"
}
```

### Chat Template Resolution

**WARNING**: The `chat_template` field in `run_config.json` is only the **fallback** specified in the benchmark config. The actual template used follows this resolution priority:

1. Model's `tokenizer_config.json` `chat_template` field (vLLM uses automatically)
2. Model's `chat_template.jinja` file in HuggingFace repo
3. Benchmark's local template from `templates/{chat_template}.jinja`

To determine the actual template used, check the model's HuggingFace repo (Claude's HF MCP tool can help with this).

See `serve.py:resolve_chat_template()` for implementation details.

### metadata.json (after eval completion)
```json
{
  "env_id": "livecodebench-modal",
  "model": "org/model",
  "num_examples": 454,
  "rollouts_per_example": 4,
  "sampling_args": { "temperature": 0.6, "max_completion_tokens": 32768 },
  "time_ms": 22077930.43,
  "avg_reward": 0.297,
  "avg_metrics": { "passed": 0.297, "num_test_cases": 32.3, "pass_rate": 0.39 }
}
```

### results.jsonl (one JSON object per line)
```json
{
  "example_id": 0,
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "completion": [{"role": "assistant", "content": "..."}],
  "task": "default",
  "reward": 1.0,
  "generation_ms": 258824.18,
  "scoring_ms": 2.32,
  "total_ms": 258826.50,
  "info": { "platform": "atcoder", "difficulty": "easy", "question_id": "..." },
  "passed": 1.0,
  "num_test_cases": 15.0,
  "pass_rate": 1.0,
  "has_error": 0.0
}
```

### checkpoint_meta.json
```json
{
  "checkpoint_version": "1.0",
  "run_id": "d5825468",
  "env_id": "livecodebench-modal",
  "model": "org/model",
  "rollouts_per_example": 2,
  "completed_example_ids": [0, 1, 2, ...],
  "total_examples": 454,
  "completed_rollouts": 908,
  "start_time": "2026-01-03T11:59:31.505165",
  "last_checkpoint_time": "2026-01-03T18:07:29.164556"
}
```

## Available Benchmarks (Default Configs)

| Name | env_id | Default GPU |
|------|--------|-------------|
| livecodebench | livecodebench-modal | 2x A100 |
| aime2025 | aime2025 | 2x A100 |
| math500 | math500 | 1x A100 |
| gsm8k | gsm8k | 1x A100 |
| ifeval | ifeval | 2x A100 |

All configs are overridable via CLI (`--infra.gpu`, `--vllm.*`, `--sampling.*`, `--eval.*`).

## Log File Contents

### vf_eval.log
- Environment loading: `Loading environment: livecodebench-modal`
- Progress: `Processing 454 groups (1816 total rollouts): 50%|...`
- Errors from verifiers framework
- Checkpoint saves

### proxy.log
- Startup: `Uvicorn running on http://0.0.0.0:58402`
- Modal URL: `Forwarding to: https://...modal.run`
- Request logs: `127.0.0.1:... - "POST /v1/chat/completions HTTP/1.1" 200`

### modal_vllm.log
- Modal CLI output from `modal app logs` command
- May contain errors if Modal logs command fails

## Evaluation Pipeline Flow

```
CLI (cli.py) → Build RunConfig → Save run_config.json
     ↓
Deploy vLLM to Modal (serve.py) if needed
     ↓
Start local proxy (proxy.py) on ephemeral port
     ↓
Run vf-eval subprocess (verifiers) pointing to proxy
     ↓
Results written to checkpoint.jsonl → results.jsonl + metadata.json
     ↓
Cleanup: stop proxy, optionally stop Modal app
```

## Key Files for Common Tasks

- **Add new benchmark**: Create `config/mybenchmark.yaml`
- **Modify vLLM settings**: `models.py:VLLMConfig`, override via `--vllm.*`
- **Change evaluation logic**: `runner.py:EvalRunner._run_vf_eval()`
- **Debug proxy issues**: `proxy.py`, check `logs/proxy.log`
- **Resume logic**: `checkpoint.py:CheckpointStore`
