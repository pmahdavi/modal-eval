# modal-eval

LLM evaluation framework on [Modal](https://modal.com) with [vLLM](https://github.com/vllm-project/vllm) serving, using [verifiers](https://github.com/PrimeIntellect-ai/verifiers) from PrimeIntellect.

## Leaderboards

Evaluation results for cross-capability merging of OLMo-3 RL-Zero models:

- **[LiveCodeBench v6](https://huggingface.co/datasets/pmahdavi/livecodebench-merging-leaderboard)** — 454 coding problems, 4 rollouts @ temp=0.8
- **[AIME 2025](https://huggingface.co/datasets/pmahdavi/aime2025-merging-leaderboard)** — 30 competition math problems, 32 rollouts @ temp=0.8

## Quick Start

```bash
# Install
uv sync --all-extras

# Run evaluation
uv run modal-eval run -m allenai/Olmo-3-7B-Think -b livecodebench

# With overrides
uv run modal-eval run -m model -b benchmark \
    --infra.gpu="H100:2" \
    --sampling.temperature=0.6 \
    --eval.rollouts=8
```

## CLI Commands

```bash
modal-eval run -m MODEL -b BENCHMARK    # Deploy + evaluate
modal-eval resume -m MODEL -b BENCHMARK # Resume interrupted run
modal-eval deploy -m MODEL -b BENCHMARK # Deploy only
modal-eval stop --all                   # Stop Modal apps
modal-eval status                       # Show deployed models
modal-eval runs                         # List recent runs
modal-eval benchmarks                   # List available benchmarks
```

## Benchmarks

| Benchmark | Problems | Rollouts | GPUs |
|-----------|----------|----------|------|
| `livecodebench` | 454 | 4 | 2x A100 |
| `aime2025_clean` | 30 | 32 | 2x A100 |
| `math500` | 500 | 4 | 1x A100 |
| `gsm8k` | 1319 | 4 | 1x A100 |
| `ifeval` | 541 | 4 | 2x A100 |

## Architecture

```
Local                              Modal (Cloud)
┌──────────┐    ┌───────┐         ┌─────────────────┐
│  vf-eval │───▶│ Proxy │────────▶│ vLLM (A100/H100)│
└──────────┘    └───────┘         └─────────────────┘
```

The proxy handles Modal's 150s timeout limitation by following redirect chains.

## Output Structure

```
outputs/evals/{benchmark}--{org}--{model}/{run_id}/
├── run_config.json    # Configuration
├── results.jsonl      # Per-example results
├── metadata.json      # Summary metrics
└── logs/              # vf_eval.log, proxy.log, modal_vllm.log
```

## Push to Hub

```bash
# Combine multiple evals into a leaderboard
uv run python -m modal_eval.push_to_hub --combine \
    outputs/evals/benchmark--org--model1/run_id1 \
    outputs/evals/benchmark--org--model2/run_id2 \
    --name username/leaderboard-name

# Append to existing leaderboard
uv run python -m modal_eval.push_to_hub --append \
    outputs/evals/benchmark--org--new_model/run_id \
    --name username/leaderboard-name
```

## Prerequisites

1. Modal account + token: `modal token new`
2. HuggingFace secret in Modal: `modal secret create huggingface HF_TOKEN=your_token`

## License

MIT
