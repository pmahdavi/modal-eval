#!/usr/bin/env python3
"""Push existing eval results to HuggingFace Hub.

Usage:
    python -m modal_eval.push_to_hub <results_dir> [--name DATASET_NAME] [--private]

Examples:
    # Push with auto-generated name
    python -m modal_eval.push_to_hub ./outputs/evals/livecodebench--model/abc123

    # Push with custom name
    python -m modal_eval.push_to_hub ./outputs/evals/livecodebench--model/abc123 --name pmahdavi/my-eval

    # Push as private dataset
    python -m modal_eval.push_to_hub ./outputs/evals/livecodebench--model/abc123 --private

    # List all available eval runs
    python -m modal_eval.push_to_hub --list

    # Dry run to see what would be pushed
    python -m modal_eval.push_to_hub ./outputs/evals/livecodebench--model/abc123 --dry-run

    # Combine multiple evals into a leaderboard dataset
    python -m modal_eval.push_to_hub --combine dir1 dir2 dir3 --name pmahdavi/my-leaderboard

    # Append new evals to an existing leaderboard dataset
    python -m modal_eval.push_to_hub --append dir1 dir2 --name pmahdavi/my-leaderboard
"""

import argparse
import io
import json
import logging
import re
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Core columns for the simplified schema
CORE_COLUMNS = {"model", "example_id", "prompt", "completion", "reward"}


def sanitize_for_hf(name: str) -> str:
    """Sanitize a string for use in HuggingFace dataset names."""
    # HF allows: alphanumeric, hyphens, underscores, dots
    # Replace slashes and other chars
    return name.replace("/", "-").replace("--", "-")


def slugify(model_name: str) -> str:
    """Convert model name to a slug for file naming.

    Example: 'allenai/Olmo-3-7B-Think' -> 'olmo-3-7b-think'
    """
    # Remove org prefix, lowercase, replace special chars
    name = model_name.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def load_metadata(eval_dir: Path) -> dict:
    """Load metadata.json from an eval directory."""
    metadata_file = eval_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return {}


def load_run_config(eval_dir: Path) -> dict | None:
    """Load run_config.json from an eval directory if it exists."""
    config_file = eval_dir / "run_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return None


def load_results(eval_dir: Path) -> list[dict]:
    """Load results.jsonl from an eval directory."""
    results_file = eval_dir / "results.jsonl"
    records = []
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def calculate_pass_at_n_metrics(records: list[dict]) -> dict:
    """Calculate pass@N and avg@N metrics from a list of records.

    Groups records by example_id and calculates:
    - pass_n: Fraction of examples where at least one rollout has reward=1.0
    - avg_n: Average reward across all rollouts (equivalent to pass@1)
    - num_problems: Number of unique example_ids
    - rollouts_per_problem: Average number of rollouts per example

    Args:
        records: List of record dicts with example_id and reward fields

    Returns:
        Dict with pass_n, avg_n, num_problems, rollouts_per_problem
    """
    from collections import defaultdict

    if not records:
        return {"pass_n": 0.0, "avg_n": 0.0, "num_problems": 0, "rollouts_per_problem": 0}

    # Group by example_id
    by_example: dict[int, list[float]] = defaultdict(list)
    for r in records:
        example_id = r.get("example_id", 0)
        reward = r.get("reward", 0.0)
        by_example[example_id].append(reward)

    num_problems = len(by_example)

    # Calculate pass@N: fraction where at least one rollout is correct
    num_passed = sum(1 for rewards in by_example.values() if any(r == 1.0 for r in rewards))
    pass_n = num_passed / num_problems if num_problems > 0 else 0.0

    # Calculate avg@N: average reward across all rollouts
    all_rewards = [r.get("reward", 0.0) for r in records]
    avg_n = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    # Calculate average rollouts per problem
    rollouts_per_problem = len(records) // num_problems if num_problems > 0 else 0

    return {
        "pass_n": pass_n,
        "avg_n": avg_n,
        "num_problems": num_problems,
        "rollouts_per_problem": rollouts_per_problem,
    }


def generate_dataset_name(
    metadata: dict, results_dir: Path, hf_username: str | None = None
) -> str:
    """Generate a HuggingFace dataset name from metadata and directory structure.

    Format: {username}/{benchmark}-{model_name}-{run_hash}
    Example: pmahdavi/livecodebench-olmo-3-7b-rl-zero-code-4a479cc7
    """
    # Try to extract info from directory path
    # Expected: outputs/evals/{benchmark}--{model_org}--{model_name}/{run_hash}
    run_hash = results_dir.name[:8]  # Short hash
    parent_name = results_dir.parent.name  # e.g., livecodebench-modal--allenai--Olmo-3-7B-RL-Zero-Code

    # Parse parent directory name
    parts = parent_name.split("--")
    if len(parts) >= 2:
        benchmark = parts[0]
        model_name = "-".join(parts[1:]).lower()  # e.g., allenai-olmo-3-7b-rl-zero-code
    else:
        # Fallback to metadata
        benchmark = metadata.get("env_id", "eval")
        model_name = metadata.get("model", "unknown").replace("/", "-").lower()

    # Clean up the name
    dataset_slug = sanitize_for_hf(f"{benchmark}-{model_name}-{run_hash}")

    # Add username prefix if provided
    if hf_username:
        return f"{hf_username}/{dataset_slug}"
    return dataset_slug


def normalize_record(record: dict) -> dict:
    """Normalize a record for consistent types in the dataset.

    Handles:
    - None values in info dict fields
    - Nested dicts that might have inconsistent types
    """
    # Normalize info dict - convert None values to empty strings for consistency
    if "info" in record and isinstance(record["info"], dict):
        info = record["info"]
        for key in list(info.keys()):
            val = info[key]
            if val is None:
                info[key] = ""
            elif isinstance(val, dict):
                # Convert nested dicts to JSON strings for HF compatibility
                info[key] = json.dumps(val)
            elif not isinstance(val, (str, int, float, bool)):
                info[key] = str(val)

    # Ensure prompt and completion are properly formatted
    # They should be lists of message dicts
    for field in ["prompt", "completion"]:
        if field in record and record[field] is None:
            record[field] = []

    return record


def list_eval_runs(evals_dir: Path) -> list[dict]:
    """List all eval runs in the evals directory with their metadata."""
    runs = []
    if not evals_dir.exists():
        return runs

    for benchmark_dir in sorted(evals_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        for run_dir in sorted(benchmark_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            metadata_file = run_dir / "metadata.json"
            results_file = run_dir / "results.jsonl"

            if not results_file.exists():
                continue

            run_info = {
                "path": str(run_dir),
                "benchmark": benchmark_dir.name,
                "run_hash": run_dir.name,
                "has_metadata": metadata_file.exists(),
                "has_results": results_file.exists(),
            }

            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        meta = json.load(f)
                    run_info["model"] = meta.get("model", "unknown")
                    run_info["num_examples"] = meta.get("num_examples", 0)
                    run_info["rollouts"] = meta.get("rollouts_per_example", 1)
                    run_info["avg_reward"] = meta.get("avg_reward")
                except Exception:
                    pass

            runs.append(run_info)

    return runs


def generate_livecodebench_leaderboard_card(
    model_metrics: list[dict],
    benchmark: str = "LiveCodeBench",
) -> str:
    """Generate a LiveCodeBench-style leaderboard dataset card README.

    Args:
        model_metrics: List of dicts with model, pass1, pass_rate, test_cases, etc.
        benchmark: Name of the benchmark

    Returns:
        Markdown string for the README
    """
    # Sort by pass@1 descending
    sorted_metrics = sorted(model_metrics, key=lambda x: x["pass1"], reverse=True)

    # Build leaderboard table
    table_rows = []
    for m in sorted_metrics:
        model_link = f"[{m['model']}](https://huggingface.co/{m['model']})"
        pass1 = f"{m['pass1']*100:.1f}%"
        pass_rate = f"{m['pass_rate']*100:.1f}%"
        test_cases = f"{m['test_cases']:.1f}"
        table_rows.append(f"| {model_link} | {pass1} | {pass_rate} | {test_cases} |")

    table = "\n".join(table_rows)

    # Build ASCII bar chart
    max_width = 50
    chart_lines = []
    for m in sorted_metrics:
        short_name = m["model"].split("/")[-1][:25].ljust(25)
        bar_width = int(m["pass1"] * max_width)
        bar = "█" * bar_width
        pct = f"{m['pass1']*100:.1f}%"
        chart_lines.append(f"{short_name} {bar} {pct}")

    chart = "\n".join(chart_lines)

    # Count total examples
    total_examples = sum(m.get("num_examples", 0) for m in model_metrics)
    num_models = len(model_metrics)

    readme = f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - code
  - evaluation
  - livecodebench
  - benchmark
  - leaderboard
size_categories:
  - 1K<n<10K
---

# {benchmark} Evaluation Leaderboard

Evaluation results for {num_models} models on {benchmark} with {total_examples} total examples.

## Leaderboard

| Model | Pass@1 | Pass Rate | Avg Test Cases |
|-------|--------|-----------|----------------|
{table}

## Performance Chart

```
{chart}
```

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model identifier (e.g., "allenai/Olmo-3-7B-Think") |
| `example_id` | int | Problem ID from {benchmark} |
| `prompt` | list[dict] | Chat messages input |
| `completion` | list[dict] | Model response |
| `reward` | float | 1.0 if passed all tests, 0.0 otherwise |
| `metadata` | dict | Additional fields: pass_rate, num_test_cases, generation_ms, etc. |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("pmahdavi/livecodebench-leaderboard")

# Filter by model
olmo_results = ds.filter(lambda x: "Olmo" in x["model"])

# Get all passing examples
passing = ds.filter(lambda x: x["reward"] == 1.0)
```

## Run Configurations

See the `configs/` directory for full vLLM and sampling configurations used for each model.

## Evaluation Details

- **Benchmark**: {benchmark}
- **Rollouts per example**: 2
- **Temperature**: 0.6
- **Top-p**: 0.95
- **Max tokens**: 32768+

## Citation

If you use this dataset, please cite the original LiveCodeBench paper:

```bibtex
@article{{jain2024livecodebench,
  title={{LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code}},
  author={{Jain, Naman and others}},
  journal={{arXiv preprint arXiv:2403.07974}},
  year={{2024}}
}}
```
"""
    return readme


def generate_aime_leaderboard_card(
    model_metrics: list[dict],
    num_problems: int = 30,
    rollouts_per_problem: int = 32,
) -> str:
    """Generate an AIME-style leaderboard dataset card README.

    Args:
        model_metrics: List of dicts with model, pass_n, avg_n, num_problems, rollouts
        num_problems: Number of problems in the benchmark
        rollouts_per_problem: Number of rollouts per problem

    Returns:
        Markdown string for the README
    """
    # Sort by pass@N descending, then by avg@N
    sorted_metrics = sorted(
        model_metrics,
        key=lambda x: (x.get("pass_n", 0), x.get("avg_n", 0)),
        reverse=True,
    )

    # Build leaderboard table
    table_rows = []
    for m in sorted_metrics:
        model_link = f"[{m['model']}](https://huggingface.co/{m['model']})"
        pass_n = f"{m.get('pass_n', 0)*100:.1f}%"
        avg_n = f"{m.get('avg_n', 0)*100:.1f}%"
        table_rows.append(f"| {model_link} | {pass_n} | {avg_n} |")

    table = "\n".join(table_rows)

    # Build ASCII bar chart (using pass@N)
    max_width = 50
    chart_lines = []
    for m in sorted_metrics:
        short_name = m["model"].split("/")[-1][:25].ljust(25)
        bar_width = int(m.get("pass_n", 0) * max_width)
        bar = "█" * bar_width
        pct = f"{m.get('pass_n', 0)*100:.1f}%"
        chart_lines.append(f"{short_name} {bar} {pct}")

    chart = "\n".join(chart_lines)

    num_models = len(model_metrics)
    total_rollouts = num_problems * rollouts_per_problem

    readme = f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - math
  - evaluation
  - aime
  - benchmark
  - leaderboard
size_categories:
  - 1K<n<10K
---

# AIME 2025 Evaluation Leaderboard

Evaluation results for {num_models} models on AIME 2025 with {num_problems} problems and {rollouts_per_problem} rollouts per problem.

## Leaderboard

| Model | pass@{rollouts_per_problem} | avg@{rollouts_per_problem} |
|-------|------------|------------|
{table}

**Metrics:**
- **pass@{rollouts_per_problem}**: Fraction of problems where at least 1 of {rollouts_per_problem} rollouts got the correct answer
- **avg@{rollouts_per_problem}**: Average accuracy across all rollouts (equivalent to pass@1 with temperature sampling)

## Performance Chart (pass@{rollouts_per_problem})

```
{chart}
```

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model identifier (e.g., "allenai/Olmo-3-7B-Think") |
| `example_id` | int | Problem ID (0-{num_problems - 1}) |
| `prompt` | list[dict] | Chat messages input |
| `completion` | list[dict] | Model response |
| `reward` | float | 1.0 if correct, 0.0 otherwise |
| `metadata` | dict | Additional fields: answer, generation_ms, etc. |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("pmahdavi/aime2025-leaderboard", split="train")

# Filter by model
qwen_results = ds.filter(lambda x: "Qwen" in x["model"])

# Get all correct answers
correct = ds.filter(lambda x: x["reward"] == 1.0)

# Calculate pass@{rollouts_per_problem} for a model
model_data = ds.filter(lambda x: x["model"] == "allenai/Olmo-3-7B-RL-Zero-Math")
from collections import defaultdict
by_problem = defaultdict(list)
for ex in model_data:
    by_problem[ex["example_id"]].append(ex["reward"])
pass_n = sum(1 for rewards in by_problem.values() if any(r == 1.0 for r in rewards)) / len(by_problem)
print(f"pass@{rollouts_per_problem}: {{pass_n:.1%}}")
```

## Evaluation Details

- **Benchmark**: AIME 2025 (American Invitational Mathematics Examination)
- **Problems**: {num_problems}
- **Rollouts per problem**: {rollouts_per_problem}
- **Temperature**: 0.8
- **Top-p**: 0.95
- **Max tokens**: 28672

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{aime2025eval,
  title={{AIME 2025 Model Evaluation Leaderboard}},
  author={{Mahdavi, Parsa}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/datasets/pmahdavi/aime2025-leaderboard}}}}
}}
```
"""
    return readme


def generate_leaderboard_card(
    model_metrics: list[dict],
    benchmark: str,
    num_problems: int = 30,
    rollouts_per_problem: int = 32,
) -> str:
    """Dispatch to the appropriate leaderboard card generator based on benchmark.

    Args:
        model_metrics: List of dicts with model metrics
        benchmark: Benchmark env_id (e.g., "aime2025", "livecodebench-modal")
        num_problems: Number of problems (for AIME)
        rollouts_per_problem: Number of rollouts per problem (for AIME)

    Returns:
        Markdown string for the README
    """
    if benchmark.startswith("aime"):
        return generate_aime_leaderboard_card(
            model_metrics,
            num_problems=num_problems,
            rollouts_per_problem=rollouts_per_problem,
        )
    elif "livecodebench" in benchmark.lower():
        return generate_livecodebench_leaderboard_card(model_metrics, benchmark)
    else:
        # Default to LiveCodeBench format for unknown benchmarks
        return generate_livecodebench_leaderboard_card(model_metrics, benchmark)


def transform_record_for_combine(record: dict, model: str) -> dict:
    """Transform a record to the simplified schema for combined datasets.

    Keeps: model, example_id, prompt, completion, reward
    Moves everything else to metadata column.
    """
    # Normalize the record first
    record = normalize_record(record)

    # Build the simplified record
    simplified = {
        "model": model,
        "example_id": record.get("example_id", 0),
        "prompt": record.get("prompt", []),
        "completion": record.get("completion", []),
        "reward": record.get("reward", 0.0),
    }

    # Put everything else in metadata
    metadata = {}
    for key, value in record.items():
        if key not in CORE_COLUMNS and key != "metadata":
            metadata[key] = value

    # Include original metadata if present
    if "metadata" in record and isinstance(record["metadata"], dict):
        metadata.update(record["metadata"])

    simplified["metadata"] = metadata
    return simplified


def push_combined_evals_to_hf(
    eval_dirs: list[Path],
    dataset_name: str,
    private: bool = False,
    dry_run: bool = False,
) -> bool:
    """Combine multiple eval runs into a single leaderboard dataset.

    Args:
        eval_dirs: List of paths to eval result directories
        dataset_name: Name for the HuggingFace dataset
        private: Whether to make the dataset private
        dry_run: If True, only show what would be done

    Returns:
        True if successful, False otherwise
    """
    all_records = []
    run_configs: dict[str, dict] = {}  # model_slug -> config
    model_metrics: list[dict] = []  # For leaderboard generation
    benchmark = "LiveCodeBench"
    num_problems = 30
    rollouts_per_problem = 32

    logger.info(f"Combining {len(eval_dirs)} eval runs...")

    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            logger.error(f"Directory not found: {eval_dir}")
            return False

        metadata = load_metadata(eval_dir)
        if not metadata:
            logger.warning(f"No metadata found in {eval_dir}, skipping")
            continue

        model = metadata.get("model", "unknown")
        model_slug = slugify(model)
        benchmark = metadata.get("env_id", benchmark)
        num_problems = metadata.get("num_examples", num_problems)
        rollouts_per_problem = metadata.get("rollouts_per_example", rollouts_per_problem)

        logger.info(f"  Loading {model}...")

        # Collect run config
        config = load_run_config(eval_dir)
        if config:
            run_configs[model_slug] = config

        # Load results first (needed for AIME metrics)
        results = load_results(eval_dir)

        # Collect metrics for leaderboard (both formats for dispatcher)
        avg_metrics = metadata.get("avg_metrics", {})
        metrics = {
            "model": model,
            # LiveCodeBench metrics
            "pass1": metadata.get("avg_reward", 0.0),
            "pass_rate": avg_metrics.get("pass_rate", 0.0),
            "test_cases": avg_metrics.get("num_test_cases", 0.0),
            "num_examples": metadata.get("num_examples", 0),
        }

        # Calculate AIME metrics from results
        if benchmark.startswith("aime"):
            pass_n_metrics = calculate_pass_at_n_metrics(results)
            metrics["pass_n"] = pass_n_metrics["pass_n"]
            metrics["avg_n"] = pass_n_metrics["avg_n"]
        else:
            # For non-AIME, use avg_reward as avg_n
            metrics["pass_n"] = metadata.get("avg_reward", 0.0)
            metrics["avg_n"] = metadata.get("avg_reward", 0.0)

        model_metrics.append(metrics)

        # Transform and collect records
        for record in results:
            simplified = transform_record_for_combine(record, model)
            all_records.append(simplified)

        logger.info(f"    Loaded {len(results)} examples (avg: {metadata.get('avg_reward', 0):.1%})")

    if not all_records:
        logger.error("No records found in any eval directory")
        return False

    logger.info(f"\nTotal: {len(all_records)} examples from {len(model_metrics)} models")

    # Generate README with benchmark-appropriate format
    readme_content = generate_leaderboard_card(
        model_metrics,
        benchmark,
        num_problems=num_problems,
        rollouts_per_problem=rollouts_per_problem,
    )

    if dry_run:
        visibility = "private" if private else "public"
        logger.info(f"\n[DRY RUN] Would push to: {dataset_name} ({visibility})")
        logger.info(f"[DRY RUN] Dataset would have {len(all_records)} rows")
        logger.info(f"[DRY RUN] Would upload {len(run_configs)} config files")
        logger.info(f"\n[DRY RUN] Generated README preview:\n")
        # Show first part of README
        preview_lines = readme_content.split("\n")[:40]
        logger.info("\n".join(preview_lines))
        logger.info("\n... (truncated)")
        return True

    # Create and push dataset
    logger.info(f"\nCreating dataset...")
    dataset = Dataset.from_list(all_records)

    visibility = "private" if private else "public"
    logger.info(f"Pushing dataset to HuggingFace Hub: {dataset_name} ({visibility})")
    dataset.push_to_hub(dataset_name, private=private)

    # Upload README and config files using HfApi
    api = HfApi()

    # Upload README
    logger.info("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
    )

    # Upload config files
    if run_configs:
        logger.info(f"Uploading {len(run_configs)} config files...")
        for model_slug, config in run_configs.items():
            config_json = json.dumps(config, indent=2)
            api.upload_file(
                path_or_fileobj=io.BytesIO(config_json.encode("utf-8")),
                path_in_repo=f"configs/{model_slug}.json",
                repo_id=dataset_name,
                repo_type="dataset",
            )
            logger.info(f"  Uploaded configs/{model_slug}.json")

    logger.info(f"\nSuccessfully pushed to https://huggingface.co/datasets/{dataset_name}")
    return True


def append_evals_to_hf(
    eval_dirs: list[Path],
    dataset_name: str,
    private: bool = False,
    dry_run: bool = False,
) -> bool:
    """Append new eval runs to an existing leaderboard dataset on HuggingFace.

    Args:
        eval_dirs: List of paths to new eval result directories to add
        dataset_name: Name of the existing HuggingFace dataset
        private: Whether the dataset is private
        dry_run: If True, only show what would be done

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Loading existing dataset from HuggingFace: {dataset_name}")

    # Load existing dataset
    try:
        existing_ds = load_dataset(dataset_name, split="train")
        existing_records = list(existing_ds)
        logger.info(f"  Loaded {len(existing_records)} existing records")
    except Exception as e:
        logger.error(f"Failed to load existing dataset: {e}")
        logger.info("Use --combine instead to create a new dataset")
        return False

    # Get existing models to check for duplicates
    existing_models = set()
    for record in existing_records:
        existing_models.add(record.get("model", ""))
    logger.info(f"  Existing models: {sorted(existing_models)}")

    # Load new eval results
    new_records = []
    new_model_metrics: list[dict] = []
    benchmark = "LiveCodeBench"
    num_problems = 30
    rollouts_per_problem = 32

    logger.info(f"\nLoading {len(eval_dirs)} new eval runs...")

    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            logger.error(f"Directory not found: {eval_dir}")
            return False

        metadata = load_metadata(eval_dir)
        if not metadata:
            logger.warning(f"No metadata found in {eval_dir}, skipping")
            continue

        model = metadata.get("model", "unknown")
        benchmark = metadata.get("env_id", benchmark)
        num_problems = metadata.get("num_examples", num_problems)
        rollouts_per_problem = metadata.get("rollouts_per_example", rollouts_per_problem)

        # Check for duplicate
        if model in existing_models:
            logger.warning(f"  Model {model} already exists in dataset, skipping")
            continue

        logger.info(f"  Loading {model}...")

        # Load results first (needed for AIME metrics)
        results = load_results(eval_dir)

        # Collect metrics for leaderboard (both formats)
        avg_metrics = metadata.get("avg_metrics", {})
        metrics = {
            "model": model,
            # LiveCodeBench metrics
            "pass1": metadata.get("avg_reward", 0.0),
            "pass_rate": avg_metrics.get("pass_rate", 0.0),
            "test_cases": avg_metrics.get("num_test_cases", 0.0),
            "num_examples": metadata.get("num_examples", 0),
        }

        # Calculate AIME metrics from results
        if benchmark.startswith("aime"):
            pass_n_metrics = calculate_pass_at_n_metrics(results)
            metrics["pass_n"] = pass_n_metrics["pass_n"]
            metrics["avg_n"] = pass_n_metrics["avg_n"]
        else:
            metrics["pass_n"] = metadata.get("avg_reward", 0.0)
            metrics["avg_n"] = metadata.get("avg_reward", 0.0)

        new_model_metrics.append(metrics)

        # Transform and collect records
        for record in results:
            simplified = transform_record_for_combine(record, model)
            new_records.append(simplified)

        logger.info(f"    Loaded {len(results)} examples (avg: {metadata.get('avg_reward', 0):.1%})")

    if not new_records:
        logger.error("No new records to add (all models may already exist)")
        return False

    # Reconstruct metrics for existing models from records
    existing_model_metrics = []
    for model in existing_models:
        model_records = [r for r in existing_records if r.get("model") == model]
        if model_records:
            # Calculate pass@N and avg@N from records
            pass_n_metrics = calculate_pass_at_n_metrics(model_records)

            # Also collect LiveCodeBench metrics from metadata column
            pass_rates = []
            test_cases = []
            for r in model_records:
                meta = r.get("metadata", {})
                if isinstance(meta, dict):
                    if "pass_rate" in meta:
                        pass_rates.append(meta["pass_rate"])
                    if "num_test_cases" in meta:
                        test_cases.append(meta["num_test_cases"])

            existing_model_metrics.append({
                "model": model,
                # AIME metrics
                "pass_n": pass_n_metrics["pass_n"],
                "avg_n": pass_n_metrics["avg_n"],
                # LiveCodeBench metrics
                "pass1": pass_n_metrics["avg_n"],  # avg_n is equivalent to pass@1
                "pass_rate": sum(pass_rates) / len(pass_rates) if pass_rates else 0,
                "test_cases": sum(test_cases) / len(test_cases) if test_cases else 0,
                "num_examples": len(model_records),
            })

    # Combine all metrics
    all_model_metrics = existing_model_metrics + new_model_metrics
    total_records = len(existing_records) + len(new_records)

    logger.info(f"\nTotal: {total_records} examples from {len(all_model_metrics)} models")
    logger.info(f"  New models added: {[m['model'] for m in new_model_metrics]}")

    # Generate updated README with benchmark-appropriate format
    readme_content = generate_leaderboard_card(
        all_model_metrics,
        benchmark,
        num_problems=num_problems,
        rollouts_per_problem=rollouts_per_problem,
    )

    if dry_run:
        visibility = "private" if private else "public"
        logger.info(f"\n[DRY RUN] Would update: {dataset_name} ({visibility})")
        logger.info(f"[DRY RUN] Dataset would have {total_records} rows (+{len(new_records)} new)")
        logger.info(f"[DRY RUN] New models: {[m['model'] for m in new_model_metrics]}")
        logger.info(f"\n[DRY RUN] Updated leaderboard preview:\n")
        # Show leaderboard section
        for line in readme_content.split("\n"):
            if line.startswith("|") or line.startswith("```") or "%" in line:
                logger.info(line)
        return True

    # Combine datasets
    logger.info(f"\nCombining datasets...")
    new_ds = Dataset.from_list(new_records)
    combined_ds = concatenate_datasets([existing_ds, new_ds])

    visibility = "private" if private else "public"
    logger.info(f"Pushing updated dataset to HuggingFace Hub: {dataset_name} ({visibility})")
    combined_ds.push_to_hub(dataset_name, private=private)

    # Upload updated README
    api = HfApi()
    logger.info("Uploading updated README.md...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
    )

    # Upload config files for new models
    for eval_dir in eval_dirs:
        metadata = load_metadata(eval_dir)
        model = metadata.get("model", "unknown")
        if model in existing_models:
            continue  # Skip existing
        model_slug = slugify(model)
        config = load_run_config(eval_dir)
        if config:
            config_json = json.dumps(config, indent=2)
            api.upload_file(
                path_or_fileobj=io.BytesIO(config_json.encode("utf-8")),
                path_in_repo=f"configs/{model_slug}.json",
                repo_id=dataset_name,
                repo_type="dataset",
            )
            logger.info(f"  Uploaded configs/{model_slug}.json")

    logger.info(f"\nSuccessfully updated https://huggingface.co/datasets/{dataset_name}")
    return True


def push_eval_to_hf(
    results_dir: Path,
    dataset_name: str | None = None,
    private: bool = False,
    dry_run: bool = False,
    hf_username: str | None = None,
) -> bool:
    """Push eval results from a directory to HuggingFace Hub.

    Args:
        results_dir: Path to the eval results directory containing results.jsonl
        dataset_name: Name for the HuggingFace dataset (auto-generated if not provided)
        private: Whether to make the dataset private
        dry_run: If True, only show what would be done without actually pushing
        hf_username: HuggingFace username for auto-generated names

    Returns:
        True if successful, False otherwise
    """
    results_file = results_dir / "results.jsonl"
    metadata_file = results_dir / "metadata.json"
    run_config_file = results_dir / "run_config.json"

    if not results_file.exists():
        logger.error(f"Error: {results_file} not found")
        return False

    # Load metadata
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_file}")
        logger.info(f"  env_id: {metadata.get('env_id')}")
        logger.info(f"  model: {metadata.get('model')}")
        logger.info(f"  num_examples: {metadata.get('num_examples')}")
        logger.info(f"  rollouts_per_example: {metadata.get('rollouts_per_example')}")
        avg_reward = metadata.get("avg_reward")
        if avg_reward is not None:
            logger.info(f"  avg_reward: {avg_reward:.4f}")
    else:
        logger.warning(f"Warning: {metadata_file} not found")

    # Load run config if available
    run_config = {}
    if run_config_file.exists():
        with open(run_config_file) as f:
            run_config = json.load(f)
        logger.info(f"Found run config with {len(run_config)} fields")

    # Generate dataset name if not provided
    if dataset_name is None:
        dataset_name = generate_dataset_name(metadata, results_dir, hf_username)
        logger.info(f"Auto-generated dataset name: {dataset_name}")

    # Load results as dataset
    logger.info(f"Loading results from {results_file}...")
    records = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                record = normalize_record(record)
                records.append(record)

    if not records:
        logger.error("Error: No records found in results file")
        return False

    logger.info(f"Loaded {len(records)} examples")

    # Show sample record structure
    sample_keys = list(records[0].keys())
    logger.info(f"Record fields: {sample_keys}")

    if dry_run:
        visibility = "private" if private else "public"
        logger.info(f"\n[DRY RUN] Would push to: {dataset_name} ({visibility})")
        logger.info(f"[DRY RUN] Dataset would have {len(records)} rows")
        logger.info(f"[DRY RUN] URL would be: https://huggingface.co/datasets/{dataset_name}")
        return True

    # Create dataset
    dataset = Dataset.from_list(records)

    # Add metadata as dataset info
    dataset_info = {
        "description": f"Eval results for {metadata.get('model', 'unknown')} on {metadata.get('env_id', 'unknown')}",
        "run_metadata": metadata,
    }
    if run_config:
        dataset_info["run_config"] = run_config

    # Push to hub
    visibility = "private" if private else "public"
    logger.info(f"Pushing to HuggingFace Hub: {dataset_name} ({visibility})")

    dataset.push_to_hub(dataset_name, private=private)
    logger.info(f"Successfully pushed to https://huggingface.co/datasets/{dataset_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Push existing eval results to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the eval results directory containing results.jsonl",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name for the HuggingFace dataset (required for --combine)",
    )
    parser.add_argument(
        "--private",
        "-p",
        action="store_true",
        help="Make the dataset private",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available eval runs",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show what would be done without actually pushing",
    )
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=Path("outputs/evals"),
        help="Directory containing eval results (default: outputs/evals)",
    )
    parser.add_argument(
        "--username",
        "-u",
        type=str,
        default=None,
        help="HuggingFace username for auto-generated dataset names",
    )
    parser.add_argument(
        "--combine",
        "-c",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="Combine multiple eval directories into a single leaderboard dataset",
    )
    parser.add_argument(
        "--append",
        "-a",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="Append new eval directories to an existing leaderboard dataset",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        runs = list_eval_runs(args.evals_dir)
        if not runs:
            logger.info(f"No eval runs found in {args.evals_dir}")
            return 0

        logger.info(f"Found {len(runs)} eval runs:\n")
        for run in runs:
            model = run.get("model", "unknown")
            n = run.get("num_examples", "?")
            r = run.get("rollouts", "?")
            avg = run.get("avg_reward")
            avg_str = f"{avg:.3f}" if avg is not None else "N/A"
            logger.info(f"  {run['benchmark']}/{run['run_hash']}")
            logger.info(f"    model: {model}")
            logger.info(f"    examples: {n} x {r} rollouts, avg_reward: {avg_str}")
            logger.info(f"    path: {run['path']}\n")
        return 0

    # Combine mode
    if args.combine:
        if not args.name:
            parser.error("--name is required when using --combine")

        success = push_combined_evals_to_hf(
            args.combine,
            args.name,
            args.private,
            args.dry_run,
        )
        return 0 if success else 1

    # Append mode
    if args.append:
        if not args.name:
            parser.error("--name is required when using --append")

        success = append_evals_to_hf(
            args.append,
            args.name,
            args.private,
            args.dry_run,
        )
        return 0 if success else 1

    # Single push mode requires results_dir
    if args.results_dir is None:
        parser.error("results_dir is required unless --list or --combine is specified")

    if not args.results_dir.exists():
        logger.error(f"Error: Directory {args.results_dir} does not exist")
        return 1

    success = push_eval_to_hf(
        args.results_dir,
        args.name,
        args.private,
        args.dry_run,
        args.username,
    )
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
