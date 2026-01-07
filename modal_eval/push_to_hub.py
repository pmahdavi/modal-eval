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
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_dataset_name(metadata: dict) -> str:
    """Generate a HuggingFace dataset name from metadata."""
    env_id = metadata.get("env_id", "unknown")
    model = metadata.get("model", "unknown").replace("/", "_")
    num_examples = metadata.get("num_examples", 0)
    rollouts = metadata.get("rollouts_per_example", 0)
    return f"{env_id}_{model}_n{num_examples}_r{rollouts}"


def push_eval_to_hf(
    results_dir: Path,
    dataset_name: str | None = None,
    private: bool = False,
) -> bool:
    """Push eval results from a directory to HuggingFace Hub.

    Args:
        results_dir: Path to the eval results directory containing results.jsonl
        dataset_name: Name for the HuggingFace dataset (auto-generated if not provided)
        private: Whether to make the dataset private

    Returns:
        True if successful, False otherwise
    """
    results_file = results_dir / "results.jsonl"
    metadata_file = results_dir / "metadata.json"

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
        logger.info(f"  avg_reward: {metadata.get('avg_reward'):.4f}")
    else:
        logger.warning(f"Warning: {metadata_file} not found")

    # Generate dataset name if not provided
    if dataset_name is None:
        dataset_name = generate_dataset_name(metadata)
        logger.info(f"Auto-generated dataset name: {dataset_name}")

    # Load results as dataset
    # We load line-by-line to handle mixed types (e.g., timestamps that are sometimes None)
    logger.info(f"Loading results from {results_file}...")
    records = []
    with open(results_file) as f:
        for line in f:
            record = json.loads(line)
            # Normalize info dict - convert None timestamps to empty strings for consistency
            if "info" in record and isinstance(record["info"], dict):
                for key in ["contest_date", "metadata"]:
                    if key in record["info"]:
                        val = record["info"][key]
                        if val is None:
                            record["info"][key] = ""
                        elif not isinstance(val, str):
                            record["info"][key] = str(val)
            records.append(record)
    dataset = Dataset.from_list(records)
    logger.info(f"Loaded {len(dataset)} examples")

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
        help="Path to the eval results directory containing results.jsonl",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name for the HuggingFace dataset (auto-generated if not provided)",
    )
    parser.add_argument(
        "--private",
        "-p",
        action="store_true",
        help="Make the dataset private",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        logger.error(f"Error: Directory {args.results_dir} does not exist")
        return 1

    success = push_eval_to_hf(args.results_dir, args.name, args.private)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
