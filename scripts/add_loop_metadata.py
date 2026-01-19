#!/usr/bin/env python3
"""Add loop detection metadata to HuggingFace datasets.

This script merges pre-computed loop analysis results into an existing HF dataset,
adding loop detection fields to each record's metadata and updating the README
leaderboard with a Loop Rate column.

Requirements:
    - Loop analysis results JSON (from analyze_loops.py)
    - Both datasets must have the same row order (verified automatically)

Usage:
    # From project root:
    uv run python scripts/add_loop_metadata.py <dataset_name> --loop-results <path>

    # Examples:
    uv run python scripts/add_loop_metadata.py pmahdavi/livecodebench-merging-leaderboard \\
        --loop-results loop_analysis_results.json --dry-run

    uv run python scripts/add_loop_metadata.py pmahdavi/aime2025-merging-leaderboard \\
        --loop-results aime_loop_analysis_results.json
"""

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download

# Import significance filtering from analyze_loops.py (same directory)
from analyze_loops import (
    SHORT_PATTERN_THRESHOLD,
    SHORT_PATTERN_MIN_COVERAGE,
    LONG_PATTERN_MIN_COVERAGE,
    is_significant_loop,
)


def load_loop_results(path: Path) -> list[dict]:
    """Load loop analysis results from JSON file."""
    print(f"Loading loop results from {path}...")
    with open(path) as f:
        data = json.load(f)
    print(f"  {len(data):,} records loaded")
    return data


def apply_significance_filter(loop_data: list[dict]) -> list[dict]:
    """Apply significance filtering to determine final loop_detected status.

    Uses the criteria from analyze_loops.py:
    - Short patterns (< 10 chars): need > 40% coverage
    - Long patterns (>= 10 chars): need > 10% coverage
    """
    print("\nApplying significance filtering...")
    print(f"  Short patterns (< {SHORT_PATTERN_THRESHOLD} chars): > {SHORT_PATTERN_MIN_COVERAGE}% coverage required")
    print(f"  Long patterns (>= {SHORT_PATTERN_THRESHOLD} chars): > {LONG_PATTERN_MIN_COVERAGE}% coverage required")

    significant_count = 0
    for record in loop_data:
        # Original has_loop from raw detection
        raw_has_loop = record.get("has_loop", False)

        if raw_has_loop:
            # Apply significance filter
            pattern_length = record.get("worst_pattern_length", 0)
            loop_percentage = record.get("loop_percentage", 0.0)
            record["loop_detected"] = is_significant_loop(pattern_length, loop_percentage)
        else:
            record["loop_detected"] = False

        if record["loop_detected"]:
            significant_count += 1

    print(f"  {significant_count:,} significant loops ({100*significant_count/len(loop_data):.1f}% of records)")
    return loop_data


def verify_row_alignment(ds, loop_data: list[dict]) -> bool:
    """Verify that HF dataset and loop results have matching row order.

    Checks that (model, example_id) pairs match for all rows.
    Returns True if aligned, False otherwise.
    """
    print("\nVerifying row alignment...")
    mismatches = 0
    sample_mismatches = []

    for i in range(len(loop_data)):
        hf_key = (ds[i]["model"], ds[i]["example_id"])
        loop_key = (loop_data[i]["model"], loop_data[i]["example_id"])

        if hf_key != loop_key:
            mismatches += 1
            if len(sample_mismatches) < 3:
                sample_mismatches.append((i, hf_key, loop_key))

    if mismatches == 0:
        print(f"  ✓ All {len(loop_data):,} rows aligned by (model, example_id)")
        return True
    else:
        print(f"  ✗ {mismatches:,} rows misaligned!")
        for i, hf_key, loop_key in sample_mismatches:
            print(f"    Row {i}: HF={hf_key}, Loop={loop_key}")
        return False


def merge_loop_into_record(hf_record: dict, loop_record: dict) -> dict:
    """Merge loop fields into HF record's metadata."""
    # Get or create metadata dict
    metadata = hf_record.get("metadata", {})
    if metadata is None:
        metadata = {}

    # Add loop fields - include the whole loops array
    metadata["loop_detected"] = loop_record.get("loop_detected", False)
    metadata["loop_percentage"] = loop_record.get("loop_percentage", 0.0)
    metadata["loops"] = loop_record.get("loops", [])

    hf_record["metadata"] = metadata
    return hf_record


def calculate_loop_stats(records: list[dict]) -> dict[str, dict]:
    """Calculate loop statistics per model."""
    stats = defaultdict(lambda: {"total": 0, "with_loops": 0})

    for record in records:
        model = record.get("model", "unknown")
        stats[model]["total"] += 1

        metadata = record.get("metadata", {})
        if metadata and metadata.get("loop_detected", False):
            stats[model]["with_loops"] += 1

    # Calculate rates
    for model, s in stats.items():
        s["loop_rate"] = s["with_loops"] / max(s["total"], 1)

    return dict(stats)


def update_readme_with_loop_rate(readme: str, model_stats: dict[str, dict]) -> str:
    """Parse README and add Loop Rate column to leaderboard table.

    Expects a markdown table with:
    - Header containing "| Model |"
    - Model links in format [name](https://huggingface.co/org/model)
    """
    lines = readme.split("\n")
    new_lines = []
    in_table = False

    for line in lines:
        # Detect table header (line with | Model |)
        if "| Model |" in line and "Loop Rate" not in line:
            # Add Loop Rate column to header: | ... | -> | ... | Loop Rate |
            line = line.rstrip() + " Loop Rate |"
            in_table = True
            new_lines.append(line)
            continue

        # Detect table separator (|---|---|...)
        if in_table and line.startswith("|") and "---" in line:
            # Add separator for new column
            line = line.rstrip() + "-----------|"
            new_lines.append(line)
            continue

        # Detect table data rows
        if in_table and line.startswith("|") and "---" not in line:
            # Extract model name from the row
            # Format: | [model](url) | metric1 | metric2 | ... |
            match = re.search(r'\[([^\]]+)\]\(https://huggingface\.co/([^\)]+)\)', line)
            if match:
                model_name = match.group(2)  # e.g., "allenai/Olmo-3-1025-7B"

                # Get loop rate for this model
                if model_name in model_stats:
                    loop_rate = model_stats[model_name]["loop_rate"]
                    loop_str = f"{loop_rate*100:.1f}%"
                else:
                    loop_str = "-"

                # Add Loop Rate column: | ... | -> | ... | value |
                line = line.rstrip() + f" {loop_str} |"

            new_lines.append(line)
            continue

        # End of table detection (empty line or non-table content)
        if in_table and (not line.strip() or (not line.startswith("|") and line.strip())):
            in_table = False

        new_lines.append(line)

    return "\n".join(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Add loop detection metadata to HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/add_loop_metadata.py pmahdavi/livecodebench-merging-leaderboard \\
      --loop-results loop_analysis_results.json --dry-run

  uv run python scripts/add_loop_metadata.py pmahdavi/aime2025-merging-leaderboard \\
      --loop-results aime_loop_analysis_results.json
        """
    )
    parser.add_argument(
        "dataset_name",
        help="HuggingFace dataset name (e.g., pmahdavi/livecodebench-merging-leaderboard)"
    )
    parser.add_argument(
        "--loop-results",
        type=Path,
        required=True,
        help="Path to loop analysis results JSON (from analyze_loops.py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without uploading"
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip README update"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip row alignment verification (use with caution)"
    )
    args = parser.parse_args()

    # Load loop results
    if not args.loop_results.exists():
        print(f"Error: Loop results file not found: {args.loop_results}")
        sys.exit(1)

    loop_data = load_loop_results(args.loop_results)

    # Apply significance filtering
    loop_data = apply_significance_filter(loop_data)

    # Load HF dataset
    print(f"\nLoading HF dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name, split="train")
    print(f"  {len(ds):,} records loaded")

    # Verify sizes match
    if len(ds) != len(loop_data):
        print(f"Error: Size mismatch - HF dataset has {len(ds)} rows, loop results has {len(loop_data)}")
        sys.exit(1)

    # Verify row alignment
    if not args.skip_verification:
        if not verify_row_alignment(ds, loop_data):
            print("\nError: Row alignment mismatch. Ensure loop results were generated from the same dataset.")
            print("Use --skip-verification to bypass this check (not recommended).")
            sys.exit(1)

    # Merge loop data by row index
    print("\nMerging loop data into records...")
    enhanced_records = []
    for i, (hf_record, loop_record) in enumerate(zip(ds, loop_data)):
        # Convert to dict if needed
        record = dict(hf_record)
        record = merge_loop_into_record(record, loop_record)
        enhanced_records.append(record)

    print(f"  {len(enhanced_records):,} records merged")

    # Calculate statistics
    model_stats = calculate_loop_stats(enhanced_records)

    print("\nLoop Statistics (after filtering):")
    for model in sorted(model_stats.keys()):
        s = model_stats[model]
        print(f"  {model}: {s['loop_rate']*100:.1f}% ({s['with_loops']}/{s['total']})")

    # Update README if requested
    readme_content = None
    if not args.no_readme:
        print("\nDownloading current README...")
        try:
            readme_path = hf_hub_download(
                repo_id=args.dataset_name,
                filename="README.md",
                repo_type="dataset"
            )
            with open(readme_path) as f:
                readme_content = f.read()

            readme_content = update_readme_with_loop_rate(readme_content, model_stats)
            print("  README updated with Loop Rate column")
        except Exception as e:
            print(f"  Warning: Could not update README: {e}")
            readme_content = None

    if args.dry_run:
        print("\n" + "=" * 60)
        print("[DRY RUN] Would upload enhanced dataset and README")
        print("=" * 60)

        if readme_content:
            # Show table preview
            print("\nUpdated README table preview:")
            in_table = False
            for line in readme_content.split("\n"):
                if "| Model |" in line:
                    in_table = True
                if in_table:
                    print(line)
                if in_table and not line.strip():
                    break
        return

    # Upload enhanced dataset
    print("\nCreating enhanced dataset...")
    new_ds = Dataset.from_list(enhanced_records)

    print(f"Pushing to HuggingFace Hub: {args.dataset_name}")
    new_ds.push_to_hub(args.dataset_name)

    # Upload README
    if readme_content:
        print("Uploading updated README.md...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=args.dataset_name,
            repo_type="dataset",
        )

    print(f"\nSuccessfully updated https://huggingface.co/datasets/{args.dataset_name}")


if __name__ == "__main__":
    main()
