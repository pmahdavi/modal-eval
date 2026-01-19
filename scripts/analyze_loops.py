#!/usr/bin/env python3
"""
Loop Detection in LLM Rollouts

Standalone script to detect repetitive looping patterns in LLM completions.
A "loop" is a substring (1-400 chars) that repeats consecutively 20+ times.

Significance Criteria (to filter formatting noise):
  - Short patterns (< 10 chars): need > 40% coverage to be considered pathological
  - Long patterns (>= 10 chars): need > 10% coverage to be considered pathological

Dependencies:
    pip install datasets  # Only needed for --dataset mode
    pip install tqdm      # Optional: progress bar

Usage (from project root):
    uv run python scripts/analyze_loops.py --test                              # Run tests
    uv run python scripts/analyze_loops.py --text "hello" * 50                 # Analyze string
    uv run python scripts/analyze_loops.py --file output.txt                   # Analyze file
    uv run python scripts/analyze_loops.py --dataset pmahdavi/livecodebench-merging-leaderboard
    uv run python scripts/analyze_loops.py --dataset pmahdavi/aime2025-merging-leaderboard

Parallel Processing (for large datasets, use PBS job scripts):
    uv run python scripts/analyze_loops.py --dataset <name> --workers 16       # Use 16 CPU cores
    uv run python scripts/analyze_loops.py --dataset <name> -w -1              # Use all available CPUs
    uv run python scripts/analyze_loops.py --dataset <name> -w 16 --batch-size 100  # Custom batch
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Optional


# ============================================================================
# Coverage Thresholds for Significant Loops
# ============================================================================
# Short patterns (< 10 chars) like single spaces are often just formatting
# artifacts (indentation, etc.) and need higher coverage to be considered
# pathological loops. Longer patterns are more likely to be real failures.

SHORT_PATTERN_THRESHOLD = 10  # Pattern length threshold
SHORT_PATTERN_MIN_COVERAGE = 40.0  # Short patterns need > 40% coverage
LONG_PATTERN_MIN_COVERAGE = 10.0  # Long patterns need > 10% coverage


def is_significant_loop(pattern_length: int, loop_percentage: float) -> bool:
    """
    Determine if a loop is significant based on pattern length and coverage.

    Short patterns (< 10 chars) like spaces or single characters are often
    just formatting artifacts. They need > 40% coverage to be considered
    pathological. Longer patterns need > 10% coverage.

    Args:
        pattern_length: Length of the repeating pattern
        loop_percentage: Percentage of text covered by this loop

    Returns:
        True if the loop is significant, False if it's likely noise
    """
    if pattern_length < SHORT_PATTERN_THRESHOLD:
        return loop_percentage > SHORT_PATTERN_MIN_COVERAGE
    else:
        return loop_percentage > LONG_PATTERN_MIN_COVERAGE


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LoopInfo:
    """Information about a single detected loop."""
    start: int
    end: int
    pattern: str
    pattern_length: int
    repetitions: int

    @property
    def total_chars(self) -> int:
        return self.pattern_length * self.repetitions

    def to_dict(self) -> dict:
        # Truncate pattern for readability (keep first 200 chars)
        display_pattern = self.pattern if len(self.pattern) <= 200 else self.pattern[:200] + "..."
        return {
            "start": self.start,
            "end": self.end,
            "pattern": display_pattern,
            "pattern_length": self.pattern_length,
            "repetitions": self.repetitions,
            "total_chars": self.total_chars,
        }


@dataclass
class LoopDetectionResult:
    """Result of loop detection on a single text."""
    has_loop: bool
    worst_pattern: Optional[str] = None
    worst_pattern_length: int = 0
    worst_repetitions: int = 0
    worst_total_chars: int = 0
    total_loop_chars: int = 0
    loop_percentage: float = 0.0
    text_length: int = 0
    all_loops: list[LoopInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_loop": self.has_loop,
            "worst_pattern": self.worst_pattern,
            "worst_pattern_length": self.worst_pattern_length,
            "worst_repetitions": self.worst_repetitions,
            "worst_total_chars": self.worst_total_chars,
            "total_loop_chars": self.total_loop_chars,
            "loop_percentage": self.loop_percentage,
            "text_length": self.text_length,
            "num_loops": len(self.all_loops),
            "loops": [loop.to_dict() for loop in self.all_loops],
        }


# ============================================================================
# Core Algorithm: Rolling Hash Loop Detection
# ============================================================================

def detect_loops(
    text: str,
    min_pattern_len: int = 1,
    max_pattern_len: int = 400,
    min_repetitions: int = 20,
) -> LoopDetectionResult:
    """
    Detect repetitive loops in text using rolling hash algorithm.

    A loop is defined as a substring that repeats consecutively
    at least `min_repetitions` times. Additionally, loops are filtered
    by significance criteria to exclude formatting noise:
    - Short patterns (< 10 chars): need > 40% coverage
    - Long patterns (>= 10 chars): need > 10% coverage

    Algorithm: O(n * L) where L = max_pattern_len
    - Precompute rolling hashes for O(1) substring comparison
    - Scan each position, checking patterns of length 1 to max_pattern_len
    - Track the most severe loop (max pattern_length * repetitions)
    - Filter out insignificant loops based on coverage thresholds

    Args:
        text: The input text to analyze
        min_pattern_len: Minimum pattern length to consider (default: 1)
        max_pattern_len: Maximum pattern length to consider (default: 400)
        min_repetitions: Minimum consecutive repetitions to count as loop (default: 20)

    Returns:
        LoopDetectionResult with detection metrics (has_loop=False if no significant loops)
    """
    n = len(text)

    # Early exit for short texts
    if n < min_pattern_len * min_repetitions:
        return LoopDetectionResult(has_loop=False, text_length=n)

    # Precompute rolling hashes for O(1) substring hash lookup
    # Using polynomial hash: h(s) = sum(s[i] * BASE^(n-1-i)) mod MOD
    BASE = 31
    MOD = (1 << 61) - 1  # Large Mersenne prime for minimal collisions

    prefix_hash = [0] * (n + 1)
    power = [1] * (n + 1)

    for i in range(n):
        prefix_hash[i + 1] = (prefix_hash[i] * BASE + ord(text[i])) % MOD
        power[i + 1] = (power[i] * BASE) % MOD

    def get_hash(l: int, r: int) -> int:
        """Get hash of text[l:r] in O(1)."""
        return (prefix_hash[r] - prefix_hash[l] * power[r - l] % MOD + MOD) % MOD

    # Find all loops
    loops: list[LoopInfo] = []
    i = 0

    while i < n - min_pattern_len * min_repetitions + 1:
        best_loop_at_i: Optional[LoopInfo] = None

        # Check patterns from shortest to longest
        max_L = min(max_pattern_len, (n - i) // min_repetitions)

        for L in range(min_pattern_len, max_L + 1):
            pattern_hash = get_hash(i, i + L)
            repetitions = 1
            j = i + L

            # Count consecutive repetitions using hash comparison
            while j + L <= n:
                if get_hash(j, j + L) == pattern_hash:
                    # Verify to avoid hash collisions (rare but possible)
                    if text[j:j + L] == text[i:i + L]:
                        repetitions += 1
                        j += L
                    else:
                        break
                else:
                    break

            if repetitions >= min_repetitions:
                total_chars = L * repetitions
                # Keep the loop with most total chars (most severe)
                if best_loop_at_i is None or total_chars > best_loop_at_i.total_chars:
                    best_loop_at_i = LoopInfo(
                        start=i,
                        end=j,
                        pattern=text[i:i + L],
                        pattern_length=L,
                        repetitions=repetitions,
                    )

        if best_loop_at_i:
            loops.append(best_loop_at_i)
            i = best_loop_at_i.end  # Skip past this loop
        else:
            i += 1

    # Compute summary metrics
    if not loops:
        return LoopDetectionResult(has_loop=False, text_length=n)

    # Calculate total loop coverage
    total_loop_chars = sum(loop.total_chars for loop in loops)
    loop_percentage = total_loop_chars / n * 100

    # Find most severe loop
    most_severe = max(loops, key=lambda x: x.total_chars)

    # Check if the most severe loop is significant
    # Short patterns need higher coverage to be considered pathological
    if not is_significant_loop(most_severe.pattern_length, loop_percentage):
        return LoopDetectionResult(has_loop=False, text_length=n)

    # Truncate pattern for display (keep first 50 chars)
    display_pattern = most_severe.pattern
    if len(display_pattern) > 50:
        display_pattern = display_pattern[:50] + "..."

    return LoopDetectionResult(
        has_loop=True,
        worst_pattern=display_pattern,
        worst_pattern_length=most_severe.pattern_length,
        worst_repetitions=most_severe.repetitions,
        worst_total_chars=most_severe.total_chars,
        total_loop_chars=total_loop_chars,
        loop_percentage=loop_percentage,
        text_length=n,
        all_loops=loops,
    )


# ============================================================================
# Parallel Processing Helpers
# ============================================================================

def _worker_init():
    """
    Initialize worker process.
    Ignore SIGINT in workers - let the parent handle it.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _sigterm_handler(signum, frame):
    """
    Convert SIGTERM to SystemExit for graceful shutdown.
    PBS sends SIGTERM when using qdel.
    """
    raise SystemExit(f"Received SIGTERM (signal {signum})")


@contextmanager
def _managed_pool(num_workers: int, maxtasksperchild: int = 100, terminate_timeout: float = 5.0):
    """
    Context manager for Pool with proper cleanup on errors and signals.
    """
    pool = None
    clean_exit = False
    try:
        pool = Pool(
            processes=num_workers,
            initializer=_worker_init,
            maxtasksperchild=maxtasksperchild,
        )
        yield pool
        clean_exit = True
    finally:
        if pool is not None:
            if clean_exit:
                pool.close()
                pool.join()
            else:
                pool.terminate()
                deadline = time.time() + terminate_timeout
                workers = getattr(pool, '_pool', None) or []
                for worker in workers:
                    remaining = max(0.1, deadline - time.time())
                    worker.join(timeout=remaining)
                    if worker.is_alive():
                        try:
                            worker.kill()
                            worker.join(timeout=1.0)
                        except (OSError, AttributeError):
                            pass


def _extract_text_from_row(row: dict) -> tuple[str, str, int]:
    """Extract model, text, and example_id from a dataset row."""
    model = row.get("model", "unknown")
    example_id = row.get("example_id", -1)
    completion = row.get("completion", [])
    if not completion:
        return model, "", example_id
    if isinstance(completion, list) and len(completion) > 0:
        text = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
    else:
        text = str(completion)
    return model, text, example_id


def _process_row(row: dict) -> dict:
    """Process a single row - catches exceptions to prevent worker crashes."""
    model, text, example_id = _extract_text_from_row(row)
    if not text:
        return {"model": model, "example_id": example_id, "skipped": True}
    try:
        result = detect_loops(text)
        return {"model": model, "example_id": example_id, "skipped": False, **result.to_dict()}
    except Exception as e:
        return {"model": model, "example_id": example_id, "skipped": True, "error": str(e)}


def _process_batch(batch: list[dict]) -> list[dict]:
    """Process a batch of rows - reduces IPC overhead."""
    return [_process_row(row) for row in batch]


def _update_stats(model_stats: dict, result: dict) -> None:
    """Update per-model statistics with a result."""
    model = result["model"]
    stats = model_stats[model]
    stats["total"] += 1

    if result.get("has_loop"):
        stats["with_loops"] += 1
        stats["total_loop_percentage"] += result.get("loop_percentage", 0)

        if result.get("worst_repetitions", 0) > stats["max_repetitions"]:
            stats["max_repetitions"] = result["worst_repetitions"]
            stats["max_pattern"] = result.get("worst_pattern") or ""

        if len(stats["loop_examples"]) < 3:
            stats["loop_examples"].append({
                "example_id": result.get("example_id"),
                "pattern": result.get("worst_pattern"),
                "repetitions": result.get("worst_repetitions"),
                "percentage": result.get("loop_percentage"),
            })


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_text(text: str, verbose: bool = True) -> LoopDetectionResult:
    """Analyze a single text and optionally print results."""
    result = detect_loops(text)

    if verbose:
        print(f"Text length: {result.text_length:,} chars")
        print(f"Has loop: {result.has_loop}")
        if result.has_loop:
            print(f"Worst pattern: {repr(result.worst_pattern)}")
            print(f"Pattern length: {result.worst_pattern_length}")
            print(f"Repetitions: {result.worst_repetitions:,}")
            print(f"Loop coverage: {result.loop_percentage:.1f}%")
            print(f"Total loops found: {len(result.all_loops)}")

    return result


def analyze_file(filepath: str) -> LoopDetectionResult:
    """Analyze a text file for loops."""
    print(f"Reading file: {filepath}")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return analyze_text(text)


def analyze_dataset(
    dataset_name: str,
    sample_size: Optional[int] = None,
    output_file: str = "loop_analysis_results.json",
    num_workers: int = 1,
    batch_size: int = 50,
):
    """Analyze a HuggingFace dataset for loops with optional parallel processing."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install it with: pip install datasets")
        sys.exit(1)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("(Install tqdm for progress bar: pip install tqdm)")

    print("=" * 60)
    print(f"Loading dataset: {dataset_name}")
    print("=" * 60)

    ds = load_dataset(dataset_name, split="train")

    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))
        print(f"Analyzing {len(ds)} samples...")
    else:
        print(f"Analyzing all {len(ds)} rollouts...")

    model_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "with_loops": 0,
        "total_loop_percentage": 0.0,
        "max_repetitions": 0,
        "max_pattern": "",
        "loop_examples": [],
    })

    all_results = []
    rows = list(ds)
    total_rows = len(rows)
    error_count = 0

    # Install SIGTERM handler for PBS qdel support
    original_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        if num_workers > 1:
            # Parallel processing
            print(f"Using {num_workers} workers with batch size {batch_size}")
            batches = [rows[i:i + batch_size] for i in range(0, total_rows, batch_size)]
            total_batches = len(batches)
            start_time = time.time()

            try:
                with _managed_pool(num_workers, maxtasksperchild=100) as pool:
                    iterator = pool.imap(_process_batch, batches)
                    if has_tqdm:
                        iterator = tqdm(iterator, total=total_batches, desc=f"Analyzing ({num_workers} workers)", unit="batch")

                    for batch_idx, batch_results in enumerate(iterator):
                        # Progress logging for monitoring
                        if (batch_idx + 1) % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                            processed = min((batch_idx + 1) * batch_size, total_rows)
                            pct = (batch_idx + 1) / total_batches * 100
                            elapsed = time.time() - start_time
                            eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
                            print(f"[Progress] {processed}/{total_rows} rows ({pct:.1f}%) | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min", flush=True)

                        for result in batch_results:
                            if result.get("error"):
                                error_count += 1
                                continue
                            if result.get("skipped"):
                                continue
                            all_results.append(result)
                            _update_stats(model_stats, result)

            except (KeyboardInterrupt, SystemExit) as e:
                print(f"\n\nInterrupted ({type(e).__name__}). Saving partial results...")

        else:
            # Sequential processing
            iterator = tqdm(rows, desc="Analyzing") if has_tqdm else rows

            try:
                for idx, row in enumerate(iterator):
                    if not has_tqdm and (idx + 1) % 100 == 0:
                        print(f"  Processed {idx + 1}/{total_rows}...")

                    result = _process_row(row)
                    if result.get("error"):
                        error_count += 1
                        continue
                    if result.get("skipped"):
                        continue
                    all_results.append(result)
                    _update_stats(model_stats, result)

            except (KeyboardInterrupt, SystemExit) as e:
                print(f"\n\nInterrupted ({type(e).__name__}). Saving partial results...")

    finally:
        # Restore original SIGTERM handler
        signal.signal(signal.SIGTERM, original_sigterm)

    # Print summary
    if error_count > 0:
        print(f"\nWarning: {error_count} rows had processing errors")
    print("\n" + "=" * 60)
    print("LOOP DETECTION RESULTS BY MODEL")
    print("=" * 60)

    sorted_models = sorted(
        model_stats.items(),
        key=lambda x: x[1]["with_loops"] / max(x[1]["total"], 1),
        reverse=True,
    )

    print(f"\n{'Model':<55} {'Loop Rate':>10} {'Avg %':>8} {'Max Reps':>10}")
    print("-" * 85)

    for model, stats in sorted_models:
        if stats["total"] == 0:
            continue
        loop_rate = stats["with_loops"] / stats["total"] * 100
        avg_pct = stats["total_loop_percentage"] / max(stats["with_loops"], 1)
        model_display = model if len(model) <= 55 else "..." + model[-52:]
        print(f"{model_display:<55} {loop_rate:>9.1f}% {avg_pct:>7.1f}% {stats['max_repetitions']:>10}")

    # Print example loops
    print("\n" + "=" * 60)
    print("EXAMPLE LOOPS (showing worst per model)")
    print("=" * 60)

    for model, stats in sorted_models:
        if stats["with_loops"] > 0 and stats["loop_examples"]:
            print(f"\n{model}:")
            for ex in stats["loop_examples"][:1]:
                pattern_display = ex["pattern"][:60] + "..." if len(ex["pattern"] or "") > 60 else ex["pattern"]
                print(f"  Pattern: {repr(pattern_display)}")
                print(f"  Repetitions: {ex['repetitions']}, Coverage: {ex['percentage']:.1f}%")

    # Save detailed results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    return all_results


# ============================================================================
# Tests
# ============================================================================

def run_tests():
    """Run synthetic tests to verify correctness."""
    print("=" * 60)
    print("Running tests...")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    def test(name: str, condition: bool, msg: str = ""):
        nonlocal tests_passed, tests_failed
        if condition:
            print(f"  [PASS] {name}")
            tests_passed += 1
        else:
            print(f"  [FAIL] {name}: {msg}")
            tests_failed += 1

    # Test 1: Simple repeated string
    print("\nTest 1: Simple repeated string")
    t1 = "abc" * 25
    r1 = detect_loops(t1)
    test("Detects loop", r1.has_loop)
    test("Correct pattern", r1.worst_pattern == "abc", f"got {repr(r1.worst_pattern)}")
    test("Correct repetitions", r1.worst_repetitions == 25, f"got {r1.worst_repetitions}")

    # Test 2: Below threshold
    print("\nTest 2: Below threshold (15 reps < 20)")
    t2 = "hello" * 15
    r2 = detect_loops(t2)
    test("No loop detected", not r2.has_loop)

    # Test 3: Exactly at threshold
    print("\nTest 3: Exactly at threshold (20 reps)")
    t3 = "xyz" * 20
    r3 = detect_loops(t3)
    test("Detects loop", r3.has_loop)
    test("Correct repetitions", r3.worst_repetitions == 20, f"got {r3.worst_repetitions}")

    # Test 4: Long pattern
    print("\nTest 4: Long pattern")
    t4 = "I need to think carefully. " * 25
    r4 = detect_loops(t4)
    test("Detects loop", r4.has_loop)
    test("Correct repetitions", r4.worst_repetitions == 25, f"got {r4.worst_repetitions}")

    # Test 5: Nested pattern
    print("\nTest 5: Nested pattern (abab * 20 = ab * 40)")
    t5 = "abab" * 20
    r5 = detect_loops(t5)
    test("Detects loop", r5.has_loop)
    test("Total chars correct", r5.worst_total_chars == 80, f"got {r5.worst_total_chars}")

    # Test 6: Loop in middle of text
    print("\nTest 6: Loop embedded in normal text")
    t6 = "Normal start. " + "LOOP" * 30 + " Normal end."
    r6 = detect_loops(t6)
    test("Detects loop", r6.has_loop)
    test("Correct pattern", r6.worst_pattern == "LOOP", f"got {repr(r6.worst_pattern)}")

    # Test 7: Multiple loops
    print("\nTest 7: Multiple distinct loops")
    t7 = "A" * 50 + "BREAK" + "B" * 40
    r7 = detect_loops(t7)
    test("Detects loops", r7.has_loop)
    test("Finds 2 loops", len(r7.all_loops) == 2, f"got {len(r7.all_loops)}")

    # Test 8: No loops
    print("\nTest 8: No loops in normal text")
    t8 = "The quick brown fox jumps over the lazy dog. " * 5
    r8 = detect_loops(t8)
    test("No loop detected", not r8.has_loop)

    # Test 9: Single character loop
    print("\nTest 9: Single character loop")
    t9 = "x" * 100
    r9 = detect_loops(t9)
    test("Detects loop", r9.has_loop)
    test("Correct repetitions", r9.worst_repetitions == 100, f"got {r9.worst_repetitions}")

    # Test 10: Whitespace loop
    print("\nTest 10: Whitespace loop")
    t10 = " " * 50
    r10 = detect_loops(t10)
    test("Detects whitespace loop", r10.has_loop)

    # Summary
    print("\n" + "=" * 60)
    total = tests_passed + tests_failed
    print(f"Results: {tests_passed}/{total} tests passed")
    if tests_failed == 0:
        print("All tests PASSED!")
    else:
        print(f"WARNING: {tests_failed} tests FAILED")
    print("=" * 60)

    return tests_failed == 0


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Detect repetitive loops in LLM completions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_loops.py --test
  python analyze_loops.py --text "hello hello hello ..."
  python analyze_loops.py --file completion.txt
  python analyze_loops.py --dataset pmahdavi/livecodebench-merging-leaderboard
  python analyze_loops.py --dataset pmahdavi/livecodebench-merging-leaderboard --sample 100

Parallel Processing (for HPC clusters):
  python analyze_loops.py --dataset <name> --workers 16
  python analyze_loops.py --dataset <name> -w -1              # Use all CPUs
        """
    )

    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--text", type=str, help="Analyze a text string directly")
    parser.add_argument("--file", type=str, help="Analyze a text file")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset to analyze")
    parser.add_argument("--sample", type=int, help="Number of samples (for --dataset)")
    parser.add_argument("--output", type=str, default="loop_analysis_results.json",
                        help="Output JSON file (default: loop_analysis_results.json)")
    parser.add_argument("--min-reps", type=int, default=20,
                        help="Minimum repetitions to count as loop (default: 20)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel workers (default: 1, -1 for all CPUs)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Rows per batch for parallel processing (default: 50)")

    args = parser.parse_args()

    # No arguments - show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.text:
        result = analyze_text(args.text, verbose=not args.json)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))

    elif args.file:
        result = analyze_file(args.file)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))

    elif args.dataset:
        num_workers = args.workers if args.workers > 0 else cpu_count()
        analyze_dataset(
            args.dataset,
            sample_size=args.sample,
            output_file=args.output,
            num_workers=num_workers,
            batch_size=args.batch_size,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
