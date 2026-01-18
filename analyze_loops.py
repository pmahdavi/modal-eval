#!/usr/bin/env python3
"""
Loop Detection in LLM Rollouts

Standalone script to detect repetitive looping patterns in LLM completions.
A "loop" is a substring (1-400 chars) that repeats consecutively 20+ times.

Dependencies:
    pip install datasets  # Only needed for --dataset mode

Usage:
    python analyze_loops.py --test                              # Run tests
    python analyze_loops.py --text "hello" * 50                 # Analyze string
    python analyze_loops.py --file output.txt                   # Analyze file
    python analyze_loops.py --dataset pmahdavi/livecodebench-merging-leaderboard
    python analyze_loops.py --dataset pmahdavi/livecodebench-merging-leaderboard --sample 100
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


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
    at least `min_repetitions` times.

    Algorithm: O(n * L) where L = max_pattern_len
    - Precompute rolling hashes for O(1) substring comparison
    - Scan each position, checking patterns of length 1 to max_pattern_len
    - Track the most severe loop (max pattern_length * repetitions)

    Args:
        text: The input text to analyze
        min_pattern_len: Minimum pattern length to consider (default: 1)
        max_pattern_len: Maximum pattern length to consider (default: 200)
        min_repetitions: Minimum consecutive repetitions to count as loop (default: 20)

    Returns:
        LoopDetectionResult with detection metrics
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

    # Find most severe loop
    most_severe = max(loops, key=lambda x: x.total_chars)
    total_loop_chars = sum(loop.total_chars for loop in loops)

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
        loop_percentage=total_loop_chars / n * 100,
        text_length=n,
        all_loops=loops,
    )


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
):
    """Analyze a HuggingFace dataset for loops."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install it with: pip install datasets")
        sys.exit(1)

    # Optional: try to import tqdm for progress bar
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

    # Per-model statistics
    model_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "with_loops": 0,
        "total_loop_percentage": 0.0,
        "max_repetitions": 0,
        "max_pattern": "",
        "loop_examples": [],
    })

    all_results = []

    # Iterate with or without progress bar
    iterator = tqdm(ds, desc="Analyzing") if has_tqdm else ds
    count = 0

    for row in iterator:
        count += 1
        if not has_tqdm and count % 100 == 0:
            print(f"  Processed {count}/{len(ds)}...")

        model = row.get("model", "unknown")
        completion = row.get("completion", [])

        if not completion:
            continue

        # Extract text from completion
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
        else:
            text = str(completion)

        if not text:
            continue

        result = detect_loops(text)
        all_results.append({
            "model": model,
            "example_id": row.get("example_id", -1),
            **result.to_dict(),
        })

        stats = model_stats[model]
        stats["total"] += 1

        if result.has_loop:
            stats["with_loops"] += 1
            stats["total_loop_percentage"] += result.loop_percentage

            if result.worst_repetitions > stats["max_repetitions"]:
                stats["max_repetitions"] = result.worst_repetitions
                stats["max_pattern"] = result.worst_pattern or ""

            if len(stats["loop_examples"]) < 3:
                stats["loop_examples"].append({
                    "example_id": row.get("example_id"),
                    "pattern": result.worst_pattern,
                    "repetitions": result.worst_repetitions,
                    "percentage": result.loop_percentage,
                })

    # Print summary
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
        analyze_dataset(args.dataset, sample_size=args.sample, output_file=args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
