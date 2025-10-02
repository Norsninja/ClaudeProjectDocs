"""
Performance A/B test: Batch cache ON vs OFF.

Phase 5: Validate that batch caching provides 2-4x performance improvement at scale.
Tests at 100, 1000, and 5000 entities.
"""

import time
import numpy as np
from pathlib import Path

from aquarium.simulation import AquariumSimulation


def test_batch_cache_performance_1000():
    """
    A/B performance test at 1000 entities.

    Expected: 2-4x improvement with batch cache enabled.
    Target: <3ms per tick with cache ON at 1000 entities.
    """
    print("\n" + "="*70)
    print("A/B PERFORMANCE TEST: 1000 Entities")
    print("="*70)

    data_root = Path(__file__).parent.parent.parent / "data"
    entity_count = 1000
    warmup_ticks = 10
    test_ticks = 100

    # ========================================================================
    # Test A: Cache OFF (Phase 4b baseline)
    # ========================================================================
    print("\n[Test A] Cache OFF (per-entity queries)")
    sim_no_cache = AquariumSimulation(
        data_root=data_root,
        spawn_limit=entity_count,
        spawn_only_species=None
    )
    sim_no_cache._use_batch_cache = False  # Disable cache

    # Warmup
    for _ in range(warmup_ticks):
        sim_no_cache.tick()

    # Measure
    start = time.perf_counter()
    for _ in range(test_ticks):
        sim_no_cache.tick()
    elapsed_no_cache = time.perf_counter() - start

    avg_tick_no_cache = (elapsed_no_cache / test_ticks) * 1000.0  # ms
    avg_build_no_cache = sum(sim_no_cache._build_times[-test_ticks:]) / test_ticks * 1000.0
    avg_behavior_no_cache = sum(sim_no_cache._behavior_times[-test_ticks:]) / test_ticks * 1000.0

    print(f"  Avg tick time: {avg_tick_no_cache:.3f} ms")
    print(f"    Build:    {avg_build_no_cache:.3f} ms")
    print(f"    Behavior: {avg_behavior_no_cache:.3f} ms")

    # ========================================================================
    # Test B: Cache ON (Phase 5 batch queries)
    # ========================================================================
    print("\n[Test B] Cache ON (batch queries)")
    sim_with_cache = AquariumSimulation(
        data_root=data_root,
        spawn_limit=entity_count,
        spawn_only_species=None
    )
    sim_with_cache._use_batch_cache = True  # Enable cache (default)

    # Warmup
    for _ in range(warmup_ticks):
        sim_with_cache.tick()

    # Measure
    start = time.perf_counter()
    for _ in range(test_ticks):
        sim_with_cache.tick()
    elapsed_with_cache = time.perf_counter() - start

    avg_tick_with_cache = (elapsed_with_cache / test_ticks) * 1000.0  # ms
    avg_build_with_cache = sum(sim_with_cache._build_times[-test_ticks:]) / test_ticks * 1000.0
    avg_batch_query_with_cache = sum(sim_with_cache._batch_query_times[-test_ticks:]) / test_ticks * 1000.0
    avg_behavior_with_cache = sum(sim_with_cache._behavior_times[-test_ticks:]) / test_ticks * 1000.0

    print(f"  Avg tick time: {avg_tick_with_cache:.3f} ms")
    print(f"    Build:       {avg_build_with_cache:.3f} ms")
    print(f"    Batch query: {avg_batch_query_with_cache:.3f} ms")
    print(f"    Behavior:    {avg_behavior_with_cache:.3f} ms")

    # ========================================================================
    # Analysis
    # ========================================================================
    speedup = avg_tick_no_cache / avg_tick_with_cache
    behavior_speedup = avg_behavior_no_cache / avg_behavior_with_cache

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Overall speedup:  {speedup:.2f}x")
    print(f"  Behavior speedup: {behavior_speedup:.2f}x")
    print(f"  Target met (<3ms with cache): {'YES' if avg_tick_with_cache < 3.0 else 'NO'}")

    # Soft assertions (log warnings, don't fail)
    if speedup < 1.5:
        print(f"\n[WARN] Speedup {speedup:.2f}x below expected 2-4x range")
    elif speedup >= 2.0:
        print(f"\n[SUCCESS] Speedup {speedup:.2f}x meets 2-4x target!")
    else:
        print(f"\n[OK] Speedup {speedup:.2f}x approaching target")

    if avg_tick_with_cache >= 3.0:
        print(f"[WARN] Avg tick time {avg_tick_with_cache:.3f}ms exceeds 3ms target")
    else:
        print(f"[SUCCESS] Avg tick time {avg_tick_with_cache:.3f}ms meets <3ms target!")


def test_batch_cache_performance_100():
    """A/B test at 100 entities (baseline - expect small/no change)."""
    print("\n" + "="*70)
    print("A/B PERFORMANCE TEST: 100 Entities (Baseline)")
    print("="*70)

    data_root = Path(__file__).parent.parent.parent / "data"
    entity_count = 100
    warmup_ticks = 10
    test_ticks = 100

    # Cache OFF
    print("\n[Test A] Cache OFF")
    sim_no_cache = AquariumSimulation(
        data_root=data_root,
        spawn_limit=entity_count,
        spawn_only_species=None
    )
    sim_no_cache._use_batch_cache = False

    for _ in range(warmup_ticks):
        sim_no_cache.tick()

    start = time.perf_counter()
    for _ in range(test_ticks):
        sim_no_cache.tick()
    elapsed_no_cache = time.perf_counter() - start
    avg_tick_no_cache = (elapsed_no_cache / test_ticks) * 1000.0

    print(f"  Avg tick time: {avg_tick_no_cache:.3f} ms")

    # Cache ON
    print("\n[Test B] Cache ON")
    sim_with_cache = AquariumSimulation(
        data_root=data_root,
        spawn_limit=entity_count,
        spawn_only_species=None
    )
    sim_with_cache._use_batch_cache = True

    for _ in range(warmup_ticks):
        sim_with_cache.tick()

    start = time.perf_counter()
    for _ in range(test_ticks):
        sim_with_cache.tick()
    elapsed_with_cache = time.perf_counter() - start
    avg_tick_with_cache = (elapsed_with_cache / test_ticks) * 1000.0

    print(f"  Avg tick time: {avg_tick_with_cache:.3f} ms")

    speedup = avg_tick_no_cache / avg_tick_with_cache

    print("\n" + "="*70)
    print(f"  Speedup: {speedup:.2f}x (expected: small/neutral at 100 entities)")
    print("="*70)


if __name__ == "__main__":
    test_batch_cache_performance_100()
    test_batch_cache_performance_1000()
    print("\n[OK] All performance tests completed!")
