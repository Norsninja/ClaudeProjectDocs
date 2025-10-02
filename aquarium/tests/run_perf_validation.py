"""
Phase 5 Performance Validation Suite

Tests scaling curve at 143, 500, and 1000 entities.
Includes A/B comparison with legacy batch implementation.
"""

from pathlib import Path
from aquarium.tests.perf_harness import build_perf_scenario


def run_perf_test(entity_count: int, ticks: int = 200, warmup: int = 10):
    """Run performance test at given entity count."""
    print(f"\n{'='*70}")
    print(f"PERFORMANCE TEST: {entity_count} Entities")
    print(f"{'='*70}")

    data_root = Path(__file__).parent.parent.parent / "data"

    # Build scenario
    sim = build_perf_scenario(
        data_root=data_root,
        N=entity_count,
        predator_ratio=0.01,
        ship_count=1,
        seed=42
    )

    # Warmup
    print(f"\n[Warmup] Running {warmup} warmup ticks...")
    for _ in range(warmup):
        sim.tick()

    # Test run with breakdown logging
    print(f"\n[Test] Running {ticks} ticks with breakdown logging every 200 ticks...")
    for i in range(ticks):
        sim.tick()
        sim.print_perf_breakdown(every=200)

    # Summary
    stats = sim.get_tick_stats()
    print(f"\n{'='*70}")
    print(f"RESULTS: {entity_count} Entities")
    print(f"{'='*70}")
    print(f"  Total ticks:      {stats['tick_count']}")
    print(f"  Avg tick time:    {stats['avg_tick_time_ms']:.3f} ms")
    print(f"  Last tick time:   {stats['last_tick_time_ms']:.3f} ms")
    print(f"{'='*70}\n")

    return stats['avg_tick_time_ms']


def run_legacy_comparison():
    """Run A/B comparison with legacy batch implementation."""
    import aquarium.constants as constants

    print(f"\n{'='*70}")
    print(f"A/B COMPARISON: Legacy vs Vectorized Batch (1000 entities)")
    print(f"{'='*70}")

    data_root = Path(__file__).parent.parent.parent / "data"
    entity_count = 1000
    warmup = 10
    test_ticks = 100

    # Test A: Vectorized (default)
    print(f"\n[Test A] Vectorized Batch (USE_LEGACY_BATCH=False)")
    sim_vectorized = build_perf_scenario(data_root, N=entity_count, seed=42)

    for _ in range(warmup):
        sim_vectorized.tick()

    for _ in range(test_ticks):
        sim_vectorized.tick()

    stats_vec = sim_vectorized.get_tick_stats()
    avg_vec = stats_vec['avg_tick_time_ms']
    print(f"  Avg tick time: {avg_vec:.3f} ms")

    # Test B: Legacy
    print(f"\n[Test B] Legacy Batch (USE_LEGACY_BATCH=True)")

    # Temporarily enable legacy
    original_flag = constants.USE_LEGACY_BATCH
    constants.USE_LEGACY_BATCH = True

    # Reimport to pick up flag change
    import importlib
    import aquarium.spatial_queries
    importlib.reload(aquarium.spatial_queries)

    sim_legacy = build_perf_scenario(data_root, N=entity_count, seed=42)

    for _ in range(warmup):
        sim_legacy.tick()

    for _ in range(test_ticks):
        sim_legacy.tick()

    stats_legacy = sim_legacy.get_tick_stats()
    avg_legacy = stats_legacy['avg_tick_time_ms']
    print(f"  Avg tick time: {avg_legacy:.3f} ms")

    # Restore flag
    constants.USE_LEGACY_BATCH = original_flag
    importlib.reload(aquarium.spatial_queries)

    # Summary
    speedup = avg_legacy / avg_vec
    print(f"\n{'='*70}")
    print(f"A/B RESULTS")
    print(f"{'='*70}")
    print(f"  Vectorized: {avg_vec:.3f} ms")
    print(f"  Legacy:     {avg_legacy:.3f} ms")
    print(f"  Speedup:    {speedup:.2f}x")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 5 PERFORMANCE VALIDATION SUITE")
    print("="*70)

    # Run scaling tests
    results = {}
    results[143] = run_perf_test(143, ticks=200)
    results[500] = run_perf_test(500, ticks=200)
    results[1000] = run_perf_test(1000, ticks=200)

    # Scaling curve summary
    print("\n" + "="*70)
    print("SCALING CURVE SUMMARY")
    print("="*70)
    print(f"  143 entities:   {results[143]:.3f} ms/tick")
    print(f"  500 entities:   {results[500]:.3f} ms/tick")
    print(f"  1000 entities:  {results[1000]:.3f} ms/tick")
    print(f"\n  Scaling factor (500/143):  {results[500]/results[143]:.2f}x")
    print(f"  Scaling factor (1000/500): {results[1000]/results[500]:.2f}x")
    print(f"  Scaling factor (1000/143): {results[1000]/results[143]:.2f}x")
    print("="*70)

    # A/B legacy comparison
    run_legacy_comparison()

    print("\n[OK] Performance validation complete!")
