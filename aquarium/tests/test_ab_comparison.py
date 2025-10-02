"""
Test A/B Comparison: USE_CKDTREE False vs True

Verifies that cKDTree backend produces identical results to O(n) fallback
and measures performance improvement.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aquarium.simulation import AquariumSimulation
from aquarium import constants


def test_ab_determinism(entity_count=100):
    """
    Compare USE_CKDTREE=False vs True for identical results.

    Args:
        entity_count: Number of entities to spawn for test

    Expects:
    - Identical positions (within floating point tolerance)
    - Identical active_behavior_id values
    - Significant performance improvement with cKDTree at high entity counts
    """
    print("=" * 60)
    print(f"A/B Comparison Test: O(n) vs cKDTree ({entity_count} entities)")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Run 1: O(n) fallback
    print("\n[RUN 1] USE_CKDTREE=False (O(n) fallback)")
    print("-" * 60)

    # Temporarily disable cKDTree
    original_setting = constants.USE_CKDTREE
    constants.USE_CKDTREE = False

    sim1 = AquariumSimulation(
        data_root=data_root,
        spawn_limit=entity_count,
        spawn_only_species=['sp-001-drifter']
    )

    # Add ship for behavior variation
    sim1.update_ship(np.array([200.0, -100.0, 0.0], dtype=np.float64))

    # Warm-up
    for _ in range(10):
        sim1.tick(evaluate_behaviors=True)

    # Benchmark 100 ticks
    start_time = time.perf_counter()
    for _ in range(100):
        sim1.tick(evaluate_behaviors=True)
    elapsed_on = time.perf_counter() - start_time

    snapshot1 = sim1.get_snapshot()
    stats1 = sim1.get_tick_stats()

    print(f"Entities: {len(sim1.entities)}")
    print(f"Total time: {elapsed_on:.3f}s")
    print(f"Average tick: {stats1['avg_tick_time_ms']:.3f}ms")

    # Run 2: cKDTree
    print("\n[RUN 2] USE_CKDTREE=True (cKDTree backend)")
    print("-" * 60)

    # Enable cKDTree
    constants.USE_CKDTREE = True

    sim2 = AquariumSimulation(
        data_root=data_root,
        spawn_limit=entity_count,
        spawn_only_species=['sp-001-drifter']
    )

    sim2.update_ship(np.array([200.0, -100.0, 0.0], dtype=np.float64))

    # Warm-up
    for _ in range(10):
        sim2.tick(evaluate_behaviors=True)

    # Benchmark 100 ticks
    start_time = time.perf_counter()
    for _ in range(100):
        sim2.tick(evaluate_behaviors=True)
    elapsed_ckd = time.perf_counter() - start_time

    snapshot2 = sim2.get_snapshot()
    stats2 = sim2.get_tick_stats()

    print(f"Entities: {len(sim2.entities)}")
    print(f"Total time: {elapsed_ckd:.3f}s")
    print(f"Average tick: {stats2['avg_tick_time_ms']:.3f}ms")

    # Restore original setting
    constants.USE_CKDTREE = original_setting

    # Compare results
    print("\n[COMPARISON]")
    print("-" * 60)

    entities1 = sorted(snapshot1['entities'], key=lambda e: e['instance_id'])
    entities2 = sorted(snapshot2['entities'], key=lambda e: e['instance_id'])

    assert len(entities1) == len(entities2), "Entity count mismatch"

    max_pos_diff = 0.0
    max_vel_diff = 0.0
    behavior_mismatches = 0

    for e1, e2 in zip(entities1, entities2):
        # Compare positions
        pos1 = np.array(e1['position'])
        pos2 = np.array(e2['position'])
        pos_diff = np.linalg.norm(pos1 - pos2)
        max_pos_diff = max(max_pos_diff, pos_diff)

        # Compare velocities
        vel1 = np.array(e1['velocity'])
        vel2 = np.array(e2['velocity'])
        vel_diff = np.linalg.norm(vel1 - vel2)
        max_vel_diff = max(max_vel_diff, vel_diff)

        # Compare behaviors
        if e1['active_behavior_id'] != e2['active_behavior_id']:
            behavior_mismatches += 1

        # Assert tight tolerance
        assert pos_diff < 1e-9, f"Position mismatch: {pos_diff}"
        assert vel_diff < 1e-9, f"Velocity mismatch: {vel_diff}"
        assert e1['active_behavior_id'] == e2['active_behavior_id'], \
            f"Behavior mismatch: {e1['active_behavior_id']} vs {e2['active_behavior_id']}"

    # Performance comparison
    speedup = elapsed_on / elapsed_ckd
    improvement_pct = (elapsed_on - elapsed_ckd) / elapsed_on * 100

    print(f"Max position difference: {max_pos_diff:.12f}m")
    print(f"Max velocity difference: {max_vel_diff:.12f}m/s")
    print(f"Behavior mismatches: {behavior_mismatches}")
    print()
    print(f"O(n) time: {elapsed_on:.3f}s ({stats1['avg_tick_time_ms']:.3f}ms/tick)")
    print(f"cKDTree time: {elapsed_ckd:.3f}s ({stats2['avg_tick_time_ms']:.3f}ms/tick)")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {improvement_pct:.1f}%")

    # Verify meaningful speedup (should be at least 2x for 100 entities)
    if speedup >= 2.0:
        print(f"\n[OK] Significant speedup achieved ({speedup:.2f}x)")
    elif speedup >= 1.1:
        print(f"\n[WARN] Modest speedup ({speedup:.2f}x), expected >2x for cKDTree")
    else:
        print(f"\n[WARN] Minimal speedup ({speedup:.2f}x), cKDTree may not be effective")

    print("\n[OK] Determinism verified - results identical")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 4b A/B Test: Verify cKDTree Correctness & Performance")
    print("=" * 60)
    print()

    # Test at multiple scales to find crossover point
    test_counts = [100, 500, 1000]

    try:
        for count in test_counts:
            test_ab_determinism(entity_count=count)
            print()  # Blank line between tests

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("[PASS] All A/B comparison tests passed!")
    print("=" * 60)
