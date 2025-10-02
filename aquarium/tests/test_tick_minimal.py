"""
Test Phase 2: Minimal tick loop with movement and bounds reflection.

Verifies:
- Entities spawn correctly
- Positions update each tick
- Entities remain within biome bounds (reflection works)
- Determinism (same seed = identical results)
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aquarium.simulation import AquariumSimulation
from aquarium.spatial import distance_to_sphere


def test_movement():
    """Test that entities move over time"""
    print("=" * 60)
    print("Test 1: Entity Movement")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Spawn 10 Drifters only
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=10,
        spawn_only_species=['sp-001-drifter']
    )

    print(f"Spawned {len(sim.entities)} entities")
    assert len(sim.entities) == 10, f"Expected 10 entities, got {len(sim.entities)}"

    # Record initial positions
    initial_positions = [e.position.copy() for e in sim.entities]

    # Tick 100 times (without behavior evaluation for Phase 2)
    print("Ticking simulation 100 times...")
    for i in range(100):
        sim.tick(evaluate_behaviors=False)

        # Print progress every 25 ticks
        if (i + 1) % 25 == 0:
            sim.print_tick_summary()

    # Verify positions changed
    print("\nVerifying positions changed...")
    for i, entity in enumerate(sim.entities):
        distance_moved = np.linalg.norm(entity.position - initial_positions[i])
        print(f"  {entity.instance_id}: moved {distance_moved:.2f} m")

        # With cruise speed = 2.0 m/s (20% of 10.0 m/s) and dt=1.0s,
        # minimum expected movement = 100 ticks * 2.0 m/s = 200m
        # (unless entity hit bounds and reflected)
        assert distance_moved > 10.0, f"Entity {entity.instance_id} barely moved ({distance_moved:.2f}m)"

    print("[OK] All entities moved significantly\n")


def test_bounds_reflection():
    """Test that entities stay within biome bounds"""
    print("=" * 60)
    print("Test 2: Bounds Reflection")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=10,
        spawn_only_species=['sp-001-drifter']
    )

    # Get biome bounds
    biome = list(sim.biomes.values())[0]
    center = np.array(biome.bounds['center'], dtype=np.float64)
    radius = biome.bounds['radius']

    print(f"Biome: {biome.name}, center={center.tolist()}, radius={radius}m")

    # Tick 100 times (without behavior evaluation for Phase 2)
    print("Ticking simulation 100 times...")
    for i in range(100):
        sim.tick(evaluate_behaviors=False)

    # Verify all entities within bounds
    print("\nVerifying entities within bounds...")
    for entity in sim.entities:
        dist = distance_to_sphere(entity.position, center, radius)
        offset = entity.position - center
        distance_from_center = np.linalg.norm(offset)

        print(f"  {entity.instance_id}: distance from center = {distance_from_center:.2f}m (signed dist = {dist:.2f}m)")

        # Allow small epsilon for floating point error
        assert dist <= 0.01, f"Entity {entity.instance_id} outside bounds (dist={dist:.2f}m)"

    print("[OK] All entities within biome bounds\n")


def test_determinism():
    """Test that same seed produces identical results"""
    print("=" * 60)
    print("Test 3: Determinism")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Run simulation 1
    print("Running simulation 1 (seed=12345)...")
    sim1 = AquariumSimulation(
        data_root=data_root,
        spawn_limit=10,
        spawn_only_species=['sp-001-drifter']
    )

    for _ in range(100):
        sim1.tick(evaluate_behaviors=False)

    snapshot1 = sim1.get_snapshot()

    # Run simulation 2 with same seed
    print("Running simulation 2 (seed=12345)...")
    sim2 = AquariumSimulation(
        data_root=data_root,
        spawn_limit=10,
        spawn_only_species=['sp-001-drifter']
    )

    for _ in range(100):
        sim2.tick(evaluate_behaviors=False)

    snapshot2 = sim2.get_snapshot()

    # Compare snapshots
    print("\nComparing snapshots...")
    assert snapshot1['tick_count'] == snapshot2['tick_count'], "Tick counts differ"
    assert snapshot1['entity_count'] == snapshot2['entity_count'], "Entity counts differ"

    # Compare entity positions
    entities1 = sorted(snapshot1['entities'], key=lambda e: e['instance_id'])
    entities2 = sorted(snapshot2['entities'], key=lambda e: e['instance_id'])

    for e1, e2 in zip(entities1, entities2):
        assert e1['instance_id'] == e2['instance_id'], "Instance IDs differ"

        pos1 = np.array(e1['position'])
        pos2 = np.array(e2['position'])
        pos_diff = np.linalg.norm(pos1 - pos2)

        vel1 = np.array(e1['velocity'])
        vel2 = np.array(e2['velocity'])
        vel_diff = np.linalg.norm(vel1 - vel2)

        print(f"  {e1['instance_id']}: pos_diff={pos_diff:.9f}m, vel_diff={vel_diff:.9f}m/s")

        assert pos_diff < 1e-9, f"Position mismatch for {e1['instance_id']}: {pos_diff}"
        assert vel_diff < 1e-9, f"Velocity mismatch for {e1['instance_id']}: {vel_diff}"

    print("[OK] Simulations are bit-for-bit identical (determinism verified)\n")


def test_timing():
    """Test tick timing measurement"""
    print("=" * 60)
    print("Test 4: Tick Timing")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=10,
        spawn_only_species=['sp-001-drifter']
    )

    print("Running 100 ticks with timing...")
    for i in range(100):
        sim.tick(evaluate_behaviors=False)

    stats = sim.get_tick_stats()
    print(f"\nTiming results:")
    print(f"  Total ticks: {stats['tick_count']}")
    print(f"  Average tick time: {stats['avg_tick_time_ms']:.3f} ms")
    print(f"  Last tick time: {stats['last_tick_time_ms']:.3f} ms")

    # No assertion on timing (too variable), just verify non-zero
    assert stats['avg_tick_time_ms'] > 0, "Timing not recorded"

    print("[OK] Timing measurement working\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 2 Test: Minimal Tick Loop")
    print("=" * 60)
    print()

    try:
        test_movement()
        test_bounds_reflection()
        test_determinism()
        test_timing()

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("[PASS] All Phase 2 tests passed!")
    print("=" * 60)
