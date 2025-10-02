"""
Tests for obstacle avoidance (Phase 4).

Per SD guidance:
- No penetration: entities stay outside sphere radius
- Determinism: identical runs produce identical positions
- Speed limits: velocities stay within max_speed after avoidance
"""

import numpy as np
from pathlib import Path

from aquarium.simulation import AquariumSimulation
from aquarium.tests.perf_harness import build_perf_scenario


def test_sphere_avoidance_no_penetration():
    """
    Test that entities do not penetrate sphere obstacles.

    After N ticks, signed distance to every sphere should be >= 0.
    """
    data_root = Path(__file__).parent.parent.parent / "data"

    # Build scenario with entities near vents
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=50,  # Small population for focused test
        spawn_only_species=None
    )

    # Run for 100 ticks
    for _ in range(100):
        sim.tick()

    # Check all entities against all spheres in their biome
    violations = []
    for entity in sim.entities:
        biome = sim.biomes[entity.biome_id]
        spheres = biome.obstacles.get('spheres', [])

        for sphere in spheres:
            sphere_center = np.array(sphere.center, dtype=np.float64)
            distance_to_center = np.linalg.norm(entity.position - sphere_center)
            signed_distance = distance_to_center - sphere.radius

            if signed_distance < 0:
                violations.append({
                    'entity_id': entity.instance_id,
                    'sphere_id': sphere.id,
                    'penetration_depth': abs(signed_distance)
                })

    if violations:
        print(f"\n[WARN] Found {len(violations)} penetrations:")
        for v in violations[:5]:  # Print first 5
            print(f"  Entity {v['entity_id']} penetrates {v['sphere_id']} by {v['penetration_depth']:.3f}m")

    # Soft assertion: warn but don't fail (MVP may have edge cases)
    assert len(violations) == 0, f"Found {len(violations)} sphere penetrations"


def test_sphere_avoidance_determinism():
    """
    Test that avoidance is deterministic.

    Two runs with same seed should produce identical entity positions.
    """
    data_root = Path(__file__).parent.parent.parent / "data"

    # Run A
    sim_a = build_perf_scenario(data_root, N=100, seed=42)
    for _ in range(50):
        sim_a.tick()

    positions_a = {e.instance_id: e.position.copy() for e in sim_a.entities}

    # Run B (same seed)
    sim_b = build_perf_scenario(data_root, N=100, seed=42)
    for _ in range(50):
        sim_b.tick()

    positions_b = {e.instance_id: e.position.copy() for e in sim_b.entities}

    # Compare positions
    mismatches = []
    for entity_id in positions_a.keys():
        pos_a = positions_a[entity_id]
        pos_b = positions_b[entity_id]
        diff = np.linalg.norm(pos_a - pos_b)

        if diff > 1e-9:  # Floating point tolerance
            mismatches.append({
                'entity_id': entity_id,
                'diff': diff,
                'pos_a': pos_a,
                'pos_b': pos_b
            })

    if mismatches:
        print(f"\n[FAIL] Found {len(mismatches)} position mismatches:")
        for m in mismatches[:3]:
            print(f"  Entity {m['entity_id']}: diff={m['diff']:.6f}m")
            print(f"    A: {m['pos_a']}")
            print(f"    B: {m['pos_b']}")

    assert len(mismatches) == 0, f"Determinism violated: {len(mismatches)} position mismatches"


def test_sphere_avoidance_speed_limits():
    """
    Test that avoidance respects species max speed.

    After avoidance, |velocity| <= species.movement.max_speed_ms.
    """
    data_root = Path(__file__).parent.parent.parent / "data"

    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=50,
        spawn_only_species=None
    )

    # Run for 50 ticks
    for _ in range(50):
        sim.tick()

    # Check all entity speeds
    violations = []
    for entity in sim.entities:
        species = sim.species_registry.get(entity.species_id)
        if not species:
            continue

        max_speed = species.movement.max_speed_ms
        current_speed = np.linalg.norm(entity.velocity)

        # Allow small floating point tolerance (1e-6)
        if current_speed > max_speed + 1e-6:
            violations.append({
                'entity_id': entity.instance_id,
                'species_id': entity.species_id,
                'current_speed': current_speed,
                'max_speed': max_speed,
                'excess': current_speed - max_speed
            })

    if violations:
        print(f"\n[FAIL] Found {len(violations)} speed violations:")
        for v in violations[:5]:
            print(f"  Entity {v['entity_id']} ({v['species_id']}): "
                  f"{v['current_speed']:.3f} m/s > {v['max_speed']:.3f} m/s "
                  f"(excess: {v['excess']:.6f})")

    assert len(violations) == 0, f"Speed limit violated: {len(violations)} entities too fast"


def test_sphere_avoidance_performance():
    """
    Spot check that avoidance overhead is reasonable.

    Expected: Avoidance adds <5ms at 143 entities.
    This is not a hard limit, just monitoring.
    """
    data_root = Path(__file__).parent.parent.parent / "data"

    sim = build_perf_scenario(data_root, N=143, seed=42)

    # Warmup
    for _ in range(10):
        sim.tick()

    # Measure
    for _ in range(100):
        sim.tick()

    stats = sim.get_tick_stats()
    avg_tick_ms = stats['avg_tick_time_ms']

    print(f"\n[Perf] Avg tick time with avoidance (143 entities): {avg_tick_ms:.3f} ms")

    # Soft assertion: warn if significantly slower than expected
    # Baseline without avoidance: ~4ms
    # Expected with sphere avoidance: <10ms
    if avg_tick_ms > 20.0:
        print(f"[WARN] Avoidance overhead higher than expected: {avg_tick_ms:.3f} ms > 20ms")

    # Don't fail test, just monitor
    assert avg_tick_ms < 50.0, f"Avoidance performance degraded: {avg_tick_ms:.3f} ms"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
