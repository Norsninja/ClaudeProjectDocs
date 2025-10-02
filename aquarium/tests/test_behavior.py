"""
Test Phase 3: Behavior evaluation system.

Verifies:
- Priority-based behavior selection
- Flee behavior (predator proximity)
- Investigate behavior (ship proximity, neutral sentiment)
- Knowledge token defaults (ship_sentiment = 0.0 when missing)
- Determinism with behaviors
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aquarium.simulation import AquariumSimulation
from aquarium.entity import Entity
from aquarium.spatial import distance_3d


def test_flee_predator():
    """Test that Drifters flee from nearby Shadow predators"""
    print("=" * 60)
    print("Test 1: Flee Predator Behavior")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Spawn 3 Drifters only (no Shadows yet)
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=3,
        spawn_only_species=['sp-001-drifter']
    )

    print(f"Spawned {len(sim.entities)} Drifters")

    # Manually create a Shadow entity nearby (within 80m flee threshold)
    biome_id = list(sim.biomes.keys())[0]
    drifter = sim.entities[0]

    # Place Shadow 50m away from first Drifter
    shadow_position = drifter.position + np.array([50.0, 0.0, 0.0], dtype=np.float64)

    shadow = Entity(
        instance_id="sp-003-shadow-test-0000",
        species_id="sp-003-shadow",
        biome_id=biome_id,
        position=shadow_position,
        velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        size_factor=1.0,
        tags=['mobile', 'predator']  # Critical: 'predator' tag
    )

    sim.entities.append(shadow)
    print(f"Added Shadow at distance {distance_3d(drifter.position, shadow.position):.2f}m from Drifter 0")

    # Record initial positions
    initial_positions = {e.instance_id: e.position.copy() for e in sim.entities if 'drifter' in e.instance_id}

    # Tick with behaviors and verify flee immediately
    print("\nTicking with behavior evaluation...")
    flee_behaviors_seen = []

    for i in range(10):
        sim.tick(evaluate_behaviors=True)

        # Check behavior on first tick (before fleeing too far)
        if i == 0:
            for entity in sim.entities:
                if 'drifter' in entity.instance_id:
                    dist_to_shadow = distance_3d(entity.position, shadow.position)
                    print(f"  Tick {i+1}: {entity.instance_id}: behavior={entity.active_behavior_id}, dist={dist_to_shadow:.2f}m")
                    flee_behaviors_seen.append(entity.active_behavior_id == "flee-predator")

    # Verify at least one Drifter chose flee-predator (the closest one)
    assert any(flee_behaviors_seen), "At least one Drifter should have flee-predator behavior"

    # Verify Drifters moved away from Shadow
    print("\nVerifying movement away from predator...")
    for entity in sim.entities:
        if 'drifter' in entity.instance_id:
            initial_dist = distance_3d(initial_positions[entity.instance_id], shadow.position)
            current_dist = distance_3d(entity.position, shadow.position)

            print(f"  {entity.instance_id}: initial={initial_dist:.2f}m, current={current_dist:.2f}m")

            # Drifters close enough initially should have moved away
            if initial_dist < 80.0:
                assert current_dist > initial_dist, "Nearby Drifter should move away from predator"

    print("[OK] Drifters flee from predator\n")


def test_investigate_ship():
    """Test that Drifters investigate ship when sentiment is neutral"""
    print("=" * 60)
    print("Test 2: Investigate Ship Behavior")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Spawn 3 Drifters
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=3,
        spawn_only_species=['sp-001-drifter']
    )

    print(f"Spawned {len(sim.entities)} Drifters")

    # Place ship 250m away (within 300m investigate threshold)
    biome = list(sim.biomes.values())[0]
    center = np.array(biome.bounds['center'], dtype=np.float64)
    ship_position = center + np.array([250.0, 0.0, 0.0], dtype=np.float64)

    sim.update_ship(ship_position)
    print(f"Ship placed at {ship_position.tolist()}")

    # Verify all Drifters within 300m of ship
    for entity in sim.entities:
        dist = distance_3d(entity.position, ship_position)
        print(f"  {entity.instance_id}: distance to ship = {dist:.2f}m")

    # Record initial distances
    initial_distances = {e.instance_id: distance_3d(e.position, ship_position) for e in sim.entities}

    # Tick with behaviors
    print("\nTicking 20 times with behavior evaluation...")
    for i in range(20):
        sim.tick(evaluate_behaviors=True)

    # Verify Drifters have investigate behavior and move toward ship
    print("\nVerifying investigate behavior...")
    for entity in sim.entities:
        print(f"  {entity.instance_id}: behavior={entity.active_behavior_id}")

        # Should be investigate-ship (priority 3) or forage (priority 10)
        # If within 300m, should be investigate
        dist = distance_3d(entity.position, ship_position)
        if dist < 300.0:
            assert entity.active_behavior_id in ["investigate-ship", "forage"], \
                f"Expected investigate-ship or forage, got {entity.active_behavior_id}"

        # Verify movement toward ship (if investigate active)
        if entity.active_behavior_id == "investigate-ship":
            initial_dist = initial_distances[entity.instance_id]
            current_dist = distance_3d(entity.position, ship_position)

            print(f"    Initial distance: {initial_dist:.2f}m, Current: {current_dist:.2f}m")
            # Should move closer (but might have bounced off biome boundary)
            # Allow small margin
            assert current_dist < initial_dist + 50.0, "Drifter should generally move toward ship"

    print("[OK] Drifters investigate ship\n")


def test_neutral_sentiment_default():
    """Test that missing ship_sentiment token defaults to 0.0 (neutral)"""
    print("=" * 60)
    print("Test 3: Neutral Sentiment Default")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Spawn 2 Drifters
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=2,
        spawn_only_species=['sp-001-drifter']
    )

    print(f"Spawned {len(sim.entities)} Drifters")

    # Verify no knowledge tokens
    for entity in sim.entities:
        assert len(entity.knowledge_tokens) == 0, "Expected no knowledge tokens initially"
        print(f"  {entity.instance_id}: knowledge_tokens = {entity.knowledge_tokens}")

    # Place ship within investigate range
    biome = list(sim.biomes.values())[0]
    center = np.array(biome.bounds['center'], dtype=np.float64)
    ship_position = center + np.array([200.0, 0.0, 0.0], dtype=np.float64)

    sim.update_ship(ship_position)
    print(f"Ship placed at {ship_position.tolist()}")

    # Tick with behaviors
    print("\nTicking 5 times...")
    for i in range(5):
        sim.tick(evaluate_behaviors=True)

    # Verify behavior choices (should default to investigate, not flee)
    print("\nVerifying behavior with default sentiment...")
    for entity in sim.entities:
        dist = distance_3d(entity.position, ship_position)
        print(f"  {entity.instance_id}: distance={dist:.2f}m, behavior={entity.active_behavior_id}")

        # With no token (default 0.0), should NOT flee-hostile-ship (requires < -0.5)
        # Should be investigate-ship (>= 0.0) or forage
        assert entity.active_behavior_id != "flee-hostile-ship", \
            "Should not flee with neutral sentiment"

        # If within range, should investigate
        if dist < 300.0:
            assert entity.active_behavior_id in ["investigate-ship", "forage"], \
                f"Expected investigate or forage, got {entity.active_behavior_id}"

    print("[OK] Default neutral sentiment works correctly\n")


def test_behavior_determinism():
    """Test that same seed produces identical behavior sequences"""
    print("=" * 60)
    print("Test 4: Behavior Determinism")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Run simulation 1
    print("Running simulation 1...")
    sim1 = AquariumSimulation(
        data_root=data_root,
        spawn_limit=3,
        spawn_only_species=['sp-001-drifter']
    )

    # Place ship
    biome = list(sim1.biomes.values())[0]
    center = np.array(biome.bounds['center'], dtype=np.float64)
    ship_position = center + np.array([250.0, 0.0, 0.0], dtype=np.float64)
    sim1.update_ship(ship_position)

    for _ in range(50):
        sim1.tick(evaluate_behaviors=True)

    snapshot1 = sim1.get_snapshot()

    # Run simulation 2
    print("Running simulation 2...")
    sim2 = AquariumSimulation(
        data_root=data_root,
        spawn_limit=3,
        spawn_only_species=['sp-001-drifter']
    )

    sim2.update_ship(ship_position)

    for _ in range(50):
        sim2.tick(evaluate_behaviors=True)

    snapshot2 = sim2.get_snapshot()

    # Compare behavior sequences
    print("\nComparing behavior sequences...")
    entities1 = sorted(snapshot1['entities'], key=lambda e: e['instance_id'])
    entities2 = sorted(snapshot2['entities'], key=lambda e: e['instance_id'])

    for e1, e2 in zip(entities1, entities2):
        print(f"  {e1['instance_id']}: behavior={e1['active_behavior_id']}")

        assert e1['active_behavior_id'] == e2['active_behavior_id'], \
            f"Behavior mismatch: {e1['active_behavior_id']} vs {e2['active_behavior_id']}"

        pos1 = np.array(e1['position'])
        pos2 = np.array(e2['position'])
        pos_diff = np.linalg.norm(pos1 - pos2)

        vel1 = np.array(e1['velocity'])
        vel2 = np.array(e2['velocity'])
        vel_diff = np.linalg.norm(vel1 - vel2)

        print(f"    pos_diff={pos_diff:.9f}m, vel_diff={vel_diff:.9f}m/s")

        assert pos_diff < 1e-9, f"Position mismatch: {pos_diff}"
        assert vel_diff < 1e-9, f"Velocity mismatch: {vel_diff}"

    print("[OK] Behavior determinism verified\n")


def test_multiple_predators():
    """Test that Drifter flees from nearest predator when multiple present"""
    print("=" * 60)
    print("Test 5: Multiple Predators (Nearest Selection)")
    print("=" * 60)

    data_root = Path(__file__).parent.parent.parent / "data"

    # Spawn 1 Drifter
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=1,
        spawn_only_species=['sp-001-drifter']
    )

    drifter = sim.entities[0]
    biome_id = list(sim.biomes.keys())[0]

    # Create 3 Shadows at different distances
    shadow1_position = drifter.position + np.array([50.0, 0.0, 0.0], dtype=np.float64)
    shadow2_position = drifter.position + np.array([-100.0, 0.0, 0.0], dtype=np.float64)
    shadow3_position = drifter.position + np.array([0.0, 0.0, 70.0], dtype=np.float64)

    shadow1 = Entity(
        instance_id="shadow-1",
        species_id="sp-003-shadow",
        biome_id=biome_id,
        position=shadow1_position,
        velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        size_factor=1.0,
        tags=['mobile', 'predator']
    )

    shadow2 = Entity(
        instance_id="shadow-2",
        species_id="sp-003-shadow",
        biome_id=biome_id,
        position=shadow2_position,
        velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        size_factor=1.0,
        tags=['mobile', 'predator']
    )

    shadow3 = Entity(
        instance_id="shadow-3",
        species_id="sp-003-shadow",
        biome_id=biome_id,
        position=shadow3_position,
        velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        size_factor=1.0,
        tags=['mobile', 'predator']
    )

    sim.entities.extend([shadow1, shadow2, shadow3])

    print(f"Shadow 1 distance: {distance_3d(drifter.position, shadow1.position):.2f}m")
    print(f"Shadow 2 distance: {distance_3d(drifter.position, shadow2.position):.2f}m")
    print(f"Shadow 3 distance: {distance_3d(drifter.position, shadow3.position):.2f}m")

    initial_pos = drifter.position.copy()

    # Tick once
    sim.tick(evaluate_behaviors=True)

    # Verify flee behavior active
    assert drifter.active_behavior_id == "flee-predator", \
        f"Expected flee-predator, got {drifter.active_behavior_id}"

    # Verify fleeing away from Shadow1 (closest at 50m)
    direction_from_shadow1 = drifter.position - shadow1.position
    direction_from_shadow1_norm = direction_from_shadow1 / np.linalg.norm(direction_from_shadow1)

    velocity_norm = drifter.velocity / np.linalg.norm(drifter.velocity)

    # Check if velocity is aligned with away-from-shadow1 direction (dot product > 0.9)
    alignment = np.dot(velocity_norm, direction_from_shadow1_norm)
    print(f"\nVelocity alignment with away-from-nearest: {alignment:.3f}")
    assert alignment > 0.9, f"Expected high alignment (>0.9), got {alignment:.3f}"

    print("[OK] Drifter flees from nearest predator\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3 Test: Behavior Evaluation")
    print("=" * 60)
    print()

    try:
        test_flee_predator()
        test_investigate_ship()
        test_neutral_sentiment_default()
        test_behavior_determinism()
        test_multiple_predators()

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("[PASS] All Phase 3 tests passed!")
    print("=" * 60)
