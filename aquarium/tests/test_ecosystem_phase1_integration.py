"""
Integration tests for Phase 1 Ecosystem Foundation.

Tests resource extraction, starvation thresholds, density sampling, revival guard,
compaction integrity, behavior priorities, and gossip v2 compatibility.

No fixtures - uses inline helpers with explicit seeds per project conventions.
"""

import numpy as np
import pytest
from pathlib import Path

from aquarium.simulation import AquariumSimulation
from aquarium.entity import Entity


class TestResourceProviderExtraction:
    """Test resource provider extraction from biome YAML."""

    def test_resource_extraction_counts(self):
        """Verify 12 vents extracted with correct peaks and base density."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(data_root=data_root, spawn_limit=10)

        # Assert arrays exist and have correct length
        assert hasattr(sim, '_resource_centers'), "sim._resource_centers missing"
        assert hasattr(sim, '_resource_peaks'), "sim._resource_peaks missing"
        assert hasattr(sim, '_resource_sigmas'), "sim._resource_sigmas missing"
        assert hasattr(sim, '_resource_base_density'), "sim._resource_base_density missing"

        num_providers = len(sim._resource_centers)
        print(f"  Resource providers extracted: {num_providers}")
        print(f"  Peak values: {sim._resource_peaks}")
        print(f"  Base density: {sim._resource_base_density}")

        # Verify counts align
        assert len(sim._resource_centers) == 12, f"Expected 12 resource centers, got {len(sim._resource_centers)}"
        assert len(sim._resource_peaks) == 12, f"Expected 12 peaks, got {len(sim._resource_peaks)}"
        assert len(sim._resource_sigmas) == 12, f"Expected 12 sigmas, got {len(sim._resource_sigmas)}"

        # Verify all peaks = 50.0 (per vent-field-alpha.yaml)
        assert np.allclose(sim._resource_peaks, 50.0), f"Expected all peaks=50.0, got {sim._resource_peaks}"

        # Verify base density = 0.0 (no ambient food)
        assert sim._resource_base_density == 0.0, f"Expected base_density=0.0, got {sim._resource_base_density}"

        print("  [OK] 12 resource providers with peaks=50.0, base=0.0")


class TestStarvationThresholds:
    """Test starvation threshold population from species YAML."""

    def test_threshold_population_for_drifters(self):
        """Verify drifter-only spawn populates threshold=10.0 for all entities."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=20,
            spawn_only_species=['sp-001-drifter']
        )

        N = len(sim.entities)
        print(f"  Spawned {N} drifter entities")

        assert N > 0, "No entities spawned"

        # All drifters should have threshold=10.0 per sp-001-drifter.yaml
        thresholds = sim._starvation_threshold[:N]
        print(f"  Unique thresholds: {np.unique(thresholds)}")

        assert np.allclose(thresholds, 10.0), \
            f"Expected all thresholds=10.0 for drifters, got {np.unique(thresholds)}"

        # Optional: verify metabolism parsing (catches loader regressions)
        base_drains = sim._base_drain_per_tick[:N]
        intake_rates = sim._intake_rate_per_tick[:N]

        print(f"  Base drains: {np.unique(base_drains)}")
        print(f"  Intake rates: {np.unique(intake_rates)}")

        assert np.allclose(base_drains, 0.5), \
            f"Expected base_drain=0.5 for drifters, got {np.unique(base_drains)}"
        assert np.allclose(intake_rates, 1.0), \
            f"Expected intake_rate=1.0 for drifters, got {np.unique(intake_rates)}"

        print("  [OK] Thresholds=10.0, metabolism parsed correctly")


class TestDensitySampling:
    """Test plankton density sampling with spatial gradient."""

    def test_density_sampling_near_and_far(self):
        """Verify Gaussian falloff: vent center high, far away zero."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(data_root=data_root, spawn_limit=1)

        # Sample at vent center (should be exactly base + peak)
        vent_center = sim._resource_centers[0]
        vent_peak = sim._resource_peaks[0]
        base = sim._resource_base_density

        print(f"  Vent 0 center: {vent_center}")
        print(f"  Vent 0 peak: {vent_peak}, base: {base}")

        density_at_vent = sim._sample_plankton_density(vent_center.reshape(1, 3))[0]
        print(f"  Density at vent center: {density_at_vent:.6f}")

        expected_at_vent = base + vent_peak
        assert abs(density_at_vent - expected_at_vent) < 1e-6, \
            f"Vent center density {density_at_vent:.6f} != expected {expected_at_vent:.6f}"

        # Sample far from all vents (>5 sigma away)
        far_point = np.array([[1000.0, -100.0, 1000.0]])
        density_far = sim._sample_plankton_density(far_point)[0]
        print(f"  Density far from vents: {density_far:.6f}")

        assert density_far <= 1e-6, \
            f"Far density {density_far:.6f} should be ~0.0 (>5σ from all vents)"

        print("  [OK] Spatial gradient confirmed (near=50.00, far≈0.00)")


class TestRevivalGuard:
    """Test that entities at/below threshold don't feed back to life."""

    def test_no_same_tick_revival_guard(self):
        """Verify entity that crosses threshold in A.75 doesn't feed in B.5."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=5,
            spawn_only_species=['sp-001-drifter']
        )

        N_initial = len(sim.entities)
        print(f"  Initial entity count: {N_initial}")

        assert N_initial > 0, "No entities spawned"

        # Get threshold for drifters (should be 10.0)
        threshold = sim._starvation_threshold[0]
        print(f"  Drifter starvation threshold: {threshold}")

        # Make drain deterministic (no movement cost)
        # 1. Zero all velocities (stored on Entity objects, not SoA arrays)
        for entity in sim.entities:
            entity.velocity[:] = 0.0

        # 2. Disable avoidance (Phase A.5 can inject velocity)
        sim.world.simulation.avoidance_weight = 0.0

        print(f"  Zeroed velocities and disabled avoidance (deterministic drain)")

        # Arrange: Set entities 0-2 to cross threshold during A.75 drain
        # Base drain = 0.5, threshold = 10.0
        # Set energy = threshold + 0.25 = 10.25
        # After A.75 drain: energy = 10.25 - 0.5 = 9.75 (below threshold)
        num_to_kill = min(3, N_initial)
        for i in range(num_to_kill):
            sim._energy_current[i] = threshold + 0.25

        print(f"  Set entities [0:{num_to_kill}] energy to {threshold + 0.25:.2f} (will drain to {threshold - 0.25:.2f})")

        # Tick once with behaviors enabled (A.75 only runs when evaluate_behaviors=True)
        # A.75: drains energy below threshold
        # B.5: alive_after_drain mask excludes threshold-crossers (no feeding)
        # B.75: removes dead entities
        sim.tick(evaluate_behaviors=True)

        N_after = len(sim.entities)
        print(f"  Entity count after tick: {N_after}")

        # Entities that crossed threshold should be dead and compacted away
        expected_survivors = N_initial - num_to_kill

        assert N_after == expected_survivors, \
            f"Expected {expected_survivors} survivors after {num_to_kill} deaths, got {N_after}"

        print(f"  [OK] {num_to_kill} entities died after crossing threshold (revival guard active)")


class TestCompactionIntegrity:
    """Test that death compaction maintains SoA array synchronization."""

    def test_compaction_integrity(self):
        """Verify threshold array compacts in lockstep with other SoA arrays."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=10,
            spawn_only_species=['sp-001-drifter']
        )

        N_initial = len(sim.entities)
        print(f"  Initial entity count: {N_initial}")

        assert N_initial >= 5, "Need at least 5 entities for compaction test"

        # Record initial state (instance IDs for verification)
        initial_ids = [e.instance_id for e in sim.entities]
        print(f"  Initial instance IDs: {initial_ids[:5]}...")

        # Isolate compaction (prevent drain interference)
        # Set base_drain_per_tick to 0.0 so energy changes are only from explicit test setup
        sim._base_drain_per_tick[:] = 0.0
        print(f"  Zeroed base_drain_per_tick (isolate compaction from drain)")

        # Arrange: Mark entities at indices [1, 3] for death by setting energy <= threshold
        # Drifter threshold = 10.0, so set to 0.0 (well below threshold)
        kill_indices = [1, 3]
        for idx in kill_indices:
            sim._energy_current[idx] = 0.0

        print(f"  Marked entities at indices {kill_indices} for death (energy=0.0 <= threshold=10.0)")

        # Tick once with behaviors enabled (compaction runs at B.75)
        # B.75: identifies newly_dead (energy <= threshold & alive), compacts arrays
        sim.tick(evaluate_behaviors=True)

        N_after = len(sim.entities)
        expected_survivors = N_initial - len(kill_indices)

        print(f"  Entity count after tick: {N_after}")

        assert N_after == expected_survivors, \
            f"Expected {expected_survivors} survivors, got {N_after}"

        # Verify compaction: survivors should be at front of arrays [0:new_N]
        surviving_ids = [e.instance_id for e in sim.entities]
        print(f"  Surviving instance IDs: {surviving_ids[:5]}...")

        # Verify deaths occurred via telemetry (proves B.75 ran)
        deaths_this_tick = sim._ecosystem_telemetry.get('deaths_this_tick', 0)
        assert deaths_this_tick == len(kill_indices), \
            f"Expected {len(kill_indices)} deaths, telemetry shows {deaths_this_tick}"

        # Verify invariant: _count == len(entities) after tick
        # (After _count fix, this invariant should hold)
        assert sim._count == len(sim.entities), \
            f"_count ({sim._count}) != len(entities) ({len(sim.entities)})"

        # Verify all survivors are alive in the alive mask
        assert np.all(sim._alive[:N_after]), \
            "Compacted entities should all have alive=True"

        # Verify energy arrays are positive for all survivors (didn't die)
        survivor_energies = sim._energy_current[:N_after]
        assert np.all(survivor_energies > 0), \
            f"All survivors should have energy > 0, got min={np.min(survivor_energies)}"

        # Verify threshold array compacted correctly (10.0 for drifters)
        survivor_thresholds = sim._starvation_threshold[:N_after]
        assert np.allclose(survivor_thresholds, 10.0), \
            f"Compacted threshold array should be 10.0, got {np.unique(survivor_thresholds)}"

        # Verify expected survivor IDs (compaction preserves order, removes killed indices)
        expected_survivors_ids = [initial_ids[i] for i in range(N_initial) if i not in kill_indices]
        assert surviving_ids == expected_survivors_ids, \
            f"Survivor order mismatch: expected {expected_survivors_ids[:5]}..., got {surviving_ids[:5]}..."

        print(f"  [OK] Compaction preserved SoA integrity (alive, energy, threshold, order, _count invariant)")


class TestCountInvariantWithShip:
    """Test that ship presence doesn't leak into _count."""

    def test_count_invariant_with_ship_present(self):
        """Verify _count == len(entities) even when ship is in spatial builds."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=10,
            spawn_only_species=['sp-001-drifter']
        )

        N_initial = len(sim.entities)
        print(f"  Initial entity count: {N_initial}")

        # Add ship to simulation (will be included in spatial builds)
        ship_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        sim.update_ship(ship_position)

        assert sim.ship_entity is not None, "Ship should be present"
        print(f"  Ship added at {ship_position}")

        # Tick once (spatial builds will include ship)
        sim.tick(evaluate_behaviors=True)

        # _count should ONLY track entities, NOT ship
        assert sim._count == len(sim.entities), \
            f"_count ({sim._count}) != len(entities) ({len(sim.entities)}) - ship leaked into _count!"

        # Spatial adapter should have entity + ship references
        # (spatial._entities includes ship, but _count should not)
        print(f"  _count: {sim._count}")
        print(f"  len(entities): {len(sim.entities)}")
        print(f"  Ship present: {sim.ship_entity is not None}")

        print(f"  [OK] _count invariant holds with ship present (ship doesn't leak into _count)")


class TestKDTreeBuildFrequency:
    """Test that KD-tree builds occur exactly twice per tick (Phase A + B.9)."""

    def test_kdtree_builds_twice_per_tick(self):
        """Verify spatial index builds exactly twice: Phase A and B.9 gossip rebuild."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=10,
            spawn_only_species=['sp-001-drifter']
        )

        # Check that USE_GOSSIP is enabled (required for B.9 rebuild)
        from aquarium.simulation import USE_GOSSIP
        if not USE_GOSSIP:
            import pytest
            pytest.skip("USE_GOSSIP is False, B.9 rebuild won't occur")

        print(f"  Initial entities: {len(sim.entities)}")

        # Record build sequence before tick
        seq_before = sim.spatial._build_seq
        print(f"  Build sequence before tick: {seq_before}")

        # Tick once (should trigger Phase A build + B.9 rebuild)
        sim.tick(evaluate_behaviors=True)

        # Record build sequence after tick
        seq_after = sim.spatial._build_seq
        print(f"  Build sequence after tick: {seq_after}")

        # Verify exactly 2 builds occurred (Phase A + B.9)
        builds_this_tick = seq_after - seq_before
        print(f"  Builds this tick: {builds_this_tick}")

        assert builds_this_tick == 2, \
            f"Expected 2 KD builds (Phase A + B.9), got {builds_this_tick}"

        # Optional: verify gossip telemetry exists (confirms B.9 ran)
        assert sim._gossip_telemetry is not None, \
            "Gossip telemetry should exist after B.9 gossip phase"

        print(f"  [OK] Exactly 2 KD builds per tick confirmed (Phase A + B.9)")


class TestBehaviorPriorityInteraction:
    """Test that behavior priorities work correctly when multiple conditions are met."""

    def test_flee_predator_beats_urgent_forage(self):
        """Verify flee-predator (priority 1) wins over urgent-forage (priority 3)."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=103,  # Spawn both species: 100 drifters + 3 shadows
            spawn_only_species=['sp-001-drifter', 'sp-003-shadow']
        )

        # Pick one drifter and one shadow deterministically (sorted by instance_id)
        drifters = sorted([e for e in sim.entities if e.species_id == 'sp-001-drifter'],
                          key=lambda e: e.instance_id)
        shadows = sorted([e for e in sim.entities if e.species_id == 'sp-003-shadow'],
                         key=lambda e: e.instance_id)

        assert len(drifters) >= 1, "Need at least 1 drifter"
        assert len(shadows) >= 1, "Need at least 1 shadow (predator)"

        drifter = drifters[0]
        predator = shadows[0]

        print(f"  Test subjects: {drifter.instance_id} (drifter), {predator.instance_id} (shadow)")
        print(f"  Population: {len(drifters)} drifters, {len(shadows)} shadows")

        # Positioning: Place chosen predator 50m from drifter (within 80m flee range)
        predator.position = drifter.position + np.array([50.0, 0.0, 0.0], dtype=np.float64)

        # Move other shadows far away to prevent interference (beyond flee range)
        for i, shadow in enumerate(shadows):
            if shadow.instance_id != predator.instance_id:
                shadow.position = drifter.position + np.array([1000.0, 0.0, 0.0], dtype=np.float64)
                print(f"  Moved shadow {i+1} to 1000m (out of range)")

        # Stability: Disable avoidance, zero all velocities
        sim.world.simulation.avoidance_weight = 0.0
        for entity in sim.entities:
            entity.velocity[:] = 0.0

        # Energy setup: Drifter alive, hungry (below 30.0 for urgent-forage), above threshold (10.0)
        drifter_idx = sim.entities.index(drifter)
        sim._energy_current[drifter_idx] = 25.0

        print(f"  Test predator positioned 50m from drifter (flee range: 80m)")
        print(f"  Drifter energy: 25.0 (urgent-forage: <30, alive: >10)")
        print(f"  Avoidance disabled, velocities zeroed")

        # Act: Tick with both conditions met (predator nearby + hungry)
        sim.tick(evaluate_behaviors=True)

        # Assert: flee-predator (priority 1) should win over urgent-forage (priority 3)
        assert drifter.active_behavior_id == 'flee-predator', \
            f"Expected 'flee-predator' (priority 1), got '{drifter.active_behavior_id}'"

        print(f"  [Part 1] Drifter behavior: {drifter.active_behavior_id} (flee wins)")

        # Remove predator influence: Move predator beyond 80m (don't delete!)
        predator.position = drifter.position + np.array([150.0, 0.0, 0.0], dtype=np.float64)
        print(f"  Test predator moved to 150m (beyond flee range)")

        # Act: Tick again without predator influence
        sim.tick(evaluate_behaviors=True)

        # Assert: urgent-forage should now activate (no higher priority behavior)
        assert drifter.active_behavior_id == 'urgent-forage', \
            f"Expected 'urgent-forage' without predator, got '{drifter.active_behavior_id}'"

        print(f"  [Part 2] Drifter behavior: {drifter.active_behavior_id} (urgent-forage activates)")
        print(f"  [OK] Behavior priorities correct: flee-predator (1) > urgent-forage (3)")


class TestKnowledgeTokenDictGuard:
    """Test gossip v2 dict token compatibility with behavior conditions."""

    def test_knowledge_token_dict_guard_investigate(self):
        """Verify positive ship_sentiment (dict token) triggers investigate-ship."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=3,  # Small population, drifters only
            spawn_only_species=['sp-001-drifter']
        )

        # Pick first drifter deterministically
        drifters = sorted(sim.entities, key=lambda e: e.instance_id)
        assert len(drifters) >= 1, "Need at least 1 drifter"
        drifter = drifters[0]

        print(f"  Drifter: {drifter.instance_id}")

        # Place ship within 150m (cache radius) for investigate range test
        ship_position = drifter.position + np.array([100.0, 0.0, 0.0], dtype=np.float64)
        sim.update_ship(ship_position)

        # Stability: Disable avoidance, zero velocities
        sim.world.simulation.avoidance_weight = 0.0
        for entity in sim.entities:
            entity.velocity[:] = 0.0

        # Energy setup: Safe energy (>30, no hunger interference)
        drifter_idx = sim.entities.index(drifter)
        sim._energy_current[drifter_idx] = 60.0

        # Set ship_sentiment token as gossip v2 dict (positive value)
        drifter.knowledge_tokens['ship_sentiment'] = {
            'kind': 'ship_sentiment',
            'value': 0.6,  # Positive sentiment
            'version': 1,
            'last_tick': 0.0,
            'reliability': 1.0,
            'source': 'direct'
        }

        print(f"  Ship positioned 100m from drifter (within cache radius: 150m)")
        print(f"  Drifter energy: 60.0 (no hunger interference)")
        print(f"  ship_sentiment token: value=0.6 (dict format)")

        # Act: Tick with evaluate_behaviors=True (integrated path)
        sim.tick(evaluate_behaviors=True)

        # Assert: investigate-ship should activate (positive sentiment + ship in range)
        assert drifter.active_behavior_id == 'investigate-ship', \
            f"Expected 'investigate-ship', got '{drifter.active_behavior_id}'"

        print(f"  Drifter behavior: {drifter.active_behavior_id}")
        print(f"  [OK] Dict guard works: token['value'] extracted for investigate condition")

    def test_knowledge_token_dict_guard_hostile_flee(self):
        """Verify negative ship_sentiment (dict token) triggers flee-hostile-ship."""
        data_root = Path(__file__).parent.parent.parent / "data"
        sim = AquariumSimulation(
            data_root=data_root,
            spawn_limit=3,  # Small population, drifters only
            spawn_only_species=['sp-001-drifter']
        )

        # Pick first drifter deterministically
        drifters = sorted(sim.entities, key=lambda e: e.instance_id)
        assert len(drifters) >= 1, "Need at least 1 drifter"
        drifter = drifters[0]

        print(f"  Drifter: {drifter.instance_id}")

        # Place ship within 150m (flee-hostile range)
        ship_position = drifter.position + np.array([100.0, 0.0, 0.0], dtype=np.float64)
        sim.update_ship(ship_position)

        # Stability: Disable avoidance, zero velocities
        sim.world.simulation.avoidance_weight = 0.0
        for entity in sim.entities:
            entity.velocity[:] = 0.0

        # Energy setup: Safe energy (>30, no hunger interference)
        drifter_idx = sim.entities.index(drifter)
        sim._energy_current[drifter_idx] = 60.0

        # Set ship_sentiment token as gossip v2 dict (negative value)
        drifter.knowledge_tokens['ship_sentiment'] = {
            'kind': 'ship_sentiment',
            'value': -0.6,  # Negative sentiment (hostile)
            'version': 1,
            'last_tick': 0.0,
            'reliability': 1.0,
            'source': 'direct'
        }

        print(f"  Ship positioned 100m from drifter (flee-hostile range: 150m)")
        print(f"  Drifter energy: 60.0 (no hunger interference)")
        print(f"  ship_sentiment token: value=-0.6 (dict format)")

        # Act: Tick with evaluate_behaviors=True (integrated path)
        sim.tick(evaluate_behaviors=True)

        # Assert: flee-hostile-ship should activate (negative sentiment + ship in range)
        assert drifter.active_behavior_id == 'flee-hostile-ship', \
            f"Expected 'flee-hostile-ship', got '{drifter.active_behavior_id}'"

        print(f"  Drifter behavior: {drifter.active_behavior_id}")
        print(f"  [OK] Dict guard works: token['value'] extracted for hostile flee condition")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
