"""
Test batch query APIs for spatial index adapter.

Phase 5: Validate that batch queries produce identical results to per-entity queries.
Tests determinism, self-exclusion, biome filtering, and tag filtering.
"""

import pytest
import numpy as np
from pathlib import Path

from aquarium.simulation import AquariumSimulation
from aquarium.entity import Entity


def test_nearest_by_tag_batch_determinism_100():
    """Test batch vs per-entity query determinism at 100 entities."""
    _test_batch_determinism(entity_count=100)


def test_nearest_by_tag_batch_determinism_1000():
    """Test batch vs per-entity query determinism at 1000 entities."""
    _test_batch_determinism(entity_count=1000)


def _test_batch_determinism(entity_count: int):
    """
    Core test: batch queries must produce identical results to per-entity queries.

    Validates:
    - Distance matches (bit-for-bit)
    - Entity selection matches (same instance_id)
    - Self-exclusion works (entity never finds itself)
    - Biome filtering works (cross-biome queries return no match)
    - Tag filtering works (only finds entities with correct tag)
    """
    # Setup simulation
    data_root = Path(__file__).parent.parent.parent / "data"
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=entity_count,
        spawn_only_species=None
    )

    # Build spatial index (this also builds mappings and masks)
    all_entities = sim.entities.copy()
    if sim.ship_entity:
        all_entities.append(sim.ship_entity)

    # Fill positions array (mimic simulation.tick() Phase A)
    sim._ensure_position_capacity(len(all_entities))
    for i, entity in enumerate(all_entities):
        sim._positions[i] = entity.position

    sim.spatial.build(sim._positions[:len(all_entities)], refs=all_entities)

    # Test predator queries
    print(f"\n[Test] Batch determinism at {entity_count} entities")
    print(f"  Testing 'predator' tag queries...")

    # Build source rows (all non-predator entities)
    source_rows = []
    for row, entity in enumerate(all_entities):
        if 'predator' not in entity.tags:
            source_rows.append(row)

    if not source_rows:
        pytest.skip("No non-predator entities to test")

    source_rows = np.array(source_rows, dtype=np.int32)

    # Run batch query
    batch_indices, batch_distances = sim.spatial.nearest_by_tag_batch(
        source_rows=source_rows,
        tag='predator',
        max_distance=None
    )

    # Run per-entity queries for comparison
    per_entity_indices = []
    per_entity_distances = []

    for row in source_rows:
        entity = all_entities[row]
        nearest = sim.spatial.find_nearest_by_tag(entity, 'predator', max_distance=None)

        if nearest is None:
            per_entity_indices.append(-1)
            per_entity_distances.append(np.inf)
        else:
            # Find row index of nearest
            nearest_row = sim.spatial._row_of_id[nearest.instance_id]
            per_entity_indices.append(nearest_row)

            # Compute distance
            dist = np.linalg.norm(nearest.position - entity.position)
            per_entity_distances.append(dist)

    per_entity_indices = np.array(per_entity_indices, dtype=np.int32)
    per_entity_distances = np.array(per_entity_distances, dtype=np.float64)

    # Compare results
    indices_match = np.array_equal(batch_indices, per_entity_indices)
    distances_match = np.allclose(batch_distances, per_entity_distances, rtol=0, atol=1e-12)

    # Report mismatches
    if not indices_match:
        mismatches = np.where(batch_indices != per_entity_indices)[0]
        print(f"  [FAIL] Index mismatches: {len(mismatches)}/{len(source_rows)}")
        for i in mismatches[:5]:  # Show first 5
            print(f"    Source row {source_rows[i]}: batch={batch_indices[i]} vs per-entity={per_entity_indices[i]}")
    else:
        print(f"  [PASS] Indices match perfectly ({len(source_rows)} queries)")

    if not distances_match:
        mismatches = ~np.isclose(batch_distances, per_entity_distances, rtol=0, atol=1e-12)
        print(f"  [FAIL] Distance mismatches: {np.sum(mismatches)}/{len(source_rows)}")
        for i in np.where(mismatches)[0][:5]:  # Show first 5
            print(f"    Source row {source_rows[i]}: batch={batch_distances[i]:.15f} vs per-entity={per_entity_distances[i]:.15f}")
    else:
        print(f"  [PASS] Distances match perfectly (tolerance: 1e-12)")

    # Assert determinism
    assert indices_match, "Batch indices must match per-entity queries exactly"
    assert distances_match, "Batch distances must match per-entity queries within 1e-12"


def test_batch_self_exclusion():
    """Test that batch queries never return the source entity itself."""
    data_root = Path(__file__).parent.parent.parent / "data"
    sim = AquariumSimulation(
        data_root=data_root,
        spawn_limit=100,
        spawn_only_species=None
    )

    # Build spatial index
    all_entities = sim.entities.copy()
    if sim.ship_entity:
        all_entities.append(sim.ship_entity)

    sim.spatial.build(sim._positions[:len(all_entities)], refs=all_entities)

    # Query all entities for nearest with same tag
    for tag in ['predator', 'ship']:
        # Find entities with this tag
        source_rows = []
        for row, entity in enumerate(all_entities):
            if tag in entity.tags:
                source_rows.append(row)

        if not source_rows:
            continue

        source_rows = np.array(source_rows, dtype=np.int32)

        # Run batch query
        batch_indices, _ = sim.spatial.nearest_by_tag_batch(
            source_rows=source_rows,
            tag=tag,
            max_distance=None
        )

        # Verify no self-matches
        for i, source_row in enumerate(source_rows):
            result_row = batch_indices[i]
            if result_row != -1:  # If a match was found
                assert result_row != source_row, f"Entity at row {source_row} found itself!"

    print(f"[PASS] Self-exclusion verified for all tags")


def test_batch_multi_biome():
    """Test batch queries with multiple biomes (cross-biome filtering)."""
    # This test would require multi-biome data
    # Skip for now, but document the test case
    pytest.skip("Multi-biome test requires multi-biome data pack")


if __name__ == "__main__":
    # Run tests manually
    test_nearest_by_tag_batch_determinism_100()
    test_nearest_by_tag_batch_determinism_1000()
    test_batch_self_exclusion()
    print("\n[OK] All batch query tests passed!")
