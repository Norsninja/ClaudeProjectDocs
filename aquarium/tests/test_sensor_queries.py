"""
Tests for sensor query API (Phase 6).

Per SD guidance:
- Cone selectivity: range and angle variations affect hit count
- Flag presence: only requested fields are populated
- Determinism: identical results across runs with same state
- Performance: ≤2ms at 1000 entities for typical cone queries

Step 1: Basic fields only (POSITION, VELOCITY, DISTANCE)
"""

import numpy as np
import time
from typing import List

from aquarium.entity import Entity
from aquarium.spatial_queries import (
    SpatialIndexAdapter,
    QUERY_FLAG_POSITION,
    QUERY_FLAG_VELOCITY,
    QUERY_FLAG_DISTANCE,
    QUERY_FLAG_ACOUSTIC,
    QUERY_FLAG_BIOLUMINESCENT,
    QUERY_FLAG_OPTICAL,
    QUERY_FLAG_THERMAL
)


def create_test_entities(count: int, biome_id: str = "test-biome", seed: int = 42) -> List[Entity]:
    """
    Create test entities in a grid pattern for predictable testing.

    Args:
        count: Number of entities to create
        biome_id: Biome ID for all entities
        seed: Random seed for reproducibility

    Returns:
        List of Entity objects
    """
    np.random.seed(seed)
    entities = []

    for i in range(count):
        # Grid placement with some randomness
        x = (i % 10) * 10.0 + np.random.uniform(-2, 2)
        y = -((i // 10) % 10) * 10.0 + np.random.uniform(-2, 2)
        z = (i // 100) * 10.0 + np.random.uniform(-2, 2)

        # Random velocity
        vel = np.random.uniform(-1, 1, size=3)

        # Base emissions for testing (moderate values for variety)
        base_emissions = {
            'acoustic': {
                'amplitude': 0.5,
                'peak_hz': 100.0
            },
            'bioluminescent': {
                'intensity': 0.3,
                'wavelength_nm': 500.0
            }
        }

        entity = Entity(
            instance_id=f"test-{i:04d}",
            species_id="sp-test",
            biome_id=biome_id,
            position=np.array([x, y, z], dtype=np.float64),
            velocity=vel,
            size_factor=1.0,
            tags=['mobile'],
            base_emissions=base_emissions
        )
        entities.append(entity)

    return entities


def test_cone_range_selectivity():
    """
    Test that increasing range increases hit count monotonically.

    Fixed angle, varying range: more entities should be detected as range increases.
    """
    entities = create_test_entities(100, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([50.0, -50.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])  # Point along +X
    angle_deg = 30.0
    flags = QUERY_FLAG_DISTANCE

    # Test increasing ranges
    ranges = [10.0, 20.0, 40.0, 80.0, 160.0]
    hit_counts = []

    for range_m in ranges:
        result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)
        hit_counts.append(len(result.entities))

    # Verify monotonic increase (or equal)
    for i in range(1, len(hit_counts)):
        assert hit_counts[i] >= hit_counts[i-1], \
            f"Hit count decreased with range: {hit_counts[i-1]} -> {hit_counts[i]} at range {ranges[i]}"

    # Verify we actually detect different counts (not all zero or all same)
    assert len(set(hit_counts)) > 1, "Range variation should produce different hit counts"

    print(f"[OK] Cone range selectivity: {hit_counts} hits for ranges {ranges}")


def test_cone_angle_selectivity():
    """
    Test that increasing angle increases hit count monotonically.

    Fixed range, varying angle: more entities should be detected as angle increases.
    """
    entities = create_test_entities(100, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([50.0, -50.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])  # Point along +X
    range_m = 50.0
    flags = QUERY_FLAG_DISTANCE

    # Test increasing angles (half-angles)
    angles = [5.0, 15.0, 30.0, 60.0, 90.0]
    hit_counts = []

    for angle_deg in angles:
        result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)
        hit_counts.append(len(result.entities))

    # Verify monotonic increase (or equal)
    for i in range(1, len(hit_counts)):
        assert hit_counts[i] >= hit_counts[i-1], \
            f"Hit count decreased with angle: {hit_counts[i-1]} -> {hit_counts[i]} at angle {angles[i]}"

    # Verify we actually detect different counts
    assert len(set(hit_counts)) > 1, "Angle variation should produce different hit counts"

    print(f"[OK] Cone angle selectivity: {hit_counts} hits for angles {angles}")


def test_flag_presence_distance_only():
    """
    Test that only DISTANCE flag populates distance field, not position or velocity.
    """
    entities = create_test_entities(20, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 100.0
    flags = QUERY_FLAG_DISTANCE  # Only distance requested

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) > 0, "Should detect entities in cone"

    for hit in result.entities:
        # Distance should be present
        assert hit.distance is not None, f"Distance missing for {hit.entity_id}"
        assert hit.distance >= 0, f"Distance should be non-negative, got {hit.distance}"

        # Position should NOT be present
        assert hit.pos_x is None, "pos_x should be None when POSITION flag not set"
        assert hit.pos_y is None, "pos_y should be None when POSITION flag not set"
        assert hit.pos_z is None, "pos_z should be None when POSITION flag not set"

        # Velocity should NOT be present
        assert hit.vel_x is None, "vel_x should be None when VELOCITY flag not set"
        assert hit.vel_y is None, "vel_y should be None when VELOCITY flag not set"
        assert hit.vel_z is None, "vel_z should be None when VELOCITY flag not set"

    print(f"[OK] Flag presence (DISTANCE only): {len(result.entities)} hits, distance-only fields populated")


def test_flag_presence_all_basic():
    """
    Test that POSITION|VELOCITY|DISTANCE populates all three field groups.
    """
    entities = create_test_entities(20, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 100.0
    flags = QUERY_FLAG_POSITION | QUERY_FLAG_VELOCITY | QUERY_FLAG_DISTANCE

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) > 0, "Should detect entities in cone"

    for hit in result.entities:
        # All fields should be present
        assert hit.distance is not None, f"Distance missing for {hit.entity_id}"
        assert hit.pos_x is not None, f"pos_x missing for {hit.entity_id}"
        assert hit.pos_y is not None, f"pos_y missing for {hit.entity_id}"
        assert hit.pos_z is not None, f"pos_z missing for {hit.entity_id}"
        assert hit.vel_x is not None, f"vel_x missing for {hit.entity_id}"
        assert hit.vel_y is not None, f"vel_y missing for {hit.entity_id}"
        assert hit.vel_z is not None, f"vel_z missing for {hit.entity_id}"

        # entity_id and species_id always present
        assert hit.entity_id.startswith("test-"), f"Invalid entity_id: {hit.entity_id}"
        assert hit.species_id == "sp-test", f"Invalid species_id: {hit.species_id}"

    print(f"[OK] Flag presence (all basic): {len(result.entities)} hits, all fields populated")


def test_determinism():
    """
    Test that two queries with same input produce identical results.

    Results should be deterministic: same entities, same order, same field values.
    """
    entities = create_test_entities(50, seed=42)

    # Run query twice with same parameters
    adapter1 = SpatialIndexAdapter()
    adapter1.build(entities)

    adapter2 = SpatialIndexAdapter()
    adapter2.build(entities)

    origin = np.array([25.0, -25.0, 0.0])
    direction = np.array([1.0, 0.5, 0.0])
    angle_deg = 30.0
    range_m = 60.0
    flags = QUERY_FLAG_POSITION | QUERY_FLAG_VELOCITY | QUERY_FLAG_DISTANCE
    timestamp = 123.456  # Fixed timestamp

    result1 = adapter1.query_cone(origin, direction, angle_deg, range_m, flags, timestamp)
    result2 = adapter2.query_cone(origin, direction, angle_deg, range_m, flags, timestamp)

    # Convert to dicts for comparison
    dict1 = result1.to_dict()
    dict2 = result2.to_dict()

    # Check metadata
    assert dict1['timestamp'] == dict2['timestamp'], "Timestamps should match"
    assert dict1['query_origin'] == dict2['query_origin'], "Query origins should match"

    # Check entity count
    assert len(dict1['entities']) == len(dict2['entities']), \
        f"Entity counts differ: {len(dict1['entities'])} vs {len(dict2['entities'])}"

    # Check each entity (order and values)
    for i, (e1, e2) in enumerate(zip(dict1['entities'], dict2['entities'])):
        assert e1 == e2, f"Entity {i} differs:\n{e1}\n vs \n{e2}"

    print(f"[OK] Determinism: {len(result1.entities)} hits, identical across runs")


def test_edge_case_zero_direction():
    """
    Test that zero-length direction vector returns empty result without crashing.
    """
    entities = create_test_entities(10, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 0.0])  # Zero-length
    angle_deg = 45.0
    range_m = 50.0
    flags = QUERY_FLAG_DISTANCE

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 0, "Zero-length direction should return empty result"
    assert result.query_origin == (0.0, 0.0, 0.0), "Query origin should be preserved"

    print("[OK] Edge case: zero-length direction handled")


def test_edge_case_entity_at_origin():
    """
    Test that entity at query origin is included correctly.
    """
    # Create entity exactly at origin
    entity = Entity(
        instance_id="at-origin",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile']
    )

    adapter = SpatialIndexAdapter()
    adapter.build([entity])

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 10.0
    flags = QUERY_FLAG_DISTANCE

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    # Entity at origin should be included (distance = 0)
    assert len(result.entities) == 1, "Entity at origin should be detected"
    assert result.entities[0].entity_id == "at-origin"
    assert result.entities[0].distance == 0.0, "Distance should be 0 for entity at origin"

    print("[OK] Edge case: entity at origin handled (distance=0)")


def test_result_ordering():
    """
    Test that results are ordered by distance, then entity_id.
    """
    # Create entities at known distances
    entities = []
    distances_expected = [5.0, 5.0, 10.0, 15.0, 15.0, 20.0]  # Some duplicates for tie-breaking test

    for i, dist in enumerate(distances_expected):
        entity = Entity(
            instance_id=f"ent-{i:03d}",  # Lexicographic ordering for tie-breaking
            species_id="sp-test",
            biome_id="test-biome",
            position=np.array([dist, 0.0, 0.0]),  # Along +X axis
            velocity=np.array([0.0, 0.0, 0.0]),
            size_factor=1.0,
            tags=['mobile']
        )
        entities.append(entity)

    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 10.0  # Narrow cone along +X
    range_m = 50.0
    flags = QUERY_FLAG_DISTANCE

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == len(entities), "All entities should be in narrow +X cone"

    # Check ordering: distance ascending, then entity_id lexicographic
    prev_distance = -1.0
    prev_entity_id = ""

    for hit in result.entities:
        if hit.distance > prev_distance:
            # Distance increased, reset entity_id tracking
            prev_distance = hit.distance
            prev_entity_id = hit.entity_id
        elif hit.distance == prev_distance:
            # Same distance, check entity_id ordering
            assert hit.entity_id > prev_entity_id, \
                f"Entity IDs not ordered at same distance: {prev_entity_id} vs {hit.entity_id}"
            prev_entity_id = hit.entity_id
        else:
            # Distance decreased - ordering violation
            assert False, f"Distances not ordered: {prev_distance} -> {hit.distance}"

    print(f"[OK] Result ordering: {len(result.entities)} hits correctly ordered by distance, then entity_id")


def test_performance_basic():
    """
    Test query performance at different entity counts.

    Target: ≤2ms at 1000 entities for typical cone queries.
    Note: This is an observational test, not a hard assertion (performance varies by machine).
    """
    entity_counts = [143, 500, 1000]
    timings = {}

    for count in entity_counts:
        entities = create_test_entities(count, seed=42)
        adapter = SpatialIndexAdapter()
        adapter.build(entities)

        origin = np.array([50.0, -50.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        angle_deg = 30.0
        range_m = 80.0
        flags = QUERY_FLAG_POSITION | QUERY_FLAG_VELOCITY | QUERY_FLAG_DISTANCE

        # Warm-up query
        adapter.query_cone(origin, direction, angle_deg, range_m, flags)

        # Timed queries (average over 10 runs)
        start = time.perf_counter()
        for _ in range(10):
            result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)
        elapsed = (time.perf_counter() - start) / 10.0 * 1000.0  # ms per query

        timings[count] = elapsed
        hit_count = len(result.entities)

        print(f"  {count:4d} entities: {elapsed:6.3f} ms/query ({hit_count} hits)")

    # Observational: log if performance exceeds target
    if timings[1000] > 2.0:
        print(f"  [WARN] Performance at 1000 entities ({timings[1000]:.3f} ms) exceeds 2ms target")
    else:
        print(f"  [OK] Performance at 1000 entities ({timings[1000]:.3f} ms) within 2ms target")


def test_acoustic_flag_only():
    """
    Test that ACOUSTIC flag populates only acoustic fields, not bioluminescent/optical.
    """
    entities = create_test_entities(20, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 100.0
    flags = QUERY_FLAG_ACOUSTIC

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) > 0, "Should detect entities in cone"

    for hit in result.entities:
        # Acoustic fields should be present
        assert hit.acoustic_amplitude is not None, f"acoustic_amplitude missing for {hit.entity_id}"
        assert 0.0 <= hit.acoustic_amplitude <= 1.0, f"Amplitude out of range: {hit.acoustic_amplitude}"
        # With default multiplier=1.0, amplitude should match base (0.5)
        assert abs(hit.acoustic_amplitude - 0.5) < 0.01, f"Unexpected amplitude: {hit.acoustic_amplitude}"
        assert hit.acoustic_peak_hz == 100.0, f"Unexpected peak_hz: {hit.acoustic_peak_hz}"

        # Bioluminescent and optical should NOT be present
        assert hit.bioluminescent_intensity is None, "bioluminescent_intensity should be None"
        assert hit.bioluminescent_wavelength_nm is None, "bioluminescent_wavelength_nm should be None"
        assert hit.optical_intensity is None, "optical_intensity should be None"
        assert hit.optical_wavelength_nm is None, "optical_wavelength_nm should be None"
        assert hit.optical_components is None, "optical_components should be None"

    print(f"[OK] Acoustic flag only: {len(result.entities)} hits, acoustic fields populated")


def test_bioluminescent_flag_only():
    """
    Test that BIOLUMINESCENT flag populates only bioluminescent fields, not optical.
    """
    entities = create_test_entities(20, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 100.0
    flags = QUERY_FLAG_BIOLUMINESCENT

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) > 0, "Should detect entities in cone"

    for hit in result.entities:
        # Bioluminescent fields should be present
        assert hit.bioluminescent_intensity is not None, f"bioluminescent_intensity missing for {hit.entity_id}"
        assert 0.0 <= hit.bioluminescent_intensity <= 1.0, f"Intensity out of range: {hit.bioluminescent_intensity}"
        # With default multiplier=1.0, intensity should match base (0.3)
        assert abs(hit.bioluminescent_intensity - 0.3) < 0.01, f"Unexpected intensity: {hit.bioluminescent_intensity}"
        assert hit.bioluminescent_wavelength_nm == 500.0, f"Unexpected wavelength: {hit.bioluminescent_wavelength_nm}"

        # Optical should NOT be present
        assert hit.optical_intensity is None, "optical_intensity should be None"
        assert hit.optical_wavelength_nm is None, "optical_wavelength_nm should be None"
        assert hit.optical_components is None, "optical_components should be None"

    print(f"[OK] Bioluminescent flag only: {len(result.entities)} hits, bioluminescent fields populated")


def test_optical_flag_only():
    """
    Test that OPTICAL flag populates optical fields + components, not bioluminescent.
    """
    entities = create_test_entities(20, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 100.0
    flags = QUERY_FLAG_OPTICAL

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) > 0, "Should detect entities in cone"

    for hit in result.entities:
        # Optical fields should be present
        assert hit.optical_intensity is not None, f"optical_intensity missing for {hit.entity_id}"
        assert 0.0 <= hit.optical_intensity <= 1.0, f"Intensity out of range: {hit.optical_intensity}"
        # Should match bioluminescent base (0.3)
        assert abs(hit.optical_intensity - 0.3) < 0.01, f"Unexpected intensity: {hit.optical_intensity}"
        assert hit.optical_wavelength_nm == 500.0, f"Unexpected wavelength: {hit.optical_wavelength_nm}"
        assert hit.optical_components == ['bioluminescent'], f"Unexpected components: {hit.optical_components}"

        # Bioluminescent should NOT be present
        assert hit.bioluminescent_intensity is None, "bioluminescent_intensity should be None"
        assert hit.bioluminescent_wavelength_nm is None, "bioluminescent_wavelength_nm should be None"

    print(f"[OK] Optical flag only: {len(result.entities)} hits, optical fields + components populated")


def test_dual_optical_flags():
    """
    Test that both BIOLUMINESCENT and OPTICAL flags populate both field sets with equal values.
    """
    entities = create_test_entities(20, seed=42)
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 100.0
    flags = QUERY_FLAG_BIOLUMINESCENT | QUERY_FLAG_OPTICAL

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) > 0, "Should detect entities in cone"

    for hit in result.entities:
        # Both bioluminescent and optical should be present
        assert hit.bioluminescent_intensity is not None, "bioluminescent_intensity missing"
        assert hit.bioluminescent_wavelength_nm is not None, "bioluminescent_wavelength_nm missing"
        assert hit.optical_intensity is not None, "optical_intensity missing"
        assert hit.optical_wavelength_nm is not None, "optical_wavelength_nm missing"
        assert hit.optical_components is not None, "optical_components missing"

        # Values should be equal
        assert hit.bioluminescent_intensity == hit.optical_intensity, \
            f"Intensities differ: bio={hit.bioluminescent_intensity}, optical={hit.optical_intensity}"
        assert hit.bioluminescent_wavelength_nm == hit.optical_wavelength_nm, \
            f"Wavelengths differ: bio={hit.bioluminescent_wavelength_nm}, optical={hit.optical_wavelength_nm}"
        assert hit.optical_components == ['bioluminescent'], f"Unexpected components: {hit.optical_components}"

    print(f"[OK] Dual optical flags: {len(result.entities)} hits, both sets populated with equal values")


def test_wavelength_none_when_zero_intensity():
    """
    Test that wavelength_nm is None when intensity is 0 (sensor reading semantic).
    """
    # Create entity with zero bioluminescent multiplier
    entity = Entity(
        instance_id="zero-intensity",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile'],
        base_emissions={
            'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0},
            'bioluminescent': {'intensity': 0.5, 'wavelength_nm': 500.0}
        }
    )
    # Set multiplier to 0 (behavior-driven emission suppression)
    entity.emission_multipliers['bioluminescent'] = 0.0

    adapter = SpatialIndexAdapter()
    adapter.build([entity])

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 50.0
    flags = QUERY_FLAG_BIOLUMINESCENT | QUERY_FLAG_OPTICAL

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 1, "Should detect entity"
    hit = result.entities[0]

    # Intensity should be 0
    assert hit.bioluminescent_intensity == 0.0, f"Expected intensity=0, got {hit.bioluminescent_intensity}"
    assert hit.optical_intensity == 0.0, f"Expected intensity=0, got {hit.optical_intensity}"

    # Wavelength should be None (sensor reading: no photons = no wavelength)
    assert hit.bioluminescent_wavelength_nm is None, "wavelength_nm should be None when intensity=0"
    assert hit.optical_wavelength_nm is None, "optical_wavelength_nm should be None when intensity=0"

    print("[OK] Wavelength None when zero intensity: sensor reading semantic verified")


def test_acoustic_multiplier_scaling():
    """
    Test that acoustic amplitude scales with multiplier and is clamped to [0,1].
    """
    # Create entity with custom multiplier
    entity = Entity(
        instance_id="loud-entity",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile'],
        base_emissions={
            'acoustic': {'amplitude': 0.6, 'peak_hz': 150.0},
            'bioluminescent': {'intensity': 0.3, 'wavelength_nm': 500.0}
        }
    )
    # Set multiplier to 2.0 (should clamp to 1.0)
    entity.emission_multipliers['acoustic'] = 2.0

    adapter = SpatialIndexAdapter()
    adapter.build([entity])

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 50.0
    flags = QUERY_FLAG_ACOUSTIC

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 1, "Should detect entity"
    hit = result.entities[0]

    # Amplitude should be clamped to 1.0 (0.6 * 2.0 = 1.2, clamped)
    assert hit.acoustic_amplitude == 1.0, f"Expected clamped amplitude=1.0, got {hit.acoustic_amplitude}"
    assert hit.acoustic_peak_hz == 150.0, f"Unexpected peak_hz: {hit.acoustic_peak_hz}"

    print("[OK] Acoustic multiplier scaling: amplitude clamped to [0,1]")


def test_bioluminescent_multiplier_scaling():
    """
    Test that bioluminescent intensity scales with multiplier and is clamped to [0,1].
    """
    # Create entity with custom multiplier
    entity = Entity(
        instance_id="bright-entity",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile'],
        base_emissions={
            'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0},
            'bioluminescent': {'intensity': 0.7, 'wavelength_nm': 480.0}
        }
    )
    # Set multiplier to 0.5 (dim behavior)
    entity.emission_multipliers['bioluminescent'] = 0.5

    adapter = SpatialIndexAdapter()
    adapter.build([entity])

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 50.0
    flags = QUERY_FLAG_BIOLUMINESCENT | QUERY_FLAG_OPTICAL

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 1, "Should detect entity"
    hit = result.entities[0]

    # Intensity should be scaled (0.7 * 0.5 = 0.35)
    expected_intensity = 0.35
    assert abs(hit.bioluminescent_intensity - expected_intensity) < 0.01, \
        f"Expected intensity={expected_intensity}, got {hit.bioluminescent_intensity}"
    assert abs(hit.optical_intensity - expected_intensity) < 0.01, \
        f"Expected optical_intensity={expected_intensity}, got {hit.optical_intensity}"

    print("[OK] Bioluminescent multiplier scaling: intensity scaled correctly")


def test_performance_all_channels():
    """
    Performance test with all emission channels enabled.
    Target: ≤2ms at 1000 entities (Step 1 used 0.508ms, expect <1.5ms with channels).
    """
    entity_counts = [143, 500, 1000]
    timings = {}

    for count in entity_counts:
        entities = create_test_entities(count, seed=42)
        adapter = SpatialIndexAdapter()
        adapter.build(entities)

        origin = np.array([50.0, -50.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        angle_deg = 30.0
        range_m = 80.0
        flags = QUERY_FLAG_POSITION | QUERY_FLAG_VELOCITY | QUERY_FLAG_DISTANCE | \
                QUERY_FLAG_ACOUSTIC | QUERY_FLAG_BIOLUMINESCENT | QUERY_FLAG_OPTICAL

        # Warm-up query
        adapter.query_cone(origin, direction, angle_deg, range_m, flags)

        # Timed queries (average over 10 runs)
        start = time.perf_counter()
        for _ in range(10):
            result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)
        elapsed = (time.perf_counter() - start) / 10.0 * 1000.0  # ms per query

        timings[count] = elapsed
        hit_count = len(result.entities)

        print(f"  {count:4d} entities: {elapsed:6.3f} ms/query ({hit_count} hits, all channels)")

    # Check if performance is within target
    if timings[1000] > 2.0:
        print(f"  [WARN] Performance at 1000 entities ({timings[1000]:.3f} ms) exceeds 2ms target")
    else:
        print(f"  [OK] Performance at 1000 entities ({timings[1000]:.3f} ms) within 2ms target")


def test_thermal_linear_falloff():
    """
    Test thermal linear falloff: base_delta at center, 0.0 at influence boundary.
    """
    # Create single entity near a vent
    entity = Entity(
        instance_id="thermal-test",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([5.0, 0.0, 0.0]),  # 5m from vent center
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile'],
        base_emissions={'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0}, 'bioluminescent': {'intensity': 0.3, 'wavelength_nm': 500.0}}
    )

    adapter = SpatialIndexAdapter()

    # Setup thermal provider: vent at origin with 10m influence, 5.0°C base_delta
    thermal_centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    thermal_base_deltas = np.array([5.0], dtype=np.float64)
    thermal_influences = np.array([10.0], dtype=np.float64)

    adapter.build([entity], thermal_centers=thermal_centers,
                  thermal_base_deltas=thermal_base_deltas,
                  thermal_influences=thermal_influences)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 20.0
    flags = QUERY_FLAG_THERMAL | QUERY_FLAG_DISTANCE

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 1, "Should detect entity"
    hit = result.entities[0]

    # At 5m from center, with 10m influence: delta = 5.0 × (1 - 5/10) = 2.5°C
    expected_delta = 5.0 * (1.0 - 5.0/10.0)
    assert hit.thermal_temperature_delta is not None, "Thermal delta should be present"
    assert abs(hit.thermal_temperature_delta - expected_delta) < 0.01, \
        f"Expected delta={expected_delta}, got {hit.thermal_temperature_delta}"

    print(f"[OK] Thermal linear falloff: 5m from vent = {hit.thermal_temperature_delta:.2f}C (expected {expected_delta:.2f}C)")


def test_thermal_nearest_vent():
    """
    Test that nearest vent is chosen when multiple vents present.
    """
    # Entity at position [15, 0, 0] - closer to vent2 at [20,0,0] than vent1 at [0,0,0]
    entity = Entity(
        instance_id="multi-vent-test",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([15.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile'],
        base_emissions={'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0}, 'bioluminescent': {'intensity': 0.3, 'wavelength_nm': 500.0}}
    )

    adapter = SpatialIndexAdapter()

    # Two vents with different base_deltas
    thermal_centers = np.array([
        [0.0, 0.0, 0.0],   # Vent1: 15m away
        [20.0, 0.0, 0.0]   # Vent2: 5m away (nearest)
    ], dtype=np.float64)
    thermal_base_deltas = np.array([10.0, 8.0], dtype=np.float64)  # Different deltas
    thermal_influences = np.array([20.0, 15.0], dtype=np.float64)

    adapter.build([entity], thermal_centers=thermal_centers,
                  thermal_base_deltas=thermal_base_deltas,
                  thermal_influences=thermal_influences)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 30.0
    flags = QUERY_FLAG_THERMAL

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 1, "Should detect entity"
    hit = result.entities[0]

    # Nearest is vent2 (5m away), base_delta=8.0, influence=15.0
    # delta = 8.0 × (1 - 5/15) = 8.0 × 0.667 = 5.33°C
    expected_delta = 8.0 * (1.0 - 5.0/15.0)
    assert hit.thermal_temperature_delta is not None, "Thermal delta should be present"
    assert abs(hit.thermal_temperature_delta - expected_delta) < 0.01, \
        f"Expected nearest vent contribution={expected_delta:.2f}, got {hit.thermal_temperature_delta:.2f}"

    print(f"[OK] Thermal nearest vent: chose vent2 (5m) over vent1 (15m), delta={hit.thermal_temperature_delta:.2f}C")


def test_thermal_no_vents():
    """
    Test that thermal field is omitted when M=0 (no vents in biome).
    """
    entity = Entity(
        instance_id="no-vents-test",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile'],
        base_emissions={'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0}, 'bioluminescent': {'intensity': 0.3, 'wavelength_nm': 500.0}}
    )

    adapter = SpatialIndexAdapter()
    # Build with no thermal providers (M=0)
    adapter.build([entity])

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 20.0
    flags = QUERY_FLAG_THERMAL

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 1, "Should detect entity"
    hit = result.entities[0]

    # M=0: thermal field should be None (omitted from to_dict)
    assert hit.thermal_temperature_delta is None, "Thermal delta should be None when M=0"

    # Verify omission from to_dict
    hit_dict = hit.to_dict()
    assert 'thermal_temperature_delta' not in hit_dict, "Thermal field should be omitted from to_dict when None"

    print("[OK] Thermal no vents: M=0, field omitted from to_dict")


def test_thermal_outside_influence():
    """
    Test that thermal_temperature_delta is 0.0 when outside all vent influence (M>0).
    """
    # Entity at [50, 0, 0] - far from vent at origin
    entity = Entity(
        instance_id="outside-influence",
        species_id="sp-test",
        biome_id="test-biome",
        position=np.array([50.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        size_factor=1.0,
        tags=['mobile'],
        base_emissions={'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0}, 'bioluminescent': {'intensity': 0.3, 'wavelength_nm': 500.0}}
    )

    adapter = SpatialIndexAdapter()

    # Vent at origin with 10m influence
    thermal_centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    thermal_base_deltas = np.array([5.0], dtype=np.float64)
    thermal_influences = np.array([10.0], dtype=np.float64)

    adapter.build([entity], thermal_centers=thermal_centers,
                  thermal_base_deltas=thermal_base_deltas,
                  thermal_influences=thermal_influences)

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    angle_deg = 45.0
    range_m = 60.0
    flags = QUERY_FLAG_THERMAL

    result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)

    assert len(result.entities) == 1, "Should detect entity"
    hit = result.entities[0]

    # At 50m from vent with 10m influence: delta = 5.0 × (1 - 50/10) = negative, clamped to 0.0
    assert hit.thermal_temperature_delta is not None, "Thermal delta should be present (M>0)"
    assert hit.thermal_temperature_delta == 0.0, \
        f"Expected delta=0.0 outside influence, got {hit.thermal_temperature_delta}"

    # Verify included in to_dict (M>0, field present even if 0.0)
    hit_dict = hit.to_dict()
    assert 'thermal_temperature_delta' in hit_dict, "Thermal field should be present in to_dict when M>0"
    assert hit_dict['thermal_temperature_delta'] == 0.0, "Thermal delta should be 0.0 in dict"

    print("[OK] Thermal outside influence: M>0, distance > influence = delta=0.0 (included in to_dict)")


def test_thermal_performance():
    """
    Performance test with THERMAL flag enabled.
    Target: sensor_ms < 2ms @ 1000 entities with all channels including thermal.
    """
    entity_counts = [143, 500, 1000]
    timings = {}

    # Setup thermal providers (12 vents, matching vent-field-alpha.yaml)
    thermal_centers = []
    thermal_base_deltas = []
    thermal_influences = []
    for i in range(12):
        x = (i - 6) * 30.0
        y = -100.0
        z = (i % 3 - 1) * 40.0
        thermal_centers.append([x, y, z])
        thermal_base_deltas.append(5.0)  # Default base_delta
        thermal_influences.append(10.0)  # 10m influence

    thermal_centers_arr = np.array(thermal_centers, dtype=np.float64)
    thermal_base_deltas_arr = np.array(thermal_base_deltas, dtype=np.float64)
    thermal_influences_arr = np.array(thermal_influences, dtype=np.float64)

    for count in entity_counts:
        entities = create_test_entities(count, seed=42)
        adapter = SpatialIndexAdapter()
        adapter.build(entities, thermal_centers=thermal_centers_arr,
                      thermal_base_deltas=thermal_base_deltas_arr,
                      thermal_influences=thermal_influences_arr)

        origin = np.array([50.0, -50.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        angle_deg = 30.0
        range_m = 80.0
        flags = QUERY_FLAG_POSITION | QUERY_FLAG_VELOCITY | QUERY_FLAG_DISTANCE | \
                QUERY_FLAG_ACOUSTIC | QUERY_FLAG_BIOLUMINESCENT | QUERY_FLAG_OPTICAL | \
                QUERY_FLAG_THERMAL

        # Warm-up query
        adapter.query_cone(origin, direction, angle_deg, range_m, flags)

        # Timed queries (average over 10 runs)
        start = time.perf_counter()
        for _ in range(10):
            result = adapter.query_cone(origin, direction, angle_deg, range_m, flags)
        elapsed = (time.perf_counter() - start) / 10.0 * 1000.0  # ms per query

        timings[count] = elapsed
        hit_count = len(result.entities)

        print(f"  {count:4d} entities: {elapsed:6.3f} ms/query ({hit_count} hits, all channels + thermal)")

    # Check if performance is within target
    if timings[1000] > 2.0:
        print(f"  [WARN] Performance at 1000 entities ({timings[1000]:.3f} ms) exceeds 2ms target")
    else:
        print(f"  [OK] Performance at 1000 entities ({timings[1000]:.3f} ms) within 2ms target")


if __name__ == "__main__":
    print("Running sensor query tests (Step 1 + Phase A + Phase C)...\n")

    print("=== Step 1: Basic fields ===")
    test_cone_range_selectivity()
    test_cone_angle_selectivity()
    test_flag_presence_distance_only()
    test_flag_presence_all_basic()
    test_determinism()
    test_edge_case_zero_direction()
    test_edge_case_entity_at_origin()
    test_result_ordering()

    print("\n=== Phase A: Emission channels ===")
    test_acoustic_flag_only()
    test_bioluminescent_flag_only()
    test_optical_flag_only()
    test_dual_optical_flags()
    test_wavelength_none_when_zero_intensity()
    test_acoustic_multiplier_scaling()
    test_bioluminescent_multiplier_scaling()

    print("\n=== Phase C: Thermal channel ===")
    test_thermal_linear_falloff()
    test_thermal_nearest_vent()
    test_thermal_no_vents()
    test_thermal_outside_influence()

    print("\n=== Performance tests ===")
    test_performance_basic()
    print("\nPhase A performance (all channels):")
    test_performance_all_channels()
    print("\nPhase C performance (all channels + thermal):")
    test_thermal_performance()

    print("\n[PASS] All sensor query tests passed (23 tests total)!")
