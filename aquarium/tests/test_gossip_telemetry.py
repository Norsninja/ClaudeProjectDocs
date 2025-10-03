"""
Tests for gossip network health telemetry (Phase 7: degree histogram).

Validates that degree histogram computation is:
- Correct (counts both endpoints of edges)
- Fast (O(E) with minimal overhead)
- Useful (distinguishes lattice vs sparse layouts)
"""

import numpy as np
import pytest
from aquarium.gossip import exchange_tokens
from aquarium.spatial_queries import SpatialIndexAdapter
from aquarium.entity import Entity


def create_test_entities(positions):
    """Create entities at specified positions."""
    entities = []
    for i, pos in enumerate(positions):
        entity = Entity(
            instance_id=f'e-{i}',
            species_id='sp-test',
            biome_id='biome-test',
            position=np.array(pos, dtype=np.float64),
            velocity=np.array([0.0, 0.0, 0.0]),
            size_factor=1.0,
            tags=['mobile'],
            base_emissions={
                'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0},
                'bioluminescent': {'intensity': 0.3, 'wavelength_nm': 500.0}
            },
            knowledge_tokens={}
        )
        entities.append(entity)
    return entities


def test_diagnostics_returned_from_exchange():
    """
    exchange_tokens returns telemetry dict with expected keys and types.
    Validates API contract for telemetry data.
    """
    positions = [(0.0, 0.0, -50.0), (5.0, 0.0, -50.0)]
    entities = create_test_entities(positions)

    entities[0].knowledge_tokens['ship_sentiment'] = {
        'kind': 'ship_sentiment',
        'value': 0.5,
        'version': 1,
        'last_tick': 0.0,
        'reliability': 1.0,
        'source': 'direct'
    }

    species_registry = {}
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    result = exchange_tokens(entities, adapter, species_registry, current_tick=1)

    # Check telemetry structure
    assert 'telemetry' in result
    telemetry = result['telemetry']

    assert 'degree_mean' in telemetry
    assert isinstance(telemetry['degree_mean'], float)

    assert 'degree_histogram' in telemetry
    hist = telemetry['degree_histogram']
    assert 'isolated' in hist
    assert 'sparse' in hist
    assert 'connected' in hist
    assert isinstance(hist['isolated'], int)
    assert isinstance(hist['sparse'], int)
    assert isinstance(hist['connected'], int)

    assert 'total_entities' in telemetry
    assert isinstance(telemetry['total_entities'], int)
    assert telemetry['total_entities'] == len(entities)

    # Histogram bins should sum to total
    assert hist['isolated'] + hist['sparse'] + hist['connected'] == len(entities)


def test_degree_histogram_two_nodes():
    """
    Sanity test: two nodes connected by single edge → both degrees = 1.
    Catches off-by-one errors in degree computation.
    """
    # Two entities 5m apart (within default gossip range 15m)
    positions = [
        (0.0, 0.0, -50.0),
        (5.0, 0.0, -50.0)
    ]
    entities = create_test_entities(positions)

    # Seed one token to enable gossip
    entities[0].knowledge_tokens['ship_sentiment'] = {
        'kind': 'ship_sentiment',
        'value': 0.5,
        'version': 1,
        'last_tick': 0.0,
        'reliability': 1.0,
        'source': 'direct'
    }

    species_registry = {}
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    result = exchange_tokens(entities, adapter, species_registry, current_tick=1)

    # Both entities should have degree = 1 (connected to each other)
    telemetry = result['telemetry']
    assert telemetry['total_entities'] == 2

    # With 2 entities, 1 edge, both have degree 1 → sparse bin
    assert telemetry['degree_histogram']['isolated'] == 0
    assert telemetry['degree_histogram']['sparse'] == 2  # Both have degree 1
    assert telemetry['degree_histogram']['connected'] == 0

    # Mean degree should be 1.0
    assert abs(telemetry['degree_mean'] - 1.0) < 1e-9


def test_degree_histogram_lattice_layout():
    """
    Entities in regular lattice have predictable degree distribution.
    Most entities should have 4-8 neighbors in interior of grid.
    """
    # Create 10×10 grid with 10m spacing (100 entities)
    positions = []
    for i in range(10):
        for j in range(10):
            positions.append((i * 10.0, j * 10.0, -50.0))

    entities = create_test_entities(positions)

    # Seed tokens across population
    for entity in entities:
        entity.knowledge_tokens['ship_sentiment'] = {
            'kind': 'ship_sentiment',
            'value': 0.0,
            'version': 1,
            'last_tick': 0.0,
            'reliability': 1.0,
            'source': 'direct'
        }

    species_registry = {}
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    result = exchange_tokens(entities, adapter, species_registry, current_tick=1)

    telemetry = result['telemetry']
    assert telemetry['total_entities'] == 100

    # In 10×10 lattice with gossip_range=15m, spacing=10m:
    # Diagonal neighbors are ~14.1m away, so they count
    # Most interior entities will have 8 neighbors (orthogonal + diagonal)

    # Isolated should be 0 (all connected)
    assert telemetry['degree_histogram']['isolated'] == 0

    # Relaxed assertion: degree_mean should be 4-8
    assert 4.0 <= telemetry['degree_mean'] <= 8.0

    # Less than 5% isolated
    pct_isolated = 100.0 * telemetry['degree_histogram']['isolated'] / 100
    assert pct_isolated < 5.0


def test_degree_histogram_sparse_layout():
    """
    Entities far apart have low degree distribution.
    Most entities should be isolated or have 1-2 neighbors.
    """
    # Create 100 entities spread across 500m × 500m (far apart)
    rng = np.random.RandomState(42)
    positions = []
    for _ in range(100):
        x = rng.uniform(0, 500)
        y = rng.uniform(0, 500)
        positions.append((x, y, -50.0))

    entities = create_test_entities(positions)

    # Seed tokens
    for entity in entities:
        entity.knowledge_tokens['ship_sentiment'] = {
            'kind': 'ship_sentiment',
            'value': 0.0,
            'version': 1,
            'last_tick': 0.0,
            'reliability': 1.0,
            'source': 'direct'
        }

    species_registry = {}
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    result = exchange_tokens(entities, adapter, species_registry, current_tick=1)

    telemetry = result['telemetry']
    assert telemetry['total_entities'] == 100

    # With 100 entities across 500×500m and gossip_range=15m,
    # most should be isolated (density is too low)
    pct_isolated = 100.0 * telemetry['degree_histogram']['isolated'] / 100
    assert pct_isolated > 50.0  # Majority isolated

    # Degree mean should be low
    assert telemetry['degree_mean'] < 2.0
