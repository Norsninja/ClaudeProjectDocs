"""
Multi-N performance validation for gossip Phase 2a.

Runs gossip at 143, 500, 1000, 2000 entities and reports median/p90.
Per SD guidance: single-threaded cKDTree, log-only for N>1000.
"""

# Pin threading for stable measurement
import os
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1'
})

import numpy as np
import time
import gc
from typing import List

from aquarium.entity import Entity
from aquarium.spatial_queries import SpatialIndexAdapter
from aquarium.gossip import exchange_tokens


def create_test_entities(count: int, grid_spacing: float = 12.0, seed: int = 42) -> List[Entity]:
    """Create test entities in grid pattern."""
    np.random.seed(seed)
    entities = []

    for i in range(count):
        x = (i % 10) * grid_spacing + np.random.uniform(-0.5, 0.5)
        y = -((i // 10) % 10) * grid_spacing + np.random.uniform(-0.5, 0.5)
        z = (i // 100) * grid_spacing + np.random.uniform(-0.5, 0.5)

        entity = Entity(
            instance_id=f"perf-{i:04d}",
            species_id="sp-test",
            biome_id="test-biome",
            position=np.array([x, y, z], dtype=np.float64),
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


def run_gossip_perf_test(entity_count: int, runs: int = 7) -> dict:
    """
    Run gossip performance test at given entity count.

    Args:
        entity_count: Number of entities to test
        runs: Number of test runs (default 7 for stable median)

    Returns:
        Dict with p50, p90, min, max, pairs, exchanges
    """
    entities = create_test_entities(entity_count, grid_spacing=12.0, seed=42)

    # Seed 10% with tokens (Phase 2a schema)
    for i in range(0, entity_count, 10):
        entities[i].knowledge_tokens['ship_sentiment'] = {
            'kind': 'ship_sentiment',
            'value': np.random.uniform(-1.0, 1.0),
            'version': 1,
            'last_tick': 0.0,
            'reliability': 1.0,
            'source': 'direct'
        }

    species_registry = {}
    adapter = SpatialIndexAdapter()
    adapter.build(entities)

    # Warmup
    exchange_tokens(entities, adapter, species_registry, current_tick=0)

    # Measure (GC disabled for stable timing)
    gc.collect()
    gc.disable()

    times_ns = []
    try:
        for tick in range(runs):
            start = time.perf_counter_ns()
            result = exchange_tokens(entities, adapter, species_registry, current_tick=tick+1)
            elapsed_ns = time.perf_counter_ns() - start
            times_ns.append(elapsed_ns)
    finally:
        gc.enable()

    # Statistics
    times_ms = np.array(times_ns) / 1_000_000
    p50_ms = np.percentile(times_ms, 50)
    p90_ms = np.percentile(times_ms, 90)
    min_ms = np.min(times_ms)
    max_ms = np.max(times_ms)

    return {
        'entity_count': entity_count,
        'runs': runs,
        'p50_ms': p50_ms,
        'p90_ms': p90_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'pairs': result['pairs_count'],
        'exchanges': result['exchanges_count']
    }


def main():
    """Run multi-N gossip performance validation."""
    print("=" * 80)
    print("Gossip Phase 2a Multi-N Performance Validation")
    print("=" * 80)
    print()

    # Test sizes per SD guidance
    test_sizes = [143, 500, 1000, 2000]

    results = []

    for entity_count in test_sizes:
        print(f"[N = {entity_count}]")

        result = run_gossip_perf_test(entity_count, runs=7)

        # Display
        print(f"  p50: {result['p50_ms']:.3f}ms")
        print(f"  p90: {result['p90_ms']:.3f}ms")
        print(f"  min: {result['min_ms']:.3f}ms, max: {result['max_ms']:.3f}ms")
        print(f"  Pairs: {result['pairs']}, Exchanges: {result['exchanges']}")

        # Validation
        if entity_count <= 1000:
            # Strict check for ≤1000 entities
            if result['p50_ms'] >= 2.0:
                print(f"  WARNING: p50 {result['p50_ms']:.3f}ms >= 2.0ms target!")
            else:
                headroom_pct = ((2.0 - result['p50_ms']) / 2.0) * 100
                print(f"  PASS: {headroom_pct:.1f}% headroom under 2ms target")
        else:
            # Log-only for >1000
            print(f"  (log-only, no assertion)")

        results.append(result)
        print()

    # Summary table
    print("=" * 80)
    print("Summary Table (for PROGRESS.md)")
    print("=" * 80)
    print()
    print("| Entities | p50 (ms) | p90 (ms) | Pairs | Exchanges |")
    print("|----------|----------|----------|-------|-----------|")
    for r in results:
        print(f"| {r['entity_count']:8d} | {r['p50_ms']:8.3f} | {r['p90_ms']:8.3f} | {r['pairs']:5d} | {r['exchanges']:9d} |")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
