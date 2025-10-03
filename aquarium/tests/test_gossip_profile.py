"""
Profiling test for gossip performance analysis.
"""

import numpy as np
import time
from aquarium.entity import Entity
from aquarium.spatial_queries import SpatialIndexAdapter
from aquarium.gossip import exchange_tokens


def create_test_entities(count: int, grid_spacing: float = 12.0, seed: int = 42):
    """Create test entities for profiling."""
    np.random.seed(seed)
    entities = []

    for i in range(count):
        x = (i % 10) * grid_spacing + np.random.uniform(-0.5, 0.5)
        y = -((i // 10) % 10) * grid_spacing + np.random.uniform(-0.5, 0.5)
        z = (i // 100) * grid_spacing + np.random.uniform(-0.5, 0.5)

        vel = np.random.uniform(-0.5, 0.5, size=3)

        base_emissions = {
            'acoustic': {'amplitude': 0.5, 'peak_hz': 100.0},
            'bioluminescent': {'intensity': 0.3, 'wavelength_nm': 500.0}
        }

        entity = Entity(
            instance_id=f"profile-{i:04d}",
            species_id="sp-test",
            biome_id="test-biome",
            position=np.array([x, y, z], dtype=np.float64),
            velocity=vel,
            size_factor=1.0,
            tags=['mobile'],
            base_emissions=base_emissions,
            knowledge_tokens={}
        )
        entities.append(entity)

    return entities


def profile_gossip():
    """Profile gossip performance with detailed timing."""
    print("=" * 70)
    print("Gossip Performance Profiling")
    print("=" * 70)

    for entity_count in [143, 500, 1000]:
        print(f"\n[Entity Count: {entity_count}]")

        entities = create_test_entities(entity_count, grid_spacing=12.0, seed=42)

        # Seed 10% with tokens (Phase 2 schema)
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

        # Time adapter build
        t0 = time.perf_counter()
        adapter = SpatialIndexAdapter()
        adapter.build(entities)
        build_ms = (time.perf_counter() - t0) * 1000

        # Instrument gossip with detailed timing
        print(f"  Adapter build: {build_ms:.3f}ms")

        # Time gossip with multiple runs for stability
        times = []
        for run in range(5):
            t0 = time.perf_counter()
            result = exchange_tokens(entities, adapter, species_registry, current_tick=run)
            gossip_ms = (time.perf_counter() - t0) * 1000
            times.append(gossip_ms)

        avg_ms = np.mean(times)
        std_ms = np.std(times)
        min_ms = np.min(times)
        max_ms = np.max(times)

        print(f"  Gossip time: {avg_ms:.3f}ms ± {std_ms:.3f}ms (min={min_ms:.3f}, max={max_ms:.3f})")
        print(f"  Pairs: {result['pairs_count']}, Exchanges: {result['exchanges_count']}")
        print(f"  Budget: {'PASS' if avg_ms < 2.0 else 'FAIL'} (target <2ms)")


def profile_gossip_internals():
    """Profile internal gossip operations."""
    print("\n" + "=" * 70)
    print("Internal Gossip Operation Profiling (1000 entities)")
    print("=" * 70)

    entities = create_test_entities(1000, grid_spacing=12.0, seed=42)

    # Seed 10% with tokens (Phase 2 schema)
    for i in range(0, 1000, 10):
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

    # Manual profiling of gossip internals
    from aquarium.gossip import load_token_definitions
    from collections import defaultdict
    from scipy.spatial import cKDTree

    t0 = time.perf_counter()
    token_defs = load_token_definitions()
    t1 = time.perf_counter()
    print(f"\n  1. load_token_definitions: {(t1-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    row_to_entity = {i: e for i, e in enumerate(adapter._entities)}
    t1 = time.perf_counter()
    print(f"  2. build row_to_entity dict: {(t1-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    positions = np.array([e.position for e in adapter._entities], dtype=np.float64)
    t1 = time.perf_counter()
    print(f"  3. extract positions array: {(t1-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    range_groups = defaultdict(list)
    for row, entity in row_to_entity.items():
        gossip_range = 15.0  # Fallback
        if gossip_range > 0:
            range_groups[gossip_range].append(row)
    t1 = time.perf_counter()
    print(f"  4. group by range: {(t1-t0)*1000:.3f}ms")

    # Process largest range group
    gossip_range = 15.0
    group_rows = range_groups[gossip_range]
    print(f"\n  Range group {gossip_range}m: {len(group_rows)} entities")

    t0 = time.perf_counter()
    R = np.array(group_rows, dtype=np.int32)
    group_positions = positions[R]
    t1 = time.perf_counter()
    print(f"    a. extract group positions: {(t1-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    tree_R = cKDTree(group_positions)
    t1 = time.perf_counter()
    print(f"    b. build group KD-tree: {(t1-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    pairs_set = tree_R.query_pairs(r=gossip_range, output_type='set')
    t1 = time.perf_counter()
    print(f"    c. query_pairs: {(t1-t0)*1000:.3f}ms ({len(pairs_set)} pairs)")

    if pairs_set:
        t0 = time.perf_counter()
        pairs_list = list(pairs_set)
        pi = np.array([p[0] for p in pairs_list], dtype=np.int32)
        pj = np.array([p[1] for p in pairs_list], dtype=np.int32)
        t1 = time.perf_counter()
        print(f"    d. convert pairs to arrays: {(t1-t0)*1000:.3f}ms")

        t0 = time.perf_counter()
        d = np.linalg.norm(group_positions[pi] - group_positions[pj], axis=1)
        t1 = time.perf_counter()
        print(f"    e. compute distances: {(t1-t0)*1000:.3f}ms")

        t0 = time.perf_counter()
        src = np.concatenate([pi, pj])
        dst = np.concatenate([pj, pi])
        dist = np.concatenate([d, d])
        src_rows = R[src]
        dst_rows = R[dst]
        t1 = time.perf_counter()
        print(f"    f. build bidirectional edges: {(t1-t0)*1000:.3f}ms")

        t0 = time.perf_counter()
        src_ids = np.array([adapter._entities[row].instance_id for row in src_rows])
        dst_ids = np.array([adapter._entities[row].instance_id for row in dst_rows])
        t1 = time.perf_counter()
        print(f"    g. extract instance_ids: {(t1-t0)*1000:.3f}ms  <-- BOTTLENECK?")

        t0 = time.perf_counter()
        dtype = [('src', 'i4'), ('dist', 'f8'), ('dst_id', 'U64'), ('dst', 'i4')]
        edges = np.empty(len(src_rows), dtype=dtype)
        edges['src'] = src_rows
        edges['dist'] = dist
        edges['dst_id'] = dst_ids
        edges['dst'] = dst_rows
        t1 = time.perf_counter()
        print(f"    h. create structured array: {(t1-t0)*1000:.3f}ms")

        t0 = time.perf_counter()
        sorted_idx = np.argsort(edges, order=['src', 'dist', 'dst_id'])
        edges_sorted = edges[sorted_idx]
        t1 = time.perf_counter()
        print(f"    i. sort edges: {(t1-t0)*1000:.3f}ms")

        t0 = time.perf_counter()
        unique_srcs, first_idx, counts = np.unique(
            edges_sorted['src'],
            return_index=True,
            return_counts=True
        )
        t1 = time.perf_counter()
        print(f"    j. unique sources: {(t1-t0)*1000:.3f}ms")

        t0 = time.perf_counter()
        pairs_to_exchange = set()
        cap_per_entity = 2
        for i, src_row in enumerate(unique_srcs):
            start = first_idx[i]
            count = min(counts[i], cap_per_entity)
            for j in range(count):
                dst_row = edges_sorted['dst'][start + j]
                pair_key = tuple(sorted([src_row, dst_row]))
                pairs_to_exchange.add(pair_key)
        t1 = time.perf_counter()
        print(f"    k. select pairs (cap={cap_per_entity}): {(t1-t0)*1000:.3f}ms ({len(pairs_to_exchange)} pairs)")

    print("\n  Total internal operations profiled")


if __name__ == '__main__':
    profile_gossip()
    profile_gossip_internals()
