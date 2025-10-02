# Knowledge Gossip v2 - Phase 1 Design

**Date:** 2025-10-02
**Status:** Approved - Ready for Implementation
**Team:** Mike, Chronus, Senior Dev

---

## Problem Statement

**Original Implementation Issues:**
1. **k=2 neighbors below percolation threshold** - Network fragments into isolated clusters
2. **Directed edges from k-NN** - Asymmetric communication prevents full propagation
3. **Push-only protocol** - Stalls at ~10% coverage due to redundant transmissions
4. **float32 precision errors** - Test failures from precision loss
5. **Freshness attenuation (0.9×)** - Non-standard approach, conflicts with monotonic versioning

**Research Findings:**
- Percolation threshold for 2D k-NN graphs: k≥3 (Balister & Bollobás)
- Gossip protocols require bidirectional edges (undirected graphs)
- Push-pull hybrid is standard practice for fast convergence
- Monotonic version numbers preferred over decay-during-propagation
- cKDTree + NumPy vectorization is proven for 1000+ agents

**External Research:** `project/research/spatial_gossip_protocols_external_research_2025-10-02.md`

---

## Design Decisions

### 1. Neighbor Selection: Radius-Based

**Choice:** Use `Species.gossip_range_m` with fallback to `GOSSIP_FALLBACK_RANGE_M = 15.0`

**Rationale:**
- Fiction-aligned: entities sense within realistic range
- Automatic bidirectionality: if A within range of B, then B within range of A
- Uses existing Species field (already in schema)
- Creates undirected graph naturally

**Implementation:**
```python
gossip_range = species.gossip_range_m or GOSSIP_FALLBACK_RANGE_M
neighbor_lists = kdtree.query_ball_point(positions, r=gossip_range)
```

**Diagnostics:**
- Log/metric entities with < `MIN_GOSSIP_NEIGHBORS = 3` neighbors
- Helps design adjust densities for sparse pockets

### 2. Versioning: Hybrid (Version + Timestamp)

**Choice:** Store `version` (int64) + `last_tick` (float64) per token

**Rationale:**
- **version**: Monotonic counter for precedence during exchange
- **last_tick**: Timestamp for Phase 2 time-based decay
- Separates propagation logic (version) from lifecycle logic (decay)
- Aligns with distributed systems best practices (Lamport timestamps)

**Token Schema v2:**
```python
{
    'kind': 'ship_sentiment',
    'value': 0.8,          # float64 - knowledge value
    'version': 1,          # int64 - monotonic version counter
    'last_tick': 42.5,     # float64 - when last updated
    'source': 'direct'     # string - origin ('direct' | 'gossip')
}
```

**Migration:**
- Remove `freshness` field (was attenuation-based)
- Add compatibility shim: treat legacy tokens as `version=0, last_tick=0.0`

### 3. Protocol: Push-Pull Hybrid

**Choice:** Symmetric bidirectional exchange per undirected edge

**Rationale:**
- Research consensus: push-pull converges fastest (O(log n))
- Push dominates early (few informed → rapid spread)
- Pull accelerates late (many informed → fill gaps)
- Prevents stalls from redundant push-only transmissions

**Semantics:**
```python
# For each undirected edge (A, B):
# Push: if A.version > B.version, copy A → B
# Pull: if B.version > A.version, copy B → A
```

**Implementation:**
- Build undirected edge set: `{(min, max), ...}` (sorted tuples)
- Convert to arrays: `edge_a, edge_b` (deterministic ordering)
- Vectorized masks: `mask_push`, `mask_pull`
- Apply transfers with epsilon: `version[a] > version[b] + 1e-12`

### 4. Attenuation: Deferred to Phase 2

**Choice:** No attenuation during gossip propagation in Phase 1

**Rationale:**
- Version/timestamp copied **unchanged** during exchange
- Decay is a separate lifecycle concern (time-based, not hop-based)
- Research shows propagation should preserve version integrity
- `tokens.yaml` attenuation field ignored until Phase 2 reliability modeling

**Phase 2 Decay (Future):**
```python
# Time-based decay using freshness_rate from tokens.yaml
freshness = exp(-(current_tick - last_tick) * freshness_rate)
if freshness < eviction_threshold:
    del entity.knowledge_tokens[kind]
```

### 5. Data Types: float64 Everywhere

**Choice:** All arrays use `np.float64` (no float32)

**Rationale:**
- Eliminates precision errors (1.19e-8 test failures)
- Python float is float64 - mixing types causes casts
- Write-back uses `float(val)` to ensure float64

**Arrays:**
```python
val = np.zeros(N, dtype=np.float64)      # was float32
version = np.zeros(N, dtype=np.int64)    # new
last_tick = np.zeros(N, dtype=np.float64) # new
```

### 6. Determinism: Sorted Edge Ordering

**Choice:** Sort edges before building arrays

**Rationale:**
- Ensures identical array construction across runs
- Same seed → same neighbor discovery → same edge ordering
- Critical for deterministic gameplay

**Implementation:**
```python
edges = set()
for i, neighbors in enumerate(neighbor_lists):
    for j in neighbors:
        if i != j:
            edges.add(tuple(sorted([i, j])))  # canonical form

edges = sorted(edges)  # deterministic ordering
```

---

## Algorithm Pseudocode

```python
def exchange_tokens(entities, adapter, species_registry):
    """
    Phase 1: Vectorized push-pull gossip with radius-based neighbors.

    Note: Attenuation deferred to Phase 2. Version/timestamp copied unchanged.
    """

    # 1. Extract positions, build spatial index
    positions = [e.position for e in adapter._entities]
    kdtree = cKDTree(positions)

    # 2. Radius-based neighbor queries (per-species range)
    gossip_range = species.gossip_range_m or GOSSIP_FALLBACK_RANGE_M
    neighbor_lists = kdtree.query_ball_point(positions, r=gossip_range)

    # 3. Extract token state to arrays (float64, version, last_tick)
    N = len(entities)
    has = np.zeros(N, dtype=bool)
    val = np.zeros(N, dtype=np.float64)
    version = np.zeros(N, dtype=np.int64)
    last_tick = np.zeros(N, dtype=np.float64)

    for i, entity in enumerate(entities):
        token = entity.knowledge_tokens.get('ship_sentiment')
        if token:
            has[i] = True
            val[i] = token['value']
            # Compatibility shim for legacy tokens
            version[i] = token.get('version', 0)
            last_tick[i] = token.get('last_tick', 0.0)

    # 4. Build undirected edge set (sorted tuples, deterministic)
    edges = set()
    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            if i != j:
                edges.add(tuple(sorted([i, j])))

    edges = sorted(edges)
    edge_a = np.array([e[0] for e in edges], dtype=np.int32)
    edge_b = np.array([e[1] for e in edges], dtype=np.int32)

    # 5. Vectorized push-pull merge
    EPSILON = 1e-12

    # Push: a → b (a has newer version)
    mask_push = has[edge_a] & (~has[edge_b] | (version[edge_a] > version[edge_b] + EPSILON))

    # Pull: b → a (b has newer version)
    mask_pull = has[edge_b] & (~has[edge_a] | (version[edge_b] > version[edge_a] + EPSILON))

    # Apply push
    if np.any(mask_push):
        dst = edge_b[mask_push]
        src = edge_a[mask_push]
        has[dst] = True
        val[dst] = val[src]
        version[dst] = version[src]
        last_tick[dst] = last_tick[src]

    # Apply pull
    if np.any(mask_pull):
        dst = edge_a[mask_pull]
        src = edge_b[mask_pull]
        has[dst] = True
        val[dst] = val[src]
        version[dst] = version[src]
        last_tick[dst] = last_tick[src]

    # 6. Write back changed rows
    changed_rows = np.unique(np.concatenate([
        edge_b[mask_push] if np.any(mask_push) else [],
        edge_a[mask_pull] if np.any(mask_pull) else []
    ]))

    for row in changed_rows:
        entities[row].knowledge_tokens['ship_sentiment'] = {
            'kind': 'ship_sentiment',
            'value': float(val[row]),
            'version': int(version[row]),
            'last_tick': float(last_tick[row]),
            'source': 'gossip'
        }

    return {
        'exchanges_count': int(np.sum(mask_push) + np.sum(mask_pull)),
        'pairs_count': len(edges)
    }
```

---

## Test Updates

### Seed Format
```python
# Old (Phase 0):
entity.knowledge_tokens['ship_sentiment'] = {
    'kind': 'ship_sentiment',
    'value': 0.8,
    'freshness': 1.0,  # REMOVE
    'source': 'direct'
}

# New (Phase 1):
entity.knowledge_tokens['ship_sentiment'] = {
    'kind': 'ship_sentiment',
    'value': 0.8,
    'version': 1,      # ADD
    'last_tick': 0.0,  # ADD
    'source': 'direct'
}
```

### Expected Behavior
- **Coverage:** ≥95% in ~10 ticks (radius=15m, grid_spacing=10m)
- **Performance:** <2ms @ 1000 entities
- **Determinism:** Same seed → same results
- **Precision:** No float errors (all float64)

---

## Performance Expectations

**Current:** 2.6ms @ 1000 entities (with broken algorithm)

**Target:** <2ms @ 1000 entities (with correct algorithm)

**Profile:**
- cKDTree build: ~0.2-0.5ms
- Radius queries: ~0.5-1.0ms
- Vectorized merge: ~0.5-1.0ms
- Write-back: ~0.1-0.5ms

**Optimizations (if needed):**
- Cache tree if positions stable
- Sparse updates (only process entities with tokens)
- Numba JIT for merge logic

**Research Validation:**
- Vicsek model: "thousands of agents on standard laptop"
- Our 2.6ms already competitive with similar systems
- Bottleneck was algorithmic (k=2, directed, push-only), not computational

---

## Phase 2 Roadmap (Future)

**Deferred to Phase 2:**
1. Time-based decay using `freshness_rate` from tokens.yaml
2. Eviction when `freshness < eviction_threshold`
3. Reliability/attenuation modeling (separate from version propagation)
4. Multiple token kinds (currently ship_sentiment only)
5. Merge algorithms (weighted_average, most_recent)
6. Multi-species gossip ranges

**Phase 1 Scope:**
- Single token kind: ship_sentiment
- Version-based precedence only
- No decay during propagation
- Radius-based neighbors
- Push-pull symmetric exchange
- Float64 precision
- Deterministic edge ordering

---

## Diagnostics & Monitoring

**Log when entity has < MIN_GOSSIP_NEIGHBORS:**
```python
low_degree_count = sum(1 for n in neighbor_lists if len(n) - 1 < MIN_GOSSIP_NEIGHBORS)
if low_degree_count > 0:
    logger.warning(f"{low_degree_count} entities have < {MIN_GOSSIP_NEIGHBORS} neighbors")
```

**Metrics to track:**
- `exchanges_count`: Total transfers (push + pull)
- `pairs_count`: Unique undirected edges
- `coverage`: % entities with token
- `avg_neighbors`: Mean neighbors per entity
- `low_degree_count`: Entities with < 3 neighbors

---

## References

- Research report: `project/research/spatial_gossip_protocols_external_research_2025-10-02.md`
- Session handoff: `project/handoffs/session_010_2025-10-02_gossip_vectorized_near_budget.md`
- Original gossip implementation: `aquarium/gossip.py` (lines 1-255, to be rewritten)
- Token definitions: `data/knowledge/tokens.yaml`

---

**Approved by:** Mike, Senior Dev
**Implementation by:** Chronus
**Review required:** Post-implementation validation (coverage, performance, determinism)
