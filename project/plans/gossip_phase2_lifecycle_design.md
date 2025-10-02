# Knowledge Gossip Phase 2: Lifecycle Design

**Created**: 2025-10-02
**Status**: Design Complete, Implementation Pending
**Prerequisites**: Phase 1 complete (radius push-pull, version+timestamp, SoA cache)

## Overview

Phase 2 adds token lifecycle management: decay over time, eviction when stale, capacity constraints, and reliability attenuation during propagation.

**Key Principles:**
- Tick-based (not wall-clock) for determinism
- Exponential decay (naturalistic, bounded)
- Vectorized computation (maintain <2ms budget)
- Deterministic eviction (stable ordering)

---

## 1. Decay Math

**Exponential Decay (Discrete per Tick):**

```python
age = current_tick - last_tick
freshness = exp(-freshness_rate · age)
reliability = exp(-reliability_rate · age)
```

**Properties:**
- **Bounded**: Always in [0, 1], no clamping needed
- **Deterministic**: Tick-based, no wall-clock jitter
- **Vectorizable**: `np.exp(-rate * age_array)` is O(N) fast
- **Composable**: Decay compounds correctly over time
- **Naturalistic**: Matches cognitive forgetting curves

**Eviction Rule:**
- Evict when `freshness < eviction_threshold` (per-kind from tokens.yaml)

**Rates Source:**
- Per-kind from `tokens.yaml` (freshness_rate, reliability_rate)
- Example: `freshness_rate = 0.01` means freshness halves every ~69 ticks

---

## 2. Token Schema Extension

**Phase 1 Schema:**
```python
{
    'kind': str,
    'value': float,
    'version': int,
    'last_tick': float,
    'source': str
}
```

**Phase 2 Schema (add reliability):**
```python
{
    'kind': str,
    'value': float,
    'version': int,
    'last_tick': float,
    'reliability': float,  # NEW: decays over time, attenuates on propagation
    'source': str
}
```

**Initialization:**
- **Direct observation**: `reliability = tokens.yaml.initial_values.reliability` (e.g., 1.0)
- **Gossip**: Inherit from source, then attenuate (see §4)

---

## 3. Capacity Constraints & Eviction

**Capacity:**
- Global default: `GOSSIP_TOKEN_CAP_DEFAULT = 16`
- Optional per-species override: `Species.knowledge_capacity` (future)
- Applies to **total tokens across all kinds** per entity

**Eviction Order (Deterministic):**

When `count(tokens) > cap`, evict tokens in this order:

1. **Primary**: Lowest `freshness` (stalest memory first)
2. **Secondary**: Lowest `reliability` (if freshness equal within epsilon)
3. **Tertiary**: Oldest `last_tick` (if reliability equal)
4. **Final**: Stable sort by `(kind, entity_id)` for absolute determinism

**Mixed Token Kinds:**
- Use **raw freshness values** (no normalization across kinds)
- Rationale: Fast-decay kinds (e.g., predator_location) are meant to be evicted sooner (time-sensitive)
- Slow-decay kinds (e.g., ship_sentiment) persist longer by design

**Implementation:**
```python
# After merge, before writeback
if len(entity_tokens) > cap:
    # Compute decay arrays (vectorized, O(N))
    freshness_arr = np.exp(-freshness_rate * (current_tick - last_tick_arr))
    reliability_arr = np.exp(-reliability_rate * (current_tick - last_tick_arr))

    # Sort by (freshness, reliability, last_tick, kind) ascending
    # Evict first (count - cap) tokens
    # Write back changed rows only
```

---

## 4. Reliability Channel & Attenuation

**Purpose:**
- Track information quality/confidence separately from recency (version/last_tick)
- Degrade during propagation (gossip reduces certainty)
- Used for merge weighting, eviction tiebreaks

**Merge Precedence (Hierarchical):**

1. **Primary**: Higher `version` wins
2. **Secondary**: Later `last_tick` wins (if versions equal)
3. **Tertiary**: Higher `reliability` wins (if version + last_tick equal within epsilon)

**Attenuation During Propagation:**

Applied **during merge** when token propagates A → B via gossip:

```python
if source == 'gossip':
    reliability_dst = reliability_src * (1 - attenuation)  # clamp [0, 1]
    version_dst = version_src      # unchanged
    last_tick_dst = last_tick_src  # unchanged
```

**Attenuation rate** from `tokens.yaml` per-kind (e.g., `attenuation = 0.05` means 5% loss per hop)

**Initial Reliability:**
- Direct observation: `reliability = initial_values.reliability` (e.g., 1.0)
- Gossip-received: Inherit from source, then attenuate (no hardcoded "gossip_reliability")

---

## 5. Weighted-Average Merging (Future)

For token kinds with `merge.algorithm = weighted_average` (not in Phase 2, documented for future):

```python
# Compute weights from freshness + reliability
weight_a = w_fresh * freshness_a + (1 - w_fresh) * reliability_a
weight_b = w_fresh * freshness_b + (1 - w_fresh) * reliability_b

# Weighted average
merged_value = (value_a * weight_a + value_b * weight_b) / (weight_a + weight_b)
merged_reliability = max(reliability_a, reliability_b)  # or weighted avg
```

Where `w_fresh` comes from `tokens.yaml.merge.weight_freshness` (e.g., 0.7 = 70% freshness, 30% reliability)

---

## 6. Tokens YAML Schema Extension

**Current (Phase 1):**
```yaml
tokens:
  - kind: ship_sentiment
    default_value: 0.0
    value_range: [-1.0, 1.0]
```

**Proposed (Phase 2):**
```yaml
tokens:
  - kind: ship_sentiment
    default_value: 0.0
    value_range: [-1.0, 1.0]

    # Lifecycle parameters (NEW)
    decay:
      freshness_rate: 0.01        # exp decay factor per tick
      reliability_rate: 0.005     # reliability decay per tick
      eviction_threshold: 0.1     # evict when freshness < 0.1

    # Gossip parameters (NEW)
    gossip:
      attenuation: 0.05           # reliability loss per propagation hop (5%)

    # Merge semantics (NEW)
    merge:
      algorithm: version_based    # version_based | most_recent | weighted_average
      weight_freshness: 0.7       # for weighted_average (0.7 fresh, 0.3 reliable)

    # Initial values (NEW)
    initial_values:
      reliability: 1.0            # for direct observation
```

**Per-Kind Configuration:**
- Different tokens can decay at different rates
- Time-sensitive tokens (predator_location) → high freshness_rate
- Long-term memory (ship_sentiment) → low freshness_rate

---

## 7. Implementation Architecture

**SoA Cache Extension:**

Phase 1 cache:
```python
{
    'build_seq': int,
    'N': int,
    'has': bool[N],
    'val': float64[N],
    'version': int64[N],
    'last_tick': float64[N]
}
```

Phase 2 cache (add reliability):
```python
{
    'build_seq': int,
    'N': int,
    'has': bool[N],
    'val': float64[N],
    'version': int64[N],
    'last_tick': float64[N],
    'reliability': float64[N]  # NEW
}
```

**Lifecycle Pass (After Exchange):**

```python
def exchange_tokens_with_lifecycle(entities, adapter, species_registry, current_tick):
    # 1. Extract state (cache hit or rebuild)
    state = _get_or_build_state_cache(adapter)
    has, val, version, last_tick, reliability = state[...]

    # 2. Compute decay arrays ONCE (vectorized, O(N), ~0.1ms)
    age = current_tick - last_tick
    freshness = np.exp(-freshness_rate * age)
    reliability_decayed = np.exp(-reliability_rate * age)

    # 3. Build edges (query_pairs, uniform-range fast path)
    edge_a, edge_b = _build_edges(adapter, species_registry)

    # 4. Merge with attenuation (use reliability in precedence + attenuate)
    _merge_with_lifecycle(has, val, version, last_tick, reliability,
                          freshness, reliability_decayed,
                          edge_a, edge_b,
                          attenuation_rates)

    # 5. Eviction (per-entity, deterministic ordering)
    _enforce_capacity(entities, cap, freshness, reliability_decayed, last_tick)

    # 6. Writeback changed rows
    _writeback_to_entities(entities, has, val, version, last_tick, reliability)
```

**Performance Target:**
- Phase 1: p50 ~1.6ms, profiled 1.29ms
- Phase 2 added cost: ~0.2-0.3ms (decay arrays + eviction)
- **Target: p50 < 2ms maintained**

---

## 8. Testing Strategy

**New Test File: `aquarium/tests/test_gossip_lifecycle.py`**

**Test Cases:**

1. **test_exponential_decay()**
   - Seed token at tick 0 with reliability=1.0
   - Run 100 ticks without propagation
   - Verify freshness = exp(-rate * 100)
   - Verify reliability = exp(-rel_rate * 100)

2. **test_eviction_by_freshness()**
   - Entity has token with freshness < threshold
   - Run one tick
   - Verify token evicted

3. **test_capacity_enforcement()**
   - Entity has 20 tokens (cap = 16)
   - Run one tick
   - Verify 4 stalest tokens evicted
   - Verify eviction order: freshness → reliability → last_tick

4. **test_gossip_attenuation()**
   - Entity A has reliability=1.0
   - Entity B receives via gossip (attenuation=0.05)
   - Verify B.reliability = 0.95

5. **test_lifecycle_determinism()**
   - Run 50 ticks twice with same seed
   - Verify identical token states (including freshness, reliability)

6. **test_lifecycle_preserves_propagation()**
   - Same setup as Phase 1 test_propagation
   - Verify ≥95% coverage even with decay/eviction
   - (Ensure lifecycle doesn't break connectivity)

**Update Existing Tests:**
- `test_gossip.py`: Update for reliability field in schema
- `test_performance.py`: Verify <2ms maintained with lifecycle

---

## 9. Constants & Configuration

**New Constants (aquarium/constants.py):**

```python
# Gossip Lifecycle (Phase 2)
GOSSIP_TOKEN_CAP_DEFAULT = 16  # Max tokens per entity (across all kinds)
GOSSIP_PROFILE = False          # Enable sub-timer profiling (set via env var)
GOSSIP_LOG_INTERVAL = 200       # Ticks between perf breakdowns
```

**Environment Variables:**
- `GOSSIP_PROFILE=1`: Enable sub-timer profiling (extract/edge/merge/writeback/decay/evict)
- Used for debugging and optimization, not production

---

## 10. Integration Points

**simulation.py Phase B.5:**

```python
if USE_GOSSIP and evaluate_behaviors:
    gossip_start = time.perf_counter()

    # Rebuild spatial index (post-movement positions)
    self.spatial.build(...)

    # Exchange with lifecycle (decay + eviction + attenuation)
    gossip_result = exchange_tokens_with_lifecycle(
        self.entities,
        self.spatial,
        self.species_registry,
        current_tick=self.tick_count  # NEW: pass current tick
    )

    gossip_elapsed = time.perf_counter() - gossip_start
    self._gossip_times.append(gossip_elapsed)

    # Log sub-timers if GOSSIP_PROFILE=1
    if gossip_result.get('profile'):
        # Log extract_ms, edge_ms, merge_ms, decay_ms, evict_ms, writeback_ms
```

**Behavior Consumption (Future):**

```python
# Read-only API for behaviors to access knowledge
def get_knowledge_freshness(entity, token_kind, current_tick):
    """
    Compute current freshness of token without modifying state.
    Returns None if token not present.
    """
    token = entity.knowledge_tokens.get(token_kind)
    if token is None:
        return None

    age = current_tick - token['last_tick']
    freshness = np.exp(-freshness_rate * age)
    return freshness

# Usage in behavior:
sentiment_freshness = get_knowledge_freshness(entity, 'ship_sentiment', current_tick)
if sentiment_freshness and sentiment_freshness > 0.5:
    # Trust the sentiment, modulate behavior
    investigate_weight *= (1 + entity.knowledge_tokens['ship_sentiment']['value'])
```

---

## 11. Acceptance Criteria

**Phase 2 Complete When:**

- ✅ Decay/eviction rules implemented per tokens.yaml
- ✅ Reliability field added to schema, decays + attenuates correctly
- ✅ Coverage remains ≥95% on connected graphs
- ✅ Lifecycle does not stall propagation
- ✅ Performance: median <2ms @ 1000 entities (p90 informational)
- ✅ Determinism holds across runs with lifecycle enabled
- ✅ All tests passing: `test_gossip.py` + `test_gossip_lifecycle.py`

---

## 12. Future Extensions (Phase 2b+)

**Additional Token Kinds:**
- `predator_location`: merge.algorithm = most_recent (last sighting wins)
- `food_source`: merge.algorithm = weighted_average (blend observations)

**Network Health Monitoring:**
- Post-spawn degree histogram (warn if many < MIN_GOSSIP_NEIGHBORS)
- Periodic connectivity checks (every N ticks)

**Species Range Tuning:**
- Populate `Species.gossip_range_m` from species YAML configs
- Validate expected degree based on density (E[k] ≈ ρ π r²)

**Performance Telemetry:**
- Configurable logging (GOSSIP_LOG_INTERVAL)
- Sub-timer breakdown when GOSSIP_PROFILE=1
- Larger-N validation (143/500/1000/2000)

---

## References

- Phase 1 Design: `project/plans/gossip_v2_phase1_design.md`
- External Research: `project/research/spatial_gossip_protocols_external_research_2025-10-02.md`
- Token Schema: `data/knowledge/tokens.yaml`
- Implementation: `aquarium/gossip.py`
- Tests: `aquarium/tests/test_gossip.py`, `aquarium/tests/test_gossip_lifecycle.py` (pending)

---

**End of Phase 2 Design**
_Ready for implementation in next session_
