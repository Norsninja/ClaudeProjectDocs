# C7 Phase 3: Multi-Kind Test Specification

**Created**: 2025-10-03
**Purpose**: Define test expectations for multi-kind gossip before Phase 4 implementation
**File**: `aquarium/tests/test_gossip_multikind.py`

---

## Test Structure

**Single file**: All 6 tests in `test_gossip_multikind.py`
- Shared helpers/fixtures
- Validates one cohesive feature (multi-kind gossip)
- ~400-500 lines total

---

## Shared Test Helpers

```python
def create_multikind_entities(count: int, grid_spacing: float = 12.0, seed: int = 42) -> List[Entity]:
    """
    Create test entities in grid pattern (similar to existing helpers).

    Args:
        count: Number of entities
        grid_spacing: Distance between grid points (meters)
        seed: Random seed for reproducibility

    Returns:
        List of Entity objects with empty knowledge_tokens
    """
    np.random.seed(seed)
    entities = []
    for i in range(count):
        x = (i % 10) * grid_spacing
        y = -((i // 10) % 10) * grid_spacing  # Titan depths: -50 to -200m
        z = (i // 100) * grid_spacing
        # ... create Entity with standard base_emissions
    return entities


def seed_predator(entity: Entity, value_xyz: tuple, version: int, last_tick: float, reliability: float = 1.0):
    """
    Seed entity with predator_location token (position type).

    Args:
        entity: Target entity
        value_xyz: 3-tuple [x, y, z] position
        version: Version counter
        last_tick: Timestamp of last update
        reliability: Reliability baseline (default 1.0)
    """
    entity.knowledge_tokens['predator_location'] = {
        'kind': 'predator_location',
        'value': list(value_xyz),  # 3-tuple as list
        'version': version,
        'last_tick': last_tick,
        'reliability': reliability,
        'source': 'direct'
    }


def seed_sentiment(entity: Entity, value: float, version: int, last_tick: float, reliability: float = 1.0):
    """
    Seed entity with ship_sentiment token (float type).

    Args:
        entity: Target entity
        value: Sentiment value [-1.0, 1.0]
        version: Version counter
        last_tick: Timestamp of last update
        reliability: Reliability baseline (default 1.0)
    """
    entity.knowledge_tokens['ship_sentiment'] = {
        'kind': 'ship_sentiment',
        'value': value,
        'version': version,
        'last_tick': last_tick,
        'reliability': reliability,
        'source': 'direct'
    }


def build_adapter(entities: List[Entity]) -> SpatialIndexAdapter:
    """
    Build spatial adapter with deterministic settings.

    Args:
        entities: List of entities to index

    Returns:
        SpatialIndexAdapter ready for gossip queries
    """
    adapter = SpatialIndexAdapter()
    adapter.build(entities)
    return adapter
```

---

## Test 1: `test_most_recent_precedence`

**Purpose**: Verify most_recent merge algorithm follows precedence hierarchy

**Precedence Rules**:
1. Higher version wins (regardless of timestamp)
2. Equal version → later last_tick wins
3. Equal version + last_tick → higher reliability wins

**Test Cases**:

### Case A: Version Precedence
**Given**:
- Entity A: predator_location(version=2, last_tick=10.0, reliability=0.8)
- Entity B: predator_location(version=1, last_tick=50.0, reliability=1.0)
- Entities within gossip range (10m apart)

**When**: exchange_tokens(current_tick=60)

**Then**:
- B receives A's token
- B.predator_location.version == 2
- B.predator_location.last_tick == 10.0
- Higher version won despite B having newer timestamp

### Case B: Timestamp Precedence (Version Equal)
**Given**:
- Entity A: predator_location(version=5, last_tick=100.0, reliability=0.9)
- Entity B: predator_location(version=5, last_tick=50.0, reliability=1.0)

**When**: exchange_tokens(current_tick=150)

**Then**:
- B receives A's token
- B.predator_location.last_tick == 100.0
- Later timestamp won when versions equal

### Case C: Reliability Tiebreaker (Version + Timestamp Equal)
**Given**:
- Entity A: predator_location(version=3, last_tick=75.0, reliability=0.95)
- Entity B: predator_location(version=3, last_tick=75.0, reliability=0.85)

**When**: exchange_tokens(current_tick=100)

**Then**:
- B receives A's token
- B.predator_location.reliability == 0.95 * (1 - 0.2) = 0.76 (attenuated)
- Higher reliability won as final tiebreaker

**Implementation Notes**:
- Use epsilon (1e-12) for float comparisons
- Set current_tick explicitly to avoid decay interfering with precedence tests
- Grid spacing: 10m (within 15m gossip range)

---

## Test 2: `test_position_value_copy`

**Purpose**: Verify position values (3-tuples) copied exactly, no averaging

**Given**:
- Entity A at [0, -50, 0]: predator_location value=[100.0, -75.0, 25.5]
- Entity B at [10, -50, 0]: no predator_location
- Entities within gossip range

**When**: exchange_tokens(current_tick=1)

**Then**:
- B receives predator_location token
- B.predator_location.value == [100.0, -75.0, 25.5] (exact match)
- No vector math performed
- No averaging (contrast with weighted_average for scalar kinds)

**Edge Case - Multiple Sources**:
**Given**:
- Entity A: predator_location value=[100.0, -75.0, 25.5], version=1
- Entity C: predator_location value=[200.0, -100.0, 50.0], version=2
- Entity B (center): receives from both

**Then**:
- B gets C's value (higher version)
- B.predator_location.value == [200.0, -100.0, 50.0]
- most_recent doesn't average, just takes highest precedence

**Implementation Check**:
- Verify no `val` SoA array allocated for predator_location kind
- Optional: Add profile/debug flag to confirm position path taken

---

## Test 3: `test_predator_attenuation`

**Purpose**: Verify reliability attenuates correctly for predator_location (0.2 attenuation per hop)

**From tokens.yaml**: predator_location has `attenuation: 0.2`

**Scenario: Multi-Hop Propagation**

**Setup**:
- 3 entities in a line: A -- B -- C
- Distance: 10m between each
- Only A starts with predator_location

**Hop 1: A → B**
**Given**:
- Entity A at [0, -50, 0]: predator_location(reliability=1.0, value=[100, -75, 25])
- Entity B at [10, -50, 0]: no token
- Entity C at [20, -50, 0]: no token

**When**: exchange_tokens(current_tick=0)

**Then**:
- B receives token from A
- B.predator_location.reliability == 1.0 * (1 - 0.2) = 0.8
- B.predator_location.value == [100, -75, 25] (unchanged)

**Hop 2: B → C**
**When**: exchange_tokens(current_tick=1)

**Then**:
- C receives token from B
- C.predator_location.reliability == 0.8 * (1 - 0.2) = 0.64
- C.predator_location.value == [100, -75, 25] (unchanged)

**Hop 3: Verify No Further Propagation Beyond C**
- Add entity D at [30, -50, 0] (outside C's 15m range)
- D should NOT receive token

**Implementation Notes**:
- Use current_tick=0,1,2,... to avoid decay (freshness stays 1.0)
- Rebuild adapter after each exchange to update positions (even though static)
- Verify attenuation compounds multiplicatively: hop_n = initial * (1-0.2)^n

---

## Test 4: `test_cross_kind_capacity`

**Purpose**: Verify fast-decay kinds (predator) evicted before slow-decay kinds (sentiment) under capacity pressure

**From tokens.yaml**:
- ship_sentiment: freshness_rate=0.01 (1% decay per tick)
- predator_location: freshness_rate=0.05 (5% decay per tick, 5x faster)

**Scenario: Capacity Overflow**

**Setup**:
- Single entity with 17 tokens (exceeds cap=16)
- Mix: 10 ship_sentiment + 7 predator_location
- All tokens have different ages to ensure deterministic sorting

**Token Distribution**:
```python
# 10 sentiment tokens (ages 0-9 ticks old)
for i in range(10):
    seed_sentiment(entity, value=0.5, version=1, last_tick=current_tick - i, reliability=1.0)

# 7 predator tokens (ages 0-6 ticks old)
for i in range(7):
    seed_predator(entity, value=[100, -75, 25], version=1, last_tick=current_tick - i, reliability=1.0)
```

**When**: exchange_tokens(current_tick=100) with cap=16

**Then**:
- Capacity enforcement runs
- 1 token evicted (17 - 16 = 1)
- **Evicted token**: Oldest predator_location (age=6 ticks)
  - Predator freshness: exp(-0.05 * 6) = 0.741
  - Sentiment freshness (age=6): exp(-0.01 * 6) = 0.942
  - Predator has lower freshness → evicted first

**Verify Eviction Order**:
- After capacity enforcement: 10 sentiment + 6 predator remain
- All predator tokens with age ≥ 6 evicted before any sentiment

**Deterministic Ordering**:
- Tokens sorted by (freshness ↑, reliability ↑, last_tick ↑, kind_lexical)
- With different ages, freshness dominates
- No normalization across kinds (raw freshness values compared)

**Implementation Notes**:
- Use `GOSSIP_TOKEN_CAP_DEFAULT = 16` from constants
- Vary last_tick for each token to ensure unique sorting
- Don't use identical freshness values (edge case testing not needed here)

---

## Test 5: `test_multikind_determinism`

**Purpose**: Verify identical results across runs with 2 kinds over multiple ticks

**Scenario: Full Simulation Determinism**

**Setup**:
- 100 entities in grid
- Seed 10% with ship_sentiment (10 entities)
- Seed 10% with predator_location (10 entities, different subset)
- Fixed seed=42 for reproducibility

**Run 1**:
```python
entities_run1 = create_multikind_entities(100, seed=42)
# Seed 10 with sentiment, 10 with predator
adapter = build_adapter(entities_run1)

for tick in range(50):
    exchange_tokens(entities_run1, adapter, species_registry, current_tick=tick)
    # Don't rebuild adapter (positions static)

# Capture final state
state_run1 = [(e.instance_id, e.knowledge_tokens.copy()) for e in entities_run1]
```

**Run 2**:
```python
entities_run2 = create_multikind_entities(100, seed=42)
# Identical seeding
adapter = build_adapter(entities_run2)

for tick in range(50):
    exchange_tokens(entities_run2, adapter, species_registry, current_tick=tick)

state_run2 = [(e.instance_id, e.knowledge_tokens.copy()) for e in entities_run2]
```

**Assertions**:
```python
for (id1, tokens1), (id2, tokens2) in zip(state_run1, state_run2):
    assert id1 == id2
    assert tokens1.keys() == tokens2.keys()  # Same kinds present

    for kind in tokens1.keys():
        t1, t2 = tokens1[kind], tokens2[kind]
        assert t1['version'] == t2['version']
        assert abs(t1['last_tick'] - t2['last_tick']) < 1e-12
        assert abs(t1['reliability'] - t2['reliability']) < 1e-12

        # Value comparison (type-specific)
        if kind == 'predator_location':
            # Position: compare 3-tuple
            assert len(t1['value']) == 3
            assert all(abs(v1 - v2) < 1e-9 for v1, v2 in zip(t1['value'], t2['value']))
        else:
            # Scalar: direct comparison
            assert abs(t1['value'] - t2['value']) < 1e-12
```

**Critical Requirements**:
- `allowed_kinds = sorted(['ship_sentiment', 'predator_location'])` for deterministic order
- Fixed seed for entity creation and token seeding
- Same grid spacing, same species_registry
- No randomness in gossip (all deterministic)

**Implementation Notes**:
- 50 ticks is sufficient to test propagation and decay over time
- Use small entity count (100) for fast test execution
- Rebuild adapter only once (positions don't change in this test)

---

## Test 6: `test_multikind_performance`

**Purpose**: Validate performance target with 2 kinds at scale

**Target**:
- **Pytest**: median < 2.2ms @ 1000 entities with 2 kinds
- **Standalone**: median < 2.0ms (if running outside pytest)

**Scenario: Performance Benchmark**

**Setup**:
```python
# Pin BLAS threads (before imports)
import os
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1'
})

entities = create_multikind_entities(1000, grid_spacing=12.0, seed=42)

# Seed 10% with each kind (non-overlapping for clarity)
for i in range(0, 100, 10):  # 10 entities
    seed_sentiment(entities[i], value=0.5, version=1, last_tick=0.0)
    seed_predator(entities[i+5], value=[100, -75, 25], version=1, last_tick=0.0)

species_registry = {}
adapter = build_adapter(entities)
```

**Measurement Protocol**:
```python
import gc
import time
import numpy as np

# Warmup (1 call)
exchange_tokens(entities, adapter, species_registry, current_tick=0)

# Measure (7 runs with GC disabled)
gc.collect()
gc.disable()

times_ns = []
try:
    for tick in range(1, 8):
        start = time.perf_counter_ns()
        exchange_tokens(entities, adapter, species_registry, current_tick=tick)
        elapsed_ns = time.perf_counter_ns() - start
        times_ns.append(elapsed_ns)
finally:
    gc.enable()

# Statistics
times_ms = np.array(times_ns) / 1_000_000
p50_ms = np.percentile(times_ms, 50)
p90_ms = np.percentile(times_ms, 90)
```

**Assertions**:
```python
# Pytest threshold (includes test harness overhead)
assert p50_ms < 2.2, f"Performance regression: {p50_ms:.3f}ms >= 2.2ms"

# Report p90 for diagnostics
print(f"  p50: {p50_ms:.3f}ms, p90: {p90_ms:.3f}ms")

# Verify edge build happened once (optional diagnostic)
# Could add internal counter or check profiling data
```

**Expected Performance**:
- Phase 1 baseline: 1.22ms @ 1000 entities with 1 kind
- Phase 2 result: 1.46ms @ 1000 entities with 1 kind (+0.24ms overhead)
- Phase 4 target: <2.0ms @ 1000 entities with 2 kinds (+0.3-0.5ms for second kind)

**Performance Breakdown Estimate**:
- Edge query: ~0.6ms (once per call, reused)
- Per-kind overhead:
  - Extract: ~0.3ms (cache hit)
  - Decay: ~0.1ms (vectorized)
  - Merge: ~0.3ms (vectorized)
  - Writeback: ~0.2ms
  - Total per kind: ~0.9ms
- Two kinds: 0.6 + 2*0.9 = 2.4ms (need optimization or faster ops)

**Optimization Notes**:
- Cache hit rate critical (per-kind cache should be near 100% after warmup)
- Edge reuse essential (no multiple query_pairs calls)
- Vectorized operations throughout (no per-entity loops in hot path)

**Implementation Notes**:
- Same measurement protocol as `scripts/perf_gossip_multi_n.py`
- Run in pytest: `pytest test_gossip_multikind.py::test_multikind_performance -v`
- Can also run standalone for cleaner timing (no test framework overhead)

---

## Test Execution Order

Run tests in this order for logical flow:

1. `test_position_value_copy` - Basic position type handling
2. `test_most_recent_precedence` - Merge algorithm correctness
3. `test_predator_attenuation` - Multi-hop reliability decay
4. `test_cross_kind_capacity` - Cross-kind eviction policy
5. `test_multikind_determinism` - Full simulation correctness
6. `test_multikind_performance` - Performance validation

---

## Acceptance Criteria

**Phase 3 Complete When**:
- ✅ All 6 tests written in `test_gossip_multikind.py`
- ✅ Helper functions implemented (create, seed, build)
- ✅ Tests initially FAIL (expected - most_recent not implemented)
- ✅ Clear failure messages indicate what needs implementing

**Phase 4 Complete When**:
- ✅ All 6 tests PASS
- ✅ No regression in existing 11 tests (Phase 1+2)
- ✅ Performance: pytest median < 2.2ms, standalone < 2.0ms
- ✅ Determinism holds across runs
- ✅ One KD-tree query per call (edge reuse working)

---

## Implementation Reminders for Phase 4

**When implementing to make tests pass**:

1. **Detect merge algorithm per-kind**:
   ```python
   merge_algo = token_def.get('merge', {}).get('algorithm', 'version_based')
   if merge_algo == 'most_recent':
       # version > last_tick > reliability precedence
   ```

2. **Position type detection**:
   ```python
   value_type = token_def.get('value_type', 'float')
   if value_type == 'position':
       val = None  # Don't allocate SoA array
   ```

3. **Position writeback**:
   ```python
   if value_type == 'position':
       # Copy 3-tuple directly from source entity dict
       entity.knowledge_tokens[kind]['value'] = list(source_value)
   ```

4. **Enable predator_location**:
   ```python
   # In aquarium/constants.py
   GOSSIP_ALLOWED_KINDS = ['ship_sentiment', 'predator_location']
   ```

---

**Next Session**: Implement these 6 tests (TDD approach), verify they fail appropriately, then proceed to Phase 4 implementation.
