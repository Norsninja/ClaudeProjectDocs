# C7 Phase 1 Implementation Plan: Loop Scaffold with Edge Reuse

**Created**: 2025-10-03
**Author**: Chronus (with SD guidance)
**Status**: Planning - Awaiting Approval
**Goal**: Refactor exchange_tokens() to support multi-kind gossip while maintaining bit-identical behavior with single kind

---

## Overview

Phase 1 introduces the loop scaffold and edge reuse infrastructure without changing behavior. All existing tests must pass with identical performance.

**Key Constraint**: Single-kind only in Phase 1. Multi-kind support deferred to Phase 2.

---

## Current Structure (Phase 2a - Single Kind)

```
exchange_tokens() - 468 lines:
  Lines 121-172: Setup, validation, early exits
  Lines 174-177: Hard-coded kind = 'ship_sentiment'
  Lines 179-232: Extract token state to SoA arrays (cache keyed by adapter_id)
  Lines 234-253: Compute decay arrays (freshness, reliability_current)
  Lines 254-309: Build edges via query_pairs (edge_a, edge_b, entity_degrees)
  Lines 311-352: Vectorized push-pull merge (version_based)
  Lines 354-362: Staleness eviction (threshold check)
  Lines 364-412: Capacity enforcement (multi-kind ready, deterministic sort)
  Lines 414-446: Writeback changed rows to entities
  Lines 448-467: Diagnostics and return
```

**Critical Variables**:
- `kind`: Hard-coded 'ship_sentiment' (line 177)
- `token_def`: Loaded from `_TOKEN_DEFINITIONS[kind]`
- `edge_a, edge_b, entity_degrees`: Built once, used in merge phase
- SoA arrays: `has, val, version, last_tick, reliability` (per-entity state)
- Cache key: `_TOKEN_STATE_CACHE[adapter_id]` (not per-kind yet)

---

## Target Structure (Phase 1 - Loop Scaffold)

```
exchange_tokens() - ~200 lines:
  Lines 121-172: Setup, validation, early exits (UNCHANGED)
  Lines 174-177: Remove hard-coded kind
  Lines 179-XXX: Build edges ONCE (move from lines 254-309)
              → edge_a, edge_b, entity_degrees, pairs_count
  Lines XXX:   Single-kind guard (Phase 1 safety):
              assert len(allowed_kinds) == 1, "Multi-kind requires Phase 2 cache refactor"

  Lines XXX:   Loop over allowed_kinds:
              for kind in allowed_kinds:
                  token_def = _TOKEN_DEFINITIONS[kind]
                  exchanges_count, changed_rows, profile = _process_kind(
                      kind, token_def, entities, adapter, edge_a, edge_b,
                      entity_degrees, current_tick, EPSILON, timer
                  )
                  total_exchanges += exchanges_count

  Lines XXX:   Capacity enforcement (UNCHANGED from 364-412)
  Lines XXX:   Diagnostics and return (UNCHANGED from 448-467)

_process_kind() - NEW HELPER (~250 lines):
  Extract token state to SoA (lines 179-232 logic)
  Compute decay arrays (lines 234-253 logic)
  Merge with edges (lines 311-352 logic)
  Staleness eviction (lines 354-362 logic)
  Writeback changed rows (lines 414-446 logic)
  Return: exchanges_count, changed_rows, profile_dict
```

---

## Helper Function Design

### Signature

```python
def _process_kind(
    kind: str,
    token_def: Dict,
    entities: List[Entity],
    adapter: SpatialIndexAdapter,
    edge_a: np.ndarray,  # int32[E] - source indices
    edge_b: np.ndarray,  # int32[E] - dest indices
    entity_degrees: np.ndarray,  # int32[N] - neighbor counts (unused in Phase 1)
    current_tick: int,
    epsilon: float,
    timer: _GossipTimer
) -> Tuple[int, np.ndarray, Dict]:
    """
    Process gossip for a single token kind.

    Phases:
    1. Extract: Build SoA arrays from entity.knowledge_tokens[kind]
    2. Decay: Compute freshness and reliability_current from age
    3. Merge: Vectorized push-pull using edge_a/edge_b
    4. Evict: Remove stale tokens below eviction_threshold
    5. Writeback: Update entity.knowledge_tokens[kind]

    Args:
        kind: Token kind to process (e.g., 'ship_sentiment')
        token_def: Token definition from _TOKEN_DEFINITIONS[kind]
        entities: Entity list (via adapter._entities)
        adapter: SpatialIndexAdapter (for cache keys and _entities)
        edge_a, edge_b: Precomputed edge arrays (reused across kinds)
        entity_degrees: Neighbor counts (for diagnostics, unused in Phase 1)
        current_tick: Current simulation tick
        epsilon: Float comparison epsilon (1e-12)
        timer: Profiling timer context

    Returns:
        exchanges_count: Number of token transfers (push + pull)
        changed_rows: np.ndarray[int] of entity indices that need writeback
        profile: Dict of sub-phase timings (if GOSSIP_PROFILE=1, else empty)
    """
```

### Internal Structure

```python
def _process_kind(...):
    N = len(adapter._entities)

    # Phase 1: Extract (with timer.time('extract'))
    # - Check SoA cache: _TOKEN_STATE_CACHE[adapter_id]
    # - If valid: reuse arrays
    # - If invalid: rebuild from entities
    # - Arrays: has, val, version, last_tick, reliability

    # Phase 2: Decay (with timer.time('decay'))
    # - Get rates from token_def['decay']
    # - age = current_tick - last_tick
    # - freshness = exp(-freshness_rate * age)
    # - reliability_current = reliability * exp(-reliability_rate * age)

    # Phase 3: Merge (with timer.time('merge'))
    # - Get attenuation from token_def['gossip']['attenuation']
    # - mask_push = has[edge_a] & version precedence
    # - mask_pull = has[edge_b] & version precedence
    # - Apply transfers with attenuation
    # - exchanges_count = sum(mask_push) + sum(mask_pull)

    # Phase 4: Staleness Eviction (with timer.time('staleness_evict'))
    # - eviction_threshold from token_def['decay']['eviction_threshold']
    # - stale_mask = has & (freshness < threshold)
    # - has[stale_mask] = False

    # Phase 5: Writeback (with timer.time('writeback'))
    # - Collect changed_rows from merge + eviction
    # - For each changed row:
    #     - If has[row]: write token dict to entity.knowledge_tokens[kind]
    #     - Else: delete entity.knowledge_tokens[kind]

    # Return
    profile_dict = timer.timings.copy() if timer.enabled else {}
    return exchanges_count, changed_rows, profile_dict
```

---

## Phase 1 Changes: Step-by-Step

### Change 1: Remove Hard-Coded Kind

**File**: `aquarium/gossip.py`
**Lines**: 176-177
**Action**: DELETE

```python
# BEFORE (lines 176-177):
    # Phase 1: Only ship_sentiment token (single kind optimization)
    kind = 'ship_sentiment'

# AFTER:
    # (removed - kind now comes from loop)
```

---

### Change 2: Move Edge Building to Top

**File**: `aquarium/gossip.py`
**Lines**: 254-309 → Move to line 179 (before kind loop)
**Action**: RELOCATE (cut and paste, update comment)

```python
# NEW location (line ~179, after setup):
    # ========================================================================
    # Build edges ONCE per tick (reused across all token kinds)
    # ========================================================================
    with timer.time('edge_query'):
        # Get or compute per-row gossip ranges (cached by adapter build_seq)
        adapter_id = id(adapter)
        cache_entry = _RANGE_CACHE.get(adapter_id)

        # ... (existing logic from lines 254-309, unchanged)

        # Result: edge_a, edge_b, entity_degrees, pairs_count
```

**Variables to preserve**:
- `edge_a`: np.ndarray[int32, E] - source indices
- `edge_b`: np.ndarray[int32, E] - dest indices
- `entity_degrees`: np.ndarray[int32, N] - neighbor counts
- `pairs_count`: int - number of edges

---

### Change 3: Add Single-Kind Guard

**File**: `aquarium/gossip.py`
**Lines**: Insert after edge building (~line 250)
**Action**: ADD

```python
    # ========================================================================
    # Phase 1 Safety: Single-kind only until Phase 2 cache refactor
    # ========================================================================
    assert len(allowed_kinds) == 1, \
        f"Multi-kind gossip requires Phase 2 per-kind cache (got {len(allowed_kinds)} kinds)"
```

**Rationale**: Prevents accidental multi-kind runs before Phase 2 refactors the cache structure.

---

### Change 4: Add Kind Loop and Helper Call

**File**: `aquarium/gossip.py`
**Lines**: Insert after guard (~line 253)
**Action**: ADD

```python
    # ========================================================================
    # Process each token kind (Phase 1: single kind only)
    # ========================================================================
    total_exchanges = 0

    for kind in allowed_kinds:
        # Get token definition for this kind
        token_def = token_defs.get(kind)
        if token_def is None:
            continue  # Skip undefined kinds (shouldn't happen with validated config)

        # Process this kind (extract → decay → merge → evict → writeback)
        exchanges_count_kind, changed_rows_kind, profile_kind = _process_kind(
            kind=kind,
            token_def=token_def,
            entities=entities,
            adapter=adapter,
            edge_a=edge_a,
            edge_b=edge_b,
            entity_degrees=entity_degrees,
            current_tick=current_tick,
            epsilon=EPSILON,
            timer=timer
        )

        total_exchanges += exchanges_count_kind
```

**Note**: `profile_kind` is currently unused but will be used for per-kind profiling in Phase 2+.

---

### Change 5: Create _process_kind() Helper

**File**: `aquarium/gossip.py`
**Lines**: Insert before `exchange_tokens()` (~line 120)
**Action**: ADD NEW FUNCTION

```python
def _process_kind(
    kind: str,
    token_def: Dict,
    entities: List[Entity],
    adapter: SpatialIndexAdapter,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    entity_degrees: np.ndarray,
    current_tick: int,
    epsilon: float,
    timer: _GossipTimer
) -> Tuple[int, np.ndarray, Dict]:
    """
    Process gossip for a single token kind.

    [Full docstring from Helper Function Design above]
    """
    N = len(adapter._entities)

    # Extract token state to arrays (MOVE from lines 179-232)
    with timer.time('extract'):
        # ... existing extract logic ...
        # NOTE: Cache key remains adapter_id (not per-kind until Phase 2)

    # Compute decay arrays (MOVE from lines 234-253)
    with timer.time('decay'):
        # ... existing decay logic ...

    # Vectorized push-pull merge (MOVE from lines 311-352)
    with timer.time('merge'):
        # ... existing merge logic ...

    # Staleness eviction (MOVE from lines 354-362)
    with timer.time('staleness_evict'):
        # ... existing eviction logic ...

    # Writeback (MOVE from lines 414-446)
    with timer.time('writeback'):
        # ... existing writeback logic ...

    # Return results
    profile_dict = timer.timings.copy() if timer.enabled else {}
    return exchanges_count, changed_rows, profile_dict
```

**Important**: Copy logic exactly from current implementation. No semantic changes in Phase 1.

---

### Change 6: Update Return Statement

**File**: `aquarium/gossip.py`
**Lines**: 456-467 (return statement)
**Action**: UPDATE

```python
# BEFORE:
    result = {
        'exchanges_count': exchanges_count,  # from merge phase
        'pairs_count': pairs_count,
        'low_degree_count': int(low_degree_count),
        'avg_degree': float(np.mean(entity_degrees)) if N > 0 else 0.0
    }

# AFTER:
    result = {
        'exchanges_count': total_exchanges,  # sum across kinds
        'pairs_count': pairs_count,
        'low_degree_count': int(low_degree_count),
        'avg_degree': float(np.mean(entity_degrees)) if N > 0 else 0.0
    }
```

**Change**: Use `total_exchanges` (summed from loop) instead of `exchanges_count` (from merge phase, now scoped inside helper).

---

## Variables That Change Scope

### Moved INTO _process_kind (per-kind):
- `has, val, version, last_tick, reliability` (SoA arrays)
- `freshness, reliability_current` (decay arrays)
- `mask_push, mask_pull` (merge masks)
- `exchanges_count` (becomes `exchanges_count_kind` in loop)
- `changed_rows` (becomes `changed_rows_kind` in loop)

### Remain in exchange_tokens (global):
- `edge_a, edge_b, entity_degrees, pairs_count` (built once, reused)
- `total_exchanges` (accumulated across kinds)
- `token_defs` (already global, loaded at module import)

### Unchanged (stay in exchange_tokens):
- `EPSILON` (float comparison tolerance)
- `timer` (profiling context)
- `adapter, entities, species_registry, current_tick` (passed to helper)
- Capacity enforcement logic (lines 364-412, runs after loop)

---

## Cache Strategy (Phase 1)

**Decision**: Keep current cache structure (keyed by `adapter_id` only).

**Rationale**:
- Limits change surface in Phase 1
- Single-kind guard prevents multi-kind runs
- Behavior must be bit-identical to baseline

**Phase 2 Change**: Switch to per-kind cache key `(adapter_id, build_seq, kind)`.

**Code Location**: `aquarium/gossip.py`, lines ~186 in _process_kind:

```python
# Phase 1 (unchanged):
adapter_id = id(adapter)
state_cache = _TOKEN_STATE_CACHE.get(adapter_id)

# Phase 2 (future):
cache_key = (id(adapter), adapter._build_seq, kind)
state_cache = _TOKEN_STATE_CACHE.get(cache_key)
```

---

## Test Strategy

### Phase 1: No New Tests (A/B Validation)

**Approach**: Run existing tests unchanged. They should pass with identical results.

**Tests to Run**:
1. `aquarium/tests/test_gossip.py` (5 Phase 1 tests)
2. `aquarium/tests/test_gossip_lifecycle.py` (6 Phase 2a lifecycle tests)
3. `scripts/perf_gossip_multi_n.py` (performance validation)

**Expected Results**:
- All 11 tests pass
- Performance: 1000 entities @ 1.88ms median (pytest), no regression
- Determinism: Bit-identical results across runs

**Validation Checklist**:
- [ ] All existing tests pass (11/11)
- [ ] Performance within 5% of baseline (1.88ms ± 0.09ms)
- [ ] No behavior changes (same propagation, same eviction, same capacity)
- [ ] Single-kind guard prevents accidental multi-kind runs

### Optional Sanity Check (Non-Blocking):

Add temporary diagnostic in test to confirm edge build happens once:

```python
# In test_gossip.py or test_gossip_lifecycle.py (temporary):
result = exchange_tokens(entities, adapter, species_registry, current_tick=1)
assert result['pairs_count'] > 0, "Edges should be built"
# Could add internal counter to verify query_pairs called exactly once per tick
```

---

## Acceptance Criteria

### Phase 1 Complete When:

1. **Code Changes**:
   - [ ] Hard-coded `kind = 'ship_sentiment'` removed
   - [ ] Edge building moved before kind loop
   - [ ] Single-kind guard added (assert len==1)
   - [ ] `_process_kind()` helper created with full logic
   - [ ] Kind loop added in exchange_tokens
   - [ ] Return statement updated to use `total_exchanges`

2. **Tests Pass**:
   - [ ] `test_gossip.py`: 5/5 passing
   - [ ] `test_gossip_lifecycle.py`: 6/6 passing
   - [ ] Performance test: 1000 entities @ <2.0ms (within 5% of baseline)

3. **Behavior Validation**:
   - [ ] Bit-identical determinism (same entity states across runs)
   - [ ] No perf regression (median within noise)
   - [ ] Profiling breakdown preserved (extract/decay/merge/evict/writeback timings)

4. **Code Quality**:
   - [ ] No linting errors
   - [ ] Docstrings complete for `_process_kind()`
   - [ ] Comments updated to reflect new structure

---

## Risks and Mitigations

### Risk 1: Accidentally Change Merge Semantics
**Mitigation**: Copy logic exactly. Do not refactor merge algorithm in Phase 1.

### Risk 2: Cache Invalidation Bug
**Mitigation**: Keep cache key unchanged (adapter_id only). Defer per-kind cache to Phase 2.

### Risk 3: Performance Regression from Helper Call Overhead
**Mitigation**: Python function call is ~100ns, negligible. Verify with perf test.

### Risk 4: Profiling Timer Breaks with Nested Contexts
**Mitigation**: Timer already supports nested contexts. Test with GOSSIP_PROFILE=1.

---

## Next Steps After Phase 1

### Phase 2: Per-Kind SoA Cache
- Change cache key to `(adapter_id, build_seq, kind)`
- Remove single-kind guard
- Test with `allowed_kinds = ['ship_sentiment', 'ship_capabilities']` (both float)

### Phase 3: most_recent + Position Types
- Implement most_recent merge algorithm
- Skip val SoA for position kinds
- Copy 3-tuples on writeback

### Phase 4: Enable predator_location
- Add to GOSSIP_ALLOWED_KINDS
- Create test_gossip_multikind.py
- Validate performance with 2 kinds (<2ms)

---

## Questions for Review

1. **Helper function scope**: Is `_process_kind()` signature correct? Any missing parameters?

2. **Cache strategy**: Confirm we keep current cache in Phase 1, defer per-kind to Phase 2?

3. **Edge build timing**: Edges built once before loop - any edge cases where query_pairs fails?

4. **Single-kind guard**: Should we use assert or log warning + early return?

5. **Test validation**: A/B comparison sufficient for Phase 1, or should we add explicit edge-count check?

---

**Status**: Ready for review by Mike and SD.
**Next Action**: Await approval, then implement changes.
