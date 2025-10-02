# Session Handoff: Phase 5 Subtree Foundation Complete

**Created**: 2025-10-01
**From Session**: Session 005
**To**: Next Chronus Instance
**Context Window**: 66% - Ready for handoff

## Critical Context

Phase 5 batch query optimization is 80% complete. Subtree infrastructure built and validated. Next session must implement vectorized batch queries (Step 2) and array caching (Step 3) to unlock 2-4x performance gains at 1000 entities.

## What Was Accomplished

### 1. Phase 5 Initial Implementation (Batch Cache with Dicts)

- Implemented `nearest_by_tag_batch()` API with per-biome grouping
- Added stable index mapping (row_of_id, id_of_row, biome_of_row)
- Built tag masks for hot tags (predator, ship)
- Integrated batch cache into simulation.tick() Phase A
- Refactored behavior.py to consult cache first, fallback to per-entity queries
- Result: 4.95x behavior speedup (3.433ms → 0.693ms at 143 entities)

### 2. Synthetic Performance Test Harness

- Created `aquarium/tests/perf_harness.py` for controlled entity populations
- Function: `build_perf_scenario(N, predator_ratio=0.01, ship_count=1, seed=42)`
- Generates deterministic entity distributions: 989 drifters, 10 predators, 1 ship
- Validated at 1000 entities without modifying biome data
- Baseline: 118ms/tick at 1000 entities (dict-based cache)

### 3. Per-Tag, Per-Biome Subtree Building (Step 1 Complete)

- Added `_tag_biome_trees` storage to SpatialIndexAdapter
- Implemented `_build_tag_biome_subtrees()` method
- Builds cKDTree for each (tag, biome) pair with entities
- Stores: (kdtree, rows_subset[int64], instance_ids_subset)
- Validated: 2 subtrees built for 1000 entities (10 predators, 1 ship)
- Added constants: USE_PER_TAG_TREES, CKDTREE_WORKERS, NEAREST_PREDATOR_RADIUS, NEAREST_SHIP_RADIUS

## Current Working State

### What IS Working:

- ✅ Batch query API - deterministic results match per-entity queries (100% at 100/1000 entities)
- ✅ Subtree building - correctly identifies tag/biome combinations, builds cKDTree
- ✅ Synthetic 1000-entity harness - realistic tag distribution, deterministic placement
- ✅ All existing tests passing (8/8 behavior + batch determinism tests)
- ✅ Behavior speedup validated - 4.95x improvement proves architecture works

### What is PARTIALLY Working:

- ⏳ Batch query performance - works correctly but Python loops dominate (batch_query_ms=1.740ms at 143 entities)
- ⏳ Overall speedup - 1.31x at 143 entities (behavior: 4.95x, but batch overhead negates gains)
- ⏳ Dict-based cache - functional but adds lookup overhead in hot path

### What is NOT Working:

- ❌ Performance at scale - 118ms/tick at 1000 entities (need vectorized queries to reach 2-4x target)
- ❌ Vectorized batch queries - still using per-source Python loops (Step 2 not implemented)
- ❌ Array-based cache - still using dicts (Step 3 not implemented)

### Known Issues:

- Per-source iteration in `_nearest_by_tag_batch_ckdtree()` defeats vectorization (lines 557-630)
- Dict cache lookups in behavior.py add overhead (lines 123-142, 231-232, 256)
- Biome spawn limits cap at 143 entities (production data, not a bug)

## Next Immediate Steps

### Priority 1: Implement Vectorized Batch Nearest (Step 2)

**Goal:** Eliminate Python loops in batch queries using single cKDTree.query() call

**Implementation:**
- Refactor `_nearest_by_tag_batch_ckdtree()` to use subtrees
- Replace per-source loop with: `d, local_idx = subtree.query(positions[sources], k=1, workers=CKDTREE_WORKERS)`
- Map local indices to global: `global_idx = rows_subset[local_idx]`
- Apply masks in numpy:
  - Self-exclusion: `mask_self = (global_idx == sources); global_idx[mask_self] = -1; d[mask_self] = inf`
  - Max distance: `mask_far = (d > max_distance); global_idx[mask_far] = -1; d[mask_far] = inf`
- Return (d, global_idx) arrays directly

**Testing:**
- Test at 143 entities first (known baseline)
- Validate determinism: results must match current dict-based implementation
- Then test at 1000 entities, measure speedup

**Expected Outcome:**
- Batch query time drops from 1.740ms to <0.5ms
- Overall tick time improves 2-4x at 1000 entities

### Priority 2: Convert Cache to Numpy Arrays (Step 3)

**Goal:** Eliminate dict overhead in behavior hot path

**Implementation:**
- Preallocate arrays in `_build_query_cache()`:
  - `nearest_predator_dist = np.full(N, np.inf, dtype=np.float64)`
  - `nearest_predator_idx = np.full(N, -1, dtype=np.int64)`
  - Same for ship
- Fill slices per biome group: `nearest_predator_dist[group_rows] = d`
- Change cache structure: `cache['nearest']['predator'] = (dist_arr, idx_arr)`
- Update behavior lookups:
  - `dist_arr, idx_arr = cache['nearest']['predator']`
  - `dist = dist_arr[row]; idx = idx_arr[row]`

**Testing:**
- A/B test: cache OFF vs cache ON (with arrays)
- Validate determinism preserved

**Expected Outcome:**
- Small additional speedup (~10-20%)
- Cleaner code, better cache locality

### Priority 3: Performance Validation

- Run synthetic perf test at 500 and 1000 entities
- Compare: baseline (cache OFF) vs optimized (cache ON with subtrees)
- Log breakdown: build_main_ms, build_tag_ms, batch_query_ms, behavior_ms, movement_ms
- Target: 2-4x improvement, 1000 entities <30ms/tick (down from 118ms baseline)

### Priority 4: Determinism Validation

- A/B test: USE_CKDTREE False vs True
- A/B test: cache OFF vs cache ON
- Assert positions and behavior IDs match exactly
- Test multi-biome scenario if available

## Files Created/Modified

**Created:**
- `aquarium/tests/perf_harness.py` - Synthetic performance test harness (1000 entities)
- `aquarium/tests/test_batch_queries.py` - Batch query determinism tests (100/1000 entities)
- `aquarium/tests/test_batch_performance.py` - A/B performance comparison tests

**Modified:**
- `aquarium/spatial_queries.py` - Added subtree building, batch query API, index mapping
- `aquarium/simulation.py` - Integrated batch cache into tick Phase A, added timing breakdown
- `aquarium/behavior.py` - Cache-first lookups with per-entity fallback
- `aquarium/constants.py` - Added USE_PER_TAG_TREES, CKDTREE_WORKERS, NEAREST_*_RADIUS

## Key Insights/Learnings

**Behavior Speedup Validates Architecture:**
- 4.95x behavior improvement (3.433ms → 0.693ms) proves cache concept works
- Bottleneck shifted from behavior evaluation to batch query building
- Need to optimize cache building, not cache reading

**Synthetic Harness Critical:**
- Production biome data caps at 143 entities (intentional game balance)
- Synthetic harness enables scale testing without affecting game data
- Deterministic seeded spawns ensure reproducible tests

**Subtrees Cheap to Build:**
- Building 2 subtrees (10 predators, 1 ship) adds negligible overhead
- Cost is in per-source Python iteration, not tree construction
- Vectorization will unlock full cKDTree benefits

**Incremental Validation Essential:**
- Testing subtree building separately (Step 1) caught issues early
- Staged approach (Steps 1-2-3) reduces risk
- Each step validates before proceeding

**Performance Expectations:**
- Don't expect 24x improvement (unrealistic)
- Target: 2-4x at 1000 entities (30ms/tick, down from 118ms)
- Build time has a floor cost (main tree + subtrees)

## Technical Notes

**Subtree Storage Structure:**
```python
_tag_biome_trees: Dict[Tuple[str, str], Tuple[cKDTree, np.ndarray, np.ndarray]]
# Key: (tag, biome_id)
# Value: (kdtree, rows_subset, instance_ids_subset)
```

**Current Batch Query Flow (Needs Optimization):**
```python
# CURRENT (Python loops):
for i, (src_pos, src_row) in enumerate(zip(source_positions, source_rows)):
    indices = tree.query_ball_point(src_pos, r=max_distance)  # Per-source call!
    # Filter, sort, pick nearest...

# TARGET (Vectorized):
d, local_idx = subtree.query(positions[sources], k=1, workers=-1)  # Single call!
global_idx = rows_subset[local_idx]
# Apply masks in numpy
```

**Critical Edge Cases:**
- Empty subtree (no predators in biome) → return inf/-1 arrays without tree call
- Self-match (entity finds itself) → mask where global_idx == source_row
- Beyond max_distance → mask where d > max_distance

**Determinism Guardrails:**
- cKDTree.query with k=1 returns stable results (distance-sorted)
- Equal distances rare with k=1, but tie-break by instance_id if needed
- Self-exclusion MUST happen before returning results
- workers=-1 safe if post-processing maintains deterministic order

**Performance Constants:**
```python
USE_PER_TAG_TREES = True      # Enable subtree optimization
CKDTREE_WORKERS = -1          # Use all cores for parallel queries
NEAREST_PREDATOR_RADIUS = 80.0   # Max detection distance
NEAREST_SHIP_RADIUS = 150.0      # Max detection distance
```

## Progress Metrics

- Phase 5 Progress: 80% (subtree foundation complete, vectorization pending)
- Tests Passing: 8/8 (5 behavior + 3 batch tests)
- Performance: 1.31x at 143 entities, 118ms/tick at 1000 entities (baseline)
- Context Window at Handoff: 66%

## SD's Technical Guidance (Preserved for Next Session)

**Step 2 Implementation Details:**
- Use `tree.query(positions[sources_rows], k=1, workers=CKDTREE_WORKERS)`
- Map local_idx to global via rows_subset
- Self-exclusion: `mask_self = (global_idx == sources_rows)`
- Radius limit: `mask_far = (d > max_distance)`
- Return numpy arrays, no per-source Python loops

**Step 3 Cache Structure:**
- Preallocate: `dist_arr = np.full(N, inf)`; `idx_arr = np.full(N, -1, int64)`
- Fill slices per biome: `dist_arr[group_rows] = batch_results`
- Behavior lookup: `row = row_of_id[entity_id]`; `dist = dist_arr[row]`

**Testing Priorities:**
- A/B determinism at 143/500/1000 entities
- Perf breakdown logging every 200 ticks
- Multi-biome test if available

**Expected Outcomes:**
- 143 entities: modest gain (build overhead dominates)
- 500 entities: clear 2-3x improvement
- 1000 entities: target 2-4x, <30ms/tick

## Collaboration Notes

**What Worked Well:**
- Research → Discuss → Plan → Implement → Test cycle
- Incremental validation caught issues early
- SD's staged approach reduced risk
- Todo list kept progress honest

**Technical Wins:**
- Synthetic harness cleanly separates perf testing from game balance
- Subtree validation before vectorization prevented wasted effort
- Determinism tests ensure correctness at each step

**For Next Session:**
- Code well-documented with inline comments
- Tests validate each milestone
- Clear technical roadmap from SD
- Performance baseline established (118ms → target <30ms)

---

_Handoff prepared by Chronus Session 005_
_Phase 5 subtree foundation complete - vectorization unlocks 2-4x gains_
