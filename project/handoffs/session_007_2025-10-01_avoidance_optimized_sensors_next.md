# Session Handoff: Obstacle Avoidance Optimized, Sensor API Next

**Created**: 2025-10-01
**From Session**: Session 007
**To**: Next Chronus Instance
**Context Window**: 70% - Handoff recommended

## Critical Context

Obstacle avoidance system fully optimized (77x speedup: 218ms → 2.8ms). All tests passing, performance budget met with 35% headroom at 1000 entities. Next task: Implement sensor query API (query_cone) with position/velocity/distance + acoustic/bioluminescent/thermal channels. Sensors first, then gossip.

## What Was Accomplished

### 1. Obstacle Avoidance Performance Optimization

- Researched optimization techniques via technical-research-scout agent
- Implemented obstacle-centric architecture: loop over M=15 obstacles instead of N=1000 entities
- Added 3D spatial hash for entity culling (O(1) lookups, ~85-90% reduction)
- Vectorized per-obstacle math: process ~100 candidates at once per obstacle
- Used direct indexing for force accumulation (avoided numpy.add.at performance trap)
- Sequential blending: sphere → cylinder → plane (plane last for vertical safety)

**Performance Results:**
- Avoidance: 218ms → 2.8ms (77x faster)
- Total tick at 1000 entities: 192ms → 19.5ms (under 30ms budget, 35% headroom)
- Breakdown at 1000e: sphere 0.3ms (was 119ms), cylinder 0.9ms (was 85ms), plane 0.06ms (was 11ms)

### 2. Research Documentation

- Created `project/research/obstacle_avoidance_optimization_external_research_2025-10-01.md`
- Key findings: spatial hash > KD-tree for dynamic entities, numpy.add.at is 10-25x slower, obstacle-centric validated
- Documented expected performance gains (7-10x from spatial culling alone)

### 3. Testing and Validation

- All 4 avoidance tests passing: no penetration, determinism, speed limits, performance
- Determinism verified with spatial hash (sorted indices before accumulation)
- Correctness preserved through refactor

## Current Working State

### What IS Working:

- Phase 5 spatial optimization - 6.95x speedup, 16.978ms at 1000 entities (batch queries)
- Obstacle avoidance (spheres/cylinders/planes) - 2.8ms at 1000 entities, all correctness tests passing
- Spatial hash for entity culling - ~85-90% reduction in checks per obstacle
- Obstacle-centric processing - M=15 Python loops instead of N=1000
- Vectorized per-obstacle math - candidates processed in NumPy batches
- Performance instrumentation - timing breakdown available via AVOIDANCE_TIMING_BREAKDOWN flag

### What is PARTIALLY Working:

- None - avoidance system is production-ready

### What is NOT Working:

- Sensor query API - not yet implemented (next task)
- Knowledge gossip system - not yet implemented (after sensors)
- Spawn-inside-sphere robustness test - deferred, not critical

### Known Issues:

- None identified - system is stable and performant

## Next Immediate Steps

### Priority 1: Implement Sensor Query API (2-3 hours estimated)

**API Signature:**
```python
adapter.query_cone(
    origin: np.ndarray,      # (3,) world position
    direction: np.ndarray,   # (3,) normalized direction vector
    angle_deg: float,        # Cone half-angle in degrees
    range_m: float,          # Max distance
    flags: Set[str]          # {'position', 'velocity', 'distance', 'acoustic', 'bioluminescent', 'thermal'}
) -> SpatialQueryResult     # Dataclass with requested fields
```

**Implementation Plan:**
1. Define `SpatialQueryResult` dataclass in `aquarium/data_types.py`
2. Add emission fields to Species or use placeholders (acoustic_amplitude, acoustic_peak_hz, biolum_intensity, biolum_wavelength)
3. Implement `Entity.get_emission_multipliers()` if not exists (scales emissions by behavior state)
4. Add `query_cone()` to `aquarium/spatial_queries.py`:
   - Spatial culling: candidates within range_m sphere
   - Cone filtering: vectorized dot product for angle check (cos(theta) >= cos(angle_deg/2))
   - Flag-driven field population (only compute requested fields)
   - Channel scaling: multiply base emissions by behavior multipliers
   - Thermal sampling: distance to nearest vent sphere, radial falloff (thermal_delta = vent_base * (1 - d/influence))
5. Add sensor_ms timing to simulation.tick() performance breakdown

**Scope for MVP:**
- Position/velocity/distance (basic)
- Acoustic (amplitude, peak_hz)
- Bioluminescent (intensity, wavelength)
- Thermal (temperature_delta from vents)
- Magnetic/chemical (stubs, not implemented)

**Performance Target:** ≤2ms at 1000 entities for typical cone queries

**Tests Required:**
- Cone selectivity (angle/range variations)
- Flag presence/absence (requested fields present, others absent)
- Channel scaling (behavior multipliers affect output)
- Thermal falloff (distance to vent affects temperature)
- Determinism (identical results across runs)
- Performance at 143/500/1000 entities

**Design Decisions to Discuss:**
- Q1: Entity emission data source - Species fields vs placeholders?
- Q2: QueryFlags pattern - String set vs Enum vs Dict?
- Q3: SpatialQueryResult - Dataclass vs Dict vs NumPy array?
- Q4: Thermal implementation - All vents vs nearest vent vs pre-computed grid?

**Recommendations:**
- Q1: Pull from Species (cleaner, already loaded)
- Q2: String set (Python-friendly, clear intent)
- Q3: Dataclass (type-safe, good IDE support)
- Q4: Nearest vent sphere (fast, deterministic, good enough for MVP)

### Priority 2: Knowledge Gossip System (next session)

**Deferred to next session after sensors complete.**

Mechanics:
- Use `neighbors_within(entity, gossip_range, tag_filter={'social'})`
- Cap 1-2 exchanges per entity per tick, one exchange per pair per tick
- Decay tokens after gossip, evict by freshness threshold
- Enforce token capacity per entity
- First pass: ship_sentiment only, generic pipeline for future tokens

Tests:
- Propagation: seed ship_sentiment in 1 agent → 95% coverage by T with freshness > 0.2
- Determinism, bounded CPU, gossip_ms logging

## Files Created/Modified

**Created:**
- `project/research/obstacle_avoidance_optimization_external_research_2025-10-01.md` - Comprehensive research on optimization techniques
- `project/handoffs/session_007_2025-10-01_avoidance_optimized_sensors_next.md` - This handoff

**Modified:**
- `aquarium/avoidance.py` - Complete rewrite with obstacle-centric + spatial hash architecture
- `aquarium/constants.py` - Added AVOIDANCE_TIMING_BREAKDOWN flag (line 48), set to False for production
- `aquarium/simulation.py` - Updated timing instrumentation for avoidance_ms breakdown (lines 341-368)
- `aquarium/simulation.py` - Updated print_perf_breakdown() to show avoidance_ms (lines 492-539)

**Key Code Locations:**
- Spatial hash implementation: `aquarium/avoidance.py:209-234` (_build_spatial_hash)
- Sphere candidate query: `aquarium/avoidance.py:237-284` (_query_sphere_candidates)
- Cylinder candidate query: `aquarium/avoidance.py:287-332` (_query_cylinder_candidates)
- Sphere avoidance accumulation: `aquarium/avoidance.py:335-431` (_accumulate_sphere_avoidance)
- Cylinder avoidance accumulation: `aquarium/avoidance.py:433-549` (_accumulate_cylinder_avoidance)
- Speed clamping: `aquarium/avoidance.py:552-573` (_clamp_speeds)
- Phase A.5 integration: `aquarium/simulation.py:334-368`

## Key Insights/Learnings

**Architectural Insights:**
- Obstacle-centric beats entity-centric when M << N (15 vs 1000)
- Spatial culling is non-negotiable: 7-10x gain even with vectorization
- numpy.add.at() is 10-25x slower than direct indexing (confirmed by research)
- Python loop overhead negligible at M=15 (~30μs total)
- Vectorized operations on ~100 entities fit in L1 cache (ideal performance)

**Implementation Patterns:**
- Precompute obstacle arrays once per tick (centers, radii, influence_radii, ab vectors)
- Query spatial hash per obstacle, deduplicate with np.unique(), sort for determinism
- Vectorize math over candidates: broadcasting, masking, accumulation
- Sequential blending with speed clamping after each type preserves stability
- Direct indexing: `avoid_accumulator[affected_indices] += forces` (NOT np.add.at)

**Performance Validation:**
- Spatial hash cell size: 10m (tunable constant HASH_CELL_SIZE)
- Expected culling ratio: 85-90% (observed in practice)
- Sphere avoidance: 377x faster (119ms → 0.3ms)
- Cylinder avoidance: 98x faster (85ms → 0.9ms)
- Plane push: 186x faster (11ms → 0.06ms)

**Research Findings:**
- Spatial hash > KD-tree for dynamic uniform entities (no rebuild cost)
- Numba JIT would provide additional 3-10x if needed (not required at current scale)
- GPU acceleration counterproductive at N=1000 (transfer overhead > compute gain)
- BVH/Octree overkill for M=15 obstacles (simple array iteration optimal)

## Technical Notes

**Spatial Hash Implementation:**
- Cell size: 10.0 meters (HASH_CELL_SIZE constant in avoidance.py:25)
- 3D grid: (x, y, z) cell coordinates from floor(pos / cell_size)
- Neighbor query: check 27 cells (center + 26 neighbors) for spheres
- Neighbor query: check AABB cells for cylinders (expanded by influence + lookahead)
- Determinism: np.unique() deduplicates and sorts candidate indices before processing

**Obstacle Processing Order:**
- Sphere → Cylinder → Plane (plane LAST for vertical safety)
- Sequential blending: apply avoidance_weight (0.6) for lateral, seabed_weight (0.8) for vertical
- Speed clamping after each obstacle type to prevent over-correction

**Lookahead Distances:**
- Defined in AVOIDANCE_LOOKAHEAD_DISTANCES = [0.5, 2.0] meters
- Samples: entity.position + direction * d for each distance
- Find nearest obstacle to any sample (minimum distance across all samples)

**Performance Instrumentation:**
- Set AVOIDANCE_TIMING_BREAKDOWN = True in constants.py for per-type breakdown
- Adds sphere_ms, cylinder_ms, plane_ms to perf output
- Currently disabled (False) for production (zero overhead)
- Timing captured in simulation.tick() Phase A.5 (lines 345-368)

**Type Handling:**
- Obstacle start/end/center are lists from JSON, must convert to numpy arrays
- Fixed in avoidance.py:364 (sphere_center_arr), 464-465 (cylinder_start_arr/end_arr)
- All distance checks use np.linalg.norm, influence checks use linear falloff

**Constants and Configuration:**
- HASH_CELL_SIZE = 10.0 (avoidance.py:25) - tunable based on obstacle density
- AVOIDANCE_LOOKAHEAD_DISTANCES = [0.5, 2.0] (constants.py:36)
- AVOIDANCE_WEIGHT_DEFAULT = 0.6 (constants.py:39)
- INFLUENCE_RADIUS_FACTOR_DEFAULT = 2.5 (constants.py:42)

## Progress Metrics

- Phase 4 (Avoidance): 100% complete (optimized, tested, production-ready)
- Phase 5 (Spatial Optimization): 100% complete (6.95x speedup documented)
- Slice Exit Criteria Progress:
  - ✅ Phase 5 optimization complete (16.978ms at 1000e)
  - ✅ Obstacle avoidance complete (2.8ms at 1000e, 77x faster)
  - ✅ Performance ≤30ms at 1000e (19.5ms achieved, 35% headroom)
  - ✅ Determinism preserved (all tests passing)
  - ❌ Knowledge gossip + decay (not started, next after sensors)
  - ❌ Sensor query API (not started, immediate next task)
- Tests Passing: 14/14 (4 avoidance + 6 geometry + 4 behavior/batch)
- Context Window at Handoff: 70%

**Estimated Slice Completion:** 75% (avoidance done, sensors + gossip remain)

---

## Recommended Reading Order for Next Session

1. **START HERE**: This handoff (`project/handoffs/session_007_2025-10-01_avoidance_optimized_sensors_next.md`)
2. **Performance baseline**: `project/status/phase5_performance.md` (16.978ms at 1000e before avoidance optimization)
3. **Research findings**: `project/research/obstacle_avoidance_optimization_external_research_2025-10-01.md` (empirical evidence for architecture decisions)
4. **Optimized avoidance code**: `aquarium/avoidance.py` (complete obstacle-centric implementation with spatial hash)
5. **Simulation integration**: `aquarium/simulation.py:334-368` (Phase A.5 avoidance call and timing)
6. **Constants**: `aquarium/constants.py:31-48` (avoidance configuration and timing flag)
7. **Previous handoff**: `project/handoffs/session_006_2025-10-01_obstacle_avoidance_complete.md` (context before optimization)

**Optional (if implementing sensors immediately):**
8. **Spatial adapter**: `aquarium/spatial_queries.py` (where query_cone will be added)
9. **Data types**: `aquarium/data_types.py` (where SpatialQueryResult dataclass will be defined)
10. **Entity class**: `aquarium/entity.py` (check if get_emission_multipliers exists)
11. **Species definitions**: `data/species/*.json` (emission field candidates)

---

_Handoff prepared by Chronus Session 007_
_Obstacle avoidance optimized 77x (218ms → 2.8ms), sensor API next_
