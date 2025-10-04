# Shipmind Aquarium — Progress Report

Last Updated: 2025-10-04

**Status**
- Phases 1–3 complete: data loader, minimal tick, behavior evaluation.
- Phase 4/5 spatial optimization complete: cKDTree + vectorized batch via per-tag/per-biome subtrees; array caches; legacy path retained for A/B.
- Determinism preserved; regression and batch determinism tests passing.
- Obstacle avoidance optimized (obstacle-centric + spatial hash) and validated within budget.
- Sensor API (Step 1) complete: cone query (position/velocity/distance) with perf headroom.
- Sensor channels (Phase A) complete: acoustic + bioluminescent, with OPTICAL alias + provenance.
- **Knowledge Gossip (Phase 1) COMPLETE**: radius-based push-pull v2 with version+timestamp; SoA cache; ~1.3ms @ 1000 entities.
- **Knowledge Gossip (Phase 2a) COMPLETE**: exponential decay, attenuation, capacity enforcement; p50 1.88ms @ 1000 entities.
- **Knowledge Gossip (C7 Phase 4) COMPLETE**: Multi-kind support (ship_sentiment + predator_location); most_recent merge algorithm; position value_type; species-level capacity.
- **Knowledge Gossip (Phase 7 Network Telemetry) COMPLETE**: Degree histogram (isolated/sparse/connected) with O(E) computation; logged every GOSSIP_LOG_INTERVAL ticks; test suite validated.
- **Ecosystem Phase 1 (A+B+C) COMPLETE**: Resource fields, energy drain/feeding, death/compaction, hunger-driven behaviors; integration test suite (10 tests); _count invariant fix; behavior cache fallback fix.

**What's New**
- **Ecosystem Phase 1 Complete**:
  - Resource field extraction from biome obstacles (Gaussian density fields with peak/sigma)
  - Energy drain/feeding cycle: base metabolic drain + movement cost; density sampling with feeding efficiency
  - Death detection (energy <= starvation_threshold) + compaction (SoA arrays + entity list in lockstep)
  - Hunger-driven behaviors: urgent-forage (<30 energy), flee-predator/flee-hostile-ship priority validation
  - Alive-after-drain revival guard (prevents same-tick revival at threshold)
  - Integration test suite (10 tests): resources, thresholds, density, revival, compaction, behaviors, dict guards
- **Critical Fixes**:
  - `_count` invariant fix: Phase A/B.9 use local `build_count`; B.75 remains sole writer of `_count == len(entities)`
  - Behavior cache fallback: nearest_entity now falls back to per-entity query when cache misses (handles condition.max_distance > cache radius)
- Priority behavior engine (flee > investigate > forage) with first-match wins.
- Ship modeled as inert actor (tag-based targeting; update_ship injection).
- Per-entity emission multipliers tracked (sensor-ready).
- Central constants module (token defaults, cruise fraction, timing window).
- Multiple-predators test validates nearest selection in two-phase tick.
- Spatial Adapter: cKDTree integrated with per-tag/per-biome subtrees and vectorized k=1 batch queries; array-based caches eliminate dict overhead.
- Sensor cone query implemented (KD-tree range + vectorized angle filter); DTOs (EntityHit, SpatialQueryResult) with to_dict() serialization; perf timing (`sensor_ms`).
- Optical plan (Path B+): keep explicit `bioluminescent_*` fields; add optional `optical_*` alias with `optical_components` provenance ['bioluminescent'] for future aggregation of artificial/reflected sources.
- Sensor channels Phase A: acoustic + bioluminescent populated from baked base_emissions × behavior multipliers; OPTICAL alias populated with provenance; single compute path for dual‑flag requests.

**Metrics**
- Phase 2 avg tick (10 entities, no behaviors): ~0.042 ms.
- 143 entities (single biome): behavior 4.95x faster with batch; modest overall gain (build dominates).
- 1000 entities (pre-avoidance opt): 22.348 ms/tick (down from 118 ms baseline), 5.28x overall speedup; determinism intact.
- 1000 entities (with optimized avoidance): 19.5 ms/tick total; avoidance 2.8 ms (was 104–218 ms) — 77x faster; ~35% headroom under 30 ms target.
- Sensor cone (Step 1, basic DTO only):
  - 143 entities: ~0.171 ms
  - 500 entities: ~0.437 ms
  - 1000 entities: ~0.508 ms (≈4x under ≤2 ms target)
- Sensor channels Phase A (acoustic + bioluminescent + OPTICAL alias):
  - 143 entities: ~0.470 ms
  - 500 entities: ~0.955 ms
  - 1000 entities: ~1.162 ms (≈58% under ≤2 ms target)
- Knowledge Gossip Phase 1 (radius push-pull, version+timestamp, SoA cache):
  - 1000 entities: p50 ~1.6ms, profiled 1.29ms (35% under <2ms target)
  - Profiled breakdown: extract 0.41ms (31%), edge_query 0.60ms (46%), merge 0.15ms, writeback 0.14ms
  - Test coverage: ≥95% propagation, determinism, pair constraint, version precedence
- Knowledge Gossip Phase 2a (lifecycle: decay, attenuation, eviction, capacity):
  - Multi-N performance (median of 7 runs):
    - 143 entities: p50 0.36ms, p90 0.53ms
    - 500 entities: p50 1.69ms, p90 2.25ms (15.7% headroom)
    - 1000 entities: p50 1.88–2.06ms (pytest typically 1.88ms), p90 ~2.96ms
    - 2000 entities: p50 4.14ms, p90 5.22ms (log-only, no assertion)
  - Phase 2 overhead: +0.27ms over Phase 1 (decay+attenuation vectorized)
  - Test coverage: decay curves, staleness eviction, capacity enforcement, attenuation, determinism, propagation preserved
  - Note: pytest shows 1.88ms @ 1000 entities (test harness overhead differs from standalone)
- Knowledge Gossip C7 Phase 4 (multi-kind: ship_sentiment + predator_location):
  - Single-kind performance (1000 entities, 7 runs):
    - p50: ~1.5-1.7ms, p90: ~2.7-2.9ms (target <2.2ms pytest median) ✅
  - Multi-kind performance (1000 entities, 2 kinds, 7 runs):
    - Profiled breakdown: edge_query 1.25ms (once), extract 0.36ms, decay 0.12ms, merge 0.54ms, writeback 0.22ms = ~2.5ms core
    - Measured (pytest): p50 ~2.7-3.3ms, p90 ~4.5-6.0ms (target <3.5ms pytest median) ✅
    - Standalone (no pytest): typically ~2.6-3.0ms
  - Features:
    - most_recent merge algorithm (version > last_tick > reliability precedence with epsilon 1e-12)
    - Position value_type (3-tuple copy, no SoA val array, no averaging)
    - Species-level capacity override (dict/object support, fallback to GOSSIP_TOKEN_CAP_DEFAULT=16)
    - Capacity skip optimization (skip loop when max_tokens ≤ min_cap)
    - Profiler accumulates per-kind times correctly
  - Test coverage (23/23 functional passing):
    - Phase 1: 5 tests (propagation, determinism, pair constraint, performance, freshness)
    - Phase 2a: 6 tests (decay, eviction, capacity, attenuation, determinism, propagation preserved)
    - C7 Phase 4: 8 tests (most_recent precedence ×3, position copy, attenuation, capacity, determinism, performance)
    - Phase 7 Telemetry: 4 tests (API contract, two-node sanity, lattice layout, sparse layout)
- Determinism: bit-for-bit identical positions and behavior IDs across backends/toggles.

**Risks**
- O(n) neighbor scans won’t scale — replace with cKDTree (Phase 4).
- Investigate standoff/steering deferred until avoidance exists (Phase 5+).
- Schema/YAML drift risk — keep emission multipliers and depth range conventions aligned.

**Next**
- Species gossip ranges: populate gossip_range_m in species YAMLs (currently using fallback 15m)
- Optional optimizations (backlog):
  - Combine per-kind passes to reuse edge arrays in one sweep
  - Pre-allocate scratch buffers across kinds
  - Tune CKDTREE_LEAFSIZE for query_pairs optimization
- Performance validation at 143/500/1000/2000 with multi-kind enabled; log breakdowns every 200 ticks.

Spatial Adapter API
- Locked for Phase 4; see `project/docs/SPATIAL_ADAPTER_API.md` for function signatures and DTOs.

**Decisions (Confirmed)**
- Same-biome search scope for nearest/neighbor queries.
- Cruise fraction (0.2) only for fallback; behaviors set speed via multipliers.
- Dynamic emissions via simple per-channel multipliers.
- Two-phase tick: A) evaluate desired velocities from start-of-tick state; B) apply movement/reflect/clamp.

**Artifacts**
- Schemas: `data/schemas/*.json` (species, biome, world, interaction, knowledge_token).
- Content: `data/world/titan.yaml`, `data/biomes/vent-field-alpha.yaml`, `data/species/*.yaml`, `data/interactions/rules.yaml`, `data/knowledge/tokens.yaml`.
- Python: `aquarium/loader.py`, `aquarium/data_types.py`, `aquarium/simulation.py`, `aquarium/spawning.py`, `aquarium/entity.py`, `aquarium/behavior.py`, `aquarium/spatial.py`, `aquarium/rng.py`, `aquarium/constants.py`.
- Spatial Adapter: `aquarium/spatial_queries.py` (cKDTree backend + per-tag/per-biome subtrees; vectorized batch; legacy toggle).
- Perf Harness/Tests: `aquarium/tests/perf_harness.py`, `aquarium/tests/test_batch_queries.py`, `aquarium/tests/test_batch_performance.py`.
- Sensors: `aquarium/tests/test_sensor_queries.py`; DTOs in code; `sensor_ms` perf timing in `simulation.py`.
- Ecosystem: `aquarium/tests/test_ecosystem_phase1_integration.py` (10 integration tests covering resources, thresholds, feeding, death, compaction, behaviors, dict guards).
- Tests: `aquarium/tests/test_loader.py`, `aquarium/tests/test_tick_minimal.py`, `aquarium/tests/test_behavior.py`.

**Open Items / Notes**
- Add depth-band unit tests when a species uses `within_depth_band`.
- Ensure species depth ranges use min ≤ max with negative depths across all YAMLs.
- Align schema to YAML for `emission_multipliers` (per-channel) if not already updated.
- Consider CKDTREE_LEAFSIZE/CKDTREE_WORKERS tuning at higher N if batch_query_ms grows.

**Planned Checkpoints**
- C1: Vectorized batch via subtrees; array caches; A/B determinism at 143/500/1000; 1000 @ <30 ms/tick. (Completed)
- C2: Obstacle avoidance validated against vents/ridges/seabed; perf within budget at 1000 entities. (Completed)
- C3: Sensor cone (Step 1) implemented and validated under ≤2 ms at 1000 entities. (Completed)
- C4a: Sensor channels Phase A (acoustic + bioluminescent + OPTICAL alias) implemented and tested. (Completed)
- C4b: Sensor thermal channel implemented and tested. **COMPLETED** ✅
- C5: Knowledge propagation test passes (coverage ≥95%, version precedence). **COMPLETED** ✅
- C6: Gossip Phase 2a lifecycle complete (decay, attenuation, eviction, capacity); perf p50 <2ms @ 1000. **COMPLETED** ✅
- C7: Multi-kind tokens (ship_sentiment + predator_location); most_recent merge; position value_type. **COMPLETED** ✅
- C7-Phase7: Network health telemetry (degree histogram) with O(E) computation; interval logging; test validation. **COMPLETED** ✅
- C8: Ecosystem Phase 1 (A+B+C) complete; resource fields, energy/feeding, death/compaction, hunger behaviors; integration tests (10); _count invariant; cache fallback. **COMPLETED** ✅

**Slice Exit Criteria (Vent Field Alpha)**
- Avoidance: entities do not penetrate obstacles; speeds clamped; determinism intact; tests green.
- Knowledge: propagation from 1 → population with decay/evict; ≥95% coverage by target T; tests green.
- Sensors: adapter.query_cone returns flat DTO by flags; emissions scaled by behavior state; thermal sampling around vents; tests green.
- Performance: 1000 entities ≤ 30 ms/tick with avoidance + gossip enabled; determinism preserved.
