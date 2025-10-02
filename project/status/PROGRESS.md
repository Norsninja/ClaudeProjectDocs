# Shipmind Aquarium — Progress Report

Last Updated: 2025-09-30

**Status**
- Phases 1–3 complete: data loader, minimal tick, behavior evaluation.
- Phase 4/5 spatial optimization complete: cKDTree + vectorized batch via per-tag/per-biome subtrees; array caches; legacy path retained for A/B.
- Determinism preserved; regression and batch determinism tests passing.
- Obstacle avoidance optimized (obstacle-centric + spatial hash) and validated within budget.
- Sensor API (Step 1) complete: cone query (position/velocity/distance) with perf headroom.
- Sensor channels (Phase A) complete: acoustic + bioluminescent, with OPTICAL alias + provenance.
- **Knowledge Gossip (Phase 1) COMPLETE**: radius-based push-pull v2 with version+timestamp; SoA cache; ~1.3ms @ 1000 entities.
- Remaining for the slice: thermal channel (nearest‑vent falloff) and gossip decay/eviction (Phase 2).

**What’s New**
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
- Determinism: bit-for-bit identical positions and behavior IDs across backends/toggles.

**Risks**
- O(n) neighbor scans won’t scale — replace with cKDTree (Phase 4).
- Investigate standoff/steering deferred until avoidance exists (Phase 5+).
- Schema/YAML drift risk — keep emission multipliers and depth range conventions aligned.

**Next**
- Sensor channels (Step 2 & 3):
  - Acoustic + bioluminescent fields; add `optical_*` alias with `optical_components=['bioluminescent']`; scale via behavior multipliers; clamp ranges.
  - Thermal via nearest‑vent radial falloff (use `thermal_base_delta` on vent spheres; fallback constant).
  - Extend tests for scaling, flags, falloff; keep `sensor_ms` within target.
- Knowledge gossip + decay:
  - neighbors_within via adapter; cap exchanges; decay → evict; enforce token capacity; propagation test ≥95% coverage by T.
- Performance validation at 143/500/1000 with channels and gossip enabled; log breakdowns every 200 ticks.

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
- C4b: Sensor thermal channel implemented and tested.
- C5: Knowledge propagation test passes (coverage ≥95%, version precedence). **COMPLETED** ✅
- C6: 500/1000-entity perf meets targets; determinism holds after avoidance + sensors + gossip.

**Slice Exit Criteria (Vent Field Alpha)**
- Avoidance: entities do not penetrate obstacles; speeds clamped; determinism intact; tests green.
- Knowledge: propagation from 1 → population with decay/evict; ≥95% coverage by target T; tests green.
- Sensors: adapter.query_cone returns flat DTO by flags; emissions scaled by behavior state; thermal sampling around vents; tests green.
- Performance: 1000 entities ≤ 30 ms/tick with avoidance + gossip enabled; determinism preserved.
