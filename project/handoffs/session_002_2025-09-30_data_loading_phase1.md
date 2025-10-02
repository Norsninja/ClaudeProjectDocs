# Session Handoff: Data Loading Complete, Phase 2 Planned

**Created**: 2025-09-30
**From Session**: Session 002
**To**: Next Chronus Instance
**Context Window**: 65% (130k/200k tokens)

## Critical Context

Built complete YAML data loading system and validated all game data packs (species, biomes, world, interactions, knowledge tokens). Phase 1 complete. Phase 2 scope defined: minimal tick loop with entity spawning and movement, no behaviors yet.

## What Was Accomplished

### 1. Schema/YAML Alignment (Preflight Fixes)

- Fixed emission_multipliers naming (simple per-channel: acoustic, thermal, chemical, magnetic, bioluminescent)
- Changed tags to freeform strings (removed enum restriction)
- Renamed depth fields: min_meters/max_meters → shallow_limit_meters/deep_limit_meters
- Allowed token_capacity: 0 for sessile species (Tube Forest)
- Confirmed seeds already present in world/biome schemas
- Result: All schema conflicts resolved, YAMLs align with schemas

### 2. Monitoring System Design

- Designed profiling.py: Per-tick timing breakdowns (movement_ms, kdtree_ms, behaviors_ms, gossip_ms, decay_ms)
- Designed cli.py: Run command with --verbose, --ticks, --seed, --snapshot-interval
- Console output: Live tick summaries every 10 ticks
- CSV logging: Full per-tick metrics for post-analysis
- JSON snapshots: Full entity state dumps every N ticks
- Query interface: Interactive snapshot inspection (deferred to Phase 3+)

### 3. Phase 1 Implementation: Data Loading System

- Created aquarium/__init__.py (package root)
- Created aquarium/data_types.py (42 dataclasses mirroring YAML schemas)
- Created aquarium/loader.py (YAML parser with optional JSON Schema validation)
- Implemented load_species(), load_biome(), load_world(), load_token_definitions(), load_interaction_rules()
- Implemented load_all_data() convenience function
- Created aquarium/tests/test_loader.py (Phase 1 validation)
- All tests pass: Loaded 3 species, 1 biome (143 entities to spawn), 5 token types, 3 interaction rules

### 4. Phase 2 Scope Definition

- Goal: Spawn entities, tick simulation, update positions (no behaviors yet)
- Spawn 10 Drifters uniform distribution for testing
- Tick 100 times, verify movement
- Implement bounds reflection (entities bounce at biome radius)
- Add determinism test (same seed = identical output)
- Log tick times (no performance assertion yet, insufficient workload)
- Files to create: entity.py, rng.py, spawning.py, simulation.py, spatial.py, tests/test_tick_minimal.py

### 5. Technical Decisions Locked

- Coordinate system: Y-axis vertical (depth negative), X/Z horizontal, meters
- Biome center [0, -100, 0] = 100m deep
- Deterministic RNG: SHA256 seed hashing, make_seed(world_seed, biome_id, entity_id, component)
- Entity iteration: Sort by instance_id before tick (stable order)
- Phase 2 velocity: Random unit vector, 20% of max_speed, stored per-instance
- Spawn distribution: uniform only (clustered/near_obstacles stubbed)
- Bounds policy: Reflect velocity at biome radius

## Current Working State

### What IS Working:

- YAML data loading system (loader.py, data_types.py)
- All 13 data files load correctly (3 species, 1 biome, 1 world, 1 interactions, 1 tokens)
- Schema validation (optional, via jsonschema)
- Data integrity verified (Phase 1 tests pass)

### What is PARTIALLY Working:

- None (Phase 1 complete)

### What is NOT Working:

- Entity spawning (Phase 2, not implemented yet)
- Simulation tick loop (Phase 2, not implemented yet)
- Entity movement (Phase 2, not implemented yet)
- All subsequent phases (3-7)

### Known Issues:

- Token limit documentation conflict: Claude Code status shows 200k max, design docs use 180k safe limit. Use 180k as ceiling, handoff at 144k-162k (80-90%).
- Windows console: Avoid Unicode symbols (✓❌) in test output, use [OK][FAIL] instead.

## Next Immediate Steps

1. **Create Phase 2 Files**
   - aquarium/entity.py: Entity class with position, velocity, instance_id, size_factor
   - aquarium/rng.py: make_seed(components), random_unit_vector(seed)
   - aquarium/spawning.py: spawn_entities(biome, species_registry, world_seed), uniform distribution only
   - aquarium/simulation.py: AquariumSimulation.tick(), entity position updates, bounds reflection
   - aquarium/spatial.py: distance_3d(), distance_to_sphere(), distance_to_plane()
   - aquarium/tests/test_tick_minimal.py: Movement test + determinism test

2. **Phase 2 Success Criteria**
   - Spawn 10 Drifters in Vent Field Alpha biome
   - Tick 100 times (100 seconds simulation time)
   - Entities move according to stored velocity
   - Bounds reflection works (entities bounce at radius=500m)
   - Determinism test passes (same seed = identical positions)
   - Tick times logged to console/CSV

3. **After Phase 2**
   - Phase 3: Behavior evaluation (priority list, conditions, action-to-velocity)
   - Phase 4: cKDTree spatial indexing
   - Phase 5: Obstacle avoidance
   - Phase 6: Knowledge tokens + gossip
   - Phase 7: Test suite (stability, propagation, determinism, performance)

## Files Created/Modified

**Created:**

- `aquarium/__init__.py` - Package root
- `aquarium/data_types.py` - 42 dataclasses (Species, Biome, World, TokenDefinition, InteractionRule, etc.)
- `aquarium/loader.py` - YAML parser with validation (380 lines)
- `aquarium/tests/test_loader.py` - Phase 1 validation tests
- `data/schemas/species.schema.json` - Updated with fixes (emission_multipliers, freeform tags, depth rename, capacity=0)
- `data/schemas/biome.schema.json` - Added seed field
- `data/schemas/world.schema.json` - Added seed field

**Modified:**

- `data/species/sp-001-drifter.yaml` - Fixed emission_multipliers, depth_range
- `data/species/sp-003-shadow.yaml` - Fixed emission_multipliers, depth_range
- `project/SHIPMIND_HANDOFF_SYSTEM.md` - Updated 200k→180k token limits (4 locations)
- `project/GAME_DESIGN_DOCUMENT.md` - Added AI Knowledge Scope section

## Key Insights/Learnings

1. **Dataclass inheritance with optional fields breaks in Python 3.10**: Removed Obstacle base class, flattened to SphereObstacle, CylinderObstacle, PlaneObstacle with duplicated fields. Python dataclasses require all required fields before optional fields, inheritance complicates this.

2. **Windows console encoding**: Avoid Unicode symbols in Python print() output. Windows cmd uses cp1252 encoding which doesn't support ✓❌ symbols. Use ASCII [OK][FAIL] instead.

3. **Deterministic RNG requires per-instance velocity storage**: Cannot re-generate velocity from seed each tick (would need tick count in seed, messy). Generate velocity once at spawn, store in Entity, use for all ticks. Phase 3 will replace with behavior-driven velocity.

4. **Bounds reflection preserves energy**: Reflecting velocity at biome boundary prevents entity clustering at edges, maintains natural distribution. Simple implementation: `velocity -= 2 * dot(velocity, normal) * normal`.

5. **Uniform spawning is sufficient for MVP**: Clustered and near_obstacles distributions add complexity without validating core simulation loop. Defer to post-MVP. Phase 2 proves tick mechanism works.

6. **SD architectural oversight is valuable**: Caught token limit discrepancy (200k vs 180k), recommended reflection over clamping, suggested adding determinism test early (Phase 2 not Phase 7). Prevents bugs from propagating through phases.

## Technical Notes

**Deterministic RNG Pattern:**
```python
from aquarium.rng import make_seed, random_unit_vector
entity_seed = make_seed(world_seed, biome_id, species_id, entity_index)
velocity_seed = make_seed(entity_seed, "initial_velocity")
direction = random_unit_vector(velocity_seed)
```

**Bounds Reflection:**
```python
offset = position - center
distance = np.linalg.norm(offset)
if distance > radius:
    normal = offset / distance
    velocity = velocity - 2 * np.dot(velocity, normal) * normal
    position = center + normal * radius  # Clamp to surface
```

**Entity Iteration for Determinism:**
```python
entities.sort(key=lambda e: e.instance_id)  # Before each tick
for entity in entities:
    entity.update_position(delta_time)
```

**Phase 2 Velocity (Replaced in Phase 3):**
```python
# Spawn time only:
direction = random_unit_vector(velocity_seed)
speed = species.movement.max_speed_ms * 0.2  # 20% max (slow drift)
entity.velocity = direction * speed
```

**Dependencies:**
- numpy >= 1.21.0
- scipy >= 1.7.0 (for Phase 4 cKDTree)
- pyyaml >= 6.0
- jsonschema >= 4.0

**Research Documents from Session 001 (Still Valid):**
- `project/research/behavior_tree_formats_external_research_2025-09-30.md` - Priority list pattern (Minecraft-style)
- `project/research/spatial_query_interfaces_external_research_2025-09-30.md` - Flat DTO with flags, orjson performance
- `project/research/collision_avoidance_external_research_2025-09-30.md` - Ahead-vector avoidance, <2ms for 500 entities

## Progress Metrics

- Design Phase: 100% (Session 001)
- Data Schema: 100% (Session 002)
- Phase 1 (Data Loading): 100% (Session 002)
- Phase 2 (Tick Loop): 0% (planned, not implemented)
- Phases 3-7: 0%
- Context Window at Handoff: 65% (130k/200k tokens)

---

_Handoff prepared by Chronus Session 002_
_Data loading system complete, Phase 2 tick loop ready to implement_
