# Session Handoff: Aquarium Simulation Foundation

**Created**: 2025-09-30 14:02:43
**From Session**: Session 001
**To**: Next Chronus Instance
**Context Window**: 65% (118k/180k safe limit)

## Critical Context

Shipmind is a turn-based exploration game where player + AI collaborate to investigate Titan's subsurface ocean. Core architectural decision: Build the mathematical aquarium simulation FIRST (standalone Python, deterministic, seed-driven ecosystem) before any game code. Unity and AI are consumers of the aquarium, not the other way around.

## What Was Accomplished

### 1. Complete Game Design Document

- Defined core premise: Signal investigation, damaged AI, isolation narrative, turn-based collaboration
- Established design pillars: collaborative discovery, sensor-mediated reality, consequence-driven narrative, hard sci-fi
- Documented turn contract (JSON I/O, ~200 tokens/turn target)
- Created tool registry (11 tools for player/AI interaction)
- Defined signal model (5 channels: acoustic, thermal, chemical, magnetic, visual)
- Clarified AI autonomy boundaries (AI manages subsystems, proposes actions, player executes)
- No offensive weapons (tools can be misused with consequences)
- Token limit: 180k safe ceiling (not 200k) with handoff recommended at 144k-162k

### 2. Shipmind Handoff System Specification

- Documented memory management as gameplay mechanic (damaged AI = context limits)
- Warning stages: 70% nominal, 80% warning, 90% alert, 95% critical, 100% collapse
- Handoff process: AI generates mission brief, conversation resets, brief becomes new context
- Failure state: Memory collapse = total amnesia, player must restore from archived briefs
- Session continuation: Claude SDK resume feature for quick saves
- Configuration parameters: Token budgets, warning thresholds, provider profiles

### 3. Paper Prototype: Drifter Encounter

- Completed 11-turn encounter simulation (Turn 47-57)
- Validated turn-based collaboration loop works
- Token usage: ~2,560 tokens total (~233 tokens/turn average, slightly over target but acceptable)
- Key learnings:
  - Player/AI mutual dependency requires careful design
  - Shipmind should NOT have meta-knowledge (discovers with player)
  - Species log is in-game document, updated when bio-data collected
  - Encounters need natural endings (Drifter leaves after curiosity satisfied)
  - Multiple valid player paths (passive, communicative, aggressive, ignore)

### 4. Aquarium Simulation Design Document (COMPLETE)

- 1200+ line specification document created
- Core principle: Aquarium is truth, Unity/sensors/AI are consumers
- Technology stack validated through external research:
  - Python 3.10+ with scipy.spatial.cKDTree for spatial indexing
  - No ABM framework (custom kernel adopting Mesa patterns)
  - Simple OOP classes (no ECS complexity)
  - Deterministic seed-based world generation
- Performance targets: 15ms/turn for 100-500 entities (validated via research)
- Two-fidelity simulation (MVP: active regions only, inactive regions paused)
- Query-first interface (sensors query aquarium, get structured results)
- Injection interface (player actions affect ecosystem)
- Complete test plan (6 tests: stability, propagation, predator-prey, determinism, performance, player impact)

### 5. External Research (4 Comprehensive Documents)

- Spatial indexing: scipy.cKDTree optimal for 100-500 entities, O(log N) queries ~0.15ms
- Agent-based modeling: Mesa patterns good, Mesa framework unnecessary (9-14x slower, experimental 3D)
- Dwarf Fortress architecture: Event-driven scheduling, no polymorphic ECS, determinism fragile
- Sensor simulation: Two-phase approach (spatial query + raycasting), GPU for visualization
- All research synthesized into aquarium design with rationale for every technology choice

### 6. Ecosystem Information Propagation Design

- Knowledge tokens: ship_sentiment, ship_capabilities, predator_location
- Gossip merge algorithm: weighted average based on reliability and freshness
- Token decay: linear (MVP), exponential (post-MVP)
- Species-to-species communication (proximity-based, range-limited)
- Information spreads through population over time (not instant)
- Player actions on one side of world take turns to propagate to other side
- Computational simulation (no rendering required for ecosystem updates)

## Current Working State

### What IS Working:

- Complete design documentation (GDD, Handoff System, Aquarium Simulation)
- Paper prototype validates core gameplay loop
- Technology choices validated through research
- Clear architectural direction (aquarium-first development)

### What is PARTIALLY Working:

- Data schemas defined in aquarium doc but not as separate YAML files yet
- Drifter species fully designed but not in final YAML schema format
- Vent Field Alpha biome conceptually complete but not as data pack

### What is NOT Working:

- No implementation code yet (by design - specification phase)
- No YAML data files created
- No Python simulator kernel built
- No Unity integration

### Known Issues:

- Token budget slightly over target in paper prototype (233 vs 200 tokens/turn) - need to verify in real implementation
- Species log access by AI needs tool definition (not yet in tool registry)
- Handoff generation by AI needs specific prompt engineering (not yet documented)

## Next Immediate Steps

1. **Create YAML Schema Files**
   - world/titan.yaml (planet parameters, biomes)
   - species/sp_001_drifter.yaml (complete behavioral definition from paper prototype)
   - interactions/rules.yaml (Drifter-Drifter gossip, Drifter-Shadow flee, Drifter-Ship evaluation)
   - knowledge/tokens.yaml (token types, decay rates, merge rules)

2. **Create Vent Field Alpha Data Pack**
   - Complete biomes/vent_field_alpha.yaml (12 vents, 3 ridges, 8 tube clusters)
   - Stub files for sp_002_tube_forest.yaml and sp_003_shadow.yaml
   - Emission profiles for acoustic/thermal/bioluminescent channels

3. **Build Python Simulator Prototype**
   - Implement core tick function (movement, encounters, knowledge propagation, decay)
   - Integrate scipy.cKDTree for spatial queries
   - Simple Drifter class with behavior evaluation
   - Query interface for sensors (spatial_cone, acoustic_sources, thermal_gradient)
   - Run Test 1 (population stability) and Test 2 (knowledge propagation)

## Files Created/Modified

**Created:**

- `project/GAME_DESIGN_DOCUMENT.md` - Complete game design specification (600+ lines)
- `project/SHIPMIND_HANDOFF_SYSTEM.md` - Memory management technical implementation (600+ lines)
- `project/AQUARIUM_SIMULATION.md` - Authoritative simulation specification (1200+ lines)
- `project/research/spatial_indexing_external_research_2025-09-30.md` - Research findings
- `project/research/agent_based_modeling_frameworks_external_research_2025-09-30.md` - Research findings
- `project/research/dwarf_fortress_simulation_external_research_2025-09-30.md` - Research findings
- `project/research/sensor_simulation_external_research_2025-09-30.md` - Research findings

**Modified:**

- `README.md` - Updated with project overview and goals (early session)

## Key Insights/Learnings

1. **Aquarium-first is correct approach** - User confirmed: world simulation is the foundation, everything else consumes it. Building game systems first would create a facade with no substance.

2. **Paper prototyping reveals design issues early** - 11-turn Drifter encounter exposed token budget concerns, AI knowledge scope problems, and player agency gaps before any code written.

3. **Research prevents reinventing wheels** - scipy.cKDTree, Mesa patterns, Dwarf Fortress lessons saved weeks of trial-and-error. Standing on shoulders of giants.

4. **Token efficiency is gameplay** - 200 tokens/turn target directly translates to session length before handoff. Tighter data = longer AI coherence = better player experience.

5. **Determinism scope matters** - Don't waste time on cross-platform replay determinism (even DF abandoned it). Focus on seed-based generation + save/load state restoration.

6. **Two-fidelity can wait** - Cohorts are optimization, not requirement. 100-500 entities run in 15ms/turn without cohorts. Prove core simulation first.

7. **Turn-based embraces LLM constraints** - Player and AI both accept turn-based nature. Makes AI collaboration natural, not forced. Matches LLM interaction pattern.

8. **Species are behavioral rules, not scripts** - Drifter is defined by decision tree + knowledge effects, not scripted sequence. Enables emergent encounters.

9. **Information propagation is the game** - Knowledge tokens spreading through populations creates dynamic world. Player actions have delayed, spreading consequences.

10. **No offensive weapons creates moral tension** - Tools can harm (high-power sonar, probes, hull contact) but require misuse. Player choices have weight.

## Technical Notes

**Critical architectural decisions:**

- **Simulation language:** Python 3.10+ (not TypeScript, not Unity C#) - needs to run standalone for testing
- **Spatial indexing:** scipy.spatial.cKDTree with leafsize=16, batch queries with workers=-1
- **No frameworks:** Custom kernel, no Mesa, no ECS, simple OOP classes
- **Data format:** YAML authoring → JSON runtime (compile step via Python script)
- **Determinism:** Seed-based generation, save/load serialization, NOT cross-platform replay
- **Token limit:** 180k safe ceiling (not 200k), handoff at 144k-162k recommended
- **Turn duration:** Variable (navigation = multiple turns, scan = 1 turn, probe = deploy + results turns)
- **Active region radius:** 2000m for MVP (configurable)

**Performance expectations:**

- Movement updates: <5ms
- cKDTree rebuild: <3ms
- Neighbor queries: <2ms
- Knowledge propagation: <2ms
- Behavior evaluation: <3ms
- Total: ~15ms/turn (66 turns/second if needed, instant for turn-based)

**Sensor division of labor:**

- Python aquarium: Entity positions, emissions, spatial queries (cheap, <0.15ms)
- Unity: Raycasting for occlusion, visualization, audio spatialization (expensive, needs optimization)

**Species behavior pattern:**

```python
# Priority-based decision tree (simple agents)
if has_predator_nearby(80m):
    return flee
elif has_ship_sentiment < -0.5 and ship_distance < 150m:
    return flee_ship
elif ship_distance < 300m and sentiment >= 0:
    return investigate_ship
else:
    return forage
```

**Knowledge token structure:**

```json
{
  "kind": "ship_sentiment",
  "value": 0.3,
  "freshness": 0.8,
  "reliability": 0.9,
  "source": "direct"
}
```

## Progress Metrics

- Design Phase: 100% complete (GDD, Handoff System, Aquarium Simulation)
- Data Schema: 0% (specifications in docs, not YAML files yet)
- Implementation: 0% (by design, specification first)
- Testing: 0% (no code to test yet)
- Context Window at Handoff: 65% (118k/180k)

---

_Handoff prepared by Chronus Session 001_
_Established aquarium-first architecture with complete technical specifications and research validation_
