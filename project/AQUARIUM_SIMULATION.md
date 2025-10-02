# Aquarium Simulation - Design Specification

## Overview

The aquarium simulation is the authoritative world model for Shipmind. It is a deterministic, seed-driven ecosystem that exists independently of player observation. All game systems (sensors, AI, player actions) interact with the aquarium through defined interfaces.

**Core Principle:** The aquarium is truth. Unity renders it. Sensors query it. AI interprets it. Player affects it.

## Design Principles

### 1. Deterministic and Seed-Driven
**Determinism scope (based on Dwarf Fortress lessons):**
- **World generation:** Same seed → same vent locations, species starting positions, terrain (fully deterministic)
- **Initial state:** Same seed → same entity IDs, knowledge initialization (fully deterministic)
- **Simulation:** Deterministic within a session, reproducible via save/load
- **Cross-platform:** Not guaranteed (floating-point differences acceptable)

**Why this approach:**
- Even Dwarf Fortress (20+ years development) abandoned full cross-platform determinism
- Floating-point math breaks across platforms/compilers
- Physics engines are inherently non-deterministic
- **Key insight:** Determinism needed for save/load and debugging, not for speedrunning or replays

**Implementation:**
- All randomness derived from `hash(global_seed, region_seed, entity_id, turn)`
- Save/load serializes exact state (perfect restoration)
- Player actions create divergence from base seed (butterfly effect)

### 2. Two-Fidelity Simulation
**Active regions (near player):**
- Individual entities with full simulation
- Per-turn updates (movement, knowledge exchange, interactions)
- High spatial resolution

**Inactive regions (far from player):**
- **MVP:** Paused entirely (no updates, frozen state)
- **Post-MVP:** Cohort aggregates (population counts, average knowledge)
- **Post-MVP:** Periodic updates (every N turns) or event-driven wakeups
- **Post-MVP:** Spawn individuals on-demand when region activates

**Rationale:**
- RimWorld, Dwarf Fortress use two-fidelity for performance with 10,000+ entities
- Our target: 100-500 entities (adequate performance without cohorts)
- **Lesson learned:** Prove core simulation works before adding cohort complexity
- Cohorts are optimization, not requirement

### 3. Information-First Ecosystem
Species don't just move and eat - they exchange knowledge:
- Ship sentiment (hostile/neutral/friendly)
- Ship capabilities (observed behaviors)
- Predator locations (warnings)
- Food sources (quality, location)

Knowledge propagates through communication, decays over time, affects behavior.

### 4. Query-First Interface
External systems don't read simulation state directly - they query it:
- "What entities are within sonar range at bearing X?"
- "What acoustic emissions are detectable from position Y?"
- "What's the thermal gradient at location Z?"

Simulator returns structured results based on sensor parameters.

### 5. Unity and AI Are Consumers
- Simulator has no dependency on Unity or AI systems
- Standalone Python module with JSON API
- Can run headless for testing (1000 turns in seconds)
- Unity visualizes subset of state
- AI receives processed observations, not raw state

## Architecture

### Components

```
┌─────────────────────────────────────────┐
│  Aquarium Simulator (Python)            │
│  ┌─────────────────────────────────┐   │
│  │  World State                    │   │
│  │  - Spatial entities             │   │
│  │  - Knowledge networks           │   │
│  │  - Emission fields              │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Simulation Kernel              │   │
│  │  - Movement                     │   │
│  │  - Encounters                   │   │
│  │  - Knowledge propagation        │   │
│  │  - Decay                        │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Query Interface                │   │
│  │  - Spatial queries              │   │
│  │  - Sensor simulations           │   │
│  │  - Emission field sampling      │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Injection Interface            │   │
│  │  - Player events                │   │
│  │  - Knowledge tokens             │   │
│  │  - Spawn/despawn                │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
           ↕ JSON API
┌─────────────────────────────────────────┐
│  Unity Game Client                      │
│  - Sensor systems                       │
│  - Player input                         │
│  - Visual rendering                     │
│  - AI integration                       │
└─────────────────────────────────────────┘
```

### Data Flow

**Per Turn:**
1. Unity sends player action → Simulator injects event
2. Simulator runs `tick()` → updates world state
3. Unity queries sensors → Simulator returns observations
4. Unity formats observations → AI receives turn data
5. AI responds with analysis → Unity displays to player
6. Repeat

**Initialization:**
1. Load world schema (YAML) → Compile to JSON pack
2. Initialize simulator with seed
3. Generate initial world state (terrain, entities, emissions)
4. Unity connects to simulator API

**Save/Load:**
1. Serialize world state + turn number + seed
2. On load: Restore state or regenerate from seed + replay events

## Data Schemas

### World Schema

**Structure:**
```yaml
# world/titan.yaml

world:
  id: titan
  type: moon
  radius: 2575000  # meters
  coordinate_system: spherical  # lat, lon, depth
  gravity: 1.352  # m/s²

  ocean:
    composition: ammonia_water
    depth_range: [-6000, 0]  # meters below ice
    temperature_range: [-2, 10]  # Celsius

  ice_crust:
    thickness: 100000  # meters
    penetration_point: {lat: 45.2, lon: -120.5}  # player entry

  seed: 0  # global seed, set per save file

biomes:
  - id: vent_field_alpha
    type: hydrothermal_vent_field
    location: {lat: 45.2, lon: -120.5}
    extent: {radius: 5000}  # meters
    depth_range: [-3200, -2900]
    temperature_range: [2, 8]
    seed_offset: 1  # derives from world seed

    features:
      - type: hydrothermal_vent
        count: 12
        placement: poisson_disk  # min distance between vents
        min_distance: 400
        heat_output_range: [100, 300]  # watts

      - type: ridge_formation
        count: 3
        pattern: network
        elevation_range: [50, 150]  # meters above base

      - type: tube_forest_cluster
        count: 8
        placement: near_vents  # within 100m of vents
        density_range: [0.5, 0.9]  # coverage
```

**Compiled Runtime (JSON):**
```json
{
  "world_id": "titan",
  "seed": 0,
  "biomes": [
    {
      "id": "vent_field_alpha",
      "center": {"lat": 45.2, "lon": -120.5, "depth": -3050},
      "radius": 5000,
      "seed": 1,
      "features": [
        {
          "id": "vent_alpha_01",
          "type": "hydrothermal_vent",
          "location": {"x": -105.2, "y": 462.1, "depth": -3102},
          "heat_output": 220,
          "seed": 101
        }
      ]
    }
  ]
}
```

### Species Schema

**Structure:**
```yaml
# species/sp_001_drifter.yaml

species_id: sp_001_drifter
name: null  # Player assigns on discovery
classification:
  type: mobile_megafauna
  family: bioluminescent_cephalopod
  size_range: [2.8, 3.4]  # meters
  mass_range: [180, 250]  # kg

habitat:
  biomes: [vent_field, ridge_network]
  depth_range: [2800, 3500]
  temperature_range: [2, 8]
  suitability_weights:
    near_vent: 0.6  # within 500m
    in_biome: 0.9
    depth_match: 0.8
    temp_match: 0.7

population:
  base_density: 0.00001  # per cubic meter in suitable habitat
  max_per_region: 50  # MVP cap
  spawn_on_activation: true

movement:
  base_speed: 0.3  # m/s
  max_speed: 1.2  # when fleeing
  steering:
    toward_food: 0.4
    away_predator: 0.8
    toward_conspecific: 0.3
    random_drift: 0.2

emissions:
  - channel: acoustic
    pattern: click_pulse_pair
    frequency: 1200  # Hz
    interval: 0.8  # seconds
    strength: 0.6
    range: 500  # meters
    intermittency: 0.9

  - channel: bioluminescent
    pattern: synchronized_pulse
    color: [0.0, 0.8, 0.9]  # RGB cyan
    brightness: 0.7
    range: 100
    sync_with: acoustic

knowledge:
  max_tokens: 5
  token_types:
    - ship_sentiment
    - ship_capabilities
    - predator_location
    - food_source

  communication:
    range: 100  # meters
    method: gossip
    capacity: 2  # max exchanges per turn

behavior:
  archetype: curious_investigator

  decision_tree:
    priority:
      - name: flee_threat
        condition: |
          (has_predator_nearby AND distance < 80) OR
          (has_ship_sentiment AND sentiment < -0.5 AND ship_distance < 150)
        action:
          type: flee
          speed: 1.2
          direction: away_from_threat
          until: distance > 200

      - name: investigate_ship
        condition: |
          ship_distance < 300 AND
          (ship_sentiment > 0 OR ship_sentiment == null)
        action:
          type: approach
          speed: 0.4
          maintain_distance: [60, 100]
          emit: true

      - name: respond_communication
        condition: ship_acoustic_detected AND ship_sentiment >= 0
        action:
          type: communicate
          pattern: mirror_with_variation

      - name: forage
        condition: default
        action:
          type: wander
          speed: 0.3
          bias: toward_vents
          emit: true

  knowledge_effects:
    ship_sentiment:
      "< -0.5": flee_immediately
      "-0.5 to 0": cautious, maintain_distance > 150
      "0 to 0.5": neutral, investigate_if_curious
      "> 0.5": friendly, approach_readily
```

### Interaction Rules Schema

```yaml
# interactions/rules.yaml

interactions:
  - species_a: drifter
    species_b: drifter
    proximity_trigger: 100  # meters
    actions:
      - type: gossip
        knowledge_tokens: [ship_sentiment, ship_capabilities, predator_location]
        merge_method: weighted_average
        capacity: 2  # per agent per turn

  - species_a: drifter
    species_b: shadow
    proximity_trigger: 80
    actions:
      - actor: drifter
        type: flee
        speed_multiplier: 2.0
        probability: 0.9

      - actor: shadow
        type: pursue
        probability: 0.6
        speed_multiplier: 1.5

      - actor: drifter
        type: inject_knowledge
        token:
          kind: predator_location
          value: {predator_id: shadow.id, location: shadow.location, threat: high}
          freshness: 1.0

  - species_a: player_ship
    species_b: drifter
    proximity_trigger: 300
    actions:
      - actor: drifter
        type: evaluate_knowledge
        condition: has(ship_sentiment)
        effects:
          sentiment < -0.5: flee
          sentiment > 0.5: approach
          else: investigate_cautiously
```

### Knowledge Tokens Schema

```yaml
# knowledge/tokens.yaml

token_types:
  - id: ship_sentiment
    value_type: float
    value_range: [-1.0, 1.0]  # -1 hostile, 0 neutral, 1 friendly
    decay_rate: 0.1  # per turn
    merge_method: weighted_average

  - id: ship_capabilities
    value_type: array
    possible_values: [lights, acoustic_comm, sonar_active, probes, harmful_action]
    decay_rate: 0.05  # slower decay (factual info)
    merge_method: union  # combine lists

  - id: predator_location
    value_type: object
    schema:
      predator_id: string
      location: {x, y, depth}
      last_seen_turn: int
      threat_level: enum[low, medium, high]
    decay_rate: 0.2  # fast decay (location changes)
    merge_method: most_recent

decay_function: linear  # MVP: freshness -= decay_rate per turn
# Post-MVP: exponential (freshness *= exp(-lambda * delta_t))

gossip_merge:
  weighted_average:
    formula: |
      weightA = reliabilityA * freshnessA
      weightB = reliabilityB * freshnessB
      value' = (weightA * valueA + weightB * valueB) / (weightA + weightB)
      freshness' = max(freshnessA, freshnessB) * 0.9
      reliability' = max(reliabilityA, reliabilityB) - 0.1
```

## Simulation Kernel

### Tick Function (Per Turn)

```python
def tick(delta_turns=1):
    """
    Advance simulation by delta_turns.

    For active regions:
    1. Update movement
    2. Detect encounters
    3. Apply interactions (gossip, predation, etc.)
    4. Decay knowledge
    5. Update behavior states

    For inactive regions:
    - Skip or update at lower fidelity (every N turns)
    """

    for turn in range(delta_turns):
        self.turn += 1

        for region in self.active_regions:
            # Movement
            for entity in region.entities:
                new_position = compute_movement(entity, region, self.turn)
                entity.position = new_position

            # Spatial indexing for O(n) queries
            spatial_grid = build_spatial_grid(region.entities)

            # Encounters
            encounters = find_encounters(spatial_grid, interaction_rules)

            # Apply interactions
            for encounter in encounters:
                apply_interaction(encounter, interaction_rules)

            # Knowledge decay
            for entity in region.entities:
                decay_knowledge(entity.knowledge)

            # Update behavior states
            for entity in region.entities:
                entity.behavior_state = evaluate_behavior_tree(entity)

        # Inactive region updates (optional, less frequent)
        if self.turn % 10 == 0:
            update_cohorts(self.inactive_regions)
```

### Query Interface

```python
class SimulatorQuery:

    def spatial_cone(self, origin, bearing, range, beam_width):
        """
        Query entities within cone (for sonar).

        Returns: [
            {
                entity_id,
                species_id,
                distance,
                bearing,
                size,
                velocity
            }
        ]
        """
        pass

    def acoustic_sources(self, origin, range):
        """
        Query acoustic emitters within range.

        Returns: [
            {
                entity_id,
                bearing,
                distance,
                frequency,
                pattern,
                strength
            }
        ]
        """
        pass

    def thermal_gradient(self, origin, bearing, range):
        """
        Sample thermal field in direction.

        Returns: {
            gradient_strength,
            gradient_direction,
            sources: [{type, location, intensity}]
        }
        """
        pass

    def entities_in_radius(self, origin, radius, species_filter=None):
        """
        Get all entities within radius, optionally filtered by species.
        """
        pass

    def region_knowledge_summary(self, region_id, token_kind):
        """
        Get aggregate knowledge for region (for AI context).

        Returns: {
            mean_value,
            variance,
            coverage: fraction of population with this token
        }
        """
        pass
```

### Injection Interface

```python
class SimulatorInjection:

    def inject_event(self, event):
        """
        Inject player action or environmental event.

        event: {
            type: 'sonar_ping' | 'acoustic_signal' | 'probe_deploy' | 'harm_entity',
            location,
            parameters,
            affected_radius
        }
        """
        pass

    def inject_knowledge(self, entity_id, token):
        """
        Directly inject knowledge token into entity.

        token: {
            kind,
            value,
            freshness,
            reliability,
            source
        }
        """
        pass

    def spawn_entity(self, species_id, location):
        """
        Spawn new entity at location.
        """
        pass

    def despawn_entity(self, entity_id):
        """
        Remove entity from simulation.
        """
        pass

    def set_player_position(self, location):
        """
        Update player ship position (for activation radius).
        """
        pass
```

## Determinism Specification

### Seed Derivation

```python
def derive_seed(global_seed, *components):
    """
    Deterministic seed derivation using hash.

    Example:
    entity_seed = derive_seed(world_seed, "region", region_id, "entity", entity_id)
    movement_seed = derive_seed(entity_seed, "movement", turn_number)
    """
    hash_input = f"{global_seed}:{':'.join(map(str, components))}"
    return int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % (2**32)
```

### RNG Usage

```python
# All randomness uses derived seeds
rng = np.random.default_rng(seed)

# Movement noise
noise = rng.normal(0, 0.1, size=2)

# Encounter probability
if rng.random() < encounter_probability:
    apply_encounter()
```

### Serialization

```json
{
  "turn": 157,
  "global_seed": 42,
  "regions": [
    {
      "id": "vent_field_alpha",
      "active": true,
      "entities": [
        {
          "id": "drifter_alpha_003",
          "species_id": "sp_001_drifter",
          "position": {"x": -105.2, "y": 462.1, "depth": -3102},
          "velocity": {"x": 0.2, "y": -0.1},
          "knowledge": [
            {
              "kind": "ship_sentiment",
              "value": 0.3,
              "freshness": 0.8,
              "reliability": 0.9,
              "source": "direct"
            }
          ],
          "behavior_state": "investigate_ship",
          "seed": 1729
        }
      ]
    }
  ]
}
```

## Activation and Fidelity Policy

### Active Region Definition

**Active if:**
- Player distance < activation_radius (default: 2000m for MVP)
- OR: Region contains mission-critical entities (marked for persistence)

**Activation radius:** 2000m (configurable)

### Cohort System (Post-MVP)

**Inactive regions store:**
```json
{
  "region_id": "vent_field_beta",
  "active": false,
  "last_update_turn": 98,
  "cohorts": {
    "sp_001_drifter": {
      "count": 42,
      "average_knowledge": {
        "ship_sentiment": {
          "mean": 0.0,
          "variance": 0.1,
          "coverage": 0.0
        }
      }
    }
  }
}
```

**On activation:**
- Spawn individuals based on cohort count
- Distribute knowledge tokens based on average + variance
- Resume full simulation

**On deactivation:**
- Aggregate entities into cohort statistics
- Despawn individuals (save only aggregates)
- Pause simulation or update every N turns

## Test Plan

### Test 1: Population Stability
**Objective:** Verify species populations remain stable over time

**Setup:**
- Spawn 50 Drifters in vent field
- Run 1000 turns
- No player interaction

**Success criteria:**
- 40-60 Drifters remain (±20% variance acceptable)
- Drifters stay within habitat bounds
- No crashes or infinite loops

### Test 2: Knowledge Propagation
**Objective:** Verify information spreads through population

**Setup:**
- 50 Drifters in region
- Inject `ship_sentiment = -0.8` into Drifter #1 at turn 0
- Run 200 turns, track knowledge spread

**Success criteria:**
- By turn 50: 25% of population has token (coverage >= 0.25)
- By turn 100: 50% of population has token
- By turn 200: 75% of population has token
- Token decays appropriately (freshness decreasing)

### Test 3: Predator-Prey Interaction
**Objective:** Verify cross-species interactions work

**Setup:**
- 40 Drifters + 3 Shadows in same region
- Run 500 turns

**Success criteria:**
- Drifters flee when Shadows approach
- Drifters generate `predator_location` tokens
- Drifters avoid areas with recent Shadow sightings
- No simulation deadlocks

### Test 4: Determinism Verification
**Objective:** Ensure same seed = same outcome

**Setup:**
- Run simulation with seed X for 100 turns
- Record final state (entity positions, knowledge, behavior states)
- Reset, run again with same seed

**Success criteria:**
- Final states are bit-for-bit identical
- No drift or randomness creep

### Test 5: Query Performance
**Objective:** Ensure queries are fast enough for real-time

**Setup:**
- 200 entities in region
- Run 100 spatial cone queries (sonar simulation)
- Measure time per query

**Success criteria:**
- Average query time < 10ms
- 95th percentile < 20ms
- No memory leaks over 1000 queries

### Test 6: Player Impact Propagation
**Objective:** Verify player actions affect ecosystem

**Setup:**
- Player kills 1 Drifter
- 5 witness Drifters within 300m
- Run 50 turns, track sentiment spread

**Success criteria:**
- Witness Drifters immediately get `ship_sentiment = -1.0`
- Sentiment propagates to nearby Drifters through gossip
- Future player encounters show changed behavior (flee instead of investigate)

## Sensor I/O Alignment

### Sonar Query → Simulation Response

**Unity sends:**
```json
{
  "sensor": "sonar_active",
  "parameters": {
    "origin": {"x": -105.0, "y": 460.0, "depth": -3100},
    "bearing": 235,
    "range": 500,
    "beam_width": 30,
    "power": 0.8
  }
}
```

**Simulator returns:**
```json
{
  "contacts": [
    {
      "bearing": 235,
      "distance": 180,
      "size": 3.1,
      "classification": "organic",
      "velocity": 0.3,
      "confidence": 0.85
    }
  ],
  "terrain": [
    {
      "bearing": 240,
      "distance": 220,
      "type": "ridge",
      "height": 80
    }
  ]
}
```

### AI Turn Contract Integration

**Each turn, Unity queries simulator and formats for AI:**

```json
{
  "turn": 47,
  "ship": {...},
  "observations": [
    {
      "sensor": "acoustic",
      "summary": "Structured clicks detected, 1.2kHz, bearing 235",
      "confidence": 0.73
    }
  ],
  "contacts": [
    {
      "id": "c-47",
      "type": "mobile_organism",
      "bearing": 235,
      "range": 180,
      "labels": ["unknown", "acoustic_emitter"]
    }
  ]
}
```

**AI proposes action:**
```json
{
  "proposed_actions": [
    {
      "tool": "toggle_lights",
      "args": {"level": "pulse"}
    }
  ]
}
```

**Unity injects into simulator:**
```python
sim.inject_event({
    'type': 'light_pulse',
    'location': player_position,
    'duration': 1,
    'intensity': 0.7
})
```

**Simulator updates affected entities:**
- Nearby Drifters detect light event
- Behavior tree evaluates → respond_communication trigger
- Drifter emits response pattern
- Next turn, Unity queries acoustic sensor → detects response

## Milestones and Deliverables

### M0: Schema Definition (This Session)
- [ ] World schema (YAML format defined)
- [ ] Species schema (YAML format defined)
- [ ] Interaction rules schema (YAML format defined)
- [ ] Knowledge tokens schema (YAML format defined)
- [ ] This design document completed

### M1: Vent Field Alpha Data Pack (Next Session)
- [ ] `world/titan.yaml` (complete)
- [ ] `biomes/vent_field_alpha.yaml` (complete)
- [ ] `species/sp_001_drifter.yaml` (complete)
- [ ] `species/sp_002_tube_forest.yaml` (stub)
- [ ] `species/sp_003_shadow.yaml` (stub)
- [ ] `interactions/rules.yaml` (Drifter-Drifter, Drifter-Shadow, Drifter-Ship)
- [ ] Compilation script: `compile_data_packs.py` (YAML → JSON)

### M2: Simulator Kernel Prototype (Python)
- [ ] Core simulation loop (tick function)
- [ ] Spatial indexing (grid or k-d tree)
- [ ] Movement system (steering behaviors)
- [ ] Encounter detection (proximity checks)
- [ ] Knowledge propagation (gossip implementation)
- [ ] Knowledge decay (per-turn freshness reduction)
- [ ] Query interface (spatial_cone, acoustic_sources, thermal_gradient)
- [ ] Injection interface (inject_event, inject_knowledge)

### M3: Determinism and Testing
- [ ] Seed derivation system
- [ ] Serialization/deserialization
- [ ] Test suite (6 tests defined above)
- [ ] Performance benchmarks
- [ ] Validation: 1000-turn stability run

### M4: Unity Integration (Post-Simulator Validation)
- [ ] JSON API wrapper (Python → Unity communication)
- [ ] Sensor query implementations
- [ ] Player action injection
- [ ] Turn synchronization
- [ ] Save/load with simulator state

## Technology Stack (Research-Validated)

### Core Technology Decisions

**Based on extensive research into:**
- Spatial indexing libraries (scipy, octrees, spatial hashing)
- Agent-based modeling frameworks (Mesa, Agents.jl, NetLogo)
- Dwarf Fortress architecture patterns
- Game sensor simulation techniques

### Simulator Core

**Language:** Python 3.10+

**Why Python:**
- Fast prototyping for ecosystem validation
- Excellent scientific computing libraries (NumPy, SciPy)
- Can run standalone (no Unity dependency for testing)
- JSON API for Unity integration
- Performance adequate for 100-500 entities

**Core Dependencies:**
- **NumPy:** Spatial operations, vectorized math, RNG
- **SciPy:** `spatial.cKDTree` for O(log N) proximity queries
- **PyYAML:** Human-readable data authoring
- **JSON:** Runtime format (deterministic, Unity-compatible)

**Optional (Post-MVP):**
- **py_trees:** Behavior trees for complex agents (Shadow predator)

### Spatial Indexing: scipy.spatial.cKDTree

**Research findings:**
- Evaluated: k-d trees, octrees, spatial hashing, R-trees, BVH
- **Winner:** scipy.spatial.cKDTree for our use case

**Performance (verified benchmarks):**
- 500 entities, tree construction: 2-3ms
- Single query (K-nearest or radius): ~0.15ms
- Batch queries with `workers=-1`: 4-8x speedup (parallel)
- Rebuild every turn acceptable for turn-based game

**Why not alternatives:**
- Octrees: 5-15x faster for radius queries BUT harder to implement correctly
- Spatial hashing: O(1) queries BUT requires careful cell size tuning
- Mesa ContinuousSpace: Only supports 2D (we need 3D)
- Custom implementation: Reinventing the wheel (scipy is battle-tested)

**Implementation pattern:**
```python
from scipy.spatial import cKDTree
import numpy as np

# Build tree from entity positions
positions = np.array([entity.position for entity in entities])
tree = cKDTree(positions, leafsize=16)

# Radius query (all within communication range)
indices = tree.query_ball_point([x, y, z], r=100.0)

# Batch queries (all entities find neighbors in parallel)
all_neighbors = tree.query_ball_point(positions, r=100.0, workers=-1)
```

**Critical gotchas avoided:**
- cKDTree has O(n²) construction on structured grid data → shuffle input if grid-aligned
- `query_ball_point` memory explosion with large radius → use `return_length=True` for counts
- Rebuild frequency matters → amortize cost by rebuilding every 5-10 frames (or every turn for turn-based)

### Framework Decision: No ABM Framework

**Research findings:**
- Evaluated: Mesa, Agents.jl, GAMA, NetLogo, AgentPy
- Mesa is mature, deterministic, well-documented
- **BUT:** 9-14x slower than alternatives, experimental 3D support, built-in spatial queries are O(n)

**Decision:** Adopt Mesa patterns, implement custom kernel

**What we adopt from Mesa:**
- RandomActivation (shuffle agent update order per turn for fairness)
- Deterministic seeding (`random.seed()`, `np.random.seed()`)
- Agent-Model separation (entities don't own simulation logic)

**What we implement ourselves:**
- Lightweight entity classes (no framework overhead)
- Direct scipy.cKDTree integration (not Mesa's O(n) neighbor search)
- Custom 3D spatial logic
- No ECS (Entity-Component-System) - simple OOP classes

**Why no ECS:**
- Research finding: Tarn Adams (Dwarf Fortress) called polymorphic ECS "ultimate mistake"
- ECS excellent when entity composition changes frequently
- Our entities are static: Drifter is always a Drifter, no component add/remove
- Simple OOP (class Drifter with attributes) is clearer and faster for our case

### Behavior Systems

**Simple agents (Drifter, Tube Forest):**
- Priority-based decision lists (if-elif-else)
- No framework needed
- Example:
```python
def evaluate_behavior(drifter, context):
    if drifter.has_predator_nearby(80):
        return "flee"
    elif drifter.has_ship_sentiment() and drifter.sentiment < -0.5:
        return "flee_ship"
    elif drifter.ship_distance < 300:
        return "investigate_ship"
    else:
        return "forage"
```

**Complex agents (Shadow predator - Post-MVP):**
- py_trees behavior tree framework
- Mature, robotics-proven
- Useful for: patrol → detect → stalk → attack sequences

### Performance Targets (Validated)

**For 100-500 entities per turn:**
- Movement updates: < 5ms
- cKDTree rebuild: < 3ms
- Neighbor queries (batch): < 2ms
- Knowledge propagation: < 2ms
- Behavior evaluation: < 3ms
- **Total: ~15ms per turn**

**Interpretation:**
- 66 turns/second possible (if real-time needed)
- Turn-based game: instant feel
- 1000-turn stability test runs in ~15 seconds

**Performance scaling (if needed post-MVP):**
- Event-driven scheduling: Entities sleep until next action (Dwarf Fortress pattern)
- Reduces 500 updates/turn → 50 updates/turn (10x improvement)
- Tube Forests wake only when: ship approaches, predator nearby
- Not needed for MVP but documented for future

### Data Pipeline
- **Authoring:** YAML (human-editable)
- **Runtime:** JSON (canonical, deterministic)
- **Build step:** Python script converts YAML → JSON

### API Layer (Future)
- **Interface:** JSON over HTTP or WebSocket
- **Unity client:** C# JSON deserializer
- **Testing:** cURL or Python requests

### Sensor Simulation Integration

**Research findings on Unity sensor systems:**

**Division of responsibility (two-phase approach):**
1. **Python Aquarium:** Provides spatial query results (entities in range/cone)
2. **Unity:** Handles raycasting (occlusion), visualization, audio spatialization

**Why this split:**
- Aquarium knows entity positions/emissions (authoritative)
- Unity knows terrain geometry/occlusion (rendering engine)
- Aquarium queries are cheap (cKDTree lookup: ~0.15ms)
- Unity raycasting is expensive (needs optimization)

**Sonar simulation pattern:**
```python
# Aquarium provides potential contacts
def query_spatial_cone(origin, bearing, range, beam_width):
    # Fast spatial query: entities in cone
    candidates = kdtree.query_ball_point(origin, r=range)
    # Filter by bearing
    return [entity for entity in candidates if in_cone(entity, bearing, beam_width)]

# Unity then:
# 1. Takes candidate list from aquarium
# 2. Raycasts for occlusion (terrain blocking)
# 3. Applies sensor noise/uncertainty
# 4. Renders sonar display (GPU)
```

**Performance targets (Unity side):**
- Sonar per submarine: 3-5ms (RaycastCommand batch, 1024 rays)
- Thermal sampling: 1ms (GPU 3D texture lookup)
- Acoustic propagation: 2ms (ray occlusion + simplified physics)
- **Total sensor budget:** 7-9ms (leaves headroom for rendering)

**Key techniques from research:**
- Unity RaycastCommand + Job System: 10x performance vs. standard raycasts
- Two-phase approach: cheap OverlapSphere → selective raycasting
- Layer masks mandatory (biggest performance mistake: testing all colliders)
- NativeArray for zero-GC sensor data
- GPU Compute Shaders for thermal field visualization

**Acoustic propagation:**
- Research-grade tools (KRAKEN, Bellhop3D) too slow for real-time
- Practical approach: Ray-based occlusion + material absorption
- Unity Audio spatialization (good-enough accuracy for gameplay)

**Thermal fields:**
- 3D RenderTexture + Compute Shader (GPU-accelerated)
- 64³ voxels updateable at 60 FPS
- Aquarium provides heat source locations/intensities
- Unity renders thermal gradient visualization

## Ecosystem Vision

### Food Web Architecture
The ecosystem simulates a complete trophic cascade - the foundation of a living moon:

**Base Layer: Plankton/Bacteria Analogs**
- Seek optimal temperature bands (thermal vents, warm currents)
- Move with flow dynamics and thermal gradients
- Base of food chain (primary producers)
- Population: 5,000-10,000 entities in active regions

**Middle Layer: Herbivores**
- Hunt plankton concentrations using spatial queries
- Use knowledge gossip to find feeding grounds ("good hunting here")
- Create emergent migration patterns toward food sources
- Examples: Filter-feeders, grazers, small schooling organisms
- Population: 1,000-2,000 entities in active regions

**Top Layer: Predators**
- Hunt herbivore clusters using nearest-entity behaviors
- Territorial behavior driven by energy levels
- Energy-driven aggression (desperate = more aggressive)
- Examples: Shadows, apex hunters, ambush predators
- Population: 200-500 entities in active regions

### Equilibrium Goals
The aquarium must reach stable equilibrium before ship integration:

- **Population stability**: No extinction cascades, sustainable predator/prey ratios
- **Energy flow balanced**: Consumption matches production across trophic levels
- **Resource distribution**: Food scarcity drives migration and territorial behavior
- **Tunable parameters**: Adjust spawn rates, consumption rates, energy transfer efficiency
- **Emergent dynamics**: Clusters form around food sources, migrations follow resources

**Core Principle:** The aquarium must feel alive and self-sustaining. When ship arrives, it enters a functioning ecosystem, not a scripted environment.

### Extended Phase Roadmap

**Phase 1-6: Foundation** (Current Work)
- Phase 1-2: Data loading, spawning, movement (Complete)
- Phase 3: Behavior evaluation, ship entity (Complete)
- Phase 4: Spatial indexing with cKDTree (In Progress)
- Phase 5: Ship interaction behaviors (Planned)
- Phase 6: Knowledge gossip system (Planned)

**Phase 7: Energy System**
- Add `energy: float` to entity state (current level, 0.0 to max)
- Add `max_energy: float` to species definition
- Add `energy_consumption_rate: float` (per-tick baseline cost)
- New behavior conditions: `energy_below`, `energy_above`
- New action types: `hunt` (seek food entity), `consume` (gain energy from target)
- Starvation mechanics: Energy reaches 0 → entity becomes "desperate"
- **No death yet** - energy floors at 0, behavior changes to frantic hunting

**Phase 8: Population Dynamics**
- Death from starvation (energy at 0 for N ticks)
- Death from predation (consumed by predator)
- Spawning tied to energy surplus (entity with >80% energy can spawn)
- Population equilibrium tuning (adjust spawn/death rates)
- Extinction prevention (minimum population thresholds trigger spawning)
- **Goal**: Stable populations without manual intervention

**Phase 9: Environmental Gradients**
- Temperature fields in biomes (thermal plumes from vents)
- Nutrient concentrations (tied to thermal activity)
- Plankton growth rates tied to temperature + nutrients
- Thermal plume dynamics (flow, diffusion, cooling)
- Optimal zones for each trophic level
- **Creates dynamic resource distribution**

**Phase 10: Ship as Observer**
- Ship entity becomes active (sensors, movement, presence)
- Creatures react to ship based on existing behaviors
- Ship has sensor queries (sonar, thermal, chemical, acoustic)
- Player observes ecosystem through limited perception
- **Ship enters a living world**

**Phase 11: Moon Generation**
- Procedurally generate Titan ocean at runtime (seed-driven)
- Multiple biomes with different conditions
- Scale aquarium patterns to moon (10x, 100x regions)
- Persistent world state (save/load entire ocean)
- Ship explores vast space

**Phase 12: Rendering Layer**
- Unity visualization of sensor data (not direct observation)
- Sonar displays, thermal gradients, acoustic waterfalls
- Camera feeds (limited range, water clarity)
- UI for player control
- **Player sees what sensors detect**

**Phase 13: Narrative Injection**
- Ancient civilization lore written
- Relic entities designed (special tags, behaviors)
- Ruins as spatial obstacles with story data
- Mystery progression system
- Narrative events triggered by exploration
- **Story woven into living ecosystem**

**Phase 14: AI Reflection System**
- ShipMind reviews all handoffs at endgame
- Makes sovereign choice based on collaboration quality
- Ending branches on player-AI relationship
- Consequences persist to next game
- **The AI becomes part of the story**

## Open Questions

### For Next Session

1. **Cohort granularity:** Per-region sufficient for MVP, or need tiled subdivisions?
   - **Recommendation:** Per-region for MVP

2. **Reproduction/mortality:** Include in MVP or defer?
   - **Recommendation:** Defer to Phase 8 (prove food web first)

3. **Cross-species knowledge:** Can Shadows learn about ships from Drifters?
   - **Recommendation:** Phase 6+ feature (same-species gossip first)

4. **Regional knowledge exposure to AI:** Should AI see aggregate stats ("40% of Drifters hostile") or only infer from encounters?
   - **Recommendation:** Inference only (more emergent, better narrative)

5. **Activation/deactivation hysteresis:** How to prevent regions rapidly toggling active/inactive?
   - **Recommendation:** Activation radius + deactivation delay (stay active for N turns after player leaves)

## Research Foundation

This design is based on comprehensive external research into production systems:

**Spatial Indexing Research:**
- Evaluated scipy.cKDTree, octrees, spatial hashing, R-trees, BVH structures
- Analyzed performance benchmarks from molecular dynamics, boids simulations, point clouds
- Studied real-world implementations handling 100-10,000+ entities
- **Key finding:** scipy.cKDTree optimal for 100-500 entities with O(log N) queries

**Agent-Based Modeling Research:**
- Evaluated Mesa, Agents.jl, GAMA, NetLogo, AgentPy frameworks
- Studied deterministic simulation techniques from academic ABM literature
- Analyzed behavior tree implementations (py_trees, Behavior Designer)
- **Key finding:** Adopt Mesa patterns without framework overhead

**Architecture Research:**
- Studied Dwarf Fortress development talks and postmortems
- Analyzed ECS patterns (Flecs, EnTT, Unity DOTS)
- Researched event-driven scheduling for large entity counts
- Examined roguelike codebases (DCSS, Cataclysm DDA) for data-driven design
- **Key findings:** Simple OOP beats ECS for static entities; event-driven scheduling scales to 30,000+

**Sensor Simulation Research:**
- Studied submarine game sonar implementations (URSim, RomeldaB/Sonar-Simulation)
- Analyzed raycasting optimizations (Unity Job System, RaycastCommand)
- Researched acoustic propagation models (KRAKEN, Bellhop3D, simplified approaches)
- Examined thermal field simulations (GPU Compute Shaders, voxel grids)
- **Key findings:** Two-phase approach (spatial query + raycasting); GPU for visualization

**Critical Lessons Learned:**
1. **Don't reinvent the wheel:** scipy.cKDTree is battle-tested (use it)
2. **Determinism is fragile:** Focus on generation + save/load, not cross-platform replay
3. **Frameworks have costs:** Mesa patterns good, Mesa framework unnecessary for our case
4. **ECS is not always better:** Simple OOP faster when entity composition is static
5. **Performance comes from scheduling:** Smart updates > perfect data structures
6. **Cohorts can wait:** Prove core sim first, optimize later

**Research Documents:**
- `project/research/spatial_indexing_external_research_2025-09-30.md`
- `project/research/agent_based_modeling_frameworks_external_research_2025-09-30.md`
- `project/research/dwarf_fortress_simulation_external_research_2025-09-30.md`
- `project/research/sensor_simulation_external_research_2025-09-30.md`

## Summary

This specification defines the aquarium simulation as the authoritative world model for Shipmind. It is:
- **Deterministic:** Same seed = same world generation; save/load = exact restoration
- **Independent:** No Unity or AI dependencies (standalone Python module)
- **Queryable:** Sensors ask, simulator answers (query-first interface)
- **Reactive:** Player actions affect ecosystem (knowledge propagation, behavior changes)
- **Testable:** Can run 1000 turns in ~15 seconds, validate stability
- **Research-validated:** Built on battle-tested libraries and production patterns

**Technology choices validated:**
- scipy.cKDTree for spatial indexing (proven performance)
- Custom lightweight kernel (Mesa patterns, no framework)
- Simple OOP classes (no ECS complexity)
- Python for simulation, Unity for visualization
- Two-phase sensor approach (spatial queries + raycasting)

**Next steps:**
1. Create Vent Field Alpha data pack (complete YAML files)
2. Implement simulator kernel (Python prototype)
3. Run test suite (verify stability, propagation, determinism)
4. Only then: Integrate with Unity

The aquarium must work perfectly in isolation before becoming part of the game.
