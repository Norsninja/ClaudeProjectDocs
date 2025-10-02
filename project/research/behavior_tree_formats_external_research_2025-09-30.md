# Behavior Tree YAML/JSON Formats - External Research

**Research Date:** 2025-09-30
**Researcher:** Technical Research Scout
**Focus:** Data-driven behavior tree formats for aquarium simulation entities

---

## Executive Summary

For simple priority-based agent AI with 100-500 entities and <3ms budget, **avoid traditional behavior tree libraries**. Research reveals three critical findings:

1. **Traditional BT libraries (py_trees, BehaviorTree.CPP) are over-engineered** for your use case - designed for complex robotics with hundreds of behaviors per agent, not hundreds of agents
2. **Simple priority-based selectors outperform full BT implementations** - Minecraft's JSON format and Terasology's lightweight approach are better models
3. **C++ BT overhead is 1.4-15ns per tick, but Python adds 1000x overhead** - py_trees explicitly states it's "not scoped to enable hundreds of characters"

**Recommendation:** Use a lightweight priority list format (like Minecraft Bedrock entities) rather than full behavior trees. Your "if predator → flee, elif hostile_ship → flee, elif ship → investigate, else → forage" is a **Selector node** - don't build more than you need.

---

## Implementation Patterns

### 1. Terasology Game Engine (Lightweight BT in JSON)

**Source:** https://github.com/Terasology/Behaviors (Production game with creatures)

**Format:** Simple JSON with minimal nesting
```json
{
  "sequence": [
    {"timer": {"time": 5}},
    {"log": {"message": "Hello!"}}
  ]
}
```

**Key Patterns:**
- Nodes are objects with single key (node type) and value (parameters or children)
- Quotes optional but canonical JSON preferred
- Composite nodes use arrays: `"sequence": [child1, child2]`
- Decorators take single child: `{"decorator": {"child": {...}}}`
- Actions are leaf nodes: `{"action_name": {"param": "value"}}`

**Example Complex Tree:**
```json
{
  "sequence": [
    {
      "animation": {
        "play": "engine:Stand.animationPool",
        "loop": "engine:Stand.animationPool"
      }
    },
    {
      "sleep": {"time": 3}
    }
  ]
}
```

**Why It Works:**
- Dead simple to parse (Python `json.load()`)
- Easy to author and read
- No external dependencies
- Performant for entity-based games

---

### 2. Behavior3 Editor Format (Visual Tool + Runtime)

**Source:** https://github.com/behavior3/behavior3editor (Multi-language: JS/Python/TypeScript)

**Format:** More verbose, designed for visual editing tools

**Tree Structure:**
```json
{
  "title": "TreeName",
  "description": "What this tree does",
  "root": "node_id_1",
  "properties": {},
  "nodes": {
    "node_id_1": {
      "id": "node_id_1",
      "name": "Sequence",
      "title": "Main Sequence",
      "description": "",
      "children": ["node_id_2", "node_id_3"],
      "parameters": {},
      "properties": {}
    },
    "node_id_2": {
      "id": "node_id_2",
      "name": "CheckDistance",
      "title": "Check Predator Distance",
      "parameters": {
        "threshold": 80.0,
        "target_type": "predator"
      }
    }
  }
}
```

**Pros:**
- Visual editor available (https://behavior3.github.io/behavior3editor/)
- Cross-platform (behavior3py library exists)
- Formal structure with IDs for debugging

**Cons:**
- Verbose - lots of IDs and metadata
- Designed for complex trees (overkill for simple priority lists)
- No formal JSON schema (requested but never created)

**Production Use:** Designed for "hundreds of agents" - explicitly optimized for multi-agent systems

---

### 3. BehaviorTree.CPP XML Format (Robotics Standard)

**Source:** https://www.behaviortree.dev (C++ library, industry standard for robotics/ROS)

**Format:** XML with simple schema
```xml
<root BTCPP_format="4">
  <BehaviorTree ID="DrifterBehavior">
    <Sequence name="main">
      <Action ID="SaySomething" message="Hello"/>
      <Action ID="OpenGripper"/>
      <SubTree ID="GraspObject"/>
    </Sequence>
  </BehaviorTree>
</root>
```

**Compact Syntax:**
```xml
<Selector name="priority_check">
  <CheckPredatorDistance threshold="80.0"/>
  <Sequence>
    <CheckShipSentiment min="-0.5"/>
    <FleeFromShip/>
  </Sequence>
  <InvestigateShip max_distance="300.0"/>
  <Forage/>
</Selector>
```

**Blackboard (State Management):**
```xml
<Action message="{blackboard_key}"/>
```
Use `{key}` to reference shared state (like knowledge tokens)

**Pros:**
- Industry standard (Halo, Unreal Engine inspired)
- Excellent tooling (Groot visual editor)
- Performance: 1.4ns overhead (C++) vs 15ns (OOP style)

**Cons:**
- XML (verbose, not human-friendly for authoring)
- C++ library (not Python)
- Over-engineered for simple priority lists

**Performance Data (C++):**
- Tuple-based: ~1.4-1.7ns per tick
- OOP inheritance: ~9-15ns per tick
- Function pointers: ~11-14ns per tick
(Benchmarked on Ubuntu 18.04, GCC 8)

---

### 4. Minecraft Bedrock Entity AI (Production Game Format)

**Source:** https://learn.microsoft.com/en-us/minecraft/creator/reference/content/entityreference/examples/aigoallist

**Format:** Priority-based JSON goals (exactly what you need!)

**Structure:**
```json
{
  "minecraft:behavior.avoid_mob_type": {
    "priority": 1,
    "max_dist": 16,
    "entity_types": [
      {"filters": {"test": "is_family", "value": "zombie"}}
    ],
    "speed_multiplier": 1.2
  },
  "minecraft:behavior.panic": {
    "priority": 2,
    "speed_multiplier": 1.5
  },
  "minecraft:behavior.random_stroll": {
    "priority": 6,
    "speed_multiplier": 1.0
  }
}
```

**Behavior Goals Relevant to Drifters:**
- `minecraft:behavior.avoid_mob_type` - Flee from specific entities (predators, ships)
- `minecraft:behavior.investigate_suspicious_location` - Investigate positions
- `minecraft:behavior.move_towards_target` - Move toward target
- `minecraft:behavior.random_stroll` - Forage/wander
- `minecraft:behavior.panic` - Emergency flee

**Priority System:**
- Lower number = higher priority
- Behaviors checked in priority order
- First valid behavior wins (Selector pattern!)

**Distance/Condition Patterns:**
```json
{
  "filters": {
    "test": "distance_to_nearest_player",
    "operator": "<",
    "value": 150
  }
}
```

**Why This Fits Your Use Case:**
- Simple priority list (exactly your "if/elif/else" logic)
- Distance checks built-in
- Filter system for conditions
- Production-proven for 100+ entities per world
- Extremely readable and authorable

---

### 5. Unity ML-Agents (YAML Configuration)

**Source:** Unity ML-Agents toolkit

**Format:** YAML for hyperparameters, not behavior trees
```yaml
behaviors:
  DrifterBrain:
    trainer_type: ppo
    max_steps: 5.0e5
    batch_size: 128
    buffer_size: 2048
```

**Not Relevant:** ML-Agents uses neural networks, not behavior trees. YAML is for training config, not behavior specification.

---

## Battle-Tested Patterns

### Pattern 1: Priority-Based Selector (Universal)

All successful implementations use this core pattern:

```
Selector (try in order, stop at first success):
  1. Condition: Predator within 80m → Action: Flee
  2. Condition: Hostile ship within 150m → Action: Flee
  3. Condition: Ship within 300m → Action: Investigate
  4. Default → Action: Forage
```

**In Terasology JSON:**
```json
{
  "selector": [
    {
      "sequence": [
        {"check_distance": {"entity_type": "predator", "max": 80}},
        {"flee": {"speed_multiplier": 1.5}}
      ]
    },
    {
      "sequence": [
        {"check_distance": {"entity_type": "ship", "max": 150}},
        {"check_sentiment": {"key": "ship_sentiment", "max": -0.5}},
        {"flee_from_ship": {}}
      ]
    },
    {
      "sequence": [
        {"check_distance": {"entity_type": "ship", "max": 300}},
        {"investigate": {}}
      ]
    },
    {"forage": {}}
  ]
}
```

**In Minecraft-style JSON (Recommended):**
```json
{
  "behaviors": {
    "drifter:flee_predator": {
      "priority": 1,
      "max_distance": 80,
      "entity_types": [{"filters": {"test": "is_family", "value": "predator"}}],
      "speed_multiplier": 1.5
    },
    "drifter:flee_hostile_ship": {
      "priority": 2,
      "conditions": [
        {"test": "distance_to_nearest", "subject": "other", "value": 150, "operator": "<"},
        {"test": "knowledge_token", "key": "ship_sentiment", "value": -0.5, "operator": "<"}
      ],
      "action": "flee"
    },
    "drifter:investigate_ship": {
      "priority": 3,
      "max_distance": 300,
      "action": "investigate"
    },
    "drifter:forage": {
      "priority": 10,
      "action": "forage"
    }
  }
}
```

**Why Priority Lists Win:**
- O(n) where n = number of behaviors (~4-10 for most species)
- No tree traversal overhead
- Cache-friendly (linear array scan)
- Easy to debug (print which behavior won)

---

### Pattern 2: Condition Encoding (From Multiple Sources)

**Spatial Conditions:**
```yaml
# Distance to entity
condition:
  type: distance_check
  target: predator
  operator: "<"
  threshold: 80.0

# Distance to position
condition:
  type: distance_to_point
  position: [x, y, z]
  threshold: 50.0

# Angle/visibility check
condition:
  type: in_view_cone
  target: ship
  angle: 120  # degrees
  max_distance: 300
```

**State Conditions (Knowledge Tokens):**
```yaml
condition:
  type: knowledge_check
  key: ship_sentiment
  operator: "<"
  value: -0.5

# Or compound:
conditions:
  - {type: knowledge_check, key: ship_sentiment, operator: ">", value: 0.5}
  - {type: distance_check, target: ship, operator: "<", threshold: 300}
```

**Entity Type/Tag Filters (Minecraft pattern):**
```json
{
  "filters": {
    "all_of": [
      {"test": "is_family", "value": "ship"},
      {"test": "has_component", "value": "dda:hostile"}
    ]
  }
}
```

---

### Pattern 3: Action Encoding

**Abstract Actions (recommended):**
```yaml
action:
  type: flee
  speed_multiplier: 1.5
  duration: 5.0  # seconds

action:
  type: investigate
  target: entity  # or "position"
  approach_distance: 50.0

action:
  type: forage
  wander_radius: 100.0
  speed_multiplier: 0.8
```

**Why Abstract?** Kernel handles motion/pathfinding. Data specifies **what** to do, kernel figures out **how**.

**Parameter References:**
```yaml
action:
  type: flee
  speed: "{species_config.flee_speed}"  # reference from species config
  distance: "{behaviors.flee_distance}"
```

---

### Pattern 4: Species Configuration (Recommended Structure)

```yaml
species: drifter
version: 1.0

# Global parameters (referenced by behaviors)
parameters:
  flee_speed: 15.0  # m/s
  investigate_speed: 8.0
  forage_speed: 5.0
  predator_detection_range: 80.0
  ship_detection_range: 300.0

# Priority-based behavior list
behaviors:
  - id: flee_predator
    priority: 1
    conditions:
      - type: distance_check
        target_tag: predator
        operator: "<"
        threshold: "{parameters.predator_detection_range}"
    action:
      type: flee
      speed: "{parameters.flee_speed}"

  - id: flee_hostile_ship
    priority: 2
    conditions:
      - type: distance_check
        target_tag: ship
        operator: "<"
        threshold: 150.0
      - type: knowledge_check
        key: ship_sentiment
        operator: "<"
        value: -0.5
    action:
      type: flee
      speed: "{parameters.flee_speed}"

  - id: investigate_ship
    priority: 3
    conditions:
      - type: distance_check
        target_tag: ship
        operator: "<"
        threshold: "{parameters.ship_detection_range}"
    action:
      type: investigate
      speed: "{parameters.investigate_speed}"
      approach_distance: 50.0

  - id: forage
    priority: 10  # default/lowest priority
    conditions: []  # always valid
    action:
      type: forage
      speed: "{parameters.forage_speed}"
      wander_radius: 100.0
```

---

## Critical Gotchas

### 1. Don't Over-Engineer (Major Anti-Pattern)

**From "Overcoming Pitfalls in Behavior Tree Design" (Game AI Pro 3):**

**Pitfall #1: Needlessly Multiplying Core Primitives**
- Don't add 10 types of Selector nodes (priority, random, weighted, utility, etc.)
- Start with Selector, Sequence, Action, Condition - add more only when proven necessary
- Halo 2 shipped with ~5 node types

**Pitfall #2: Inventing a Whole Programming Language**
- Don't add loops, variables, functions to your BT format
- If you need that complexity, use Python, not YAML gymnastics
- Keep data as **data**, logic as **code**

**Pitfall #3: Forcing All Communication Through Blackboard**
- Don't route everything through global state
- Direct entity references are fine (entity.position, entity.velocity)
- Blackboard for shared decision data, not everything

**For Your Use Case:** You have 4 behaviors per species. Don't build a framework for 400.

---

### 2. Performance Cliffs (Python-Specific)

**py_trees Documentation Warning:**
> "Robotic scenarios for a single robot tend to be, maximally in the order of hundreds of behaviours... [py_trees is] not scoped to enable an NPC gaming engine with hundreds of characters and thousands of behaviours for each character."

**Reality Check:**
- C++ BT overhead: 1-15 nanoseconds per tick
- Python overhead: ~1000x (microseconds, not nanoseconds)
- **Your budget: 3ms / 500 entities = 6 microseconds per entity**

**Math:**
- 500 entities × 4 behaviors = 2000 condition checks per tick
- Python function call overhead: ~0.3µs each = 600µs (20% of budget)
- Full py_trees: ~5-10µs per tree tick = 2500-5000µs (83-167% of budget) ❌

**Solution:** Lightweight priority list, not full BT library. Simple Python loop:
```python
for behavior in sorted_behaviors:  # already sorted by priority
    if behavior.check_conditions(entity, world):
        behavior.execute(entity, world)
        break  # first success wins (Selector pattern)
```

---

### 3. Event-Driven vs Polling Trade-offs

**Unreal Engine Approach (Event-Driven):**
- BTs passively listen for events
- Only re-evaluate when state changes
- **Pro:** Scales to 1000+ entities
- **Con:** Complex architecture, requires event system

**Simple Polling Approach:**
- Tick every entity every frame
- Check conditions each tick
- **Pro:** Dead simple, deterministic
- **Con:** Doesn't scale past ~500 entities

**Hybrid (Recommended for MVP):**
- Tick behaviors every frame (simple)
- Cache results that don't change often (entity positions, etc.)
- Optimize after profiling shows it's slow

---

### 4. YAML vs JSON vs XML

**YAML:**
- ✅ Human-readable, easy to author
- ✅ Comments supported
- ✅ Less verbose (no quotes/brackets)
- ❌ Slower to parse (PyYAML is not fast)
- ❌ Whitespace-sensitive (error-prone)

**JSON:**
- ✅ Fast to parse (`json.load()` is C-optimized)
- ✅ Supported everywhere
- ✅ Strict syntax (catches errors)
- ❌ No comments
- ❌ More verbose

**XML:**
- ✅ Industry standard (BehaviorTree.CPP, Unreal)
- ✅ Schema validation (XSD)
- ❌ Very verbose
- ❌ Hard to author by hand

**Recommendation:** **YAML for authoring, JSON for runtime**
- Write species configs in YAML (comments, readability)
- Convert to JSON at load time (`yaml.safe_load()` → `json.dump()`)
- Ship JSON with game (faster parsing)

---

### 5. Condition Evaluation Optimization

**Anti-Pattern (Terasology mistake):**
```python
# Recalculate expensive checks every frame
def check_conditions(entity):
    nearest_predator = world.find_nearest(entity, "predator")  # O(n) search!
    if nearest_predator and distance(entity, nearest_predator) < 80:
        return True
```

**Better Pattern:**
```python
# Cache spatial queries
class SpatialCache:
    def __init__(self):
        self.cache = {}  # entity_id -> nearest_threat
        self.ttl = 0.1  # 100ms cache

    def get_nearest_threat(self, entity, current_time):
        if entity.id not in self.cache or current_time - self.cache[entity.id].time > self.ttl:
            self.cache[entity.id] = world.spatial_query(entity.position, "threat")
        return self.cache[entity.id].result
```

**Minecraft Approach:**
- Update entity AI every 2-4 ticks (not every tick)
- Spatial queries use chunk-based grid (O(1) lookups)
- AI "thinks" at 10Hz, moves at 60Hz

---

### 6. Interruption Handling

**Problem:** What if Drifter is investigating ship, then predator appears?

**Bad Pattern:**
```python
# Finish current action before checking new priorities
current_action.execute()  # keeps investigating even though predator is near!
```

**Good Pattern (Selector Natural Interruption):**
```python
# Re-evaluate priorities every tick
for behavior in sorted_behaviors:
    if behavior.check_conditions():
        if current_behavior != behavior:
            current_behavior.interrupt()  # clean up old action
            current_behavior = behavior
        behavior.execute()
        break
```

**From rubenwardy's blog:**
> "Decorators can interrupt work based on dynamic conditions like health or energy"

**Implementation:**
```yaml
behaviors:
  - id: flee_predator
    priority: 1
    interrupt: true  # can interrupt lower-priority behaviors
    conditions: [...]

  - id: investigate_ship
    priority: 3
    interruptible: true  # can be interrupted by higher priority
    conditions: [...]
```

---

## Trade-off Analysis

### Option A: Full Behavior Tree Library (py_trees, behavior3py)

**Pros:**
- Mature, tested code
- Visual editors available
- Structured node types

**Cons:**
- Performance: Not scoped for 100+ agents
- Over-engineered for simple priority lists
- Learning curve for library API
- Dependency management

**Use When:**
- Complex per-agent logic (20+ behaviors)
- Need visual debugging tools
- Single agent or <10 agents

**Don't Use When:**
- Simple priority lists (your case)
- 100+ agents
- <3ms performance budget

---

### Option B: Lightweight Priority List (Custom)

**Pros:**
- Exact fit for your use case
- Performance: O(n) where n = behaviors per species (~4-10)
- No dependencies
- Easy to debug (print winning behavior)

**Cons:**
- No visual editor
- Must implement yourself (~200 LOC)
- No advanced features (utility AI, planners, etc.)

**Use When:**
- Simple priority-based decisions
- Many agents (100+)
- Tight performance budget
- Data-driven species configs

**Implementation Effort:** ~4 hours
- YAML parser: 30 min
- Condition evaluators: 90 min
- Action executors: 60 min
- Integration: 60 min

---

### Option C: Utility AI (Weighted Scoring)

**Concept:** Score each action, pick highest
```python
scores = {
    "flee": calculate_flee_urgency(entity),      # 0.0-1.0
    "investigate": calculate_curiosity(entity),  # 0.0-1.0
    "forage": calculate_hunger(entity)           # 0.0-1.0
}
action = max(scores, key=scores.get)
```

**Pros:**
- More nuanced than binary conditions
- XCOM, Sims use this successfully
- Natural tie-breaking (scores vs priorities)

**Cons:**
- Harder to author (tune weights)
- Harder to debug (why did it choose X?)
- More computation (score everything, not just check conditions)

**Use When:**
- Need subtle behavior variation
- Many competing motivations
- Tuning/balancing is acceptable

**Don't Use When:**
- Clear priority rules (your "if predator, flee" is clear)
- Deterministic behavior desired

---

### Option D: Hybrid (Priority List + Utility Scoring)

**Pattern:**
```yaml
behaviors:
  - id: flee
    priority: 1
    conditions:
      - type: threat_nearby
    scoring:  # only if multiple threats
      function: closest_threat_urgency

  - id: investigate
    priority: 3
    conditions:
      - type: curiosity_triggered
    scoring:
      function: investigation_value
```

**Use When:**
- Most behaviors are simple (priority list)
- Some behaviors need nuance (utility scoring)
- Post-MVP enhancement

---

## Recommendation for Shipmind Drifter MVP

**Format: Lightweight YAML Priority List (Minecraft-inspired)**

```yaml
# drifter.behavior.yaml
species: drifter
description: Peaceful space jellyfish, curious but cautious

parameters:
  # Detection ranges
  predator_detect_range: 80.0
  ship_detect_range: 300.0
  hostile_ship_range: 150.0

  # Movement speeds (m/s)
  flee_speed: 15.0
  investigate_speed: 8.0
  forage_speed: 5.0

  # Thresholds
  hostile_sentiment_threshold: -0.5
  friendly_sentiment_threshold: 0.5

behaviors:
  # Priority 1: Flee from predators (highest priority)
  - id: flee_predator
    priority: 1
    conditions:
      - type: nearest_entity
        tag: predator
        max_distance: "{parameters.predator_detect_range}"
    action:
      type: flee
      speed: "{parameters.flee_speed}"

  # Priority 2: Flee from hostile ships
  - id: flee_hostile_ship
    priority: 2
    conditions:
      - type: nearest_entity
        tag: ship
        max_distance: "{parameters.hostile_ship_range}"
      - type: knowledge_token
        key: ship_sentiment
        operator: "<"
        value: "{parameters.hostile_sentiment_threshold}"
    action:
      type: flee
      speed: "{parameters.flee_speed}"

  # Priority 3: Investigate neutral/friendly ships
  - id: investigate_ship
    priority: 3
    conditions:
      - type: nearest_entity
        tag: ship
        max_distance: "{parameters.ship_detect_range}"
    action:
      type: investigate
      speed: "{parameters.investigate_speed}"
      approach_distance: 50.0

  # Priority 10: Default behavior - forage/wander
  - id: forage
    priority: 10
    conditions: []  # always valid (fallback)
    action:
      type: forage
      speed: "{parameters.forage_speed}"
      wander_radius: 100.0
```

**Why This Format:**
1. **Readable:** Game designers can author without coding
2. **Fast:** Simple linear scan, no tree traversal
3. **Debuggable:** Print which behavior ID won
4. **Extensible:** Add new behaviors without changing code
5. **Proven:** Minecraft uses this for 100+ mob types

**Implementation (~200 LOC):**
```python
# behavior_engine.py
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Condition:
    type: str
    params: Dict[str, Any]

    def evaluate(self, entity, world) -> bool:
        if self.type == "nearest_entity":
            target = world.find_nearest(entity, tag=self.params["tag"])
            if target:
                dist = world.distance(entity, target)
                return dist < self.params["max_distance"]
            return False
        elif self.type == "knowledge_token":
            value = entity.knowledge.get(self.params["key"], 0.0)
            op = self.params["operator"]
            threshold = self.params["value"]
            if op == "<": return value < threshold
            elif op == ">": return value > threshold
            # ... etc
        return False

@dataclass
class Behavior:
    id: str
    priority: int
    conditions: List[Condition]
    action: Dict[str, Any]

    def check(self, entity, world) -> bool:
        return all(c.evaluate(entity, world) for c in self.conditions)

class BehaviorEngine:
    def __init__(self, yaml_path: str):
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Parse behaviors
        self.behaviors = []
        for b in config["behaviors"]:
            conditions = [Condition(c["type"], c) for c in b.get("conditions", [])]
            self.behaviors.append(Behavior(
                id=b["id"],
                priority=b["priority"],
                conditions=conditions,
                action=b["action"]
            ))

        # Sort by priority (lower = higher priority)
        self.behaviors.sort(key=lambda b: b.priority)

    def tick(self, entity, world):
        """Select and execute highest-priority valid behavior"""
        for behavior in self.behaviors:
            if behavior.check(entity, world):
                self.execute_action(entity, behavior.action, world)
                return behavior.id  # for debugging
        return None

    def execute_action(self, entity, action, world):
        action_type = action["type"]
        if action_type == "flee":
            # Set entity velocity away from threat
            threat = world.get_current_threat(entity)
            entity.set_flee_velocity(threat, speed=action["speed"])
        elif action_type == "investigate":
            target = world.get_investigation_target(entity)
            entity.set_approach_velocity(target,
                                        speed=action["speed"],
                                        min_distance=action["approach_distance"])
        elif action_type == "forage":
            entity.set_wander_velocity(speed=action["speed"],
                                      radius=action["wander_radius"])

# Usage
engine = BehaviorEngine("species/drifter.behavior.yaml")
for entity in world.entities:
    active_behavior = engine.tick(entity, world)
    print(f"{entity.id}: {active_behavior}")
```

---

## Red Flags

### 1. Using py_trees for 100+ Agents
**Evidence:** Documentation explicitly warns against this
> "not scoped to enable an NPC gaming engine with hundreds of characters"

**Alternative:** Custom priority list

---

### 2. Deep Tree Nesting
**Anti-Pattern:**
```yaml
selector:
  - sequence:
      - selector:
          - sequence:
              - condition: ...
              - action: ...
```

**Why Bad:** Hard to author, hard to debug, traversal overhead

**Instead:** Flat priority list with compound conditions
```yaml
- priority: 1
  conditions: [A, B, C]  # implicit AND
  action: ...
```

---

### 3. Recalculating Expensive Queries Every Frame
**Example:** `world.find_all_entities()` every tick

**Solution:**
- Spatial caching (chunk grids)
- Update AI every N ticks
- Event-driven updates (only when entities move into range)

---

### 4. No Interruption System
**Problem:** Low-priority behavior blocks high-priority

**Example:** Drifter keeps investigating ship even though predator appeared

**Solution:** Re-evaluate priorities every tick (Selector pattern handles this naturally)

---

### 5. Overly Generic "One Size Fits All" Format
**Temptation:** Build format that handles all species + future Unity integration

**Reality:** YAGNI (You Aren't Gonna Need It)

**Start Simple:**
- MVP: 1 species (Drifter)
- Post-MVP: 10 species
- Unity port: Cross that bridge later (Unity has its own BT system)

---

## Additional Resources

### Academic Papers
- "A Survey of Behavior Trees in Robotics and AI" (2022) - https://www.sciencedirect.com/science/article/pii/S0921889022000513
- Comparison between Behavior Trees and FSMs - https://arxiv.org/html/2405.16137v1

### Game AI Pro Book Series
- Chapter 6: "The Behavior Tree Starter Kit" (Game AI Pro 1)
- Chapter 9: "Overcoming Pitfalls in Behavior Tree Design" (Game AI Pro 3)
- Chapter 10: "Building Utility Decisions into Your Existing Behavior Tree" (Game AI Pro 1)

### GDC Talks
- "Handling Complexity in the Halo 2 AI" (2005) - Damian Isla
- "Three Approaches to Halo-style Behavior Tree AI" (2007) - https://www.gdcvault.com/play/760/

### Open Source Examples
- Terasology Behaviors: https://github.com/Terasology/Behaviors
- Behavior3 Editor: https://behavior3.github.io/behavior3editor/
- BehaviorTree.CPP: https://www.behaviortree.dev
- Minecraft Bedrock Docs: https://learn.microsoft.com/en-us/minecraft/creator/reference/content/entityreference/examples/aigoallist

### Python Libraries (if you go that route)
- py_trees: https://py-trees.readthedocs.io/ (robotics, not games)
- behavior3py: https://github.com/behavior3/behavior3py (multi-agent optimized)

---

## Next Steps

1. **Prototype lightweight priority list** (4 hours)
   - Parse YAML config
   - Implement 3-4 condition types
   - Implement 3 action types (flee, investigate, forage)
   - Test with 10 Drifters

2. **Benchmark** (2 hours)
   - Profile with 100, 250, 500 entities
   - Measure condition evaluation time
   - Measure action execution time
   - Confirm <3ms budget met

3. **Iterate** (ongoing)
   - Add species #2 (test config reusability)
   - Add condition types as needed
   - Add action types as needed
   - Don't add features until proven necessary

4. **Consider Unity integration** (post-MVP)
   - Export YAML → Unity BT format
   - Or embed Python in Unity (IronPython?)
   - Or rewrite kernel in C# (if Python too slow)

---

**End of Research Document**
