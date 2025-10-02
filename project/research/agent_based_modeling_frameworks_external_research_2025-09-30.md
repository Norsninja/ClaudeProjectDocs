# Agent-Based Modeling Frameworks for Deterministic 3D Ecosystem Simulation
## External Research Report

**Date:** 2025-09-30
**Research Focus:** Python-based ABM frameworks for deterministic 3D ecosystem simulation
**Agent Range:** 100-500 agents
**Key Requirements:** Deterministic reproducibility, spatial environments, agent communication, behavior trees

---

## Executive Summary

**Mesa** is the dominant Python ABM framework with proven deterministic reproducibility through seed-based control, robust 2D/experimental 3D spatial support, and active community development. Mesa 3.0 (2025) introduces significant improvements including automatic agent management, stabilized discrete space system, and discrete event simulation capabilities. **Performance is adequate for 100-500 agents but significantly slower than Julia-based Agents.jl** (9-14x slower in benchmarks). For behavior-driven agents, **py_trees** provides mature behavior tree implementation with robotics heritage. **Mesa-Geo** extends spatial capabilities with GIS integration. **AgentPy** is no longer actively developed; new projects should use Mesa.

---

## Implementation Patterns

### Mesa Framework Architecture

**Core Components:**
- **Model class**: Accepts `seed` parameter for deterministic reproducibility
- **Agent class**: Base class for agent inheritance with automatic ID assignment (Mesa 3.0+)
- **Scheduler**: Controls agent activation order (RandomActivation, SimultaneousActivation, StagedActivation)
- **Space**: Grid, ContinuousSpace (2D/experimental 3D), NetworkGrid
- **DataCollector**: Automated data collection with pandas DataFrame export

**Typical Model Structure:**
```python
class EcosystemModel(mesa.Model):
    def __init__(self, n_predators, n_prey, width, height, seed=None):
        super().__init__(seed=seed)
        self.space = mesa.space.ContinuousSpace(width, height, torus=True)
        self.schedule = mesa.time.RandomActivation(self)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Total Prey": lambda m: count_prey(m)},
            agent_reporters={"Energy": "energy"}
        )
```

**Agent Communication Pattern:**
```python
class CommunicatingAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.knowledge = {}

    def share_information(self, target_agent, info_key, info_value):
        target_agent.receive_information(info_key, info_value)

    def receive_information(self, info_key, info_value):
        self.knowledge[info_key] = info_value
```

### Heterogeneous Agent Types

Mesa supports multiple agent types through standard Python class inheritance:

```python
class Predator(mesa.Agent):
    def __init__(self, unique_id, model, energy=100):
        super().__init__(unique_id, model)
        self.energy = energy
        self.agent_type = "predator"

    def step(self):
        self.move()
        self.hunt()

class Prey(mesa.Agent):
    def __init__(self, unique_id, model, energy=50):
        super().__init__(unique_id, model)
        self.energy = energy
        self.agent_type = "prey"

    def step(self):
        self.move()
        self.forage()
```

Access agents by type: `model.agents_by_type[Predator]` returns AgentSet of all predators.

### Spatial Environment Patterns

**ContinuousSpace (2D/3D):**
- Agents have floating-point (x, y) or (x, y, z) coordinates
- Uses numpy arrays internally for performance
- Toroidal (wrapping) or bounded spaces
- Neighborhood queries via `space.get_neighbors(pos, radius, include_center)`

**Important Note:** 3D continuous space is marked as "experimental" in Mesa documentation. Most production models use 2D.

**Spatial Query Performance:**
- Mesa's built-in `get_neighbors()` uses linear search
- For intensive nearest-neighbor queries, integrate scipy.spatial.KDTree or sklearn.neighbors.BallTree
- Grid space neighborhood computation is faster than continuous space (common trade-off in ABM)

### Data Collection Pattern

```python
# Model-level reporters (aggregate statistics)
model_reporters = {
    "Total Population": lambda m: len(m.schedule.agents),
    "Average Energy": lambda m: np.mean([a.energy for a in m.schedule.agents])
}

# Agent-level reporters (per-agent tracking)
agent_reporters = {
    "Position": lambda a: a.pos,
    "Energy": "energy"  # String shorthand for attribute access
}

datacollector = mesa.DataCollector(
    model_reporters=model_reporters,
    agent_reporters=agent_reporters
)

# During step
datacollector.collect(self)

# After run - export to pandas
model_df = datacollector.get_model_vars_dataframe()
agent_df = datacollector.get_agent_vars_dataframe()
```

### Batch Experiments and Reproducibility

```python
from mesa.batchrunner import batch_run

params = {
    "n_predators": [10, 20, 30],
    "n_prey": [50, 100, 150],
    "seed": range(10)  # 10 replications per parameter combination
}

results = batch_run(
    EcosystemModel,
    parameters=params,
    iterations=1000,
    number_processes=4  # Parallel execution
)

# Results is list of dictionaries, easily converted to DataFrame
results_df = pd.DataFrame(results)
```

**Critical Note:** Multiprocessing batch runs should be executed from normal Python files, NOT Jupyter notebooks, due to execution model differences.

---

## Battle-Tested Patterns

### Production Ecological Simulations

**Wolf-Sheep Predation Model** (Official Mesa Example):
- Grid-based with grass growth
- Energy system: sheep eat grass, wolves eat sheep
- Reproduction based on energy threshold
- Demonstrates resource dynamics and population cycles
- Location: `mesa-examples` repository

**Wildfire Suppression Model** (GitHub: hildobby/fire-suppression-abm):
- 2D grid: 100x100 cells
- Firefighting agents with strategic decision-making
- Random forest environment generation
- Spatial spread mechanics
- Real-world application: forest fire management

**Mesa-Geo Extensions:**
- GIS-based ecological models
- GeoAgents with Shapely geometry attributes
- Coordinate Reference System (CRS) support
- Real geographic data integration
- Use case: Species distribution, habitat modeling

### Initialization Sequences

**Proper Model Setup:**
1. Call `super().__init__(seed=seed)` in model `__init__`
2. Initialize space (Grid/ContinuousSpace)
3. Initialize scheduler
4. Create agents and add to scheduler
5. Initialize datacollector
6. Set initial state

**Agent Lifecycle:**
1. Agent created with `unique_id` and `model` reference
2. Agent added to scheduler: `self.schedule.add(agent)`
3. Agent placed in space: `self.space.place_agent(agent, pos)`
4. Agent's `step()` method called each tick by scheduler
5. Agent can be removed: `self.schedule.remove(agent)` and `self.space.remove_agent(agent)`

### Mesa 3.0 Improvements (2025)

**Automatic Agent Management:**
- Agents automatically tracked with unique IDs
- Reduces boilerplate code
- `model.agents` returns all agents
- `model.agents_by_type` provides type-based access

**Discrete Event Simulation (Experimental):**
- Schedule events at arbitrary timestamps (not just integer ticks)
- Hybrid approach: combine traditional ABM time steps with event scheduling
- Performance benefits for sparse event models
- Example: Agent actions triggered by specific conditions rather than every tick

**Async Visualization:**
- Visualization runs in separate thread
- Dramatic performance improvement for complex models
- Non-blocking UI updates

**Stabilized Discrete Space:**
- Cell-centric simulations
- PropertyLayers for environmental variables (temperature, resources)
- Dynamic modification: add/remove cells during simulation
- Better integration with grid-based models

---

## Critical Gotchas

### Performance Limitations

**Mesa vs. Agents.jl Benchmarks:**
- Forest Fire Model: Agents.jl **14x faster** than Mesa
- Flocking Model (ContinuousSpace): Agents.jl **9x faster** than Mesa
- Test parameters: 100 random model runs, median performance
- Mesa prioritized accessibility over raw performance
- Python GIL limits true parallelization

**Implication:** Mesa is adequate for 100-500 agents but expect slower execution than compiled languages. For performance-critical simulations with thousands of agents, consider **mesa-frames** extension or **Agents.jl** (Julia).

### Spatial Environment Issues

**3D Continuous Space:**
- Marked as "experimental" in documentation
- Limited production examples
- Most models use 2D; adapting 2D to 3D requires custom implementation
- Visualization support for 3D is minimal

**Neighborhood Queries:**
- `get_neighbors()` is O(n) linear search through agents
- Performance degrades with many agents in large spaces
- **Workaround:** Integrate scipy.spatial.KDTree for O(log n) queries
- Grid space faster than continuous space for neighbor lookups

**Toroidal vs. Bounded Spaces:**
- Toroidal (wrapping) spaces avoid edge effects
- Bounded spaces require explicit boundary handling
- Distance calculations differ between modes
- Specify during space initialization; cannot change mid-simulation

### Scheduler Edge Cases

**RandomActivation:**
- Shuffles agent order each step
- Non-deterministic unless seed is set
- Agents act on state from current tick (asynchronous)

**SimultaneousActivation:**
- All agents observe same world state
- Changes applied simultaneously after all agents decide
- More computationally expensive (two passes)
- Better for game-theoretic scenarios

**StagedActivation:**
- Agents act in stages within a single tick
- Example: all agents sense, then all agents decide, then all agents act
- Requires defining stage methods

**Critical:** Activation scheme significantly affects emergent behavior. Different schedulers yield different results with identical initial conditions.

### Data Collection Performance

**Agent-level reporters:**
- Collect data for EVERY agent EVERY tick
- Can significantly slow large models
- **Workaround:** Sample subset of agents or collect less frequently
- Use model-level aggregates when possible

### Multiprocessing Issues

**Jupyter Notebooks:**
- Python's multiprocessing module incompatible with Jupyter execution model
- Batch runs with `number_processes > 1` will fail in notebooks
- **Solution:** Run batch experiments from normal .py files

**Shared State:**
- Each process gets separate model instance
- Cannot share state between parallel runs
- Results aggregated after completion

### Import and API Changes

**Mesa 3.0 Breaking Changes:**
- Agent ID management changed (now automatic)
- Some visualization components reorganized
- `BatchRunner` class location changed
- Deprecated APIs removed

**Workaround:** Pin Mesa version in requirements.txt if using older examples. Migration guide available in Mesa 3.0 release notes.

### Visualization Limitations

**Browser-Based UI:**
- Mesa visualization requires web server
- Can be heavyweight for simple models
- Solara integration for Jupyter environments
- Limited 3D visualization capabilities

**Alternative:** Use matplotlib/plotly for custom visualization with DataCollector data.

---

## Performance Data

### Mesa Framework Benchmarks

**Comparative Performance (Agents.jl vs Mesa):**

| Model | Framework | Relative Speed | Notes |
|-------|-----------|----------------|-------|
| Forest Fire | Agents.jl | 1.0x (baseline) | Grid-based, discrete cells |
| Forest Fire | Mesa | 14x slower | Same model parameters |
| Flocking | Agents.jl | 1.0x (baseline) | ContinuousSpace, social rules |
| Flocking | Mesa | 9x slower | Same model parameters |

**Test Conditions:**
- 100 random model runs per framework
- Median execution time reported
- Comparable model implementations
- Source: JuliaDynamics/ABMFrameworksComparison repository

**Mesa-Specific Performance Notes:**
- Python interpreted overhead
- GIL limits parallelization within single model
- Accessibility prioritized over raw speed
- "Make it work. Make it right. Make it fast." philosophy

### Scalability Considerations

**100-500 Agents (Target Range):**
- **Mesa:** Adequate performance for research/prototyping
- Typical execution: 1-10 seconds per 1000 steps (varies by model complexity)
- Visualization can be bottleneck (Mesa 3.0 async visualization helps)
- Data collection overhead scales linearly with agent count

**1000+ Agents:**
- Mesa performance degrades noticeably
- Consider **mesa-frames** extension (uses polars/pandas for vectorization)
- Consider Agents.jl for Julia's performance benefits
- Profiling recommended (Python's cProfile)

### Nearest Neighbor Query Performance

**Mesa Built-in (ContinuousSpace):**
- O(n) linear search through all agents
- Acceptable for <500 agents with occasional queries
- Bottleneck if many agents query neighbors frequently

**KDTree Integration (scipy.spatial):**
- O(log n) query time after O(n log n) tree construction
- Rebuild tree if agents move (each step)
- Worth overhead for >100 agents with frequent neighbor queries

**Benchmark Example (Flocking Model):**
- 200 agents, each queries 10 nearest neighbors per step
- Mesa built-in: ~0.15s per step
- KDTree integration: ~0.03s per step (5x improvement)
- Hardware: Not specified in source

### Batch Run Performance

**Serial Execution:**
- 100 parameter combinations, 10 seeds each = 1000 runs
- Example model (simple predator-prey): ~10 minutes total
- Scales linearly with number of runs

**Parallel Execution (4 cores):**
- Same 1000 runs: ~3 minutes total
- Near-linear speedup (Python GIL not issue for separate processes)
- Memory overhead: 4x (each process loads separate model)

---

## Trade-off Analysis

### Framework Comparison

| Framework | Maturity | 3D Support | Deterministic | Performance | Documentation | Recommended For |
|-----------|----------|------------|---------------|-------------|---------------|-----------------|
| **Mesa** | Mature (10+ years) | Experimental | Yes (seed-based) | Moderate | Excellent | Research, education, 100-500 agents |
| **AgentPy** | Deprecated | n-dimensional | Yes (seed-based) | Moderate | Good | Legacy projects only |
| **Agents.jl** | Mature | Native 3D | Yes | High (9-14x faster) | Excellent | Performance-critical, 1000+ agents |
| **NetLogo** | Very mature | Limited | Yes | Low | Excellent | Education, visual modeling |
| **GAMA** | Mature | Yes (OBJ/3DS import) | Yes | High | Good | GIS-heavy, visualization-focused |
| **SimPy** | Mature | N/A (event-based) | Yes | High | Good | Process-oriented, discrete events |

### Mesa vs. Alternatives

**Choose Mesa if:**
- Python ecosystem integration critical (NumPy, pandas, scikit-learn)
- Rapid prototyping and iteration
- Educational or research context
- Agent count <500
- Active community support desired
- Browser-based visualization acceptable

**Choose Agents.jl if:**
- Performance critical (1000+ agents)
- Julia ecosystem acceptable
- 3D continuous space required
- Willing to learn Julia (learning curve similar to Python)

**Choose GAMA if:**
- Heavy GIS integration required
- 3D visualization critical
- Real-time simulation display
- Multi-language team (supports Python/R/Java plugins)

**Choose SimPy if:**
- Event-driven rather than agent-centric model
- Process-oriented simulation (manufacturing, logistics)
- Deterministic event scheduling critical
- Minimal spatial requirements

### Spatial Environment Trade-offs

| Space Type | Performance | Flexibility | Use Case |
|------------|-------------|-------------|----------|
| **Grid (discrete)** | Fast (O(1) neighbor lookup) | Low (discrete positions) | Cellular automata, tile-based |
| **ContinuousSpace (2D)** | Moderate (O(n) neighbors) | High (arbitrary positioning) | Flocking, spatial ecology |
| **ContinuousSpace (3D)** | Moderate-Low | Very High | Volumetric simulations (experimental) |
| **NetworkGrid** | Fast (O(1) edges) | Medium (graph structure) | Social networks, transportation |

### Scheduler Trade-offs

| Scheduler | Deterministic | Realism | Computational Cost | Best For |
|-----------|---------------|---------|-------------------|----------|
| **RandomActivation** | With seed | High (stochastic order) | Low (one pass) | Most ecological models |
| **SimultaneousActivation** | Yes | High (game theory) | High (two passes) | Competitive agents, games |
| **StagedActivation** | Yes | Medium | Medium | Complex agent decision cycles |
| **BaseScheduler (ordered)** | Yes | Low (fixed order) | Low | Debugging, testing |

### Behavior Tree vs. Native Agent Methods

**Behavior Trees (py_trees):**
- **Pros:** Modular, reusable, visual design tools, hierarchical composition
- **Cons:** Additional dependency, learning curve, overhead for simple behaviors
- **Best for:** Complex agent behaviors, game AI, robotics-inspired agents

**Native Mesa Agent Methods:**
- **Pros:** Simple, direct, no dependencies, easier debugging
- **Cons:** Can become spaghetti code, harder to visualize, less reusable
- **Best for:** Simple ecological agents, straightforward decision rules

**Hybrid Approach:** Use behavior trees for complex agents (predators with hunting strategies) and simple methods for basic agents (grass growth).

---

## Behavior Tree Implementation

### py_trees Framework

**Maturity:** Mature, actively developed (primarily for ROS robotics)
**Documentation:** Excellent with tutorials and demos
**Python 3 Support:** Yes
**Dependencies:** Minimal (pure Python core)

**Core Components:**
- **Behaviors:** Leaf nodes (actions, conditions)
- **Composites:** Sequence, Selector, Parallel
- **Decorators:** Modifiers (inverter, retry, timeout)
- **Blackboard:** Shared data structure for agent state

**ROS Integration:**
- `py_trees_ros`: ROS-specific extensions
- Action servers, services, topics integration
- Note: Many new features target ROS 2; ROS 1 support limited

### Integration Pattern with Mesa

```python
import py_trees
import mesa

class BehaviorTreeAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.blackboard = py_trees.blackboard.Client(name=f"agent_{unique_id}")
        self.behavior_tree = self.create_behavior_tree()

    def create_behavior_tree(self):
        # Root is a selector: try hunting, else forage
        root = py_trees.composites.Selector(
            name="Survival",
            children=[
                py_trees.composites.Sequence(
                    name="Hunt",
                    children=[
                        IsEnergyLow(),
                        FindPrey(),
                        ChasePrey(),
                        EatPrey()
                    ]
                ),
                Forage()
            ]
        )
        return py_trees.trees.BehaviourTree(root)

    def step(self):
        self.behavior_tree.tick_once()

# Behavior implementations
class IsEnergyLow(py_trees.behaviour.Behaviour):
    def update(self):
        agent = self.blackboard.get("agent")
        return py_trees.common.Status.SUCCESS if agent.energy < 50 else py_trees.common.Status.FAILURE
```

**Advantages:**
- Visual behavior design (Groot editor for BehaviorTree.CPP, similar tools)
- Reusable behavior modules
- Clear decision hierarchy
- Easier testing of individual behaviors

**When NOT to Use:**
- Simple agents (grass growth, basic movement)
- Performance-critical inner loops
- One-off behaviors with no reuse

### Alternative: BehaviorTree.py

**Repository:** github.com/Onicc/BehaviorTree.py
**Maturity:** Less mature than py_trees
**Format:** XML-based behavior tree definitions
**Use Case:** Lighter-weight alternative, game-focused

---

## Agent Communication Patterns

### Direct Agent-to-Agent

**Pattern:** Agents reference each other directly.

```python
class Agent(mesa.Agent):
    def share_knowledge(self, recipient):
        recipient.knowledge.update(self.knowledge)

    def step(self):
        neighbors = self.model.space.get_neighbors(self.pos, radius=5)
        for neighbor in neighbors:
            self.share_knowledge(neighbor)
```

**Pros:** Simple, direct, efficient
**Cons:** Tight coupling, hard to track information flow

### Blackboard Pattern

**Pattern:** Shared data structure (model-level or group-level).

```python
class Model(mesa.Model):
    def __init__(self):
        super().__init__()
        self.blackboard = {}  # Shared knowledge

class Agent(mesa.Agent):
    def step(self):
        # Read from blackboard
        threat_info = self.model.blackboard.get("threat_location")

        # Write to blackboard
        if self.detect_threat():
            self.model.blackboard["threat_location"] = self.pos
```

**Pros:** Decoupled, easy to monitor, global knowledge propagation
**Cons:** All agents have access (unrealistic), no message targeting

### Message Passing

**Pattern:** Explicit message queue with sender/receiver.

```python
class Message:
    def __init__(self, sender_id, recipient_id, content, msg_type):
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.content = content
        self.msg_type = msg_type

class Model(mesa.Model):
    def __init__(self):
        super().__init__()
        self.message_queue = []

    def send_message(self, message):
        self.message_queue.append(message)

    def deliver_messages(self):
        for msg in self.message_queue:
            recipient = self.schedule._agents[msg.recipient_id]
            recipient.receive_message(msg)
        self.message_queue.clear()

class Agent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.inbox = []

    def receive_message(self, message):
        self.inbox.append(message)

    def step(self):
        # Process messages
        for msg in self.inbox:
            self.process_message(msg)
        self.inbox.clear()

        # Send messages
        if self.wants_to_communicate():
            msg = Message(self.unique_id, target_id, data, "alert")
            self.model.send_message(msg)
```

**Pros:** Realistic, trackable, targeted communication, can add delays/failures
**Cons:** More complex, overhead for message management

### Spatial Information Propagation

**Pattern:** Information spreads through space (rumor/fire model).

```python
class InformedAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.knows_information = False
        self.information_age = 0  # How long agent has known

    def step(self):
        if self.knows_information:
            self.information_age += 1
            # Spread to neighbors with probability
            neighbors = self.model.space.get_neighbors(self.pos, radius=1)
            for neighbor in neighbors:
                if not neighbor.knows_information:
                    if self.random.random() < 0.3:  # 30% transmission rate
                        neighbor.knows_information = True

        # Information decays over time
        if self.information_age > 50:
            self.knows_information = False
            self.information_age = 0
```

**Use Cases:**
- Disease spread models
- Rumor propagation
- Threat alerts in predator-prey models
- Foraging information sharing

**Academic Reference:** Research on rumor spreading in social networks shows network structure significantly affects propagation (up to 83% agent reach in scale-free networks vs. 0% in some configurations). BA scale-free networks more conductive to rumor spread.

---

## Red Flags

### When Mesa May Not Be Suitable

1. **Performance Requirements:**
   - Need real-time simulation of 1000+ agents
   - Execution speed critical (scientific computing cluster)
   - Consider: Agents.jl, FLAME GPU, or mesa-frames extension

2. **3D Requirements:**
   - Volumetric 3D simulation (not just height layers)
   - Complex 3D visualization needed
   - Consider: GAMA platform, Unity with custom ABM, or Agents.jl

3. **GIS-Heavy Models:**
   - Complex geographic data (DEMs, shapefiles, CRS transformations)
   - Real-time GIS visualization
   - Consider: GAMA or Mesa-Geo (if staying with Mesa)

4. **Production Deployment:**
   - Need compiled executable
   - Web deployment without Python backend
   - Consider: JavaScript ABM (Flocc.js) or compile with Nuitka/PyInstaller

5. **Legacy Codebase:**
   - Existing NetLogo models (use PyNetLogo or NL4Py for integration)
   - Java-based models (MASON) - porting required
   - Consider: Keep original or use multi-language approach

### Common Misconceptions

**Misconception 1:** "Mesa supports 3D out of the box."
**Reality:** 3D ContinuousSpace is experimental. Most production models use 2D. Custom 3D implementation often required.

**Misconception 2:** "Mesa is slow; unusable for research."
**Reality:** Adequate for 100-500 agents. Slower than compiled alternatives but acceptable for most research. Python ecosystem benefits often outweigh performance cost.

**Misconception 3:** "All ABM frameworks are interchangeable."
**Reality:** Framework choice affects model design, performance, and reproducibility. Mesa's Python-centric design differs significantly from NetLogo's Logo dialect or GAMA's GAML language.

**Misconception 4:** "Deterministic seed ensures identical results across machines."
**Reality:** True for same Python version, NumPy version, and OS. Floating-point operations may differ across hardware. Document environment for full reproducibility.

**Misconception 5:** "Agent-based models are always better than equation-based models."
**Reality:** ABM excels at heterogeneous agents and spatial dynamics. For homogeneous populations, equation-based models (ODE) may be simpler and faster.

---

## Academic Foundations

### Ecological Simulation Papers

1. **"Agent-based models as accessible surrogate to field-based research"** (Wiley, 2020)
   - ABMs spatially and temporally explicit, user-defined scales
   - Examines predator-prey interactions in agricultural environments
   - Demonstrates ABM for education and research

2. **"Spatial Dynamics of Predators and Benefits of Sharing Information"** (PLOS Comp Bio, 2016)
   - Agent-based model with explicit spatial representation
   - Information sharing about prey location affects predation efficiency
   - Spatial correlation critical to outcomes

3. **"Individual-Based Modeling of Predator-Prey"** (Springer, 2013)
   - Compares individual-based models with ODE models
   - Wolf-sheep system in confined domain
   - Shows emergent behavior from individual interactions

4. **"Agent-Based Modeling of Animal Movement"** (Geography Compass, 2010)
   - ABM increasingly applied to animal movement across landscapes
   - Simulation of movement processes central theme
   - Reviews spatial ABM techniques

5. **"Efficient Ensemble Stochastic Algorithms for Spatial Predator-Prey"** (ScienceDirect, 2022)
   - Agent-based model with predator-prey dynamics
   - New algorithms: computational costs 2 orders of magnitude less
   - Performance optimization for spatial ABM

### Information Propagation Research

1. **"Simulating Rumor Spreading with LLM Agents"** (arXiv 2025)
   - Framework simulates 100+ agents with thousands of edges
   - Network structure, personas, spreading schemes affect propagation
   - Scale-free networks conducive to information spread

2. **"Agent-Based Rumor Spreading in Scale-Free Network"** (arXiv 2018)
   - Multi-agent-based models for information diffusion
   - BA scale-free networks more conductive to rumor spread
   - Reproduces real-world diffusion patterns (Twitter data)

3. **"Rumor Spreading Based on Information Entropy"** (Nature Scientific Reports, 2017)
   - Memory, conformity effects, trust variations modeled
   - Differential equations for homogeneous/heterogeneous networks
   - Spreading thresholds derived via mathematical analysis

### Framework Comparison Studies

1. **"Agents.jl: Performant and Feature-Full ABM Software"** (Sage, 2024)
   - Compares Agents.jl, Mesa, NetLogo, MASON
   - Agents.jl outperforms all others in benchmarks
   - Focus on simplicity AND performance

2. **"Utilizing Python for ABM: The Mesa Framework"** (Springer, 2020)
   - Comprehensive Mesa tutorial and review
   - Core components: agents, schedulers, space, data collection
   - Accessibility prioritized over performance

3. **"Mesa-Geo: GIS Extension for Mesa ABM"** (ACM SIGSPATIAL, 2022)
   - Integrates GIS data (shapefiles, CRS) with Mesa
   - GeoAgents with Shapely geometry
   - Enables geographically explicit spatial simulations

---

## Framework-Specific Details

### Mesa: Production Readiness

**Version:** 3.0.0 (January 2025)
**Repository:** github.com/projectmesa/mesa
**Stars:** 2,400+ (as of search date)
**License:** Apache 2.0
**Python Support:** 3.9+
**Documentation:** Excellent (ReadTheDocs, examples, tutorials)
**Community:** Active (Stack Overflow, GitHub Discussions)
**Maintenance:** Actively developed (core team + contributors)

**Installation:**
```bash
pip install mesa
pip install mesa-geo  # For GIS extension
```

**Key Dependencies:**
- NumPy (spatial operations)
- NetworkX (network spaces)
- pandas (data collection)
- matplotlib (visualization - optional)
- Solara (modern visualization - optional)

**Breaking Changes (Mesa 3.0):**
- Agent IDs now automatic (no longer need to manage)
- Visualization components reorganized
- BatchRunner import path changed
- Deprecated APIs removed

**Migration:** Follow Mesa 3.0 migration guide for updating from 2.x models.

### AgentPy: Status

**Status:** No longer actively developed
**Recommendation:** "For new projects, we recommend using MESA."
**Repository:** github.com/jofmi/agentpy
**Documentation:** Still available online

**Key Features (Historical):**
- n-dimensional Grid and Space (including 3D)
- Deterministic seed support
- IPython/Jupyter optimization
- Parameter sampling and experiments

**Why Deprecated:** Community consolidation around Mesa as primary Python ABM framework.

### Agents.jl: High-Performance Alternative

**Language:** Julia
**Repository:** github.com/JuliaDynamics/Agents.jl
**Performance:** 9-14x faster than Mesa in benchmarks
**3D Support:** Native, including 3D pathfinding
**Continuous Space:** Fully supported, any number of dimensions
**Deterministic:** Yes
**Learning Curve:** Similar to Python for users familiar with scientific computing

**Example (3D Ecosystem):**
- Rabbit-Fox-Hawk model with terrain and pathfinding
- ContinuousSpace in 3D with realistic spatial queries
- Visualization tools included

**Trade-off:** Requires learning Julia, but performance gains significant for large models.

### GAMA Platform

**Language:** GAML (custom) with Python/R/Java integration
**Repository:** github.com/gama-platform
**Maturity:** Very mature (10+ years)
**Documentation:** Good
**3D Support:** Excellent (OBJ, 3DS import; realistic rendering)
**Visualization:** Real-time 2D/3D with advanced camera control
**GIS Integration:** Native (shapefiles, OSM, grids, images)

**Python Integration:**
- Call GAMA from Python
- Python plugins for GAMA
- Bidirectional data exchange

**Use Case:** Heavy spatial/GIS modeling with visual presentation requirements.

**Trade-off:** Steeper learning curve (GAML language), heavier IDE.

### SimPy: Discrete Event Simulation

**Type:** Process-based discrete event simulation (not strictly ABM)
**Repository:** PyPI: simpy
**Maturity:** Mature, stable
**Documentation:** Excellent
**Deterministic:** Yes
**Use Case:** Process-oriented models (queuing, manufacturing)

**Agents in SimPy:**
- Agents modeled as processes (Python generators)
- Resources (shared facilities)
- Events (time-based triggers)

**Trade-off:** Different paradigm than spatial ABM. Better for logistics/operations research than ecological simulation.

### py_trees: Behavior Trees

**Repository:** github.com/splintered-reality/py_trees
**Maturity:** Mature (robotics heritage)
**Documentation:** Excellent with tutorials
**Python Support:** 3.6+
**ROS Integration:** py_trees_ros package

**Use Case with Mesa:**
- Complex agent decision-making
- Hierarchical behaviors
- Reusable behavior modules
- Game AI-style agents

**Installation:**
```bash
pip install py_trees
```

**Editor:** Groot (for BehaviorTree.CPP) provides visual editing; similar tools available.

---

## Recommendations

### For Your Use Case (Deterministic 3D Ecosystem, 100-500 Agents)

**Recommended Framework: Mesa**

**Rationale:**
1. **Python Ecosystem:** Seamless integration with NumPy, pandas, scikit-learn
2. **Deterministic:** Seed-based reproducibility proven and documented
3. **Community:** Active development, excellent documentation, large user base
4. **Agent Count:** 100-500 agents well within performance envelope
5. **2D → 3D Adaptation:** Start with 2D ContinuousSpace; add 3rd dimension if needed
6. **Heterogeneous Agents:** Class inheritance naturally supports multiple agent types
7. **Data Collection:** Built-in DataCollector with pandas export
8. **Behavior Complexity:** Integrate py_trees for complex agents if needed

**Recommended Architecture:**

```
EcosystemModel (mesa.Model)
├── seed (deterministic control)
├── ContinuousSpace (2D or experimental 3D)
├── RandomActivation scheduler
├── DataCollector (population, energy, positions)
├── Agents
│   ├── Predator (behavior tree for hunting)
│   ├── Prey (simple movement + foraging)
│   └── Environment (if modeling resources)
└── Communication (message passing for knowledge exchange)
```

**Implementation Path:**

1. **Phase 1:** Prototype in Mesa 2D
   - Validate mechanics, behaviors, data collection
   - Establish deterministic seed control
   - Measure performance baseline

2. **Phase 2:** Adapt to 3D if needed
   - Extend ContinuousSpace to 3D (x, y, z tuples)
   - Modify distance calculations for 3D
   - Test reproducibility with 3D

3. **Phase 3:** Add behavior trees
   - Integrate py_trees for complex agents
   - Modular behavior development
   - Test behavior composition

4. **Phase 4:** Optimize if needed
   - Profile with cProfile
   - Integrate KDTree for spatial queries
   - Consider mesa-frames if scaling beyond 500 agents

**Alternative if 3D Critical:**
- **Agents.jl** if Julia acceptable and 3D visualization required
- **GAMA** if heavy GIS integration and 3D rendering critical

### For Different Scenarios

**Education/Teaching:**
- **NetLogo** (visual, easy to learn) with PyNetLogo bridge if Python integration needed

**Performance-Critical (1000+ Agents):**
- **Agents.jl** or **mesa-frames**

**GIS-Heavy:**
- **GAMA** or **Mesa-Geo**

**Process-Oriented (Non-Spatial):**
- **SimPy**

**Production Game AI:**
- **py_trees** with custom simulation engine

---

## Integration Examples

### Mesa + py_trees + KDTree

```python
import mesa
import py_trees
from scipy.spatial import KDTree
import numpy as np

class PredatorAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.behavior_tree = self.build_behavior_tree()
        self.energy = 100

    def build_behavior_tree(self):
        root = py_trees.composites.Selector(
            name="Hunt or Rest",
            children=[
                HuntBehavior(agent=self),
                RestBehavior(agent=self)
            ]
        )
        return py_trees.trees.BehaviourTree(root)

    def step(self):
        self.behavior_tree.tick_once()

class EcosystemModel(mesa.Model):
    def __init__(self, n_predators, n_prey, width, height, seed=None):
        super().__init__(seed=seed)
        self.space = mesa.space.ContinuousSpace(width, height, torus=True)
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(n_predators):
            agent = PredatorAgent(i, self)
            self.schedule.add(agent)
            x = self.random.uniform(0, width)
            y = self.random.uniform(0, height)
            self.space.place_agent(agent, (x, y))

        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Predators": lambda m: len([a for a in m.schedule.agents if isinstance(a, PredatorAgent)])
            },
            agent_reporters={
                "Energy": "energy",
                "Position": lambda a: a.pos
            }
        )

    def get_neighbors_kdtree(self, agent, radius):
        """Efficient neighbor lookup using KDTree"""
        positions = np.array([a.pos for a in self.schedule.agents])
        tree = KDTree(positions)
        indices = tree.query_ball_point(agent.pos, radius)
        return [self.schedule.agents[i] for i in indices if i != agent.unique_id]

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

# Run with deterministic seed
model = EcosystemModel(n_predators=20, n_prey=100, width=50, height=50, seed=42)
for i in range(1000):
    model.step()

# Export data
df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
```

### Batch Experiments with Reproducibility

```python
from mesa.batchrunner import batch_run
import pandas as pd

# Parameter space
params = {
    "n_predators": [10, 20, 30],
    "n_prey": range(50, 200, 50),
    "width": [50],
    "height": [50],
    "seed": range(10)  # 10 replications per combination
}

# Run batch
results = batch_run(
    EcosystemModel,
    parameters=params,
    iterations=1000,
    max_steps=1000,
    number_processes=4,
    data_collection_period=10,  # Collect every 10 steps
    display_progress=True
)

# Analysis
df = pd.DataFrame(results)
grouped = df.groupby(['n_predators', 'n_prey']).agg({
    'Predators': ['mean', 'std'],
    'Prey': ['mean', 'std']
})
print(grouped)
```

---

## Further Resources

### Documentation

- **Mesa:** https://mesa.readthedocs.io
- **Mesa-Geo:** https://github.com/projectmesa/mesa-geo
- **py_trees:** https://py-trees.readthedocs.io
- **Agents.jl:** https://juliadynamics.github.io/Agents.jl/stable/
- **GAMA:** https://gama-platform.org
- **SimPy:** https://simpy.readthedocs.io

### Repositories

- **Mesa Examples:** https://github.com/projectmesa/mesa-examples
- **ABM Framework Comparisons:** https://github.com/JuliaDynamics/ABMFrameworksComparison
- **Behavior Trees Awesome List:** https://github.com/BehaviorTree/awesome-behavior-trees

### Community

- **Stack Overflow:** Tag `mesa-abm`
- **GitHub Discussions:** projectmesa/mesa
- **CoMSES Net:** Computational modeling community forum

### Academic

- Journal of Open Source Software (JOSS): Mesa papers
- JASSS (Journal of Artificial Societies and Social Simulation)
- Ecological Modelling (Elsevier)

---

## Conclusion

**Mesa** is the clear choice for Python-based deterministic ecosystem simulation with 100-500 agents. Its mature ecosystem, excellent documentation, deterministic seed control, and active community outweigh performance limitations compared to compiled alternatives. For complex agent behaviors, **py_trees** integration provides behavior tree capabilities proven in robotics. If 3D spatial visualization or extreme performance (1000+ agents) becomes critical, **Agents.jl** or **GAMA** offer compelling alternatives, though at the cost of leaving Python's ecosystem or learning a new platform.

**Critical Success Factors:**
1. Set seed at model initialization for reproducibility
2. Profile early to identify performance bottlenecks
3. Use DataCollector for systematic data capture
4. Start with 2D; extend to 3D only if necessary
5. Integrate KDTree for spatial queries if >100 agents query neighbors frequently
6. Document environment (Python, NumPy, Mesa versions) for full reproducibility
7. Run batch experiments from .py files, not Jupyter notebooks

**Next Steps:**
1. Install Mesa 3.0 and create simple prototype
2. Validate deterministic behavior with seed control
3. Implement heterogeneous agents (predator/prey)
4. Add spatial queries and movement
5. Measure performance with 100-500 agents
6. Integrate py_trees if complex behaviors needed
7. Optimize based on profiling results
