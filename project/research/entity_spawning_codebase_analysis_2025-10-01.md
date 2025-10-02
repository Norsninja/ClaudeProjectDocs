# Entity Spawning Research - 2025-10-01

## Executive Summary

Entity spawning occurs in `spawning.py` through the `_spawn_species()` function, which has direct access to the complete Species object including emissions data. Currently, only 4 fields are copied from Species to Entity at spawn time (species_id, tags, size_factor, and initial velocity). Species.emissions data is NOT pre-baked into Entity instances. Pre-baking base emissions at spawn time (Option C) is completely feasible with zero architectural barriers.

## Scope

Investigation focused on:
- Entity creation flow and data copying patterns
- Species registry access at spawn time
- Current fields baked into Entity from Species
- Feasibility of adding base_emissions field to Entity
- Sensor query implementation expecting emission data

## Key Findings

### Pattern Analysis

**Current Data Flow Pattern:**
```
YAML Species → Species Registry (Dict[str, Species]) → Spawning → Entity
```

The spawning system follows a clear pattern:
1. Species loaded from YAML into Species dataclasses (`loader.py`)
2. Species stored in `species_registry` dict (key: species_id)
3. Spawning retrieves Species object from registry
4. Spawning creates Entity, copying select Species fields

**Species Registry Access:**
File: `aquarium/spawning.py`
Lines: 18-96

The `spawn_entities()` function receives `species_registry: Dict[str, Species]` as a parameter and passes it to `_spawn_species()`, providing full Species object access during entity creation.

### Implementation Details

#### Spawning Call Chain

1. **Entry Point:** `simulation.py::_spawn_all_biomes()` (lines 104-136)
```python
spawned = spawn_entities(
    biome=biome,
    species_registry=self.species_registry,  # Full registry passed
    world_seed=world_seed,
    limit=remaining_limit,
    only_species=only_species
)
```

2. **Per-Species Spawning:** `spawning.py::spawn_entities()` (lines 18-96)
```python
species = species_registry[species_id]  # Species object retrieved

spawned = _spawn_species(
    biome=biome,
    species=species,  # Full Species object passed
    count=count,
    distribution=spawn_config.distribution,
    world_seed=world_seed,
    center=center,
    radius=radius
)
```

3. **Entity Creation:** `spawning.py::_spawn_species()` (lines 99-170)
```python
# Line 158-166: Entity creation with Species data access
entity = Entity(
    instance_id=instance_id,
    species_id=species.species_id,
    biome_id=biome.biome_id,
    position=position,
    velocity=velocity,
    size_factor=size_factor,
    tags=species.tags.copy()  # Species tags copied here
)
```

**Critical Observation:** The Species object is available at line 158 where Entity is constructed. `species.emissions` is fully accessible.

#### Current Data Copied from Species to Entity

File: `aquarium/spawning.py`
Lines: 158-166

Fields currently baked into Entity at spawn:
1. `species_id` - Direct copy
2. `tags` - Shallow copy via `.copy()`
3. `size_factor` - Generated from Species.physical.size_range
4. `velocity` - Calculated from Species.movement.max_speed_ms

**Pattern Established:** Entity already receives "baked" data from Species at spawn time (tags, derived size_factor). Adding base_emissions would follow this exact pattern.

#### Species Emissions Data Structure

File: `aquarium/data_types.py`
Lines: 17-24

```python
@dataclass
class EmissionProfile:
    """Base emission profile for a species"""
    acoustic: Optional[Dict[str, float]] = None  # {peak_hz, bandwidth_hz, amplitude}
    thermal: Optional[Dict[str, float]] = None  # {delta_celsius}
    chemical: Optional[Dict[str, float]] = None  # {compound, concentration}
    magnetic: Optional[Dict[str, float]] = None  # {delta_microtesla}
    bioluminescent: Optional[Dict[str, float]] = None  # {intensity, wavelength_nm}
```

Example from `data/species/sp-001-drifter.yaml` (lines 33-45):
```yaml
emissions:
  acoustic:
    peak_hz: 1000
    bandwidth_hz: 400
    amplitude: 0.6
  thermal:
    delta_celsius: 0.1
  chemical:
    compound: organic_trace
    concentration: 0.2
  bioluminescent:
    intensity: 0.4
    wavelength_nm: 480
```

#### Sensor Query Implementation (Phase 6)

File: `aquarium/spatial_queries.py`
Lines: 815-962

The `query_cone()` method is partially implemented:

```python
# Lines 947-948: TODO comment for emission channels
# TODO: Phase 1b - Acoustic/Bioluminescent/Thermal channels (Step 2 & 3)
# For now, these channels are not populated (Step 1: basic fields only)
```

Current implementation:
- Step 1 COMPLETE: Basic fields (POSITION, VELOCITY, DISTANCE)
- Step 2 PENDING: Emission channels (ACOUSTIC, BIOLUMINESCENT, THERMAL)

**Critical for Implementation:** Sensor queries build EntityHit DTOs (lines 922-950) during cone queries. This is where emission data needs to be accessed.

#### Entity Runtime Representation

File: `aquarium/entity.py`
Lines: 13-65

Current Entity fields:
```python
@dataclass
class Entity:
    instance_id: str
    species_id: str
    biome_id: str
    position: np.ndarray
    velocity: np.ndarray
    size_factor: float
    tags: list
    active_behavior_id: str = None
    emission_multipliers: dict = None  # {channel: multiplier}
    knowledge_tokens: dict = None
```

**Emission System Design (Existing):**
- `emission_multipliers` dict initialized in `__post_init__` (lines 52-60)
- Default multipliers: all channels set to 1.0 (baseline)
- Modified by behavior actions (flee = 0.3 acoustic, investigate = 2.0 bioluminescent)

**Missing Component:** Base emission values (Species.emissions) are NOT stored in Entity.

### Code Flow

**Current Emission Access Pattern:**
1. Sensor query runs during tick (not yet implemented for emission channels)
2. Would need to look up Species object from species_registry using entity.species_id
3. Read Species.emissions
4. Apply Entity.emission_multipliers
5. Calculate final values

**Problems with Current Pattern:**
- Requires species_registry access during sensor queries
- Dictionary lookup per entity: `species_registry[entity.species_id]`
- Increases coupling between spatial queries and simulation state
- Performance cost: 1000 entities × dict lookup overhead

**Proposed Pattern (Option C - Pre-bake):**
1. At spawn time, copy Species.emissions → Entity.base_emissions
2. During sensor query, read Entity.base_emissions directly
3. Apply Entity.emission_multipliers
4. Calculate final values

**Performance Benefit:**
- Zero runtime lookups
- No species_registry dependency in sensor queries
- Data locality: emissions stored with entity position/velocity

### Related Components

#### Behavior System Integration

File: `aquarium/behavior.py`
Lines: 83-95

Behaviors already modify emission_multipliers:
```python
behavior_id, velocity, emissions = evaluate_behavior(entity, species, entities, 10.0, spatial, cache)

# Line 93: Emission multipliers returned from behavior
emissions = behavior.action.emission_multipliers or {}
```

File: `aquarium/behavior.py`
Lines: 344-360

Entity.emission_multipliers updated during behavior evaluation:
```python
behavior_id, velocity, emissions = evaluate_behavior(...)

# Lines 358-359: Apply emission multipliers
for channel, multiplier in emissions.items():
    entity.emission_multipliers[channel] = multiplier
```

**Integration Point:** Sensor queries would multiply `base_emissions × emission_multipliers` to get final values.

#### Fallback Emission Values

File: `aquarium/constants.py`
Lines: 55-61

Constants defined for missing emission data:
```python
ACOUSTIC_DEFAULT_AMPLITUDE = 0.1
ACOUSTIC_DEFAULT_PEAK_HZ = 50.0
BIOLUM_DEFAULT_INTENSITY = 0.05
BIOLUM_DEFAULT_WAVELENGTH = 480.0
VENT_THERMAL_BASE_DELTA = 5.0
```

**Use Case:** When Species.emissions.acoustic is None, use fallback values during baking.

## File Inventory

### Primary Spawning Files
- `aquarium/spawning.py` - Entity spawning logic (158 lines)
  - `spawn_entities()` - Entry point, calls per-species spawning
  - `_spawn_species()` - Creates Entity instances, **TARGET for base_emissions baking**
  - `_spawn_uniform()` - Position generation helper

### Data Loading Files
- `aquarium/loader.py` - YAML to dataclass conversion (277 lines)
  - `load_species()` - Creates Species objects with EmissionProfile
  - `load_species_registry()` - Builds species_registry dict
  - `load_all_data()` - Simulation initialization entry point

### Entity Definition Files
- `aquarium/entity.py` - Entity runtime representation (127 lines)
  - `Entity` dataclass - **TARGET for new base_emissions field**
  - `__post_init__()` - Initializes emission_multipliers dict
  - `get_emission_multipliers()` - Returns current multipliers

### Data Type Definitions
- `aquarium/data_types.py` - Schema dataclasses (347 lines)
  - `EmissionProfile` - Species base emissions structure (lines 17-24)
  - `Species` - Contains EmissionProfile (line 88)
  - `EntityHit` - Sensor query result DTO (lines 281-327)

### Sensor Query Implementation
- `aquarium/spatial_queries.py` - Spatial indexing and sensor queries (978 lines)
  - `SpatialIndexAdapter.query_cone()` - Cone sensor implementation (lines 815-962)
  - **Line 947-948:** TODO for emission channels (Step 2 pending)
  - EntityHit construction happens at lines 922-950

### Simulation Orchestration
- `aquarium/simulation.py` - Main simulation loop (545 lines)
  - `_spawn_all_biomes()` - Calls spawn_entities() (lines 104-136)
  - Stores `self.species_registry` for behavior evaluation

### Behavior System
- `aquarium/behavior.py` - Behavior evaluation and emission multipliers
  - Modifies Entity.emission_multipliers based on active behavior
  - Already integrates with emission system

### Configuration
- `aquarium/constants.py` - Emission fallback values (90 lines)
  - Defines defaults for missing Species.emissions data

## Technical Notes

### For Technical Planner

**Exact Files Requiring Modification:**

1. **`aquarium/entity.py`** (lines 13-39)
   - Add `base_emissions: dict = None` field to Entity dataclass
   - Modify `__post_init__()` to initialize base_emissions if None

2. **`aquarium/spawning.py`** (lines 158-166)
   - Add emission baking logic in `_spawn_species()` after line 156 (size_factor generation)
   - Copy Species.emissions to dict, apply fallback values for None channels
   - Pass base_emissions to Entity constructor

3. **`aquarium/spatial_queries.py`** (lines 922-950)
   - Implement emission channel population in EntityHit construction
   - Access entity.base_emissions instead of species_registry lookup
   - Apply entity.emission_multipliers for behavior modification
   - Populate EntityHit fields based on QUERY_FLAG_* bitmask

**Current Pattern to Follow:**

Tags are already baked at spawn (line 165):
```python
tags=species.tags.copy()  # Copy species tags
```

Follow this exact pattern for emissions:
```python
base_emissions=_bake_emissions(species.emissions)
```

**Integration Points with Line Numbers:**

1. **Spawning (emission baking):**
   - File: `aquarium/spawning.py`
   - Function: `_spawn_species()`
   - Insert after: Line 156 (size_factor generation)
   - Insert before: Line 158 (Entity constructor)

2. **Sensor Queries (emission reading):**
   - File: `aquarium/spatial_queries.py`
   - Function: `query_cone()`
   - Location: Lines 947-948 (existing TODO)
   - Context: Inside EntityHit construction loop (lines 922-950)

3. **Entity Definition (field addition):**
   - File: `aquarium/entity.py`
   - Class: `Entity`
   - Location: After line 35 (tags field)
   - Also modify: `__post_init__()` line 52-60 block

**Existing Conventions to Maintain:**

1. **Defensive Copying:** Use `.copy()` for mutable objects (see tags at line 165)
2. **Float64 Arrays:** All numpy arrays are float64 (position/velocity pattern)
3. **None Checks:** Initialize defaults in `__post_init__()` if field is None
4. **Fallback Values:** Use constants from `constants.py` for missing data
5. **Dict Field Naming:** Follow existing pattern (emission_multipliers → base_emissions)

**No Architectural Barriers:**

- ✓ Species object available at spawn time
- ✓ Pattern established for baking data (tags)
- ✓ Entity already has emission_multipliers dict
- ✓ Sensor queries already designed for emission fields (EntityHit DTOs ready)
- ✓ No async loading or lazy initialization
- ✓ No circular dependencies

**Performance Considerations:**

- Baking happens once per entity at spawn (1000 entities = 1000 bakes, one-time cost)
- Runtime: Zero species_registry lookups during sensor queries
- Memory: ~200 bytes per entity for base_emissions dict (negligible)
- Cache efficiency: Improved (emissions co-located with entity data)

## Recommendation

**Option C (pre-bake base_emissions at spawn) is HIGHLY FEASIBLE and RECOMMENDED.**

**Rationale:**
1. Species.emissions is available at spawn time (line 158 in spawning.py)
2. Established pattern exists for baking Species data into Entity (tags)
3. Zero architectural barriers or coupling issues
4. Performance benefit: eliminates runtime species_registry lookups
5. Aligns with existing emission_multipliers system design
6. Sensor query implementation expects emission data in EntityHit (DTOs ready)

**Implementation Complexity:** LOW
- Add 1 field to Entity dataclass
- Add 1 helper function to bake emissions dict
- Add emission access code in sensor queries (TODO already marked)

**Risk Assessment:** MINIMAL
- No breaking changes to existing systems
- Behavior system already uses emission_multipliers
- Follows established Entity initialization patterns
- Test coverage exists for sensor queries (test_sensor_queries.py)
