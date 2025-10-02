# Sensor Query API Documentation

**Version**: 1.0 (Phase A+C Complete)
**Performance**: 1.526ms @ 1000 entities (24% under 2ms budget)
**Status**: Production-ready

## Overview

The sensor query API provides a cone-based spatial query system for entity detection with multiple sensory channels. AI agents use this API to perceive their environment through acoustic, optical, bioluminescent, and thermal sensing.

## Core Function: `query_cone()`

### Signature

```python
def query_cone(
    origin: np.ndarray,      # (3,) sensor position [x, y, z]
    direction: np.ndarray,   # (3,) normalized direction vector
    angle_deg: float,        # cone half-angle in degrees
    range_m: float,          # maximum detection range in meters
    flags: int = 0           # bitwise OR of QUERY_FLAG_* constants
) -> List[EntityHit]
```

### Query Flags

Flags control which fields are populated in returned `EntityHit` objects:

| Flag | Value | Fields Populated | Cost (ms @ 1000e) |
|------|-------|------------------|-------------------|
| `QUERY_FLAG_DISTANCE` | 1 << 0 | `distance` | ~0.1ms |
| `QUERY_FLAG_POSITION` | 1 << 1 | `pos_x`, `pos_y`, `pos_z` | ~0.2ms |
| `QUERY_FLAG_VELOCITY` | 1 << 2 | `vel_x`, `vel_y`, `vel_z` | ~0.2ms |
| `QUERY_FLAG_ACOUSTIC` | 1 << 3 | `acoustic_amplitude`, `acoustic_peak_hz` | ~0.2ms |
| `QUERY_FLAG_BIOLUMINESCENT` | 1 << 4 | `bioluminescent_intensity`, `bioluminescent_wavelength_nm` | ~0.2ms |
| `QUERY_FLAG_OPTICAL` | 1 << 8 | `optical_intensity`, `optical_wavelength_nm`, `optical_components` | ~0.2ms |
| `QUERY_FLAG_THERMAL` | 1 << 9 | `thermal_temperature_delta` | ~0.3ms |

**Performance Note**: Flags can be combined with bitwise OR. The `OPTICAL` and `BIOLUMINESCENT` flags share a single compute path for efficiency.

### Example Usage

```python
from aquarium.spatial_queries import (
    SpatialIndexAdapter,
    QUERY_FLAG_DISTANCE,
    QUERY_FLAG_POSITION,
    QUERY_FLAG_OPTICAL
)

# Build spatial index
adapter = SpatialIndexAdapter()
adapter.build(entities, thermal_centers, thermal_base_deltas, thermal_influences)

# Query forward cone
origin = agent.position
direction = agent.forward_vector
hits = adapter.query_cone(
    origin=origin,
    direction=direction,
    angle_deg=45.0,
    range_m=100.0,
    flags=QUERY_FLAG_DISTANCE | QUERY_FLAG_POSITION | QUERY_FLAG_OPTICAL
)

# Process results
for hit in hits:
    print(f"Detected {hit.entity_id} at distance {hit.distance}m")
    if hit.optical_intensity > 0.5:
        print(f"  Bright optical signal: {hit.optical_wavelength_nm}nm")
```

## EntityHit Data Transfer Object

### Fields

All fields are optional (None when not requested or not applicable):

#### Basic Fields
- `entity_id: str` - Always populated
- `distance: Optional[float]` - Range to entity (meters)
- `pos_x, pos_y, pos_z: Optional[float]` - Entity position (meters)
- `vel_x, vel_y, vel_z: Optional[float]` - Entity velocity (m/s)

#### Acoustic Channel
- `acoustic_amplitude: Optional[float]` - Sound amplitude [0.0, 1.0]
- `acoustic_peak_hz: Optional[float]` - Peak frequency (Hz), None if amplitude=0

#### Bioluminescent Channel
- `bioluminescent_intensity: Optional[float]` - Light intensity [0.0, 1.0]
- `bioluminescent_wavelength_nm: Optional[float]` - Wavelength (nm), None if intensity=0

#### Optical Channel (Path B+)
- `optical_intensity: Optional[float]` - Total optical intensity [0.0, 1.0]
- `optical_wavelength_nm: Optional[float]` - Combined wavelength (nm), None if intensity=0
- `optical_components: Optional[List[str]]` - Source provenance (e.g., `['bioluminescent']`)

**Path B+ Philosophy**: The optical channel currently equals bioluminescent (single source), but includes provenance metadata (`optical_components`) for future extension. This provides honest current capability with future-proof infrastructure.

#### Thermal Channel
- `thermal_temperature_delta: Optional[float]` - Temperature delta from ambient (°C)
  - `None`: No thermal providers in biome (omitted from `to_dict()`)
  - `0.0`: Thermal providers exist, but entity outside all influence radii
  - `> 0.0`: Entity within influence radius of nearest vent

### Field Omission Rules

**Sensor Reading Semantics**: Fields distinguish "no sources exist" from "no signal detected"

| Channel | Zero Signal | No Sources | Rationale |
|---------|-------------|------------|-----------|
| Acoustic | `amplitude=0, peak_hz=None` | N/A | No sound = no measurable frequency |
| Bioluminescent | `intensity=0, wavelength_nm=None` | N/A | No photons = no measurable wavelength |
| Optical | `intensity=0, wavelength_nm=None` | N/A | No photons = no measurable wavelength |
| Thermal | `thermal_delta=0.0` | `thermal_delta=None` | Helps AI distinguish environment types |

**Design Rationale**: Wavelength/frequency are physical properties that only exist when the corresponding wave/particle exists. Zero intensity means no photons detected, so wavelength is physically unmeasurable (not zero, but undefined). This helps AI agents reason correctly about sensor readings.

## Implementation Details

### Emission Baking

Entity emissions are baked at spawn time from `Species.emissions` into `Entity.base_emissions`:

```python
entity.base_emissions = {
    'acoustic': {
        'amplitude': 0.5,    # [0.0, 1.0]
        'peak_hz': 120.0     # Hz
    },
    'bioluminescent': {
        'intensity': 0.3,    # [0.0, 1.0]
        'wavelength_nm': 480.0  # nanometers
    }
}
```

Runtime behavior modulates these base values via `entity.emission_multipliers` (e.g., excited entities emit more light). Final sensor readings:

```python
final_intensity = base_emissions['bioluminescent']['intensity'] *
                  emission_multipliers['bioluminescent']
final_intensity = np.clip(final_intensity, 0.0, 1.0)
```

### Thermal Provider Extraction

Thermal providers (hydrothermal vents) are extracted once at simulation init from biome obstacles:

```python
# In Simulation.__init__()
self._extract_thermal_providers()  # Called after biome loading

# Providers stored as numpy arrays:
self._thermal_centers: np.ndarray     # (M, 3) vent positions
self._thermal_base_deltas: np.ndarray # (M,) temperature deltas
self._thermal_influences: np.ndarray  # (M,) influence radii
```

**Extraction Rules**:
- Include sphere only if `thermal_base_delta > 0` AND `influence_radius > 0`
- Fallback: `thermal_base_delta → VENT_THERMAL_BASE_DELTA` (5.0°C)
- Fallback: `influence_radius → sphere.radius × INFLUENCE_RADIUS_FACTOR_DEFAULT` (2.5)

### Thermal Computation (Vectorized)

For K entities detected in cone with M thermal providers:

```python
# Vectorized nearest-vent computation (12× optimization)
H = np.array(hit_positions, dtype=np.float64)  # (K, 3)
C = self._thermal_centers  # (M, 3)

# Compute squared distances (avoid K×M sqrt operations)
dist_sq = ((H[:,None,:] - C[None,:,:])**2).sum(axis=2)  # (K, M)

# Find nearest vent per hit (argmin works on squared distances)
nearest_idx = np.argmin(dist_sq, axis=1)  # (K,)

# Now compute actual distance for only the K nearest vents
nearest_dist = np.sqrt(dist_sq[np.arange(K), nearest_idx])  # K sqrt ops

# Get base_deltas and influences for nearest vents
base = self._thermal_base_deltas[nearest_idx]  # (K,)
infl = self._thermal_influences[nearest_idx]  # (K,)

# Linear falloff with clamping
thermal_deltas = base * (1.0 - nearest_dist / infl)
thermal_deltas = np.maximum(thermal_deltas, 0.0)
```

**Performance**: At 23 hits × 12 vents = 12× fewer sqrt operations (23 vs 276).

## Performance Characteristics

Measured @ 1000 entities, typical cone query (45° half-angle, 100m range):

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| KD-tree build | 0.300 | Once per tick, all entities |
| Cone query (distance only) | 0.508 | Step 1 baseline |
| + Acoustic channel | +0.226 | Emission lookup + scaling |
| + Bioluminescent channel | +0.226 | Emission lookup + scaling |
| + Optical channel | +0.000 | Shared compute with bioluminescent |
| + Thermal channel | +0.343 | Vectorized nearest-vent |
| **Total (all channels)** | **1.526** | **24% under 2ms budget** |

**Budget Headroom**: 0.474ms available for future channels or optimization slack.

**Scaling**: Performance is O(K log N) where K is cone result size (~20-30 entities typical), N is total entities. KD-tree build is O(N log N) once per tick.

## Future Extensions

### Phase 2: Additional Optical Sources
When artificial lights and reflections are added:

```python
# optical_components will expand
optical_components = ['bioluminescent', 'artificial', 'reflected']

# Wavelength blending (SD guidance)
# Multiple sources: weighted average by intensity
# Single source: use source wavelength directly
```

### Phase 3: Attenuation
Distance-based signal attenuation for realism:
- Acoustic: inverse square law with medium absorption
- Optical: inverse square with scattering
- Thermal: already has linear falloff per vent

### Phase 4: Occlusion
Obstacle blocking for line-of-sight:
- Ray-cast check against biome geometry
- Flag: `QUERY_FLAG_CHECK_OCCLUSION`
- Cost: +0.5ms estimated (needs profiling)

## Testing

Test suite: `aquarium/tests/test_sensor_queries.py` (22 tests, all passing)

### Coverage
- **Step 1 (8 tests)**: Cone geometry, flag presence, determinism, performance
- **Phase A (7 tests)**: Acoustic, bioluminescent, optical flags, multiplier scaling, wavelength omission
- **Phase C (4 tests)**: Thermal falloff, nearest vent, M=0 omission, outside influence
- **Performance (3 tests)**: 143/500/1000 entity scaling validation

### Run Tests
```bash
python -m pytest aquarium/tests/test_sensor_queries.py -v
```

Expected: 22 passed in ~0.5s

## Troubleshooting

### EntityHit fields are None
- **Cause**: Flag not set in query
- **Fix**: Add appropriate `QUERY_FLAG_*` to flags parameter

### optical_wavelength_nm is None despite optical_intensity > 0
- **Bug**: This should never happen (intensity > 0 implies wavelength exists)
- **Report**: Check `_compute_optical_components()` logic

### thermal_temperature_delta always None
- **Cause**: No thermal providers in biome (M=0)
- **Check**: `Simulation._extract_thermal_providers()` found no vents
- **Verify**: Biome YAML has spheres with `thermal_base_delta > 0`

### Performance degradation
- **Target**: <2ms @ 1000 entities with all flags
- **Check**: Number of hits in cone (should be ~20-30 typical)
- **Profile**: If >50 hits, consider narrowing cone angle or range

## References

- **Handoff**: `project/handoffs/session_009_2025-10-02_phase_ac_complete_gossip_next.md`
- **Implementation**: `aquarium/spatial_queries.py:815-1057`
- **Tests**: `aquarium/tests/test_sensor_queries.py`
- **Thermal Extraction**: `aquarium/simulation.py:153-203`

---

**Authors**: Chronus (Sessions 008-009), Senior Dev (design guidance), Mike (requirements)
**Last Updated**: 2025-10-02
**Status**: Production-ready, all 22 tests passing
