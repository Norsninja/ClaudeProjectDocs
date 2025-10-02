# Spatial Index Adapter API

Purpose
- Provide a stable interface for nearest/neighbor queries and sensor-facing cone queries.
- Phase 3 fallback uses O(n) scans; Phase 4 swaps in scipy.cKDTree without changing call sites.
- Enforce same-biome scope, deterministic tie-breaking, and flat DTO with flags.

Conventions
- Units: meters. `y` is vertical (depth, negative below surface).
- Same-biome scope: queries only consider entities within the caller’s biome.
- Determinism: stable iteration order; tie-breaks by `instance_id` then `entity_id`.

Types
- EntityRef
  - `entity_id: str`
  - `instance_id: int`
  - `species_id: str`
  - `tags: set[str]`
  - `pos: np.ndarray` (shape (3,), float64)
  - `vel: np.ndarray` (shape (3,), float64)

- QueryFlags (bitmask)
  - `POSITION = 1 << 0`
  - `VELOCITY = 1 << 1`
  - `DISTANCE = 1 << 2`
  - `ACOUSTIC = 1 << 3`
  - `THERMAL = 1 << 4`
  - `CHEMICAL = 1 << 5`
  - `MAGNETIC = 1 << 6`
  - `BIOLUMINESCENT = 1 << 7`

- EntityHit (flat DTO)
  - `entity_id: str`
  - `species_id: str`
  - Optional (per flags):
    - `pos_x, pos_y, pos_z: float`
    - `vel_x, vel_y, vel_z: float`
    - `distance: float`
    - `acoustic_frequency: float`, `acoustic_amplitude: float`
    - `thermal_temperature_delta: float`
    - `chemical_concentration: float`
    - `magnetic_field_strength: float`
    - `bioluminescent_intensity: float`, `bioluminescent_wavelength: float`

- SpatialQueryResult
  - `entities: list[EntityHit]`
  - `timestamp: float`
  - `query_origin: tuple[float, float, float]`

Interface
- Class: `SpatialIndexAdapter`
  - `build(entities: list[EntityRef]) -> None`
    - Build or rebuild the spatial index (Phase 3: noop; Phase 4: cKDTree).
  - `update_entity(e: EntityRef) -> None`
    - Optional incremental update (Phase 4+: may rebuild lazily or batch).
  - `find_nearest_by_tag(source: EntityRef, tag: str, max_distance: float | None = None) -> EntityRef | None`
    - Same-biome search; fallback to O(n) if index absent.
  - `neighbors_within(source: EntityRef, radius: float, tag_filter: set[str] | None = None) -> list[EntityRef]`
    - Same-biome search; returns stable-sorted results by distance then `instance_id`.
  - `query_cone(origin: tuple[float,float,float], direction: tuple[float,float,float], angle_deg: float, range_m: float, flags: int) -> SpatialQueryResult`
    - Sensor-facing cone query; returns flat DTO filtered by `flags`.

Behavior & Tie-breaking
- Distances computed in double precision; equal-distance ties resolved by `instance_id` then `entity_id`.
- Caller’s biome ID must be provided or inferred; cross-biome queries return no results.

Performance Targets
- Build time (500 entities): < 3 ms.
- Nearest query: O(log N) with cKDTree, typical < 0.2 ms.
- Neighbors within r: query_ball_point with post-filtering by tags; typical < 0.5 ms.

Migration Plan
- Phase 3: adapter delegates to naive scans; signatures stable.
- Phase 4: adapter uses cKDTree under-the-hood; zero call site changes.

Notes
- Sensor DTO uses integer `flags` bitmask for payload minimization.
- Emission channel values may be scaled by per-entity emission multipliers.
