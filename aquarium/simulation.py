"""
Aquarium simulation kernel.

Main simulation class that manages entity lifecycle, tick loop, and physics.
Phase 3: Added behavior evaluation and ship entity support.
"""

import numpy as np
import os
import time
from typing import Dict, List, Optional
from pathlib import Path

from .entity import Entity
from .data_types import World, Biome, Species
from .spawning import spawn_entities
from .spatial import distance_to_sphere, reflect_velocity, normalize, clamp_speed
from .loader import load_all_data
from .behavior import update_entity_behavior
from .constants import (
    CRUISE_SPEED_FRACTION,
    TICK_TIME_WINDOW,
    USE_CKDTREE,
    CKDTREE_LEAFSIZE,
    VENT_THERMAL_BASE_DELTA,
    INFLUENCE_RADIUS_FACTOR_DEFAULT,
    USE_GOSSIP,
    GOSSIP_LOG_INTERVAL,
    USE_ECOSYSTEM,
    ECOSYSTEM_LOG_INTERVAL,
    RESOURCE_DEFAULT_BASE,
    RESOURCE_DEFAULT_PEAK,
    RESOURCE_DEFAULT_SIGMA,
    ENERGY_DRAIN_DEFAULT,
    MOVEMENT_COST_FACTOR_DEFAULT,
    FEEDING_EFFICIENCY_DEFAULT,
    INTAKE_RATE_DEFAULT,
)
from .spatial_queries import SpatialIndexAdapter
from .gossip import exchange_tokens


class EnergyView:
    """
    Read-only view into entity energy state for behavior evaluation.

    Provides safe access to current energy and starvation thresholds without
    exposing full SoA arrays. Constructed once per Phase A, passed via
    behavior_context to avoid per-entity allocations.

    Phase 8: Ecosystem foundation (hunger-driven behaviors).
    """

    def __init__(self, row_of_id: Dict[str, int], energy_current: np.ndarray, starvation_threshold: np.ndarray):
        """
        Args:
            row_of_id: Spatial index mapping instance_id -> array row
            energy_current: (N,) array of current energy values
            starvation_threshold: (N,) array of per-entity death thresholds
        """
        self._row_of_id = row_of_id
        self._energy_current = energy_current
        self._starvation_threshold = starvation_threshold

    def get_energy(self, instance_id: str) -> float:
        """
        Get current energy for entity.

        Returns:
            Current energy value, or 0.0 if instance_id not found
        """
        row = self._row_of_id.get(instance_id, None)
        if row is None:
            return 0.0
        return float(self._energy_current[row])

    def get_threshold(self, instance_id: str) -> float:
        """
        Get starvation threshold for entity.

        Returns:
            Starvation threshold, or 0.0 if instance_id not found
        """
        row = self._row_of_id.get(instance_id, None)
        if row is None:
            return 0.0
        return float(self._starvation_threshold[row])


class AquariumSimulation:
    """
    Main simulation class for aquarium ecosystem.

    Manages entity lifecycle, tick loop, and physics simulation.
    Phase 2: Minimal tick loop with movement and bounds reflection.
    """

    def __init__(
        self,
        data_root: Path,
        schema_dir: Optional[Path] = None,
        spawn_limit: Optional[int] = None,
        spawn_only_species: Optional[List[str]] = None
    ):
        """
        Initialize simulation from data pack.

        Args:
            data_root: Path to data directory
            schema_dir: Optional path to JSON schemas
            spawn_limit: Optional limit on spawned entities (for testing)
            spawn_only_species: Optional list of species to spawn (for testing)
        """
        # Load data pack
        print("Loading data pack...")
        data = load_all_data(data_root, schema_dir)

        self.world: World = data['world']
        self.biomes: Dict[str, Biome] = data['biomes']
        self.species_registry: Dict[str, Species] = data['species']
        self.token_definitions = data['tokens']
        self.interaction_rules = data['interactions']

        # Simulation state
        self.entities: List[Entity] = []
        self.ship_entity: Optional[Entity] = None  # Player ship (inert entity)
        self.tick_count: int = 0
        self.dt: float = self.world.simulation.tick_delta_seconds

        # Performance metrics
        self._tick_times: List[float] = []
        self._tick_time_sum: float = 0.0
        self._tick_time_window: int = TICK_TIME_WINDOW  # Rolling average window

        # Phase timing breakdown (Phase 4b)
        self._build_times: List[float] = []
        self._behavior_times: List[float] = []
        self._movement_times: List[float] = []

        # Phase 5 timing breakdown
        self._batch_query_times: List[float] = []

        # Phase A.5 timing (avoidance)
        self._avoidance_times: List[float] = []
        self._avoidance_sphere_times: List[float] = []
        self._avoidance_cylinder_times: List[float] = []
        self._avoidance_plane_times: List[float] = []

        # Phase B.5 timing (gossip)
        self._gossip_times: List[float] = []
        self._gossip_telemetry: Optional[Dict] = None  # Latest telemetry from exchange_tokens

        # Phase 6 timing (sensor queries)
        self._sensor_times: List[float] = []

        # Spatial indexing (Phase 4b)
        self.spatial = SpatialIndexAdapter(use_ckdtree=USE_CKDTREE)
        self._positions: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._capacity: int = 0
        # _count tracks live simulation entities only (excludes ship)
        # Invariant: _count == len(self.entities) at end of every tick
        # Spatial builds use local build_count which may include ship
        self._count: int = 0

        # Phase 5: Batch query cache control
        self._use_batch_cache: bool = True  # Set False for A/B testing

        # Phase C: Thermal providers (cached, extracted once)
        self._thermal_centers: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._thermal_base_deltas: np.ndarray = np.empty(0, dtype=np.float64)
        self._thermal_influences: np.ndarray = np.empty(0, dtype=np.float64)

        # Phase 8+: Ecosystem - Resource field providers (cached, extracted once)
        self._resource_centers: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._resource_peaks: np.ndarray = np.empty(0, dtype=np.float64)
        self._resource_sigmas: np.ndarray = np.empty(0, dtype=np.float64)
        self._resource_base_density: float = RESOURCE_DEFAULT_BASE

        # Phase 8+: Ecosystem - SoA energy arrays (hot path)
        self._energy_current: np.ndarray = np.empty(0, dtype=np.float64)
        self._energy_max: np.ndarray = np.empty(0, dtype=np.float64)
        self._alive: np.ndarray = np.empty(0, dtype=bool)
        self._last_feed_tick: np.ndarray = np.empty(0, dtype=np.int32)

        # Phase 8+: Ecosystem - SoA metabolism arrays (precomputed per-row from species)
        self._base_drain_per_tick: np.ndarray = np.empty(0, dtype=np.float64)
        self._move_cost_factor: np.ndarray = np.empty(0, dtype=np.float64)
        self._intake_rate_per_tick: np.ndarray = np.empty(0, dtype=np.float64)
        self._feeding_efficiency: np.ndarray = np.empty(0, dtype=np.float64)
        self._starvation_threshold: np.ndarray = np.empty(0, dtype=np.float64)

        # Phase 8+: Ecosystem timing breakdown
        self._energy_drain_times: List[float] = []
        self._feeding_times: List[float] = []
        self._death_times: List[float] = []

        # Phase 8+: Ecosystem telemetry
        self._ecosystem_telemetry: Dict = {
            'total_deaths': 0,
            'deaths_this_tick': 0,
            'mean_energy': 0.0,
            'min_energy': 0.0,
            'starving_count': 0,
            'mean_density_sampled': 0.0,
        }

        # Spawn entities
        print(f"Spawning entities (limit={spawn_limit}, species={spawn_only_species})...")
        self._spawn_all_biomes(spawn_limit, spawn_only_species)

        # Initialize position array capacity
        self._ensure_position_capacity(len(self.entities))

        # Initialize _count to match spawned entities (after capacity ensured)
        self._count = len(self.entities)

        # Extract thermal providers from biomes (cached for all ticks)
        self._extract_thermal_providers()

        # Extract resource field providers from biomes (Phase 8+: Ecosystem)
        if USE_ECOSYSTEM:
            self._extract_resource_providers()
            self._initialize_energy_arrays()

        print(f"[OK] Simulation initialized: {len(self.entities)} entities, "
              f"dt={self.dt}s, seed={self.world.parameters.seed}")

    def _spawn_all_biomes(self, limit: Optional[int], only_species: Optional[List[str]]):
        """
        Spawn entities in all active biomes.

        Args:
            limit: Optional global entity limit
            only_species: Optional species filter
        """
        world_seed = self.world.parameters.seed
        total_spawned = 0

        for biome_id, biome in self.biomes.items():
            # Calculate remaining limit
            remaining_limit = None
            if limit is not None:
                remaining_limit = limit - total_spawned

                if remaining_limit <= 0:
                    break

            # Spawn entities in this biome
            spawned = spawn_entities(
                biome=biome,
                species_registry=self.species_registry,
                world_seed=world_seed,
                limit=remaining_limit,
                only_species=only_species
            )

            self.entities.extend(spawned)
            total_spawned += len(spawned)

            print(f"  {biome.name}: spawned {len(spawned)} entities")

    def _extract_thermal_providers(self):
        """
        Extract thermal providers from biome obstacles (spheres only).

        Processes all sphere obstacles and builds numpy arrays for thermal sensors.
        Providers are filtered to include only those with positive thermal_base_delta
        and positive influence_radius.

        Arrays are cached for the lifetime of the simulation (obstacles are static).

        Sets:
            self._thermal_centers: (M, 3) array of vent centers
            self._thermal_base_deltas: (M,) array of base temperature deltas (°C)
            self._thermal_influences: (M,) array of influence radii (meters)
        """
        thermal_centers = []
        thermal_base_deltas = []
        thermal_influences = []

        # Process all biomes
        for biome in self.biomes.values():
            # Extract thermal providers from sphere obstacles
            for sphere in biome.obstacles.get('spheres', []):
                # Get base_delta with fallback
                base_delta = sphere.thermal_base_delta if sphere.thermal_base_delta is not None \
                             else VENT_THERMAL_BASE_DELTA

                # Get influence_radius with fallback
                if sphere.influence_radius is not None:
                    influence = sphere.influence_radius
                else:
                    influence = sphere.radius * INFLUENCE_RADIUS_FACTOR_DEFAULT

                # Include only if both are positive (avoid divide-by-zero, meaningless providers)
                if base_delta > 0 and influence > 0:
                    thermal_centers.append(sphere.center)
                    thermal_base_deltas.append(base_delta)
                    thermal_influences.append(influence)

        # Convert to numpy arrays (empty arrays if no providers)
        if thermal_centers:
            self._thermal_centers = np.array(thermal_centers, dtype=np.float64)
            self._thermal_base_deltas = np.array(thermal_base_deltas, dtype=np.float64)
            self._thermal_influences = np.array(thermal_influences, dtype=np.float64)
            print(f"  Thermal providers: {len(thermal_centers)} vents extracted")
        else:
            # Empty arrays (M=0)
            self._thermal_centers = np.empty((0, 3), dtype=np.float64)
            self._thermal_base_deltas = np.empty(0, dtype=np.float64)
            self._thermal_influences = np.empty(0, dtype=np.float64)
            print("  Thermal providers: none found")

    def _extract_resource_providers(self):
        """
        Extract resource field providers from biome obstacles (spheres only).

        Processes sphere obstacles that have resource_peak attribute and builds
        numpy arrays for plankton density sampling. Uses nearest-vent Gaussian model.

        Providers are filtered to include only those with positive peak density
        and positive sigma (falloff distance).

        Arrays are cached for the lifetime of the simulation (obstacles are static).

        Sets:
            self._resource_centers: (R, 3) array of resource patch centers
            self._resource_peaks: (R,) array of peak densities (units/m³)
            self._resource_sigmas: (R,) array of Gaussian falloff distances (meters)
            self._resource_base_density: Background density everywhere (scalar)
        """
        resource_centers = []
        resource_peaks = []
        resource_sigmas = []

        # Process all biomes
        for biome in self.biomes.values():
            # Extract resource providers from sphere obstacles
            for sphere in biome.obstacles.get('spheres', []):
                # Check if sphere has resource_peak attribute
                peak = getattr(sphere, 'resource_peak', None)
                if peak is None or peak <= 0:
                    continue  # Skip non-resource spheres

                # Get sigma (falloff distance) with fallback
                sigma = getattr(sphere, 'resource_sigma', None)
                if sigma is None:
                    sigma = RESOURCE_DEFAULT_SIGMA

                # Include only if both are positive
                if peak > 0 and sigma > 0:
                    resource_centers.append(sphere.center)
                    resource_peaks.append(peak)
                    resource_sigmas.append(sigma)

        # Convert to numpy arrays (empty arrays if no providers)
        if resource_centers:
            self._resource_centers = np.array(resource_centers, dtype=np.float64)
            self._resource_peaks = np.array(resource_peaks, dtype=np.float64)
            self._resource_sigmas = np.array(resource_sigmas, dtype=np.float64)
            print(f"  Resource providers: {len(resource_centers)} fields extracted")
        else:
            # Empty arrays (R=0)
            self._resource_centers = np.empty((0, 3), dtype=np.float64)
            self._resource_peaks = np.empty(0, dtype=np.float64)
            self._resource_sigmas = np.empty(0, dtype=np.float64)
            print("  Resource providers: none found (entities won't feed)")

    def _sample_plankton_density(self, positions: np.ndarray) -> np.ndarray:
        """
        Sample plankton density at positions using nearest-vent Gaussian falloff.

        Args:
            positions: (N, 3) array of sample positions

        Returns:
            (N,) array of densities at each position

        Algorithm:
            For each position, find nearest resource provider and compute:
            density = base + peak * exp(-dist^2 / (2*sigma^2))

            Vectorized using broadcasting for O(N*R) distance computation.
        """
        N = len(positions)

        # Handle no resource providers case
        if len(self._resource_centers) == 0:
            return np.full(N, self._resource_base_density, dtype=np.float64)

        # Compute squared distances: (N, R) array
        # positions: (N, 3), resource_centers: (R, 3)
        diff = positions[:, np.newaxis, :] - self._resource_centers[np.newaxis, :, :]  # (N, R, 3)
        dist_sq = np.sum(diff ** 2, axis=2)  # (N, R)

        # Find nearest provider for each position
        nearest_idx = np.argmin(dist_sq, axis=1)  # (N,) - index of nearest provider
        nearest_dist_sq = dist_sq[np.arange(N), nearest_idx]  # (N,) - squared distance to nearest

        # Get parameters of nearest provider
        nearest_peak = self._resource_peaks[nearest_idx]  # (N,)
        nearest_sigma = self._resource_sigmas[nearest_idx]  # (N,)

        # Gaussian falloff: peak * exp(-dist^2 / (2*sigma^2))
        gaussian = nearest_peak * np.exp(-nearest_dist_sq / (2 * nearest_sigma ** 2))

        # Total density = base + Gaussian contribution
        density = self._resource_base_density + gaussian

        return density

    def _rebuild_metabolism_arrays(self):
        """
        Rebuild per-row metabolism parameter arrays from species config.

        Called during initialization and after compaction (when population changes).
        Precomputes species-specific parameters to avoid per-tick lookups.

        Arrays built:
            - self._base_drain_per_tick[N]: Passive metabolic drain
            - self._move_cost_factor[N]: Movement energy cost scaling
            - self._intake_rate_per_tick[N]: Max resource consumption per tick
            - self._feeding_efficiency[N]: Energy gained per resource unit
        """
        from .constants import (
            ENERGY_DRAIN_DEFAULT,
            MOVEMENT_COST_FACTOR_DEFAULT,
            INTAKE_RATE_DEFAULT,
            FEEDING_EFFICIENCY_DEFAULT
        )

        N = len(self.entities)
        if N == 0:
            return

        # Allocate arrays
        self._base_drain_per_tick = np.empty(N, dtype=np.float64)
        self._move_cost_factor = np.empty(N, dtype=np.float64)
        self._intake_rate_per_tick = np.empty(N, dtype=np.float64)
        self._feeding_efficiency = np.empty(N, dtype=np.float64)
        self._starvation_threshold = np.empty(N, dtype=np.float64)

        # Fill from species config
        for i, entity in enumerate(self.entities):
            species = self.species_registry.get(entity.species_id)

            # Metabolism parameters
            if species and species.metabolism:
                self._base_drain_per_tick[i] = species.metabolism.energy_drain_per_tick
                self._move_cost_factor[i] = species.metabolism.movement_cost_factor
                self._starvation_threshold[i] = species.metabolism.starvation_threshold
            else:
                self._base_drain_per_tick[i] = ENERGY_DRAIN_DEFAULT
                self._move_cost_factor[i] = MOVEMENT_COST_FACTOR_DEFAULT
                self._starvation_threshold[i] = 0.0  # Default: die at zero energy

            # Feeding parameters
            if species and species.feeding:
                self._intake_rate_per_tick[i] = species.feeding.intake_rate_per_tick
            else:
                self._intake_rate_per_tick[i] = INTAKE_RATE_DEFAULT

            # Feeding efficiency comes from metabolism (energy conversion)
            if species and species.metabolism:
                self._feeding_efficiency[i] = species.metabolism.feeding_efficiency
            else:
                self._feeding_efficiency[i] = FEEDING_EFFICIENCY_DEFAULT

    def _initialize_energy_arrays(self):
        """
        Initialize energy SoA arrays from species metabolism config.

        Called once after spawning completes. Sets initial energy to max capacity
        and builds metabolism parameter arrays.
        """
        from .constants import ENERGY_MAX_DEFAULT

        N = len(self.entities)
        if N == 0:
            return

        # Allocate energy arrays
        self._energy_current = np.empty(N, dtype=np.float64)
        self._energy_max = np.empty(N, dtype=np.float64)
        self._alive = np.ones(N, dtype=bool)
        self._last_feed_tick = np.zeros(N, dtype=np.int32)

        # Fill from species config
        for i, entity in enumerate(self.entities):
            species = self.species_registry.get(entity.species_id)
            if species and species.metabolism:
                self._energy_max[i] = species.metabolism.energy_max
                self._energy_current[i] = species.metabolism.energy_max  # Start at full energy
            else:
                # Fallback to defaults
                self._energy_max[i] = ENERGY_MAX_DEFAULT
                self._energy_current[i] = ENERGY_MAX_DEFAULT

        # Build metabolism parameter arrays
        self._rebuild_metabolism_arrays()

    def _apply_energy_drain(self):
        """
        Phase A.75: Apply energy drain based on metabolism and movement.

        Drains energy from alive entities based on:
        - Base metabolic drain (passive energy cost)
        - Movement cost (proportional to velocity magnitude)

        Uses pre-movement velocities (Phase A finalized velocities).
        """
        N = len(self.entities)
        if N == 0:
            return

        # Alive mask (only drain alive entities)
        alive_mask = self._alive[:N]

        # Extract velocities from entities
        velocities = np.array([entity.velocity for entity in self.entities], dtype=np.float64)

        # Compute speed (magnitude of velocity)
        speeds = np.linalg.norm(velocities, axis=1)

        # Total drain = base + movement cost
        movement_cost = self._move_cost_factor[:N] * speeds
        total_drain = self._base_drain_per_tick[:N] + movement_cost

        # Apply drain to alive entities only, clamp at zero
        self._energy_current[:N][alive_mask] = np.maximum(
            0.0,
            self._energy_current[:N][alive_mask] - total_drain[alive_mask]
        )

    def _apply_feeding(self):
        """
        Phase B.5: Apply feeding from resource fields.

        Entities gain energy by consuming plankton at their current positions.
        Uses post-movement positions to sample resource density.

        Feeding constraints:
        - Only alive entities feed
        - Intake capped by intake_rate_per_tick
        - Energy gain = consumption * feeding_efficiency
        - Energy clamped to energy_max
        """
        N = len(self.entities)
        if N == 0:
            return

        # Alive-after-drain mask (exclude entities that hit threshold this tick)
        # Prevents same-tick revival where entity drains to threshold then feeds back up
        alive_after_drain = self._alive[:N] & (self._energy_current[:N] > self._starvation_threshold[:N])
        alive_indices = np.where(alive_after_drain)[0]

        if len(alive_indices) == 0:
            return

        # Extract positions of alive entities
        positions = np.array([self.entities[i].position for i in alive_indices], dtype=np.float64)

        # Sample resource density at positions
        densities = self._sample_plankton_density(positions)

        # Compute intake (capped by intake_rate)
        intake = np.minimum(densities, self._intake_rate_per_tick[alive_indices])

        # Energy gain from consumption
        energy_gain = intake * self._feeding_efficiency[alive_indices]

        # Apply energy gain, clamp to max
        self._energy_current[alive_indices] = np.minimum(
            self._energy_max[alive_indices],
            self._energy_current[alive_indices] + energy_gain
        )

        # Update telemetry
        self._ecosystem_telemetry['mean_density_sampled'] = np.mean(densities) if len(densities) > 0 else 0.0

    def _process_deaths(self) -> int:
        """
        Phase B.75: Mark dead entities and compact SoA arrays.

        Entities with energy <= 0 are marked dead and removed from:
        - self.entities list
        - All SoA arrays (energy, metabolism, positions)

        Must be called BEFORE gossip spatial rebuild to maintain index integrity.
        The scheduled B.9 rebuild will increment _build_seq automatically.

        Returns:
            Number of entities that died this tick
        """
        N = len(self.entities)
        if N == 0:
            return 0

        # Find newly dead entities (energy <= threshold and currently alive)
        newly_dead = (self._energy_current[:N] <= self._starvation_threshold[:N]) & self._alive[:N]
        dead_indices = np.where(newly_dead)[0]

        if len(dead_indices) == 0:
            return 0  # No deaths this tick

        # Mark as dead
        self._alive[dead_indices] = False

        # Compact arrays - keep only alive entities
        alive_indices = np.where(self._alive[:N])[0]
        new_N = len(alive_indices)

        # Compact all SoA arrays
        self._energy_current[:new_N] = self._energy_current[alive_indices]
        self._energy_max[:new_N] = self._energy_max[alive_indices]
        self._alive[:new_N] = True
        self._alive[new_N:N] = False
        self._last_feed_tick[:new_N] = self._last_feed_tick[alive_indices]

        # Compact metabolism arrays
        self._base_drain_per_tick[:new_N] = self._base_drain_per_tick[alive_indices]
        self._move_cost_factor[:new_N] = self._move_cost_factor[alive_indices]
        self._intake_rate_per_tick[:new_N] = self._intake_rate_per_tick[alive_indices]
        self._feeding_efficiency[:new_N] = self._feeding_efficiency[alive_indices]
        self._starvation_threshold[:new_N] = self._starvation_threshold[alive_indices]

        # Compact position array (used by spatial adapter)
        self._positions[:new_N] = self._positions[alive_indices]

        # Update entity list
        self.entities = [self.entities[i] for i in alive_indices]

        # Update count
        self._count = new_N

        return N - new_N  # Number of deaths

    def update_ship(self, position: np.ndarray, biome_id: str = None):
        """
        Update or create ship entity (inert, used for behavior conditions).

        Args:
            position: Ship position [x, y, z]
            biome_id: Biome ID (defaults to first biome if not specified)
        """
        if biome_id is None:
            biome_id = list(self.biomes.keys())[0]

        if self.ship_entity is None:
            # Create ship entity
            self.ship_entity = Entity(
                instance_id="ship-player-0000",
                species_id="ship",
                biome_id=biome_id,
                position=np.array(position, dtype=np.float64),
                velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                size_factor=1.0,
                tags=['ship']
            )
        else:
            # Update position
            self.ship_entity.position = np.array(position, dtype=np.float64)
            self.ship_entity.biome_id = biome_id

    def _ensure_position_capacity(self, n: int):
        """
        Ensure position array has capacity for at least n entities.

        Uses amortized growth (2x) to avoid per-tick allocations.
        When entity count shrinks, array is not shrunk.

        Args:
            n: Required capacity
        """
        if n > self._capacity:
            # Allocate new array with 2x growth or exact size
            new_capacity = max(2 * self._capacity, n)
            new_positions = np.empty((new_capacity, 3), dtype=np.float64)

            # Copy existing data if any
            if self._count > 0:
                new_positions[:self._count] = self._positions[:self._count]

            # Swap arrays
            self._positions = new_positions
            self._capacity = new_capacity

    def _build_query_cache(self, all_entities: List) -> dict:
        """
        Build batch query cache for hot tags (Phase 5 refined).

        Phase 5 refined: Preallocated numpy arrays instead of dicts.
        Per SD's guidance:
        - Preallocate dist/idx arrays (N,) filled with inf/-1
        - Fill slices per biome (no dict overhead)
        - Cache structure: cache['nearest'][tag] = (dist_arr, idx_arr)

        Args:
            all_entities: All entities (simulation entities + ship if present)

        Returns:
            Cache dict with structure: {'nearest': {'predator': (dist_arr, idx_arr), 'ship': (...)}}
        """
        N = len(all_entities)

        # Phase 5 refined: Preallocate arrays for each tag
        cache = {
            'nearest': {
                'predator': (
                    np.full(N, np.inf, dtype=np.float64),  # distances
                    np.full(N, -1, dtype=np.int64)         # indices
                ),
                'ship': (
                    np.full(N, np.inf, dtype=np.float64),
                    np.full(N, -1, dtype=np.int64)
                )
            }
        }

        # Group entity rows by biome
        biome_groups = {}  # biome_id -> list of row indices
        for row in range(N):
            biome_id = self.spatial._biome_of_row[row]
            if biome_id not in biome_groups:
                biome_groups[biome_id] = []
            biome_groups[biome_id].append(row)

        # Query configurations per SD guidance
        tag_configs = {
            'predator': {'max_distance': 80.0},
            'ship': {'max_distance': 150.0}
        }

        # Run batch queries per biome, per tag
        for tag, config in tag_configs.items():
            max_distance = config['max_distance']
            dist_arr, idx_arr = cache['nearest'][tag]

            for biome_id, rows in biome_groups.items():
                if not rows:
                    continue

                rows_array = np.array(rows, dtype=np.int32)

                # Run batch query for this biome
                indices, distances = self.spatial.nearest_by_tag_batch(
                    source_rows=rows_array,
                    tag=tag,
                    max_distance=max_distance
                )

                # Write directly to cache arrays by biome slice (no concatenation)
                dist_arr[rows_array] = distances
                idx_arr[rows_array] = indices

        return cache

    def tick(self, evaluate_behaviors: bool = True):
        """
        Advance simulation by one time step.

        TWO-PHASE TICK CONTRACT (Critical Invariant):

        Phase A: Behavior Evaluation (Read-Only)
        ----------------------------------------
        All entities evaluate behaviors using world state at tick start (t=N).
        Behaviors read positions/velocities from t=N and compute desired velocities.
        No entity positions are updated during this phase.

        Result: All entity velocities determined based on consistent world state.

        Phase B: Physics Integration (Write)
        ------------------------------------
        All entities update positions using velocities from Phase A.
        Bounds reflection applied after movement.

        Result: World state advanced to t=N+1.

        Why This Matters:
        - Prevents order-dependent behavior (entity A sees entity B's old position)
        - Ensures deterministic outcomes (evaluation order doesn't affect results)
        - Critical for nearest-entity queries (all distances computed from t=N)

        Args:
            evaluate_behaviors: If True, run Phase A (behavior evaluation)
                               If False, skip Phase A (Phase 2 compatibility)
        """
        start_time = time.perf_counter()

        # Sort entities by instance_id for determinism
        self.entities.sort(key=lambda e: e.instance_id)

        # Build entity list including ship (if present)
        all_entities = self.entities.copy()
        if self.ship_entity is not None:
            all_entities.append(self.ship_entity)

        # ============================================================
        # PHASE A: BEHAVIOR EVALUATION (Read-Only)
        # ============================================================
        # All behaviors see world state from tick start (t=N)
        # Velocities are updated, but positions remain at t=N

        if evaluate_behaviors:
            # Build spatial index ONCE at Phase A start (t=N snapshot)
            build_start = time.perf_counter()

            # Ensure capacity and fill positions array
            # Use local build_count for spatial build (may include ship)
            # _count tracks live entities only (excludes ship)
            build_count = len(all_entities)
            self._ensure_position_capacity(build_count)
            for i, entity in enumerate(all_entities):
                self._positions[i] = entity.position

            # Build index from prebuilt positions (with cached thermal providers)
            self.spatial.build(
                self._positions[:build_count],
                refs=all_entities,
                thermal_centers=self._thermal_centers,
                thermal_base_deltas=self._thermal_base_deltas,
                thermal_influences=self._thermal_influences
            )
            build_elapsed = time.perf_counter() - build_start
            self._build_times.append(build_elapsed)

            # Phase 5: Build batch query cache
            batch_query_start = time.perf_counter()
            if self._use_batch_cache:
                query_cache = self._build_query_cache(all_entities)
            else:
                query_cache = None  # Fallback to per-entity queries
            batch_query_elapsed = time.perf_counter() - batch_query_start
            self._batch_query_times.append(batch_query_elapsed)

            # Phase 8: Construct EnergyView for behavior evaluation (hunger-driven behaviors)
            energy_view = EnergyView(
                row_of_id=self.spatial._row_of_id,
                energy_current=self._energy_current,
                starvation_threshold=self._starvation_threshold
            )
            behavior_context = {'energy_view': energy_view}

            # Evaluate behaviors using spatial adapter + cache + context
            behavior_start = time.perf_counter()
            for entity in self.entities:
                species = self.species_registry.get(entity.species_id)
                if species:
                    update_entity_behavior(entity, species, all_entities, self.spatial, query_cache, behavior_context)
            behavior_elapsed = time.perf_counter() - behavior_start
            self._behavior_times.append(behavior_elapsed)

        # ============================================================
        # PHASE A.5: OBSTACLE AVOIDANCE (Velocity Modification)
        # ============================================================
        # Apply obstacle avoidance to entity velocities
        # Modifies velocities in place, does not update positions
        # Stateless per entity (no inter-entity velocity dependencies)

        if evaluate_behaviors:
            from .avoidance import apply_avoidance
            from .constants import AVOIDANCE_TIMING_BREAKDOWN

            avoidance_start = time.perf_counter()

            # Apply avoidance per biome (obstacles are biome-specific)
            for biome_id, biome in self.biomes.items():
                biome_entities = [e for e in self.entities if e.biome_id == biome_id]

                if biome_entities:
                    breakdown = apply_avoidance(
                        entities=biome_entities,
                        obstacles=biome.obstacles,
                        influence_radius_factor=self.world.simulation.influence_radius_factor,
                        avoidance_weight=self.world.simulation.avoidance_weight,
                        seabed_influence_distance=self.world.simulation.seabed_influence_distance,
                        dt=self.dt
                    )

                    # Capture per-type breakdown if enabled
                    if AVOIDANCE_TIMING_BREAKDOWN and breakdown is not None:
                        self._avoidance_sphere_times.append(breakdown['sphere_ms'] / 1000.0)
                        self._avoidance_cylinder_times.append(breakdown['cylinder_ms'] / 1000.0)
                        self._avoidance_plane_times.append(breakdown['plane_ms'] / 1000.0)

            avoidance_elapsed = time.perf_counter() - avoidance_start
            self._avoidance_times.append(avoidance_elapsed)

        # ============================================================
        # PHASE A.75: ENERGY DRAIN (Ecosystem)
        # ============================================================
        # Apply per-tick energy drain based on metabolism and movement costs
        # Uses pre-movement velocities (Phase A finalized velocities)

        if USE_ECOSYSTEM and evaluate_behaviors:
            energy_drain_start = time.perf_counter()
            self._apply_energy_drain()
            energy_drain_elapsed = time.perf_counter() - energy_drain_start
            self._energy_drain_times.append(energy_drain_elapsed)

        # ============================================================
        # PHASE B: PHYSICS INTEGRATION (Write)
        # ============================================================
        # Now that all velocities determined, update positions to t=N+1

        movement_start = time.perf_counter()

        for entity in self.entities:
            entity.update_position(self.dt)

        # Apply bounds reflection for each biome
        for biome_id, biome in self.biomes.items():
            biome_entities = [e for e in self.entities if e.biome_id == biome_id]
            self._apply_bounds_reflection(biome, biome_entities)

        movement_elapsed = time.perf_counter() - movement_start
        self._movement_times.append(movement_elapsed)

        # ============================================================
        # PHASE B.5: FEEDING (Ecosystem)
        # ============================================================
        # Sample resource density and apply feeding
        # Entities gain energy from plankton fields at current positions

        if USE_ECOSYSTEM and evaluate_behaviors:
            feeding_start = time.perf_counter()
            self._apply_feeding()
            feeding_elapsed = time.perf_counter() - feeding_start
            self._feeding_times.append(feeding_elapsed)

        # ============================================================
        # PHASE B.75: DEATH PROCESSING (Ecosystem)
        # ============================================================
        # Mark dead entities (energy <= 0) and compact SoA arrays
        # Must occur BEFORE gossip spatial rebuild to maintain index integrity

        if USE_ECOSYSTEM and evaluate_behaviors:
            death_start = time.perf_counter()
            deaths_count = self._process_deaths()
            death_elapsed = time.perf_counter() - death_start
            self._death_times.append(death_elapsed)

            self._ecosystem_telemetry['deaths_this_tick'] = deaths_count
            self._ecosystem_telemetry['total_deaths'] += deaths_count

        # ============================================================
        # PHASE B.9: KNOWLEDGE GOSSIP (after movement, rebuild spatial index)
        # ============================================================
        # Entities exchange knowledge tokens with nearby entities
        # Requires spatial index rebuild since positions changed in Phase B

        if USE_GOSSIP and evaluate_behaviors:
            gossip_start = time.perf_counter()

            # Rebuild all_entities AFTER compaction (fresh list, no stale references)
            all_entities = self.entities.copy()
            if self.ship_entity is not None:
                all_entities.append(self.ship_entity)

            # Rebuild spatial index with post-movement positions for gossip queries
            # Use local build_count for spatial build (may include ship)
            # _count tracks live entities only (excludes ship)
            build_count = len(all_entities)
            self._ensure_position_capacity(build_count)
            for i, entity in enumerate(all_entities):
                self._positions[i] = entity.position

            self.spatial.build(
                self._positions[:build_count],
                refs=all_entities,
                thermal_centers=self._thermal_centers,
                thermal_base_deltas=self._thermal_base_deltas,
                thermal_influences=self._thermal_influences
            )

            # Exchange knowledge tokens (vectorized, radius-based, push-pull, Phase 2: with lifecycle)
            gossip_result = exchange_tokens(
                self.entities,
                self.spatial,
                self.species_registry,
                current_tick=self.tick_count
            )

            gossip_elapsed = time.perf_counter() - gossip_start
            self._gossip_times.append(gossip_elapsed)

            # Store telemetry for network health diagnostics (Phase 7)
            self._gossip_telemetry = gossip_result.get('telemetry')

        # Increment tick count
        self.tick_count += 1

        # Record timing
        elapsed = time.perf_counter() - start_time
        self._record_tick_time(elapsed)

        # Debug invariant check (zero perf impact when env var not set)
        # Invariant: _count tracks live simulation entities only (excludes ship)
        if os.getenv('SIM_DEBUG_INVARIANTS') == '1':
            assert self._count == len(self.entities), \
                f"_count ({self._count}) != len(entities) ({len(self.entities)})"

    def _apply_bounds_reflection(self, biome: Biome, entities: List[Entity]):
        """
        Reflect entities at biome boundary sphere.

        When entity exceeds biome radius:
        1. Calculate outward normal
        2. Reflect velocity across normal
        3. Project position back to sphere surface
        4. Re-clamp velocity to cruise speed

        Args:
            biome: Biome definition
            entities: Entities in this biome
        """
        center = np.array(biome.bounds['center'], dtype=np.float64)
        radius = biome.bounds['radius']

        for entity in entities:
            # Check if outside bounds
            dist = distance_to_sphere(entity.position, center, radius)

            if dist > 0:
                # Entity outside sphere, reflect
                offset = entity.position - center
                normal, _ = normalize(offset)

                # Reflect velocity
                entity.velocity = reflect_velocity(entity.velocity, normal)

                # Project position to surface
                entity.position = center + normal * radius

                # Re-clamp to cruise speed
                species = self.species_registry[entity.species_id]
                cruise_speed = species.movement.max_speed_ms * CRUISE_SPEED_FRACTION
                entity.velocity = clamp_speed(entity.velocity, cruise_speed)

    def get_tick_stats(self) -> dict:
        """
        Get current tick timing statistics.

        Returns:
            Dict with tick_count, avg_tick_time_ms, last_tick_time_ms
        """
        if not self._tick_times:
            return {
                'tick_count': self.tick_count,
                'avg_tick_time_ms': 0.0,
                'last_tick_time_ms': 0.0
            }

        avg_time = self._tick_time_sum / len(self._tick_times)
        last_time = self._tick_times[-1]

        return {
            'tick_count': self.tick_count,
            'avg_tick_time_ms': avg_time * 1000.0,
            'last_tick_time_ms': last_time * 1000.0
        }

    def _record_tick_time(self, elapsed: float):
        """
        Record tick timing for rolling average.

        Args:
            elapsed: Tick time in seconds
        """
        self._tick_times.append(elapsed)
        self._tick_time_sum += elapsed

        # Maintain rolling window
        if len(self._tick_times) > self._tick_time_window:
            removed = self._tick_times.pop(0)
            self._tick_time_sum -= removed

    def get_snapshot(self) -> dict:
        """
        Get complete simulation state snapshot.

        Returns:
            Dict with tick_count, entities, timing
        """
        return {
            'tick_count': self.tick_count,
            'entity_count': len(self.entities),
            'entities': [e.to_dict() for e in self.entities],
            'timing': self.get_tick_stats()
        }

    def print_tick_summary(self):
        """Print tick summary to console (lightweight monitoring)"""
        stats = self.get_tick_stats()
        print(f"Tick {stats['tick_count']:5d} | "
              f"Avg: {stats['avg_tick_time_ms']:6.3f} ms | "
              f"Last: {stats['last_tick_time_ms']:6.3f} ms | "
              f"Entities: {len(self.entities)}")

    def print_perf_breakdown(self, every: int = 200):
        """
        Print performance breakdown on interval (Phase 6).

        Logs timing breakdown: build_main_ms, build_tag_ms, batch_query_ms,
        behavior_ms, avoidance_ms, sensor_ms (new), movement_ms. Optional per-type breakdown
        if AVOIDANCE_TIMING_BREAKDOWN=True. Only prints every N ticks to reduce overhead.

        Args:
            every: Print interval in ticks (default 200)
        """
        if self.tick_count % every != 0:
            return

        # Calculate averages over last window
        window = min(every, len(self._build_times))
        if window == 0:
            return

        from .constants import AVOIDANCE_TIMING_BREAKDOWN

        avg_build_main = self.spatial.last_build_main_ms
        avg_build_tag = self.spatial.last_build_tag_ms
        avg_batch_query = sum(self._batch_query_times[-window:]) / window * 1000.0
        avg_behavior = sum(self._behavior_times[-window:]) / window * 1000.0
        avg_avoidance = sum(self._avoidance_times[-window:]) / window * 1000.0 if self._avoidance_times else 0.0
        avg_sensor = sum(self._sensor_times[-window:]) / window * 1000.0 if self._sensor_times else 0.0
        avg_movement = sum(self._movement_times[-window:]) / window * 1000.0
        avg_gossip = sum(self._gossip_times[-window:]) / window * 1000.0 if self._gossip_times else 0.0
        avg_total = sum(self._tick_times[-window:]) / window * 1000.0

        print(f"\n[Perf Breakdown] Tick {self.tick_count} ({len(self.entities)} entities)")
        print(f"  Build main:   {avg_build_main:6.3f} ms")
        print(f"  Build tags:   {avg_build_tag:6.3f} ms")
        print(f"  Batch query:  {avg_batch_query:6.3f} ms")
        print(f"  Behavior:     {avg_behavior:6.3f} ms")
        print(f"  Avoidance:    {avg_avoidance:6.3f} ms")

        # Optional per-type breakdown
        if AVOIDANCE_TIMING_BREAKDOWN and self._avoidance_sphere_times:
            avg_sphere = sum(self._avoidance_sphere_times[-window:]) / min(window, len(self._avoidance_sphere_times)) * 1000.0
            avg_cylinder = sum(self._avoidance_cylinder_times[-window:]) / min(window, len(self._avoidance_cylinder_times)) * 1000.0
            avg_plane = sum(self._avoidance_plane_times[-window:]) / min(window, len(self._avoidance_plane_times)) * 1000.0
            print(f"    -> Sphere:  {avg_sphere:6.3f} ms")
            print(f"    -> Cylinder:{avg_cylinder:6.3f} ms")
            print(f"    -> Plane:   {avg_plane:6.3f} ms")

        print(f"  Sensor:       {avg_sensor:6.3f} ms")
        print(f"  Movement:     {avg_movement:6.3f} ms")
        print(f"  Gossip:       {avg_gossip:6.3f} ms")

        # Print gossip network telemetry on GOSSIP_LOG_INTERVAL (Phase 7: network health EKG)
        if USE_GOSSIP and self.tick_count % GOSSIP_LOG_INTERVAL == 0 and self._gossip_telemetry:
            telemetry = self._gossip_telemetry
            pct_isolated = 100.0 * telemetry['degree_histogram']['isolated'] / telemetry['total_entities']
            pct_sparse = 100.0 * telemetry['degree_histogram']['sparse'] / telemetry['total_entities']
            pct_connected = 100.0 * telemetry['degree_histogram']['connected'] / telemetry['total_entities']
            print(f"  [Gossip Network] degree_mean={telemetry['degree_mean']:.2f} | "
                  f"isolated={pct_isolated:.1f}% sparse={pct_sparse:.1f}% connected={pct_connected:.1f}%")

        # Print ecosystem telemetry on ECOSYSTEM_LOG_INTERVAL (Phase 8+: ecosystem health)
        if USE_ECOSYSTEM and self.tick_count % ECOSYSTEM_LOG_INTERVAL == 0:
            # Compute mean energy of alive entities
            N = len(self.entities)
            if N > 0:
                alive_mask = self._alive[:N]
                alive_energies = self._energy_current[:N][alive_mask]
                mean_energy = np.mean(alive_energies) if len(alive_energies) > 0 else 0.0
                min_energy = np.min(alive_energies) if len(alive_energies) > 0 else 0.0
                starving_count = np.sum(alive_energies < 10.0)  # Arbitrary threshold
            else:
                mean_energy = 0.0
                min_energy = 0.0
                starving_count = 0

            # Compute ecosystem timing averages
            avg_drain = sum(self._energy_drain_times[-window:]) / window * 1000.0 if self._energy_drain_times else 0.0
            avg_feeding = sum(self._feeding_times[-window:]) / window * 1000.0 if self._feeding_times else 0.0
            avg_death = sum(self._death_times[-window:]) / window * 1000.0 if self._death_times else 0.0

            print(f"  [Ecosystem] energy_mean={mean_energy:.1f} min={min_energy:.1f} | "
                  f"deaths={self._ecosystem_telemetry['total_deaths']} starving={starving_count} | "
                  f"drain={avg_drain:.3f}ms feed={avg_feeding:.3f}ms death={avg_death:.3f}ms")

        print(f"  Total:        {avg_total:6.3f} ms")
        print(f"  Overhead:     {(avg_total - avg_build_main - avg_build_tag - avg_batch_query - avg_behavior - avg_avoidance - avg_sensor - avg_movement - avg_gossip):6.3f} ms")
