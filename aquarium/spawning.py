"""
Entity spawning system.

Spawns entities from biome spawning configuration with deterministic placement.
Supports uniform, clustered, and near_obstacles distributions (Phase 2: uniform only).
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from .entity import Entity
from .data_types import Biome, Species, SpawningConfig
from .rng import make_seed, random_position_in_sphere, random_unit_vector, random_size_factor
from .constants import (
    CRUISE_SPEED_FRACTION,
    ACOUSTIC_DEFAULT_AMPLITUDE,
    ACOUSTIC_DEFAULT_PEAK_HZ,
    BIOLUM_DEFAULT_INTENSITY,
    BIOLUM_DEFAULT_WAVELENGTH
)


def spawn_entities(
    biome: Biome,
    species_registry: Dict[str, Species],
    world_seed: int,
    limit: Optional[int] = None,
    only_species: Optional[List[str]] = None
) -> List[Entity]:
    """
    Spawn entities according to biome spawning configuration.

    Phase 2: Only uniform distribution implemented.
    Clustered and near_obstacles distributions stubbed for future.

    Args:
        biome: Biome definition with spawning config
        species_registry: Dict of species_id -> Species
        world_seed: World generation seed
        limit: Optional limit on total entities spawned (for testing)
        only_species: Optional list of species IDs to spawn (for testing)

    Returns:
        List of spawned Entity instances

    Example (test override):
        entities = spawn_entities(biome, registry, seed, limit=10, only_species=['sp-001-drifter'])
    """
    if not biome.spawning or 'species' not in biome.spawning:
        return []

    entities = []
    total_spawned = 0

    # Extract biome bounds
    center = np.array(biome.bounds['center'], dtype=np.float64)
    radius = biome.bounds['radius']

    # Spawn each species
    for spawn_config in biome.spawning['species']:
        species_id = spawn_config.species_id

        # Filter by only_species if specified
        if only_species and species_id not in only_species:
            continue

        # Check if species exists
        if species_id not in species_registry:
            print(f"[WARN] Species {species_id} not found in registry, skipping")
            continue

        species = species_registry[species_id]

        # Determine spawn count (respect limit)
        count = spawn_config.count
        if limit is not None:
            remaining = limit - total_spawned
            count = min(count, remaining)

        if count <= 0:
            break

        # Spawn entities
        spawned = _spawn_species(
            biome=biome,
            species=species,
            count=count,
            distribution=spawn_config.distribution,
            world_seed=world_seed,
            center=center,
            radius=radius
        )

        entities.extend(spawned)
        total_spawned += len(spawned)

        # Stop if limit reached
        if limit is not None and total_spawned >= limit:
            break

    return entities


def _spawn_species(
    biome: Biome,
    species: Species,
    count: int,
    distribution: str,
    world_seed: int,
    center: np.ndarray,
    radius: float
) -> List[Entity]:
    """
    Spawn multiple entities of a single species.

    Args:
        biome: Biome definition
        species: Species definition
        count: Number of entities to spawn
        distribution: Distribution type (uniform, clustered, near_obstacles)
        world_seed: World generation seed
        center: Biome center [x, y, z]
        radius: Biome radius

    Returns:
        List of spawned entities
    """
    entities = []

    for i in range(count):
        # Deterministic seed per entity
        entity_seed = make_seed(world_seed, biome.biome_id, species.species_id, i)

        # Generate instance ID
        instance_id = f"{species.species_id}-{biome.biome_id}-{i:04d}"

        # Generate position based on distribution
        if distribution == "uniform":
            position = _spawn_uniform(entity_seed, center, radius)
        elif distribution == "clustered":
            # Phase 3+: Implement clustered distribution (near vents)
            position = _spawn_uniform(entity_seed, center, radius)
            print(f"[WARN] Clustered distribution not implemented, using uniform for {instance_id}")
        elif distribution == "near_obstacles":
            # Phase 3+: Implement near_obstacles distribution (attached to ridges/vents)
            position = _spawn_uniform(entity_seed, center, radius)
            print(f"[WARN] Near_obstacles distribution not implemented, using uniform for {instance_id}")
        else:
            position = _spawn_uniform(entity_seed, center, radius)
            print(f"[WARN] Unknown distribution '{distribution}', using uniform for {instance_id}")

        # Generate initial velocity (random drift, 20% max speed)
        velocity_seed = make_seed(entity_seed, "initial_velocity")
        direction = random_unit_vector(velocity_seed)
        cruise_speed = species.movement.max_speed_ms * CRUISE_SPEED_FRACTION
        velocity = direction * cruise_speed

        # Generate size factor
        size_seed = make_seed(entity_seed, "size_factor")
        size_factor = random_size_factor(size_seed)

        # Bake base emissions from species
        base_emissions = _extract_base_emissions(species)

        # Create entity
        entity = Entity(
            instance_id=instance_id,
            species_id=species.species_id,
            biome_id=biome.biome_id,
            position=position,
            velocity=velocity,
            size_factor=size_factor,
            tags=species.tags.copy(),  # Copy species tags
            base_emissions=base_emissions  # Baked emission values
        )

        entities.append(entity)

    return entities


def _extract_base_emissions(species: Species) -> dict:
    """
    Extract base emission values from Species, applying fallback defaults.

    Bakes acoustic and bioluminescent channels from Species.emissions.
    Future channels (thermal, chemical, magnetic) can be added here.

    Args:
        species: Species definition with optional emissions profile

    Returns:
        Dict with structure: {
            'acoustic': {'amplitude': float, 'peak_hz': float},
            'bioluminescent': {'intensity': float, 'wavelength_nm': float}
        }
    """
    base_emissions = {}

    # Extract acoustic channel
    if species.emissions and species.emissions.acoustic:
        base_emissions['acoustic'] = {
            'amplitude': species.emissions.acoustic.get('amplitude', ACOUSTIC_DEFAULT_AMPLITUDE),
            'peak_hz': species.emissions.acoustic.get('peak_hz', ACOUSTIC_DEFAULT_PEAK_HZ)
        }
    else:
        # Fallback: all species have some acoustic signature
        base_emissions['acoustic'] = {
            'amplitude': ACOUSTIC_DEFAULT_AMPLITUDE,
            'peak_hz': ACOUSTIC_DEFAULT_PEAK_HZ
        }

    # Extract bioluminescent channel
    if species.emissions and species.emissions.bioluminescent:
        base_emissions['bioluminescent'] = {
            'intensity': species.emissions.bioluminescent.get('intensity', BIOLUM_DEFAULT_INTENSITY),
            'wavelength_nm': species.emissions.bioluminescent.get('wavelength_nm', BIOLUM_DEFAULT_WAVELENGTH)
        }
    else:
        # Fallback: dim bioluminescence
        base_emissions['bioluminescent'] = {
            'intensity': BIOLUM_DEFAULT_INTENSITY,
            'wavelength_nm': BIOLUM_DEFAULT_WAVELENGTH
        }

    return base_emissions


def _spawn_uniform(seed: int, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Spawn entity at random position uniformly within sphere.

    Args:
        seed: RNG seed
        center: Sphere center [x, y, z]
        radius: Sphere radius

    Returns:
        Random position within sphere
    """
    return random_position_in_sphere(seed, center, radius)
