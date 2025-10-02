"""
Deterministic RNG utilities for aquarium simulation.

Uses SHA256 hashing to derive stable seeds from hierarchical components
(world_seed, biome_id, entity_id, component_name). All randomness uses
numpy.random.Generator(PCG64) for reproducible cross-session results.
"""

import hashlib
import numpy as np
from typing import Any


def make_seed(*components: Any) -> int:
    """
    Generate deterministic 64-bit seed from hierarchical components.

    Uses SHA256 to hash components into stable seed value.

    Args:
        *components: Seed components (world_seed, biome_id, entity_index, etc.)

    Returns:
        64-bit integer seed for numpy RNG

    Example:
        entity_seed = make_seed(world_seed, biome_id, species_id, entity_index)
        velocity_seed = make_seed(entity_seed, "initial_velocity")
    """
    # Join all components with colon separator
    hash_input = ":".join(str(c) for c in components)

    # SHA256 hash and extract 64-bit integer
    hash_bytes = hashlib.sha256(hash_input.encode('utf-8')).digest()
    seed = int.from_bytes(hash_bytes[:8], byteorder='big')

    return seed


def random_unit_vector(seed: int) -> np.ndarray:
    """
    Generate random 3D unit vector (uniform distribution on sphere surface).

    Uses rejection sampling: generate random point in [-1, 1]^3 cube,
    reject if outside unit sphere, normalize.

    Args:
        seed: RNG seed (from make_seed())

    Returns:
        3D unit vector as numpy array [x, y, z]
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    # Rejection sampling for uniform sphere surface
    while True:
        vec = rng.uniform(-1.0, 1.0, size=3)
        length_sq = np.dot(vec, vec)

        # Reject if outside unit sphere or too close to origin
        if 0.01 < length_sq <= 1.0:
            return vec / np.sqrt(length_sq)


def random_position_in_sphere(seed: int, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Generate random position uniformly distributed within sphere.

    Uses rejection sampling in bounding cube.

    Args:
        seed: RNG seed
        center: Sphere center [x, y, z]
        radius: Sphere radius

    Returns:
        Random position within sphere as numpy array [x, y, z]
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    # Rejection sampling for uniform sphere volume
    while True:
        # Random point in [-radius, radius]^3 cube
        offset = rng.uniform(-radius, radius, size=3)

        # Accept if inside sphere
        if np.dot(offset, offset) <= radius * radius:
            return center + offset


def random_size_factor(seed: int, min_val: float = 0.8, max_val: float = 1.2) -> float:
    """
    Generate random size factor for entity size variation.

    Args:
        seed: RNG seed
        min_val: Minimum size factor (default 0.8 = 80% of nominal size)
        max_val: Maximum size factor (default 1.2 = 120% of nominal size)

    Returns:
        Random size factor in [min_val, max_val]
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.uniform(min_val, max_val)
