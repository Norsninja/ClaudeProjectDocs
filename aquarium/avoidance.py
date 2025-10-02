"""
Obstacle avoidance for entities (Phase 4 - Optimized with Spatial Hash).

Implements ahead-vector repulsion from spheres, cylinders, and seabed.
Applied in Phase A.5 (after behavior evaluation, before movement).

Architecture (per SD + research):
- Obstacle-centric: Loop over M obstacles (not N entities)
- Spatial hash culling: O(1) entity lookups, 85-90% reduction
- Vectorized per-obstacle math: Process ~100 entities at once
- Direct indexing for force accumulation (NOT numpy.add.at)
- Sequential blending: sphere → cylinder → plane (plane last for safety)
"""

import numpy as np
import time
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from .entity import Entity
from .data_types import SphereObstacle, CylinderObstacle, PlaneObstacle
from .constants import AVOIDANCE_TIMING_BREAKDOWN, AVOIDANCE_LOOKAHEAD_DISTANCES


# Spatial hash cell size (tunable constant)
HASH_CELL_SIZE = 10.0  # meters (approx max_influence_radius + max_lookahead)


def apply_avoidance(
    entities: List[Entity],
    obstacles: dict,
    influence_radius_factor: float,
    avoidance_weight: float,
    seabed_influence_distance: float,
    dt: float
) -> Optional[Dict[str, float]]:
    """
    Apply obstacle avoidance using obstacle-centric processing + spatial hash.

    Modifies entity.velocity in place. Uses spatial hash to cull entities
    far from obstacles, then vectorizes per-obstacle math.

    Args:
        entities: List of entities to apply avoidance to
        obstacles: Dict with keys 'spheres', 'cylinders', 'planes'
        influence_radius_factor: Default multiplier for obstacle radius
        avoidance_weight: Blend weight for avoidance force (0-1)
        seabed_influence_distance: Distance above seabed to start pushing up
        dt: Time delta (for lookahead distance scaling)

    Returns:
        Optional timing breakdown dict if AVOIDANCE_TIMING_BREAKDOWN=True
    """
    if len(entities) == 0:
        return None

    # Build arrays for vectorized operations
    N = len(entities)
    positions = np.array([e.position for e in entities], dtype=np.float64)  # (N, 3)
    velocities = np.array([e.velocity for e in entities], dtype=np.float64)  # (N, 3)
    speeds = np.linalg.norm(velocities, axis=1)  # (N,)

    # Get max speeds from entities (for clamping)
    # Note: This assumes entities are sorted by instance_id for determinism
    # We'll restore this ordering at the end

    # Track original order
    entity_ids = [e.instance_id for e in entities]
    id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    # Skip stationary entities
    moving_mask = speeds > 0.01
    moving_indices = np.where(moving_mask)[0]

    if len(moving_indices) == 0:
        return None  # No moving entities

    # Build spatial hash for moving entities only
    spatial_hash = _build_spatial_hash(positions[moving_indices], moving_indices, HASH_CELL_SIZE)

    # Initialize force accumulators
    avoid_sphere = np.zeros((N, 3), dtype=np.float64)
    avoid_cylinder = np.zeros((N, 3), dtype=np.float64)
    avoid_plane = np.zeros((N, 3), dtype=np.float64)

    # Get obstacle lists
    spheres = obstacles.get('spheres', [])
    cylinders = obstacles.get('cylinders', [])
    planes = obstacles.get('planes', [])

    # Optional per-type timing
    if AVOIDANCE_TIMING_BREAKDOWN:
        sphere_time = 0.0
        cylinder_time = 0.0
        plane_time = 0.0

    # Process spheres (obstacle-centric)
    if spheres and AVOIDANCE_TIMING_BREAKDOWN:
        t0 = time.perf_counter()

    for sphere in spheres:
        # Determine influence radius
        if sphere.influence_radius is not None:
            influence_radius = sphere.influence_radius
        else:
            influence_radius = sphere.radius * influence_radius_factor

        # Get candidate entities from spatial hash
        candidate_indices = _query_sphere_candidates(
            spatial_hash, sphere.center, influence_radius, AVOIDANCE_LOOKAHEAD_DISTANCES
        )

        if len(candidate_indices) == 0:
            continue

        # Vectorized sphere avoidance for candidates
        _accumulate_sphere_avoidance(
            positions, velocities, speeds, candidate_indices,
            sphere.center, sphere.radius, influence_radius,
            avoid_sphere, AVOIDANCE_LOOKAHEAD_DISTANCES
        )

    if spheres and AVOIDANCE_TIMING_BREAKDOWN:
        sphere_time = time.perf_counter() - t0

    # Process cylinders (obstacle-centric)
    if cylinders and AVOIDANCE_TIMING_BREAKDOWN:
        t0 = time.perf_counter()

    for cylinder in cylinders:
        # Determine influence radius
        if cylinder.influence_radius is not None:
            influence_radius = cylinder.influence_radius
        else:
            influence_radius = cylinder.radius * influence_radius_factor

        # Get candidate entities from spatial hash (AABB query)
        candidate_indices = _query_cylinder_candidates(
            spatial_hash, cylinder.start, cylinder.end,
            influence_radius, AVOIDANCE_LOOKAHEAD_DISTANCES
        )

        if len(candidate_indices) == 0:
            continue

        # Vectorized cylinder avoidance for candidates
        _accumulate_cylinder_avoidance(
            positions, velocities, speeds, candidate_indices,
            cylinder.start, cylinder.end, cylinder.radius, influence_radius,
            avoid_cylinder, AVOIDANCE_LOOKAHEAD_DISTANCES
        )

    if cylinders and AVOIDANCE_TIMING_BREAKDOWN:
        cylinder_time = time.perf_counter() - t0

    # Process plane (seabed) - vectorized across ALL entities
    if planes and AVOIDANCE_TIMING_BREAKDOWN:
        t0 = time.perf_counter()

    if planes:
        seabed_planes = [p for p in planes if p.type == "seabed"]
        if seabed_planes:
            seabed = seabed_planes[0]
            influence_distance = getattr(seabed, 'influence_distance', seabed_influence_distance)

            # Vectorized plane push for all entities
            distance_above = positions[:, 1] - seabed.y_level
            mask = distance_above < influence_distance

            strength = np.where(mask, 1.0 - (distance_above / influence_distance), 0.0)
            strength = np.clip(strength, 0.0, 1.0)

            # Use entity speed for push magnitude (or default for stationary)
            push_speeds = np.where(speeds > 0.01, speeds, 1.0)
            avoid_plane[:, 1] = strength * push_speeds

    if planes and AVOIDANCE_TIMING_BREAKDOWN:
        plane_time = time.perf_counter() - t0

    # Blend forces into velocities (sequential: sphere → cylinder → plane)
    # Sphere forces
    velocities = velocities * (1.0 - avoidance_weight) + avoid_sphere * avoidance_weight
    velocities = _clamp_speeds(velocities, speeds)

    # Cylinder forces
    velocities = velocities * (1.0 - avoidance_weight) + avoid_cylinder * avoidance_weight
    velocities = _clamp_speeds(velocities, speeds)

    # Plane forces (stronger weight for safety)
    seabed_weight = 0.8
    velocities = velocities * (1.0 - seabed_weight) + avoid_plane * seabed_weight
    velocities = _clamp_speeds(velocities, speeds)

    # Write back to entities (in deterministic order)
    sorted_entities = sorted(entities, key=lambda e: e.instance_id)
    for entity in sorted_entities:
        idx = id_to_idx[entity.instance_id]
        entity.velocity = velocities[idx]

    # Return timing breakdown if flag enabled
    if AVOIDANCE_TIMING_BREAKDOWN:
        return {
            'sphere_ms': sphere_time * 1000.0,
            'cylinder_ms': cylinder_time * 1000.0,
            'plane_ms': plane_time * 1000.0
        }
    return None


def _build_spatial_hash(
    positions: np.ndarray,
    indices: np.ndarray,
    cell_size: float
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Build 3D spatial hash from entity positions.

    Args:
        positions: (N, 3) array of positions
        indices: (N,) array of entity indices
        cell_size: Size of hash grid cells (meters)

    Returns:
        Dict mapping (ix, iy, iz) cell coords to list of entity indices
    """
    grid = defaultdict(list)

    for i, pos in zip(indices, positions):
        cell_x = int(np.floor(pos[0] / cell_size))
        cell_y = int(np.floor(pos[1] / cell_size))
        cell_z = int(np.floor(pos[2] / cell_size))
        cell_key = (cell_x, cell_y, cell_z)
        grid[cell_key].append(i)

    return grid


def _query_sphere_candidates(
    spatial_hash: Dict,
    center: np.ndarray,
    influence_radius: float,
    lookahead_distances: List[float]
) -> np.ndarray:
    """
    Query spatial hash for entities near sphere obstacle.

    Args:
        spatial_hash: Spatial hash grid
        center: Sphere center position
        influence_radius: Sphere influence radius
        lookahead_distances: Lookahead sample distances

    Returns:
        Array of unique candidate entity indices
    """
    # Query radius includes influence + max lookahead
    query_radius = influence_radius + max(lookahead_distances)

    # Determine cell range to check
    cell_radius = int(np.ceil(query_radius / HASH_CELL_SIZE))

    center_cell = (
        int(np.floor(center[0] / HASH_CELL_SIZE)),
        int(np.floor(center[1] / HASH_CELL_SIZE)),
        int(np.floor(center[2] / HASH_CELL_SIZE))
    )

    # Collect candidates from neighboring cells
    candidates = []
    for dx in range(-cell_radius, cell_radius + 1):
        for dy in range(-cell_radius, cell_radius + 1):
            for dz in range(-cell_radius, cell_radius + 1):
                cell_key = (
                    center_cell[0] + dx,
                    center_cell[1] + dy,
                    center_cell[2] + dz
                )
                if cell_key in spatial_hash:
                    candidates.extend(spatial_hash[cell_key])

    # Deduplicate and sort for determinism
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    return np.unique(np.array(candidates, dtype=np.int64))


def _query_cylinder_candidates(
    spatial_hash: Dict,
    start: np.ndarray,
    end: np.ndarray,
    influence_radius: float,
    lookahead_distances: List[float]
) -> np.ndarray:
    """
    Query spatial hash for entities near cylinder obstacle.

    Uses AABB (axis-aligned bounding box) expanded by influence radius.

    Args:
        spatial_hash: Spatial hash grid
        start: Cylinder start point
        end: Cylinder end point
        influence_radius: Cylinder influence radius
        lookahead_distances: Lookahead sample distances

    Returns:
        Array of unique candidate entity indices
    """
    # AABB around cylinder, expanded by influence + max lookahead
    expansion = influence_radius + max(lookahead_distances)

    min_corner = np.minimum(start, end) - expansion
    max_corner = np.maximum(start, end) + expansion

    # Convert to cell coordinates
    min_cell = tuple(int(np.floor(x / HASH_CELL_SIZE)) for x in min_corner)
    max_cell = tuple(int(np.floor(x / HASH_CELL_SIZE)) for x in max_corner)

    # Collect candidates from cells in AABB
    candidates = []
    for cx in range(min_cell[0], max_cell[0] + 1):
        for cy in range(min_cell[1], max_cell[1] + 1):
            for cz in range(min_cell[2], max_cell[2] + 1):
                cell_key = (cx, cy, cz)
                if cell_key in spatial_hash:
                    candidates.extend(spatial_hash[cell_key])

    # Deduplicate and sort
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    return np.unique(np.array(candidates, dtype=np.int64))


def _accumulate_sphere_avoidance(
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    candidate_indices: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
    influence_radius: float,
    avoid_accumulator: np.ndarray,
    lookahead_distances: List[float]
) -> None:
    """
    Accumulate sphere avoidance forces for candidate entities (vectorized).

    Args:
        positions: (N, 3) entity positions
        velocities: (N, 3) entity velocities
        speeds: (N,) entity speeds
        candidate_indices: Indices of entities to process
        sphere_center: Sphere center position
        sphere_radius: Sphere radius
        influence_radius: Sphere influence radius
        avoid_accumulator: (N, 3) accumulator for avoidance forces
        lookahead_distances: Lookahead sample distances
    """
    if len(candidate_indices) == 0:
        return

    # Ensure sphere_center is numpy array
    sphere_center_arr = np.array(sphere_center, dtype=np.float64)

    # Extract candidate data
    cand_positions = positions[candidate_indices]  # (C, 3)
    cand_velocities = velocities[candidate_indices]  # (C, 3)
    cand_speeds = speeds[candidate_indices]  # (C,)

    # Compute direction vectors
    cand_directions = np.zeros_like(cand_velocities)
    moving_mask = cand_speeds > 0.01
    cand_directions[moving_mask] = cand_velocities[moving_mask] / cand_speeds[moving_mask, None]

    # Compute lookahead samples for all candidates
    n_samples = len(lookahead_distances)
    samples = np.zeros((len(candidate_indices), n_samples, 3), dtype=np.float64)
    for i, d in enumerate(lookahead_distances):
        samples[:, i, :] = cand_positions + cand_directions * d

    # Vectorized distance to sphere center for all samples
    # Shape: (C, n_samples, 3)
    deltas = sphere_center_arr[None, None, :] - samples  # Broadcast
    distances_to_center = np.linalg.norm(deltas, axis=2)  # (C, n_samples)

    # Distance to surface
    distances_to_surface = distances_to_center - sphere_radius

    # Find nearest sample per candidate
    nearest_sample_idx = np.argmin(distances_to_surface, axis=1)  # (C,)
    nearest_distances = distances_to_surface[np.arange(len(candidate_indices)), nearest_sample_idx]

    # Mask: within influence
    influence_mask = nearest_distances < influence_radius

    if not np.any(influence_mask):
        return

    # Get affected candidates
    affected_indices = candidate_indices[influence_mask]
    affected_sample_idx = nearest_sample_idx[influence_mask]
    affected_distances = nearest_distances[influence_mask]
    affected_speeds = cand_speeds[influence_mask]

    # Get nearest sample positions
    affected_samples = samples[influence_mask, affected_sample_idx, :]  # Advanced indexing trick
    affected_samples = np.array([samples[i, affected_sample_idx[i]] for i in range(len(affected_sample_idx))])

    # Repulsion magnitude (linear falloff)
    repulsion_magnitude = 1.0 - (np.maximum(0.0, affected_distances) / influence_radius)
    repulsion_magnitude = np.clip(repulsion_magnitude, 0.0, 1.0)

    # Repulsion direction (away from nearest point on sphere)
    # Point on sphere surface closest to sample
    affected_deltas = sphere_center_arr - affected_samples
    dist_to_center = np.linalg.norm(affected_deltas, axis=1)

    # Normalize direction (outward from sphere)
    repulsion_directions = np.zeros_like(affected_samples)
    valid_mask = dist_to_center > 0
    repulsion_directions[valid_mask] = -affected_deltas[valid_mask] / dist_to_center[valid_mask, None]

    # For samples at center (rare), use velocity direction
    zero_mask = ~valid_mask
    if np.any(zero_mask):
        repulsion_directions[zero_mask] = cand_directions[influence_mask][zero_mask]

    # Repulsion vectors (scaled by speed)
    repulsion_vectors = repulsion_directions * (repulsion_magnitude[:, None] * affected_speeds[:, None])

    # Accumulate forces (direct indexing, NOT add.at)
    avoid_accumulator[affected_indices] += repulsion_vectors


def _accumulate_cylinder_avoidance(
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    candidate_indices: np.ndarray,
    cylinder_start: np.ndarray,
    cylinder_end: np.ndarray,
    cylinder_radius: float,
    influence_radius: float,
    avoid_accumulator: np.ndarray,
    lookahead_distances: List[float]
) -> None:
    """
    Accumulate cylinder avoidance forces for candidate entities (vectorized).

    Args:
        positions: (N, 3) entity positions
        velocities: (N, 3) entity velocities
        speeds: (N,) entity speeds
        candidate_indices: Indices of entities to process
        cylinder_start: Cylinder start point
        cylinder_end: Cylinder end point
        cylinder_radius: Cylinder radius
        influence_radius: Cylinder influence radius
        avoid_accumulator: (N, 3) accumulator for avoidance forces
        lookahead_distances: Lookahead sample distances
    """
    if len(candidate_indices) == 0:
        return

    # Precompute cylinder axis
    cylinder_start_arr = np.array(cylinder_start, dtype=np.float64)
    cylinder_end_arr = np.array(cylinder_end, dtype=np.float64)
    ab = cylinder_end_arr - cylinder_start_arr
    ab_len_sq = np.dot(ab, ab)

    if ab_len_sq < 1e-9:  # Degenerate cylinder
        return

    # Extract candidate data
    cand_positions = positions[candidate_indices]
    cand_velocities = velocities[candidate_indices]
    cand_speeds = speeds[candidate_indices]

    # Compute direction vectors
    cand_directions = np.zeros_like(cand_velocities)
    moving_mask = cand_speeds > 0.01
    cand_directions[moving_mask] = cand_velocities[moving_mask] / cand_speeds[moving_mask, None]

    # Compute lookahead samples
    n_samples = len(lookahead_distances)
    samples = np.zeros((len(candidate_indices), n_samples, 3), dtype=np.float64)
    for i, d in enumerate(lookahead_distances):
        samples[:, i, :] = cand_positions + cand_directions * d

    # Vectorized point-to-segment distance for all samples
    # Flatten samples: (C*n_samples, 3)
    samples_flat = samples.reshape(-1, 3)

    # Project onto segment
    ap = samples_flat - cylinder_start_arr  # (C*n_samples, 3)
    t = np.dot(ap, ab) / ab_len_sq  # (C*n_samples,)
    t = np.clip(t, 0.0, 1.0)

    # Closest points on segment
    closest_points = cylinder_start_arr + t[:, None] * ab  # (C*n_samples, 3)

    # Distances to axis
    deltas = samples_flat - closest_points
    distances_to_axis = np.linalg.norm(deltas, axis=1)  # (C*n_samples,)

    # Distance to surface
    distances_to_surface = distances_to_axis - cylinder_radius

    # Reshape back: (C, n_samples)
    distances_to_surface = distances_to_surface.reshape(len(candidate_indices), n_samples)
    closest_points = closest_points.reshape(len(candidate_indices), n_samples, 3)
    deltas = deltas.reshape(len(candidate_indices), n_samples, 3)

    # Find nearest sample per candidate
    nearest_sample_idx = np.argmin(distances_to_surface, axis=1)
    nearest_distances = distances_to_surface[np.arange(len(candidate_indices)), nearest_sample_idx]

    # Mask: within influence
    influence_mask = nearest_distances < influence_radius

    if not np.any(influence_mask):
        return

    # Get affected candidates
    affected_indices = candidate_indices[influence_mask]
    affected_sample_idx = nearest_sample_idx[influence_mask]
    affected_distances = nearest_distances[influence_mask]
    affected_speeds = cand_speeds[influence_mask]

    # Get nearest sample positions and deltas
    affected_samples = np.array([samples[i, affected_sample_idx[i]] for i in range(len(affected_sample_idx))])
    affected_deltas = np.array([deltas[i, affected_sample_idx[i]] for i in range(len(affected_sample_idx))])

    # Repulsion magnitude
    repulsion_magnitude = 1.0 - (np.maximum(0.0, affected_distances) / influence_radius)
    repulsion_magnitude = np.clip(repulsion_magnitude, 0.0, 1.0)

    # Repulsion direction (away from closest point on axis)
    repulsion_directions = np.zeros_like(affected_deltas)
    norms = np.linalg.norm(affected_deltas, axis=1)
    valid_mask = norms > 0
    repulsion_directions[valid_mask] = affected_deltas[valid_mask] / norms[valid_mask, None]

    # For samples on axis (rare), use velocity direction
    zero_mask = ~valid_mask
    if np.any(zero_mask):
        repulsion_directions[zero_mask] = cand_directions[influence_mask][zero_mask]

    # Repulsion vectors
    repulsion_vectors = repulsion_directions * (repulsion_magnitude[:, None] * affected_speeds[:, None])

    # Accumulate forces
    avoid_accumulator[affected_indices] += repulsion_vectors


def _clamp_speeds(
    velocities: np.ndarray,
    original_speeds: np.ndarray
) -> np.ndarray:
    """
    Clamp velocity magnitudes to original speeds.

    Args:
        velocities: (N, 3) velocity vectors
        original_speeds: (N,) original speed magnitudes

    Returns:
        (N, 3) clamped velocities
    """
    current_speeds = np.linalg.norm(velocities, axis=1)
    exceeded_mask = current_speeds > original_speeds

    if np.any(exceeded_mask):
        scale = original_speeds[exceeded_mask] / current_speeds[exceeded_mask]
        velocities[exceeded_mask] *= scale[:, None]

    return velocities
