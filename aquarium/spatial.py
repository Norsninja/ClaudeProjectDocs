"""
Spatial utility functions for 3D geometry.

Helper functions for distance calculations, collision detection,
and geometric queries in 3D space.
"""

import numpy as np
from typing import Tuple


def distance_3d(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 3D points.

    Args:
        pos_a: Position [x, y, z]
        pos_b: Position [x, y, z]

    Returns:
        Distance in meters
    """
    diff = pos_a - pos_b
    return np.sqrt(np.dot(diff, diff))


def distance_to_sphere(pos: np.ndarray, center: np.ndarray, radius: float) -> float:
    """
    Calculate signed distance from point to sphere surface.

    Positive = outside sphere, negative = inside sphere, zero = on surface.

    Args:
        pos: Point position [x, y, z]
        center: Sphere center [x, y, z]
        radius: Sphere radius

    Returns:
        Signed distance to sphere surface (positive = outside)
    """
    offset = pos - center
    distance_from_center = np.sqrt(np.dot(offset, offset))
    return distance_from_center - radius


def distance_to_plane(pos: np.ndarray, plane_y: float) -> float:
    """
    Calculate signed distance from point to horizontal plane.

    Positive = above plane, negative = below plane.

    Args:
        pos: Point position [x, y, z]
        plane_y: Y-coordinate of plane

    Returns:
        Signed distance to plane (positive = above)
    """
    return pos[1] - plane_y


def distance_to_cylinder(pos: np.ndarray, start: np.ndarray, end: np.ndarray, radius: float) -> float:
    """
    Calculate distance from point to infinite cylinder defined by line segment.

    Args:
        pos: Point position [x, y, z]
        start: Cylinder axis start [x, y, z]
        end: Cylinder axis end [x, y, z]
        radius: Cylinder radius

    Returns:
        Distance to cylinder surface (positive = outside, negative = inside)
    """
    axis = end - start
    axis_length_sq = np.dot(axis, axis)

    if axis_length_sq < 1e-9:
        # Degenerate cylinder (start == end), treat as sphere
        return distance_to_sphere(pos, start, radius)

    # Project point onto axis
    t = np.dot(pos - start, axis) / axis_length_sq
    t = np.clip(t, 0.0, 1.0)  # Clamp to segment

    # Closest point on axis
    closest = start + t * axis

    # Distance from point to axis
    offset = pos - closest
    distance_from_axis = np.sqrt(np.dot(offset, offset))

    return distance_from_axis - radius


def reflect_velocity(velocity: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Reflect velocity vector across surface normal.

    Uses formula: v' = v - 2 * (v · n) * n

    Args:
        velocity: Incident velocity vector [vx, vy, vz]
        normal: Surface normal (must be unit vector)

    Returns:
        Reflected velocity vector
    """
    return velocity - 2.0 * np.dot(velocity, normal) * normal


def normalize(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize vector to unit length.

    Args:
        vec: Vector to normalize [x, y, z]

    Returns:
        Tuple of (normalized vector, original length)
    """
    length = np.sqrt(np.dot(vec, vec))

    if length < 1e-9:
        # Zero vector, return arbitrary unit vector
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0

    return vec / length, length


def clamp_speed(velocity: np.ndarray, max_speed: float) -> np.ndarray:
    """
    Clamp velocity magnitude to maximum speed.

    Args:
        velocity: Velocity vector [vx, vy, vz]
        max_speed: Maximum allowed speed

    Returns:
        Velocity with clamped magnitude
    """
    speed_sq = np.dot(velocity, velocity)

    if speed_sq > max_speed * max_speed:
        # Rescale to max_speed
        speed = np.sqrt(speed_sq)
        return velocity * (max_speed / speed)

    return velocity
