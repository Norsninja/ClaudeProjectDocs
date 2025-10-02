"""
Geometry helper utilities for avoidance and spatial calculations.

This module provides small, focused functions with no simulation
state. All helpers operate on float64 numpy arrays and are safe
to use in deterministic, per‑entity calculations.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the closest point on the line segment AB to point P.

    Handles degenerate segments (A == B) by returning A.

    Parameters
    - p: (3,) point
    - a: (3,) segment start
    - b: (3,) segment end

    Returns
    - (3,) closest point on segment AB to P
    """
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return a.copy()

    t = float(np.dot(p - a, ab)) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return a + t * ab


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Euclidean distance from point P to the line segment AB.

    Parameters
    - p: (3,) point
    - a: (3,) segment start
    - b: (3,) segment end

    Returns
    - distance as float64
    """
    c = closest_point_on_segment(p, a, b)
    return float(np.linalg.norm(p - c))


def signed_distance_to_sphere(point: np.ndarray, center: np.ndarray, radius: float) -> float:
    """
    Signed distance from point to the surface of a sphere.

    Negative means inside the sphere, positive outside.
    """
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return float(np.linalg.norm(point - center) - radius)


def project_above_plane_y(point: np.ndarray, y_level: float) -> Tuple[np.ndarray, float]:
    """
    Project a point vertically above a horizontal plane at y_level.

    Returns a new point with y >= y_level and the vertical penetration depth
    (positive when the original point was below the plane).
    """
    p = np.asarray(point, dtype=np.float64).copy()
    penetration = max(0.0, y_level - float(p[1]))
    if penetration > 0.0:
        p[1] = y_level
    return p, penetration

