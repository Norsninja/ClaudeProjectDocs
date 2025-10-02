import numpy as np

from aquarium.geometry import (
    closest_point_on_segment,
    point_to_segment_distance,
    signed_distance_to_sphere,
    project_above_plane_y,
)


def test_closest_point_on_segment_midpoint():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 0.0])
    c = closest_point_on_segment(p, a, b)
    assert np.allclose(c, np.array([1.0, 0.0, 0.0]))


def test_closest_point_on_segment_clamps_to_start_end():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    # Before start
    p1 = np.array([-2.0, 0.5, 0.0])
    c1 = closest_point_on_segment(p1, a, b)
    assert np.allclose(c1, a)
    # Beyond end
    p2 = np.array([3.0, -0.25, 0.0])
    c2 = closest_point_on_segment(p2, a, b)
    assert np.allclose(c2, b)


def test_point_to_segment_distance_matches_manual():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    p = np.array([1.0, 0.5, 0.0])
    d = point_to_segment_distance(p, a, b)
    assert np.isclose(d, 1.0)


def test_degenerate_segment():
    a = np.array([2.0, -1.0, 3.0])
    b = a.copy()
    p = np.array([3.0, 4.0, 5.0])
    c = closest_point_on_segment(p, a, b)
    assert np.allclose(c, a)
    d = point_to_segment_distance(p, a, b)
    assert np.isfinite(d) and d >= 0.0


def test_signed_distance_to_sphere():
    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    outside = np.array([3.0, 0.0, 0.0])
    inside = np.array([1.0, 0.0, 0.0])
    assert np.isclose(signed_distance_to_sphere(outside, center, radius), 1.0)
    assert np.isclose(signed_distance_to_sphere(inside, center, radius), -1.0)


def test_project_above_plane_y():
    p = np.array([0.0, -3.0, 0.0])
    y_level = -1.0
    p2, pen = project_above_plane_y(p, y_level)
    assert np.isclose(pen, 2.0)
    assert np.isclose(p2[1], y_level)
    # No change when already above
    p3 = np.array([0.0, 0.5, 0.0])
    q, pen2 = project_above_plane_y(p3, y_level)
    assert np.isclose(pen2, 0.0)
    assert np.allclose(q, p3)

