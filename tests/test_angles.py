import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movement_analysis import angle_2d, frame_head_turn_proxy, line_angle_to_horizontal_degrees  # noqa: E402


def test_straight_line_is_180_degrees():
    assert angle_2d((0, 1), (0, 0), (0, -1)) == 180.0


def test_right_angle_is_90_degrees():
    assert round(angle_2d((1, 0), (0, 0), (0, 1)), 4) == 90.0


def test_same_direction_is_0_degrees():
    assert angle_2d((1, 0), (0, 0), (2, 0)) == 0.0


def test_60_degree_angle():
    import math

    c = (math.cos(math.radians(60)), math.sin(math.radians(60)))
    assert round(angle_2d((1, 0), (0, 0), c), 4) == 60.0


def test_degenerate_zero_length_ray_returns_zero():
    assert angle_2d((0, 0), (0, 0), (1, 1)) == 0.0


def test_elbow_bend_example():
    # Shoulder above elbow, wrist bent back up toward shoulder: sharp bend.
    shoulder = (0.5, 0.2)
    elbow = (0.5, 0.5)
    wrist = (0.6, 0.25)
    angle = angle_2d(shoulder, elbow, wrist)
    assert 0 < angle < 90


def test_ear_line_angle_is_zero_when_level_and_45_when_tilted():
    assert line_angle_to_horizontal_degrees((0, 0), (1, 0)) == 0.0
    assert round(line_angle_to_horizontal_degrees((0, 0), (1, 1)), 1) == 45.0


HEAD_FRAME = {
    "nose": {"x": 0.55, "y": 0.2, "visibility": 1.0},
    "left_ear": {"x": 0.4, "y": 0.2, "visibility": 1.0},
    "right_ear": {"x": 0.6, "y": 0.2, "visibility": 1.0},
}


def test_frame_head_turn_proxy_computes_offset():
    assert frame_head_turn_proxy(HEAD_FRAME) == 0.25


def test_frame_head_turn_proxy_returns_none_when_frame_is_none():
    assert frame_head_turn_proxy(None) is None


def test_frame_head_turn_proxy_returns_none_when_ear_missing():
    frame = dict(HEAD_FRAME)
    frame["left_ear"] = None
    assert frame_head_turn_proxy(frame) is None
