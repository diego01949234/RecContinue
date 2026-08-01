import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movement_analysis import frame_elbow_angle  # noqa: E402


RIGHT_ARM_FRAME = {
    "left_shoulder": {"x": 0.4, "y": 0.3, "visibility": 1.0},
    "right_shoulder": {"x": 0.6, "y": 0.3, "visibility": 1.0},
    "elbow": {"x": 0.6, "y": 0.5, "visibility": 1.0},
    "wrist": {"x": 0.6, "y": 0.7, "visibility": 1.0},
}


def test_frame_elbow_angle_computes_for_right_arm():
    angle = frame_elbow_angle(RIGHT_ARM_FRAME, "right")
    assert angle == 180.0


def test_frame_elbow_angle_returns_none_when_frame_is_none():
    assert frame_elbow_angle(None, "right") is None


def test_frame_elbow_angle_returns_none_when_elbow_missing():
    frame = dict(RIGHT_ARM_FRAME)
    frame["elbow"] = None
    assert frame_elbow_angle(frame, "right") is None


def test_frame_elbow_angle_returns_none_when_wrist_missing():
    frame = dict(RIGHT_ARM_FRAME)
    frame["wrist"] = None
    assert frame_elbow_angle(frame, "right") is None


def test_frame_elbow_angle_uses_left_shoulder_for_left_arm():
    frame = {
        "left_shoulder": {"x": 0.4, "y": 0.3, "visibility": 1.0},
        "right_shoulder": {"x": 0.6, "y": 0.3, "visibility": 1.0},
        "elbow": {"x": 0.4, "y": 0.5, "visibility": 1.0},
        "wrist": {"x": 0.4, "y": 0.7, "visibility": 1.0},
    }
    angle = frame_elbow_angle(frame, "left")
    assert angle == 180.0
