import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movement_analysis import (  # noqa: E402
    NoPoseDetectedError,
    RepetitionCounter,
    compute_session_metrics,
)

FLEXED = 80.0  # well below the 115-degree flexed threshold
EXTENDED = 175.0  # well above the 145-degree extended threshold
DEAD_ZONE = 130.0  # between the two thresholds


def feed(counter: RepetitionCounter, angles: list[float]) -> None:
    for angle in angles:
        counter.update(angle)


def test_no_movement_counts_zero():
    counter = RepetitionCounter(smoothing_window=1, confirm_frames=1)
    feed(counter, [FLEXED] * 10)
    assert counter.repetition_count == 0


def test_single_clean_repetition_counts_one():
    counter = RepetitionCounter(smoothing_window=1, confirm_frames=1)
    feed(counter, [FLEXED, FLEXED, EXTENDED, EXTENDED, FLEXED, FLEXED])
    assert counter.repetition_count == 1


def test_three_repetitions_count_three():
    counter = RepetitionCounter(smoothing_window=1, confirm_frames=1)
    cycle = [EXTENDED, EXTENDED, FLEXED, FLEXED]
    feed(counter, [FLEXED, FLEXED] + cycle * 3)
    assert counter.repetition_count == 3


def test_does_not_count_until_returning_to_flexed():
    counter = RepetitionCounter(smoothing_window=1, confirm_frames=1)
    feed(counter, [FLEXED, FLEXED, EXTENDED, EXTENDED, EXTENDED])
    assert counter.repetition_count == 0


def test_dead_zone_readings_do_not_confirm_a_transition():
    counter = RepetitionCounter(smoothing_window=1, confirm_frames=1)
    feed(counter, [FLEXED] * 3 + [DEAD_ZONE] * 5 + [FLEXED] * 3)
    assert counter.repetition_count == 0


def test_default_smoothing_and_confirm_filter_a_brief_noise_spike():
    # Defaults: 5-frame moving average, 3-frame confirm. A single-frame
    # spike to an extended angle should never hold the smoothed average
    # past the extended threshold for 3 consecutive frames.
    counter = RepetitionCounter()
    feed(counter, [FLEXED] * 8 + [EXTENDED] + [FLEXED] * 8)
    assert counter.repetition_count == 0


def test_default_smoothing_and_confirm_count_a_sustained_repetition():
    counter = RepetitionCounter()
    feed(counter, [FLEXED] * 8 + [EXTENDED] * 8 + [FLEXED] * 8)
    assert counter.repetition_count == 1


def _make_frame(
    palm_y=0.7, nose_x=0.5, wrist_visibility=1.0, index_visibility=1.0, pinky_visibility=1.0,
    left_shoulder_x=0.4, right_shoulder_x=0.6, shoulder_y=0.5, elbow=(0.6, 0.4), wrist_x=0.62,
):
    return {
        "nose": {"x": nose_x, "y": 0.1, "visibility": 1.0},
        "left_ear": {"x": 0.35, "y": 0.1, "visibility": 1.0},
        "right_ear": {"x": 0.65, "y": 0.1, "visibility": 1.0},
        "left_shoulder": {"x": left_shoulder_x, "y": shoulder_y, "visibility": 1.0},
        "right_shoulder": {"x": right_shoulder_x, "y": shoulder_y, "visibility": 1.0},
        "elbow": {"x": elbow[0], "y": elbow[1], "visibility": 1.0},
        "wrist": {"x": wrist_x, "y": palm_y, "visibility": wrist_visibility},
        "index": {"x": wrist_x + 0.01, "y": palm_y, "visibility": index_visibility},
        "pinky": {"x": wrist_x - 0.01, "y": palm_y, "visibility": pinky_visibility},
    }


def _angle_frame(elbow_angle_degrees: float):
    """A frame whose shoulder-elbow-wrist angle is exactly `elbow_angle_degrees`.

    The elbow is fixed relative to the (right) shoulder; the wrist is
    placed at a fixed distance from the elbow, rotated by
    `elbow_angle_degrees` away from the elbow->shoulder direction, so the
    angle between elbow->shoulder and elbow->wrist is exactly the
    requested value.
    """
    shoulder = (0.6, 0.3)
    elbow = (0.6, 0.5)
    theta = math.radians(elbow_angle_degrees)
    wrist = (elbow[0] + 0.2 * math.sin(theta), elbow[1] - 0.2 * math.cos(theta))
    return {
        "nose": {"x": 0.5, "y": 0.1, "visibility": 1.0},
        "left_ear": {"x": 0.35, "y": 0.1, "visibility": 1.0},
        "right_ear": {"x": 0.65, "y": 0.1, "visibility": 1.0},
        "left_shoulder": {"x": 0.4, "y": shoulder[1], "visibility": 1.0},
        "right_shoulder": {"x": shoulder[0], "y": shoulder[1], "visibility": 1.0},
        "elbow": {"x": elbow[0], "y": elbow[1], "visibility": 1.0},
        "wrist": {"x": wrist[0], "y": wrist[1], "visibility": 1.0},
        "index": {"x": wrist[0] + 0.005, "y": wrist[1], "visibility": 1.0},
        "pinky": {"x": wrist[0] - 0.005, "y": wrist[1], "visibility": 1.0},
    }


def test_compute_session_metrics_raises_on_no_frames():
    with pytest.raises(NoPoseDetectedError):
        compute_session_metrics([], fps=30.0, selected_arm="right")


def test_compute_session_metrics_basic_shape():
    frames = [_angle_frame(FLEXED)] * 8 + [_angle_frame(EXTENDED)] * 8 + [_angle_frame(FLEXED)] * 8
    metrics = compute_session_metrics(frames, fps=15.0, selected_arm="right")

    assert metrics["repetition_count"] == 1
    assert metrics["session_duration_seconds"] == 1.6
    assert metrics["selected_arm"] == "right"
    assert metrics["measurement_basis"] == "palm"
    assert metrics["landmark_confidence"]["average"] == 1.0
    assert metrics["landmark_confidence"]["low_confidence"] is False
    assert metrics["maximum_observed_hand_height_relative_to_shoulder"] is not None
    assert metrics["estimated_2d_elbow_angle_at_peak_reach_degrees"] is not None


def test_compute_session_metrics_falls_back_to_wrist_when_hand_points_low_confidence():
    frames = [_make_frame(index_visibility=0.1, pinky_visibility=0.1)] * 5
    metrics = compute_session_metrics(frames, fps=30.0, selected_arm="right")
    assert metrics["measurement_basis"] == "wrist-based"


def test_compute_session_metrics_flags_low_confidence():
    low_visibility_frame = {
        "nose": {"x": 0.5, "y": 0.1, "visibility": 0.1},
        "left_ear": {"x": 0.35, "y": 0.1, "visibility": 0.1},
        "right_ear": {"x": 0.65, "y": 0.1, "visibility": 0.1},
        "left_shoulder": {"x": 0.4, "y": 0.5, "visibility": 0.1},
        "right_shoulder": {"x": 0.6, "y": 0.5, "visibility": 0.1},
        "elbow": {"x": 0.6, "y": 0.4, "visibility": 0.1},
        "wrist": {"x": 0.62, "y": 0.7, "visibility": 0.1},
        "index": {"x": 0.63, "y": 0.7, "visibility": 0.1},
        "pinky": {"x": 0.61, "y": 0.7, "visibility": 0.1},
    }
    metrics = compute_session_metrics([low_visibility_frame] * 5, fps=30.0, selected_arm="right")
    assert metrics["landmark_confidence"]["low_confidence"] is True
