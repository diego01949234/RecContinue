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


import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import movement_analysis as ma


def test_analyze_video_is_a_generator_function():
    assert inspect.isgeneratorfunction(ma.analyze_video)


class _FakeLandmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def _fake_pose_landmarks():
    fixed = {
        ma.NOSE: (0.5, 0.2), ma.LEFT_EAR: (0.45, 0.2), ma.RIGHT_EAR: (0.55, 0.2),
        ma.LEFT_SHOULDER: (0.4, 0.3), ma.RIGHT_SHOULDER: (0.6, 0.3),
        ma.RIGHT_ELBOW: (0.6, 0.5), ma.RIGHT_WRIST: (0.6, 0.7),
        ma.RIGHT_INDEX: (0.61, 0.71), ma.RIGHT_PINKY: (0.62, 0.72), ma.RIGHT_THUMB: (0.59, 0.71),
        ma.LEFT_ELBOW: (0.4, 0.5), ma.LEFT_WRIST: (0.4, 0.7),
        ma.LEFT_INDEX: (0.39, 0.71), ma.LEFT_PINKY: (0.38, 0.72), ma.LEFT_THUMB: (0.41, 0.71),
    }
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(23)]
    for idx, (x, y) in fixed.items():
        landmarks[idx] = _FakeLandmark(x, y)
    return [landmarks]


class _FakeCapture:
    def __init__(self, total_frames=12):
        self._remaining = total_frames
        self._total = total_frames

    def isOpened(self):
        return True

    def get(self, prop_id):
        return {
            "fps": 30.0,
            "width": 64,
            "height": 48,
            "frame_count": self._total,
        }.get(prop_id, 0)

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        pass


class _FakeLandmarker:
    def detect_for_video(self, mp_image, timestamp_ms):
        return SimpleNamespace(pose_landmarks=_fake_pose_landmarks())

    def close(self):
        pass


def test_analyze_video_yields_progress_before_final_result(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS="fps",
        CAP_PROP_FRAME_WIDTH="width",
        CAP_PROP_FRAME_HEIGHT="height",
        CAP_PROP_FRAME_COUNT="frame_count",
        COLOR_BGR2RGB="bgr_to_rgb",
        VideoCapture=lambda path: _FakeCapture(total_frames=12),
        VideoWriter=lambda *a, **kw: MagicMock(),
        VideoWriter_fourcc=lambda *args: 0,
        cvtColor=lambda frame, conversion: frame,
        circle=lambda *args, **kwargs: None,
        line=lambda *args, **kwargs: None,
    )
    fake_mp = SimpleNamespace(
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    fake_vision = SimpleNamespace(
        RunningMode=SimpleNamespace(VIDEO="video"),
        PoseLandmarkerOptions=lambda **kwargs: kwargs,
        PoseLandmarker=SimpleNamespace(create_from_options=lambda options: _FakeLandmarker()),
    )
    with patch.object(ma, "_require_cv", lambda: None), \
         patch.multiple(ma, cv2=fake_cv2, mp=fake_mp, mp_vision=fake_vision, BaseOptions=lambda **kwargs: kwargs, create=True):
        items = list(ma.analyze_video("fake.mp4", "right", output_path=output_path))

    progress_items = [item for item in items if isinstance(item, dict)]
    final_items = [item for item in items if not isinstance(item, dict)]

    assert len(progress_items) >= 1
    assert progress_items[0]["frame_index"] == 0
    assert progress_items[0]["total_frames"] == 12
    assert progress_items[0]["current_angle"] == 180.0

    assert len(final_items) == 1
    annotated_path, metrics = final_items[0]
    assert annotated_path == output_path
    assert metrics["repetition_count"] == 0
    assert metrics["selected_arm"] == "right"
