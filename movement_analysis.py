"""Local MediaPipe/OpenCV movement analysis for RecContinue (Phase 3).

Only the limited upper-body landmark subset described in SPEC.md section 6
is read or displayed: head (nose, ears), shoulders, the selected arm's
elbow/wrist, and an approximate palm point built from the selected arm's
wrist/index/pinky. Hips, knees, ankles, feet, full facial mesh, and
individual finger joints are never read or displayed.

All processing happens locally against a video file already on disk;
nothing here makes a network call.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False

BASE_DIR = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"

# MediaPipe Pose landmark indices used by RecContinue (SPEC.md section 6).
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22

Arm = Literal["left", "right"]

CONFIDENCE_THRESHOLD = 0.5
BRAND_TEAL = (154, 168, 31)  # BGR for #1FA89A
BRAND_CORAL = (104, 122, 255)  # BGR for #FF7A68
BRAND_MINT = (243, 247, 234)  # BGR for #EAF7F3


class NoPoseDetectedError(Exception):
    """Raised when no pose landmarks could be detected in any frame."""


class MovementAnalysisUnavailableError(Exception):
    """Raised when cv2/mediapipe or the pose landmarker model are unavailable."""


def arm_landmark_indices(selected_arm: Arm) -> dict[str, int]:
    if selected_arm == "left":
        return {"shoulder": LEFT_SHOULDER, "elbow": LEFT_ELBOW, "wrist": LEFT_WRIST,
                 "pinky": LEFT_PINKY, "index": LEFT_INDEX, "thumb": LEFT_THUMB}
    return {"shoulder": RIGHT_SHOULDER, "elbow": RIGHT_ELBOW, "wrist": RIGHT_WRIST,
             "pinky": RIGHT_PINKY, "index": RIGHT_INDEX, "thumb": RIGHT_THUMB}


def angle_2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Return the 2D angle in degrees at vertex b, formed by rays b->a and b->c."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a == 0 or mag_c == 0:
        return 0.0
    cos_angle = (ax * cx + ay * cy) / (mag_a * mag_c)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def approximate_palm_center(
    wrist: Optional[dict], index: Optional[dict], pinky: Optional[dict],
    min_visibility: float = CONFIDENCE_THRESHOLD,
) -> tuple[Optional[dict], bool]:
    """Approximate the palm center from wrist/index/pinky points.

    Returns (point, is_wrist_based). Falls back to the wrist point alone,
    labeled wrist-based, if index/pinky are missing or below the
    visibility threshold, per SPEC.md section 6.
    """
    if wrist is None:
        return None, True
    usable = [p for p in (index, pinky) if p is not None and p.get("visibility", 1.0) >= min_visibility]
    if len(usable) < 2:
        return {"x": wrist["x"], "y": wrist["y"]}, True
    xs = [wrist["x"]] + [p["x"] for p in usable]
    ys = [wrist["y"]] + [p["y"] for p in usable]
    return {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys)}, False


@dataclass
class RepetitionCounter:
    """Counts a repetition using an elbow-angle state machine (SPEC.md
    section 7A): a session starts flexed (elbow bent), the angle rises
    past `extended_above_degrees` during the reach, then drops back below
    `flexed_below_degrees` to complete one repetition.

    Angle readings are smoothed with a trailing moving average over
    `smoothing_window` frames before classification, and a candidate state
    must hold for `confirm_frames` consecutive readings before it is
    accepted, to prevent single-frame jitter from double-counting or
    missing repetitions. Readings between the two thresholds are a dead
    zone that neither confirms nor resets a pending transition.
    """

    flexed_below_degrees: float = 115.0
    extended_above_degrees: float = 145.0
    smoothing_window: int = 5
    confirm_frames: int = 3

    _angle_window: list[float] = field(default_factory=list, init=False, repr=False)
    _state: str = field(default="flexed", init=False, repr=False)
    _pending_state: Optional[str] = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)
    repetition_count: int = field(default=0, init=False)

    def update(self, elbow_angle_degrees: Optional[float]) -> None:
        if elbow_angle_degrees is None:
            return

        self._angle_window.append(elbow_angle_degrees)
        if len(self._angle_window) > self.smoothing_window:
            self._angle_window.pop(0)
        smoothed = sum(self._angle_window) / len(self._angle_window)

        if smoothed < self.flexed_below_degrees:
            candidate_state = "flexed"
        elif smoothed > self.extended_above_degrees:
            candidate_state = "extended"
        else:
            candidate_state = None  # dead zone between thresholds

        if candidate_state is None or candidate_state == self._state:
            self._pending_state = None
            self._pending_count = 0
            return

        if candidate_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = candidate_state
            self._pending_count = 1

        if self._pending_count >= self.confirm_frames:
            if self._state == "extended" and candidate_state == "flexed":
                self.repetition_count += 1
            self._state = candidate_state
            self._pending_state = None
            self._pending_count = 0


HEAD_AND_ARM_KEYS = ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                      "wrist", "index", "pinky")


def compute_session_metrics(frames: list[dict], fps: float, selected_arm: Arm) -> dict:
    """Aggregate per-frame landmark dicts into the five SPEC.md section 7 metrics.

    `frames` is a list of dicts, one per video frame in which a pose was
    detected, each mapping landmark name -> {"x", "y", "visibility"} in
    normalized (0-1) image coordinates, already resolved to the selected
    arm (keys: nose, left_ear, right_ear, left_shoulder, right_shoulder,
    elbow, wrist, index, pinky, thumb).
    """
    if not frames:
        raise NoPoseDetectedError("No pose was detected in this video.")

    rep_counter = RepetitionCounter()
    palm_heights: list[float] = []
    head_offsets: list[float] = []
    visibilities: list[float] = []
    measurement_basis = "palm"
    peak_frame = None
    peak_height = float("-inf")

    for frame in frames:
        left_shoulder, right_shoulder = frame["left_shoulder"], frame["right_shoulder"]
        shoulder_mid_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"]) or 1e-6
        shoulder_pt = left_shoulder if selected_arm == "left" else right_shoulder
        shoulder_y = shoulder_pt["y"]

        palm, is_wrist_based = approximate_palm_center(frame.get("wrist"), frame.get("index"), frame.get("pinky"))
        if is_wrist_based:
            measurement_basis = "wrist-based"
        if palm is None:
            continue

        elbow, wrist = frame.get("elbow"), frame.get("wrist")
        frame_elbow_angle = (
            angle_2d((shoulder_pt["x"], shoulder_pt["y"]), (elbow["x"], elbow["y"]), (wrist["x"], wrist["y"]))
            if elbow is not None and wrist is not None
            else None
        )
        rep_counter.update(frame_elbow_angle)

        palm_height = (shoulder_y - palm["y"]) / shoulder_width
        palm_heights.append(palm_height)
        if palm_height > peak_height:
            peak_height = palm_height
            peak_frame = frame

        head_offsets.append((frame["nose"]["x"] - shoulder_mid_x) / shoulder_width)

        for key in HEAD_AND_ARM_KEYS:
            lm = frame.get(key)
            if lm is not None:
                visibilities.append(lm.get("visibility", 1.0))

    peak_elbow_angle = None
    if peak_frame is not None:
        shoulder_key = "left_shoulder" if selected_arm == "left" else "right_shoulder"
        shoulder_pt = peak_frame[shoulder_key]
        peak_elbow_angle = round(
            angle_2d((shoulder_pt["x"], shoulder_pt["y"]),
                     (peak_frame["elbow"]["x"], peak_frame["elbow"]["y"]),
                     (peak_frame["wrist"]["x"], peak_frame["wrist"]["y"])),
            1,
        )

    average_confidence = round(sum(visibilities) / len(visibilities), 2) if visibilities else 0.0

    return {
        "data_status": "Camera-based measurement",
        "selected_arm": selected_arm,
        "repetition_count": rep_counter.repetition_count,
        "session_duration_seconds": round(len(frames) / fps, 1) if fps else None,
        "estimated_2d_elbow_angle_at_peak_reach_degrees": peak_elbow_angle,
        "maximum_observed_hand_height_relative_to_shoulder": (
            round(max(palm_heights), 2) if palm_heights else None
        ),
        "observed_head_position_variation": (
            round(max(head_offsets) - min(head_offsets), 2) if head_offsets else None
        ),
        "landmark_confidence": {
            "average": average_confidence,
            "low_confidence": average_confidence < CONFIDENCE_THRESHOLD,
        },
        "measurement_basis": measurement_basis,
    }


def frame_elbow_angle(frame_data: Optional[dict], selected_arm: Arm) -> Optional[float]:
    """Compute the 2D elbow angle for one frame's landmark data.

    Returns ``None`` when the frame has no detected pose or the elbow or
    wrist landmark is unavailable. Full-session metrics remain the
    authoritative source for the reported peak-reach angle.
    """
    if frame_data is None:
        return None
    shoulder_key = "left_shoulder" if selected_arm == "left" else "right_shoulder"
    shoulder_pt = frame_data[shoulder_key]
    elbow, wrist = frame_data.get("elbow"), frame_data.get("wrist")
    if elbow is None or wrist is None:
        return None
    return round(
        angle_2d(
            (shoulder_pt["x"], shoulder_pt["y"]),
            (elbow["x"], elbow["y"]),
            (wrist["x"], wrist["y"]),
        ),
        1,
    )


def _require_cv() -> None:
    if not _CV_AVAILABLE:
        raise MovementAnalysisUnavailableError(
            "opencv-python and mediapipe are not installed. Run `pip install -r requirements.txt`."
        )
    if not MODEL_PATH.exists():
        raise MovementAnalysisUnavailableError(
            f"MediaPipe Pose Landmarker model not found at {MODEL_PATH}. See models/README.md "
            "for how to download it."
        )


def _landmark_point(landmarks, index: int) -> dict:
    lm = landmarks[index]
    return {"x": lm.x, "y": lm.y, "visibility": getattr(lm, "visibility", 1.0)}


def _draw_overlay(frame, frame_data: dict, palm_point: Optional[dict], width: int, height: int) -> None:
    """Draw only the selected landmarks/lines: head points, shoulder line,
    selected upper arm, selected forearm, and the approximate palm/wrist
    point (SPEC.md section 6)."""

    def px(point: dict) -> tuple[int, int]:
        return int(point["x"] * width), int(point["y"] * height)

    for key in ("nose", "left_ear", "right_ear"):
        cv2.circle(frame, px(frame_data[key]), 4, BRAND_MINT, -1)

    cv2.line(frame, px(frame_data["left_shoulder"]), px(frame_data["right_shoulder"]), BRAND_TEAL, 2)
    cv2.line(frame, px(frame_data["left_shoulder" if frame_data["_arm"] == "left" else "right_shoulder"]),
              px(frame_data["elbow"]), BRAND_TEAL, 2)
    cv2.line(frame, px(frame_data["elbow"]), px(frame_data["wrist"]), BRAND_TEAL, 2)

    if palm_point is not None:
        cv2.circle(frame, px(palm_point), 6, BRAND_CORAL, -1)


def analyze_video(video_path: str, selected_arm: Arm, output_path: Optional[str] = None) -> tuple[str, dict]:
    """Run local MediaPipe/OpenCV analysis on a video and return
    (annotated_video_path, metrics_dict).

    Raises NoPoseDetectedError if no pose was found in any frame, and
    MovementAnalysisUnavailableError if the required dependencies/model
    are not installed. Never fabricates a result for a video it could not
    analyze.
    """
    _require_cv()
    indices = arm_landmark_indices(selected_arm)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        output_path = str(pathlib.Path(video_path).with_name(pathlib.Path(video_path).stem + "_annotated.mp4"))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    base_options = BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO)
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    frames_data: list[dict] = []
    frame_index = 0
    try:
        while True:
            ok, bgr_frame = cap.read()
            if not ok:
                break
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * (1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                frame_data = {
                    "_arm": selected_arm,
                    "nose": _landmark_point(landmarks, NOSE),
                    "left_ear": _landmark_point(landmarks, LEFT_EAR),
                    "right_ear": _landmark_point(landmarks, RIGHT_EAR),
                    "left_shoulder": _landmark_point(landmarks, LEFT_SHOULDER),
                    "right_shoulder": _landmark_point(landmarks, RIGHT_SHOULDER),
                    "elbow": _landmark_point(landmarks, indices["elbow"]),
                    "wrist": _landmark_point(landmarks, indices["wrist"]),
                    "index": _landmark_point(landmarks, indices["index"]),
                    "pinky": _landmark_point(landmarks, indices["pinky"]),
                }
                palm_point, _ = approximate_palm_center(frame_data["wrist"], frame_data["index"], frame_data["pinky"])
                _draw_overlay(bgr_frame, frame_data, palm_point, width, height)
                frames_data.append(frame_data)

            writer.write(bgr_frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        landmarker.close()

    if not frames_data:
        raise NoPoseDetectedError("No pose was detected in this video.")

    metrics = compute_session_metrics(frames_data, fps, selected_arm)
    return output_path, metrics
