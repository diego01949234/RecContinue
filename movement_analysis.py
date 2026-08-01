"""Local MediaPipe/OpenCV movement analysis for RecContinue (Phase 3).

Only the landmarks required by the three selected tests are read or displayed:
head/neck (nose and ears), the selected arm (shoulder, elbow, wrist), or the
palm (wrist, knuckle reference, and fingertips). Hips, knees, ankles, feet,
and full facial mesh are never read or displayed.

All processing happens locally against a video file already on disk;
nothing here makes a network call.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional

# OpenCV and MediaPipe are optional native runtimes. The patient intake,
# reporting, export, and clinician-review paths must remain usable even when
# a platform cannot initialize one of them, so load both only on Analyze.
cv2 = None
mp = None
mp_vision = None
BaseOptions = None


def _load_analysis_runtime() -> bool:
    """Load OpenCV and MediaPipe only when local analysis begins."""
    global cv2, mp, mp_vision, BaseOptions
    if mp is not None and mp_vision is not None and BaseOptions is not None:
        return True
    try:
        import cv2 as open_cv
        import mediapipe as media_pipe
        from mediapipe.tasks.python import vision as media_pipe_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions as media_pipe_base_options
    except ImportError:
        return False
    cv2 = open_cv
    mp = media_pipe
    mp_vision = media_pipe_vision
    BaseOptions = media_pipe_base_options
    return True

BASE_DIR = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

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

# Recorded clips can arrive at high camera resolution, but the 2D angle/ratio
# proxies used here need only a modest number of pixels. Capping the longest
# side keeps per-frame MediaPipe inference and video-writer encoding fast
# (this is the main lever for keeping a short clip's analysis to ~10s) without
# changing any landmark coordinate, since MediaPipe returns normalized [0,1]
# positions regardless of input size.
MAX_ANALYSIS_DIMENSION = 480


def _capped_frame_size(width: int, height: int, max_dimension: int = MAX_ANALYSIS_DIMENSION) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_dimension:
        return width, height
    scale = max_dimension / longest
    return max(2, int(width * scale)), max(2, int(height * scale))


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


def line_angle_to_horizontal_degrees(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the signed 2D angle of a line relative to the image horizontal.

    The ear-to-ear line is used as an observable head-orientation proxy. It is
    intentionally a camera-plane angle only and is not interpreted as posture,
    impairment, or a clinical finding.
    """
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def palm_closure_ratio(hand_landmarks: list[dict]) -> Optional[float]:
    """Estimate palm opening from MediaPipe Hands landmarks in one frame.

    The returned value is the mean fingertip-to-wrist distance, normalized by
    the wrist-to-middle-knuckle distance. A smaller value means the detected
    fingertips are closer to the wrist in the camera plane; it is not a grip
    strength, range-of-motion, or clinical measure.
    """
    if len(hand_landmarks) < 21:
        return None
    wrist, middle_mcp = hand_landmarks[0], hand_landmarks[9]
    scale = math.hypot(middle_mcp["x"] - wrist["x"], middle_mcp["y"] - wrist["y"])
    if scale <= 1e-6:
        return None
    tip_distances = [
        math.hypot(hand_landmarks[index]["x"] - wrist["x"], hand_landmarks[index]["y"] - wrist["y"])
        for index in (4, 8, 12, 16, 20)
    ]
    return sum(tip_distances) / len(tip_distances) / scale


def compute_palm_closure_metrics(hand_frames: list[list[dict]], fps: float, selected_arm: Arm) -> dict:
    """Aggregate local MediaPipe Hands observations for a palm open/close test."""
    ratios = [ratio for frame in hand_frames if (ratio := palm_closure_ratio(frame)) is not None]
    if not ratios:
        raise NoPoseDetectedError("No usable hand landmarks were detected in this video.")
    return {
        "data_status": "Camera-based measurement",
        "observation_module": "palm",
        "selected_arm": selected_arm,
        "detected_frame_count": len(ratios),
        "session_duration_seconds": round(len(ratios) / fps, 1) if fps else None,
        "palm_opening_ratio_min": round(min(ratios), 2),
        "palm_opening_ratio_max": round(max(ratios), 2),
        "palm_opening_ratio_change": round(max(ratios) - min(ratios), 2),
        "measurement_basis": "MediaPipe Hands fingertip-to-wrist distance ratio",
    }


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
    head_tilt_angles: list[float] = []
    head_turn_proxies: list[float] = []
    elbow_angles: list[float] = []
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
        if frame_elbow_angle is not None:
            elbow_angles.append(frame_elbow_angle)

        palm_height = (shoulder_y - palm["y"]) / shoulder_width
        palm_heights.append(palm_height)
        if palm_height > peak_height:
            peak_height = palm_height
            peak_frame = frame

        head_offsets.append((frame["nose"]["x"] - shoulder_mid_x) / shoulder_width)
        left_ear, right_ear = frame.get("left_ear"), frame.get("right_ear")
        if left_ear is not None and right_ear is not None:
            ear_mid_x = (left_ear["x"] + right_ear["x"]) / 2
            ear_width = abs(right_ear["x"] - left_ear["x"])
            if ear_width > 1e-6:
                head_turn_proxies.append((frame["nose"]["x"] - ear_mid_x) / ear_width)
            head_tilt_angles.append(
                line_angle_to_horizontal_degrees(
                    (left_ear["x"], left_ear["y"]),
                    (right_ear["x"], right_ear["y"]),
                )
            )

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
        "observation_module": "arm",
        "selected_arm": selected_arm,
        "repetition_count": rep_counter.repetition_count,
        "detected_frame_count": len(frames),
        "session_duration_seconds": round(len(frames) / fps, 1) if fps else None,
        "estimated_2d_elbow_angle_at_peak_reach_degrees": peak_elbow_angle,
        "observed_2d_elbow_angle_min_degrees": round(min(elbow_angles), 1) if elbow_angles else None,
        "observed_2d_elbow_angle_max_degrees": round(max(elbow_angles), 1) if elbow_angles else None,
        "observed_2d_elbow_angle_change_degrees": (
            round(max(elbow_angles) - min(elbow_angles), 1) if elbow_angles else None
        ),
        "maximum_observed_hand_height_relative_to_shoulder": (
            round(max(palm_heights), 2) if palm_heights else None
        ),
        "observed_head_position_variation": (
            round(max(head_offsets) - min(head_offsets), 2) if head_offsets else None
        ),
        "observed_head_tilt_angle_variation_degrees": (
            round(max(head_tilt_angles) - min(head_tilt_angles), 1) if head_tilt_angles else None
        ),
        "observed_head_turn_proxy_range": (
            round(max(head_turn_proxies) - min(head_turn_proxies), 2) if head_turn_proxies else None
        ),
        "landmark_confidence": {
            "average": average_confidence,
            "low_confidence": average_confidence < CONFIDENCE_THRESHOLD,
        },
        "measurement_basis": measurement_basis,
    }


HEAD_KEYS = ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder")


def compute_head_turn_metrics(frames: list[dict], fps: float) -> dict:
    """Aggregate per-frame head/ear landmarks into the head test's metrics.

    Unlike `compute_session_metrics`, this never depends on the arm, wrist,
    or hand being visible — a head-turning recording is not expected to show
    them, and requiring them silently dropped every frame of that test.
    """
    if not frames:
        raise NoPoseDetectedError("No pose was detected in this video.")

    head_offsets: list[float] = []
    head_tilt_angles: list[float] = []
    head_turn_proxies: list[float] = []
    visibilities: list[float] = []

    for frame in frames:
        left_shoulder, right_shoulder = frame["left_shoulder"], frame["right_shoulder"]
        shoulder_mid_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"]) or 1e-6
        head_offsets.append((frame["nose"]["x"] - shoulder_mid_x) / shoulder_width)

        left_ear, right_ear = frame["left_ear"], frame["right_ear"]
        ear_mid_x = (left_ear["x"] + right_ear["x"]) / 2
        ear_width = abs(right_ear["x"] - left_ear["x"])
        if ear_width > 1e-6:
            head_turn_proxies.append((frame["nose"]["x"] - ear_mid_x) / ear_width)
        head_tilt_angles.append(
            line_angle_to_horizontal_degrees(
                (left_ear["x"], left_ear["y"]),
                (right_ear["x"], right_ear["y"]),
            )
        )

        for key in HEAD_KEYS:
            lm = frame.get(key)
            if lm is not None:
                visibilities.append(lm.get("visibility", 1.0))

    average_confidence = round(sum(visibilities) / len(visibilities), 2) if visibilities else 0.0

    return {
        "data_status": "Camera-based measurement",
        "observation_module": "head",
        "detected_frame_count": len(frames),
        "session_duration_seconds": round(len(frames) / fps, 1) if fps else None,
        "observed_head_turn_proxy_range": (
            round(max(head_turn_proxies) - min(head_turn_proxies), 2) if head_turn_proxies else None
        ),
        "observed_head_position_variation": (
            round(max(head_offsets) - min(head_offsets), 2) if head_offsets else None
        ),
        "observed_head_tilt_angle_variation_degrees": (
            round(max(head_tilt_angles) - min(head_tilt_angles), 1) if head_tilt_angles else None
        ),
        "landmark_confidence": {
            "average": average_confidence,
            "low_confidence": average_confidence < CONFIDENCE_THRESHOLD,
        },
        "measurement_basis": "head/ear landmarks",
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


def frame_head_turn_proxy(frame_data: Optional[dict]) -> Optional[float]:
    """Compute the head-turn proxy for one frame's nose/ear landmarks.

    Returns ``None`` when the frame has no detected pose or an ear/nose
    landmark is unavailable. Mirrors `frame_elbow_angle`'s per-frame shape
    so a live session can report a value every frame, not only in
    aggregate the way `compute_head_turn_metrics` does.
    """
    if frame_data is None:
        return None
    nose = frame_data.get("nose")
    left_ear, right_ear = frame_data.get("left_ear"), frame_data.get("right_ear")
    if nose is None or left_ear is None or right_ear is None:
        return None
    ear_mid_x = (left_ear["x"] + right_ear["x"]) / 2
    ear_width = abs(right_ear["x"] - left_ear["x"])
    if ear_width <= 1e-6:
        return None
    return round((nose["x"] - ear_mid_x) / ear_width, 2)


def _require_cv() -> None:
    if not _load_analysis_runtime():
        raise MovementAnalysisUnavailableError(
            "opencv-python and mediapipe are not installed. Run `pip install -r requirements.txt`."
        )


def _require_hand_runtime() -> None:
    """Ensure the optional local Hand Landmarker model exists for palm testing."""
    if not _load_analysis_runtime():
        raise MovementAnalysisUnavailableError(
            "opencv-python and mediapipe are not installed. Run `pip install -r requirements.txt`."
        )
    if not HAND_MODEL_PATH.exists():
        raise MovementAnalysisUnavailableError(
            f"MediaPipe Hand Landmarker model not found at {HAND_MODEL_PATH}. See models/README.md "
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


def _draw_head_overlay(frame, frame_data: dict, width: int, height: int) -> None:
    """Draw only the nose/ear points and the ear-to-ear line for head (no arm)."""

    def px(point: dict) -> tuple[int, int]:
        return int(point["x"] * width), int(point["y"] * height)

    cv2.line(frame, px(frame_data["left_ear"]), px(frame_data["right_ear"]), BRAND_TEAL, 2)
    for key in ("nose", "left_ear", "right_ear"):
        cv2.circle(frame, px(frame_data[key]), 5, BRAND_CORAL, -1)


def _draw_hand_overlay(frame, hand_landmarks: list[dict], width: int, height: int) -> None:
    """Draw only palm/wrist and fingertips needed for the open-close measure."""
    def px(point: dict) -> tuple[int, int]:
        return int(point["x"] * width), int(point["y"] * height)

    wrist = hand_landmarks[0]
    cv2.circle(frame, px(wrist), 6, BRAND_TEAL, -1)
    for tip_index in (4, 8, 12, 16, 20):
        tip = hand_landmarks[tip_index]
        cv2.line(frame, px(wrist), px(tip), BRAND_TEAL, 2)
        cv2.circle(frame, px(tip), 5, BRAND_CORAL, -1)


PROGRESS_YIELD_EVERY_N_FRAMES = 5


def _analyze_palm_video(video_path: str, selected_arm: Arm, output_path: Optional[str] = None):
    """Analyze a palm opening/closing recording with local MediaPipe Hands."""
    _require_hand_runtime()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = _capped_frame_size(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if output_path is None:
        output_path = str(pathlib.Path(video_path).with_name(pathlib.Path(video_path).stem + "_palm_annotated.mp4"))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    base_options = BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    hand_frames: list[list[dict]] = []
    frame_index = 0
    try:
        while True:
            ok, bgr_frame = cap.read()
            if not ok:
                break
            bgr_frame = cv2.resize(bgr_frame, (width, height))
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, int(frame_index * (1000.0 / fps)))
            current_opening = None
            if result.hand_landmarks:
                hand_landmarks = [
                    {"x": landmark.x, "y": landmark.y, "visibility": 1.0}
                    for landmark in result.hand_landmarks[0]
                ]
                if len(hand_landmarks) >= 21:
                    _draw_hand_overlay(bgr_frame, hand_landmarks, width, height)
                    hand_frames.append(hand_landmarks)
                    current_opening = palm_closure_ratio(hand_landmarks)
            writer.write(bgr_frame)
            if frame_index % PROGRESS_YIELD_EVERY_N_FRAMES == 0:
                yield {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "observation_module": "palm",
                    "current_opening_ratio": current_opening,
                }
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        landmarker.close()

    yield output_path, compute_palm_closure_metrics(hand_frames, fps, selected_arm)


def _analyze_head_video(video_path: str, output_path: Optional[str] = None):
    """Analyze a head/neck-turning recording. Only nose/ear/shoulder landmarks
    are read; the arm, wrist, and hand are never required, since a head-turn
    recording has no reason to show them (unlike the shared pose pipeline this
    replaced, which silently dropped every frame here for lacking a palm)."""
    _require_cv()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = _capped_frame_size(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    if output_path is None:
        output_path = str(pathlib.Path(video_path).with_name(pathlib.Path(video_path).stem + "_head_annotated.mp4"))
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
            bgr_frame = cv2.resize(bgr_frame, (width, height))
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * (1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_turn = None
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                frame_data = {
                    "nose": _landmark_point(landmarks, NOSE),
                    "left_ear": _landmark_point(landmarks, LEFT_EAR),
                    "right_ear": _landmark_point(landmarks, RIGHT_EAR),
                    "left_shoulder": _landmark_point(landmarks, LEFT_SHOULDER),
                    "right_shoulder": _landmark_point(landmarks, RIGHT_SHOULDER),
                }
                _draw_head_overlay(bgr_frame, frame_data, width, height)
                frames_data.append(frame_data)
                ear_width = abs(frame_data["right_ear"]["x"] - frame_data["left_ear"]["x"])
                if ear_width > 1e-6:
                    ear_mid_x = (frame_data["left_ear"]["x"] + frame_data["right_ear"]["x"]) / 2
                    current_turn = round((frame_data["nose"]["x"] - ear_mid_x) / ear_width, 2)

            writer.write(bgr_frame)
            if frame_index % PROGRESS_YIELD_EVERY_N_FRAMES == 0:
                yield {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "observation_module": "head",
                    "current_turn_proxy": current_turn,
                }
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        landmarker.close()

    if not frames_data:
        raise NoPoseDetectedError("No pose was detected in this video.")

    yield output_path, compute_head_turn_metrics(frames_data, fps)


def _analyze_arm_video(video_path: str, selected_arm: Arm, output_path: Optional[str] = None):
    """Analyze an elbow-bending recording with the full arm pose pipeline."""
    _require_cv()
    indices = arm_landmark_indices(selected_arm)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = _capped_frame_size(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    if output_path is None:
        output_path = str(pathlib.Path(video_path).with_name(pathlib.Path(video_path).stem + "_annotated.mp4"))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    base_options = BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO)
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    frames_data: list[dict] = []
    live_rep_counter = RepetitionCounter()
    frame_index = 0
    try:
        while True:
            ok, bgr_frame = cap.read()
            if not ok:
                break
            bgr_frame = cv2.resize(bgr_frame, (width, height))
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * (1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_data = None
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
            current_angle = frame_elbow_angle(frame_data, selected_arm)
            live_rep_counter.update(current_angle)
            if frame_index % PROGRESS_YIELD_EVERY_N_FRAMES == 0:
                yield {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "observation_module": "arm",
                    "current_angle": current_angle,
                    "reps_so_far": live_rep_counter.repetition_count,
                }
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        landmarker.close()

    if not frames_data:
        raise NoPoseDetectedError("No pose was detected in this video.")

    metrics = compute_session_metrics(frames_data, fps, selected_arm)
    yield output_path, metrics


def analyze_video(
    video_path: str,
    selected_arm: Arm,
    output_path: Optional[str] = None,
    observation_module: str = "arm",
):
    """Dispatch to the analysis pipeline matching the selected observation
    module. Each of the three modules (head/palm/arm) has its own dedicated
    function reading only the landmarks it needs, so choosing one always
    runs the matching detection rather than falling back to a shared,
    arm-centric pipeline.

    Each intermediate yielded item is a live-progress dict; the final item
    is the existing ``(annotated_video_path, metrics_dict)`` tuple.
    """
    if observation_module == "palm":
        yield from _analyze_palm_video(video_path, selected_arm, output_path)
    elif observation_module == "head":
        yield from _analyze_head_video(video_path, output_path)
    else:
        yield from _analyze_arm_video(video_path, selected_arm, output_path)
