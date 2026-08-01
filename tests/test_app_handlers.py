import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

gradio_stub = types.ModuleType("gradio")
gradio_stub.SelectData = object
gradio_stub.update = lambda **kwargs: kwargs
sys.modules.setdefault("gradio", gradio_stub)

import app  # noqa: E402


def _fake_analyze_video(video_path, selected_arm):
    yield {"frame_index": 5, "total_frames": 12, "current_angle": 96.0, "reps_so_far": 1}
    yield "annotated.mp4", {"repetition_count": 1, "selected_arm": selected_arm}


def test_analyze_video_handler_yields_progress_then_result():
    with patch("app.analyze_video", side_effect=_fake_analyze_video):
        updates = list(app.analyze_video_handler("session.mp4", "right"))

    assert len(updates) == 2
    progress_update = updates[0]
    assert "frame 5/12" in progress_update[3]
    assert "96.0" in progress_update[3]
    assert "reps so far: 1" in progress_update[3]

    final_update = updates[1]
    assert final_update[0] == "annotated.mp4"
    assert final_update[2] == {"repetition_count": 1, "selected_arm": "right"}
    assert final_update[3] == ""


def test_analyze_video_handler_no_video_selected():
    updates = list(app.analyze_video_handler(None, "right"))
    assert len(updates) == 1
    assert updates[0][0] is None
    assert "No video selected" in updates[0][1]


def test_analyze_video_handler_reports_unavailable_error():
    def _raises(video_path, selected_arm):
        raise app.MovementAnalysisUnavailableError("mediapipe not installed")
        yield  # pragma: no cover

    with patch("app.analyze_video", side_effect=_raises):
        updates = list(app.analyze_video_handler("session.mp4", "right"))

    assert len(updates) == 1
    assert updates[0][0] is None
    assert "mediapipe not installed" in updates[0][1]


def test_use_synthetic_metrics_fallback_clears_progress_text():
    metrics, markdown, progress = app.use_synthetic_metrics_fallback()
    assert metrics == app.SYNTHETIC_METRICS
    assert progress == ""
