"""Test-only stand-ins for MediaPipe's optional native import tree.

The suite exercises RecContinue's movement logic with synthetic landmarks;
the video pipeline itself replaces these objects with focused fakes. Keeping
the import local to pytest avoids MediaPipe's platform font discovery during
unit-test collection.
"""
from __future__ import annotations

import sys
import types


def _install_mediapipe_stub() -> None:
    if "mediapipe" in sys.modules:
        return

    mediapipe = types.ModuleType("mediapipe")
    tasks = types.ModuleType("mediapipe.tasks")
    python = types.ModuleType("mediapipe.tasks.python")
    vision = types.ModuleType("mediapipe.tasks.python.vision")
    base_options = types.ModuleType("mediapipe.tasks.python.core.base_options")

    vision.RunningMode = types.SimpleNamespace(VIDEO="video")
    vision.PoseLandmarkerOptions = lambda **kwargs: kwargs
    vision.PoseLandmarker = types.SimpleNamespace()
    base_options.BaseOptions = lambda **kwargs: kwargs

    mediapipe.tasks = tasks
    tasks.python = python
    python.vision = vision

    sys.modules.update(
        {
            "mediapipe": mediapipe,
            "mediapipe.tasks": tasks,
            "mediapipe.tasks.python": python,
            "mediapipe.tasks.python.vision": vision,
            "mediapipe.tasks.python.core": types.ModuleType("mediapipe.tasks.python.core"),
            "mediapipe.tasks.python.core.base_options": base_options,
        }
    )


_install_mediapipe_stub()
