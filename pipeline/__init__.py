"""
pipeline — Unified Moshi + Bridge + Ditto integration layer.

Sub-modules
-----------
moshi_runner    : Runs Moshi inference; captures acoustic tokens + output audio.
bridge_runner   : Converts Mimi tokens → HuBERT-like features via the bridge model.
ditto_runner    : Drives Ditto StreamSDK with pre-computed audio features + image.
merge_audio_video : FFmpeg utility to mux audio into a silent video.
"""

from .moshi_runner import MoshiTokenRunner
from .bridge_runner import BridgeRunner
from .ditto_runner import DittoRunner
from .merge_audio_video import merge_audio_into_video

__all__ = [
    "MoshiTokenRunner",
    "BridgeRunner",
    "DittoRunner",
    "merge_audio_into_video",
]
