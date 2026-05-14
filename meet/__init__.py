"""meetscribe-offline — full meeting transcription pipeline.

Builds on meetscribe-record (capture primitives) and adds transcription,
diarization, summarization, PDF, GUI, sync.
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("meetscribe-offline")
except Exception:
    __version__ = "0.7.1"
