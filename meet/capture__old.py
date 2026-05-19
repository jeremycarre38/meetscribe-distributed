"""Backward-compat shim — capture lives in `meetscribe-record` since 0.5.0.

This module re-exports everything from `meet_record.capture` so that
existing code (and any third-party importers) continue to work with
`from meet.capture import ...`. All real implementation lives upstream
in the meetscribe-record package.

To install just the capture primitives without meetscribe-offline's
heavy deps:

    pip install meetscribe-record

To get the full pipeline:

    pip install meetscribe-offline
"""

from meet_record.capture import *  # noqa: F401,F403
from meet_record.capture import (  # noqa: F401  re-exported names
    DRAIN_SECONDS,
    RecordingSession,
    create_session,
    check_prerequisites,
    list_sources,
    get_default_sink,
    get_default_source,
)
