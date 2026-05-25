"""Sample-accurate dual-source capture via a PulseAudio null-sink combiner.

Architecture:

    mic ──► remap[front-left] ──► loopback ──┐
                                              ▼
                          meet_capture_<pid> (stereo null-sink)
                                              ▲
    G935.monitor ──► remap[front-right] ──► loopback ──┘

    meet_capture_<pid>.monitor ──► parec ──► stereo WAV

Why this works:
- A single ``parec`` client consumes the null-sink's monitor. There is no
  second PulseAudio client to set up, so the 1.5–3 s inter-client
  establishment delay that wrecked the previous designs is gone.
- Mic and system audio meet inside PulseAudio's mixer, which runs on one
  internal clock. L and R channels of the captured stereo are therefore
  sample-accurate by construction — no offset to detect, no merge to do.
- The user's default sink is untouched: apps keep playing to their normal
  output (e.g. the headset), and the same sink's existing ``.monitor`` is
  the source we route into the null-sink. No app reconfiguration needed.
- All transient PA modules are unloaded in ``stop()``, restoring the
  system audio state exactly. A best-effort orphan cleanup at ``start()``
  catches modules left behind by a previous crashed session.

Pause/resume:
- Reader thread skips writes while paused. The parec process keeps
  running so the null-sink keeps draining; otherwise PulseAudio would
  back-pressure on the loopbacks and create unbounded latency.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DRAIN_SECONDS: int = 2

SAMPLE_RATE = 16_000
READ_BUFSIZE = 8192
FRAME_BYTES = 4  # 2 channels × s16le

# Module-name prefix used for the null-sink and remap sources. Includes the
# PID so concurrent sessions don't collide; also used by the orphan cleanup.
_MODULE_PREFIX = "meetscribe"


@dataclass
class RecordingStatus:
    is_alive: bool
    elapsed_seconds: float
    file_size_bytes: int
    restart_count: int
    failed: bool
    fail_reason: str = ""
    paused: bool = False


def _pactl(*args: str) -> str:
    """Run a pactl command, return stdout. Raises CalledProcessError on failure."""
    result = subprocess.run(
        ["pactl", *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _try_pactl(*args: str) -> None:
    """Run pactl, swallow failures (used for best-effort cleanup)."""
    subprocess.run(["pactl", *args], capture_output=True)


def _cleanup_orphan_modules() -> None:
    """Unload any leftover ``meetscribe_*`` PA modules from prior crashed runs.

    Walks ``pactl list short modules`` and unloads any module whose argument
    string mentions our prefix. Idempotent and safe to call before each
    session.
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "modules"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError:
        return
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        mod_id = parts[0]
        argstr = parts[2] if len(parts) >= 3 else ""
        if _MODULE_PREFIX in argstr:
            _try_pactl("unload-module", mod_id)


class RecordingSession:
    def __init__(
        self,
        output_file: Path,
        mic_device: str | int,
        monitor_device: str | int,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.output_file = Path(output_file)
        self.mic_source: str = str(mic_device)
        self.monitor_source: str = str(monitor_device)
        self._sr = sample_rate
        self._recording = False
        self._paused = False
        self._failed = False
        self._fail_reason = ""
        self._start_time: Optional[float] = None
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._wav: Optional[wave.Wave_write] = None
        self._stderr_log = self.output_file.with_suffix(".parec.log")
        self._stderr_fp = None
        self._pa_modules: list[int] = []
        self._sink_name = ""

    # ------------------------------------------------------------------
    # PulseAudio routing

    def _load_pa_modules(self) -> None:
        """Load the null-sink + remap-source + loopback modules.

        Module IDs are appended in load order; ``_unload_pa_modules`` walks
        them in reverse so dependencies are removed cleanly.
        """
        pid = os.getpid()
        self._sink_name = f"{_MODULE_PREFIX}_capture_{pid}"
        mic_remap = f"{_MODULE_PREFIX}_mic_L_{pid}"
        sys_remap = f"{_MODULE_PREFIX}_sys_R_{pid}"

        try:
            # 1. Stereo null-sink that the recording is captured from.
            mod = _pactl(
                "load-module", "module-null-sink",
                f"sink_name={self._sink_name}",
                f"rate={self._sr}",
                "channels=2",
                "channel_map=front-left,front-right",
                "sink_properties=device.description=Meetscribe",
            )
            self._pa_modules.append(int(mod))

            # 2. Mic → mono virtual source mapped to front-left.
            # Default remix=yes is required here: with remix=no PulseAudio
            # only forwards channels whose *names* match between master and
            # remapped device. The mic master is labeled "mono" while we
            # want the output labeled "front-left" — those names don't match
            # under remix=no, and the remapped source delivers pure silence.
            mod = _pactl(
                "load-module", "module-remap-source",
                f"source_name={mic_remap}",
                f"master={self.mic_source}",
                "channels=1",
                "channel_map=front-left",
            )
            self._pa_modules.append(int(mod))

            # 3. Loopback that mono source into the stereo null-sink.
            mod = _pactl(
                "load-module", "module-loopback",
                f"source={mic_remap}",
                f"sink={self._sink_name}",
                "latency_msec=20",
                f"rate={self._sr}",
            )
            self._pa_modules.append(int(mod))

            # 4. System monitor → mono virtual source mapped to front-right.
            # Default remix=yes mixes the stereo monitor down to mono before
            # placing it on front-right of our null-sink — intentional, we
            # don't care about preserving L/R of the original system audio.
            mod = _pactl(
                "load-module", "module-remap-source",
                f"source_name={sys_remap}",
                f"master={self.monitor_source}",
                "channels=1",
                "channel_map=front-right",
            )
            self._pa_modules.append(int(mod))

            # 5. Loopback that mono source into the stereo null-sink.
            mod = _pactl(
                "load-module", "module-loopback",
                f"source={sys_remap}",
                f"sink={self._sink_name}",
                "latency_msec=20",
                f"rate={self._sr}",
            )
            self._pa_modules.append(int(mod))
        except subprocess.CalledProcessError as e:
            self._unload_pa_modules()
            stderr = (e.stderr or "").strip() or str(e)
            raise RuntimeError(f"PulseAudio routing setup failed: {stderr}")

    def _unload_pa_modules(self) -> None:
        for mod_id in reversed(self._pa_modules):
            _try_pactl("unload-module", str(mod_id))
        self._pa_modules.clear()

    # ------------------------------------------------------------------
    # Capture lifecycle

    def start(self) -> None:
        _cleanup_orphan_modules()
        self._load_pa_modules()
        # Brief settle time so loopbacks are actually pumping data before
        # parec starts. Without this the first ~100 ms of the capture can
        # be silence even if the sources are active.
        time.sleep(0.2)

        self._wav = wave.open(str(self.output_file), "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self._sr)

        self._stderr_fp = open(self._stderr_log, "wb")
        self._proc = subprocess.Popen(
            [
                "parec",
                f"--device={self._sink_name}.monitor",
                f"--rate={self._sr}",
                "--channels=2",
                "--format=s16le",
                "--latency-msec=20",
                "--raw",
            ],
            stdout=subprocess.PIPE,
            stderr=self._stderr_fp,
        )
        self._recording = True
        self._start_time = time.monotonic()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        assert self._wav is not None
        leftover = b""
        try:
            while True:
                data = self._proc.stdout.read(READ_BUFSIZE)
                if not data:
                    break
                if self._paused:
                    leftover = b""
                    continue
                data = leftover + data
                aligned = len(data) - (len(data) % FRAME_BYTES)
                if aligned > 0:
                    self._wav.writeframesraw(data[:aligned])
                leftover = data[aligned:]
        except (BrokenPipeError, ValueError, OSError):
            pass

    def stop(self) -> Path:
        if self._paused:
            self._send_cont()
            self._paused = False
        self._recording = False

        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            except ProcessLookupError:
                pass

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=DRAIN_SECONDS)

        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass
            self._wav = None

        if self._stderr_fp is not None:
            try:
                self._stderr_fp.close()
            except OSError:
                pass
            self._stderr_fp = None

        # Always unload PA modules — even if something above failed, we
        # don't want to leave the user's audio routing modified.
        self._unload_pa_modules()

        self._write_session_meta()
        return self.output_file

    def _write_session_meta(self) -> None:
        meta_path = self.output_file.with_suffix(".session.json")
        meta = {
            "output_file": str(self.output_file),
            "mic_source": self.mic_source,
            "monitor_source": self.monitor_source,
            "parec_log": str(self._stderr_log),
            "architecture": "pa-null-sink-combiner",
            "failed": self._failed,
            "fail_reason": self._fail_reason,
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Pause / resume

    def _send_stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGSTOP)
            except ProcessLookupError:
                pass

    def _send_cont(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGCONT)
            except ProcessLookupError:
                pass

    def pause(self) -> None:
        if not self._paused:
            # We drop frames in the reader instead of SIGSTOP-ing parec
            # so the null-sink keeps draining; otherwise PA would back-
            # pressure on the loopbacks and grow unbounded latency.
            self._paused = True

    def resume(self) -> None:
        if self._paused:
            self._paused = False

    # ------------------------------------------------------------------
    # Status

    def status(self) -> RecordingStatus:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        size = self.output_file.stat().st_size if self.output_file.exists() else 0
        is_alive = (
            self._recording
            and self._proc is not None
            and self._proc.poll() is None
        )
        return RecordingStatus(
            is_alive=is_alive,
            elapsed_seconds=elapsed,
            file_size_bytes=size,
            restart_count=0,
            failed=self._failed,
            fail_reason=self._fail_reason,
            paused=self._paused,
        )


# ---------------------------------------------------------------------------
# Device helpers


def list_sources() -> list[dict]:
    result = subprocess.run(
        ["pactl", "list", "short", "sources"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    out: list[dict] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            out.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver": parts[2],
                    "sample_spec": parts[3],
                    "state": parts[4],
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                }
            )
    return out


def get_default_source() -> str:
    return _pactl("get-default-source")


def get_default_sink() -> str:
    """Return the .monitor source name of the default PulseAudio output sink."""
    return _pactl("get-default-sink") + ".monitor"


# ---------------------------------------------------------------------------
# Public factory


def create_session(
    output_dir: str | Path | None = None,
    virtual_sink: bool = False,
    mic: str | None = None,
    monitor: str | None = None,
) -> RecordingSession:
    """Create a new recording session.

    Session artefacts (WAV, parec log, session.json) all live inside a
    per-session ``meeting-YYYYMMDD-HHMMSS/`` subdir under *output_dir*.

    Args:
        output_dir: Root directory under which the session subdir is created.
                    Defaults to ``~/meetscribe-recordings/``.
        virtual_sink: Accepted for API compatibility; ignored — the null-sink
                      combiner architecture supersedes the legacy virtual-sink
                      mode.
        mic: PulseAudio source name for the microphone input.
             Defaults to the system default source.
        monitor: PulseAudio monitor source name for the system audio.
                 Defaults to ``<default-sink>.monitor``.
    """
    del virtual_sink
    base = Path(output_dir) if output_dir else Path.home() / "meetscribe-recordings"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_name = f"meeting-{ts}"
    session_dir = base / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    output_file = session_dir / f"{session_name}.wav"

    mic_device = mic if mic is not None else get_default_source()
    monitor_device = monitor if monitor is not None else get_default_sink()

    return RecordingSession(output_file, mic_device, monitor_device)


def check_prerequisites() -> list[str]:
    """Return a list of error strings; empty list means all good."""
    issues: list[str] = []
    try:
        subprocess.run(["parec", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        issues.append("parec is not available: sudo apt install pulseaudio-utils")
    try:
        subprocess.run(
            ["pactl", "info"], capture_output=True, check=True, timeout=2
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        issues.append("PulseAudio/PipeWire server is not running.")
    return issues


# Note: the `remix_session()` and offset-override env vars from the previous
# architecture are gone — there is no offset to tune now. Sync is sample-
# accurate by construction (single PA capture client).
