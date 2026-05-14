# Changelog

## v0.7.1 — 2026-05-14

### Fixes

- **Transcription auto-falls back to CPU + `int8` when CUDA is
  unavailable** (#19, thanks @fadenb) — running `meet run` on a machine
  without a GPU (laptop, container without passthrough, CI runner)
  previously crashed with `ValueError: device='cuda' but CUDA is not
  available`. `TranscriptionConfig` now warns and falls back to
  `device=cpu`, downgrading `compute_type=float16` to `int8` (float16
  is unsupported on CPU). The model-load log line annotates whether
  CPU was forced (`--device cpu`) or auto-selected because no GPU was
  found, so the diagnostic distinction is preserved.

### Internals

- `TranscriptionConfig` gains an internal `_device_auto_fallback`
  flag set in `__post_init__` when the device is auto-flipped, so
  `_load_whisperx_asr_model` can label the load line accurately
  without re-sniffing torch at print time.

## v0.7.0 — 2026-05-08

### Features

- **Structured YAML frontmatter on every summary (schema_version 1)** —
  `.summary.md` now begins with a typed YAML frontmatter block carrying
  `participants`, `topics`, `action_items` (with assignee, task, due,
  status), `decisions` (text, topic), `language`, `duration`, and a
  `source` pointer. A matching `.frontmatter.json` sidecar is written
  next to it for tools that don't want to parse YAML. The schema is
  intentionally small in v1; downstream tools (e.g. the
  [vezir](https://github.com/pretyflaco/vezir) 0.2.0+ indexer) build
  richer derived views over this stable surface. See the README's
  "Structured frontmatter" section for the schema.
- **`meet ingest` subcommand** — re-extract structured frontmatter for
  one or more existing session directories. Idempotent: skips sessions
  whose `.summary.meta.json` already records `data_extracted: true`
  unless `--force` is passed. Accepts the standard summary-backend
  flags. `--dry-run` previews without invoking the LLM. `--no-pdf`
  skips PDF regeneration.
- **LLM contract: fenced JSON data block** — every summarization prompt
  (single-pass, two-pass formatter, and inline fallbacks) instructs
  the model to append exactly one fenced ```json block at the end of
  its output with the structured fields. Single source of truth: the
  Markdown body still drives the PDF, the JSON block populates the
  frontmatter. JSON is required to be in English even when the body is
  in another language so cross-language indexing works.

### Internals

- New module `meet/frontmatter.py`: schema, build/parse/validate, YAML
  render and read-back, and a `context_from_transcript()` helper that
  pulls `started_at` / `title` from the session's `*.session.json`.
  No PyYAML dependency added; the writer is small enough to maintain
  in-tree and the reader prefers PyYAML when installed but falls back
  to a tightly-scoped subset parser otherwise.
- `MeetingSummary.save(out_dir, basename, *, frontmatter_context=...)` —
  new keyword argument. When provided, the saved Markdown is prefixed
  with the YAML block and a `.frontmatter.json` sidecar is written.
  When omitted, behavior is unchanged from 0.6.x for backward
  compatibility.
- `_dispatch()` in `summarize.py` strips the trailing JSON block off
  every backend's output once, so PDF rendering keeps using a clean
  Markdown body and `MeetingSummary.data` exposes the parsed dict to
  callers.
- `meet label` (find-and-replace fallback) splits, replaces, and
  re-renders both the YAML frontmatter and the JSON sidecar in step
  with the body, so renames stay consistent across all four artifacts.
- The summary `.summary.meta.json` sidecar now records
  `data_extracted: true` on success or `data_error: "<reason>"` on
  failure to extract.

### Backwards compatibility

- All callers that don't pass `frontmatter_context=` to
  `MeetingSummary.save()` continue to produce the legacy artifacts.
- Sessions recorded before 0.7.0 work unchanged; run `meet ingest` to
  upgrade them to schema_version 1.

### Tests

- 33 new tests across `tests/test_frontmatter.py` (24) and
  `tests/test_ingest.py` (9), plus 3 new assertions in
  `tests/test_summarize.py` confirming both the on-disk and inline
  fallback prompts carry the JSON contract.

---

## v0.4.2 — 2026-04-24

### Improvements

- **Two-pass Ollama summarization (default for local LLMs)** — the local
  Ollama backend now runs a separate extraction pass (Pass 1: pull topics,
  actions, decisions, questions out of the transcript with a wide context
  window) followed by a formatting pass (Pass 2: organize the extracted
  data into the canonical Markdown structure with a small 8K context).
  This dramatically improves format compliance and reduces hallucinations
  on 20B-class local models like `gpt-oss:20b`. Cloud backends
  (claudemax, openrouter, openai) are unchanged — they remain single-pass.
- **Improved cloud-summary prompt** — the system prompt used by the
  cloud backends has been rewritten based on A/B-tested results: more
  topics extracted, ~20% faster on Sonnet, no regressions.
- **`--ollama-singlepass` opt-out flag** — added to `transcribe`, `run`,
  `gui`, and `label` commands for users who want the previous single-pass
  behavior. Also configurable via `MEETSCRIBE_OLLAMA_SINGLEPASS=1`.
- **Per-pass timing in summary sidecar** — the `.summary.meta.json`
  sidecar now records `mode: "two_pass"`, `pass1_seconds`,
  `pass2_seconds`, and `pass1_chars` when two-pass was used.

### Documentation

- Added `docs/local-model-evaluation.md` — full evaluation of local
  20B-class models on 4 reference transcripts, including known
  failure modes (gpt-oss:20b unreliability on low-information short
  transcripts, qwen3.6:27b reasoning-mode bottleneck, and the
  rationale for the two-pass design).

### Testing

- Added 27 new tests covering env-var resolution, two-pass system
  prompts (en + de), two-pass call flow, dispatcher routing, and
  sidecar serialization. All 127 tests pass.

### Known limitations

- `gpt-oss:20b` may hallucinate on transcripts dominated by very
  short low-information utterances ("yes", "okay"). For such meetings
  the cloud backends produce more reliable summaries — the fallback
  chain (claudemax → openrouter → ollama) handles this automatically
  if a cloud backend is configured.
- `gpt-oss:20b` may exceed the default 600s timeout on very large
  (>100 KB) non-English transcripts during Pass 1. The fallback chain
  catches this; alternatively pass `--summary-timeout 1200` or use a
  cloud backend.

---

## v0.4.1 — 2026-04-13

### Improvements

- **Speaker labeling and sync prompts no longer deferred during recording** —
  previously, the speaker labeling dialog and sync confirmation prompt would
  wait until the user stopped recording before appearing. They now appear
  immediately, allowing users to label speakers from a previous meeting while
  the next one records.

---

## v0.4.0 — 2026-04-13

### New features

- **Background post-processing for back-to-back meetings** — after stopping a
  recording, the GUI returns to idle within seconds (drain time) so you can
  immediately start recording the next meeting. Transcription, speaker labeling,
  summarization, PDF generation, and sync all run in a background job queue.
  A small status line at the bottom of the window shows background progress
  (e.g., "Transcribing: meeting-20260413-143453..."). Interactive dialogs
  (speaker labeling, alignment model prompts, sync confirmation) are deferred
  until the user is not actively recording.

### Improvements

- Simplified GUI state machine: removed 8 post-processing states that blocked
  the recording controls. Primary states are now: idle, recording, paused,
  draining, done, error.
- Background jobs process sequentially via a FIFO queue, ensuring GPU resources
  are not contended between concurrent transcriptions.
- Clean shutdown: closing the window unblocks any background threads waiting
  for user input.

### Testing

- All 100 tests pass (99 + 1 pre-existing environment-dependent skip).

---

## v0.3.3 — 2026-04-13

### New features

- **GUI Pause/Resume** — the recording widget now shows side-by-side Pause and
  Stop buttons while recording. Pressing Pause stops the current ffmpeg chunk
  and freezes the timer; pressing Resume starts a new chunk. Stopping from
  either recording or paused state works seamlessly — chunks are stitched
  together automatically. The idle/done/error states still show a single
  centered Record button as before.

### Improvements

- `RecordingSession` in `capture.py` gained `pause()` and `resume()` methods
  and a `paused` field on `RecordingStatus`, making pause/resume available to
  any future consumer (CLI, scripts, etc.) without GUI dependency.
- The watchdog thread now skips health checks while paused, preventing false
  stall-restart triggers.
- Stopping from the paused state skips the 10-second drain buffer since there
  is no active ffmpeg pipeline to flush.

### Bug fixes

- **CLI version string** — `meet --version` now reports the correct version
  (`0.3.3`) instead of the stale `0.1.0` it has shown since the initial release.

### Testing

- 13 new tests for pause/resume functionality (`tests/test_capture.py`):
  pause flag, ffmpeg stop, error cases, resume chunk creation, status reporting,
  elapsed-time freezing, stop-from-paused, and watchdog behaviour.
- All 100 tests pass.

---

## v0.3.2 — 2026-04-10

### New features

- **`--mixdown dual` mode for headphone users** — new CLI flag on `meet transcribe`
  and `meet run` that transcribes each stereo channel independently (mic → YOU,
  system → REMOTE) instead of mixing to mono. This fixes transcription for
  headphone setups where the ~20× energy difference between mic and system
  channels causes WhisperX to suppress the quieter voice. Diarization is skipped
  in dual mode since channel identity equals speaker identity. Default behavior
  (`--mixdown mono`) is unchanged.
  *(Contributed by [@Rolloniel](https://github.com/Rolloniel) in [#1](https://github.com/pretyflaco/meetscribe/pull/1))*

### Bug fixes

- **Speaker labeling threshold** — `_label_speakers_from_channels()` now requires
  `mic_ratio > 0.5` before labeling a speaker as YOU. Previously, the speaker
  with the highest mic ratio was always labeled YOU even when no speaker was
  actually mic-dominant (e.g. system-only audio capture). When no speaker exceeds
  the threshold, all speakers are labeled REMOTE.
  *(Contributed by [@Rolloniel](https://github.com/Rolloniel) in [#1](https://github.com/pretyflaco/meetscribe/pull/1))*

---

## v0.3.1 — 2026-04-10

### Bug fixes

- **CUDA NVRTC JIT fix** — replaced `_ensure_nvrtc_compat()` symlink approach
  with `_preload_nvrtc_builtins()` using `ctypes.CDLL`. The old method created
  a wrong-version symlink and set `LD_LIBRARY_PATH` too late (after
  `libnvrtc.so` was already loaded). The new approach preloads the correct
  `libnvrtc-builtins.so` into the process address space before NVRTC needs it,
  with automatic version detection across `nvidia-cuda-nvrtc` pip packages.

- **Channel-based diarization fallback** — added `_split_by_channel()` for
  stereo recordings where pyannote detects only 0–1 speakers. This can happen
  on short recordings or when GPU-dependent floating-point differences in
  WeSpeaker speaker embeddings cause VBx clustering to collapse multiple
  speakers into one. The fallback uses per-segment and per-word mic vs system
  channel RMS energy to assign YOU/REMOTE labels, which is hardware-independent
  and reliable when stereo channels are cleanly separated.

---

## v0.3.0 — 2026-04-01

### New features

- **Multi-backend summarization** — supports four backends with automatic
  fallback: `claudemax` (Claude Max API Proxy), `openrouter` (OpenRouter API),
  `openai` (any OpenAI-compatible endpoint), and `ollama` (local). If the
  configured backend is unavailable, meetscribe automatically tries the next
  one. Use `--summary-backend` and `--summary-model` flags, or set
  `MEETSCRIBE_SUMMARY_BACKEND` / `MEETSCRIBE_SUMMARY_MODEL` env vars.

- **Generic OpenAI-compatible backend** — use any OpenAI-compatible API for
  summarization (Lemonade, LiteLLM, vLLM, LocalAI, self-hosted endpoints).
  Set `MEETSCRIBE_OPENAI_BASE_URL` and optionally `MEETSCRIBE_OPENAI_API_KEY`.

- **Voiceprint speaker recognition** — automatically identifies speakers across
  meetings using voice embeddings. After labeling a meeting, speaker profiles
  are stored in `~/.config/meet/speaker_profiles.json`. Future meetings match
  voices against the database using cosine similarity. Use `meet enroll` to
  build profiles from past sessions, or let the GUI update profiles
  automatically after each labeling.

- **Meeting sync** — push meeting artifacts (transcript, summary, PDF, SRT) to
  any configured Git repository on a schedule. Configure your repo URL and
  meeting schedule in `~/.config/meet/sync_config.json`. Use `meet sync` to
  push manually or let the GUI auto-sync after recording. Run
  `meet sync --init-config` to generate an example config.

- **Improved summarization prompts** — prompt templates extracted to standalone
  markdown files (`meet/prompts/summarize_system.md`, etc.) for easy iteration
  without touching Python code. Prompt rewritten for better results with
  local/open-source models: more information-dense, preserves technical
  specificity, captures implied action items, provides format guidance.

### Improvements

- Dynamic context window sizing for ollama — automatically sizes `num_ctx` to
  fit long transcripts (up to 64K tokens) instead of truncating.
- Response validation catches upstream API errors (expired tokens, rate limits)
  that would otherwise be silently saved as the meeting summary.
- Thinking mode explicitly disabled for ollama models (`think: false`) to avoid
  wasting tokens on hidden reasoning with models like GLM-4.7-flash and Qwen 3.5.
- GUI auto-sync guarded by `is_sync_configured()` — silently skips if no repo
  is configured.

### Testing

- All 81 existing tests pass with the new prompt loading system.

---

## v0.2.0 — 2026-03-14

### New features

- **Multilingual support** — Whisper large-v3-turbo supports 99 languages.
  meetscribe now passes language hints through the full pipeline: transcription,
  wav2vec2 alignment, Ollama summary (prompted in the source language), and PDF.
  Use `--language auto` (default) or specify a code: `en`, `de`, `tr`, `fr`,
  `es`, `fa`.

- **Farsi / RTL support** — Farsi transcripts render correctly in PDF using
  Noto Naskh Arabic with arabic-reshaper + python-bidi for right-to-left layout.
  Install optional deps with `pip install "meetscribe-offline[rtl]"`.

- **`meet label` CLI command** — assign real names to speakers after the fact.
  For each speaker: shows a summary table, plays a short audio clip from the
  correct stereo channel (via ffplay), prompts for a name. Regenerates all
  outputs (txt, srt, json, summary.md, pdf) with the new names. Options:
  `--no-audio`, `--no-summary`.

- **GUI speaker labeling dialog** — when 2+ speakers are detected, a dialog
  appears before results are saved. Shows each speaker's channel and a sample
  line. Labels are applied before writing any output files.

### Improvements

- PDF now uses DejaVu Sans for full Unicode coverage (replaces previous
  Latin-only font). Handles Cyrillic, Greek, Turkish special characters, etc.
- Ollama summary prompts are now language-aware: when a non-English language is
  detected, the prompt instructs the LLM to write the summary in that language.
- `post_process()` function centralises all output generation (txt, srt, json,
  pdf) so that `meet label` and the GUI dialog share the same code path.
- Shared utilities extracted to `meet/audio.py`, `meet/languages.py`,
  `meet/utils.py`, `meet/label.py` for cleaner architecture.

### Testing

- 81-test suite added covering `label`, `pdf`, `summarize`, `transcribe`, and
  `utils` modules.

### Package

- PyPI package renamed to `meetscribe-offline` to distinguish from an unrelated
  squatted project. Install with `pip install meetscribe-offline`.

---

## v0.1.0 — 2026-03-01

Initial release.

- Dual-channel audio capture (mic left, system audio right) via
  PipeWire/PulseAudio + ffmpeg
- WhisperX transcription (faster-whisper + wav2vec2 alignment)
- pyannote-audio speaker diarization with YOU/REMOTE channel mapping
- Ollama AI meeting summaries (qwen3.5:9b default)
- PDF output (summary + full transcript)
- Output formats: `.txt`, `.srt`, `.json`, `.summary.md`, `.pdf`
- GTK3 GUI widget (always-on-top, record/stop, live timer, open results)
- CLI: `meet run`, `meet record`, `meet transcribe`, `meet gui`, `meet devices`,
  `meet check`
