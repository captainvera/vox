# Parakeet MLX Integration Plan

Add `parakeet-mlx` as an alternative transcription backend.

## Why

- Native MLX on Apple Silicon (same runtime as Voxtral, no ONNX/PyTorch)
- 600M params / 2.5 GB (vs Voxtral 4B) — faster, lighter
- 25 languages, built-in punctuation + capitalization
- Streaming via `transcribe_stream` with caching
- CC-BY-4.0 license

Model: `mlx-community/parakeet-tdt-0.6b-v3` via `parakeet-mlx` library.

## API Reference (verified from source)

### Batch

```python
from parakeet_mlx import from_pretrained

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
result = model.transcribe("audio_file.wav")  # AlignedResult
result.text   # full transcribed text
result.sentences  # list[AlignedSentence] with timestamps
```

- `from_pretrained(hf_id_or_path, dtype=mx.bfloat16, cache_dir=None)` — downloads from HF hub or loads local dir.
- `model.transcribe(path, dtype=mx.bfloat16, chunk_duration=None, overlap_duration=15.0)` — takes file path, returns `AlignedResult`.
- `model.preprocessor_config.sample_rate` — model's expected sample rate (typically 16000).

### Streaming

```python
with model.transcribe_stream(
    context_size=(256, 256),  # (left_context, right_context) frames
    depth=1,
) as ctx:
    ctx.add_audio(mx_array_chunk)  # 1D mx.array
    ctx.result          # AlignedResult (finalized + draft)
    ctx.finalized_tokens  # confirmed tokens
    ctx.draft_tokens      # tentative tokens (may change)
```

- `add_audio(audio: mx.array)` — **blocks significantly**: computes mel spectrogram, runs Conformer encoder forward pass, two-phase decode (finalized + draft). Cannot be called from sounddevice callback thread.
- No callback API — must diff `ctx.result.text` against previous text to detect new tokens.
- Context manager switches encoder to local attention on `__enter__`, restores on `__exit__`.

## Changes

### 1. `src/vox/config.py` — add backend selection

- Add `VALID_BACKENDS = ("voxtral", "parakeet")`.
- Add fields: `backend: str = "voxtral"`, `parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v3"`.
- Validate `backend` in `__post_init__` (same pattern as `mode`).
- Keep `model_path` for Voxtral backward compat.

### 2. `src/vox/parakeet.py` — new file

**`ParakeetTranscriber`** — implements `Transcriber` protocol:

- `load()`: calls `parakeet_mlx.from_pretrained(model_name)`. Stores model + sample rate.
- `transcribe(audio)`: write temp WAV via soundfile, call `model.transcribe(path)`, return `result.text`.
- `supports_streaming`: `True`.
- `create_stream(on_token)`: return `ParakeetStream`.

**`ParakeetStream`** — implements `TranscriptionStream` protocol:

- Constructor: enter `model.transcribe_stream(context_size=(256, 256))` context, spawn background thread.
- `feed(chunk: np.ndarray)`: append to thread-safe audio buffer (fast, no ML work). Called from sounddevice callback.
- Background thread (`_process_loop`):
  1. Drain audio buffer under lock.
  2. Convert to `mx.array`, call `ctx.add_audio(chunk)`.
  3. Diff `ctx.result.text` against `_prev_text`.
  4. If new text detected, call `on_token(delta)` and update `_prev_text`.
  5. Sleep briefly, repeat.
- `flush()`: signal thread to stop, wait for exit, return `ctx.result.text`.
- `close()`: exit streaming context manager (`__exit__`), null all state.

### 3. `src/vox/__main__.py` — factory dispatch

Replace hardcoded `VoxtralTranscriber` instantiation with:

```python
def _make_transcriber(config: Config):
    if config.backend == "parakeet":
        from .parakeet import ParakeetTranscriber
        return ParakeetTranscriber(model_name=config.parakeet_model)
    from .transcriber import VoxtralTranscriber
    return VoxtralTranscriber(model_path=config.model_path)
```

Lazy imports so only the selected backend's deps are loaded.

### 4. `pyproject.toml` — optional dependency

```toml
[project.optional-dependencies]
parakeet = ["parakeet-mlx>=0.3.0"]
```

Install with `uv tool install -e ".[parakeet]"`.

### 5. `src/vox/formatter.py` — no code change

Parakeet outputs punctuated/capitalized text. Existing transforms are idempotent on already-punctuated input (`fix_capitalization`, `ensure_trailing_punctuation`). `strip_fillers` is still useful since Parakeet may transcribe filler words faithfully.

### 6. Tests — TDD (red/green)

**`tests/test_config.py`** (additions):

- `backend` field exists, defaults to `"voxtral"`.
- `backend="parakeet"` accepted, `backend="invalid"` raises `ValueError`.
- `parakeet_model` field exists with correct default.
- Round-trip persistence for backend field.

**`tests/test_parakeet.py`** (new file):

- Protocol conformance: `ParakeetTranscriber` satisfies `Transcriber`, `ParakeetStream` satisfies `TranscriptionStream`.
- `load()` calls `from_pretrained` with correct model name, idempotent.
- `is_loaded` transitions false -> true after `load()`.
- `transcribe(audio)` returns string, writes temp WAV.
- `supports_streaming` is `True`.
- `create_stream(on_token)` returns a `ParakeetStream`.
- Stream `feed()` accepts numpy, is thread-safe (multi-thread stress test).
- Stream `flush()` returns string (accumulated text).
- Stream `close()` releases state.
- Stream text diffing: `on_token` fires with correct deltas.
- Factory dispatch: `_make_transcriber` returns `ParakeetTranscriber` for `backend="parakeet"`, `VoxtralTranscriber` for `backend="voxtral"`.

## Execution order

TDD ��� write failing tests first (RED), then implement until green (GREEN).

1. RED: config tests (`backend`, `parakeet_model` fields, validation)
2. GREEN: config changes
3. RED: `test_parakeet.py` — all `ParakeetTranscriber` + `ParakeetStream` tests
4. GREEN: `src/vox/parakeet.py` — batch + streaming in one pass
5. RED: factory dispatch tests
6. GREEN: `__main__.py` factory + `pyproject.toml` optional dep
7. Verify all tests pass (`pytest`)
