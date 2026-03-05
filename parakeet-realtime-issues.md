# Parakeet Real-Time Streaming Issues

## Status: OPEN — Periodic Batch Transcription Planned

**`supports_streaming = False`** as of 2026-03-02. Native
`transcribe_stream` produces garbage for cursor-typing use cases.
Batch mode works perfectly.

**Next step**: Implement periodic batch transcription �� re-run
`model.generate()` on the growing audio buffer every ~2s during
recording, diff against previous output, emit stable deltas at cursor.
This gives realtime feedback using the exact code path that produces
perfect text.

### Why not `transcribe_stream`?

parakeet-mlx's `transcribe_stream` is a post-hoc bolt-on to a batch-only
model (`parakeet-tdt-0.6b-v3`). It works for **display-repaint** use
cases (the library author's demo reprints the full `result.text` via
`Rich.Live` every call — users report it's "spot-on"). But for
**cursor-typing** (our use case), draft tokens change completely between
calls so you can't diff-emit stable deltas. Finalized tokens trickle in
slowly and are corrupted by four structural defects in the streaming
pipeline (see Bug 4 below).

NVIDIA ships two distinct model families:
- **parakeet-tdt-0.6b-v3** — standard Conformer, batch-only by design
- **parakeet-eou** — cache-aware FastConformer, designed for streaming

parakeet-mlx wraps the batch-only model. parakeet-rs uses the
streaming-native model. That's the fundamental difference.

### Community evidence (GitHub issues)

- **senstella/parakeet-mlx#22**: Library author's streaming demo uses
  `result.text` with `Rich.Live` repaint. User reports "accuracy is
  spot-on" with 1.5s chunks + default params. This works because the
  full text is repainted each call — drafts flicker but finalized text
  stabilizes. Not viable for cursor-typing.

- **senstella/parakeet-mlx#46**: User reports hallucinations even in
  batch mode with non-speech signals. Different class of problem.

- **senstella/parakeet-mlx#20**: User built push-to-talk that does
  batch transcription after recording stops (calls `get_logmel()` +
  `model.generate()` directly on numpy buffer). No realtime feedback.

- **senstella/parakeet-mlx#42**: User found mel computation
  discrepancies vs NVIDIA's NeMo reference. Later said it didn't matter.

Nobody in the community is doing periodic batch transcription during
recording. Existing approaches are either display-repaint (streaming)
or batch-after-stop (push-to-talk).

## Model Facts

- **Model**: `parakeet-tdt-0.6b-v3` (600M params, TDT decoder)
- **Encoder**: 24-layer Conformer, `d_model=1024`, `subsampling_factor=8`
- **Preprocessor**: `hop_length=160` (10ms stride), `win_length=400` (25ms), `n_fft=512`, 128 mel features
- **One encoded frame** = 8 mel frames = 1280 samples = 0.08s at 16kHz

## How `transcribe_stream` Works

```python
with model.transcribe_stream(context_size=(L, R), depth=D) as ctx:
    ctx.add_audio(mx_array)     # blocks: mel → encoder → decode
    ctx.finalized_tokens        # confirmed, never change
    ctx.draft_tokens            # tentative, revised each call
    ctx.result.text             # finalized + draft joined
```

Each `add_audio()`:
1. Appends to internal `audio_buffer`, computes mel spectrogram
2. Concatenates new mel frames to `mel_buffer`
3. Runs **full encoder** on `mel_buffer[:, :aligned]` with `RotatingConformerCache`
4. Produces `length` encoded features
5. Two-phase decode: finalized region (`max(0, length - drop_size)`) + draft region
6. **Trims mel_buffer** to `drop_size × subsampling_factor + leftover` frames

Key parameters:
- `keep_size = context_size[0]` — KV cache depth (left context)
- `drop_size = context_size[1] × depth` — encoded frames treated as draft (right context)
- `depth` — how many encoder layers carry accurate KV cache across chunks (1..24)

## Bug 1: Metal Crash on Small Audio Chunks (FIXED)

**Symptom**: `RuntimeError: [metal::malloc] Attempting to allocate 18446744073709547520 bytes`
(= 0xFFFFFFFFFFFFF800, integer underflow: -2048 as uint64)

**Root cause**: Processing thread drained ~50ms of audio and called `add_audio()`.
Too few samples → fewer than 8 mel frames → encoder got `mel_buffer[:, :0]`
(zero-length time dimension). The `DwStridingSubsampling` Conv2d computed output
size `(0 + 2×pad − kernel) / stride + 1` with unsigned arithmetic → underflow.

**Fix**: Minimum buffer gate — peek at `_audio_chunks_len`, don't drain until
`_MIN_FIRST_SAMPLES` (4800 = 0.3s) is available. After first successful call,
lower to `_MIN_SAMPLES` (1600 = 0.1s) since mel_buffer retains context.

Also replaced `np.append` (O(n) per feed) with `list[np.ndarray]` accumulation
+ `np.concatenate` on drain (O(1) per feed, same as VoxtralStream).

## Bug 2: Full-Text Duplication in Token Emission (FIXED)

**Symptom**: Each "token" was a complete sentence like `"It's real-time audio."`,
typed in full at cursor ~25 times. App froze for ~8s on stop (keystroke queue
backlog).

**Root cause**: `_diff_and_emit(result.text)` compared full result (finalized +
draft) against `_prev_text`. Draft tokens change completely on every `add_audio()`
call, so `new_text` never starts with `_prev_text` → `elif` branch emitted the
FULL text every time.

**Fix**: Replaced with `_emit_finalized()` — reads only `ctx.finalized_tokens`
(monotonically growing list), joins `.text` attributes, emits delta against
`_emitted_text`. Draft tokens are never typed mid-stream. On `flush()`, full
`result.text` is emitted (drafts are final since recording ended).

## Bug 3: Zero Output with context_size=(256, 256) (FIXED)

**Symptom**: `flush complete: 0 chars` — nothing emitted during streaming or
at flush.

**Root cause**: `drop_size = 256 × 1 = 256` encoded frames. Tokens only finalize
after `256 × 0.08s = 20.5 seconds` of audio. For any typical recording (<20s),
`finalized_tokens` was always empty. Even `result.text` at flush returned very
little because the model's draft decode with 256 pending frames accumulated
poorly.

**Fix**: Reduced to `context_size=(256, 16)`.

## Bug 4: Poor Quality — Native Streaming Not Viable for Cursor-Typing

All parameter combinations tried produce garbage **finalized** output.
Draft text (via `result.text`) is more coherent but changes completely
between calls — unusable for incremental cursor-typing. Batch mode on
the same audio gives perfect text every time.

**Resolution**: `supports_streaming = False` for native streaming.
Periodic batch transcription planned as replacement (see status above).

### What Was Tried

| Config | drop_size | Chunk size | Finalization delay | Result |
|--------|-----------|-----------|-------------------|--------|
| `(256,16), depth=1` | 16 | 0.1s | 1.3s | 7 tokens, 17 chars from 12.4s: `"This is just. Uh."` |
| `(256,2), depth=24` | 48 | 0.17s | 3.84s | 15 tokens from 12.3s: `"It's the Figh okay,....e in buttons"` |
| `(256,2), depth=4` + silence flush | 8 | 0.5s | 0.64s | 13 tokens from 14.7s: `"Recording time per. car playingos."` |

All produce hallucinated text unrelated to speech. Batch mode on the
same recordings gives perfect transcription every time.

### Root Cause: The Starvation Loop

Confirmed via per-call logging (see `_process_loop` instrumentation).

The mel_buffer is trimmed to `drop_size × 8 + leftover` after every
`add_audio()` call. With small audio chunks (~0.1–0.17s drained per call),
the encoder output length barely exceeds `drop_size`:

```
mel_buffer retained = drop_size × 8         ≈ 384 frames (depth=24, right=2)
new mel from 0.17s audio                    ≈  17 frames
total mel                                   ≈ 401 frames
encoder output = 401 / 8                    ≈  50 encoded frames
finalized = max(0, 50 − drop_size=48)      =   2 frames
to_cache per layer = max(0, 50 − 48)       =   2 frames   ← STARVATION
conv cache tokens = min(padding=15, 50−48)  =   2 frames   ← STARVATION
```

98% of encoder compute re-encodes the retained draft region.  Only 2%
produces new finalized frames.  The KV cache (capacity 256) fills at
~2 frames per call.  The conv cache (needs 15 frames for proper
depthwise-conv padding) gets 2 frames per call — so for the first ~8
calls the convolution runs with mostly-zero context across all 24 layers.

**This is inherent to parakeet-mlx's sliding-window design**: the retained
mel equals `drop_size × 8`, so the encoder always produces ≈ `drop_size`
frames from the retained portion plus a tiny sliver from new audio.  With
frequent small calls, the sliver is negligible.

### Observed Logs (depth=24, right=2)

```
add_audio #1–#23:  fin=0  draft=0  mel≈(1, 200→389, 128)   ← building up
add_audio #24:     fin=2  draft=8  mel=(1, 389, 128)        ← first finalization at ~4s
add_audio #25–#29: fin grows 2→9                            ← trickle
add_audio #30–#65: fin stalls at 13  mel≈(1, 384-391, 128)  ← stalled for 6+ seconds
flush:             fin=15  draft=12  result=70 chars         ← gibberish
```

### Comparison: parakeet-rs and parakeet-eou

Studied via https://deepwiki.com/altunenes/parakeet-rs — uses a **completely
different model** (`parakeet-eou`, cache-aware FastConformer with ONNX
encoder that has explicit cache tensor I/O).  This is NOT the same model
as `parakeet-tdt-0.6b-v3` — NVIDIA designed it specifically for streaming.

| Aspect | parakeet-mlx (tdt-0.6b-v3) | parakeet-rs (parakeet-eou) |
|--------|---------------------------|---------------------------|
| **Model architecture** | Standard Conformer (batch-only) | Cache-aware FastConformer (streaming-native) |
| **ONNX encoder** | Single model, no cache I/O | `encoder.onnx` with `cache_last_channel`, `cache_last_time` inputs/outputs |
| **Re-encoding** | Every call re-encodes `drop_size×8` mel frames | **Never** — encoder processes each chunk once |
| **Chunk size** | ~0.1–0.5s | **0.16s** (2560 samples) |
| **Min buffer** | 0.3–1.0s | **1.0s** (16000 samples) |
| **Encoder input/chunk** | ~400 mel frames (mostly re-encoded) | **25 frames** (9 cache + 16 new) |
| **Cache** | Bolted-on `RotatingConformerCache` | Native encoder cache (trained with it) |
| **Mel normalization** | Per-chunk independent normalization | Features from 4s ring buffer |
| **End-of-stream** | Emit result.text | Feed 3 silence chunks to flush decoder |

parakeet-rs's encoder processes each chunk **once** because the conformer
layers have built-in cache inputs/outputs.  parakeet-mlx's conformer was
not designed for streaming — the `RotatingConformerCache` and attention
mode switch are bolted on, causing the defects documented above.

Even parakeet-rs's author reports mixed real-world quality: "I must admit
that this is not work very well on my real world tests."

### Fix Plan

Three changes, all in `src/vox/parakeet.py`:

#### 1. Reduce `drop_size` from 48 to 8

```python
# In create_stream():
ctx_mgr = self._model.transcribe_stream(
    context_size=(256, 2),
    depth=4,          # was 24 — 4 layers exact cache, drop_size = 2×4 = 8
)
```

**Why depth=4 not 24**: `drop_size = right × depth`.  With depth=24,
drop_size=48 and starvation is inevitable.  With depth=4, drop_size=8:

```
mel_buffer retained = 8 × 8                =  64 frames (0.64s, not 3.84s)
new mel from 0.5s audio                    =  50 frames
total mel                                  = 114 frames
encoder output = 114 / 8                   =  14 encoded frames
finalized = max(0, 14 − 8)                =   6 frames   ← 6× more than before
to_cache = max(0, 14 − 8)                 =   6 frames   ← cache fills 6×faster
conv cache tokens = min(15, 14−8)          =   6 frames   ← conv gets real context
```

6 finalized frames per call instead of 1–2.  Cache fills to 256 in ~43
calls (~21s).  Conv cache full in 3 calls.  Finalization delay = 0.64s
instead of 3.84s.

Only 4 of 24 layers have exact cache, but this is a pragmatic tradeoff:
the first 4 layers do the heavy feature extraction in conformers, and
the mel_buffer retention is tiny (64 frames = 0.64s) so the draft region
being re-encoded is small.

#### 2. Feed larger audio chunks

```python
_MIN_FIRST_SAMPLES = 16000   # 1.0s (was 4800 = 0.3s)
_MIN_SAMPLES        = 8000   # 0.5s (was 1600 = 0.1s)
```

**Why**: Matches parakeet-rs's 1s minimum buffer.  With 0.5s chunks the
encoder gets ~50 new mel frames per call → 14 total encoded → 6 finalized.
With 0.1s chunks it was ~10 new mel → ~9 total → 1 finalized.

Text appears in ~0.5s bursts (6 frames × N tokens/frame) instead of
per-frame trickle.  Since finalization delay is 0.64s, the added 0.4s
input latency is negligible.

#### 3. Silence flushing at end of stream

```python
# In flush(), after draining remaining audio:
silence = np.zeros(drop_size * 1280, dtype=np.float32)  # 10240 samples = 0.64s
ctx.add_audio(mx.array(silence))
```

**Why**: Learned from parakeet-rs (feeds 3 silence chunks at end).
Pushes the last real audio frames through the draft zone into
finalization.  Without this, the last `drop_size` frames' worth of
speech (~0.64s) only appears as draft tokens in `result.text`.  With
silence flushing, those frames get properly finalized first.

`drop_size × 1280` samples = `8 × 1280 = 10240` = 0.64s of silence.
This matches exactly the finalization delay, ensuring all real speech
is pushed through.

### Expected Outcome

| Metric | Before (depth=24) | After (depth=4) |
|--------|-------------------|-----------------|
| drop_size | 48 | 8 |
| Finalization delay | 3.84s | 0.64s |
| Finalized frames per call | 1–2 | 6 |
| Cache fill rate | 2 frames/call | 6 frames/call |
| Conv cache health | Starved (2/15) | Healthy (6/15 → full in 3 calls) |
| Encoder waste (re-encoding) | 98% | 57% |
| _MIN_SAMPLES | 1600 (0.1s) | 8000 (0.5s) |
| _MIN_FIRST_SAMPLES | 4800 (0.3s) | 16000 (1.0s) |
| Silence flush | No | Yes (0.64s) |

### Why It's Still Broken After Fixing Starvation

The depth=4 config fixed the mechanical starvation problem: 6 finalized
frames per call, cache filling properly, conv cache healthy.  But the
output was still hallucinated garbage (`"Recording time per. car
playingos."` instead of a coherent sentence).

This means the problem goes **deeper than cache fill rates**.  The
`transcribe_stream` pipeline corrupts encoder representations through
multiple compounding factors:

1. **Attention mode switch**: `__enter__` switches the encoder from
   `rel_pos` (trained mode, full bidirectional) to `rel_pos_local_attn`
   (windowed).  The model was never trained with local attention.  Even
   though the window (258 frames) covers the full input for early calls,
   the `LocalRelPositionalEncoding` produces different position embeddings
   than `RelPositionalEncoding` — the encoding buffer is `[1, L+R+1, d]`
   vs `[1, 2*max_len-1, d]`.  This changes every attention score.

2. **Layer-wise cache degradation**: With depth=D, only layers 0..D-1
   have "exact" cache (matching a full forward pass).  Layers D..23
   have cache entries computed from PREVIOUS calls' outputs — which were
   themselves computed with approximate cache at those layers.  Error
   compounds through the layer stack.  With depth=4: 20 layers have
   degrading cache.

 3. **Mel normalization discontinuity** (most damaging defect):
    `get_logmel()` applies `per_feature` normalization (zero-mean,
    unit-variance per mel bin) over just the current chunk's time
    dimension.  Each `add_audio()` computes mel from `audio_buffer`
    (leftover <160 samples + new chunk), normalizes independently,
    then concatenates to `mel_buffer`.  Adjacent chunks have different
    normalization statistics → artificial scale/offset discontinuities
    at every chunk boundary.  In batch mode, `get_logmel()` is called
    once on the full audio — normalization is global and correct.
    Additionally, the STFT reflect-padding at chunk edges creates
    spectral leakage (each chunk's first frame has <400 samples of
    context where `win_length=400`).

    Source: `parakeet_mlx/audio.py:169-176` — `per_feature` normalize
    computes `mean = mx.mean(x, axis=1)`, `std = mx.std(x, axis=1)`
    over just the chunk.  `parakeet_mlx/parakeet.py:1011-1029` —
    `add_audio()` computes mel per-call, concatenates to mel_buffer.

4. **TDT decoder state discontinuity**: The decoder carries
   `decoder_hidden` and `last_token` across calls.  These are set from
   the FINALIZED decode phase, which sees encoder features that were
   computed differently from batch mode (due to factors 1-3 above).
   The decoder's internal language model drifts into incoherent states.

None of these can be fixed by tuning `context_size`, `depth`, or chunk
size.  They are structural to `transcribe_stream`'s design.

**Untried parameter**: `keep_original_attention=True` exists in the
`transcribe_stream` signature.  It skips the attention mode switch
(fixes defect 1) but defects 2, 3, and 4 remain.  The mel
normalization artifact alone would still corrupt encoder input.

### The Fundamental Tradeoff

| Parameter | Effect on quality | Effect on latency |
|-----------|-------------------|-------------------|
| `context_size[1]` (right) ↑ | More mel frames retained → better context | Longer finalization delay |
| `depth` ↑ | More layers with accurate cache → better quality | `drop_size = R × depth` grows → longer delay |

The constraint is `drop_size = right × depth`.  To avoid starvation,
`drop_size` must be small relative to the per-call encoded output.  This
means you **cannot** have both high depth (all 24 layers) and reasonable
latency — the library's design makes it impossible.  depth=4 is a
pragmatic compromise.

### Why Voxtral Streaming Works Better

Voxtral (4B, autoregressive encoder-decoder) emits tokens from the **decoder**
directly — each token is final, no finalized/draft split. It manages its own
incremental encoder with left/right padding and a rotating KV cache. Tokens
appear word-by-word as audio arrives with ~0.5s latency.

Parakeet (600M, CTC/TDT Conformer) chunks audio and re-runs the encoder on
each chunk with a rotating cache. The decoder runs two passes (finalized +
draft) per chunk. Quality depends heavily on cache accuracy (depth) and context
window (right context).

### Future Possibilities

1. **Library improvements**: parakeet-mlx is young. A future version might
   support cache-aware encoding (like parakeet-eou's FastConformer) that
   avoids re-encoding entirely.  Monitor the repo for architecture changes.

2. **`keep_original_attention=True`**: Untested. Fixes defect 1 but not
   2-4.  Low probability of fixing cursor-typing quality, but cheap to
   experiment with if periodic batch latency proves unacceptable.

3. **parakeet-eou via ONNX**: If an MLX port of parakeet-eou appears, or
   if we can run the ONNX encoder via coremltools, native streaming
   becomes viable.

## Periodic Batch Transcription Plan

Bypass `transcribe_stream` entirely.  Use `model.generate()` (batch mode)
periodically on the growing audio buffer during recording.  This uses the
exact code path that produces perfect text.

### Architecture

```
recorder.on_chunk → ParakeetStream.feed()   [O(1) append to buffer]
                          │
                    background thread (every ~2s):
                          │
                          ├─ concatenate ALL accumulated audio
                          ├─ get_logmel(mx.array(audio), preprocessor_config)
                          ├─ model.generate(mel)[0].text
                          ├─ diff against previously emitted text
                          └─ emit delta via on_token callback
                          │
               flush() → one final batch transcribe, emit remaining
```

### Key design decisions

1. **Never trim the audio buffer.**  Always re-transcribe the full
   recording.  This guarantees the text prefix is monotonically stable
   (batch mode is deterministic for the same audio input).

2. **No temp files.**  Compute mel directly from numpy array:
   `get_logmel(mx.array(audio), model.preprocessor_config)` →
   `model.generate(mel)`.  Avoids the ffmpeg round-trip in
   `model.transcribe(path)`.

3. **Simple prefix diff.**  Find longest common prefix between old and
   new result text, emit the suffix.  Works because batch mode produces
   stable text — extending the audio only appends to the transcription.

4. **~2s transcription interval.**  Parakeet-0.6B does 10s of audio in
   ~0.5s on Apple Silicon.  Re-transcribing every 2s means:
   - At 10s of recording: 0.5s compute, 1.5s idle → ~25% GPU
   - At 20s of recording: 1.0s compute, 1.0s idle → ~50% GPU
   - At 30s of recording: 1.5s compute, 0.5s idle → ~75% GPU
   Practical ceiling ~40s before compute exceeds interval.

5. **ParakeetStream gets the model directly.**  `create_stream()` passes
   `self._model` (not a `transcribe_stream` context).  No `__enter__` /
   `__exit__`, no attention mode switch, no `RotatingConformerCache`.

6. **`supports_streaming = True`.**  This approach provides genuine
   realtime feedback via `on_token` during recording.

### Latency profile

| Recording length | Batch time (est.) | Update interval | Perceived latency |
|-----------------|-------------------|-----------------|-------------------|
| 5s | ~0.25s | 2s | ~2.25s |
| 10s | ~0.5s | 2s | ~2.5s |
| 20s | ~1.0s | 2s | ~3.0s |
| 30s | ~1.5s | 2s | ~3.5s |

Latency = interval + batch_time.  For dictation (typically <30s), this
is acceptable.  Text appears in bursts every ~2-3s rather than
word-by-word, but it's stable and accurate.

### Changes required

All in `src/vox/parakeet.py`:

1. **`ParakeetTranscriber`**:
   - `supports_streaming` → `True`
   - `create_stream()` → pass `self._model` directly, no context manager

2. **`ParakeetStream`**:
   - `__init__` takes `model` instead of `ctx`/`_ctx_mgr`
   - `_process_loop` → periodic batch transcribe + prefix diff
   - `flush()` → final batch transcribe, emit remaining delta
   - `close()` → just set flags (no context manager to exit)
   - Remove all `transcribe_stream`-specific code (finalized_tokens,
     draft_tokens, silence flushing, mel_buffer inspection)

3. **Tests** (`tests/test_parakeet.py`):
   - Update `_make_stream` to pass mock model instead of mock ctx
   - Update `test_parakeet_supports_streaming_is_false` → `is_true`
   - Remove `test_stream_emit_finalized_*` tests (no finalized tokens)
   - Add tests for prefix-diff logic
   - Add test for periodic batch transcription loop
