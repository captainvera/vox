# 003 — Fix Voxtral streaming: freeze, fan noise, silence handling

Status: planned
Model: Voxtral-Mini-4B-Realtime-6bit (voxmlx, MLX)
Files: `src/vox/voxtral_stream.py`, `tests/test_voxtral_stream.py`

## Problem

Three compounding bugs in VoxtralStream cause freeze, sustained fan noise, and unresponsive stop:

1. **App freezes on stop after long session** — processing thread is stuck in decode loop, flush times out, then races with the still-running thread on shared model state
2. **Fans blow during silence** — model decodes at full speed (~12.5 steps/sec) during silence, producing empty tokens. Each step is a full 4B forward pass
3. **Quality degrades with length** — EOS never fires (0 EOS across entire log history), decoder context grows unbounded, model eventually enters degenerate state

## Evidence from logs

```
# 39% of all tokens are empty strings (silence decode waste)
Empty tokens: 1160 / 2962 total (39%)

# EOS never fires — across ALL sessions, ever
EOS count: 0 (every session shows eos=0)

# The freeze: processing thread unresponsive, flush never completes
03:06:47  last token (" Silence.")
          ← 3 min gap, no tokens, no EOS
03:09:51  "Stopping realtime stream"
          ← NO flush/exit log
03:10:03  PID written (app killed + restarted)

# Empty token spam visible in early session
01:34:12  stream starts
01:34:13  token: '' (×50 consecutive empty tokens before first real word)
01:34:18  token: ' This'  ← first real word, 6 seconds after start
```

## Root causes

### 1. CRITICAL — Race condition in flush() after timeout

`flush()` at `voxtral_stream.py:128-169`:

```python
self._running = False
if not self._done_event.wait(timeout=10):
    log.warning("...")

# BUG: proceeds to use model even though processing thread is still running
self._encode_chunk(flush_chunk)         # ← thread-unsafe
self._decode_available(...)             # ← thread-unsafe
```

After the 10s timeout, the processing thread is still alive inside `_decode_available()`. Both threads call `model.decode()` on the same KV cache simultaneously. Result: corrupted state, hang, or crash.

### 2. CRITICAL — Decode loop ignores _running flag

`_decode_available()` at `voxtral_stream.py:311`:

```python
for i in range(n_decodable):
    # ... model.decode() per step ...
    # _running is NEVER checked
```

If `n_decodable` is large (hundreds of accumulated embeds), this loop runs for seconds without checking if the user stopped. The processing thread cannot respond to stop requests until the entire batch is decoded.

### 3. HIGH — No silence detection, continuous GPU during silence

The model decodes silence at full speed. Each step produces an empty token but still costs a full forward pass. Over minutes of silence, this means thousands of wasted inferences, sustained GPU heat, and fan noise.

### 4. MEDIUM — Empty tokens emitted via callback

`_emit_token()` fires `_on_token(text)` for every decoded token including empty strings. The app's `_on_stream_token` filters them (`if not text: return`), but the callback overhead and accumulated list still grow. The filter should be in VoxtralStream, not the app.

### 5. LOW ��� EOS never fires, no natural segmentation

The model never produces EOS, so `_reset_state()` is dead code. The decoder accumulates the full session in KV cache. At 12.5 tokens/sec, a 90s session = ~1125 cache positions. Well within the 8192 limit, but attention cost grows linearly with context length, making each step progressively slower.

## Fix plan

### Phase 1: fix the freeze (safety)

1. **Check `_running` inside the decode loop.**
   Add `if not self._running: break` inside the `for i in range(n_decodable)` loop. This lets the processing thread respond to stop within one decode step (~50-100ms) instead of waiting for the entire batch.

2. **After flush timeout, do NOT touch the model.**
   If `_done_event.wait(timeout=10)` returns False, skip the final encode/decode entirely. Return whatever text has been accumulated so far. Losing the last few tokens is better than hanging.

   ```python
   if not self._done_event.wait(timeout=10):
       log.warning("Processing thread did not stop — returning partial text")
       return "".join(self._accumulated)
   ```

### Phase 2: silence handling (fan noise)

3. **Track consecutive empty tokens. Pause decode after threshold.**
   After N consecutive empty tokens (e.g., 25 = ~2 seconds of silence), stop calling `_decode_available()` and switch to a polling mode — only resume decoding when new non-trivial audio arrives or when audio_embeds accumulate past a threshold.

   This is the key change for fan noise. During silence:
   - Encoder still runs (cheap, must keep processing audio)
   - Decoder pauses (expensive, no useful output)
   - Resume when speech resumes

4. **Filter empty tokens in `_emit_token`, not the app.**
   Add `if not text: return` at the top of `_emit_token()`. Avoids callback overhead, keeps `_accumulated` clean.

### Phase 3: long-form robustness (quality)

5. **Force periodic state reset.**
   Since EOS never fires naturally, add a manual reset after N seconds of continuous decoding (e.g., 60s) or after a silence gap. Feed right padding before reset and left padding after, simulating a natural segment boundary. This keeps the KV cache small and the model in its training distribution.

6. **Log decode step latency.**
   Track wall-time per `model.decode()` call. Log a warning when it exceeds a threshold (e.g., 200ms). This gives early visibility into performance degradation before it causes a freeze.

### Phase 4: cleanup

7. **Remove the `_on_stream_token` empty-string filter from app.py** once VoxtralStream handles it internally.

8. **Add diagnostic stats to flush log**: consecutive empty token count, max decode latency, cache utilization.

## Test plan

### Phase 1 (freeze fix)
- Unit test: decode loop breaks when `_running` is False
- Unit test: flush returns partial text after timeout (no model access)
- Integration test: simulate long decode batch + stop request — verify clean exit

### Phase 2 (silence)
- Unit test: consecutive empty tokens trigger decode pause
- Unit test: decode resumes after speech-like audio arrives
- Unit test: `_emit_token` filters empty strings

### Phase 3 (long-form)
- Unit test: periodic reset fires after configured interval
- Unit test: reset produces right pad + left pad boundary
- Manual test: 2+ minute continuous speech — verify consistent quality

## Open questions

- What's the right silence threshold? 25 empty tokens (2s) is a guess. Need to test with real ambient noise vs intentional pauses.
- Should the periodic reset be time-based (every 60s) or silence-based (reset at first silence gap after 30s)?
- Can we detect "speech resumed" cheaply without running the decoder? Audio energy / RMS threshold on raw samples might work.
- The empty token issue �� is the tokenizer producing empty strings for some valid token IDs? Should we log the actual token IDs to understand what the model is generating during silence?
