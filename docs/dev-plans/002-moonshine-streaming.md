# 002 — Fix Moonshine streaming transcription

Status: in progress
Backend: moonshine-voice (C++ core, ONNX Runtime, Python ctypes bindings)
Model: medium-streaming (245M params, 6.65% WER)
Files: `src/vox/moonshine.py`, `tests/test_moonshine.py`

## Problem

Batch mode works. Streaming is broken in two ways:
1. **Wrong text** — model hallucinates ("Thank you" for "This is a test of Moonshine real time")
2. **No incremental output** — full phrases appear at once instead of word-by-word

## Root cause (validated)

Ran the reference streaming pipeline (`transcriber.py __main__`) with `beckett.wav` (10s). Results:

```
LineTextChanged id=735 text='Ever heard?'     ← initial hypothesis
LineTextChanged id=735 text='Ever failed.'    ← REVISION (correct)
...
LineTextChanged id=737 text='Try it.'         ← initial hypothesis
LineTextChanged id=737 text='Try again.'      ← REVISION (correct)
...
LineTextChanged id=739 text='5.'              ← initial hypothesis
LineTextChanged id=739 text='Fail better.'    ← REVISION (correct)
```

**Moonshine's streaming model routinely revises earlier text** via speculative decoding. 3 of 6 phrases were revised in the test. Our `_compute_delta()` uses strict prefix matching and silently drops ALL revisions:

```python
# "Ever failed.".startswith("Ever heard?") → False → revision DROPPED
```

Our output for beckett.wav: `"Ever tried? Ever heard? No matter. Try it. Fail again. 5."`
Correct output:              `"Ever tried? Ever failed. No matter. Try again. Fail again. Fail better."`

**The prefix-match delta strategy is the primary bug.** 50% of phrases get wrong text because revisions are silently dropped.

### Debounce hypothesis — disproved

The reference impl also feeds audio faster than real-time (10s audio in 0.77s wall time) and works correctly. The C library's "200ms debounce" is either stream-time-based or doesn't prevent updates in practice. `MOONSHINE_FLAG_FORCE_UPDATE` may help incrementality but is NOT needed for correctness.

## Deficiencies (re-ranked after validation)

### 1. Delta strategy drops model corrections (CRITICAL — root cause)

`_compute_delta()` at `moonshine.py:284-294` uses strict prefix matching. Moonshine's streaming model revises text on ~50% of phrases. Every revision is silently dropped, freezing the (often wrong) initial hypothesis.

### 2. Moonshine streaming is phrase-level, not word-level (DESIGN — not a bug)

Reference impl produced ~1.6 `LineTextChanged` per phrase. Text arrives as complete phrases with occasional revisions — NOT word-by-word like Voxtral. This is inherent to the architecture (VAD segmentation + speculative decoding). Expecting Voxtral-style token-by-token output is unrealistic.

### 3. flush() feeds remaining audio as one giant array (MEDIUM)

`flush()` at `moonshine.py:209-211` concatenates all remaining buffered chunks into one `add_audio()` call. Single call = single update check.

### 4. FORCE_UPDATE and update_interval tuning (LOW — nice to have)

`MOONSHINE_FLAG_FORCE_UPDATE` and lower `update_interval` could improve incrementality within long phrases, but the reference impl works fine without them.

## Fix plan

### Phase 1: handle model revisions (the actual fix)

Replace prefix-match delta with a strategy that handles Moonshine's text revisions. Options:

**Option A — Emit-on-complete (simplest correct approach):**
Don't emit text during `LineTextChanged`. Only emit final text on `LineCompleted`. Accumulate completed lines with spaces. Loses real-time feel entirely but guarantees correct text.

**Option B — Track full line, emit latest (incremental with correct final):**
On each `LineTextChanged`, track the full `line.text`. For the on_token callback, emit the *latest* full line text (not a delta). The keystroke layer needs to handle replacement — either:
- Buffer pending text and only actually type on `LineCompleted`
- Or use delete keystrokes to backtrack on revision

**Option C — Prefix-match with fallback (minimal change):**
Keep prefix-match delta. When revision is detected (new text doesn't start with old), emit a separator + full new text. Produces some duplicate text but no data loss.

**Recommendation:** Start with Option A (emit-on-complete). It's the smallest, most correct change. Then iterate to Option B if the latency is unacceptable.

### Phase 2: flush cleanup

Feed remaining audio in chunks during `flush()` instead of one concatenated array.

### Phase 3: tune (optional)

Lower `update_interval`, consider `FORCE_UPDATE`, add diagnostic logging for `has_text_changed` frequency.

## Test plan

- Failing test: simulate Moonshine revision sequence (TextChanged "X" → TextChanged "Y"), verify output contains "Y" not "X"
- Failing test: simulate multiple revised lines, verify final accumulated text matches all completions
- Unit test: flush feeds remaining audio correctly
- Integration test: reference impl with beckett.wav (already passing — validated above)
- Manual test: compare Moonshine streaming output vs Voxtral on same speech

## Validated assumptions

- Reference streaming pipeline produces correct output with faster-than-real-time feeding (0.77s for 10s audio) ✓
- Text revisions are common (~50% of phrases in test) ✓
- ctypes conversion of numpy arrays produces identical values to Python lists ✓
- Model architecture (medium-streaming) loads and transcribes correctly ✓
