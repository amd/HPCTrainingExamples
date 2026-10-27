#!/usr/bin/env python3
"""Measure faster-whisper word-error-rate and decode latency on recorded audio.

Place one or more recordings in eval/audio/ as NAME.wav (16-bit PCM, any sample
rate) with a matching NAME.txt holding the exact words spoken. We report WER and
the realtime factor for each model so model and setting changes can be compared.

Usage: python3 eval/stt_eval.py [model ...]   (default: tiny.en base.en)
"""
import os, sys, time, glob, wave, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from live_captions import fix_en, TECH_PROMPT
from faster_whisper import WhisperModel

SR = 16000
AUDIO_DIR = os.path.join(HERE, "audio")

def load_wav_16k(path):
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        m = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, 1, m, endpoint=False),
                      np.linspace(0, 1, len(x), endpoint=False), x).astype(np.float32)
    return x

def norm(s):
    return " ".join("".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in s).split())

def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / max(1, len(r))

def transcribe(model, audio):
    peak = float(np.max(np.abs(audio)) + 1e-6)
    if peak < 0.3:
        audio = np.clip(audio * min(6.0, 0.3 / peak), -1.0, 1.0)
    segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                               condition_on_previous_text=False, no_speech_threshold=0.6,
                               initial_prompt=TECH_PROMPT)
    return " ".join(s.text.strip() for s in segs).strip()

def gather():
    pairs = []
    for wavp in sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav"))):
        txtp = os.path.splitext(wavp)[0] + ".txt"
        if os.path.exists(txtp):
            pairs.append((open(txtp, encoding="utf-8").read().strip(), load_wav_16k(wavp)))
    return pairs

def main():
    models = sys.argv[1:] or ["tiny.en", "base.en"]
    samples = gather()
    if not samples:
        sys.exit(f"No recordings found. Add NAME.wav + NAME.txt to {AUDIO_DIR}")
    total_s = sum(len(a) for _, a in samples) / SR
    rows, detail = [], []
    for mdl in models:
        print(f"\n[stt] loading {mdl}")
        m = WhisperModel(mdl, device="cpu", compute_type="int8",
                         cpu_threads=min(8, os.cpu_count() or 8), num_workers=1)
        wers, t_dec = [], 0.0
        detail.append(f"\n### {mdl}\n")
        for ref, audio in samples:
            t = time.time()
            hyp = fix_en(transcribe(m, audio))
            t_dec += time.time() - t
            e = wer(ref, hyp); wers.append(e)
            detail.append(f"- WER {e:.2f}\n  - ref: {ref}\n  - hyp: {hyp}")
        mean_wer = sum(wers) / len(wers)
        rtf = t_dec / max(1e-6, total_s)
        rows.append((mdl, mean_wer, t_dec / len(samples) * 1000, rtf))
        print(f"[stt] {mdl}: WER {mean_wer:.3f}  decode {t_dec/len(samples)*1000:.0f} ms/utt  RTF {rtf:.2f}x")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out = [f"# STT eval ({ts})",
           f"\n{len(samples)} utterances  |  {total_s:.1f}s audio\n",
           "| model | mean WER | ms/utt | realtime factor |", "|---|---|---|---|"]
    out += [f"| {mdl} | {w:.3f} | {ms:.0f} | {rtf:.2f}x |" for mdl, w, ms, rtf in rows]
    out.append("\n(A realtime factor below one means decoding is faster than real time.)\n")
    out += detail
    rep = os.path.join(HERE, f"report_stt_{ts}.md")
    open(rep, "w", encoding="utf-8").write("\n".join(out))
    print(f"\n[stt] wrote {rep}")

if __name__ == "__main__":
    main()
