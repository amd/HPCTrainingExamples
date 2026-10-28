# KAL live captions

When we give a technical talk in English to an audience that reads Spanish, the
language gets in the way even when the slides are shared. Meeting platforms can
show live captions, but they transcribe whatever language is being spoken, so an
English talk produces English captions. What the audience needs is the reverse:
our English speech shown as Spanish subtitles, produced quickly enough to read
along as we speak.

This example is a small pipeline that does that. It listens to the microphone,
recognizes the English speech, translates each sentence into Mexican Spanish, and
writes the result to a file that a caption display reads. Nothing is spoken back,
so the audience hears our own voice while they read the translation.

## How it works

We read the microphone in short frames and track the energy of each frame to tell
speech from silence. Frames are grouped into a phrase, and the phrase is released
once a brief pause marks its end. Each finished phrase is transcribed once, on the
CPU, with faster-whisper, and the text is sent to the translator. The translation
streams back token by token, so the Spanish line grows on screen as it is produced
and a new line begins with the next phrase. Because we translate whole phrases
rather than a buffer we re-read as it fills, the translator always sees complete
sentences and no audio is lost while it is busy. The delay the audience notices is
the time to the first translated token, not the time to finish a sentence.

Silence and background noise can make the recognizer emit stray words. We guard
against this in two ways. Every recognized segment carries a no-speech probability
and an average token probability, and we drop the segments that look more like
noise than speech. We also keep a short substitution table for the coined tool
names the recognizer tends to mishear, and apply it before translation.

## Requirements

The client runs on the machine with the microphone, on Windows, macOS, or Linux.
We create a virtual environment and install the dependencies:

```
python -m venv venv
venv/bin/pip install -r requirements.txt   # venv\Scripts\pip on Windows
```

The translator runs wherever a GPU is available. Serving it is described next.

## Serving the translator

We serve the translator with vLLM, which exposes an OpenAI-compatible API on port
8000. The model used here is
[SaptivaAI/KAL-24B-mx-v1](https://huggingface.co/SaptivaAI/KAL-24B-mx-v1), trained
for Mexican Spanish. Any OpenAI-compatible endpoint can take its place by pointing
the client at a different URL and model name. The first run downloads the weights.

```
bash serve_vllm.sh
```

On a batch system we wrap that command in a scheduler job script and request one
GPU. When the GPU is on a different machine from the client, we forward the port
over SSH so the client reaches the API at `localhost:8000`:

```
ssh -N -L 8000:SERVING_NODE:8000 USER@LOGIN_HOST
```

## Running the captions

With the API reachable, we start the client:

```
python live_captions.py
```

We speak in English. The terminal prints each recognized phrase, its translation,
and the per-phrase timings, and the current caption is written to `caption.txt`
next to the script.

## Showing the captions

The caption file is meant to be read by whatever the audience already sees, and
this example comes with two displays for it.

When we share the whole screen, `caption_overlay.py` floats a translucent bar
across it that mirrors `caption.txt`. On Windows the bar lets clicks pass through
to the windows beneath it and closes with Ctrl+Alt+Q; on other systems it closes
with Escape. It clears itself after a short silence so a finished sentence does
not linger.

When we share a camera instead, OBS can read the same file: add a Text (GDI+)
source, enable read-from-file, point it at `caption.txt`, and select the OBS
virtual camera in the meeting application. A video background effect in the meeting
application hides captions drawn on the camera, so we leave it off.

## Tuning and evaluation

The pipeline has two quality stages, recognition and translation, and the `eval`
folder measures each offline so a change can be judged without a live meeting.

`eval/stt_eval.py` reports the recognizer word-error-rate and decode speed for one
or more models. We place a recording at `eval/audio/NAME.wav` with its transcript
at `eval/audio/NAME.txt`, then:

```
python eval/stt_eval.py tiny.en base.en
```

`eval/mt_eval.py` sends the lines in `eval/eval_set_en.txt` through the running
translator and checks each result: that terms meant to stay in English survive,
that the model translated rather than answered, that no formatting leaked through,
and that the latency stays low.

```
python eval/mt_eval.py
```

Each tool writes a timestamped report, so we judge a prompt or model change by
running it before and after and comparing the two reports.

Day-to-day behaviour is set through environment variables read by the client:

| variable | meaning |
|---|---|
| `KAL_API_URL` | translator endpoint (default `http://localhost:8000/v1/chat/completions`) |
| `KAL_MODEL` | served model name (default `kal`) |
| `KAL_MIC` | input device name substring (default: system default input) |
| `KAL_WHISPER` | whisper model, for example `tiny.en` or `base.en` |
| `KAL_START_RMS`, `KAL_END_RMS` | speech and silence energy gates; raise them for a noisy or far-field microphone |
| `KAL_END_SILENCE_MS` | trailing silence that ends a phrase; lower it for snappier captions |
| `KAL_NSP_MAX`, `KAL_LOGPROB_MIN` | confidence thresholds for rejecting hallucinated segments |
| `KAL_KEEP_LINES` | context lines kept on screen (default 2) |

The overlay reads `KAL_OVL_POS` (top or bottom), `KAL_OVL_FONT`, `KAL_OVL_ALPHA`,
`KAL_OVL_IDLE` (idle seconds before clearing), and `KAL_OVL_CLICKTHROUGH`.

One list is worth calling out. The KEEP list in `live_captions.py` holds the
product names and acronyms that must stay in English. Generic vocabulary such as
throughput, latency, and bandwidth is deliberately left off it, so the model
renders those words in Spanish, which reads more naturally for the audience.
`eval/mt_eval.py` keeps its own copy of the list for the preservation check, and
the two should agree.
