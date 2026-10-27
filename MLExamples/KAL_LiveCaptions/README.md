# KAL live captions

We present a talk in English and the audience reads live Mexican-Spanish
subtitles. A microphone is transcribed on the CPU with faster-whisper, each
complete phrase is translated by the KAL model served through vLLM, and the
Spanish text is streamed into a file that a display source reads. There is no
speech synthesis, so the only perceived delay is the time until the first
translated token appears.

The translator is [SaptivaAI/KAL-24B-mx-v1](https://huggingface.co/SaptivaAI/KAL-24B-mx-v1),
a Mexican-Spanish model, served behind an OpenAI-compatible API. Any
OpenAI-compatible endpoint and model can be substituted through the environment
variables below, though the prompt in `live_captions.py` is written for
English-to-Mexican-Spanish technical speech.

## How it works

The audio callback tracks short-term energy to find phrase boundaries. It
accumulates a phrase while we speak and, on a brief trailing silence, hands the
whole utterance to a worker thread. The worker transcribes that utterance once
and then streams its translation, so the caption fills in token by token and a
new line begins on the next phrase. Capturing whole phrases, rather than
repeatedly re-decoding a growing buffer, keeps the model fed with complete
sentences and avoids both dropped audio and sentences cut in the middle.

Two filters keep phantom captions off the screen. faster-whisper reports a
no-speech probability and an average log-probability per segment, and we discard
segments that score poorly on either; a short list of the words whisper tends to
emit from silence is dropped as well. A small substitution table repairs coined
tool names that whisper reliably mis-hears before they reach the translator.

## Requirements

The client runs on the machine with the microphone (Windows, macOS, or Linux):

```
python -m venv venv
venv/bin/pip install -r requirements.txt   # venv\Scripts\pip on Windows
```

The translator runs wherever a GPU is available. The first run downloads the
model weights.

## Serving the translator

`serve_vllm.sh` starts vLLM and exposes the API on port 8000:

```
bash serve_vllm.sh
```

On a batch system we wrap this invocation in a scheduler job script and request
one GPU. When the serving node is not the client machine, we forward the port
over SSH so the client reaches the API at `localhost:8000`:

```
ssh -N -L 8000:SERVING_NODE:8000 USER@LOGIN_HOST
```

## Running the captions

With the API reachable, start the client:

```
python live_captions.py
```

We speak in English and watch the terminal, which prints each English phrase, its
Spanish translation, and per-utterance timings. The current caption is written to
`caption.txt` next to the script.

## Showing the captions

For a shared screen, `caption_overlay.py` draws a semi-transparent, always-on-top
bar that reads `caption.txt`. We share the entire screen so the bar is captured
with the slides. On Windows the bar is click-through and quits with Ctrl+Alt+Q;
elsewhere it quits with Escape. The bar clears itself after a few idle seconds so
a finished line does not linger.

For a talking-head feed, OBS reads the same file: add a Text (GDI+) source, enable
read-from-file, point it at `caption.txt`, start the virtual camera, and select
that camera in the meeting application. Any video background effect in the meeting
application will cover captions burned onto the camera, so we leave it off.

## Evaluating and tuning

The pipeline has two quality stages, and `eval/` measures each offline so changes
can be compared without a live meeting.

`eval/stt_eval.py` reports transcription word-error-rate and decode speed per
model. We record a representative sample as `eval/audio/NAME.wav` with the spoken
words in `eval/audio/NAME.txt`, then:

```
python eval/stt_eval.py tiny.en base.en
```

`eval/mt_eval.py` sends the lines in `eval/eval_set_en.txt` through the running
translator and checks that terms meant to stay English survive, that the reply is
a translation rather than a chatbot answer, that no formatting leaks through, and
that latency stays low:

```
python eval/mt_eval.py
```

Each tool writes a timestamped report, so a prompt or model change is evaluated by
running it before and after and diffing the two reports.

The behaviour is adjusted through environment variables read by the client:

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

The KEEP list in `live_captions.py` names the product names and acronyms that must
stay in English. Generic vocabulary such as throughput, latency, and bandwidth is
deliberately left off that list so the model renders it in Spanish, which reads
more naturally for the audience. `eval/mt_eval.py` carries a copy of the list, and
the two must agree for its check to reflect the client.
