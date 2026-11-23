# Let's turn our voice into text, in a notebook

Record a short clip, stop, and watch the transcript appear as words are decoded.
You can upload audio or try the included sample too. PyTorch, audio processing,
compilation and model execution all run in your browser.

- [Open the WebGPU notebook](https://anutosh21.github.io/whisper/lab/index.html?path=whisper-speech-webgpu.ipynb)
- [Open the CPU notebook](https://anutosh21.github.io/whisper/lab/index.html?path=whisper-speech-cpu.ipynb)
- [Open the optional benchmark](https://anutosh21.github.io/whisper/lab/index.html?path=benchmark.ipynb)

Both demos include the sample audio and a completed transcript. To try your own
clip, select **Run → Run All Cells** in a fresh kernel. Click **Record microphone**,
allow access, and speak. **Stop & transcribe** releases the microphone and starts
recognition. Clips can be up to 15 seconds long.

The demo goes straight from loading and compiling the model to the transcript
and recording controls. Eager-reference checks and repeated timings are separate:
`content/benchmark.ipynb` contains completed CPU/WebGPU comparisons, and
`content/benchmark-results.json` keeps the recorded measurements. Each benchmark
run also saves its own JSON report. Run one notebook at a time when comparing.

Shoutout to [OpenAI's Whisper](https://github.com/openai/whisper) and the
[ExecuTorch speech-to-text playground](https://executorch.ai/playground/webgpu/#/demo/speech-to-text)
for the inspiration! Built from the
[Xeus-Lite template](https://github.com/jupyterlite/xeus-lite-demo).

We're using the original Whisper base checkpoint in FP32. Model configuration,
tokenizer files and the Python dependencies described in `vendor/README.md` are
preloaded. The model cell downloads the pinned checkpoint (about 277 MiB), checks
its SHA-256 hash and saves it in the browser cache. Later runs reuse that copy
while storage remains available. Clearing site data, private browsing or browser
eviction can require another download. Audio stays in the browser.

The sample is an excerpt from John F. Kennedy's inaugural address, a US federal
government work, distributed with Whisper's tests. Model, code and dependency
licensing details are in `content/sources.json` and the included license files.

To deploy, set **Settings → Pages → Source → GitHub Actions**, then push to `main`.

The browser bundle omits development headers, static libraries and package tests.
Both notebooks use Turbine/IREE for compilation and execution on their selected device.

To build and check the same site locally before deploying:

```bash
micromamba create -n whisper-site -f .github/build-environment.yml
micromamba run -n whisper-site python .github/prepare_whisper.py
micromamba run -n whisper-site jupyter lite build --contents content --output-dir dist
micromamba run -n whisper-site python -m http.server 8000 --directory dist
```

Open `http://localhost:8000/lab/index.html`. Saved transcripts and audio are visible
without loading the model; run all cells to recreate the recording controls.
