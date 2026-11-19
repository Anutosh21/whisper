# Let's turn our voice into text, in a notebook

Record a short clip, stop, and watch the transcript appear as words are decoded.
You can upload audio or try the included sample too. PyTorch, audio processing,
compilation and model execution all run in your browser.

- [Open the WebGPU notebook](https://anutosh21.github.io/whisper/lab/index.html?path=whisper-speech-webgpu.ipynb)
- [Open the CPU notebook](https://anutosh21.github.io/whisper/lab/index.html?path=whisper-speech-cpu.ipynb)

Run the cells from top to bottom in a fresh kernel. Click **Record microphone**,
allow access, and speak. **Stop & transcribe** releases the microphone and starts
recognition. Clips can be up to 15 seconds long. Run one notebook at a time when
comparing CPU and WebGPU.

Shoutout to [OpenAI's Whisper](https://github.com/openai/whisper) and the
[ExecuTorch speech-to-text playground](https://executorch.ai/playground/webgpu/#/demo/speech-to-text)
for the inspiration! Built from the
[Xeus-Lite template](https://github.com/jupyterlite/xeus-lite-demo).

We're using the original Whisper base checkpoint in FP32. GitHub Actions downloads
its pinned revision and checks its SHA-256 hash when building the site. Xeus
preloads it at `/models/whisper-base`, alongside the Python dependencies described
in `vendor/README.md`. Both notebooks load these local files directly. The first
visit downloads the environment and model; audio stays in the browser.

The sample is an excerpt from John F. Kennedy's inaugural address, a US federal
government work, distributed with Whisper's tests. Model, code and dependency
licensing details are in `content/sources.json` and the included license files.

To deploy, set **Settings → Pages → Source → GitHub Actions**, then push to `main`.
