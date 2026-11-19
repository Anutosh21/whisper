# Whisper's Python dependencies

This deployment includes the same dependency bundle used by the validated
Wasm64 speech notebooks. It contains Transformers 4.57.6, Hugging Face Hub
0.36.2 and their Python dependencies, including matching Emscripten 6.0.8 /
Python 3.14 Wasm64 extensions for regex, Brotli and safetensors.

The bundle has two retained Transformers portability fixes, recorded in
`transformers-wasm.patch`: load optional GGUF support lazily and use string
keys with the Python AddedToken fallback. It uses WhisperTokenizer's Python
implementation, without requiring the optional Rust tokenizers package.

Bundle SHA-256:
`89fbcc3a52bd78503544845d8d35a478d41a99bce80bab7e413ba46880e785e3`.

The build checks this hash and uses Xeus's standard filesystem mounts to
preload the dependencies into `/lib/python3.14/site-packages`. The notebook
imports Transformers normally. This bundle supplements `environment.yml`;
it is not a separately published emscripten-forge package. Its package metadata
and bundled notices remain in the archive. Transformers is Apache-2.0 licensed;
see `content/LICENSE-Transformers`.
