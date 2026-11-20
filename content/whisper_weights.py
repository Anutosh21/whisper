"""Fetch and verify the original checkpoint before loading the local model."""

from pathlib import Path

import pyjs


WEIGHTS_URL = 'https://huggingface.co/openai/whisper-base/resolve/e37978b90ca9030d5170a5c07aadb050351a65bb/pytorch_model.bin?download=true'
SHA256 = 'c37f294c9563a49217d108687fbd3cda5bdf0441a451512d87bf66343ab31a87'
CACHE_NAME = 'whisper-base-weights-v1'

_FETCH = '''async (url, expectedHash, cacheName) => {
    let cache = null;
    let response;
    try {
        cache = await caches.open(cacheName);
        response = await cache.match(url);
    } catch { cache = null; }
    const cached = Boolean(response);
    if (!response) response = await fetch(url);
    if (!response.ok) throw Error(`Weights download failed: HTTP ${response.status}`);
    const data = await response.arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)),
        byte => byte.toString(16).padStart(2, '0')).join('');
    if (hash !== expectedHash) {
        if (cache) await cache.delete(url);
        throw Error('Pretrained weights failed the SHA-256 check; rerun to download again.');
    }
    let saved = cached;
    if (cache && !cached) {
        try {
            await cache.put(url, new Response(data));
            saved = true;
        } catch { /* Downloaded weights remain usable if storage is unavailable. */ }
    }
    return {data, cached, saved};
}'''


async def ensure_checkpoint(directory):
    """Make the verified checkpoint available to Transformers' local loader."""
    checkpoint = Path(directory) / 'pytorch_model.bin'
    if checkpoint.exists():
        return checkpoint
    print('Loading Whisper base weights (first download: 277 MiB)...', flush=True)
    result = await pyjs.js.eval(_FETCH)(WEIGHTS_URL, SHA256, CACHE_NAME)
    if pyjs.to_py(result.cached):
        print('Using verified cached weights.', flush=True)
    elif pyjs.to_py(result.saved):
        print('Weights verified and saved in the browser cache.', flush=True)
    else:
        print('Weights verified; browser storage is unavailable for caching.', flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.partial')
    try:
        temporary.write_bytes(bytes(pyjs.to_py(result.data)))
        temporary.replace(checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint
