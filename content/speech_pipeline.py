"""Browser-local Whisper inference using the original pretrained PyTorch model."""
import asyncio
import time

import numpy as np
import torch
from whisper_graphs import AudioEncoder, DecoderStep, empty_self_cache, greedy_token


class WhisperSpeech:
    def __init__(self, model, tokenizer, extractor, *, device, gpu=None, cache_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.device = device
        self.gpu = gpu
        self.cache_length = cache_length
        backend = gpu.backend if gpu is not None else 'turbine_cpu'
        self.encode = torch.compile(AudioEncoder(model), backend=backend, fullgraph=True, dynamic=False)
        self.decode = torch.compile(DecoderStep(model, cache_length), backend=backend, fullgraph=True, dynamic=False)

    async def _complete(self):
        if self.gpu is not None:
            await self.gpu.synchronize()

    async def _host(self, value):
        return await self.gpu.to_cpu(value) if self.gpu is not None else value.detach()

    async def transcribe(self, audio, *, language='en', on_partial=None,
                         cancelled=lambda: False, retain_reference=False):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not 0 < len(audio) <= 15 * 16000 or not np.isfinite(audio).all():
            raise ValueError('Expected a finite mono clip of at most 15 seconds at 16 kHz.')
        duration = len(audio) / 16000
        started = time.perf_counter()
        timings = dict(features_ms=0.0, encoder_ms=0.0, decoder_ms=0.0)
        # Match the playground's silence gate, before invoking the model.
        if float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < 4e-4:
            return dict(text='', token_ids=[], audio_seconds=duration, no_speech=True,
                        cancelled=False, truncated=False, timings=timings,
                        transcription_seconds=time.perf_counter() - started)
        features = self.extractor(audio, sampling_rate=16000, return_tensors='pt').input_features
        features_device = features.to(self.device)
        states = empty_self_cache(self.model.config, self.cache_length, self.device)
        await self._complete()
        encoded_at = time.perf_counter()
        timings['features_ms'] = 1000 * (encoded_at - started)
        if cancelled():
            return dict(text='', token_ids=[], audio_seconds=duration, no_speech=False,
                        cancelled=True, truncated=False, timings=timings,
                        transcription_seconds=time.perf_counter() - started)
        with torch.inference_mode():
            hidden, cross = self.encode(features_device)
        await self._complete()
        timings['encoder_ms'] = 1000 * (time.perf_counter() - encoded_at)
        encoder_reference = (await self._host(hidden)).clone() if retain_reference else None
        prompt = self.tokenizer.get_decoder_prompt_ids(language=language, task='transcribe', no_timestamps=True)
        prefix = [self.model.generation_config.decoder_start_token_id] + [token for _, token in prompt]
        tokens = torch.tensor([prefix], dtype=torch.int64).to(self.device)
        positions = torch.arange(len(prefix), dtype=torch.int64).to(self.device)
        ids = []
        reference_logits = None
        decoder_started = time.perf_counter()
        with torch.inference_mode():
            logits, states = self.decode(tokens, positions, hidden, states, cross)
        await self._complete()
        while len(prefix) + len(ids) < self.cache_length:
            if cancelled():
                break
            host_logits = await self._host(logits)
            if reference_logits is None and retain_reference:
                reference_logits = host_logits.clone()
            next_id = greedy_token(host_logits, self.model.generation_config, first=not ids)
            if next_id == self.model.generation_config.eos_token_id:
                break
            ids.append(next_id)
            if on_partial is not None:
                on_partial(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
            # Give UI cancellation a turn on both CPU and WebGPU.
            await asyncio.sleep(0)
            if cancelled() or len(prefix) + len(ids) >= self.cache_length:
                break
            tokens = torch.tensor([[next_id]], dtype=torch.int64).to(self.device)
            positions = torch.tensor([len(prefix) + len(ids) - 1], dtype=torch.int64).to(self.device)
            with torch.inference_mode():
                logits, states = self.decode(tokens, positions, hidden, states, cross)
            await self._complete()
        timings['decoder_ms'] = 1000 * (time.perf_counter() - decoder_started)
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        result = dict(text=text, token_ids=ids, audio_seconds=duration, no_speech=False,
                      cancelled=cancelled(), truncated=len(prefix) + len(ids) >= self.cache_length,
                      timings=timings, transcription_seconds=time.perf_counter() - started)
        if retain_reference:
            result['encoder_reference'] = encoder_reference
            result['first_logits'] = reference_logits
        return result
