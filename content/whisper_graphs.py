"""Compilation boundaries around the original Transformers Whisper modules.

The model, weights and attention implementation are unchanged. Tensor-only graph
inputs/outputs keep both self-attention and cross-attention caches on the device.
"""
import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, StaticCache


class AudioEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.decoder = model.model.decoder

    def forward(self, features):
        hidden = self.encoder(features, return_dict=False)[0]
        batch = hidden.shape[0]
        cross = []
        for layer in self.decoder.layers:
            attention = layer.encoder_attn
            key = attention.k_proj(hidden).view(batch, -1, attention.num_heads, attention.head_dim)
            value = attention.v_proj(hidden).view(batch, -1, attention.num_heads, attention.head_dim)
            cross.extend((key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()))
        return hidden, tuple(cross)


class DecoderStep(torch.nn.Module):
    def __init__(self, model, cache_length):
        super().__init__()
        self.decoder = model.model.decoder
        self.output = model.proj_out
        self.config = model.config
        self.cache_length = cache_length

    def forward(self, tokens, positions, hidden, self_states, cross_states):
        cache = EncoderDecoderCache(
            StaticCache(config=self.config, max_cache_len=self.cache_length),
            DynamicCache(config=self.config),
        )
        for index in range(self.config.decoder_layers):
            key, value = self_states[index * 2:index * 2 + 2]
            layer = cache.self_attention_cache.layers[index]
            layer.keys, layer.values = key, value
            layer.max_batch_size, layer.num_heads, _, layer.head_dim = key.shape
            layer.dtype, layer.device = key.dtype, key.device
            layer.is_initialized = True
            cross = cache.cross_attention_cache.layers[index]
            cross.keys, cross.values = cross_states[index * 2:index * 2 + 2]
            cross.is_initialized = True
            cache.is_updated[index] = True
        # An explicit 4-D causal mask uses the standard decoder's supported mask
        # input and avoids host reads of the device-side cache position.
        columns = torch.arange(self.cache_length, device=positions.device)
        visible = columns[None, :] <= positions[:, None]
        mask = torch.where(visible, 0.0, torch.finfo(hidden.dtype).min)[None, None]
        result = self.decoder(
            input_ids=tokens, encoder_hidden_states=hidden, past_key_values=cache,
            cache_position=positions, position_ids=positions[None],
            attention_mask=mask, use_cache=True, return_dict=False,
        )[0]
        logits = self.output(result[:, -1:])[:, 0]
        updated = tuple(state for layer in cache.self_attention_cache.layers
                        for state in (layer.keys, layer.values))
        return logits, updated


def empty_self_cache(config, length, device):
    shape = (1, config.decoder_attention_heads, length,
             config.d_model // config.decoder_attention_heads)
    return tuple(torch.zeros(shape, dtype=torch.float32).to(device)
                 for _ in range(config.decoder_layers * 2))


def greedy_token(logits, generation_config, first=False):
    """Host-side greedy sampling with the checkpoint's token suppression rules."""
    scores = logits.clone()
    scores[:, generation_config.suppress_tokens] = -float('inf')
    scores[:, generation_config.no_timestamps_token_id + 1:] = -float('inf')
    if first:
        scores[:, generation_config.begin_suppress_tokens] = -float('inf')
    return int(scores.argmax(dim=-1).item())
