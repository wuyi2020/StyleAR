import torch
from torch import nn
from typing import Optional, Union, Tuple


import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

    
# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=1280,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=4096, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LinearLora(nn.Module):
    def __init__(self, in_features, out_features, r=4, bias=False):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(r).float())
        self.A = nn.Parameter(torch.randn(in_features, r) * std_dev)
        self.B = nn.Parameter(torch.zeros(r, out_features))
        # self.down = nn.Linear(in_features, r, bias=bias)
        # self.up = nn.Linear(r, out_features, bias=bias)
    
    def forward(self, x):
        x = x @ self.A @ self.B
        # x = self.down(x)
        # x = self.up(x)
        return x


class ChameleonSdpaAttentionLora(nn.Module):
    def __init__(self, attn, r):
        super().__init__()
        hidden_size = attn.config.hidden_size
        self.hidden_size = hidden_size
        self.r = r
        self.attn = attn

        # block diagonals that turn into the identity matrix
        
        self.q_proj_lora = LinearLora(in_features=self.attn.hidden_size, out_features=self.attn.num_heads * self.attn.head_dim, r=r, bias=self.attn.config.attention_bias)
        self.k_proj_lora = LinearLora(in_features=self.attn.hidden_size, out_features=self.attn.num_key_value_heads * self.attn.head_dim, r=r, bias=self.attn.config.attention_bias)
        self.v_proj_lora = LinearLora(in_features=self.attn.hidden_size, out_features=self.attn.num_key_value_heads * self.attn.head_dim, r=r, bias=self.attn.config.attention_bias)
        self.o_proj_lora = LinearLora(in_features=self.attn.hidden_size, out_features=self.attn.hidden_size, r=r, bias=self.attn.config.attention_bias)

    # Adapted from ChameleonAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.attn.q_proj(hidden_states) + self.q_proj_lora(hidden_states)
        key_states = self.attn.k_proj(hidden_states) + self.k_proj_lora(hidden_states)
        value_states = self.attn.v_proj(hidden_states) + self.v_proj_lora(hidden_states)

        query_states = query_states.reshape(-1, self.attn.num_heads, self.attn.head_dim)
        query_states = self.attn.q_norm(query_states)

        key_states = key_states.reshape(-1, self.attn.num_key_value_heads, self.attn.head_dim)
        key_states = self.attn.k_norm(key_states)

        query_states = query_states.reshape(bsz, q_len, self.attn.num_heads, self.attn.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.attn.num_key_value_heads, self.attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.attn.num_key_value_heads, self.attn.head_dim).transpose(1, 2)

        cos, sin = self.attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.attn.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self.attn.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attn.attention_dropout if self.attn.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.attn.hidden_size)

        attn_output = self.attn.o_proj(attn_output) + self.o_proj_lora(attn_output)

        return attn_output, None, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaMLP with Llama->Chameleon
class ChameleonMLPLora(nn.Module):
    def __init__(self, mlp, r):
        super().__init__()
        self.r = r
        self.mlp = mlp

        # block diagonals that turn into the identity matrix
        # self.gate_proj_lora = LinearLora(self.mlp.hidden_size, self.mlp.intermediate_size, r=r, bias=self.mlp.config.mlp_bias)
        # self.up_proj_lora = LinearLora(self.mlp.hidden_size, self.mlp.intermediate_size, r=r, bias=self.mlp.config.mlp_bias)
        self.down_proj_lora = LinearLora(self.mlp.intermediate_size, self.mlp.hidden_size, r=r, bias=self.mlp.config.mlp_bias)

    # Ignore copy
    def forward(self, x):
        hs = self.mlp.act_fn(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        down_proj = self.mlp.down_proj(hs) + self.down_proj_lora(hs)
        return down_proj


# Copied from transformers.models.llama.modeling_llama.LlamaMLP with Llama->Chameleon
class LMHeadLora(nn.Module):
    def __init__(self, mlp, r, bias=False):
        super().__init__()
        self.r = r
        self.mlp = mlp

        # block diagonals that turn into the identity matrix
        self.lm_head_lora = LinearLora(self.mlp.weight.shape[1], self.mlp.weight.shape[0], r=r, bias=bias)

    # Ignore copy
    def forward(self, x):
        x = self.mlp(x) + self.lm_head_lora(x)
        return x