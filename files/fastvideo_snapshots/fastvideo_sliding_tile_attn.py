"""Standalone FastVideo sliding-tile attention Triton snapshot for Apex."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _is_cdna() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip"


def _autotune_configs():
    supported_num_stages = [1, 2] if _is_cdna() else [1, 2, 3, 4]
    return [
        triton.Config({"BLOCK_Q": bq, "BLOCK_KV": bkv}, num_stages=s, num_warps=w)
        for bq in (32, 64, 128)
        for bkv in (32, 64, 128)
        for s in supported_num_stages
        for w in (4, 8)
    ]


@triton.jit
def _attn_fwd_loop(q, k, v, kv_mask, m, l, acc, sm_scale, MASK_KV: tl.constexpr):
    scores = tl.dot(q, k.T) * sm_scale
    if MASK_KV:
        scores = tl.where(kv_mask[None, :], scores, -float("inf"))
    current_m = tl.max(scores, axis=1)
    new_m = tl.maximum(m, current_m)
    exp_scores = tl.math.exp2(scores - new_m[:, None])
    current_l = tl.sum(exp_scores, axis=1)
    alpha = tl.math.exp2(m - new_m)
    l = l * alpha + current_l
    m = new_m
    acc = acc * alpha[:, None] + tl.dot(exp_scores.to(v.type.element_ty), v)
    return m, l, acc


@triton.autotune(configs=_autotune_configs(), key=["head_dim"])
@triton.jit
def triton_sta_kernel(
    Q,
    K,
    V,
    output,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    img_seq_len: int,
    text_length: int,
    canvas_t: int,
    canvas_h: int,
    canvas_w: int,
    kernel_t: int,
    kernel_h: int,
    kernel_w: int,
    tile_t: int,
    tile_h: int,
    tile_w: int,
    scale: float,
    has_text: tl.constexpr,
    text_q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    total_tile_size = tile_t * tile_h * tile_w
    q_block_per_tile = (total_tile_size + BLOCK_Q - 1) // BLOCK_Q
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2) if text_q else tl.program_id(2) % q_block_per_tile
    q_tile_flat = 0 if text_q else tl.program_id(2) // q_block_per_tile
    q_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim
    q_base_idx = img_seq_len + q_block_idx * BLOCK_Q if text_q else q_tile_flat * total_tile_size + q_block_idx * BLOCK_Q
    q_idx = q_base_idx + tl.arange(0, BLOCK_Q)
    q_mask = (q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)) < total_tile_size
    q = tl.load(Q + q_offset + q_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :], mask=q_mask[:, None], other=0.0)
    sm_scale = scale * 1.4426950408889634
    kv_idx = tl.arange(0, BLOCK_KV)
    kv_mask = kv_idx < total_tile_size
    k = tl.load(K + q_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :], mask=kv_mask[:, None], other=0.0)
    v = tl.load(V + q_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :], mask=kv_mask[:, None], other=0.0)
    m = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_DIM), dtype=tl.float32)
    m, l, acc = _attn_fwd_loop(q, k, v, kv_mask, m, l, acc, sm_scale, True)
    output_acc = acc / l[:, None]
    tl.store(output + q_offset + q_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :], output_acc, mask=q_mask[:, None])


def sliding_tile_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size,
    text_length: int,
    has_text: bool = True,
    dit_seq_shape: str = "18x48x80",
) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = q.shape
    if dit_seq_shape == "18x48x80":
        canvas_t, canvas_h, canvas_w = 18, 48, 80
        tile_t, tile_h, tile_w = 3, 8, 8
    else:
        canvas_t, canvas_h, canvas_w = 30, 48, 80
        tile_t, tile_h, tile_w = 6, 8, 8
    out = torch.empty_like(q)
    grid = (batch_size, num_heads, math.ceil((tile_t * tile_h * tile_w) / 64))
    kernel_t, kernel_h, kernel_w = window_size[0]
    triton_sta_kernel[grid](
        q,
        k,
        v,
        out,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        seq_len - text_length,
        text_length,
        canvas_t,
        canvas_h,
        canvas_w,
        kernel_t,
        kernel_h,
        kernel_w,
        tile_t,
        tile_h,
        tile_w,
        head_dim**-0.5,
        has_text=has_text,
        text_q=False,
        BLOCK_DIM=head_dim,
    )
    return out
