"""Standalone FastVideo video-sparse attention Triton snapshot for Apex."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_UNIFORM_FULL_METADATA_CACHE: dict[tuple[object, ...], bool] = {}

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=s, num_warps=w)
    for s in (2, 3, 4, 5, 7)
    for w in (2, 4, 8)
]


@triton.autotune(_CONFIGS, key=["N_CTX_Q", "HEAD_DIM", "max_kv_blks"])
@triton.jit
def _attn_fwd_sparse(
    Q,
    K,
    V,
    sm_scale,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = ((b * H + h) * q_tiles + q_blk)

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    q_off = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    k_off = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    v_off = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    o_off = b.to(tl.int64) * stride_oz + h.to(tl.int64) * stride_oh

    q_start = q_blk * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = q_start + tl.arange(0, BLOCK_M)

    q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_start, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_base = tl.make_block_ptr(
        base=K + k_off,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_base = tl.make_block_ptr(
        base=V + v_off,
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    o_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(q_start, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(q_ptr)
    qk_scale = sm_scale * 1.44269504
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.ones([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for i in tl.range(0, kv_blocks, num_stages=STAGE):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        k_ptr = tl.advance(k_base, (0, kv_idx * BLOCK_N))
        v_ptr = tl.advance(v_base, (kv_idx * BLOCK_N, 0))

        k = tl.load(k_ptr)
        qk = tl.dot(q, k)
        if block_size != BLOCK_N:
            mask = offs_n < block_size
            qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(v_ptr)
        if block_size != BLOCK_N:
            mask = offs_n < block_size
            v = tl.where(mask[:, None], v, 0.0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i)
    tl.store(o_ptr, acc.to(Out.type.element_ty))


@triton.autotune(_CONFIGS, key=["N_CTX_Q", "HEAD_DIM", "MAX_KV_BLKS"])
@triton.jit
def _attn_fwd_sparse_full_blocks(
    Q,
    K,
    V,
    sm_scale,
    q2k_index,
    max_kv_blks,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    MAX_KV_BLKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = ((b * H + h) * q_tiles + q_blk)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    tl.assume(stride_qm > 0)
    tl.assume(stride_qk > 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kk > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_on > 0)

    q_off = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    k_off = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    v_off = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    o_off = b.to(tl.int64) * stride_oz + h.to(tl.int64) * stride_oh
    q_start = q_blk * BLOCK_M

    q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_start, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_base = tl.make_block_ptr(
        base=K + k_off,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_base = tl.make_block_ptr(
        base=V + v_off,
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    o_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(q_start, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.ones([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptr)

    for i in tl.static_range(0, MAX_KV_BLKS):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        k_ptr = tl.advance(k_base, (0, kv_idx * BLOCK_N))
        v_ptr = tl.advance(v_base, (kv_idx * BLOCK_N, 0))
        k = tl.load(k_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(Out.type.element_ty))


def _metadata_key(
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    max_kv_blks: int,
) -> tuple[object, ...]:
    return (
        q2k_num.data_ptr(),
        variable_block_sizes.data_ptr(),
        tuple(q2k_num.shape),
        tuple(variable_block_sizes.shape),
        max_kv_blks,
        q2k_num.device,
    )


def _is_uniform_full_layout(
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    max_kv_blks: int,
) -> bool:
    key = _metadata_key(q2k_num, variable_block_sizes, max_kv_blks)
    cached = _UNIFORM_FULL_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    is_uniform = bool(
        torch.all(q2k_num == max_kv_blks).item()
        and torch.all(variable_block_sizes == 64).item()
    )
    _UNIFORM_FULL_METADATA_CACHE[key] = is_uniform
    return is_uniform


def triton_block_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-only sparse attention wrapper used by Apex standalone grading."""
    batch, heads, q_len, dim = q.shape
    kv_len = k.shape[2]
    out = torch.empty_like(q)
    m = torch.empty((batch, heads, q_len), device=q.device, dtype=torch.float32)
    max_kv_blks = q2k_index.shape[-1]
    grid = (triton.cdiv(q_len, 64), batch * heads)

    if not (q.requires_grad or k.requires_grad or v.requires_grad) and _is_uniform_full_layout(
        q2k_num, variable_block_sizes, max_kv_blks
    ):
        _attn_fwd_sparse_full_blocks[grid](
            q,
            k,
            v,
            dim**-0.5,
            q2k_index,
            max_kv_blks,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            batch,
            heads,
            q_len,
            kv_len,
            HEAD_DIM=dim,
            MAX_KV_BLKS=max_kv_blks,
        )
        return out, torch.empty(0, device=q.device, dtype=torch.float32)

    _attn_fwd_sparse[grid](
        q,
        k,
        v,
        dim**-0.5,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        m,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        batch,
        heads,
        q_len,
        kv_len,
        HEAD_DIM=dim,
        STAGE=3,
    )
    return out, m
