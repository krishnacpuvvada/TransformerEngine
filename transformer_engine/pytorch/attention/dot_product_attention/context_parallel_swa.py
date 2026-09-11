# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Sliding window attention with context parallelism over point-to-point communication.

With the dual-chunk-swap layout the sequence is split into ``2 * cp_size`` chunks of
length ``L`` and rank ``r`` owns chunks ``r`` and ``2 * cp_size - 1 - r``. Under a causal
left window of ``W`` tokens, the queries of chunk ``a`` only attend to keys in
``[max(0, a*L - W), (a+1)*L)``. Each rank therefore fetches the windowed keys that precede
each of its chunks (the halo) from the ranks that own them, runs one attention call per
chunk over ``[halo, chunk]`` with a bottom-right causal mask, and returns the halo
gradients to their owners in backward. Communication and extra memory scale with ``W``
rather than with the sequence length.
"""
from collections import namedtuple

import torch

from transformer_engine.pytorch.utils import nvtx_range_pop, nvtx_range_push
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
    fused_attn_bwd,
    FusedAttnBackend,
)
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
)
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils


def get_chunk_owner(chunk_id, cp_size):
    """CP rank that owns sequence chunk ``chunk_id`` in the dual-chunk-swap layout."""
    return chunk_id if chunk_id < cp_size else 2 * cp_size - 1 - chunk_id


def get_halo_length(chunk_id, chunk_len, window):
    """Number of keys before chunk ``chunk_id`` that fall inside the window."""
    return min(window, chunk_id * chunk_len)


HaloPiece = namedtuple("HaloPiece", ["dst_chunk", "src_chunk", "src_start", "dst_start", "length"])


def get_halo_pieces(cp_size, chunk_len, window):
    """Pieces of preceding chunks that each chunk needs, in one order shared by all ranks.

    Chunk ``a`` needs positions ``[a*L - halo, a*L)``, which covers the tail of chunk
    ``a-1`` and, for windows wider than a chunk, whole earlier chunks. ``src_start`` is the
    offset within the source chunk and ``dst_start`` the offset within the destination's
    halo. Every rank derives its sends and receives from this list, so two ranks always post
    their shared messages in the same order.
    """
    pieces = []
    for dst_chunk in range(2 * cp_size):
        halo_start = dst_chunk * chunk_len - get_halo_length(dst_chunk, chunk_len, window)
        for src_chunk in range(dst_chunk):
            src_end = (src_chunk + 1) * chunk_len
            if src_end <= halo_start:
                continue
            start = max(src_chunk * chunk_len, halo_start)
            pieces.append(
                HaloPiece(
                    dst_chunk,
                    src_chunk,
                    start - src_chunk * chunk_len,
                    start - halo_start,
                    src_end - start,
                )
            )
    return pieces


def get_halo_kv_seqlens(cp_size, chunk_len, window):
    """Distinct K/V lengths of the per-chunk attention calls, for backend selection."""
    return sorted(
        {
            chunk_len + get_halo_length(chunk_id, chunk_len, window)
            for chunk_id in range(2 * cp_size)
        }
    )


def use_p2p_swa(qkv_format, attn_mask_type, window_size, max_seqlen_kv, cp_size):
    """Whether sliding window attention with ``cp_comm_type="p2p"`` uses the halo exchange.

    That is dense causal attention with a left window that moves fewer tokens than an
    all-gather would. Other sliding-window requests with p2p fall back to all-gather.
    """
    if qkv_format not in ["bshd", "sbhd"] or attn_mask_type != "causal":
        return False
    if window_size is None or window_size[0] < 1 or window_size[1] != 0:
        return False
    chunk_len = max_seqlen_kv // (2 * cp_size)
    return window_size[0] <= (cp_size - 1) * chunk_len


def _seq_shape(shape, seq_dim, seqlen):
    shape = list(shape)
    shape[seq_dim] = seqlen
    return shape


def _src_slice(x, piece, seq_dim, cp_size):
    """``piece`` within the owner's two-chunk tensor ``x``."""
    chunk = x.select(seq_dim, int(piece.src_chunk >= cp_size))
    return chunk.narrow(seq_dim, piece.src_start, piece.length)


def _dst_slice(ext, piece, seq_dim, cp_size):
    """``piece`` within the destination's per-chunk tensors ``ext``, whose halos come first."""
    return ext[int(piece.dst_chunk >= cp_size)].narrow(seq_dim, piece.dst_start, piece.length)


def _exchange(sends, recvs, cp_group):
    """Post all sends and receives as one batched point-to-point operation."""
    ops = [torch.distributed.P2POp(torch.distributed.isend, t, dst, cp_group) for t, dst in sends]
    ops += [torch.distributed.P2POp(torch.distributed.irecv, t, src, cp_group) for t, src in recvs]
    return torch.distributed.batch_isend_irecv(ops)


class AttnFuncWithCPAndKVP2PSWA(torch.autograd.Function):
    """
    Attention with context parallelism for a causal sliding window, using point-to-point
    communication of only the windowed K/V that precede each sequence chunk.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        qkv_format,
        deterministic,
        return_max_logit,
        window_size,
        cp_group,
        cp_global_ranks,
    ):
        # pylint: disable=missing-function-docstring,too-many-locals
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVP2PSWA.forward")

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        seq_dim = qkv_format.index("s")
        batch_size = q.shape[qkv_format.index("b")]
        chunk_len = q.shape[seq_dim] // 2
        window = window_size[0]
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        qkv_layout = "_".join([qkv_format] * 3)
        out_shape = q.shape[:-1] + v.shape[-1:]

        # [b, s, h, d] -> [b, 2, s//2, h, d] or [s, b, h, d] -> [2, s//2, b, h, d]
        q, k, v = [
            x.view(*x.shape[:seq_dim], 2, chunk_len, *x.shape[seq_dim + 1 :]) for x in [q, k, v]
        ]
        chunk_ids = [rank, 2 * cp_size - 1 - rank]
        halo_lens = [get_halo_length(chunk_id, chunk_len, window) for chunk_id in chunk_ids]

        # K/V for each chunk's attention call: [halo, chunk] along the sequence dimension
        k_ext, v_ext = [None, None], [None, None]
        for i in range(2):
            for x, ext in [(k, k_ext), (v, v_ext)]:
                chunk = x.select(seq_dim, i)
                if halo_lens[i] == 0:
                    ext[i] = chunk.contiguous()
                else:
                    ext[i] = torch.empty(
                        _seq_shape(chunk.shape, seq_dim, chunk_len + halo_lens[i]),
                        dtype=chunk.dtype,
                        device=chunk.device,
                    )
                    ext[i].narrow(seq_dim, halo_lens[i], chunk_len).copy_(chunk)

        # exchange halo pieces; pieces a rank needs from itself are plain copies
        sends, recvs, staged = [], [], []
        needs_comm = [False, False]
        for piece in get_halo_pieces(cp_size, chunk_len, window):
            src_rank = get_chunk_owner(piece.src_chunk, cp_size)
            dst_rank = get_chunk_owner(piece.dst_chunk, cp_size)
            if src_rank == rank == dst_rank:
                for x, ext in [(k, k_ext), (v, v_ext)]:
                    _dst_slice(ext, piece, seq_dim, cp_size).copy_(
                        _src_slice(x, piece, seq_dim, cp_size)
                    )
            elif src_rank == rank:
                for x in [k, v]:
                    sends.append(
                        (
                            _src_slice(x, piece, seq_dim, cp_size).contiguous(),
                            cp_global_ranks[dst_rank],
                        )
                    )
            elif dst_rank == rank:
                for ext in [k_ext, v_ext]:
                    dst = _dst_slice(ext, piece, seq_dim, cp_size)
                    buf = torch.empty(dst.shape, dtype=dst.dtype, device=dst.device)
                    recvs.append((buf, cp_global_ranks[src_rank]))
                    staged.append((buf, dst))
                needs_comm[int(piece.dst_chunk >= cp_size)] = True
        reqs = _exchange(sends, recvs, cp_group)

        # attention per chunk, chunks that need no communication first
        out = torch.empty(*q.shape[:-1], v.shape[-1], dtype=q.dtype, device=q.device)
        cu_seqlens_q = dpa_utils.get_full_cu_seqlens(batch_size, chunk_len, q.device)
        cu_seqlens_kv, softmax_lse, rng_state = [None, None], [None, None], [None, None]
        max_logit = None
        waited = False
        for i in sorted(range(2), key=lambda i: needs_comm[i]):
            if needs_comm[i] and not waited:
                for req in reqs:
                    req.wait()
                for buf, dst in staged:
                    dst.copy_(buf)
                waited = True
            cu_seqlens_kv[i] = dpa_utils.get_full_cu_seqlens(
                batch_size, chunk_len + halo_lens[i], q.device
            )
            out_per_chunk, aux_ctx_tensors, *max_logit_per_chunk = fused_attn_fwd(
                is_training,
                chunk_len,
                chunk_len + halo_lens[i],
                cu_seqlens_q,
                cu_seqlens_kv[i],
                q.select(seq_dim, i).contiguous(),
                k_ext[i],
                v_ext[i],
                q.dtype,
                FusedAttnBackend["F16_arbitrary_seqlen"],
                attn_scale=softmax_scale,
                dropout=dropout_p,
                qkv_layout=qkv_layout,
                o_format=qkv_format,
                attn_mask_type="causal_bottom_right",
                window_size=(window, 0),
                return_max_logit=return_max_logit,
                cuda_graph=is_graph_capturing(),
            )
            out.select(seq_dim, i).copy_(out_per_chunk)
            softmax_lse[i], rng_state[i], *_ = aux_ctx_tensors
            if return_max_logit:
                max_logit = (
                    max_logit_per_chunk[0]
                    if max_logit is None
                    else torch.maximum(max_logit, max_logit_per_chunk[0])
                )
        if return_max_logit:
            torch.distributed.all_reduce(
                max_logit, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )

        ctx.save_for_backward(
            q, *k_ext, *v_ext, out, *softmax_lse, *rng_state, cu_seqlens_q, *cu_seqlens_kv
        )
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.qkv_format = qkv_format
        ctx.qkv_layout = qkv_layout
        ctx.seq_dim = seq_dim
        ctx.chunk_len = chunk_len
        ctx.window = window
        ctx.halo_lens = halo_lens
        ctx.k_shape, ctx.v_shape = k.shape, v.shape
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic

        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2PSWA.forward")
        out = out.view(out_shape)
        if return_max_logit:
            return out, max_logit
        return out

    @staticmethod
    def backward(ctx, dout, *_args):
        # pylint: disable=missing-function-docstring,too-many-locals
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVP2PSWA.backward")
        q, *saved = ctx.saved_tensors
        k_ext, v_ext = saved[0:2], saved[2:4]
        out = saved[4]
        softmax_lse, rng_state = saved[5:7], saved[7:9]
        cu_seqlens_q, cu_seqlens_kv = saved[9], saved[10:12]

        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        seq_dim, chunk_len, halo_lens = ctx.seq_dim, ctx.chunk_len, ctx.halo_lens
        dout = dout.contiguous().view(out.shape)

        dq = torch.empty_like(q)
        dk = torch.empty(ctx.k_shape, dtype=q.dtype, device=q.device)
        dv = torch.empty(ctx.v_shape, dtype=q.dtype, device=q.device)
        dk_halo, dv_halo = [None, None], [None, None]
        for i in range(2):
            dq_per_chunk, dk_per_chunk, dv_per_chunk, *_ = fused_attn_bwd(
                chunk_len,
                chunk_len + halo_lens[i],
                cu_seqlens_q,
                cu_seqlens_kv[i],
                q.select(seq_dim, i).contiguous(),
                k_ext[i],
                v_ext[i],
                out.select(seq_dim, i).contiguous(),
                dout.select(seq_dim, i).contiguous(),
                q.dtype,
                [softmax_lse[i], rng_state[i]],
                FusedAttnBackend["F16_arbitrary_seqlen"],
                attn_scale=ctx.softmax_scale,
                dropout=ctx.dropout_p,
                qkv_layout=ctx.qkv_layout,
                o_format=ctx.qkv_format,
                do_format=ctx.qkv_format,
                dqkv_layout=ctx.qkv_layout,
                attn_mask_type="causal_bottom_right",
                window_size=(ctx.window, 0),
                deterministic=ctx.deterministic,
                cuda_graph=is_graph_capturing(),
            )
            dq.select(seq_dim, i).copy_(dq_per_chunk)
            dk.select(seq_dim, i).copy_(dk_per_chunk.narrow(seq_dim, halo_lens[i], chunk_len))
            dv.select(seq_dim, i).copy_(dv_per_chunk.narrow(seq_dim, halo_lens[i], chunk_len))
            dk_halo[i] = dk_per_chunk.narrow(seq_dim, 0, halo_lens[i])
            dv_halo[i] = dv_per_chunk.narrow(seq_dim, 0, halo_lens[i])

        # return halo gradients to the owners of the pieces and accumulate them there; the
        # piece order is the same on every run, which keeps the accumulation deterministic
        sends, recvs, staged = [], [], []
        for piece in get_halo_pieces(cp_size, chunk_len, ctx.window):
            src_rank = get_chunk_owner(piece.src_chunk, cp_size)
            dst_rank = get_chunk_owner(piece.dst_chunk, cp_size)
            if src_rank == rank == dst_rank:
                for x, halo in [(dk, dk_halo), (dv, dv_halo)]:
                    _src_slice(x, piece, seq_dim, cp_size).add_(
                        _dst_slice(halo, piece, seq_dim, cp_size)
                    )
            elif dst_rank == rank:
                for halo in [dk_halo, dv_halo]:
                    sends.append(
                        (
                            _dst_slice(halo, piece, seq_dim, cp_size).contiguous(),
                            ctx.cp_global_ranks[src_rank],
                        )
                    )
            elif src_rank == rank:
                for x in [dk, dv]:
                    target = _src_slice(x, piece, seq_dim, cp_size)
                    buf = torch.empty(target.shape, dtype=target.dtype, device=target.device)
                    recvs.append((buf, ctx.cp_global_ranks[dst_rank]))
                    staged.append((buf, target))
        for req in _exchange(sends, recvs, ctx.cp_group):
            req.wait()
        for buf, target in staged:
            target.add_(buf)

        dq = dq.view(*dq.shape[:seq_dim], -1, *dq.shape[seq_dim + 2 :])
        dk = dk.view(*dk.shape[:seq_dim], -1, *dk.shape[seq_dim + 2 :])
        dv = dv.view(*dv.shape[:seq_dim], -1, *dv.shape[seq_dim + 2 :])
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2PSWA.backward")
        return (None, dq, dk, dv) + (None,) * 9
