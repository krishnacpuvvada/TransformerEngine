# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Sliding window attention with context parallelism over point-to-point communication.

With the dual-chunk-swap layout every sequence is split into ``2 * cp_size`` chunks of
length ``L`` and rank ``r`` owns chunks ``r`` and ``2 * cp_size - 1 - r``. Under a causal
left window of ``W`` tokens, the queries of chunk ``a`` only attend to keys in
``[max(0, a*L - W), (a+1)*L)``. Each rank therefore fetches the windowed keys that precede
each of its chunks (the halo) from the ranks that own them, runs one attention call per
chunk over ``[halo, chunk]`` with a bottom-right causal mask, and returns the halo
gradients to their owners in backward. Communication and extra memory scale with ``W``
rather than with the sequence length.

Dense ``bshd``/``sbhd`` inputs are one sequence. Packed ``thd`` inputs apply the same
schedule to every sequence of the pack, with one attention call per chunk half covering
all sequences.
"""
from collections import namedtuple

import torch
import transformer_engine_torch as tex

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

    That is causal attention with a left window, dense or packed, that moves fewer tokens
    than an all-gather would. Other sliding-window requests with p2p fall back to
    all-gather. For packed inputs ``max_seqlen_kv`` is the longest sequence of the pack.
    """
    dense = qkv_format in ["bshd", "sbhd"] and attn_mask_type == "causal"
    packed = qkv_format == "thd" and attn_mask_type == "padding_causal"
    if not (dense or packed):
        return False
    if window_size is None or window_size[0] < 1 or window_size[1] != 0:
        return False
    chunk_len = max_seqlen_kv // (2 * cp_size)
    return window_size[0] <= (cp_size - 1) * chunk_len


# One chunk owned by this rank: ``start`` is its offset along the local sequence axis and
# ``ext_start`` the offset of its ``[halo, chunk]`` segment in the K/V buffer of its half.
Chunk = namedtuple("Chunk", ["doc", "half", "chunk_id", "start", "length", "halo", "ext_start"])


def get_local_chunks(rank, cp_size, window, chunk_lens):
    """This rank's chunks of every sequence, two per sequence in sequence order, and the
    total length of the K/V buffer of each half. ``chunk_lens[i]`` is the chunk length of
    sequence ``i``; a dense input is one sequence."""
    chunks = []
    start = 0
    ext_total = [0, 0]
    for doc, length in enumerate(chunk_lens):
        for half, chunk_id in enumerate([rank, 2 * cp_size - 1 - rank]):
            halo = get_halo_length(chunk_id, length, window)
            chunks.append(
                Chunk(doc, half, chunk_id, start + half * length, length, halo, ext_total[half])
            )
            ext_total[half] += halo + length
        start += 2 * length
    return chunks, ext_total


def _seq_shape(shape, seq_dim, seqlen):
    shape = list(shape)
    shape[seq_dim] = seqlen
    return shape


def _src_slice(x, seq_dim, doc_start, length, piece, cp_size):
    """``piece`` within the owner's local tensor ``x``, for a sequence whose local tokens start
    at ``doc_start`` and whose chunks have ``length`` tokens."""
    half = int(piece.src_chunk >= cp_size)
    return x.narrow(seq_dim, doc_start + half * length + piece.src_start, piece.length)


def _dst_slice(ext, seq_dim, chunk, piece):
    """``piece`` within the K/V buffer of the destination chunk's half."""
    return ext[chunk.half].narrow(seq_dim, chunk.ext_start + piece.dst_start, piece.length)


def _exchange(transfers, seq_dim, cp_group, cp_global_ranks):
    """Post the halo exchange as one batched point-to-point operation.

    ``transfers`` holds ``(sends, recvs)`` pairs, one per tensor kind (K, V), where ``sends``
    and ``recvs`` map a peer rank to its slices in the global piece order. Pieces for one peer
    travel in one message, so the message count does not grow with the number of sequences.
    Returns the requests and the receive buffers paired with their destination slices.
    """
    ops, staged = [], []
    for sends, recvs in transfers:
        for peer, slices in sends.items():
            ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    torch.cat(slices, dim=seq_dim),
                    cp_global_ranks[peer],
                    cp_group,
                )
            )
        for peer, slices in recvs.items():
            total = sum(s.shape[seq_dim] for s in slices)
            buf = torch.empty(
                _seq_shape(slices[0].shape, seq_dim, total),
                dtype=slices[0].dtype,
                device=slices[0].device,
            )
            ops.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv, buf, cp_global_ranks[peer], cp_group
                )
            )
            staged.append((buf, slices))
    return torch.distributed.batch_isend_irecv(ops), staged


def _unpack(staged, seq_dim, accumulate):
    for buf, slices in staged:
        lengths = [s.shape[seq_dim] for s in slices]
        for piece, dst in zip(torch.split(buf, lengths, dim=seq_dim), slices):
            if accumulate:
                dst.add_(piece)
            else:
                dst.copy_(piece)


def _packed_seqlens(chunks, ext_total, actual_lens, total_tokens, half, device):
    """cuDNN THD metadata for one half's call over all sequences: offsets of the queries in
    the local Q and their valid counts, offsets of each ``[halo, chunk]`` segment in the
    half's K/V buffer and their valid counts. Padding sits at the end of a sequence, so a
    chunk with any valid query always has a fully valid halo."""
    q_offsets, kv_offsets, q_valid, kv_valid = [], [], [0], [0]
    for c in chunks[half::2]:
        q_offsets.append(c.start)
        kv_offsets.append(c.ext_start)
        actual = actual_lens[c.doc]
        q_valid.append(q_valid[-1] + min(max(actual - c.chunk_id * c.length, 0), c.length))
        kv_valid.append(
            kv_valid[-1] + min(max(actual - (c.chunk_id * c.length - c.halo), 0), c.halo + c.length)
        )
    q_offsets.append(total_tokens)
    kv_offsets.append(ext_total[half])
    return [
        torch.tensor(x, dtype=torch.int32, device=device)
        for x in [q_valid, q_offsets, kv_valid, kv_offsets]
    ]


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
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
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
        packed = qkv_format == "thd"
        seq_dim = 0 if packed else qkv_format.index("s")
        window = window_size[0]
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        qkv_layout = "_".join([qkv_format] * 3)
        attn_mask_type = "padding_causal_bottom_right" if packed else "causal_bottom_right"
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if packed:
            # Sequence lengths are needed as integers for the schedule and the message sizes:
            # one device-to-host copy per call.
            cu_q, cu_q_padded, cu_kv, cu_kv_padded = torch.stack(
                [cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv, cu_seqlens_kv_padded]
            ).tolist()
            assert cu_q == cu_kv and cu_q_padded == cu_kv_padded, (
                "Sliding window attention with cp_comm_type='p2p' requires the same sequence"
                " packing for Q and K/V."
            )
            seqlens = [end - start for start, end in zip(cu_kv[:-1], cu_kv[1:])]
            chunk_lens = [
                (end - start) // (2 * cp_size)
                for start, end in zip(cu_kv_padded[:-1], cu_kv_padded[1:])
            ]
        else:
            seqlens = None
            chunk_lens = [q.shape[seq_dim] // 2]
        chunks, ext_total = get_local_chunks(rank, cp_size, window, chunk_lens)
        doc_starts = [2 * sum(chunk_lens[:doc]) for doc in range(len(chunk_lens))]

        # K/V for each half's attention call: per sequence [halo, chunk] along the sequence axis
        k_ext, v_ext = (
            [
                torch.empty(_seq_shape(x.shape, seq_dim, total), dtype=x.dtype, device=x.device)
                for total in ext_total
            ]
            for x in [k, v]
        )
        for c in chunks:
            for x, ext in [(k, k_ext), (v, v_ext)]:
                ext[c.half].narrow(seq_dim, c.ext_start + c.halo, c.length).copy_(
                    x.narrow(seq_dim, c.start, c.length)
                )

        # halo pieces: copied when this rank owns both ends, otherwise exchanged
        transfers = [({}, {}), ({}, {})]
        needs_comm = [False, False]
        for doc, length in enumerate(chunk_lens):
            for piece in get_halo_pieces(cp_size, length, window):
                src_rank = get_chunk_owner(piece.src_chunk, cp_size)
                dst_rank = get_chunk_owner(piece.dst_chunk, cp_size)
                if rank not in [src_rank, dst_rank]:
                    continue
                dst_chunk = chunks[2 * doc + int(piece.dst_chunk >= cp_size)]
                for x, ext, (sends, recvs) in zip([k, v], [k_ext, v_ext], transfers):
                    if src_rank == rank == dst_rank:
                        _dst_slice(ext, seq_dim, dst_chunk, piece).copy_(
                            _src_slice(x, seq_dim, doc_starts[doc], length, piece, cp_size)
                        )
                    elif src_rank == rank:
                        sends.setdefault(dst_rank, []).append(
                            _src_slice(x, seq_dim, doc_starts[doc], length, piece, cp_size)
                        )
                    else:
                        recvs.setdefault(src_rank, []).append(
                            _dst_slice(ext, seq_dim, dst_chunk, piece)
                        )
                        needs_comm[dst_chunk.half] = True
        reqs, staged = _exchange(transfers, seq_dim, cp_group, cp_global_ranks)

        # attention per half, halves that need no communication first; packed outputs are
        # assembled from the valid tokens of each half, so they start from zeros
        out_shape = (*q.shape[:-1], v.shape[-1])
        if packed:
            out = torch.zeros(out_shape, dtype=q.dtype, device=q.device)
        else:
            out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
        cu_seqlens_per_half, max_seqlens, softmax_lse, rng_state = (
            [],
            [],
            [None, None],
            [None, None],
        )
        for half in range(2):
            if packed:
                cu_seqlens_per_half.append(
                    _packed_seqlens(chunks, ext_total, seqlens, q.shape[0], half, q.device)
                )
                max_seqlens.append(
                    (max(chunk_lens), max(c.halo + c.length for c in chunks[half::2]))
                )
            else:
                chunk = chunks[half]
                batch_size = q.shape[qkv_format.index("b")]
                cu_seqlens_per_half.append(
                    [
                        dpa_utils.get_full_cu_seqlens(batch_size, chunk.length, q.device),
                        None,
                        dpa_utils.get_full_cu_seqlens(
                            batch_size, chunk.halo + chunk.length, q.device
                        ),
                        None,
                    ]
                )
                max_seqlens.append((chunk.length, chunk.halo + chunk.length))
        max_logit = None
        waited = False
        for half in sorted(range(2), key=lambda half: needs_comm[half]):
            if needs_comm[half] and not waited:
                for req in reqs:
                    req.wait()
                _unpack(staged, seq_dim, accumulate=False)
                waited = True
            cu_q, q_offsets, cu_kv, kv_offsets = cu_seqlens_per_half[half]
            chunk = chunks[half]  # dense inputs have one chunk per half
            q_half = q if packed else q.narrow(seq_dim, chunk.start, chunk.length).contiguous()
            out_half, aux_ctx_tensors, *max_logit_half = fused_attn_fwd(
                is_training,
                *max_seqlens[half],
                cu_q,
                cu_kv,
                q_half,
                k_ext[half],
                v_ext[half],
                q.dtype,
                fused_attn_backend,
                attn_scale=softmax_scale,
                dropout=dropout_p,
                qkv_layout=qkv_layout,
                o_format=qkv_format,
                attn_mask_type=attn_mask_type,
                window_size=(window, 0),
                cu_seqlens_q_padded=q_offsets,
                cu_seqlens_kv_padded=kv_offsets,
                return_max_logit=return_max_logit,
                cuda_graph=is_graph_capturing(),
            )
            if packed:
                tex.thd_copy_valid_tokens_from_per_split_to_rank_local(
                    out, out_half, q_offsets, cu_q
                )
            else:
                out.narrow(seq_dim, chunk.start, chunk.length).copy_(out_half)
            softmax_lse[half], rng_state[half], *_ = aux_ctx_tensors
            if return_max_logit:
                max_logit = (
                    max_logit_half[0]
                    if max_logit is None
                    else torch.maximum(max_logit, max_logit_half[0])
                )
        if return_max_logit:
            torch.distributed.all_reduce(
                max_logit, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )

        ctx.save_for_backward(
            q,
            *k_ext,
            *v_ext,
            out,
            *softmax_lse,
            *rng_state,
            *cu_seqlens_per_half[0],
            *cu_seqlens_per_half[1],
        )
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.qkv_format = qkv_format
        ctx.qkv_layout = qkv_layout
        ctx.attn_mask_type = attn_mask_type
        ctx.seq_dim = seq_dim
        ctx.window = window
        ctx.chunk_lens = chunk_lens
        ctx.doc_starts = doc_starts
        ctx.chunks = chunks
        ctx.max_seqlens = max_seqlens
        ctx.seqlens = seqlens
        ctx.k_shape, ctx.v_shape = k.shape, v.shape
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic

        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2PSWA.forward")
        if return_max_logit:
            return out, max_logit
        return out

    @staticmethod
    def backward(ctx, dout, *_args):
        # pylint: disable=missing-function-docstring,too-many-locals
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVP2PSWA.backward")
        q, *saved = ctx.saved_tensors
        k_ext, v_ext, out = saved[0:2], saved[2:4], saved[4]
        softmax_lse, rng_state = saved[5:7], saved[7:9]
        cu_seqlens_per_half = [saved[9:13], saved[13:17]]

        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        packed = ctx.qkv_format == "thd"
        seq_dim, chunks = ctx.seq_dim, ctx.chunks
        dout = dout.contiguous()

        dq = torch.zeros_like(q) if packed else torch.empty_like(q)
        dk = torch.empty(ctx.k_shape, dtype=q.dtype, device=q.device)
        dv = torch.empty(ctx.v_shape, dtype=q.dtype, device=q.device)
        dk_ext, dv_ext = [None, None], [None, None]
        for half in range(2):
            cu_q, q_offsets, cu_kv, kv_offsets = cu_seqlens_per_half[half]
            chunk = chunks[half]  # dense inputs have one chunk per half
            if packed:
                q_half, out_half, dout_half = q, out, dout
            else:
                q_half, out_half, dout_half = [
                    x.narrow(seq_dim, chunk.start, chunk.length).contiguous()
                    for x in [q, out, dout]
                ]
            dq_half, dk_ext[half], dv_ext[half], *_ = fused_attn_bwd(
                *ctx.max_seqlens[half],
                cu_q,
                cu_kv,
                q_half,
                k_ext[half],
                v_ext[half],
                out_half,
                dout_half,
                q.dtype,
                [softmax_lse[half], rng_state[half]],
                FusedAttnBackend["F16_arbitrary_seqlen"],
                cu_seqlens_q_padded=q_offsets,
                cu_seqlens_kv_padded=kv_offsets,
                attn_scale=ctx.softmax_scale,
                dropout=ctx.dropout_p,
                qkv_layout=ctx.qkv_layout,
                o_format=ctx.qkv_format,
                do_format=ctx.qkv_format,
                dqkv_layout=ctx.qkv_layout,
                attn_mask_type=ctx.attn_mask_type,
                window_size=(ctx.window, 0),
                deterministic=ctx.deterministic,
                cuda_graph=is_graph_capturing(),
            )
            if packed:
                tex.thd_copy_valid_tokens_from_per_split_to_rank_local(dq, dq_half, q_offsets, cu_q)
            else:
                dq.narrow(seq_dim, chunk.start, chunk.length).copy_(dq_half)
            for chunk in chunks[half::2]:
                for grad, grad_ext in [(dk, dk_ext), (dv, dv_ext)]:
                    grad.narrow(seq_dim, chunk.start, chunk.length).copy_(
                        grad_ext[half].narrow(seq_dim, chunk.ext_start + chunk.halo, chunk.length)
                    )

        # return halo gradients to the owners of the pieces and accumulate them there; the
        # piece order is the same on every run, which keeps the accumulation deterministic
        transfers = [({}, {}), ({}, {})]
        for doc, length in enumerate(ctx.chunk_lens):
            for piece in get_halo_pieces(cp_size, length, ctx.window):
                src_rank = get_chunk_owner(piece.src_chunk, cp_size)
                dst_rank = get_chunk_owner(piece.dst_chunk, cp_size)
                if rank not in [src_rank, dst_rank]:
                    continue
                dst_chunk = chunks[2 * doc + int(piece.dst_chunk >= cp_size)]
                for grad, grad_ext, (sends, recvs) in zip([dk, dv], [dk_ext, dv_ext], transfers):
                    if src_rank == rank == dst_rank:
                        _src_slice(grad, seq_dim, ctx.doc_starts[doc], length, piece, cp_size).add_(
                            _dst_slice(grad_ext, seq_dim, dst_chunk, piece)
                        )
                    elif dst_rank == rank:
                        sends.setdefault(src_rank, []).append(
                            _dst_slice(grad_ext, seq_dim, dst_chunk, piece)
                        )
                    else:
                        recvs.setdefault(dst_rank, []).append(
                            _src_slice(grad, seq_dim, ctx.doc_starts[doc], length, piece, cp_size)
                        )
        reqs, staged = _exchange(transfers, seq_dim, ctx.cp_group, ctx.cp_global_ranks)
        for req in reqs:
            req.wait()
        _unpack(staged, seq_dim, accumulate=True)

        if packed:
            # zero the padding of every sequence; on this rank the valid tokens of a sequence
            # are a prefix of its two chunks because padding sits at the end of the sequence
            valid = [0]
            for doc, length in enumerate(ctx.chunk_lens):
                low, high = chunks[2 * doc], chunks[2 * doc + 1]
                actual = ctx.seqlens[doc]
                valid.append(
                    valid[-1]
                    + min(max(actual - low.chunk_id * length, 0), length)
                    + min(max(actual - high.chunk_id * length, 0), length)
                )
            padding = dpa_utils.get_thd_padding_mask(
                q.shape[0],
                torch.tensor(valid, dtype=torch.int32, device=q.device),
                torch.tensor(ctx.doc_starts + [q.shape[0]], dtype=torch.int32, device=q.device),
            )
            for grad in [dq, dk, dv]:
                grad[padding] = 0

        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2PSWA.backward")
        return (None, dq, dk, dv) + (None,) * 12
