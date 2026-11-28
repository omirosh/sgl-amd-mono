# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#67678 at ce476ce305 (Apache-2.0):
# tests/models/qwen3_5/amd_specialized_decode/reference.py
"""Float64 references of the wide launches' tail stages with the stock path's
bf16 rounding points (`bf`), at one TP8 rank's Qwen3.8-2.4T shapes. Tensors stay
on the device they come on."""

import torch
import torch.nn.functional as F

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = torch.tensor(_E2M1 + [-v for v in _E2M1])


def bf(t):
    return t.to(torch.bfloat16).double()


def mxfp4(n, k, gen, lo=118, hi=124, device="cpu"):
    """Random e2m1 codes and E8M0 scales: (packed [n, k / 2] uint8, scales
    [n, k / 32] uint8)."""
    codes = torch.randint(
        0, 16, (n, k), generator=gen, dtype=torch.uint8, device=device
    )
    sc = torch.randint(
        lo, hi, (n, k // 32), generator=gen, dtype=torch.uint8, device=device
    )
    return codes[..., 0::2] | (codes[..., 1::2] << 4), sc


def shuffle_data(packed):
    """AITER's `shuffle_weight(layout=(16, 16))` of fp4 codes [.., n, k / 2]."""
    *lead, n, kb = packed.shape
    v = packed.reshape(-1, n // 16, 16, kb // 16, 16).permute(0, 1, 3, 2, 4)
    return v.contiguous().view(*lead, n, kb)


def shuffle_scale(sc):
    """AITER's `e8m0_shuffle` of [rows, cols] scales (cols padded to 8)."""
    rows, cols = sc.shape
    pad = (cols + 7) // 8 * 8
    if pad != cols:
        fill = torch.full((rows, pad - cols), 127, dtype=torch.uint8, device=sc.device)
        sc = torch.cat([sc, fill], 1)
    return (
        sc.view(rows // 32, 2, 16, pad // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(rows, pad)
    )


def dequant_mx(q, s):
    """Dequantized float64 [.., k] of packed e2m1 codes `q` [.., k / 2] and
    E8M0 scales `s` [.., >= k / 32]."""
    q = q.long()
    lut = E2M1.to(q.device, torch.float64)
    vals = torch.stack([lut[q & 15], lut[q >> 4]], -1).flatten(-2)
    k = vals.shape[-1]
    scale = torch.pow(2.0, s[..., : k // 32].double() - 127)
    return vals * scale.repeat_interleave(32, -1)


def quant_act(x):
    """AITER's MXFP4 quantization of activation rows (`per_1x32_f4_quant_hip`,
    the ASM FP4 GEMM's and `fused_moe`'s) -> dequantized."""
    from aiter import per_1x32_f4_quant_hip

    xb = x.to(torch.bfloat16).contiguous()
    q, s = per_1x32_f4_quant_hip(xb, shuffle=False)
    return dequant_mx(q.view(torch.uint8), s.view(torch.uint8)[: xb.shape[0]])


def rotate32(x):
    """`block_hadamard_rotate` of bf16-valued `x`: f32 math, bf16 out, in
    the fast transform's order (`norm.hadamard32`) as the kernels sum it: a
    matmul's order rounds a rare bf16 ulp apart, which moves a whole 32-group's
    MXFP4 codes."""
    v = x.float().unflatten(-1, (-1, 32))
    h = 1
    while h < 32:
        v = v.unflatten(-1, (-1, 2, h))
        a, b = v[..., 0, :], v[..., 1, :]
        v = torch.stack([a + b, a - b], -2).flatten(-3)
        h *= 2
    return bf((v * torch.tensor(1.0 / 32**0.5, dtype=torch.float32)).flatten(-2))


def linear(x, w, fmt: str):
    """`x @ W.T` with the stock rounding: `w` a bf16 weight, or for "mxfp4" /
    "mxfp4_h32" the dequantized float64 weight (of the rotated weight for h32)
    with the input quantized (after its rotation) as the ASM FP4 GEMM does.
    Returns bf16-rounded float64."""
    if fmt == "bf16":
        return bf(x.double() @ w.double().T)
    if fmt == "mxfp4_h32":
        x = rotate32(x)
    return bf(quant_act(x) @ w.T)


def gemma_norm(z, w, eps):
    """`GemmaRMSNorm` of rows `z`: (1 + w) weights. `z` is the residual sum
    before its bf16 rounding, as SGLang's `fused_add_rms_norm` reads it."""
    rr = torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
    return bf(z * rr * (1 + w.double()))


def route(x, w_gate, experts: int, topk: int, prefer=None, margin: float = 1e-2):
    """The fused router's logits (bf16) -> (expert ids [rows, topk + 1], f64
    weights): softmax top-k of the routed experts, then the shared expert
    (index `experts`) at sigmoid of its logit. `prefer`: a kernel's
    picks [rows, topk], taken for a row when they are a top-k within
    `margin` (relative to the row's largest logit) of the boundary, as
    inputs a few bf16 ulps apart flip near-tied experts."""
    rows = x.shape[0]
    logits = bf(x.double() @ w_gate.double().T)
    routed = logits[:, :experts]
    top_v, top_i = torch.topk(routed, topk, dim=-1)
    if prefer is not None:
        prefer = prefer.long().to(x.device)
        tol = margin * routed.abs().amax(-1)
        picked = routed.gather(1, prefer)
        ok = (
            (prefer < experts).all(-1)
            & (prefer.sort(-1).values.diff(dim=-1) > 0).all(-1)
            & (picked.amin(-1) >= top_v[:, -1] - tol)
        )
        top_i = torch.where(ok[:, None], prefer, top_i)
        top_v = routed.gather(1, top_i)
    shared = torch.full((rows, 1), experts, device=x.device)
    ids = torch.cat([top_i, shared], 1)
    ws = torch.cat(
        [torch.softmax(top_v, -1), torch.sigmoid(logits[:, experts])[:, None]], 1
    )
    return ids, ws, logits


def moe(
    x,
    w_gate,
    w13,
    w13s,
    w2,
    w2s,
    experts,
    topk,
    ri,
    rip,
    prefer=None,
    xd=None,
    shared=None,
):
    """The MoE of rows `x` (one rank's intermediate slice; unshuffled MXFP4
    expert codes and scales), activations quantized to MXFP4 as AITER does
    (the input and the gate-up product). Returns (y f64, ids, weights);
    `prefer`: as `route`; `xd`: the experts' dequantized input rows
    (default `quant_act(x)`). `shared`: SGLang's bf16 shared expert
    (gate_up_proj, down_proj) of this rank, else the shared expert is expert
    `experts`: y is then the rank's bf16 partial `bf16(bf16(routed) +
    bf16(bf16(down) bf16(sigmoid)))`."""
    ids, ws, _ = route(x, w_gate, experts, topk, prefer)
    if xd is None:
        xd = quant_act(x)
    y = torch.zeros_like(xd)
    for e in ids.unique().tolist():
        if shared is not None and e == experts:
            continue
        w13d = dequant_mx(w13[e], w13s[e])
        w2d = dequant_mx(w2[e], w2s[e])[:, :ri]
        for b, j in (ids == e).nonzero().tolist():
            if shared is not None and j == topk:
                continue
            gu = w13d @ xd[b]
            act = F.silu(gu[:ri]) * gu[rip : rip + ri]
            y[b] += ws[b, j] * (w2d @ quant_act(act[None].float())[0])
    if shared is not None:
        sgu, sd = shared
        gu = bf(x.double() @ sgu.double().T)
        h = bf(F.silu(gu[:, :ri]) * gu[:, ri:])
        ws[:, topk] = bf(ws[:, topk])
        y = bf(bf(y) + bf(bf(h @ sd.double().T) * ws[:, topk : topk + 1]))
    return y, ids, ws


def unshuffle_data(packed):
    """The inverse of `shuffle_data`."""
    *lead, n, kb = packed.shape
    v = packed.reshape(-1, n // 16, kb // 16, 16, 16).permute(0, 1, 3, 2, 4)
    return v.contiguous().view(*lead, n, kb)


def unshuffle_scale(sc):
    """The inverse of `shuffle_scale` (the padded columns stay)."""
    rows, pad = sc.shape
    return (
        sc.view(rows // 32, pad // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .contiguous()
        .view(rows, pad)
    )


def gdn(x, w, conv, state, idx, T=1, eps=1e-6):
    """The GDN layer of SGLang's decode (T = 1) or target verify (T draft tokens
    a request: rows r T .. r T + T - 1, from the request's state) from the normed
    input rows `x` to the gated core, in float64 at the stock path's bf16 points
    (the projections, each conv product as Triton's bf16 multiply rounds it, the
    conv output, o, the core). `w`: `inputs.k1_weights`
    (`w_qkvzs` there means an MXFP4 in_proj_qkvz in the ASM FP4 GEMM layout);
    `conv` [slots, 2560, 3] and `state` [slots, NV, V, K] are read only; `idx`:
    one state slot a request, <= 0 a pad request (zero core rows).

    Returns the core and, for SGLang's kernel on the same inputs, the conv
    output (q | k | v), z, b and a (bf16-valued float64)."""
    hd, nv = state.shape[-1], state.shape[1]
    key = (conv.shape[1] - nv * hd) // 2
    nk = key // hd
    wq = w["w_qkvz"]
    if "w_qkvzs" in w:
        n = wq.shape[0]
        s = w["w_qkvzs"].view(torch.uint8).view(n, -1)
        wq = dequant_mx(unshuffle_data(wq.view(torch.uint8)), unshuffle_scale(s))
        proj = linear(x, wq, "mxfp4")
    else:
        proj = linear(x, wq, "bf16")
    ba = linear(x, w["w_ba"], "bf16")
    b, a = ba[:, :nv], ba[:, nv:]
    cw = w["conv_w"].double().reshape(conv.shape[1], -1)
    ch = conv.shape[1]
    z = proj[:, ch : ch + nv * hd].reshape(-1, nv, hd)
    sp = a + w["dt_bias"].double()
    sp = torch.where(sp <= 20.0, torch.log1p(torch.exp(sp)), sp)
    decay = torch.exp(-torch.exp(w["a_log"].double()) * sp)
    beta = torch.sigmoid(b)
    heads = torch.arange(nv, device=x.device) // (nv // nk)
    mixed = torch.zeros_like(proj[:, :ch])
    o = torch.zeros(x.shape[0], nv, hd, dtype=torch.float64, device=x.device)
    for r, slot in enumerate(idx.tolist()):
        if slot <= 0:
            continue
        win = conv[slot].double()
        h = state[slot].double()
        for t in range(r * T, r * T + T):
            win = torch.cat([win, proj[t, :ch, None]], 1)
            mixed[t] = bf(F.silu(bf(win * cw).sum(1)))
            win = win[:, 1:]
            q = mixed[t, :key].view(nk, hd)
            k = mixed[t, key : 2 * key].view(nk, hd)
            q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * hd**-0.5
            k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
            q, k = q[heads], k[heads]
            v = mixed[t, 2 * key :].view(nv, hd)
            h = h * decay[t, :, None, None]
            d = (v - (h @ k[..., None])[..., 0]) * beta[t, :, None]
            h = h + d[..., None] * k[:, None, :]
            o[t] = bf((h @ q[..., None])[..., 0])
    core = gated_norm(o, z, w["norm_w"], eps)
    core[~(idx > 0).repeat_interleave(T)] = 0
    return core, mixed, z, b, a


def gated_norm(o, z, norm_w, eps=1e-6):
    """SGLang's RMSNormGated of a core `o` [rows, NV, HD] (bf16-valued) with the
    gate `z` (the z part of in_proj_qkvz) -> the bf16-valued core."""
    rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    return bf(o * rstd * norm_w.double() * F.silu(z)).flatten(-2)
