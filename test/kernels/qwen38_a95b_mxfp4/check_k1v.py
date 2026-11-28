"""Check K1v (``gdn_pre`` with ``spec`` = T: the GDN layer of a NEXTN verify step
on SGLang's compact ReplaySSM ring) on one gfx950 GPU, on random weights in
SGLang's layouts:

- compile: every K1v key the spec makes (T = 4 at 4 and 8 rows, first layer,
  bf16 state, the in_proj_qkvz formats);
- chain: K1v's core and res_out rows and its conv window after each draft token
  equal T decode launches (one token of each request a launch) on an fp32 state,
  bit for bit, in the dense and in SGLang's deduplicated window layout; K1v
  writes no state and no conv state;
- bf16 state: K1v on a bf16 state equals K1v on that state widened to fp32, bit
  for bit (K1v rounds no state);
- records: SGLang's fold of the first a records (d, the normalized key and g,
  each high plus low part, in fp64) gives the decode state after a tokens, a = 1
  .. T, from a ring start that wraps; past L - write_pos K1v writes no record;
- pad request: ring row -1 and slot -1 give zero core rows and no record or conv
  window;
- graph replay: the K1v widths in one graph, replayed, bit for bit against eager.

``--f64``: K1v and SGLang's verify kernel (``fused_sigmoid_gating_delta_rule_update``
on the same q, k, v, a and b, SGLang's gated norm) against a float64 GDN layer
(``wide_reference.gdn``), for each draft token; prints only.
"""

import argparse
import sys
import time

import torch
import wide_reference as R
from inputs import BF, CORE, FP4_FORMATS, HIDDEN, NV, epoch, errors, gen, k1_state
from inputs import HD, k1_weights, rnd

from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L
from sgl_amd_mono.qwen38_a95b_mxfp4.spec import _compile_only

DEV = torch.device("cuda", 0)
T, RING, RSLOTS, SLOTS = 4, 16, 8, 8
NK, CONV, HPK = L.NK, L.CONV, NV // L.NK
SENTINEL = 7.0  # what a ring or window entry holds before the launch


def ring(r, dedup=False, wpos=0, cbase=(14, 3)):
    """The ring of one layer for r requests (ring rows 5, 2 ..), every record
    the sentinel; the intermediate conv window dense or deduplicated (SGLang's
    ``_allocate_deduplicated_conv_window``: windows overlap along the token axis)."""
    if dedup:
        phys = torch.full((r + 1, CONV, T + 2), SENTINEL, dtype=BF, device=DEV)
        icw = phys.as_strided(
            (r + 1, T, CONV, 3), (phys.stride(0), 1, phys.stride(1), 1)
        )
    else:
        icw = torch.full((r + 1, T, CONV, 3), SENTINEL, dtype=BF, device=DEV)
    full = dict(dtype=BF, device=DEV)
    rows = torch.tensor([5, 2][:r], dtype=torch.int32, device=DEV)
    wp = torch.zeros(RSLOTS, dtype=torch.int32, device=DEV)
    cb = torch.zeros(RSLOTS, dtype=torch.int32, device=DEV)
    wp[rows.long()] = wpos
    cb[rows.long()] = torch.tensor(cbase[:r], dtype=torch.int32, device=DEV)
    return gdn.Ring(
        ring_indices=rows,
        write_pos=wp,
        cache_base=cb,
        d=torch.full((RSLOTS, NV, RING, HD), SENTINEL, **full),
        rawv=torch.full((RSLOTS, NV, RING, HD), SENTINEL, **full),
        k=torch.full((RSLOTS, NK, RING, HD), SENTINEL, **full),
        rawk=torch.full((RSLOTS, NK, RING, HD), SENTINEL, **full),
        g=torch.full((RSLOTS, NV, RING), SENTINEL, dtype=torch.float32, device=DEV),
        conv_window=icw,
        conv_window_indices=torch.arange(r, dtype=torch.int32, device=DEV),
    )


def launch(key, w, hidden, residual, conv, state, idx, ep, rg=None, scr=None):
    s = key.tokens
    core = torch.full((s, CORE), float("nan"), dtype=BF, device=DEV)
    res_out = torch.empty(s, HIDDEN, dtype=BF, device=DEV)
    if scr is None:
        scr = torch.zeros(gdn.scratch_bytes(key), dtype=torch.uint8, device=DEV)
    ep[:1].add_(1)
    gdn.gdn_pre(
        key,
        **w,
        hidden=hidden,
        residual=residual,
        res_out=res_out,
        conv_state=conv,
        rstate=state,
        st_idx=idx,
        core=core,
        scratch=scr,
        epoch=ep,
        layer=0,
        ring=rg,
    )
    return core, res_out


def keys_of(r, first=False, bf16=False, qq=False, qh=False):
    common = dict(first=first, state_bf16=bf16, qq=qq, qh=qh)
    return (
        gdn.K1Build(tokens=r * T, spec=T, ring_len=RING, **common),
        gdn.K1Build(tokens=r, **common),
    )


def check_compile():
    keys = [
        gdn.K1Build(tokens=s, spec=T, ring_len=RING, first=f, state_bf16=b, qq=q, qh=h)
        for s in (T, 2 * T)
        for f in (False, True)
        for b in (False, True)
        for q, h in FP4_FORMATS
    ]
    t0 = time.time()
    for key in keys:
        _compile_only(gdn.build(key))
    torch.cuda.synchronize()
    print(f"compile: {len(keys)} K1v keys in {time.time() - t0:.0f} s", flush=True)
    return []


def inputs(seed, r, first, qq):
    g = gen(DEV, seed)
    w = k1_weights(g, qq=qq)
    conv, state = k1_state(g, SLOTS)
    hidden = rnd(g, r * T, HIDDEN, scale=2.0)
    residual = None if first else rnd(g, r * T, HIDDEN, scale=2.0)
    idx = torch.tensor([3, 6][:r], dtype=torch.int32, device=DEV)
    return w, conv, state, hidden, residual, idx


def decode_chain(key, w, hidden, residual, conv, state, idx, ep):
    """T decode launches, one token of each request a launch: per token the core
    and res_out rows, the conv state and the SSM state of the requests' slots."""
    r = key.tokens
    conv, state = conv.clone(), state.clone()
    out = []
    for s in range(T):
        rows = torch.arange(r, device=DEV) * T + s
        core, res_out = launch(
            key,
            w,
            hidden[rows].contiguous(),
            None if residual is None else residual[rows].contiguous(),
            conv,
            state,
            idx,
            ep,
        )
        out.append((core, res_out, conv[idx.long()].clone(), state[idx.long()].clone()))
    return out


def fold(rg, state0, slot, q, a):
    """SGLang's compact commit of request q's first a records into the checkpoint
    (``gdn_replayssm_compact_commit_kernel``), in fp64."""
    ri = int(rg.ring_indices[q])
    wp, cb = int(rg.write_pos[ri]), int(rg.cache_base[ri])
    h = state0[slot].double()
    heads = torch.arange(NV, device=DEV) // HPK
    for c in range(a):
        p = (cb + wp + c) % RING
        d = rg.d[ri, :, p].double() + rg.rawv[ri, :, p].double()
        kn = rg.k[ri, :, p].double() + rg.rawk[ri, :, p].double()
        g = rg.g[ri, :, p].double()
        h = g.exp()[:, None, None] * h + d[:, :, None] * kn[heads][:, None, :]
    return h


def check_chain():
    fails = []
    for r, first, (qq, qh), dedup in (
        (1, False, (False, False), False),
        (2, False, (True, False), True),
        (2, True, (True, True), False),
        (1, False, (True, True), True),
    ):
        tag = f"r={r} first={first} qq={qq} qh={qh} dedup={dedup}"
        kv, kd = keys_of(r, first=first, qq=qq, qh=qh)
        w, conv, state, hidden, residual, idx = inputs(10 + r, r, first, qq)
        ep = epoch(DEV)
        chain = decode_chain(kd, w, hidden, residual, conv, state, idx, ep)
        rg = ring(r, dedup=dedup)
        conv1, state1 = conv.clone(), state.clone()
        core, res_out = launch(kv, w, hidden, residual, conv1, state1, idx, ep, rg)
        torch.cuda.synchronize()
        bad = []
        for s, (dcore, dres, dconv, dstate) in enumerate(chain):
            rows = torch.arange(r, device=DEV) * T + s
            if not torch.equal(core[rows], dcore):
                bad.append(f"core token {s}")
            if not torch.equal(res_out[rows], dres):
                bad.append(f"res_out token {s}")
            win = rg.conv_window[rg.conv_window_indices.long(), s]
            if not torch.equal(win, dconv):
                bad.append(f"conv window token {s}")
            for q in range(r):
                want = dstate[q].double()
                got = fold(rg, state, int(idx[q]), q, s + 1)
                err = ((got - want).norm() / want.norm()).item()
                if not err < 1e-4:
                    bad.append(f"records request {q} after {s + 1}: rel err {err:.2e}")
        if not (torch.equal(conv1, conv) and torch.equal(state1, state)):
            bad.append("the conv or SSM state changed")
        if any(errors(ep).values()):
            bad.append(f"error words {errors(ep)}")
        print(f"chain {tag}: {bad or 'OK'}", flush=True)
        fails += [f"chain {tag}: {b}" for b in bad]
    return fails


def check_bf16_state():
    r = 2
    w, conv, state, hidden, residual, idx = inputs(21, r, False, True)
    state = state.to(BF)
    out = {}
    for bf in (False, True):
        kv, _ = keys_of(r, bf16=bf, qq=True)
        rg = ring(r)
        st = state.clone() if bf else state.float()
        core, res_out = launch(
            kv, w, hidden, residual, conv.clone(), st, idx, epoch(DEV), rg
        )
        out[bf] = [core, res_out, rg.d, rg.rawv, rg.k, rg.rawk, rg.g, rg.conv_window]
    torch.cuda.synchronize()
    bad = [
        i
        for i, (a, b) in enumerate(zip(out[False], out[True]))
        if not torch.equal(a, b)
    ]
    print(f"bf16 state: {bad or 'OK'}", flush=True)
    return (
        [f"bf16 state: output {bad} differs from the widened fp32 state"] if bad else []
    )


def check_mask():
    """write_pos L - 2: records for tokens 0 and 1 only; the rest keep the sentinel."""
    r = 1
    kv, _ = keys_of(r)
    w, conv, state, hidden, residual, idx = inputs(31, r, False, False)
    rg = ring(r, wpos=RING - 2, cbase=(5,))
    launch(kv, w, hidden, residual, conv, state, idx, epoch(DEV), rg)
    torch.cuda.synchronize()
    ri, bad = int(rg.ring_indices[0]), []
    for s in range(T):
        p = (5 + RING - 2 + s) % RING
        kept = [
            bool((t[ri, :, p] == SENTINEL).all())
            for t in (rg.d, rg.rawv, rg.k, rg.rawk, rg.g)
        ]
        if s < 2 and any(kept):
            bad.append(f"token {s}: a record not written")
        if s >= 2 and not all(kept):
            bad.append(f"token {s}: a record past the ring written")
    print(f"mask: {bad or 'OK'}", flush=True)
    return [f"mask: {b}" for b in bad]


def check_pad():
    r = 2
    kv, _ = keys_of(r)
    w, conv, state, hidden, residual, idx = inputs(41, r, False, False)
    rg = ring(r)
    idx[1] = -1
    rg.ring_indices[1] = -1
    core, _ = launch(kv, w, hidden, residual, conv, state, idx, epoch(DEV), rg)
    torch.cuda.synchronize()
    live = int(rg.ring_indices[0])
    others = [i for i in range(RSLOTS) if i != live]
    checks = {
        "live core finite": bool(torch.isfinite(core[:T]).all()),
        "pad core zero": bool((core[T:] == 0).all()),
        "no pad record": all(
            bool((t[others] == SENTINEL).all())
            for t in (rg.d, rg.rawv, rg.k, rg.rawk, rg.g)
        ),
        "no pad conv window": bool((rg.conv_window[1] == SENTINEL).all()),
    }
    bad = [k for k, v in checks.items() if not v]
    print(f"pad request: {bad or 'OK'}", flush=True)
    return [f"pad request: {b}" for b in bad]


def check_graph_replay(replays):
    ep = epoch(DEV)
    steps = []
    for r in (1, 2, 1):
        kv, _ = keys_of(r, qq=True)
        w, conv, state, hidden, residual, idx = inputs(50 + r, r, False, True)
        scr = torch.zeros(gdn.scratch_bytes(kv), dtype=torch.uint8, device=DEV)
        steps.append((kv, w, hidden, residual, conv, state, idx, ring(r), scr))
    outs = []

    def run_all():
        outs.clear()
        for kv, w, hidden, residual, conv, state, idx, rg, scr in steps:
            outs.extend(launch(kv, w, hidden, residual, conv, state, idx, ep, rg, scr))
            outs.extend([rg.d, rg.rawv, rg.k, rg.rawk, rg.g, rg.conv_window])

    run_all()
    torch.cuda.synchronize()
    eager = [t.clone() for t in outs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    bad = 0
    for i in range(replays):
        graph.replay()
        if i % 50 == 49 or i == replays - 1:
            torch.cuda.synchronize()
            bad += sum(not torch.equal(a, b) for a, b in zip(outs, eager))
    print(f"graph replay: {replays} replays: {bad} mismatching checks", flush=True)
    fails = [f"graph replay: {bad} mismatching checks"] if bad else []
    if any(errors(ep).values()):
        fails.append(f"graph replay: error words {errors(ep)}")
    return fails


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def check_f64(seeds):
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as sglang_verify,
    )

    key = L.NK * HD
    for r, qq, bf in ((1, False, False), (2, True, True), (1, True, True)):
        kv, _ = keys_of(r, first=True, bf16=bf, qq=qq)
        n = r * T
        stats = {}
        for seed in range(seeds):
            w, conv, state, hidden, _, idx = inputs(100 + seed, r, True, qq)
            if bf:
                state = state.to(BF)
            core, _ = launch(
                kv,
                w,
                hidden,
                None,
                conv.clone(),
                state.clone(),
                idx,
                epoch(DEV),
                ring(r),
            )
            x = R.gemma_norm(hidden.double(), w["ln_w"], 1e-6)
            ref, mixed, z, b, a = R.gdn(x, w, conv, state, idx, T)
            m = mixed.to(BF)
            got = {"k1v": core.double()}
            for rb in (False, True):
                o = sglang_verify(
                    A_log=w["a_log"],
                    a=a.to(BF),
                    dt_bias=w["dt_bias"],
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=m[:, :key].reshape(1, n, L.NK, HD),
                    k=m[:, key : 2 * key].reshape(1, n, L.NK, HD),
                    v=m[:, 2 * key :].reshape(1, n, NV, HD),
                    b=b.to(BF),
                    initial_state_source=state,
                    initial_state_indices=idx,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=torch.arange(0, n + 1, T, dtype=torch.int32, device=DEV),
                    disable_state_update=True,
                    round_beta_to_input_dtype=rb,
                )
                o = R.bf(o.reshape(n, NV, HD).double())
                got[f"sglang rb={int(rb)}"] = R.gated_norm(o, z, w["norm_w"])
            for name, c in got.items():
                for s in range(T):
                    stats.setdefault((name, s), []).append(_rel(c[s::T], ref[s::T]))
                stats.setdefault((name, "cos to k1v"), []).append(
                    torch.nn.functional.cosine_similarity(
                        c.flatten(), got["k1v"].flatten(), dim=0
                    ).item()
                )
        tag = f"r={r} qq={qq} bf16 state={bf}, {seeds} seeds"
        for name in got:
            per = " ".join(
                f"t{s} {max(stats[name, s]):.2e}/{sum(stats[name, s]) / seeds:.2e}"
                for s in range(T)
            )
            cos = min(stats[name, "cos to k1v"])
            print(
                f"f64 {tag}: {name}: rel to f64 (max/mean) {per}; "
                f"smallest cosine to k1v {cos:.7f}",
                flush=True,
            )
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--f64", type=int, default=0, help="seeds of check_f64 only")
    args = ap.parse_args()
    torch.cuda.set_device(DEV)
    if args.f64:
        check_f64(args.f64)
        return
    fails = check_compile()
    fails += check_chain()
    fails += check_bf16_state()
    fails += check_mask()
    fails += check_pad()
    fails += check_graph_replay(args.replays)
    for f in fails:
        print(f"FAIL {f}", flush=True)
    print("PASS" if not fails else f"FAIL ({len(fails)})", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
