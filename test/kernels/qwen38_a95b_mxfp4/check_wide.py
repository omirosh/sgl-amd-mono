# Adapted from vllm-project/vllm#67678 at ce476ce305 (Apache-2.0):
# tests/models/qwen3_5/amd_specialized_decode/test_layer.py
"""Check the wide launches (``wide.layer``: out_proj or o_proj, its TP reduce,
the post-attention norm, the router, the experts with the fused shared expert,
the MoE TP reduce and the next layer's input norm) on 8 gfx950 GPUs (torchrun),
on random TP8 shards in SGLang's layouts, against float64 references with the
stock rounding points (``wide_reference``):

- compile: every wide key the spec can make (``attn`` on the gated core,
  ``gdn_back`` with a bf16 or an MXFP4 out_proj) compiles on every rank, at 17
  and 32 rows;
- chain: one step of three launches (``gdn_back``, ``attn``, ``gdn_back``) at 17
  and 32 rows, with a bf16 and an MXFP4 out_proj; each launch's MoE input rows,
  routes, route weights, residual and normed rows against the reference fed the
  previous launch's outputs (relative Frobenius errors);
- fault: rank 3 stalls at its first wait (``stop=False``) in ``gdn_back`` at 32
  rows: rank 3 sets its deadline word and its flag reaches every rank.

The MoE is checked on the kernel's own MXFP4 input rows (``x4``): the
reference's input differs from the kernel's in a few bf16 ulps, which flip MXFP4
codes; routes are compared as top-k sets within a tie margin
(``wide_reference.route``).
"""

import datetime
import os
import sys
import time

import torch
import torch.distributed as dist
import wide_reference as R
from inputs import BF, epoch, errors, gen

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import layout as L
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import grid
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W
from sgl_amd_mono.qwen38_a95b_mxfp4.spec import _compile_only

WIDTHS = (17, 32)
FORMATS = ("bf16", "mxfp4")
STEP = ("gdn_back", "attn", "gdn_back")
EPS = 1e-6

rank = int(os.environ["RANK"])
assert int(os.environ["WORLD_SIZE"]) == L.TP, f"needs {L.TP} ranks"
torch.cuda.set_device(rank)
DEV = torch.device("cuda", rank)
dist.init_process_group("gloo", timeout=datetime.timedelta(hours=1))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


def all_ok(flag: bool) -> bool:
    t = torch.tensor([0 if flag else 1])
    dist.all_reduce(t)
    return int(t) == 0


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def allsum(part):
    """Every rank's bf16 partial, summed in rank order -> bf16."""
    parts = [torch.empty_like(part.cpu()) for _ in range(L.TP)]
    dist.all_gather(parts, part.cpu().contiguous())
    tot = parts[0].double()
    for p in parts[1:]:
        tot = tot + p.double()
    return R.bf(tot).to(DEV)


def key(kind, rows, fmt, **kw):
    """The spec's wide key (``spec._wide_keys``) of a layer kind."""
    if kind == "attn":
        return W.LayerBuild(kind="attn", tokens=rows, tp=L.TP, og=False, rip=L.RI, **kw)
    return W.LayerBuild(
        kind="gdn_back",
        tokens=rows,
        tp=L.TP,
        out=fmt,
        out_shuffled=fmt == "mxfp4",
        rip=L.RI,
        **kw,
    )


def compile_keys(keys):
    """Every rank compiles every key before any rank launches one: a rank still
    compiling would leave the others waiting in the kernel."""
    keys = list(keys)
    start = len(keys) * rank // L.TP  # each rank at its own share of the disk cache
    ok = True
    try:
        for k in keys[start:] + keys[:start]:
            _compile_only(W.build(k))
        torch.cuda.synchronize()
    except Exception as e:  # reported, then every rank stops
        print(f"[rank {rank}] compile: {type(e).__name__}: {e}", flush=True)
        ok = False
    return all_ok(ok)


def check_compile():
    keys = sorted(
        {key(kind, s, f) for kind in set(STEP) for s in WIDTHS for f in FORMATS},
        key=repr,
    )
    t0 = time.time()
    ok = compile_keys(keys)
    log(
        f"compile: {len(keys)} wide keys in {time.time() - t0:.0f} s: {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        raise SystemExit("a rank failed to compile")
    return []


class Side:
    """A peer buffer, a guard block, the barrier slots of one step's launches and
    a scratch of ``rows``, as the spec's ``_Wide``."""

    def __init__(self, rows, launches, seconds=30.0):
        self.peer = PeerBuffer(W.peer_buffer_bytes(), dist.group.WORLD, rank, L.TP, DEV)
        self.peer.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()  # no peer writes into a buffer before its owner zeroed it
        self.epoch = epoch(DEV, seconds)
        self.epoch[:1].add_(1)
        self.rows = rows
        self.layout = W.scratch_layout(rows, L.RI)
        self.scratch = torch.zeros(
            W.scratch_bytes(rows, L.RI), dtype=torch.uint8, device=DEV
        )
        self.sync = torch.zeros(grid.sync_words(), dtype=torch.int32, device=DEV)
        base = self.sync.data_ptr()
        slot = [base + 4 * grid.SLOT_WORDS * k for k in range(grid.SLOTS)]
        self.slots = [(slot[a], slot[b]) for a, b in grid.launch_slots(launches)]
        self.folds = base + 4 * grid.FOLD_BASE

    def view(self, name, cols, dtype=BF):
        off = self.layout[name][0]
        n = self.rows * cols * torch.tensor([], dtype=dtype).element_size()
        return self.scratch[off : off + n].view(dtype).view(self.rows, cols)

    def route(self):
        """(ids [rows, SLOTS], weights [rows, SLOTS]) of the last launch."""
        off, n = self.layout["route"][0], self.rows * L.SLOTS
        words = self.scratch[off : off + 8 * n]
        return words[: 4 * n].view(torch.int32).view(-1, L.SLOTS), words[4 * n :].view(
            torch.float32
        ).view(-1, L.SLOTS)

    def run(self, n, k, layer, core, w, n_post, n_next):
        """Launch ``n`` of the step: model layer ``layer`` from its out_proj /
        o_proj input ``core``."""
        args = dict.fromkeys(W.ABI.names, 0) | {
            "scratch": self.scratch,
            "core_out" if k.kind == "gdn_back" else "attn": core,
            "n_post": n_post,
            "n_next": n_next,
            "w_out": w["out"],
            "w_out_s": w["out_s"],
            "w_gate": w["w_gate"],
            "w13": w["w13"],
            "w13s": w["w13s"],
            "w2": w["w2"],
            "w2s": w["w2s"],
            "peers": self.peer.addresses,
            "rank": rank,
            "epoch": self.epoch,
            "layer": layer,
            "sync": self.slots[n][0],
            "sync_next": self.slots[n][1],
            "folds": self.folds,
        }
        W.run(k, args)

    def close(self):
        self.peer.close()


def weights(shared, mine, kind, fmt):
    """One layer's tail weights (kernel layouts) and their float64 references."""

    def rn(*shape, scale=1.0, g=mine):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(BF)

    cols = L.ATTN_CORE if kind == "attn" else L.VALUE_DIM
    w = {}
    if kind == "attn" or fmt == "bf16":
        w["out"] = w["out_r"] = rn(L.HIDDEN, cols, scale=0.02)
        w["out_s"] = 0
    else:
        codes, sc = R.mxfp4(L.HIDDEN, cols, mine, lo=119, hi=123, device=DEV)
        w["out"], w["out_s"] = R.shuffle_data(codes), R.shuffle_scale(sc)
        w["out_r"] = R.dequant_mx(codes, sc)
    e, rip = L.E_ALL, L.RI
    w13, w13s = R.mxfp4(e * 2 * rip, L.HIDDEN, mine, lo=119, hi=123, device=DEV)
    w2, w2s = R.mxfp4(e * L.HIDDEN, rip, mine, lo=119, hi=123, device=DEV)
    return w | {
        "w_gate": rn(e, L.HIDDEN, scale=0.02, g=shared),
        "w13": R.shuffle_data(w13).view(e, 2 * rip, L.HIDDEN // 2),
        "w13s": R.shuffle_scale(w13s).view(e, 2 * rip, -1),
        "w2": R.shuffle_data(w2).view(e, L.HIDDEN, rip // 2),
        "w2s": R.shuffle_scale(w2s).view(e, L.HIDDEN, -1),
        "r13": w13.view(e, 2 * rip, -1),
        "r13s": w13s.view(e, 2 * rip, -1),
        "r2": w2.view(e, L.HIDDEN, -1),
        "r2s": w2s.view(e, L.HIDDEN, -1),
        "fmt": "bf16" if kind == "attn" else fmt,
    }


def check_chain(rows, fmt):
    shared, mine = gen(DEV, 1234), gen(DEV, 1000 + rank)
    norms = [
        (torch.randn(L.HIDDEN, generator=shared, device=DEV) * 0.1).to(BF)
        for _ in range(2 * len(STEP))
    ]
    keys = [key(kind, rows, fmt) for kind in STEP]
    side = Side(rows, len(STEP))
    notes = []
    try:
        side.view("res", L.HIDDEN).copy_(
            (torch.randn(rows, L.HIDDEN, generator=shared, device=DEV)).to(BF)
        )
        for n, (kind, k) in enumerate(zip(STEP, keys)):
            w = weights(shared, mine, kind, fmt)
            cols = L.ATTN_CORE if kind == "attn" else L.VALUE_DIM
            core = (torch.randn(rows, cols, generator=mine, device=DEV)).to(BF)
            n_post, n_next = norms[2 * n], norms[2 * n + 1]
            res = side.view("res", L.HIDDEN).clone()
            side.run(n, k, n, core, w, n_post, n_next)
            torch.cuda.synchronize()
            words = errors(side.epoch)
            if any(words.values()):
                notes.append((f"L{n} guard error words {words}", 1.0, 0.0))
                break
            tag = f"L{n} {kind}"
            res1 = R.bf(
                res.double() + allsum(R.linear(core.double(), w["out_r"], w["fmt"]))
            )
            xn = R.gemma_norm(res1, n_post, EPS)
            kxd = R.dequant_mx(
                side.view("x4", L.HIDDEN // 2, torch.uint8),
                side.view("x4s", L.HIDDEN // 32, torch.int32),
            )
            notes.append((f"{tag} moe input (MXFP4)", rel(kxd, R.quant_act(xn)), 2e-2))
            kids, kws = side.route()
            y, ids, ws = R.moe(
                xn,
                w["w_gate"],
                w["r13"],
                w["r13s"],
                w["r2"],
                w["r2s"],
                L.EXPERTS,
                L.TOPK,
                L.RI,
                L.RI,
                prefer=kids[:, : L.TOPK],
                xd=kxd,
            )
            same = sum(
                set(kids[r].tolist()) == set(ids[r].tolist()) for r in range(rows)
            )
            notes.append((f"{tag} routes", 1 - same / rows, 0.0))
            notes.append((f"{tag} route weights", rel(kws, ws), 1e-2))
            res2 = R.bf(res1 + allsum(R.bf(y)))
            notes.append(
                (f"{tag} residual", rel(side.view("res", L.HIDDEN), res2), 1e-2)
            )
            x2 = R.gemma_norm(res2, n_next, EPS)
            notes.append((f"{tag} x", rel(side.view("x", L.HIDDEN), x2), 1e-2))
    finally:
        side.close()
    bad = [f"{name}: {v:.2e} > {tol:.0e}" for name, v, tol in notes if not v <= tol]
    for b in bad:
        print(f"[rank {rank}] chain {rows} rows {fmt}: {b}", flush=True)
    ok = all_ok(not bad)
    worst = max((v for name, v, _ in notes if "routes" not in name), default=0.0)
    log(
        f"chain ({rows} rows, {fmt} out_proj): rank 0 largest error {worst:.2e}: "
        f"{'OK' if ok else 'FAIL'}"
    )
    return [] if ok else [f"chain ({rows} rows, {fmt})"]


def check_fault(rows=32, fmt="mxfp4"):
    fault = dict(fault_site=1) if rank == 3 else {}
    k = key("gdn_back", rows, fmt, stop=False, **fault)
    if not compile_keys([k]):
        raise SystemExit("a rank failed to compile the fault key")
    shared, mine = gen(DEV, 77), gen(DEV, 78 + rank)
    w = weights(shared, mine, "gdn_back", fmt)
    core = (torch.randn(rows, L.VALUE_DIM, generator=mine, device=DEV)).to(BF)
    norm = torch.zeros(L.HIDDEN, dtype=BF, device=DEV)
    side = Side(rows, 2, seconds=0.5)
    try:
        t0 = time.time()
        side.run(0, k, 0, core, w, norm, norm)
        torch.cuda.synchronize()
        dt = time.time() - t0
        words = errors(side.epoch)
        flags = side.peer.bytes.view(torch.int32)[
            k2.FLAGS_OFF // 4 : k2.FLAGS_OFF // 4 + L.TP
        ].tolist()
    finally:
        side.close()
    bad = []
    if rank == 3 and words[guard.ERR_DEADLINE] != 2:
        bad.append(f"rank 3 error words {words}, want deadline word 2")
    if flags[3] != 1:
        bad.append(f"rank 3's flag not set here: {flags}")
    if bad:
        print(f"[rank {rank}] fault: {bad} in {dt:.2f} s", flush=True)
    ok = all_ok(not bad)
    log(
        f"fault (gdn_back at {rows} rows): rank 3 stalled, returned in {dt:.2f} s: "
        f"{'OK' if ok else 'FAIL'}"
    )
    return [] if ok else [f"fault (gdn_back at {rows} rows)"]


def main():
    fails = check_compile()
    for rows in WIDTHS:
        for fmt in FORMATS:
            dist.barrier()
            fails += check_chain(rows, fmt)
    dist.barrier()
    fails += check_fault()
    for f in fails:
        log(f"FAIL {f}")
    log("PASS" if not fails else f"FAIL ({len(fails)})")
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
