# SPDX-License-Identifier: Apache-2.0
"""Diagnosis only: the K1v checks that a check-mode server dumped
(``check.dump_k1v``). For each dump, SGLang's verify core and K1v's core against
the float64 GDN layer (``wide_reference.gdn``) for each draft token, once on the
norm of the unrounded residual sum and once on SGLang's normed rows, and the
MXFP4 input codes that differ between the two norms."""

import glob

import torch
import wide_reference as R

DUMP = "/root/.cache/huggingface/hub/k1v_dump"


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    files = sorted(glob.glob(f"{DUMP}/*.pt"))
    print(f"k1v dump: {len(files)} files in {DUMP}", flush=True)
    for f in files:
        d = torch.load(f, map_location="cuda")
        idx = d["idx"].cuda()
        reqs = idx.numel()
        T = d["rows"] // reqs
        live = idx > 0
        rows = live.repeat_interleave(T)
        loc = torch.zeros_like(idx)
        loc[live] = torch.arange(1, int(live.sum()) + 1, device=idx.device).to(
            idx.dtype
        )
        conv = torch.cat([torch.zeros_like(d["conv"][:1]), d["conv"]])
        state = torch.cat([torch.zeros_like(d["state"][:1]), d["state"]])
        w = {
            k: d[k] for k in ("w_qkvz", "w_ba", "conv_w", "a_log", "dt_bias", "norm_w")
        }
        if d["w_qkvzs"] is not None:
            w["w_qkvzs"] = d["w_qkvzs"]
        z = d["h"].double()
        if d["residual"] is not None:
            z = z + d["residual"].double()
        x_own = R.gemma_norm(z, d["ln_w"], 1e-6)
        x_sg = d["x_sglang"].double()
        moved = (R.quant_act(x_own) != R.quant_act(x_sg)).sum(-1)[rows]
        print(
            f"{f.rsplit('/', 1)[-1]}: layer {d['layer']}, {reqs} requests x {T} tokens, "
            f"k1v core cosine {d['cos']:.6f}; x: own norm vs SGLang's {_rel(x_own[rows], x_sg[rows]):.2e}, "
            f"{int((x_own != x_sg)[rows].sum())} bf16 values and {int(moved.sum())} MXFP4 codes differ "
            f"(largest in a row {int(moved.max())})",
            flush=True,
        )
        for name, x in (("own norm", x_own), ("SGLang's norm", x_sg)):
            ref = R.gdn(x, w, conv, state, loc, T, 1e-6)[0]
            for side, label in (("ref", "SGLang"), ("got", "K1v")):
                c = d[side]["core"].double()
                per = " ".join(
                    f"t{t} {_rel(c[t::T][live], ref[t::T][live]):.2e}" for t in range(T)
                )
                print(f"  f64 on {name}: {label} {per}", flush=True)
        a, b = d["got"]["core"].double(), d["ref"]["core"].double()
        e = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)
        worst = e.argsort(descending=True)[:4].tolist()
        print(
            f"  worst rows K1v vs SGLang (row, rel): {[(i, round(e[i].item(), 4)) for i in worst]}",
            flush=True,
        )


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
