# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): next to each launch, SGLang's own layer
code runs on the launch's input, and the launch's outputs are checked against it.

- K2: SGLang's o_proj, TP reduce, post-attention norm and MoE block with its TP
  reduce, on K2's own input: the layer output and the residual. The MoE block
  runs one row at a time: in a batch of 8, SGLang's router picks other experts
  for some rows than for the same row alone.

Each value is checked against its ``spec.NUMERICS`` bound on TP rank 0. Every
check is logged; a miss is logged as a violation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/models/qwen3_5.py": ("post_attention_layernorm", "mlp"),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
}


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    # equal also covers the graph warmup's zero tensors, where the cosine is 0
    return {
        "cosine": 1.0 if torch.equal(a, b) else F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def k2_reference(layer, o_proj, core, residual, forward_batch) -> dict:
    """SGLang's o_proj, post-attention norm and MoE block on K2's input. Both
    projections leave their TP sum to the layer's stage boundary: reduce here."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    a, _ = o_proj(core)
    a = tensor_model_parallel_all_reduce(a)
    x, res = layer.post_attention_layernorm(a, residual.clone())
    # the top-k the MoE block hands its experts, a row an entry (log_routes)
    routes = []
    hook = layer.mlp.experts.register_forward_pre_hook(
        lambda _, args: routes.append(args[1] if len(args) > 1 else None)
    )
    try:
        out = torch.cat(
            [
                tensor_model_parallel_all_reduce(layer.mlp(x[i : i + 1], forward_batch))
                for i in range(x.shape[0])
            ]
        )
    finally:
        hook.remove()
    return {"out": out.clone(), "res_out": res.clone(), "x": x, "routes": routes}


def compare(kind: str, layer_id: int, rows: int, ref: dict, got: dict) -> bool:
    """Log the values; False if one misses its bound (on TP rank 0 only)."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen38_a95b_mxfp4.spec import NUMERICS

    if get_parallel().tp_rank != 0:
        return True
    ok_all = True
    values = {
        f"{kind}.{name}.{metric}": v
        for name in ref
        for metric, v in _metrics(got[name], ref[name]).items()
    }
    logger.info(
        "monokernel check layer %d (%s, %d rows): %s",
        layer_id,
        kind,
        rows,
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            ok_all = False
            logger.error(
                "monokernel check violation: layer %d %s = %.6g, bound %g",
                layer_id,
                name,
                v,
                bound,
            )
    return ok_all


def log_routes(layer, layer_id: int, x, routes, k2_ids, k2_w, every_row: bool) -> None:
    """Each row's top-k by K2 (ROUTE) and by SGLang's MoE block, on TP rank 0. A row
    whose experts differ (every row if ``every_row``) is logged with the experts only
    one side picked and the bf16 router logits at ranks k - 2 .. k + 2 (torch fp32
    in brackets): a near-tie at rank k swaps the last expert."""
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_rank != 0:
        return
    k = k2_ids.shape[1]
    lg = x.float() @ layer.mlp.gate.weight.float().t()
    top = lg.bfloat16().float().topk(k + 2, dim=-1)
    for i, r in enumerate(routes):
        if getattr(r, "topk_ids", None) is None:
            continue
        ref_ids, got_ids = r.topk_ids[0, :k].tolist(), k2_ids[i].tolist()
        if set(ref_ids) == set(got_ids) and not every_row:
            continue
        cut = " ".join(
            f"#{j + 1} {e}:{top.values[i, j].item():.6g}({lg[i, e].item():.8g})"
            for j, e in enumerate(top.indices[i].tolist())
            if j >= k - 3
        )
        logger.info(
            "monokernel check route layer %d row %d: k2 only %s, sglang only %s; "
            "k2 %s w %s; sglang %s w %s; logits %s",
            layer_id,
            i,
            sorted(set(got_ids) - set(ref_ids)),
            sorted(set(ref_ids) - set(got_ids)),
            got_ids,
            [round(w, 5) for w in k2_w[i].tolist()],
            ref_ids,
            [round(w, 5) for w in r.topk_weights[0, :k].tolist()],
            cut,
        )
