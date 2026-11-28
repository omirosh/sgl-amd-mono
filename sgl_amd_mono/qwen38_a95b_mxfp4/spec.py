# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 MoE on gfx950: the monokernel spec and its forward (RFC
sgl-project/sglang#43949).

A decode step runs each decoder layer as at most two launches:

- K1 (``gfx950/gdn.py``, GDN layers): input norm, in_proj_qkvz and in_proj_ba,
  the conv and the gated delta rule with their state updates, and the gated norm.
  It writes the core output (o_proj's input) and the residual.
- K2 (``gfx950/layer.py``, every layer): o_proj and its TP reduce, the
  post-attention norm, the router, the shared and routed MXFP4 experts and the MoE
  TP reduce. It writes the layer output and the residual.

A full-attention layer runs SGLang's attention up to o_proj's input, then K2. A
target-verify step (speculative decoding) runs K2 only, and a GDN layer runs
SGLang's GDN up to out_proj's input there, except with K1v: on SGLang's compact
ReplaySSM ring (``--enable-linear-replayssm-spec`` with a bf16 SSM state, the
draft chain of 2 to 4 tokens), a verify step of up to 8 rows runs each GDN
layer's front as K1 with ``spec`` = the draft tokens. It writes the ring records
and the conv windows that SGLang's verify kernels write and no state, and
SGLang's commit after the accept stays as it is (``gfx950/gdn.py``).
``SGLANG_AMD_MONO_K1V=0`` turns K1v off. SGLang's embedding, final norm and
logits processor run around the layers.

A step of 16 rows (decode or verify) runs SGLang's GDN or attention up to
out_proj's / o_proj's input, then ``layer_block`` (``gfx950/mono/blocks.py``,
from vllm-project/vllm#67421) in place of K2. It keeps no x tile in LDS, so it
takes the rows K2 cannot, with or without the fused shared expert. With the
fused shared expert, it also runs in place of K2 at the widths of
``LAYER_WIDTHS``. In a decode step of 16 rows, a GDN layer runs its front as one split launch (``gfx950/wide/layer.py``,
the ``gdn`` kind at 16 rows, from vllm-project/vllm#67421 at 28f094565b and
vllm-project/vllm#67678 at ce476ce305) in place of SGLang's GDN: input norm, in_proj_qkvz and in_proj_ba, the conv and the gated delta rule
with their state updates, and the gated norm, as K1 does.

A step of 32 rows (decode or verify) runs SGLang's GDN or attention up to
out_proj's / o_proj's input, then one wide launch a layer (``gfx950/wide``, from
vllm-project/vllm#67421 and #67678): out_proj / o_proj and its TP reduce, the
post-attention norm, the MoE and its TP reduce, and the next layer's input norm
(the last layer's: the final norm). The launches hand the residual and the normed
rows on in their scratch, so SGLang runs no input norm between them. In a decode
step, a GDN layer's launch also runs its front from the normed rows, as K1 does:
in_proj_qkvz and in_proj_ba, the conv and the gated delta rule with their state
updates, and the gated norm. With SGLang's bf16 shared expert, the launch's
router also makes the shared gate | up, and its least busy CTAs the shared
down projection (``wide.stages.moe``).

The kernels read SGLang's weights in place at each launch (SGLang repacks
in_proj_qkvz after the build). MXFP4 GDN projections (sgl-project/sglang#43839)
run K1 ``qq`` and K2 ``oq``. The fused shared expert
(``--enforce-shared-experts-fusion``, MXFP4 with sgl-project/sglang#41899) runs K2
``fse``: the shared expert is routed expert 512.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.qwen38_a95b_mxfp4 import HF_CONFIG
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import layout as ML

logger = logging.getLogger(__name__)

# the rows a launch takes (decode requests, or verify tokens): 1 to MAX_TOKENS
# (K1, K2), SPLIT_MAX (layer_block) or WIDE (the wide launches)
WIDE = 32
WIDTHS = (*range(1, L.MAX_TOKENS + 1), ML.SPLIT_MAX, WIDE)
# the widths up to MAX_TOKENS where layer_block (fused shared expert) runs in
# place of K2: at 1 and 4 rows its gate | up tasks (32 columns, K2's 16) leave
# its busiest CTA more work than K2's
LAYER_WIDTHS = (2, 3, 5, 6, 7, 8)
# the mailbox tags a launch writes: K1 at slot i, K2 at TAG_SLOTS // 2 + i
TAG_SLOTS = 256
MAX_LAYERS = TAG_SLOTS // 2

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "enable_linear_replayssm",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "enable_linear_replayssm_spec",
        "linear_replayssm_cache_len",
    ),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "forward_metadata",
        "state_layer_cache",
        "replayssm_ring_indices",
        "state_pool",
        "req_to_token_pool",
    ),
    "srt/layers/attention/linear/gdn_backend.py": (
        "verify_intermediate_state_indices",
    ),
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    # K1v reads no history: with a bf16 state the commit folds each record, then
    # the finish kernel sets write_pos to 0
    "srt/speculative/spec_utils.py": ("commit_gdn_replayssm_after_verify",),
    "kernels/ops/attention/linear/gdn_replayssm_spec_decode.py": (
        "commit_gdn_replayssm_after_verify(fold_every_commit)",
        "_finish_gdn_replayssm_circular_fold_kernel",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "eps",
        "group_size",
        "norm_before_gate",
        "activation",
    ),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "srt/mem_cache/memory_pool.py": (
        "conv",
        "temporal",
        "replayssm_d",
        "replayssm_rawv",
        "replayssm_k",
        "replayssm_rawk",
        "replayssm_g",
        "replayssm_beta",
        "replayssm_spec_write_pos",
        "replayssm_cache_base",
        "replayssm_spec_fold",
        "replayssm_is_kda",
        "intermediate_conv_window",
    ),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py": (
        "AiterMxfp4OnlineLinearMethod",
    ),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "is_shuffled",
        '"w13_weight"',
        '"w2_weight"',
        '"w13_weight_scale"',
        '"w2_weight_scale"',
    ),
    "srt/model_executor/forward_batch_info.py": ("mrope_positions",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "gate",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "num_fused_shared_experts",
        "gate_up_proj",
        "down_proj",
    ),
    "srt/models/qwen3_5.py": (
        "embed_tokens",
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "layers_to_capture",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "linear_attn",
        "in_proj_qkvz",
        "in_proj_ba",
        "conv1d",
        "A_log",
        "dt_bias",
        "out_proj",
        "activation",
        "layer_id",
        "self_attention",
        "o_proj",
        "q_size",
        "attn_output_gate",
    ),
    "srt/models/qwen3_5_text.py": (
        "model",
        "lm_head",
        "logits_processor",
        "is_mrope_enabled",
    ),
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/runtime_context.py": (
        "get_exec",
        "get_parallel",
        "get_spec",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
}

# K1 and K2 against SGLang's layer on the same input and state (check.py,
# SGLANG_AMD_MONO_CHECK). K2 quantizes the expert inputs to MXFP4 as SGLang's
# AITER MoE does but sums in another order, so the layer output differs by about
# the MoE's rounding error. The k2.out bound holds on the rows that K2 and
# SGLang route alike (check.k2_alike).
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "k1.core.cosine": 0.9999,
        "k1.conv.cosine": 0.9999,
        "k1.state.cosine": 0.9999,
        # K1v against SGLang's verify kernel: SGLang takes the checkpoint terms
        # as bf16 dot products, K1v in fp32
        "k1v.core.cosine": 0.9999,
        "k1v.conv.cosine": 0.9999,
        "k1v.d.cosine": 0.999,
        "k1v.k.cosine": 0.9999,
        "k1v.g.cosine": 0.9999,
        "k2.out.cosine": 0.99,
        "wide.out.cosine": 0.99,
        "wide.conv.cosine": 0.9999,
        "wide.state.cosine": 0.9999,
    },
)


def _text(config):
    return getattr(config, "text_config", config)


def _is_gdn(layer) -> bool:
    return hasattr(layer, "linear_attn")


def _mxfp4(linear) -> bool:
    """An MXFP4 projection: fp4x2 bytes and their e8m0 scales."""
    return (
        linear.weight.dtype == torch.uint8
        and getattr(linear, "weight_scale", None) is not None
    )


# The states of the quant method of an MXFP4 GDN projection and whether its weight
# is rotated: SGLang before sgl-project/sglang#46059, with
# SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0, and with its default, which rotates the
# weight and the activation by Hadamard-32 (the kernels rotate the activation as
# AITER does). An exact match, so that a renamed or a new flag gives a refusal.
_ONLINE = "AiterMxfp4OnlineLinearMethod"
_STATES = (
    ({}, False),
    ({"hadamard": False, "quant_kwargs": {}}, False),
    ({"hadamard": True, "quant_kwargs": {"hadamard_rotate": True}}, True),
)


def _rotated(linear) -> bool | None:
    """An MXFP4 projection's rotation; None for a quant method or a state the
    kernels do not know."""
    method = linear.quant_method
    if type(method).__name__ != _ONLINE:
        return None
    return next((rot for state, rot in _STATES if vars(method) == state), None)


def _format(linear) -> tuple[bool, bool]:
    """(MXFP4, rotated) of a projection that the spec accepts."""
    return (True, _rotated(linear)) if _mxfp4(linear) else (False, False)


def _fse(mlp) -> bool:
    return bool(getattr(mlp, "num_fused_shared_experts", 0))


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype

    text = _text(hf_config)
    reasons = [
        f"needs Qwen3.8's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(text, key, None) != want
    ]
    if not getattr(text, "norm_topk_prob", True):
        reasons.append("needs norm_topk_prob: the router renormalizes the top-k")
    layers = getattr(text, "num_hidden_layers", 0)
    if layers > MAX_LAYERS:
        reasons.append(f"{layers} layers: the mailbox tags hold {MAX_LAYERS}")
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.temporal not in (torch.float32, torch.bfloat16):
        reasons.append(f"needs the SSM state in f32 or bf16, got {dtypes.temporal}")
    if dtypes.conv != torch.bfloat16:
        reasons.append(f"needs the conv state in bf16, got {dtypes.conv}")
    if cfg.speculative_algorithm is None:
        if cfg.enable_linear_replayssm:
            reasons.append(
                "--enable-linear-replayssm: K1 updates the SSM state in place; "
                "ReplaySSM defers it to a ring"
            )
    else:
        topk = cfg.speculative_eagle_topk or 1
        tokens, steps = cfg.speculative_num_draft_tokens, cfg.speculative_num_steps
        if topk != 1 or tokens != (steps or 0) + 1:
            reasons.append(
                "the verify step takes one draft chain: needs "
                "--speculative-eagle-topk 1 and --speculative-num-draft-tokens = "
                f"--speculative-num-steps + 1, got topk {topk}, {tokens} tokens, "
                f"{steps} steps"
            )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _bf16(*tensors) -> bool:
    return all(t.dtype == torch.bfloat16 for t in tensors)


def _layer_reasons(layer, eps: tuple) -> list[str]:
    """Why the kernels cannot run one decoder layer as SGLang loaded it."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return [f"needs the sparse MoE block, got {type(mlp).__name__}"]
    reasons = []
    se = mlp.shared_expert
    if se is None and not _fse(mlp):
        reasons.append("needs the shared expert")
    elif se is not None and not _bf16(se.gate_up_proj.weight, se.down_proj.weight):
        reasons.append("needs the shared expert in bf16")
    ex = mlp.experts
    e = L.E + _fse(mlp)
    if not getattr(ex.w13_weight, "is_shuffled", False):
        reasons.append("needs the routed experts in AITER's shuffled layout")
    if ex.w13_weight.dtype != torch.uint8 or ex.w2_weight.dtype != torch.uint8:
        reasons.append("needs MXFP4 routed experts")
    if tuple(ex.w13_weight.shape) != (e, 2 * L.RI, L.HIDDEN // 2):
        reasons.append(
            f"routed w13 shape {tuple(ex.w13_weight.shape)}: the kernels take the "
            f"experts TP-sharded, unpadded"
        )
    if not _bf16(
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        mlp.gate.weight,
        mlp.shared_expert_gate.weight,
    ):
        reasons.append("needs the norms, router and shared-expert gate in bf16")
    if _is_gdn(layer):
        a = layer.linear_attn
        o = a.out_proj
        if not all(_mxfp4(p) or _bf16(p.weight) for p in (a.in_proj_qkvz, o)):
            reasons.append("needs in_proj_qkvz and out_proj in bf16 or MXFP4")
        elif any(_rotated(p) is None for p in (a.in_proj_qkvz, o) if _mxfp4(p)):
            reasons.append(
                f"needs the MXFP4 in_proj_qkvz and out_proj of {_ONLINE} in a known "
                "state (sgl-project/sglang#46059 at 7397386194)"
            )
        if not _bf16(a.in_proj_ba.weight, a.conv1d.weight, a.dt_bias, a.norm.weight):
            reasons.append("needs the GDN in_proj_ba, conv, dt_bias and norm in bf16")
        if a.A_log.dtype != torch.float32:
            reasons.append("needs A_log in f32")
        if a.activation != "silu" or a.conv1d.bias is not None:
            reasons.append("needs the GDN conv without bias and with silu")
        norm = a.norm
        if norm.activation not in ("silu", "swish") or norm.group_size is not None:
            reasons.append("needs the GDN gated norm over the head with silu")
        if not norm.norm_before_gate:
            reasons.append("needs the GDN norm before the gate")
        layer_eps = (layer.input_layernorm.variance_epsilon, norm.eps)
    else:
        o = layer.o_proj
        if not layer.attn_output_gate or layer.q_size != L.CORE:
            reasons.append("needs the gated attention output with one rank's heads")
        if not _bf16(o.weight):
            reasons.append("needs o_proj in bf16")
        layer_eps = (layer.input_layernorm.variance_epsilon, None)
    if not _mxfp4(o) and tuple(o.weight.shape) != (L.HIDDEN, L.CORE):
        reasons.append(f"o_proj shape {tuple(o.weight.shape)}")
    want = (eps[0], eps[1] if _is_gdn(layer) else None)
    if layer_eps != want or layer.post_attention_layernorm.variance_epsilon != eps[2]:
        reasons.append("its norm eps differ from the first layer's")
    return reasons


def _eps(layers) -> tuple:
    """(input norm, GDN gated norm, post-attention norm) eps: one for all layers."""
    gdn = next((x.linear_attn for x in layers if _is_gdn(x)), None)
    return (
        layers[0].input_layernorm.variance_epsilon,
        None if gdn is None else gdn.norm.eps,
        layers[0].post_attention_layernorm.variance_epsilon,
    )


def _layers(model) -> list:
    m = model.model
    return [m.layers[i] for i in range(m.start_layer, m.end_layer)]


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    layers = _layers(model)
    eps = _eps(layers)
    return [
        f"layer {layer.layer_id}: {why}"
        for layer in layers
        for why in _layer_reasons(layer, eps)
    ]


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x, None


_IDENTITY = _Identity()


@contextmanager
def _proj_input(module, name: str):
    """``module.<name>`` returns its input, so SGLang's attention or GDN returns
    o_proj's input. ponytail: a module swap around SGLang's call; the upgrade is a
    core / projection split in SGLang's Qwen3.5 layers."""
    proj = getattr(module, name)
    setattr(module, name, _IDENTITY)
    try:
        yield
    finally:
        setattr(module, name, proj)


def _wide_format(linear) -> str:
    """A GDN projection's weight format in the split and wide launches
    (``wide.layer.FORMATS``): "mxfp4_h32" rotates the activation by
    Hadamard-32, as AITER does for a weight that sgl-project/sglang#46059
    rotated."""
    mx, rotated = _format(linear)
    return "mxfp4_h32" if rotated else "mxfp4" if mx else "bf16"


def _builds_split(layers, widths, verify) -> bool:
    """The 16-row decode steps run split launches: ``layer_block`` of the
    layer's shared-expert layout runs their MLP half."""
    return ML.SPLIT_MAX in widths and not verify


def _split_keys(layers, eps, gdn_eps, state_bf16, tp, fault_site, fault_after):
    """Each layer's split launch at 16 rows: a GDN layer's ``gdn`` key, None for
    an attention layer. Each launch runs its own input norm, so a launch with an
    MXFP4 in_proj_qkvz writes the MXFP4 rows of it."""
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W

    keys = []
    for n, x in enumerate(layers):
        if not _is_gdn(x):
            keys.append(None)
            continue
        f = _wide_format(x.linear_attn.in_proj_qkvz)
        keys.append(
            W.LayerBuild(
                kind="gdn",
                tokens=ML.SPLIT_MAX,
                tp=tp,
                rip=L.RI,
                first=n == 0,
                first_q4=W.q4_of(f) or "none",
                qkvz=f,
                qkvz_shuffled=f != "bf16",
                eps=eps,
                gdn_eps=gdn_eps,
                state_bf16=state_bf16,
                fault_site=fault_site,
                fault_after=fault_after,
            )
        )
    return keys


class _Split:
    """The split launches of a 16-row decode step: one a GDN layer (``keys[n]``,
    layer ``n`` of the step), their scratch, barrier slots and core output.
    ``peers``: the peer buffers' address table; ``epoch``: the guard block."""

    def __init__(self, keys, peers, rank, epoch, device):
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import grid
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W

        s = ML.SPLIT_MAX
        self.keys, self.peers, self.rank, self.epoch = keys, peers, rank, epoch
        self.scratch = torch.zeros(
            W.scratch_bytes(s, L.RI), dtype=torch.uint8, device=device
        )
        off, n = W.scratch_layout(s, L.RI)["res"]
        assert n >= s * L.HIDDEN * 2, n
        # the residual a launch leaves for layer_block
        self.res = self.scratch[off : off + s * L.HIDDEN * 2]
        self.res = self.res.view(torch.bfloat16).view(s, L.HIDDEN)
        self.core = torch.empty(s, ML.VALUE_DIM, dtype=torch.bfloat16, device=device)
        # zeroed: each launch's barriers count up from its slot's zeros
        self.sync = torch.zeros(grid.sync_words(), dtype=torch.int32, device=device)
        base = self.sync.data_ptr()
        slot = [base + 4 * grid.SLOT_WORDS * k for k in range(grid.SLOTS)]
        gdns = [n for n, k in enumerate(keys) if k is not None]
        pairs = grid.launch_slots(len(gdns))
        self.slots = {n: (slot[a], slot[b]) for n, (a, b) in zip(gdns, pairs)}
        self.folds = base + 4 * grid.FOLD_BASE

    def launch(self, n, i, layer, h, residual, conv, temporal, idx):
        """Layer ``n`` of the step (model layer ``i``) from the previous layer's
        output ``h`` and ``residual`` (the first layer: the embedding, None):
        returns its core and residual (buffers the next launch rewrites)."""
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W

        args = dict.fromkeys(W.ABI.names, 0) | _gdn_front(layer, h, conv, temporal, idx)
        args |= {
            "scratch": self.scratch,
            "res_in": 0 if residual is None else residual,
            "core_out": self.core,
            "peers": self.peers,
            "rank": self.rank,
            "epoch": self.epoch,
            "layer": i,
            "sync": self.slots[n][0],
            "sync_next": self.slots[n][1],
            "folds": self.folds,
        }
        W.run(self.keys[n], args)
        return self.core, self.res


def _builds_wide(model, layers, widths) -> bool:
    """Whether a WIDE-row step runs the wide launches: one shared-expert layout
    (fused, or SGLang's bf16 shared expert) for every layer, and one eps for the
    input, post-attention and final norms (a launch's closing combine is the next
    layer's input norm, with the post-attention eps)."""
    e1, _, e2 = _eps(layers)
    return (
        WIDE in widths
        and len({_fse(x.mlp) for x in layers}) == 1
        and e1 == e2 == model.model.norm.variance_epsilon
    )


def _wide_keys(layers, eps, tp, fault_site, fault_after, front=None) -> list:
    """Each layer's wide launch at WIDE rows: ``attn`` from SGLang's gated core.
    A GDN layer runs ``gdn_back`` from SGLang's GDN core, or with ``front`` (the
    decode step: the gated norm eps, a bf16 SSM state) ``gdn`` from its input
    rows: the projections, the conv and the recurrence with their state updates.
    A launch before a ``gdn`` launch with an MXFP4 in_proj_qkvz quantizes the
    rows of its closing combine for it (rotated for a rotated weight)."""
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W

    def q4(x) -> str:
        """The MXFP4 rows that gdn launch ``x`` reads, "none" for bf16."""
        if front is None or not _is_gdn(x):
            return "none"
        return W.q4_of(_wide_format(x.linear_attn.in_proj_qkvz)) or "none"

    keys = []
    for n, x in enumerate(layers):
        common = dict(
            tokens=WIDE,
            tp=tp,
            rip=L.RI,
            fse=_fse(x.mlp),
            eps=eps,
            next_q4=q4(layers[n + 1]) if n + 1 < len(layers) else "none",
            fault_site=fault_site,
            fault_after=fault_after,
        )
        if not _is_gdn(x):
            keys.append(W.LayerBuild(kind="attn", og=False, **common))
            continue
        o = _wide_format(x.linear_attn.out_proj)
        out = dict(out=o, out_shuffled=o != "bf16")
        if front is None:
            keys.append(W.LayerBuild(kind="gdn_back", **out, **common))
            continue
        gdn_eps, state_bf16 = front
        f = _wide_format(x.linear_attn.in_proj_qkvz)
        keys.append(
            W.LayerBuild(
                kind="gdn",
                first=n == 0,
                first_q4=q4(x) if n == 0 else "none",
                qkvz=f,
                qkvz_shuffled=f != "bf16",
                gdn_eps=gdn_eps,
                state_bf16=state_bf16,
                **out,
                **common,
            )
        )
    return keys


def _router_gates(layers) -> list:
    """Each layer's router matrix (``wide.stages.moe.router_rows``): the router
    gate, without the fused shared expert the shared expert's gate_up_proj, then
    the shared expert's gate as its last row, [E + 1 (+ 2 RI), H]: the wide
    router reads one matrix. ponytail: SGLang's weights become views of it, so
    the build keeps no copy; the upgrade is a weight pointer for each in the wide
    router."""
    out = []
    for x in layers:
        ws = [x.mlp.gate.weight]
        if not _fse(x.mlp):
            ws.append(x.mlp.shared_expert.gate_up_proj.weight)
        ws.append(x.mlp.shared_expert_gate.weight)
        both = torch.cat([w.data for w in ws], 0)
        at = 0
        for w in ws:
            w.data, at = both[at : at + w.shape[0]], at + w.shape[0]
        out.append(both)
    return out


class _Wide:
    """The launches of a WIDE-row step: one wide launch a layer (``keys[n]``,
    layer ``n`` of the step), their scratch, barrier slots and router gates.
    ``peers``: the peer buffers' address table; ``epoch``: the guard block."""

    def __init__(self, keys, gates, peers, rank, epoch, device):
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import grid
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W

        self.keys, self.gates = keys, gates
        self.peers, self.rank, self.epoch = peers, rank, epoch
        self.layout = W.scratch_layout(WIDE, L.RI)
        self.scratch = torch.zeros(
            W.scratch_bytes(WIDE, L.RI), dtype=torch.uint8, device=device
        )
        # zeroed: each launch's barriers count up from its slot's zeros
        self.sync = torch.zeros(grid.sync_words(), dtype=torch.int32, device=device)
        base = self.sync.data_ptr()
        slot = [base + 4 * grid.SLOT_WORDS * k for k in range(grid.SLOTS)]
        self.slots = [(slot[a], slot[b]) for a, b in grid.launch_slots(len(keys))]
        self.folds = base + 4 * grid.FOLD_BASE

    def view(self, name: str, cols: int) -> torch.Tensor:
        """Scratch region ``name`` as bf16 [WIDE, cols]."""
        off, n = self.layout[name]
        assert n >= WIDE * cols * 2, (name, n, cols)
        return (
            self.scratch[off : off + WIDE * cols * 2]
            .view(torch.bfloat16)
            .view(WIDE, cols)
        )

    def route_ids(self) -> torch.Tensor:
        """The routed experts of the last launch's rows, [WIDE, TOPK] (its route:
        [WIDE, TOPK + 1] ids, the shared expert last, then their weights)."""
        off, _ = self.layout["route"]
        ids = self.scratch[off : off + 4 * WIDE * (L.TOPK + 1)].view(torch.int32)
        return ids.view(WIDE, L.TOPK + 1)[:, : L.TOPK]

    def launch(self, n, i, layer, core, n_next, front=None):
        """Layer ``n`` of the step (model layer ``i``) from its out_proj / o_proj
        input ``core``, or for a ``gdn`` launch from the scratch rows with
        ``front`` (``_gdn_front``): the residual and the rows ``n_next`` norms stay
        in the scratch."""
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W

        o = layer.linear_attn.out_proj if _is_gdn(layer) else layer.o_proj
        ex, se = layer.mlp.experts, layer.mlp.shared_expert
        args = dict.fromkeys(W.ABI.names, 0) | {
            "scratch": self.scratch,
            "core_out" if _is_gdn(layer) else "attn": 0 if core is None else core,
            "n_post": layer.post_attention_layernorm.weight,
            "n_next": n_next,
            "w_out": o.weight,
            "w_out_s": o.weight_scale if _mxfp4(o) else 0,
            "w_gate": self.gates[n],
            "w13": ex.w13_weight,
            "w13s": ex.w13_weight_scale,
            "w2": ex.w2_weight,
            "w2s": ex.w2_weight_scale,
            "w_sd": 0 if se is None else se.down_proj.weight,
            "peers": self.peers,
            "rank": self.rank,
            "epoch": self.epoch,
            "layer": i,
            "sync": self.slots[n][0],
            "sync_next": self.slots[n][1],
            "folds": self.folds,
        }
        W.run(self.keys[n], args | (front or {}))


def _gdn_front(layer, emb, conv, temporal, idx) -> dict:
    """A ``gdn`` launch's GDN arguments: layer ``layer``'s weights, its conv and
    SSM states (``_Built._state``) and the step's state indices; ``emb``: the
    launch's input rows (a split launch: the previous layer's output; a wide
    step: the embedding rows of its first launch), else None."""
    a, qkvz = layer.linear_attn, layer.linear_attn.in_proj_qkvz
    return {
        "emb": 0 if emb is None else emb,
        "n_in": layer.input_layernorm.weight,
        "w_qkvz": qkvz.weight,
        "w_qkvz_s": qkvz.weight_scale if _mxfp4(qkvz) else 0,
        "w_ba": a.in_proj_ba.weight,
        "w_conv": a.conv1d.weight,
        "a_log": a.A_log,
        "dt_bias": a.dt_bias,
        "gdn_norm": a.norm.weight,
        "conv_state": conv,
        "conv_stride": conv.stride(0),
        "ssm": temporal,
        "ssm_stride": temporal.stride(0),
        "st_idx": idx,
    }


class _Built:
    """One rank's launches: the build keys per (width, variant), the scratch per
    width, the guard block (word 0 the step epoch), the peer buffer, the split
    launches of the 16-row decode steps (``_Split``, else None) and the wide
    launches of the 32-row steps (``_Wide``, else None)."""

    def __init__(
        self,
        lm,
        rank,
        state_bf16,
        k1_keys,
        k2_keys,
        k1_scratch,
        k2_scratch,
        epoch,
        peer,
        split=None,
        wide=None,
    ):
        self.lm, self.rank = lm, rank
        self.state_dtype = torch.bfloat16 if state_bf16 else torch.float32
        self.k1_keys, self.k2_keys = k1_keys, k2_keys
        # K1v: the draft tokens a request of the K1 keys (verify steps); 0 decode
        self.k1v = max((k.spec for k in k1_keys.values()), default=0)
        self.k1_widths = {w for w, *_ in k1_keys}
        self.wide = wide
        self.widths = {w for w, *_ in k2_keys} | ({WIDE} if wide else set())
        self.k1_scratch, self.k2_scratch = k1_scratch, k2_scratch
        self.epoch, self.peer, self.split = epoch, peer, split
        self.check_layers = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        lm = self.lm
        # SGLang sets layers_to_capture (EAGLE3 / DFlash targets) after the build,
        # and the launches keep no auxiliary hidden states.
        if (
            input_embeds is not None
            or pp_proxy_tensors is not None
            or kwargs
            or lm.model.layers_to_capture
            or plan.key.rows not in self.widths
        ):
            return type(lm).forward(
                lm,
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors,
                **kwargs,
            )
        if lm.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        m = lm.model
        s = plan.key.rows
        if s == WIDE:
            return self._wide_forward(input_ids, positions, forward_batch)
        decode = plan.key.forward_mode != "target_verify"
        k1 = s in self.k1_widths and decode == (not self.k1v)
        split = decode and s == ML.SPLIT_MAX and self.split is not None
        fb = (
            forward_batch
            if self.check_layers and not torch.cuda.is_current_stream_capturing()
            else None
        )
        self.epoch[:1].add_(1)
        h = m.embed_tokens(input_ids)
        residual = None
        reqs = s // self.k1v if k1 and self.k1v else s
        idx = self._state_indices(reqs) if k1 or split else None
        for n, i in enumerate(range(m.start_layer, m.end_layer)):
            layer = m.layers[i]
            if _is_gdn(layer) and k1:
                ring = (
                    self._ring(layer.linear_attn, forward_batch, reqs)
                    if self.k1v
                    else None
                )
                core, res_attn = self._k1(i, layer, h, residual, idx, s, fb, ring)
            elif _is_gdn(layer) and split:
                core, res_attn = self._front16(i, n, layer, h, residual, idx, fb)
            else:
                if residual is None:
                    res_attn, x = h, layer.input_layernorm(h)
                else:
                    x, res_attn = layer.input_layernorm(h, residual)
                core = self._core(layer, positions, x, forward_batch)
            h, residual = self._k2(i, layer, core, res_attn, s, fb)
        h, _ = m.norm(h, residual)
        return lm.logits_processor(input_ids, h, lm.lm_head, forward_batch, None)

    def _wide_forward(self, input_ids, positions, forward_batch):
        from sgl_amd_mono.qwen38_a95b_mxfp4 import check

        lm, m, w = self.lm, self.lm.model, self.wide
        fb = (
            forward_batch
            if self.check_layers and not torch.cuda.is_current_stream_capturing()
            else None
        )
        self.epoch[:1].add_(1)
        h = m.embed_tokens(input_ids)
        x, res = w.view("x", L.HIDDEN), w.view("res", L.HIDDEN)
        idx = None
        for n, i in enumerate(range(m.start_layer, m.end_layer)):
            layer = m.layers[i]
            core = front = None
            if w.keys[n].kind == "gdn":
                idx = self._state_indices(WIDE) if idx is None else idx
                conv, temporal = self._state(layer.linear_attn, dense=True)
                if fb is not None:
                    # SGLang's GDN on the launch's input rows: after layer 0,
                    # the rows that the launch before normed
                    head = check.k1_reference(
                        layer,
                        h if n == 0 else res,
                        None,
                        conv,
                        temporal,
                        idx[:WIDE],
                        fb,
                        x=None if n == 0 else x,
                    )
                front = _gdn_front(layer, h if n == 0 else None, conv, temporal, idx)
            elif n == 0:
                res.copy_(h)
                core = self._core(
                    layer, positions, layer.input_layernorm(h), forward_batch
                )
            else:
                core = self._core(layer, positions, x, forward_batch)
            nxt = m.layers[i + 1].input_layernorm if i + 1 < m.end_layer else m.norm
            if fb is not None:
                # SGLang's tail on the launch's input, then its next norm
                o = layer.linear_attn.out_proj if _is_gdn(layer) else layer.o_proj
                ref, ids, logits = check.k2_reference(
                    layer,
                    o,
                    core if front is None else head["core"],
                    (res if front is None else head["res_out"]).clone(),
                    fb,
                )
                ref["out"], ref["res_out"] = nxt(ref["out"], ref["res_out"])
            w.launch(n, i, layer, core, nxt.weight, front)
            if fb is not None:
                alike = check.k2_alike(
                    layer.layer_id, ids, logits, w.route_ids(), "wide"
                )
                got = {"out": x, "res_out": res}
                if front is not None:
                    st = check.k1_outputs(None, None, conv, temporal, idx[:WIDE])
                    for name in ("conv", "state"):
                        ref[name], got[name] = head[name], st[name]
                check.compare("wide", layer.layer_id, WIDE, ref, got, alike)
        # the final norm's rows: the next step's launches rewrite the scratch
        return lm.logits_processor(
            input_ids, x.clone(), lm.lm_head, forward_batch, None
        )

    @staticmethod
    def _core(layer, positions, x, forward_batch):
        if _is_gdn(layer):
            with _proj_input(layer.linear_attn, "out_proj"):
                return layer.linear_attn(x, forward_batch).contiguous()
        with _proj_input(layer, "o_proj"):
            return layer.self_attention(positions, x, forward_batch).contiguous()

    @staticmethod
    def _state_indices(s):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        md = get_attn_backend().linear_attn_backend.forward_metadata
        idx = md.mamba_cache_indices
        if idx.dtype != torch.int32 or not idx.is_contiguous() or idx.numel() < s:
            raise RuntimeError(
                f"Qwen3.8 monokernel: K1 needs {s} contiguous int32 state indices, "
                f"got {idx.dtype} {tuple(idx.shape)}"
            )
        return idx

    def _ring(self, a, forward_batch, reqs):
        """K1v's view of the layer's ReplaySSM ring (``gdn.Ring``), as
        ``GDNAttnBackend._replayssm_target_verify`` passes it to SGLang's verify
        kernel: the compact ring (low parts of d and the key, no beta), the
        request cursors, and the intermediate conv windows."""
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn

        be = get_attn_backend().linear_attn_backend
        st, pool, md = (
            be.state_layer_cache(a.layer_id),
            be.state_pool,
            be.forward_metadata,
        )
        ring = getattr(md, "replayssm_ring_indices", None)
        parts = ("replayssm_d", "replayssm_rawv", "replayssm_k", "replayssm_rawk")
        if (
            ring is None
            or any(getattr(st, n, None) is None for n in parts)
            or getattr(st, "replayssm_beta", None) is not None
            or pool.replayssm_spec_fold
            or pool.replayssm_is_kda
            or pool.replayssm_spec_write_pos is None
            or pool.replayssm_cache_base is None
        ):
            raise RuntimeError(
                f"Qwen3.8 monokernel: layer {a.layer_id}: K1v needs SGLang's compact "
                f"GDN ReplaySSM ring (--enable-linear-replayssm-spec)"
            )
        idx = md.mamba_cache_indices
        windows = select_verify_intermediate_state_indices(
            be.verify_intermediate_state_indices,
            forward_batch.req_pool_indices,
            lambda: idx[:reqs] >= 0,
            be.req_to_token_pool.size,
        )
        return gdn.Ring(
            ring_indices=ring,
            write_pos=pool.replayssm_spec_write_pos,
            cache_base=pool.replayssm_cache_base,
            d=st.replayssm_d,
            rawv=st.replayssm_rawv,
            k=st.replayssm_k,
            rawk=st.replayssm_rawk,
            g=st.replayssm_g,
            conv_window=st.intermediate_conv_window[0],
            conv_window_indices=windows,
        )

    def _state(self, a, dense=False, ring=False):
        """``dense``: a conv state slot is one dense block (the split launch and
        the wide ``gdn`` launch read no strides within it; K1 does). ``ring``:
        K1v, on a pool with the ReplaySSM ring."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        st = get_attn_backend().linear_attn_backend.state_layer_cache(a.layer_id)
        conv, temporal = st.conv[0], st.temporal
        want = self.state_dtype
        if (
            (getattr(st, "replayssm_d", None) is not None) != ring
            or temporal.dtype != want
            or conv.dtype != torch.bfloat16
            or tuple(conv.shape[1:]) != (L.CONV, L.CONV_W - 1)
            or not temporal[0].is_contiguous()
            or (dense and not conv[0].is_contiguous())
        ):
            raise RuntimeError(
                f"Qwen3.8 monokernel: layer {a.layer_id}: the GDN launches need the "
                f"{want} SSM state and the bf16 (slots, {L.CONV}, {L.CONV_W - 1}) "
                f"conv state, a slot dense, got {temporal.dtype} "
                f"{tuple(temporal.shape)} and {conv.dtype} {tuple(conv.shape)} "
                f"{conv.stride()}"
            )
        return conv, temporal

    def _k1(self, i, layer, h, residual, idx, s, check_batch=None, ring=None):
        """``check_batch``: the forward batch in check mode, else None. ``ring``:
        K1v's ``gdn.Ring`` (a verify step)."""
        from sgl_amd_mono.qwen38_a95b_mxfp4 import check
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn

        a = layer.linear_attn
        conv, temporal = self._state(a, ring=ring is not None)
        reqs = s // self.k1v if ring is not None else s
        if check_batch is not None:
            # in a verify step SGLang's GDN writes the records K1v rewrites
            ref = check.k1_reference(
                layer, h, residual, conv, temporal, idx[:reqs], check_batch
            )
            if ring is not None:
                ref |= check.k1v_records(ring, idx[:reqs], self.k1v)
        qq, qh = _format(a.in_proj_qkvz)
        core = torch.empty(s, L.CORE, dtype=h.dtype, device=h.device)
        res_out = torch.empty_like(h)
        gdn.gdn_pre(
            self.k1_keys[s, residual is None, qq, qh],
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w_qkvz=a.in_proj_qkvz.weight,
            w_ba=a.in_proj_ba.weight,
            conv_w=a.conv1d.weight,
            conv_state=conv,
            a_log=a.A_log,
            dt_bias=a.dt_bias,
            norm_w=a.norm.weight,
            rstate=temporal,
            st_idx=idx,
            core=core,
            scratch=self.k1_scratch[s],
            epoch=self.epoch,
            layer=i,
            w_qkvzs=a.in_proj_qkvz.weight_scale if qq else None,
            ring=ring,
        )
        if check_batch is not None and ring is None:
            got = check.k1_outputs(core, res_out, conv, temporal, idx[:s])
            check.compare("k1", layer.layer_id, s, ref, got)
        elif check_batch is not None:
            got = {"core": core, "res_out": res_out}
            got |= check.k1v_records(ring, idx[:reqs], self.k1v)
            ref = {n: ref[n] for n in got}
            check.compare("k1v", layer.layer_id, s, ref, got)
        return core, res_out

    def _front16(self, i, n, layer, h, residual, idx, check_batch=None):
        """GDN layer ``n`` of a 16-row decode step (model layer ``i``) as its
        split launch; checked as K1 is. ``check_batch``: the forward batch in
        check mode, else None."""
        from sgl_amd_mono.qwen38_a95b_mxfp4 import check

        s = ML.SPLIT_MAX
        conv, temporal = self._state(layer.linear_attn, dense=True)
        if check_batch is not None:
            ref = check.k1_reference(
                layer, h, residual, conv, temporal, idx[:s], check_batch
            )
        core, res = self.split.launch(n, i, layer, h, residual, conv, temporal, idx)
        if check_batch is not None:
            got = check.k1_outputs(core, res, conv, temporal, idx[:s])
            check.compare("k1", layer.layer_id, s, ref, got)
        return core, res

    def _k2(self, i, layer, core, residual, s, check_batch=None):
        from sgl_amd_mono.qwen38_a95b_mxfp4 import check
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import blocks

        mlp = layer.mlp
        o = layer.linear_attn.out_proj if _is_gdn(layer) else layer.o_proj
        if check_batch is not None:
            ref, ids, logits = check.k2_reference(layer, o, core, residual, check_batch)
        (oq, oh), fse = _format(o), _fse(mlp)
        se, ex = mlp.shared_expert, mlp.experts
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        key = self.k2_keys[s, fse, oq, oh]
        wide = isinstance(key, blocks.LayerBuild)
        (blocks.layer_block if wide else k2.layer_post)(
            key,
            core=core,
            residual=residual,
            w_o=o.weight,
            ln_w=layer.post_attention_layernorm.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            out=out,
            res_out=res_out,
            scratch=self.k2_scratch[s],
            peers=self.peer.addresses,
            rank=self.rank,
            epoch=self.epoch,
            layer=TAG_SLOTS // 2 + i,
            w_os=o.weight_scale if oq else None,
            w_sgu=None if se is None else se.gate_up_proj.weight,
            w_sd=None if se is None else se.down_proj.weight,
        )
        if check_batch is not None:
            # ROUTE: (value, tag) pairs; token t's experts at 2 k t, then its weights
            layout = blocks.layer_scratch_layout if wide else k2.scratch_layout
            off, n = layout(key)["route"]
            route = (
                self.k2_scratch[s][off : off + n].view(torch.int32).view(s, 2, -1, 2)
            )
            alike = check.k2_alike(layer.layer_id, ids, logits, route[:, 0, :, 0])
            got = {"out": out, "res_out": res_out}
            check.compare("k2", layer.layer_id, s, ref, got, alike)
        return out, res_out


def _compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(var)
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def _writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _k1v(bf16: bool) -> tuple[int, int]:
    """(draft tokens, ring length) when the verify steps run K1v, else (0, 0):
    SGLang's compact GDN ReplaySSM ring with a bf16 SSM state (each commit folds
    the records, so a verify step reads no history) and a draft chain of 2 to 4
    tokens."""
    from sglang.srt.runtime_context import get_exec, get_spec

    spec, mamba = get_spec(), get_exec().mamba
    tokens = spec.speculative_num_draft_tokens or 0
    if (
        os.environ.get("SGLANG_AMD_MONO_K1V", "1") == "0"
        or (spec.speculative_eagle_topk or 1) > 1
        or not mamba.enable_linear_replayssm_spec
        or not bf16
        or not 2 <= tokens <= 4
    ):
        return 0, 0
    return tokens, mamba.linear_replayssm_cache_len


def _scratch(keys: dict, size, device) -> dict:
    """One zeroed scratch a width, as large as its largest variant's."""
    out = {}
    for key in keys.values():
        out[key.tokens] = max(out.get(key.tokens, 0), size(key))
    return {w: torch.zeros(n, dtype=torch.uint8, device=device) for w, n in out.items()}


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype
    from sglang.srt.runtime_context import get_parallel, get_spec

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import blocks
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W

    _writable_flydsl_cache()
    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    layers = _layers(model)
    device = layers[0].input_layernorm.weight.device
    e1, e_norm, e2 = _eps(layers)
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    # the target model runs no plain decode step under speculative decoding
    verify = get_spec().speculative_algorithm is not None
    gdns = [x for x in layers if _is_gdn(x)]
    qkvz = sorted({_format(x.linear_attn.in_proj_qkvz) for x in gdns})
    bf16 = mamba2_state_dtype(hf_config).temporal == torch.bfloat16
    narrow = [w for w in ctx.widths if w <= L.MAX_TOKENS]
    k1_keys = {}
    spec, ring_len = _k1v(bf16) if verify else (0, 0)
    if not verify or spec:
        for w in narrow:
            if spec and w % spec:
                continue
            for first in sorted({x is layers[0] for x in gdns}):
                for qq, qh in qkvz:
                    k1_keys[w, first, qq, qh] = gdn.K1Build(
                        tokens=w,
                        first=first,
                        eps=e1,
                        norm_eps=e_norm,
                        state_bf16=bf16,
                        qq=qq,
                        qh=qh,
                        spec=spec,
                        ring_len=ring_len,
                        fault_site=fault_site,
                        fault_after=fault_after,
                    )
    outs = {x.linear_attn.out_proj if _is_gdn(x) else x.o_proj for x in layers}
    out_formats = sorted({_format(o) for o in outs})

    def layer_block_at(w, fse):
        return L.MAX_TOKENS < w <= ML.SPLIT_MAX or (fse and w in LAYER_WIDTHS)

    k2_keys = {
        (w, fse, oq, oh): k2.K2Build(
            tokens=w,
            eps=e2,
            fse=fse,
            oq=oq,
            oh=oh,
            fault_site=fault_site,
            fault_after=fault_after,
        )
        for w in narrow
        for fse in sorted({_fse(x.mlp) for x in layers})
        if not layer_block_at(w, fse)
        for oq, oh in out_formats
    }
    wide_keys = {
        (w, fse, oq, oh): blocks.LayerBuild(
            tokens=w,
            eps=e2,
            fse=fse,
            oq=oq,
            oh=oh,
            fault_site=fault_site,
            fault_after=fault_after,
        )
        for w in ctx.widths
        for fse in sorted({_fse(x.mlp) for x in layers})
        if layer_block_at(w, fse)
        for oq, oh in out_formats
    }
    builds_wide = _builds_wide(model, layers, ctx.widths)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(ctx.deadline_s)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    # collective: every rank builds it at the same point
    group = p.tp_group.cpu_group
    if builds_wide:
        nbytes = W.peer_buffer_bytes(tp)
    elif wide_keys:
        nbytes = blocks.peer_buffer_bytes()
    else:
        nbytes = k2.peer_buffer_bytes()
    peer = PeerBuffer(nbytes, group, rank, tp, device)
    peer.bytes.zero_()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)  # no peer writes before its owner zeroed
    k2_scratch = _scratch(
        k2_keys | wide_keys,
        lambda k: (
            blocks.scratch_bytes(blocks.layer_scratch_layout(k))
            if isinstance(k, blocks.LayerBuild)
            else k2.scratch_bytes(k)
        ),
        device,
    )
    split, split_keys = None, []
    if _builds_split(layers, ctx.widths, verify):
        keys = _split_keys(layers, e1, e_norm, bf16, tp, fault_site, fault_after)
        split = _Split(keys, peer.addresses, rank, epoch, device)
        split_keys = list(dict.fromkeys(k for k in keys if k is not None))
    wide = None
    if builds_wide:
        front = None if verify else (e_norm, bf16)
        keys = _wide_keys(layers, e2, tp, fault_site, fault_after, front)
        wide = _Wide(keys, _router_gates(layers), peer.addresses, rank, epoch, device)
    built = _Built(
        model,
        rank,
        bf16,
        k1_keys,
        k2_keys | wide_keys,
        _scratch(k1_keys, gdn.scratch_bytes, device),
        k2_scratch,
        epoch,
        peer,
        split,
        wide,
    )
    # Every rank builds every key. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    launches = [gdn.build(k) for k in k1_keys.values()]
    launches += [k2.build(k) for k in k2_keys.values()]
    launches += [blocks.build_layer(k) for k in wide_keys.values()]
    launches += [W.build(k) for k in split_keys]
    wide_builds = sorted(set(wide.keys), key=repr) if wide else []
    launches += [W.build(k) for k in wide_builds]
    start = len(launches) * rank // tp
    t0 = time.monotonic()
    for launch in launches[start:] + launches[:start]:
        _compile_only(launch)
    torch.cuda.synchronize()
    logger.info(
        "Qwen3.8 monokernel: %d K1%s, %d K2, %d layer_block, %d split and %d wide "
        "launches (widths %s, %s) in %.0f s",
        len(k1_keys),
        f" (K1v, {spec} draft tokens)" if spec else "",
        len(k2_keys),
        len(wide_keys),
        len(split_keys),
        len(wide_builds),
        sorted(built.widths),
        "verify" if verify else "decode",
        time.monotonic() - t0,
    )
    return built


SPEC = MonoSpec(
    name="qwen38_a95b_mxfp4_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=WIDTHS, tp_sizes=(L.TP,), resident_blocks=L.BLOCKS
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={"moe_runner_backend": frozenset({"auto", "aiter"})},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
