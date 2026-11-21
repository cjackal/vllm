#!/usr/bin/env python3
"""Check vLLM's per-layer quantization decisions against real checkpoints.

For every checkpoint tensor that vLLM loads into a quantizable layer:

  (b') metadata  the layer's decision equals the decision the checkpoint's own
                 quant metadata gives for the tensor's checkpoint name, looked
                 up in the checkpoint namespace (no vLLM name mapping);
  (c)  disk      the layer's decision matches the stored format (weight dtype
                 and scale tensors from the safetensors header);
  (a)  prefix    the prefix a layer passed to the quant config equals its
                 module path (reported, not graded).

Only configs and safetensors headers are downloaded. `run` builds each model on
the meta device and drives vLLM's real loader (`DefaultModelLoader` ->
`model.load_weights`) with meta tensors of the stored shape and dtype. It needs
a CUDA build of vLLM and a visible GPU, but no GPU memory for weights.

Subcommands:
  list       print the cases
  offline    (no torch) stored format vs metadata for every checkpoint module
  run        (vLLM) run the cases, check expectations, grade the patch's output
  selftest   (no torch) check the pure helpers

Typical use, on a CUDA host from the vLLM checkout:
  .venv/bin/python quant_invariant_check.py run --out qcheck/base
  .venv/bin/python quant_invariant_check.py run --out qcheck/patch \\
      --patch-env VLLM_SOME_FLAG=warn --patch-regex '<message regex>'
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fnmatch
import functools
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path

LOAD_FORMAT = "qcheck_names_only"
ERROR_VERDICTS = ("LOOKUP_BUG", "META_BUG", "MISMATCH", "HARDCODED", "DISK_MIXED")
WARN_VERDICTS = ("META_STALE",)
DEFAULT_PATCH_REGEX = r"(?i)quant\w*[^\n]{0,40}(mismatch|inconsisten|disagree)"

# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------

REV = {
    "arcee-ai/Trinity-Nano-Preview-NVFP4": "6ffbc24c4d58965cc505401f2189272fd976503c",
    "Inferact/GLM-5.3-NVFP4": "ce67b36f3669192b5bb233819f0fda6c8a9837f8",
    "nvidia/DeepSeek-R1-0528-NVFP4-v2": "25a138f28f49022958b9f2d205f9b7de0cdb6e18",
    "nvidia/DeepSeek-V3.2-NVFP4": "7c0f62c6da1da0c81c6e097010cc55854d206812",
    "nvidia/DeepSeek-V4.1-Flash-NVFP4": "c48dca0146b56d64c8116dbb6fcace8b7cc835f8",
    "nvidia/Gemma-4-31B-IT-NVFP4": "4135a98a9b728a548947683219633b25682223ac",
    "nvidia/GLM-5.3-Flash-FP4-Local": "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5",
    "nvidia/GLM-5.3-Flash-NVFP4": "1d00738646cbb0c6d0a9eb403dac5f366f1473a7",
    "nvidia/Kimi-K2.5-NVFP4": "0fd0a5e6879298d3476e3b61852a79792a35ae3d",
    "nvidia/Kimi-K2.6-NVFP4": "2fd3a800dedd098b8327eb49e93ebc75f85da19f",
    "nvidia/MiniMax-M3-NVFP4": "901464083161bf8612a29ff7ad29914cd4ab4a85",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": (
        "ae1368329b1fc76b71534c5d04b87b35d9cc6e15"
    ),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": (
        "ff433f5493e25d631c9f12b5d55c674229923d02"
    ),
    "nvidia/Qwen3.5-397B-A17B-NVFP4": "12b28061efd8145b11226e4fab2528ee4f39ef09",
    "nvidia/Qwen3.6-27B-NVFP4": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "nvidia/Qwen3.6-35B-A3B-NVFP4": "1355db6a052410cfd62085d94b58866fd0f2c3c5",
    "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4": (
        "8fb2682f136cf94d932a498f18cb1e428832a912"
    ),
    "stepfun-ai/Step-3.7-Flash-NVFP4": "4275532ffd9a9496ff36b7a2dc4a9db1048da438",
    "wangqia0309/gemma-4-26B-A4B-it_nvfp4_experts_fp8_dense_fp8_attn-kv_fp8": (
        "e52e75cef0b34aaa597df50f3eb352cbc32af458"
    ),
}
MTP = {"method": "mtp", "num_speculative_tokens": 1}


@dataclasses.dataclass
class Case:
    """One load configuration and what the checker must report for it.

    `expect` and `allow` are regexes over `"<role>:<module path>"` of layers
    with an error verdict (role is `target` or `draft`). While the bug is
    present every `expect` pattern must match; after `fixed_by` is an ancestor
    of the tested tree none may. `inject` recreates a fixed bug at runtime, so
    an injected case always has its bug present.
    """

    id: str
    model: str
    spec: dict | None = None
    lm_only: bool = False
    inject: tuple[str, ...] = ()
    expect: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()
    fixed_by: str | None = None
    ref: str = ""
    engine_args: dict = dataclasses.field(default_factory=dict)

    @property
    def revision(self) -> str | None:
        return REV.get(self.model)


def _variants(cid, model, *, spec=False, lm=False, **kw) -> list[Case]:
    out = [Case(cid, model, **kw)]
    if spec:
        out.append(Case(f"{cid}-mtp", model, spec=MTP, **kw))
    if lm:
        out.append(Case(f"{cid}-lm", model, lm_only=True, **kw))
    return out


TRINITY = "arcee-ai/Trinity-Nano-Preview-NVFP4"
GLM53 = "Inferact/GLM-5.3-NVFP4"
DSR1 = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
DSV32 = "nvidia/DeepSeek-V3.2-NVFP4"
DSV41 = "nvidia/DeepSeek-V4.1-Flash-NVFP4"
GEMMA31 = "nvidia/Gemma-4-31B-IT-NVFP4"
GLMF_MX = "nvidia/GLM-5.3-Flash-FP4-Local"
GLMF = "nvidia/GLM-5.3-Flash-NVFP4"
KIMI25 = "nvidia/Kimi-K2.5-NVFP4"
KIMI26 = "nvidia/Kimi-K2.6-NVFP4"
MINIMAX = "nvidia/MiniMax-M3-NVFP4"
NEMO_VL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
NEMO = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
QWEN35 = "nvidia/Qwen3.5-397B-A17B-NVFP4"
QWEN36D = "nvidia/Qwen3.6-27B-NVFP4"
QWEN36M = "nvidia/Qwen3.6-35B-A3B-NVFP4"
QNEXT = "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"
STEP = "stepfun-ai/Step-3.7-Flash-NVFP4"
GEMMA26 = "wangqia0309/gemma-4-26B-A4B-it_nvfp4_experts_fp8_dense_fp8_attn-kv_fp8"

CASES: list[Case] = [
    # Checkpoints as published, on the tested tree.
    Case("trinity", TRINITY, expect=(r"^target:lm_head$",), ref="#68263 (open)"),
    *_variants("glm53", GLM53, spec=True, ref="#58209 (open) may flag the draft"),
    *_variants("dsr1", DSR1, spec=True),
    *_variants(
        "dsv32",
        DSV32,
        spec=True,
        expect=(r"fused_qkv_a_proj$",),
        fixed_by="c334a4959a",
        ref="#68948",
    ),
    *_variants("dsv41", DSV41, spec=True, ref="#59655 (fixed)"),
    *_variants("gemma4-31b", GEMMA31, lm=True),
    Case(
        "glmf-mx",
        GLMF_MX,
        expect=(r"^target:.*\.self_attn\.", r"^target:visual\."),
        ref="#66882 (open)",
    ),
    Case(
        "glmf-mx-mtp",
        GLMF_MX,
        spec=MTP,
        expect=(r"^target:.*\.self_attn\.", r"^draft:.*experts$"),
        ref="#66882 (open)",
    ),
    *_variants("glmf", GLMF, spec=True, lm=True),
    *_variants("kimi25", KIMI25, lm=True),
    Case("kimi26-lm", KIMI26, lm_only=True),
    *_variants("minimax-m3", MINIMAX, lm=True),
    Case("nemo-vl", NEMO_VL),
    Case(
        "nemo-vl-mtp",
        NEMO_VL,
        spec=MTP,
        expect=(r"^draft:.*\.mixer\.(experts|shared_experts\.(up|down)_proj)$",),
        ref="#65873 (open)",
    ),
    *_variants("nemo", NEMO, spec=True),
    *_variants("qwen35", QWEN35, spec=True, lm=True),
    *_variants("qwen36-27b", QWEN36D, lm=True),
    *_variants("qwen36-35b", QWEN36M, spec=True, lm=True),
    *_variants("qnext", QNEXT, spec=True),
    *_variants("step37", STEP, spec=True, lm=True),
    *_variants("gemma4-26b", GEMMA26, lm=True),
    # Fixed bugs recreated at runtime (see `Injector`).
    Case(
        "replay-53790-nemo-mtp-mapper",
        NEMO,
        spec=MTP,
        inject=("no_mapper:^NemotronHMTP$",),
        expect=(r"^draft:",),
        allow=(r".",),
        ref="#53790",
    ),
    Case(
        "replay-42546-qwen36-prefix-map",
        QWEN36D,
        lm_only=True,
        inject=("no_mapper:^Qwen3_5",),
        expect=(r"^target:.*layers\.\d+\.",),
        allow=(r".",),
        ref="#42546",
    ),
    Case(
        "replay-47445-minimax-fused-siblings",
        MINIMAX,
        lm_only=True,
        inject=("no_packed:^(qkv_proj|gate_up_proj)$",),
        expect=(r"qkv_proj$",),
        allow=(r".",),
        ref="#47445",
    ),
    Case(
        "replay-48563-gemma4-experts",
        GEMMA26,
        lm_only=True,
        inject=("no_mapper:^Gemma4",),
        expect=(r"experts$",),
        allow=(r".",),
        ref="#48563",
    ),
    Case(
        "replay-48883-step-mtp-head",
        STEP,
        spec=MTP,
        inject=(r"prefix:^.*\.shared_head\.head$=>",),
        expect=(r"^draft:.*shared_head\.head$",),
        allow=(r".",),
        ref="#48883",
    ),
    Case(
        "replay-38832-qwen35-mtp-fc",
        QWEN35,
        spec=MTP,
        inject=(r"requant:(^|\.)fc$",),
        expect=(r"^draft:(.*\.)?fc$",),
        allow=(r".",),
        ref="#38832",
    ),
    Case(
        "replay-46316-qnext-mtp-fc",
        QNEXT,
        spec=MTP,
        inject=(r"requant:(^|\.)fc$",),
        expect=(r"^draft:(.*\.)?fc$",),
        allow=(r".",),
        ref="#46316",
    ),
]

# ----------------------------------------------------------------------------
# Pure helpers (no torch)
# ----------------------------------------------------------------------------

ALGO_CLASS = {
    "NVFP4": "nvfp4",
    "W4A16_NVFP4": "nvfp4",
    "NVFP4_AWQ": "nvfp4",
    "FP8": "fp8",
    "FP8_PER_CHANNEL_PER_TOKEN": "fp8",
    "FP8_PB_WO": "fp8",
    "MXFP8": "mxfp8",
    "MXFP4": "mxfp4",
    "INT8_PB_WO": "int8",
    "INT8_SQ": "int8",
    "W4A8_AWQ": "int4",
}
# Formats ModelOpt leaves a module in when it does not quantize it.
MODELOPT_METHODS = {
    None,
    "modelopt",
    "modelopt_fp4",
    "modelopt_fp8",
    "modelopt_mixed",
    "modelopt_mxfp8",
}
TEXT_KEYS = ("text_config", "llm_config", "language_config", "thinker_config")
FLOATS = {"BF16", "F16", "F32", "F64"}
FP8S = {"F8_E4M3", "F8_E5M2"}
E8M0 = {"U8", "F8_E8M0"}
ACT_SCALES = {
    "input_scale",
    "k_scale",
    "v_scale",
    "q_scale",
    "prob_scale",
    "output_scale",
    "weight_scale_2",
}
NUM = re.compile(r"(?<=\.)\d+(?=\.|$)")


def algo_class(algo: str | None) -> str:
    if not algo:
        return "?"
    return ALGO_CLASS.get(algo.upper(), algo.lower())


def collapse(name: str) -> str:
    return NUM.sub("N", name)


def split(name: str) -> tuple[str, str]:
    mod, _, leaf = name.rpartition(".")
    return mod, leaf


def ancestors(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def is_scale(leaf: str) -> bool:
    return "scale" in leaf


def is_bias(leaf: str) -> bool:
    return leaf == "bias" or leaf.endswith("_bias")


def is_mx(w: list[int], s: list[int]) -> bool:
    return len(w) >= 2 and len(s) == len(w) and s[-2] == w[-2] and s[-1] * 32 == w[-1]


def weight_class(
    dtype: str, shape: list[int], scales: list[tuple], global_scale: bool = False
) -> str:
    """Stored format of a weight from its dtype and its block scales.

    `global_scale`: the weight has an NVFP4-style second-level scale
    (`weight_scale_2`), which identifies NVFP4 even when the block scale is
    stored in an encoding (e.g. `weight_scale.csf32_*`).
    """
    if dtype in FLOATS:
        return "bf16"
    big = max(scales, key=lambda s: math.prod(s[1]) if s[1] else 1, default=None)
    if dtype in FP8S:
        if big is not None and big[0] in E8M0 and is_mx(shape, big[1]):
            return "mxfp8"
        return "fp8"
    if dtype in ("U8", "I8") and big is not None:
        if big[0] in FP8S:
            return "nvfp4"
        if big[0] in E8M0:
            return "mxfp4"
    if dtype == "U8" and global_scale:
        return "nvfp4"
    if dtype == "I8":
        return "int8"
    if dtype in ("I32", "U32"):
        return "int4"
    return f"?{dtype}"


class Header:
    """Tensor names, dtypes and shapes of a checkpoint (from headers only)."""

    def __init__(self, tensors: dict[str, list]):
        self.t = tensors
        self.by_mod: dict[str, list[str]] = defaultdict(list)
        for n in tensors:
            m, leaf = split(n)
            self.by_mod[m].append(leaf)

    def is_weight(self, name: str) -> bool:
        info = self.t.get(name)
        _, leaf = split(name)
        return (
            info is not None
            and not is_scale(leaf)
            and not is_bias(leaf)
            and len(info[1]) >= 2
        )

    def disk_class(self, name: str) -> str:
        dtype, shape = self.t[name][:2]
        mod, leaf = split(name)
        scales = [
            tuple(self.t[f"{mod}.{s}"][:2])
            for s in self.by_mod[mod]
            if is_scale(s)
            and s not in ACT_SCALES
            and (leaf == "weight" or s.startswith(leaf) or s in ("scale", "scales"))
        ]
        stem = "weight" if leaf == "weight" else leaf
        return weight_class(dtype, shape, scales, f"{stem}_scale_2" in self.by_mod[mod])


@dataclasses.dataclass
class QuantMeta:
    """A checkpoint's quant metadata, read in the checkpoint namespace."""

    source: str
    method: str | None
    algo: str | None
    exclude: list[str]
    layers: dict[str, str]
    native: str

    def __post_init__(self):
        self._ex = (
            re.compile("|".join(f"(?:{fnmatch.translate(p)})" for p in self.exclude))
            if self.exclude
            else None
        )
        self._cache: dict[tuple[str, bool], str] = {}

    @classmethod
    def from_configs(cls, cfg: dict | None, hfq: dict | None) -> QuantMeta | None:
        cfg = cfg or {}
        qc, src = cfg.get("quantization_config"), "config.json"
        for k in TEXT_KEYS:
            sub = cfg.get(k)
            if not qc and isinstance(sub, dict) and sub.get("quantization_config"):
                qc, src = sub["quantization_config"], f"config.json:{k}"
        if not qc and hfq and isinstance(hfq.get("quantization"), dict):
            qc, src = hfq["quantization"], "hf_quant_config.json"
        if not qc:
            return None
        method = qc.get("quant_method")
        exclude = list(
            dict.fromkeys(
                [*(qc.get("ignore") or []), *(qc.get("exclude_modules") or [])]
            )
        )
        layers = {
            k: algo_class(v.get("quant_algo") if isinstance(v, dict) else v)
            for k, v in (qc.get("quantized_layers") or {}).items()
        }
        return cls(
            source=src,
            method=method,
            algo=qc.get("quant_algo"),
            exclude=exclude,
            layers=layers,
            native="bf16" if method in MODELOPT_METHODS else "?",
        )

    def decide(self, module: str, embedding: bool = False) -> str:
        key = (module, embedding)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        anc = ancestors(module)
        if self._ex is not None and any(self._ex.match(a) for a in anc):
            res = self.native
        elif self.layers:
            res = next((self.layers[a] for a in anc if a in self.layers), self.native)
        elif embedding:
            res = self.native
        else:
            res = algo_class(self.algo)
        self._cache[key] = res
        return res


def verdict(
    vllm: str, governed: bool, metas: list[str], disks: list[str], dequant: bool
) -> str:
    """Grade one layer.

    vllm: format the layer was built for. governed: the layer asked the quant
    config (`get_quant_method`); hard-coded layers only get the disk check.
    metas/disks: metadata decision and stored format of the checkpoint modules
    whose tensors reached the layer.
    """
    if vllm.startswith("?"):
        return "UNKNOWN_VLLM"
    if any(d.startswith("?") for d in disks):
        return "UNKNOWN_DISK"
    if len(disks) != 1:
        return "DISK_MIXED"
    disk = disks[0]
    disk_ok = vllm == disk or (dequant and disk == "bf16")
    if not governed:
        return "OK" if disk_ok else "HARDCODED"
    known = {m for m in metas if m != "?"}
    if disk_ok:
        return "OK" if known <= {vllm} else "META_STALE"
    if known == {disk}:
        return "LOOKUP_BUG"
    if known and known <= {vllm}:
        return "META_BUG"
    return "MISMATCH"


def names_layer(qual: str, text: str) -> bool:
    pat = rf"(?<![\w.]){re.escape(qual)}(?!\w|\.\w)"
    return re.search(pat, text) is not None


def grade_patch(entries: list[dict], messages: list[str], regex: str) -> dict:
    """How the patch's messages line up with the oracle's verdicts."""
    picked = [m for m in messages if re.search(regex, m)]
    blob = "\n".join(picked)
    errs = [e for e in entries if e["verdict"] in ERROR_VERDICTS]
    caught = [e for e in errs if names_layer(e["qual"], blob)]
    flagged_ok = [
        f"{e['role']}:{e['qual']}"
        for e in entries
        if e["verdict"] == "OK" and names_layer(e["qual"], blob)
    ]
    return {
        "messages": len(picked),
        "errors": len(errs),
        "caught": len(caught),
        "missed": [f"{e['role']}:{e['qual']}" for e in errs if e not in caught][:50],
        "flagged_ok": flagged_ok[:50],
        "n_flagged_ok": len(flagged_ok),
        "sample": picked[:5],
    }


def evaluate(case: Case, res: dict, present: bool) -> tuple[list[str], list[str]]:
    """Return (problems, new findings) for a case result."""
    problems: list[str] = []
    if "entries" not in res:
        return [f"{res.get('status')}: {res.get('exception', '')[-300:]}"], []
    errs = [
        f"{e['role']}:{e['qual']}"
        for e in res["entries"]
        if e["verdict"] in ERROR_VERDICTS
    ]
    expect = case.expect if present else ()
    for pat in expect:
        if not any(re.search(pat, k) for k in errs):
            problems.append(f"MISSED {pat}")
    if not present and case.expect:
        still = [k for k in errs if any(re.search(p, k) for p in case.expect)]
        if still:
            problems.append(f"STILL_FLAGGED {len(still)} e.g. {still[0]}")
    new = [k for k in errs if not any(re.search(p, k) for p in (*expect, *case.allow))]
    for role, exc in res.get("load_exception", {}).items():
        problems.append(f"{role} load raised: {exc.strip().splitlines()[-1][:200]}")
    return problems, new


# ----------------------------------------------------------------------------
# Hub access (no torch)
# ----------------------------------------------------------------------------


def fetch_json(model: str, revision: str | None, filename: str) -> dict | None:
    p = Path(model)
    if p.is_dir():
        f = p / filename
        return json.loads(f.read_text()) if f.exists() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return json.loads(
            Path(hf_hub_download(model, filename, revision=revision)).read_text()
        )
    except EntryNotFoundError:
        return None


def fetch_header(model: str, revision: str | None, cache: Path) -> dict[str, list]:
    """{tensor name: [dtype, shape, file]} read from safetensors headers."""
    p = Path(model)
    out: dict[str, list] = {}
    if p.is_dir():
        for f in sorted(p.glob("*.safetensors")):
            with open(f, "rb") as fh:
                hdr = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
            for k, v in hdr.items():
                if k != "__metadata__":
                    out[k] = [v["dtype"], v["shape"], f.name]
        return out
    cf = cache / f"{model.replace('/', '__')}@{revision or 'main'}.header.json"
    if cf.exists():
        return json.loads(cf.read_text())
    from huggingface_hub import HfApi

    md = HfApi().get_safetensors_metadata(model, revision=revision)
    for fname, fm in sorted(md.files_metadata.items()):
        for name, ti in fm.tensors.items():
            out[name] = [ti.dtype, list(ti.shape), fname]
    cache.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out))
    return out


def pinned(model: str, revision: str | None) -> str | None:
    return revision or REV.get(model)


def load_meta(model: str, revision: str | None) -> QuantMeta | None:
    revision = pinned(model, revision)
    return QuantMeta.from_configs(
        fetch_json(model, revision, "config.json"),
        fetch_json(model, revision, "hf_quant_config.json"),
    )


# ----------------------------------------------------------------------------
# Worker (imports torch and vLLM)
# ----------------------------------------------------------------------------

TORCH_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
}
KV_PARAMS = {"k_scale", "v_scale", "q_scale", "prob_scale"}


def vllm_class(qm) -> tuple[str, bool]:
    """(format the layer was built for, whether it dequantizes to BF16)."""
    if qm is None:
        return "bf16", False
    name = type(qm).__name__
    if name == "DequantLinearMethod":
        return vllm_class(qm.base)[0], True
    key = getattr(getattr(qm, "spec", None), "weight", None)
    if key is not None:
        dt = str(getattr(key, "dtype", "")).lower()
        gs = tuple(getattr(getattr(key, "scale", None), "group_shape", ()) or ())
        if "float4" in dt or "fp4" in dt:
            return ("mxfp4" if gs[-1:] == (32,) else "nvfp4"), False
        if "float8" in dt:
            return ("mxfp8" if gs == (1, 32) else "fp8"), False
        if "int8" in dt:
            return "int8", False
        if "int4" in dt:
            return "int4", False
    n = name.lower()
    if "unquantized" in n:
        return "bf16", False
    for k, c in (
        ("mxfp4", "mxfp4"),
        ("nvfp4", "nvfp4"),
        ("fp4", "nvfp4"),
        ("mxfp8", "mxfp8"),
        ("fp8", "fp8"),
        ("int8", "int8"),
    ):
        if k in n:
            return c, False
    return f"?{name}", False


class Injector:
    """Recreate a fixed bug at runtime.

    no_mapper:<class regex>   the model class's hf_to_vllm_mapper is not applied
                              to the quant config
    no_packed:<key regex>     matching packed_modules_mapping keys never reach
                              the quant config
    prefix:<regex>=><repl>    get_quant_method sees re.sub(regex, repl, prefix)
    requant:<prefix regex>    a LinearBase built with quant_config=None gets the
                              model's quant config (undoes a hard-coded BF16)
    """

    def __init__(self, specs: tuple[str, ...]):
        self.no_mapper = self.no_packed = self.requant = None
        self.prefix: list[tuple[re.Pattern, str]] = []
        for s in specs:
            kind, _, arg = s.partition(":")
            if kind == "prefix":
                rx, _, repl = arg.partition("=>")
                self.prefix.append((re.compile(rx), repl))
            elif kind in ("no_mapper", "no_packed", "requant"):
                setattr(self, kind, re.compile(arg))
            else:
                raise ValueError(f"unknown injector {s!r}")

    def rewrite(self, prefix: str) -> str:
        for rx, repl in self.prefix:
            if rx.search(prefix):
                prefix = rx.sub(repl, prefix)
        return prefix


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        with contextlib.suppress(Exception):
            self.messages.append(
                f"{record.levelname} {record.name}: {record.getMessage()}"
            )


def attach(capture: LogCapture) -> None:
    """(Re)attach the capture; vLLM may reconfigure its logger lazily."""
    logger = logging.getLogger("vllm")
    if capture not in logger.handlers:
        logger.addHandler(capture)


class Recorder:
    """Records what each layer asked the quant config, and which checkpoint
    tensors reached which module."""

    def __init__(self, inj: Injector, cache: Path):
        import torch
        import torch.utils._pytree as pytree

        self.torch = torch
        self.inj = inj
        self.cache = cache
        self.role = "target"
        self.depth = 0
        self.skip_mapper = False
        self.calls: dict[int, dict] = {}
        self.keep: list = []
        self.models: dict[str, object] = {}
        self.sources: dict[str, tuple[str, str | None]] = {}
        self.tensors: dict[str, list] = {}
        self.yielded: dict[str, set[str]] = defaultdict(set)
        self.routed: dict[str, set[str]] = defaultdict(set)
        self.routes: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.hit: dict[str, set[str]] = defaultdict(set)
        self.untagged: Counter = Counter()
        self.loader_errors: Counter = Counter()
        self.loader_error_samples: list[str] = []
        self.current: str | None = None
        self.events: Counter = Counter()

        class Tagged(torch.Tensor):
            """Meta tensor carrying the checkpoint names it was derived from."""

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                flat = pytree.tree_flatten((args, kwargs))[0]
                tags: set[str] = set()
                for x in flat:
                    t = getattr(x, "_ckpt_tags", None)
                    if t:
                        tags |= t
                with torch._C.DisableTorchFunctionSubclass():
                    out = func(*args, **kwargs)
                if not tags:
                    return out
                frozen = frozenset(tags)
                inputs = {id(x) for x in flat if isinstance(x, torch.Tensor)}

                def wrap(o):
                    if (
                        isinstance(o, torch.Tensor)
                        and not isinstance(o, torch.nn.Parameter)
                        and id(o) not in inputs
                    ):
                        o = o.as_subclass(Tagged)
                        o._ckpt_tags = frozen
                    return o

                return pytree.tree_map(wrap, out)

        self.Tagged = Tagged

    def meta_tensor(self, name: str, dtype: str, shape: list[int]):
        torch = self.torch
        dt = getattr(torch, TORCH_DTYPES.get(dtype, "uint8"), torch.uint8)
        t = torch.empty(shape, dtype=dt, device="meta").as_subclass(self.Tagged)
        t._ckpt_tags = frozenset((name,))
        return t

    # -- quant config hooks ----------------------------------------------------

    def patch_quant_configs(self):
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig,
        )

        seen, stack = set(), [QuantizationConfig]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            stack.extend(c.__subclasses__())
            for attr, make in (
                ("get_quant_method", self._wrap_gqm),
                ("apply_vllm_mapper", self._wrap_mapper),
            ):
                f = c.__dict__.get(attr)
                if f is not None and not getattr(f, "_qcheck", False):
                    w = make(f)
                    w._qcheck = True
                    setattr(c, attr, w)

    def _wrap_gqm(self, f):
        rec = self

        @functools.wraps(f)
        def gqm(qc, layer, *args, **kwargs):
            if rec.depth:
                return f(qc, layer, *args, **kwargs)
            prefix = kwargs.get("prefix", args[0] if args else "")
            seen = rec.inj.rewrite(prefix)
            if seen != prefix:
                if "prefix" in kwargs:
                    kwargs["prefix"] = seen
                else:
                    args = (seen, *args[1:])
            rec.depth += 1
            try:
                ret = f(qc, layer, *args, **kwargs)
            finally:
                rec.depth -= 1
            rec.calls[id(layer)] = {
                "role": rec.role,
                "prefix": prefix,
                "seen": seen,
                "layer": type(layer).__name__,
                "config": type(qc).__name__,
                "method": type(ret).__name__ if ret is not None else None,
            }
            rec.keep.append(layer)
            return ret

        return gqm

    def _wrap_mapper(self, f):
        rec = self

        @functools.wraps(f)
        def mapper(qc, *args, **kwargs):
            if rec.skip_mapper:
                rec.events["mapper skipped"] += 1
                return None
            return f(qc, *args, **kwargs)

        return mapper

    def install_injectors(self):
        import vllm.model_executor.model_loader.utils as lu
        from vllm.model_executor.models.interfaces import SupportsQuant

        inj, rec = self.inj, self

        def drop_packed(qc):
            if qc is None or inj.no_packed is None:
                return
            pm = getattr(qc, "packed_modules_mapping", None) or {}
            kept = {k: v for k, v in pm.items() if not inj.no_packed.search(k)}
            if len(kept) != len(pm):
                rec.events["packed keys dropped"] += len(pm) - len(kept)
            qc.packed_modules_mapping = kept

        orig_cfg = lu.configure_quant_config

        def configure(qc, model_class):
            skip = inj.no_mapper is not None and bool(
                inj.no_mapper.search(model_class.__name__)
            )
            rec.skip_mapper = skip
            try:
                orig_cfg(qc, model_class)
            finally:
                rec.skip_mapper = False
            drop_packed(qc)

        lu.configure_quant_config = configure

        orig_map = SupportsQuant._maybe_apply_model_mapping

        def apply_model_mapping(model):
            skip = inj.no_mapper is not None and bool(
                inj.no_mapper.search(type(model).__name__)
            )
            rec.skip_mapper = skip
            try:
                orig_map(model)
            finally:
                rec.skip_mapper = False
            drop_packed(getattr(model, "quant_config", None))

        SupportsQuant._maybe_apply_model_mapping = apply_model_mapping

        if inj.requant is not None:
            import inspect

            from vllm.config import get_current_vllm_config
            from vllm.model_executor.layers.linear import LinearBase

            orig_init = LinearBase.__init__
            sig = inspect.signature(orig_init)

            @functools.wraps(orig_init)
            def init(self, *args, **kwargs):
                ba = sig.bind(self, *args, **kwargs)
                if ba.arguments.get("quant_config") is None and inj.requant.search(
                    ba.arguments.get("prefix", "")
                ):
                    ba.arguments["quant_config"] = (
                        get_current_vllm_config().quant_config
                    )
                    rec.events["requant"] += 1
                    return orig_init(*ba.args, **ba.kwargs)
                return orig_init(self, *args, **kwargs)

            LinearBase.__init__ = init

    # -- loading hooks ---------------------------------------------------------

    def instrument(self, model, role: str):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        owners: dict[int, list[str]] = defaultdict(list)
        params: dict[int, object] = {}
        for mname, mod in model.named_modules(remove_duplicate=False):
            for pname, p in mod.named_parameters(recurse=False):
                owners[id(p)].append(f"{mname}.{pname}" if mname else pname)
                params[id(p)] = p
        for pid, names in owners.items():
            p = params[pid]
            if getattr(p, "_qcheck_wrapped", False):
                continue
            orig = getattr(p, "weight_loader", None) or default_weight_loader
            p.weight_loader = self._wrap_loader(orig, role, tuple(names))
            p._qcheck_wrapped = True

    def _wrap_loader(self, orig, role: str, pnames: tuple[str, ...]):
        rec = self
        mods = tuple(split(n)[0] for n in pnames)

        @functools.wraps(orig)
        def loader(*args, **kwargs):
            w = args[1] if len(args) > 1 else kwargs.get("loaded_weight")
            tags = getattr(w, "_ckpt_tags", None)
            if not tags:
                rec.untagged[role] += 1
                tags = (rec.current,) if rec.current else ()
            for m in mods:
                rec.routes[role][m].update(tags)
            rec.hit[role].update(pnames)
            rec.routed[role].update(tags)
            try:
                return orig(*args, **kwargs)
            except Exception as e:
                first = str(e).splitlines()[0][:120] if str(e) else ""
                key = f"{type(e).__name__}: {first}"
                rec.loader_errors[key] += 1
                if len(rec.loader_error_samples) < 20:
                    rec.loader_error_samples.append(
                        f"{role}:{pnames[0]} <- {sorted(tags)[:2]} {key}"
                    )
                return True if kwargs.get("return_success") else None

        return loader

    def iterate(self, source):
        if source.subfolder:
            raise NotImplementedError("weight sources with a subfolder")
        hdr = fetch_header(
            source.model_or_path,
            pinned(source.model_or_path, source.revision),
            self.cache,
        )
        by_file: dict[str, list[str]] = defaultdict(list)
        for name, (_, _, f) in hdr.items():
            by_file[f].append(name)
        files = sorted(by_file)
        if source.allow_patterns_overrides:
            files = [
                f
                for f in files
                if any(fnmatch.fnmatch(f, p) for p in source.allow_patterns_overrides)
            ]
        unused = source.is_unused_weight
        if unused is not None:
            kept = [f for f in files if not all(unused(n) for n in by_file[f])]
            files = kept or files
        for f in files:
            for name in by_file[f]:
                dtype, shape, _ = hdr[name]
                full = source.prefix + name
                self.tensors[full] = [dtype, shape]
                self.yielded[self.role].add(full)
                self.current = full
                yield full, self.meta_tensor(full, dtype, shape)
        self.current = None


def register_loader(rec: Recorder):
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader(LOAD_FORMAT)
    class NamesOnlyLoader(DefaultModelLoader):
        def download_model(self, model_config) -> None:
            pass

        def _get_weights_iterator(self, source):
            return rec.iterate(source)

        def load_weights(self, model, model_config) -> None:
            rec.models[rec.role] = model
            rec.sources[rec.role] = (model_config.model, model_config.revision)
            rec.instrument(model, rec.role)
            super().load_weights(model, model_config)


def analyze(rec: Recorder, role: str) -> dict:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    model = rec.models[role]
    header = Header(rec.tensors)
    meta = load_meta(*rec.sources[role])
    # A module registered under several paths is graded once, under its
    # first path; any of its paths counts as a correct prefix.
    mods = dict(model.named_modules(remove_duplicate=False))
    quals: dict[int, list[str]] = defaultdict(list)
    for q, m in mods.items():
        quals[id(m)].append(q)

    def owner(qual: str) -> int | None:
        for q in ancestors(qual) if qual else [qual]:
            if hasattr(mods.get(q), "quant_method"):
                return id(mods[q])
        return None

    by_layer: dict[int, set[str]] = defaultdict(set)
    for qual, names in rec.routes[role].items():
        lid = owner(qual)
        if lid is not None:
            by_layer[lid] |= names

    entries = []
    for lid, names in sorted(by_layer.items(), key=lambda kv: quals[kv[0]][0]):
        weights = sorted(n for n in names if header.is_weight(n))
        if not weights:
            continue
        qual = quals[lid][0]
        mod = mods[qual]
        call = rec.calls.get(lid)
        vcls, dequant = vllm_class(getattr(mod, "quant_method", None))
        emb = isinstance(mod, VocabParallelEmbedding) and not isinstance(
            mod, ParallelLMHead
        )
        ckmods = sorted({split(n)[0] for n in weights})
        disks = sorted({header.disk_class(n) for n in weights})
        metas = sorted({meta.decide(m, emb) for m in ckmods}) if meta else ["?"]
        entries.append(
            {
                "role": role,
                "qual": qual,
                "type": type(mod).__name__,
                "method": type(getattr(mod, "quant_method", None)).__name__,
                "governed": call is not None,
                "prefix": call["seen"] if call else None,
                "vllm": vcls,
                "dequant": dequant,
                "meta": metas,
                "disk": disks,
                "ckpt_modules": sorted({collapse(m) for m in ckmods})[:8],
                "n_tensors": len(names),
                "verdict": verdict(vcls, call is not None, metas, disks, dequant),
            }
        )

    prefix_mismatch, orphans, unloaded = [], 0, []
    for lid, call in rec.calls.items():
        if call["role"] != role:
            continue
        if lid not in quals:
            orphans += 1
            continue
        q = quals[lid][0]
        if call["prefix"] not in quals[lid]:
            prefix_mismatch.append(
                {"qual": q, "prefix": call["prefix"], "layer": call["layer"]}
            )
        for pname, p in mods[q].named_parameters(recurse=False):
            full = f"{q}.{pname}" if q else pname
            if full not in rec.hit[role] and pname not in KV_PARAMS and p.numel() > 0:
                unloaded.append(full)

    routed_mods = {split(n)[0] for n in rec.routed[role]}
    dropped = Counter(
        collapse(n)
        for n in rec.yielded[role] - rec.routed[role]
        if split(n)[0] in routed_mods
    )
    unrouted = Counter(
        collapse(n)
        for n in rec.yielded[role] - rec.routed[role]
        if split(n)[0] not in routed_mods
    )
    return {
        "meta_source": meta.source if meta else None,
        "entries": entries,
        "prefix_mismatch": prefix_mismatch,
        "orphan_layers": orphans,
        "unloaded_params": sorted({collapse(u) for u in unloaded}),
        "dropped_siblings": dict(dropped.most_common(40)),
        "unrouted": dict(unrouted.most_common(40)),
        "n_yielded": len(rec.yielded[role]),
        "n_routed": len(rec.routed[role]),
        "untagged_calls": rec.untagged[role],
    }


def run_case(case: Case, args, res: dict, capture: LogCapture):
    import torch

    import vllm

    res["env"] = {
        "vllm": getattr(vllm, "__version__", "?"),
        "vllm_file": vllm.__file__,
        "torch": torch.__version__,
    }
    with contextlib.suppress(OSError):
        res["env"]["git_head"] = subprocess.run(
            ["git", "-C", str(Path(vllm.__file__).parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()

    import vllm.model_executor.model_loader.base_loader as base_loader
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs

    rec = Recorder(Injector(case.inject), Path(args.cache))
    register_loader(rec)
    base_loader.process_weights_after_loading = lambda *a, **k: None

    res["phase"] = "config"
    ea = EngineArgs(
        model=case.model,
        revision=case.revision,
        load_format=LOAD_FORMAT,
        enforce_eager=True,
        tensor_parallel_size=1,
        trust_remote_code=args.trust_remote_code,
        language_model_only=case.lm_only,
        speculative_config=case.spec,
        **case.engine_args,
    )
    vllm_config = ea.create_engine_config()
    vllm_config.load_config.device = "meta"

    rec.patch_quant_configs()
    rec.install_injectors()

    res["phase"] = "dist"
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.model_loader import get_model
    from vllm.platforms import current_platform

    fd, tmp = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp}",
            local_rank=0,
            backend=current_platform.dist_backend,
        )
        initialize_model_parallel(1, 1)

    res["load_exception"] = {}
    roles = ["target"]
    res["phase"] = "target"
    rec.role = "target"
    rec.patch_quant_configs()
    attach(capture)
    try:
        with set_current_vllm_config(vllm_config):
            get_model(vllm_config=vllm_config)
    except Exception:
        res["load_exception"]["target"] = traceback.format_exc()[-3000:]

    spec = vllm_config.speculative_config
    if spec is not None and getattr(spec, "draft_model_config", None) is not None:
        from vllm.compilation.backends import set_model_tag
        from vllm.config import replace
        from vllm.model_executor.model_loader.utils import get_draft_load_config

        res["phase"] = "draft"
        roles.append("draft")
        rec.role = "draft"
        rec.patch_quant_configs()
        attach(capture)
        try:
            dcfg = spec.apply_draft_overrides(vllm_config)
            dload = get_draft_load_config(dcfg)
            if dload is not dcfg.load_config:
                dcfg = replace(dcfg, load_config=dload)
            with set_current_vllm_config(dcfg), set_model_tag("eagle_head"):
                get_model(vllm_config=dcfg, model_config=spec.draft_model_config)
        except Exception:
            res["load_exception"]["draft"] = traceback.format_exc()[-3000:]

    res["phase"] = "analyze"
    missing = [r for r in roles if r not in rec.models]
    if missing:
        res["status"] = "construct_error"
        res["exception"] = "\n".join(res["load_exception"].get(r, "") for r in missing)
    res["roles"] = {}
    res["entries"] = []
    for r in roles:
        if r in rec.models:
            a = analyze(rec, r)
            res["entries"] += a.pop("entries")
            res["roles"][r] = a
    res["loader_errors"] = dict(rec.loader_errors.most_common(20))
    res["loader_error_samples"] = rec.loader_error_samples
    res["injector_events"] = dict(rec.events)
    res["verdicts"] = dict(Counter(e["verdict"] for e in res["entries"]))
    msgs = capture.messages + list(res["load_exception"].values())
    res["patch"] = grade_patch(res["entries"], msgs, args.patch_regex)
    res.setdefault("status", "load_error" if res["load_exception"] else "ok")


def cmd_worker(args) -> int:
    case = Case(**json.loads(args.case))
    case.inject, case.expect, case.allow = map(
        tuple, (case.inject, case.expect, case.allow)
    )
    capture = LogCapture()
    res: dict = {"case": case.id, "phase": "init"}
    try:
        run_case(case, args, res, capture)
    except BaseException:
        res["status"] = f"{res.get('phase')}_error"
        res["exception"] = traceback.format_exc()[-4000:]
    res["log_tail"] = capture.messages[-200:]
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: process-group shutdown can hang after a
    # partial load, and the result is already written.
    os._exit(0)


# ----------------------------------------------------------------------------
# Driver (no torch)
# ----------------------------------------------------------------------------


def select(pattern: str | None) -> list[Case]:
    return [c for c in CASES if not pattern or re.search(pattern, c.id)]


def bug_present(case: Case, repo: Path) -> bool | None:
    if case.inject or case.fixed_by is None:
        return True
    r = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", case.fixed_by, "HEAD"],
        capture_output=True,
    )
    return {0: False, 1: True}.get(r.returncode)


def cmd_list(args) -> int:
    for c in select(args.cases):
        opts = [
            "mtp" if c.spec else "",
            "lm-only" if c.lm_only else "",
            *c.inject,
        ]
        print(f"{c.id:40s} {c.model:60s} {' '.join(o for o in opts if o):30s} {c.ref}")
    return 0


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo)
    env = dict(os.environ)
    for kv in args.patch_env:
        k, _, v = kv.partition("=")
        env[k] = v
    rows, failed = [], False
    for c in select(args.cases):
        f = out / f"{c.id}.json"
        if not (args.resume and f.exists()):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--case",
                json.dumps(dataclasses.asdict(c)),
                "--out",
                str(f),
                "--cache",
                args.cache,
                "--patch-regex",
                args.patch_regex,
            ]
            if args.trust_remote_code:
                cmd.append("--trust-remote-code")
            try:
                p = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, timeout=args.timeout
                )
                (out / f"{c.id}.log").write_text(p.stdout + p.stderr)
                tail = p.stderr[-2000:]
            except subprocess.TimeoutExpired:
                tail = "timeout"
            if not f.exists():
                f.write_text(
                    json.dumps(
                        {"case": c.id, "status": "worker_crash", "exception": tail}
                    )
                )
        res = json.loads(f.read_text())
        present = bug_present(c, repo)
        problems, new = evaluate(c, res, present is not False)
        if present is None:
            problems.append(f"fixed_by {c.fixed_by} unknown in {repo}")
        if new and args.strict:
            problems.append(f"NEW {len(new)}")
        failed |= bool(problems)
        v = res.get("verdicts", {})
        pt = res.get("patch", {})
        row = {
            "case": c.id,
            "bug": {True: "yes", False: "fixed", None: "?"}[present],
            "status": res.get("status"),
            "errors": sum(v.get(k, 0) for k in ERROR_VERDICTS),
            "warns": sum(v.get(k, 0) for k in WARN_VERDICTS),
            "new": len(new),
            "prefix!=": sum(
                len(r["prefix_mismatch"]) for r in res.get("roles", {}).values()
            ),
            "patch": (
                f"{pt.get('caught', 0)}/{pt.get('errors', 0)} "
                f"fp={pt.get('n_flagged_ok', 0)}"
            ),
            "result": "FAIL" if problems else "ok",
            "problems": problems,
            "new_sample": new[:5],
        }
        rows.append(row)
        print(
            f"{row['result']:4s} {c.id:40s} bug={row['bug']:5s} {row['status']:16s} "
            f"err={row['errors']:<5d} warn={row['warns']:<5d} new={row['new']:<5d} "
            f"prefix!={row['prefix!=']:<4d} patch={row['patch']}",
            flush=True,
        )
        for p_ in problems:
            print(f"       - {p_}")
        for n in new[:3]:
            print(f"       + new: {n}")
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    return 1 if failed else 0


def cmd_offline(args) -> int:
    """Stored format vs metadata for every Linear-like checkpoint module."""
    models = sorted({c.model for c in select(args.cases)})
    skip = re.compile(r"(^|\.)(conv1d|conv|norm\w*|rotary\w*)$")
    linear_leaf = re.compile(r"^(weight|w\d+|\w+_proj)$")
    emb = re.compile(r"(embed_tokens|wte|word_embeddings|embeddings?)$")
    report = {}
    for m in models:
        rev = REV.get(m)
        header = Header(fetch_header(m, rev, Path(args.cache)))
        cfg = fetch_json(m, rev, "config.json")
        hfq = fetch_json(m, rev, "hf_quant_config.json")
        metas = {"active": QuantMeta.from_configs(cfg, hfq)}
        if (
            cfg
            and hfq
            and metas["active"]
            and metas["active"].source.startswith("config")
        ):
            metas["hf_quant_config.json"] = QuantMeta.from_configs(None, hfq)
        groups: Counter = Counter()
        src_diff: Counter = Counter()
        for name in header.t:
            mod, leaf = split(name)
            if not header.is_weight(name) or not linear_leaf.match(leaf):
                continue
            if skip.search(mod):
                continue
            disk = header.disk_class(name)
            e = bool(emb.search(mod))
            meta = metas["active"].decide(mod, e) if metas["active"] else "?"
            if meta not in ("?", disk):
                groups[(collapse(mod), meta, disk)] += 1
            other = metas.get("hf_quant_config.json")
            alt = other.decide(mod, e) if other is not None else "?"
            if "?" not in (meta, alt) and alt != meta:
                src_diff[(collapse(mod), meta, alt)] += 1
        report[m] = {
            "meta_source": metas["active"].source if metas["active"] else None,
            "n_tensors": len(header.t),
            "disk_formats": dict(
                Counter(header.disk_class(n) for n in header.t if header.is_weight(n))
            ),
            "meta_vs_disk": [
                {"module": k[0], "meta": k[1], "disk": k[2], "count": v}
                for k, v in sorted(groups.items())
            ],
            "config_vs_hf_quant_config": [
                {"module": k[0], "config": k[1], "hf_quant_config": k[2], "count": v}
                for k, v in sorted(src_diff.items())
            ],
        }
        r = report[m]
        print(
            f"== {m}  [{r['meta_source']}] tensors={r['n_tensors']} {r['disk_formats']}"
        )
        for g in r["meta_vs_disk"][: args.limit]:
            print(
                f"   meta={g['meta']:6s} disk={g['disk']:6s} "
                f"x{g['count']:<6d} {g['module']}"
            )
        if len(r["meta_vs_disk"]) > args.limit:
            print(f"   ... {len(r['meta_vs_disk']) - args.limit} more groups")
        for g in r["config_vs_hf_quant_config"][: args.limit]:
            print(
                f"   config={g['config']:6s} hf_quant_config={g['hf_quant_config']:6s} "
                f"x{g['count']:<6d} {g['module']}"
            )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


def cmd_selftest(args) -> int:
    meta = QuantMeta.from_configs(
        {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "ignore": ["lm_head", "model.layers.0.self_attn*", "mtp.*"],
            }
        },
        None,
    )
    assert meta.decide("lm_head") == "bf16"
    assert meta.decide("model.layers.0.self_attn.q_proj") == "bf16"
    assert meta.decide("model.layers.1.self_attn.q_proj") == "nvfp4"
    assert meta.decide("mtp.fc") == "bf16"
    assert meta.decide("model.embed_tokens", embedding=True) == "bf16"
    assert meta.decide("model.lm_head") == "nvfp4", "no substring matching"

    mixed = QuantMeta.from_configs(
        None,
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": [],
                "quantized_layers": {
                    "model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
                    "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                },
            }
        },
    )
    assert mixed.source == "hf_quant_config.json"
    assert mixed.decide("model.layers.0.mlp.experts.3.up_proj") == "nvfp4"
    assert mixed.decide("model.layers.0.self_attn.q_proj") == "fp8"
    assert mixed.decide("model.layers.0.self_attn.k_proj") == "bf16"

    dsv4 = QuantMeta.from_configs(
        {
            "quantization_config": {
                "quant_method": "fp8",
                "quant_algo": "MIXED_PRECISION",
                "ignore": ["*.attn.*"],
                "quantized_layers": {"layers.0.ffn.experts": {"quant_algo": "NVFP4"}},
            }
        },
        None,
    )
    assert dsv4.decide("layers.0.attn.wq_a") == "?"
    assert dsv4.decide("layers.0.ffn.experts.1.w1") == "nvfp4"

    h = Header(
        {
            "a.weight": ["BF16", [8, 8], "f"],
            "b.weight": ["F8_E4M3", [256, 256], "f"],
            "b.weight_scale": ["F32", [], "f"],
            "c.weight": ["F8_E4M3", [256, 256], "f"],
            "c.scale": ["F8_E8M0", [2, 2], "f"],
            "d.weight": ["F8_E4M3", [64, 256], "f"],
            "d.weight_scale": ["U8", [64, 8], "f"],
            "e.weight": ["U8", [64, 128], "f"],
            "e.weight_scale": ["F8_E4M3", [64, 16], "f"],
            "e.weight_scale_2": ["F32", [], "f"],
            "e.input_scale": ["F32", [], "f"],
            "f.weight": ["U8", [64, 128], "f"],
            "f.weight_scale": ["U8", [64, 8], "f"],
            "g.experts.gate_up_proj": ["U8", [4, 64, 128], "f"],
            "g.experts.gate_up_proj_scale": ["F8_E4M3", [4, 64, 16], "f"],
            "h.weight": ["U8", [64, 128], "f"],
            "h.weight_scale.csf32_codes": ["U8", [4096], "f"],
            "h.weight_scale_2": ["F32", [], "f"],
            "n.weight": ["BF16", [8], "f"],
        }
    )
    got = {n: h.disk_class(n) for n in h.t if h.is_weight(n)}
    assert got == {
        "a.weight": "bf16",
        "b.weight": "fp8",
        "c.weight": "fp8",
        "d.weight": "mxfp8",
        "e.weight": "nvfp4",
        "f.weight": "mxfp4",
        "g.experts.gate_up_proj": "nvfp4",
        "h.weight": "nvfp4",
    }, got

    assert verdict("nvfp4", True, ["nvfp4"], ["nvfp4"], False) == "OK"
    assert verdict("nvfp4", True, ["bf16"], ["bf16"], False) == "LOOKUP_BUG"
    assert verdict("nvfp4", True, ["nvfp4"], ["bf16"], False) == "META_BUG"
    assert verdict("bf16", True, ["nvfp4"], ["bf16"], False) == "META_STALE"
    assert verdict("bf16", False, ["nvfp4"], ["bf16"], False) == "OK"
    assert verdict("bf16", False, ["?"], ["mxfp8"], False) == "HARDCODED"
    assert verdict("fp8", True, ["fp8"], ["bf16"], True) == "OK"
    assert verdict("fp8", True, ["?"], ["fp8"], False) == "OK"
    assert verdict("nvfp4", True, ["nvfp4"], ["bf16", "nvfp4"], False) == "DISK_MIXED"
    assert verdict("nvfp4", True, ["fp8"], ["bf16"], False) == "MISMATCH"

    assert names_layer("model.layers.1.mlp", "bad layer model.layers.1.mlp.")
    assert not names_layer("model.layers.1.mlp", "model.layers.11.mlp")
    assert not names_layer("model.layers.1.mlp", "model.layers.1.mlp.experts")

    case = Case("x", "m", expect=(r"^target:lm_head$",), allow=(r"^draft:",))
    res = {
        "entries": [
            {"role": "target", "qual": "lm_head", "verdict": "LOOKUP_BUG"},
            {"role": "draft", "qual": "model.fc", "verdict": "META_BUG"},
            {"role": "target", "qual": "model.layers.0.mlp", "verdict": "HARDCODED"},
            {"role": "target", "qual": "model.layers.1.mlp", "verdict": "OK"},
        ]
    }
    assert evaluate(case, res, True) == ([], ["target:model.layers.0.mlp"])
    assert evaluate(case, res, False)[0][0].startswith("STILL_FLAGGED")
    res["entries"][0]["verdict"] = "OK"
    assert evaluate(case, res, True)[0] == ["MISSED ^target:lm_head$"]
    g = grade_patch(res["entries"], ["WARNING quant mismatch at model.fc"], r"mismatch")
    assert (g["caught"], g["errors"], g["n_flagged_ok"]) == (1, 2, 0), g

    inj = Injector((r"prefix:^.*\.shared_head\.head$=>", "no_mapper:^Foo$"))
    assert inj.rewrite("model.layers.45.shared_head.head") == ""
    assert inj.rewrite("model.layers.45.shared_head") == "model.layers.45.shared_head"
    assert (
        collapse("model.layers.12.mlp.experts.3.w1")
        == "model.layers.N.mlp.experts.N.w1"
    )
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    print(f"selftest ok ({len(CASES)} cases)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    cache_default = str(Path.home() / ".cache" / "vllm_qcheck")

    p = sub.add_parser("list")
    p.add_argument("--cases")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("offline")
    p.add_argument("--cases")
    p.add_argument("--cache", default=cache_default)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_offline)

    p = sub.add_parser("run")
    p.add_argument("--cases", help="regex over case ids")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=cache_default)
    p.add_argument("--repo", default=str(Path(__file__).resolve().parent))
    p.add_argument("--patch-env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--patch-regex", default=DEFAULT_PATCH_REGEX)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--strict", action="store_true", help="fail on new error verdicts")
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("worker")
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--patch-regex", required=True)
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
