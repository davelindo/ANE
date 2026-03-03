#!/usr/bin/env python3
"""Export nomic-embed-text-v1.5 to static ONNX for ANE compatibility testing.

This script removes dynamic RoPE table generation during export by patching each
rotary embedding module to use precomputed cos/sin buffers for a fixed sequence
length.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Iterable

_THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)


def ensure_repo_local_cache(repo_root: Path) -> None:
    cache_root = repo_root / ".local" / "cache"
    hf_home = cache_root / "hf"
    tmp_dir = cache_root / "tmp"
    for p in (cache_root, hf_home, tmp_dir):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("TMPDIR", str(tmp_dir))


def rotate_half_tensor(x, interleaved: bool):
    import torch

    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_precomputed(x, cos, sin, interleaved: bool):
    ro_dim = int(cos.shape[-1])
    if ro_dim <= 0:
        return x
    if ro_dim > int(x.shape[-1]):
        raise ValueError(f"Invalid ro_dim={ro_dim} for x.shape={tuple(x.shape)}")

    x_ro = x[..., :ro_dim]
    rotated = x_ro * cos + rotate_half_tensor(x_ro, interleaved) * sin

    if ro_dim == int(x.shape[-1]):
        return rotated
    return __import__("torch").cat((rotated, x[..., ro_dim:]), dim=-1)


def build_static_rope_tables(rotary_dim: int, base: float, seq_len: int):
    import torch

    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / float(rotary_dim))
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    cos = torch.repeat_interleave(cos_half, repeats=2, dim=-1).unsqueeze(1)
    sin = torch.repeat_interleave(sin_half, repeats=2, dim=-1).unsqueeze(1)
    return cos, sin


def patch_rotary_modules_for_static_export(model, seq_len: int) -> int:
    import torch

    patched = 0
    rotary_names = {"NomicBertRotaryEmbedding", "NomicBertDynamicNTKRotaryEmbedding"}

    for mod in model.modules():
        if type(mod).__name__ not in rotary_names:
            continue

        rotary_dim = int(round(float(getattr(mod, "dim", 0))))
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            continue

        base = float(getattr(mod, "base", 10000.0))
        interleaved = bool(getattr(mod, "interleaved", False))
        cos, sin = build_static_rope_tables(rotary_dim, base, seq_len)

        mod.register_buffer("_static_cos_export", cos, persistent=False)
        mod.register_buffer("_static_sin_export", sin, persistent=False)

        def _forward_static(self, qkv, kv=None, seqlen_offset=0, max_seqlen=None):
            seqlen = int(qkv.shape[1])
            if isinstance(seqlen_offset, torch.Tensor):
                if seqlen_offset.numel() != 1:
                    raise ValueError(
                        "Tensor seqlen_offset with numel != 1 is not supported"
                    )
                offset = int(seqlen_offset.item())
            else:
                offset = int(seqlen_offset)

            cos_local = self._static_cos_export[offset : offset + seqlen].to(
                device=qkv.device, dtype=qkv.dtype
            )
            sin_local = self._static_sin_export[offset : offset + seqlen].to(
                device=qkv.device, dtype=qkv.dtype
            )

            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            v = qkv[:, :, 2]

            q_rot = apply_rotary_precomputed(q, cos_local, sin_local, interleaved)
            k_rot = apply_rotary_precomputed(k, cos_local, sin_local, interleaved)
            return torch.stack((q_rot, k_rot, v), dim=2)

        mod.forward = types.MethodType(_forward_static, mod)
        patched += 1

    return patched


class EncoderWrapper:
    def __init__(self, model, output_kind: str, matryoshka_dim: int):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, model, output_kind: str, matryoshka_dim: int) -> None:
                super().__init__()
                self.model = model
                self.output_kind = output_kind
                self.matryoshka_dim = matryoshka_dim

            def forward(  # type: ignore[override]
                self,
                input_ids,
                attention_mask,
                token_type_ids,
                pool_mask=None,
            ):
                out = self.model(
                    input_ids=input_ids.long(),
                    attention_mask=attention_mask.long(),
                    token_type_ids=token_type_ids.long(),
                    return_dict=True,
                ).last_hidden_state
                if self.output_kind == "hidden":
                    return out

                if pool_mask is None:
                    pool_mask = attention_mask.to(dtype=out.dtype)
                mask = pool_mask.unsqueeze(-1)
                summed = (out * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1e-9)
                embeddings = summed / denom
                if 0 < self.matryoshka_dim < embeddings.shape[-1]:
                    embeddings = embeddings[:, : self.matryoshka_dim]
                squared = embeddings * embeddings
                norms = squared.sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
                return embeddings / norms

        self.module = _Wrapper(model, output_kind, matryoshka_dim)


def tokenize_example(tokenizer, seq_len: int, output_kind: str):
    enc = tokenizer(
        ["search_query: hello world"],
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    if "token_type_ids" not in enc:
        import torch

        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    if output_kind == "embedding":
        import torch

        enc["pool_mask"] = enc["attention_mask"].to(dtype=torch.float16)
    return enc


def export_static_onnx(
    model_id: str,
    revision: str | None,
    output_path: Path,
    seq_len: int,
    opset: int,
    output_kind: str,
    matryoshka_dim: int,
) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_flash_attn"):
        model.config.use_flash_attn = False
    model.eval()

    patched_count = patch_rotary_modules_for_static_export(model, seq_len)

    wrapper = EncoderWrapper(model, output_kind, matryoshka_dim).module.eval()
    example = tokenize_example(tokenizer, seq_len, output_kind)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_name = "last_hidden_state" if output_kind == "hidden" else "embedding"
    args = (
        example["input_ids"],
        example["attention_mask"],
        example["token_type_ids"],
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    if output_kind == "embedding":
        args = args + (example["pool_mask"],)
        input_names.append("pool_mask")

    torch.onnx.export(
        wrapper,
        args,
        str(output_path),
        export_params=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=[output_name],
        dynamic_axes=None,
        opset_version=opset,
    )

    return {
        "output_path": str(output_path),
        "patched_rotary_modules": patched_count,
        "seq_len": seq_len,
        "opset": opset,
        "output_kind": output_kind,
        "matryoshka_dim": matryoshka_dim,
    }


def _iter_value_infos(model) -> Iterable:
    yield from model.graph.input
    yield from model.graph.output
    yield from model.graph.value_info


def find_zero_dim_shapes(model) -> list[dict]:
    issues: list[dict] = []
    for vi in _iter_value_infos(model):
        tensor_type = vi.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        has_zero = False
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                v = int(dim.dim_value)
                dims.append(v)
                if v == 0:
                    has_zero = True
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        if has_zero:
            issues.append({"name": vi.name, "shape": dims})
    return issues


def rewrite_coreml_incompatible_ops(model) -> dict:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    init_by_name = {init.name: init for init in graph.initializer}
    scalar_const_by_output: dict[str, int] = {}
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            arr = numpy_helper.to_array(attr.t)
            if arr.ndim == 0:
                scalar_const_by_output[node.output[0]] = int(arr.item())
            break

    used_names: set[str] = set()
    for init in graph.initializer:
        used_names.add(init.name)
    for vi in _iter_value_infos(model):
        used_names.add(vi.name)
    for node in graph.node:
        used_names.add(node.name)
        for x in node.input:
            used_names.add(x)
        for x in node.output:
            used_names.add(x)

    uniq_counter = 0

    def uniq(prefix: str) -> str:
        nonlocal uniq_counter
        base = prefix or "tmp"
        name = base
        while name in used_names:
            uniq_counter += 1
            name = f"{base}_{uniq_counter}"
        used_names.add(name)
        return name

    def add_i64_initializer(name: str, values: list[int]) -> None:
        arr = np.asarray(values, dtype=np.int64)
        graph.initializer.append(numpy_helper.from_array(arr, name=name))

    dtype_by_name: dict[str, int] = {}
    for init in graph.initializer:
        dtype_by_name[init.name] = int(init.data_type)
    for vi in _iter_value_infos(model):
        tt = vi.type.tensor_type
        if tt and tt.HasField("elem_type"):
            dtype_by_name[vi.name] = int(tt.elem_type)

    zero_cache: dict[int, str] = {}

    def zero_const_for_dtype(dtype_enum: int) -> str:
        name = zero_cache.get(dtype_enum)
        if name is not None:
            return name
        name = uniq(f"__const_zero_{dtype_enum}")
        np_dtype = {
            TensorProto.FLOAT16: np.float16,
            TensorProto.FLOAT: np.float32,
            TensorProto.DOUBLE: np.float64,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
        }.get(dtype_enum, np.float32)
        graph.initializer.append(
            numpy_helper.from_array(np.asarray([0], dtype=np_dtype), name=name)
        )
        zero_cache[dtype_enum] = name
        return name

    rewritten_nodes = []
    replaced_scalar_gather = 0
    replaced_neg = 0
    skipped_non_scalar_gather = 0

    for node in graph.node:
        if node.op_type == "Gather" and len(node.input) >= 2:
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = int(attr.i)

            idx_name = node.input[1]
            idx_init = init_by_name.get(idx_name)
            idx_value = None
            if idx_init is not None:
                idx_arr = numpy_helper.to_array(idx_init)
                if idx_arr.ndim == 0:
                    idx_value = int(idx_arr.item())
            elif idx_name in scalar_const_by_output:
                idx_value = int(scalar_const_by_output[idx_name])

            if idx_value is not None:
                # CoreML EP rejects scalar-index Gather. Rewrite it to Slice+Squeeze.
                data_input = node.input[0]
                slice_out = uniq((node.output[0] or node.name or "gather") + "__slice")
                starts_name = uniq((node.name or "gather") + "__starts")
                ends_name = uniq((node.name or "gather") + "__ends")
                axes_name = uniq((node.name or "gather") + "__axes")
                steps_name = uniq((node.name or "gather") + "__steps")
                squeeze_axes_name = uniq((node.name or "gather") + "__squeeze_axes")

                add_i64_initializer(starts_name, [idx_value])
                add_i64_initializer(ends_name, [idx_value + 1])
                add_i64_initializer(axes_name, [axis])
                add_i64_initializer(steps_name, [1])
                add_i64_initializer(squeeze_axes_name, [axis])

                rewritten_nodes.append(
                    helper.make_node(
                        "Slice",
                        inputs=[
                            data_input,
                            starts_name,
                            ends_name,
                            axes_name,
                            steps_name,
                        ],
                        outputs=[slice_out],
                        name=(node.name or "Gather") + "__slice",
                    )
                )
                rewritten_nodes.append(
                    helper.make_node(
                        "Squeeze",
                        inputs=[slice_out, squeeze_axes_name],
                        outputs=list(node.output),
                        name=(node.name or "Gather") + "__squeeze",
                    )
                )
                replaced_scalar_gather += 1
                continue

            skipped_non_scalar_gather += 1

        if node.op_type == "Neg" and len(node.input) == 1:
            x = node.input[0]
            dtype_enum = int(dtype_by_name.get(x, TensorProto.FLOAT))
            zero_name = zero_const_for_dtype(dtype_enum)
            rewritten_nodes.append(
                helper.make_node(
                    "Sub",
                    inputs=[zero_name, x],
                    outputs=list(node.output),
                    name=(node.name or "Neg") + "__sub_zero",
                )
            )
            replaced_neg += 1
            continue

        rewritten_nodes.append(node)

    if replaced_scalar_gather > 0 or replaced_neg > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)

    return {
        "replaced_scalar_gather": replaced_scalar_gather,
        "replaced_neg": replaced_neg,
        "skipped_non_scalar_gather": skipped_non_scalar_gather,
    }


def sanitize_onnx(
    input_path: Path,
    output_path: Path,
    seq_len: int,
    run_simplifier: bool,
) -> dict:
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(input_path))
    rewrite_info = rewrite_coreml_incompatible_ops(model)
    onnx.checker.check_model(model)

    if run_simplifier:
        try:
            import onnxsim

            input_shapes = {
                "input_ids": [1, seq_len],
                "attention_mask": [1, seq_len],
                "token_type_ids": [1, seq_len],
            }
            if any(inp.name == "pool_mask" for inp in model.graph.input):
                input_shapes["pool_mask"] = [1, seq_len]

            try:
                model, ok = onnxsim.simplify(
                    model,
                    overwrite_input_shapes=input_shapes,
                )
            except TypeError:
                model, ok = onnxsim.simplify(model)
            if not ok:
                raise RuntimeError("onnxsim.simplify reported failure")
        except ImportError:
            pass

    model = shape_inference.infer_shapes(model)
    zero_issues = find_zero_dim_shapes(model)
    if zero_issues:
        raise RuntimeError(
            "Zero-dimension tensors remain after sanitation: "
            + json.dumps(zero_issues[:20], indent=2)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return {
        "output_path": str(output_path),
        "zero_dim_issues": len(zero_issues),
        "rewrite": rewrite_info,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export static ONNX for nomic-embed-text-v1.5"
    )
    p.add_argument("--model-id", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--revision", default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(".local/models/nomic_embed_text_v1_5/static/hidden/b1_s512.onnx"),
    )
    p.add_argument(
        "--raw-output",
        type=Path,
        default=Path(
            ".local/models/nomic_embed_text_v1_5/static/hidden/b1_s512.raw.onnx"
        ),
    )
    p.add_argument("--skip-simplify", action="store_true")
    p.add_argument(
        "--output-kind",
        default="hidden",
        choices=("hidden", "embedding"),
        help="Export raw hidden states or pooled normalized embeddings.",
    )
    p.add_argument(
        "--matryoshka-dim",
        type=int,
        default=0,
        help="Optional truncation dim applied only when --output-kind embedding.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary",
    )
    return p.parse_args()


def infer_repo_root() -> Path:
    # .../packages/ane-infer/src/ane/infer/nomic/export_static.py -> repo root
    return Path(__file__).resolve().parents[6]


def main() -> int:
    args = parse_args()
    repo_root = infer_repo_root()
    ensure_repo_local_cache(repo_root)

    out = args.output if args.output.is_absolute() else (repo_root / args.output)
    raw = (
        args.raw_output
        if args.raw_output.is_absolute()
        else (repo_root / args.raw_output)
    )

    export_info = export_static_onnx(
        model_id=args.model_id,
        revision=args.revision,
        output_path=raw,
        seq_len=args.seq_len,
        opset=args.opset,
        output_kind=args.output_kind,
        matryoshka_dim=args.matryoshka_dim,
    )

    sanitize_info = sanitize_onnx(
        input_path=raw,
        output_path=out,
        seq_len=args.seq_len,
        run_simplifier=not args.skip_simplify,
    )

    summary = {
        "status": "ok",
        "model_id": args.model_id,
        "revision": args.revision,
        "seq_len": args.seq_len,
        "opset": args.opset,
        "patched_rotary_modules": export_info["patched_rotary_modules"],
        "output_kind": export_info["output_kind"],
        "matryoshka_dim": export_info["matryoshka_dim"],
        "raw_output": str(raw),
        "output": str(out),
        "sanitized": sanitize_info,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Static ONNX export complete: {out}")
        print(f"Patched rotary modules: {export_info['patched_rotary_modules']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
