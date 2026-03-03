#!/usr/bin/env python3
"""Run nomic-embed-text-v1.5 on Apple Neural Engine paths.

Default backend uses ONNX Runtime + CoreMLExecutionProvider with
CPUAndNeuralEngine compute units. A direct coremltools conversion backend is
kept as an optional path for experimentation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional

# Avoid shadowing stdlib `tokenize` with local repo scripts.
_THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)

import numpy as np

DEFAULT_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_PREFIXED_TEXTS = (
    "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten.",
    "search_query: Who created TSNE?",
)
DEFAULT_MODEL_STORE = Path(".local/models/nomic_embed_text_v1_5")
DEFAULT_ARTIFACT = DEFAULT_MODEL_STORE / "base" / "fp16.onnx"
DEFAULT_STATIC_ARTIFACT = DEFAULT_MODEL_STORE / "static" / "hidden" / "b1_s512.onnx"
DEFAULT_STATIC_RAW_ARTIFACT = (
    DEFAULT_MODEL_STORE / "static" / "hidden" / "b1_s512.raw.onnx"
)
DEFAULT_STATIC_OUTPUT_KIND = "hidden"
DEFAULT_STATIC_OUTPUT_MATRYOSHKA_DIM = 0
DEFAULT_REPORT_DIR = Path(".local/runs/nomic_ane")
DEFAULT_BENCH_BATCH_SIZE = 8
DEFAULT_BENCH_ITERS = 30
DEFAULT_BENCH_WARMUP = 5
DEFAULT_GATE_MIN_COVERAGE_PCT = 90.0
DEFAULT_GATE_MIN_SPEEDUP = 1.7
DEFAULT_QUALITY_TOPK = 5
DEFAULT_QUALITY_MIN_SPEARMAN = 0.99
DEFAULT_QUALITY_MAX_AUC_DROP = 0.02
DEFAULT_STSB_SPLIT = "validation"
DEFAULT_STSB_MAX_SAMPLES = 0
DEFAULT_STSB_MIN_CPU_SPEARMAN = 0.75
DEFAULT_STSB_MAX_SPEARMAN_DROP = 0.01
DEFAULT_STSB_MAX_MEAN_ABS_DELTA = 0.005

# (text_a, text_b, similar_label)
QUALITY_PAIRS = (
    (
        "search_query: how to bake sourdough bread",
        "search_document: a step-by-step guide to baking sourdough bread at home",
        1,
    ),
    (
        "search_query: flu symptoms",
        "search_document: common influenza symptoms include fever, cough, and fatigue",
        1,
    ),
    (
        "search_query: python list comprehension examples",
        "search_document: tutorial on python list comprehensions with examples",
        1,
    ),
    (
        "search_query: employee annual leave policy",
        "search_document: company vacation and paid time off policy for employees",
        1,
    ),
    (
        "search_query: fix c++ memory leak",
        "search_document: debugging and fixing memory leaks in c++ applications",
        1,
    ),
    (
        "search_query: best trails in yosemite",
        "search_document: top hiking trails in yosemite national park",
        1,
    ),
    (
        "search_query: convert pdf to text",
        "search_document: extracting plain text from pdf files",
        1,
    ),
    (
        "search_query: marathon training plan beginner",
        "search_document: beginner marathon training schedule over sixteen weeks",
        1,
    ),
    (
        "search_query: nearby ev charging station",
        "search_document: finding an electric vehicle charging station near your location",
        1,
    ),
    (
        "search_query: reduce overfitting neural network",
        "search_document: techniques to prevent overfitting in neural networks",
        1,
    ),
    (
        "search_query: order refund status",
        "search_document: how to check refund status for an online order",
        1,
    ),
    (
        "search_query: grow tomatoes in pots",
        "search_document: growing tomato plants in containers on a balcony",
        1,
    ),
    (
        "search_query: how to bake sourdough bread",
        "search_document: federal reserve interest rate decision summary",
        0,
    ),
    (
        "search_query: flu symptoms",
        "search_document: installing nvidia gpu drivers on ubuntu linux",
        0,
    ),
    (
        "search_query: python list comprehension examples",
        "search_document: history of roman aqueduct architecture",
        0,
    ),
    (
        "search_query: employee annual leave policy",
        "search_document: dark chocolate cake recipe with ganache frosting",
        0,
    ),
    (
        "search_query: fix c++ memory leak",
        "search_document: best beaches to visit in thailand",
        0,
    ),
    (
        "search_query: best trails in yosemite",
        "search_document: beginner guide to sql joins and foreign keys",
        0,
    ),
    (
        "search_query: convert pdf to text",
        "search_document: dog vaccination schedule by age",
        0,
    ),
    (
        "search_query: marathon training plan beginner",
        "search_document: cloud security best practices for startups",
        0,
    ),
    (
        "search_query: nearby ev charging station",
        "search_document: how to knit a wool scarf",
        0,
    ),
    (
        "search_query: reduce overfitting neural network",
        "search_document: step-by-step car engine oil change guide",
        0,
    ),
    (
        "search_query: order refund status",
        "search_document: mindfulness breathing exercises for stress",
        0,
    ),
    (
        "search_query: grow tomatoes in pots",
        "search_document: mortgage refinance rates and lender comparison",
        0,
    ),
)


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


def tokenize_fixed(
    tokenizer: object,
    texts: Iterable[str],
    seq_len: int,
    dtype: np.dtype = np.int64,
) -> Dict[str, np.ndarray]:
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = np.zeros_like(encoded["input_ids"])
    return {
        "input_ids": encoded["input_ids"].astype(dtype),
        "attention_mask": encoded["attention_mask"].astype(dtype),
        "token_type_ids": token_type_ids.astype(dtype),
        "pool_mask": encoded["attention_mask"].astype(np.float16),
    }


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    denom = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / denom


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def build_benchmark_texts(batch_size: int) -> list[str]:
    return [f"search_query: benchmark sample {i}" for i in range(batch_size)]


def write_json(path: Optional[Path], payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------- ONNX Runtime + CoreML EP path ----------


def prepare_onnx_artifact(
    model_id: str,
    revision: Optional[str],
    onnx_filename: str,
    artifact_path: Path,
) -> None:
    from huggingface_hub import hf_hub_download

    src = Path(
        hf_hub_download(
            repo_id=model_id,
            filename=onnx_filename,
            revision=revision,
        )
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != artifact_path.resolve():
        shutil.copy2(src, artifact_path)
    print(f"Prepared ONNX artifact: {artifact_path}")


def _compute_unit_to_coreml_ep(value: str) -> str:
    mapping = {
        "cpu_only": "CPUOnly",
        "cpu_and_gpu": "CPUAndGPU",
        "cpu_and_ne": "CPUAndNeuralEngine",
        "all": "ALL",
    }
    if value not in mapping:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported compute unit '{value}'. Use one of: {allowed}")
    return mapping[value]


def _coreml_provider_with_options(compute_unit_name: str):
    return (
        "CoreMLExecutionProvider",
        {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": _compute_unit_to_coreml_ep(compute_unit_name),
        },
    )


def make_onnx_session(
    artifact_path: Path,
    providers: list[object],
    require_full_coreml: bool = False,
    disable_graph_optimizations: bool = False,
):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    if require_full_coreml:
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if disable_graph_optimizations:
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
    return ort.InferenceSession(
        str(artifact_path),
        sess_options=sess_options,
        providers=providers,
    )


def run_embedding_inference(
    session,
    feed: Dict[str, np.ndarray],
    attention_mask: np.ndarray,
    matryoshka_dim: int,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    outputs = session.run(None, feed)
    if not outputs:
        raise RuntimeError("ONNX runtime returned no outputs")
    primary = outputs[0]
    hidden: Optional[np.ndarray] = None

    if primary.ndim == 3:
        hidden = primary
        embeddings = mean_pool(hidden, attention_mask)
    elif primary.ndim == 2:
        embeddings = primary
    else:
        raise RuntimeError(f"Unexpected model output rank: {primary.shape}")

    if 0 < matryoshka_dim < embeddings.shape[1]:
        embeddings = embeddings[:, :matryoshka_dim]
    embeddings = l2_normalize(embeddings)
    return hidden, embeddings


def smoke_test_onnx_coreml(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
    matryoshka_dim: int,
    compute_unit_name: str,
    require_full_coreml: bool,
) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    providers = ort.get_available_providers()
    if "CoreMLExecutionProvider" not in providers:
        raise RuntimeError(
            "onnxruntime does not expose CoreMLExecutionProvider in this environment. "
            f"Available providers: {providers}"
        )

    session = make_onnx_session(
        artifact_path=artifact_path,
        providers=[_coreml_provider_with_options(compute_unit_name)],
        require_full_coreml=require_full_coreml,
    )

    fixed_batch = infer_fixed_batch_size(session)
    if fixed_batch is None:
        texts = list(DEFAULT_PREFIXED_TEXTS)
    elif fixed_batch <= len(DEFAULT_PREFIXED_TEXTS):
        texts = list(DEFAULT_PREFIXED_TEXTS[:fixed_batch])
    else:
        texts = build_benchmark_texts(batch_size=fixed_batch)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    tokenized = tokenize_fixed(tokenizer, texts, seq_len, dtype=np.int64)

    expected_inputs = {inp.name for inp in session.get_inputs()}
    feed: Dict[str, np.ndarray] = {}
    for name in expected_inputs:
        if name in tokenized:
            feed[name] = tokenized[name]
        else:
            raise KeyError(f"ONNX session requires unknown input '{name}'")

    hidden, embeddings = run_embedding_inference(
        session,
        feed,
        tokenized["attention_mask"],
        matryoshka_dim,
    )

    print("Backend: onnx-coreml")
    print(f"Providers used: {session.get_providers()}")
    if (not require_full_coreml) and "CPUExecutionProvider" in session.get_providers():
        print(
            "WARNING: CPUExecutionProvider is present; graph is partially partitioned."
        )
    print(f"Input shape: input_ids={tokenized['input_ids'].shape}")
    if hidden is not None:
        print(f"Hidden shape: {hidden.shape}")
    else:
        print("Hidden shape: (not returned by model; embedding output artifact)")
    print(f"Embedding shape: {embeddings.shape}")
    if embeddings.shape[0] >= 2:
        cosine = float(np.dot(embeddings[0], embeddings[1]))
        print(f"Cosine(doc, query): {cosine:.6f}")
    print("First embedding[0:8]:", np.array2string(embeddings[0, :8], precision=5))


def parse_coreml_coverage_from_text(text: str) -> dict:
    out: dict = {}
    ratio = re.search(
        r"supports\s*\[(\d+)\s*/\s*(\d+)\]\s*nodes", text, flags=re.IGNORECASE
    )
    if ratio:
        supported = int(ratio.group(1))
        total = int(ratio.group(2))
        pct = 100.0 * supported / total if total else 0.0
        out.update(
            {
                "supported_nodes": supported,
                "total_nodes": total,
                "coverage_pct": pct,
            }
        )
    else:
        alt = re.search(
            r"number of nodes in the graph:\s*(\d+)\s*number of nodes supported by CoreML:\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if alt:
            total = int(alt.group(1))
            supported = int(alt.group(2))
            pct = 100.0 * supported / total if total else 0.0
            out.update(
                {
                    "supported_nodes": supported,
                    "total_nodes": total,
                    "coverage_pct": pct,
                }
            )
    parts = re.search(
        r"number of partitions supported by CoreML:\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if parts:
        out["coreml_partitions"] = int(parts.group(1))
    return out


def probe_coreml_coverage_subprocess(
    artifact_path: Path, compute_unit_name: str
) -> dict:
    compute_unit = _compute_unit_to_coreml_ep(compute_unit_name)
    snippet = (
        "import onnxruntime as ort, sys\n"
        "artifact=sys.argv[1]\n"
        "compute_unit=sys.argv[2]\n"
        "so=ort.SessionOptions()\n"
        "so.add_session_config_entry('session.disable_cpu_ep_fallback','1')\n"
        "providers=[('CoreMLExecutionProvider',{'ModelFormat':'MLProgram','MLComputeUnits':compute_unit})]\n"
        "try:\n"
        "    ort.InferenceSession(artifact, sess_options=so, providers=providers)\n"
        "    print('STRICT_SESSION_OK')\n"
        "except Exception as exc:\n"
        "    print(f'STRICT_SESSION_ERR: {exc}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet, str(artifact_path), compute_unit],
        capture_output=True,
        text=True,
        check=False,
    )
    merged = f"{proc.stdout}\n{proc.stderr}"
    out = parse_coreml_coverage_from_text(merged)
    out["subprocess_returncode"] = proc.returncode
    return out


def probe_coreml_strict_coverage(artifact_path: Path, compute_unit_name: str) -> dict:
    result: dict = {"strict_session_ok": False}
    try:
        _ = make_onnx_session(
            artifact_path=artifact_path,
            providers=[_coreml_provider_with_options(compute_unit_name)],
            require_full_coreml=True,
        )
        result["strict_session_ok"] = True
        result["supported_nodes"] = 1
        result["total_nodes"] = 1
        result["coverage_pct"] = 100.0
        result["coreml_partitions"] = 1
        return result
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        result["error"] = msg
        parsed = parse_coreml_coverage_from_text(msg)
        if "coverage_pct" not in parsed:
            parsed.update(
                probe_coreml_coverage_subprocess(
                    artifact_path=artifact_path,
                    compute_unit_name=compute_unit_name,
                )
            )
        result.update(parsed)
        return result


def create_cpu_session_with_fallback(artifact_path: Path):
    try:
        session = make_onnx_session(
            artifact_path=artifact_path,
            providers=["CPUExecutionProvider"],
        )
        return session, "default"
    except Exception:
        session = make_onnx_session(
            artifact_path=artifact_path,
            providers=["CPUExecutionProvider"],
            disable_graph_optimizations=True,
        )
        return session, "disable_all"


def infer_fixed_batch_size(session) -> Optional[int]:
    for inp in session.get_inputs():
        shape = getattr(inp, "shape", None)
        if not shape:
            continue
        dim0 = shape[0]
        if isinstance(dim0, int) and dim0 > 0:
            return dim0
    return None


def benchmark_session(
    session,
    feed: Dict[str, np.ndarray],
    attention_mask: np.ndarray,
    matryoshka_dim: int,
    warmup: int,
    iters: int,
) -> dict:
    for _ in range(warmup):
        run_embedding_inference(session, feed, attention_mask, matryoshka_dim)

    timings_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_embedding_inference(session, feed, attention_mask, matryoshka_dim)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    return {
        "iters": iters,
        "warmup": warmup,
        "mean_ms": float(statistics.fmean(timings_ms)),
        "median_ms": float(statistics.median(timings_ms)),
        "min_ms": float(min(timings_ms)),
        "max_ms": float(max(timings_ms)),
        "all_ms": timings_ms,
    }


def benchmark_hybrid_vs_cpu(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
    matryoshka_dim: int,
    compute_unit_name: str,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    cpu_session, cpu_opt_mode = create_cpu_session_with_fallback(artifact_path)
    hybrid_session = make_onnx_session(
        artifact_path=artifact_path,
        providers=[
            _coreml_provider_with_options(compute_unit_name),
            "CPUExecutionProvider",
        ],
    )

    fixed_batch = infer_fixed_batch_size(cpu_session) or infer_fixed_batch_size(
        hybrid_session
    )
    effective_batch_size = fixed_batch if fixed_batch is not None else batch_size

    tokenized = tokenize_fixed(
        tokenizer,
        build_benchmark_texts(batch_size=effective_batch_size),
        seq_len=seq_len,
        dtype=np.int64,
    )
    cpu_expected_inputs = {inp.name for inp in cpu_session.get_inputs()}
    hybrid_expected_inputs = {inp.name for inp in hybrid_session.get_inputs()}
    cpu_feed = {name: tokenized[name] for name in cpu_expected_inputs}
    hybrid_feed = {name: tokenized[name] for name in hybrid_expected_inputs}

    cpu_report = benchmark_session(
        cpu_session,
        cpu_feed,
        tokenized["attention_mask"],
        matryoshka_dim,
        warmup,
        iters,
    )
    hybrid_report = benchmark_session(
        hybrid_session,
        hybrid_feed,
        tokenized["attention_mask"],
        matryoshka_dim,
        warmup,
        iters,
    )

    speedup = cpu_report["mean_ms"] / hybrid_report["mean_ms"]
    return {
        "status": "ok",
        "requested_batch_size": batch_size,
        "batch_size": effective_batch_size,
        "seq_len": seq_len,
        "cpu_graph_optimization_mode": cpu_opt_mode,
        "cpu": cpu_report,
        "hybrid": hybrid_report,
        "speedup_vs_cpu": float(speedup),
        "providers_hybrid": hybrid_session.get_providers(),
        "providers_cpu": cpu_session.get_providers(),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) * 0.5 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x.astype(np.float64) - float(np.mean(x))
    y0 = y.astype(np.float64) - float(np.mean(y))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 0:
        return 0.0
    return float(np.dot(x0, y0) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson_corr(_rankdata(x), _rankdata(y))


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    total = float(len(pos) * len(neg))
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def encode_texts_with_session(
    session,
    tokenizer,
    texts: list[str],
    seq_len: int,
    matryoshka_dim: int,
) -> np.ndarray:
    expected_inputs = {inp.name for inp in session.get_inputs()}
    fixed_batch = infer_fixed_batch_size(session)
    batch_size = fixed_batch if fixed_batch is not None else min(16, len(texts))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks: list[np.ndarray] = []
    idx = 0
    while idx < len(texts):
        batch_texts = list(texts[idx : idx + batch_size])
        take = len(batch_texts)
        if fixed_batch is not None and take < fixed_batch:
            batch_texts.extend([batch_texts[-1]] * (fixed_batch - take))
        tokenized = tokenize_fixed(tokenizer, batch_texts, seq_len, dtype=np.int64)
        feed = {name: tokenized[name] for name in expected_inputs}
        _, embeddings = run_embedding_inference(
            session,
            feed,
            tokenized["attention_mask"],
            matryoshka_dim,
        )
        chunks.append(embeddings[:take])
        idx += take

    return np.concatenate(chunks, axis=0)


def _topk_overlap(emb_a: np.ndarray, emb_b: np.ndarray, k: int) -> tuple[float, float]:
    n = emb_a.shape[0]
    if n <= 1:
        return 1.0, 1.0

    k_eff = min(max(k, 1), n - 1)
    sim_a = emb_a @ emb_a.T
    sim_b = emb_b @ emb_b.T
    np.fill_diagonal(sim_a, -np.inf)
    np.fill_diagonal(sim_b, -np.inf)

    topk_a = np.argpartition(-sim_a, kth=k_eff - 1, axis=1)[:, :k_eff]
    topk_b = np.argpartition(-sim_b, kth=k_eff - 1, axis=1)[:, :k_eff]
    top1_a = np.argmax(sim_a, axis=1)
    top1_b = np.argmax(sim_b, axis=1)

    overlaps = []
    for i in range(n):
        set_a = set(int(x) for x in topk_a[i])
        set_b = set(int(x) for x in topk_b[i])
        overlaps.append(len(set_a & set_b) / float(k_eff))

    topk_overlap = float(np.mean(overlaps))
    top1_agreement = float(np.mean(top1_a == top1_b))
    return topk_overlap, top1_agreement


def run_eval_quality(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
    matryoshka_dim: int,
    compute_unit_name: str,
    topk: int,
    require_full_coreml: bool,
    min_spearman: float,
    max_auc_drop: float,
) -> tuple[bool, dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    cpu_session, cpu_opt_mode = create_cpu_session_with_fallback(artifact_path)
    hybrid_providers: list[object] = [_coreml_provider_with_options(compute_unit_name)]
    if not require_full_coreml:
        hybrid_providers.append("CPUExecutionProvider")
    hybrid_session = make_onnx_session(
        artifact_path=artifact_path,
        providers=hybrid_providers,
        require_full_coreml=require_full_coreml,
    )

    pairs = list(QUALITY_PAIRS)
    unique_texts = sorted({a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    emb_cpu = encode_texts_with_session(
        cpu_session, tokenizer, unique_texts, seq_len, matryoshka_dim
    )
    emb_hybrid = encode_texts_with_session(
        hybrid_session, tokenizer, unique_texts, seq_len, matryoshka_dim
    )
    text_idx = {t: i for i, t in enumerate(unique_texts)}

    labels = np.asarray([int(lbl) for _, _, lbl in pairs], dtype=np.int64)
    scores_cpu = np.asarray(
        [
            float(np.dot(emb_cpu[text_idx[a]], emb_cpu[text_idx[b]]))
            for a, b, _ in pairs
        ],
        dtype=np.float64,
    )
    scores_hybrid = np.asarray(
        [
            float(np.dot(emb_hybrid[text_idx[a]], emb_hybrid[text_idx[b]]))
            for a, b, _ in pairs
        ],
        dtype=np.float64,
    )
    score_delta = np.abs(scores_cpu - scores_hybrid)

    cpu_auc = _binary_auc(labels, scores_cpu)
    hybrid_auc = _binary_auc(labels, scores_hybrid)
    auc_drop = None
    if cpu_auc is not None and hybrid_auc is not None:
        auc_drop = float(cpu_auc - hybrid_auc)
    spearman = _spearman_corr(scores_cpu, scores_hybrid)
    pearson = _pearson_corr(scores_cpu, scores_hybrid)

    topk_overlap, top1_agreement = _topk_overlap(emb_cpu, emb_hybrid, topk)

    pos_cpu = scores_cpu[labels == 1]
    neg_cpu = scores_cpu[labels == 0]
    pos_hybrid = scores_hybrid[labels == 1]
    neg_hybrid = scores_hybrid[labels == 0]

    auc_ok = auc_drop is not None and auc_drop <= max_auc_drop
    spearman_ok = spearman >= min_spearman
    ok = bool(auc_ok and spearman_ok)

    report = {
        "status": "pass" if ok else "fail",
        "artifact": str(artifact_path),
        "seq_len": seq_len,
        "quality_dataset": {
            "name": "builtin_semantic_pairs_v1",
            "pair_count": len(pairs),
            "unique_text_count": len(unique_texts),
            "positive_pairs": int(np.sum(labels == 1)),
            "negative_pairs": int(np.sum(labels == 0)),
        },
        "cpu_graph_optimization_mode": cpu_opt_mode,
        "providers_cpu": cpu_session.get_providers(),
        "providers_hybrid": hybrid_session.get_providers(),
        "cpu": {
            "auc": cpu_auc,
            "mean_positive_cos": float(np.mean(pos_cpu)),
            "mean_negative_cos": float(np.mean(neg_cpu)),
        },
        "hybrid": {
            "auc": hybrid_auc,
            "mean_positive_cos": float(np.mean(pos_hybrid)),
            "mean_negative_cos": float(np.mean(neg_hybrid)),
        },
        "parity": {
            "pair_score_spearman": float(spearman),
            "pair_score_pearson": float(pearson),
            "mean_abs_pair_delta": float(np.mean(score_delta)),
            "max_abs_pair_delta": float(np.max(score_delta)),
            "top1_neighbor_agreement": top1_agreement,
            "topk_neighbor_overlap": topk_overlap,
            "topk_k": int(min(max(topk, 1), len(unique_texts) - 1)),
        },
        "quality_gate": {
            "require_full_coreml": require_full_coreml,
            "min_spearman": min_spearman,
            "max_auc_drop": max_auc_drop,
            "auc_drop": auc_drop,
            "auc_ok": auc_ok,
            "spearman_ok": spearman_ok,
        },
    }
    return ok, report


def load_stsb_pairs(
    split: str,
    max_samples: int,
) -> tuple[list[tuple[str, str, float]], str]:
    from datasets import load_dataset

    ds = load_dataset("glue", "stsb", split=split)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    pairs: list[tuple[str, str, float]] = []
    for row in ds:
        a = str(row["sentence1"]).strip()
        b = str(row["sentence2"]).strip()
        score = float(row["label"]) / 5.0
        pairs.append(
            (
                f"search_document: {a}",
                f"search_document: {b}",
                score,
            )
        )
    return pairs, "glue/stsb"


def run_eval_stsb(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
    matryoshka_dim: int,
    compute_unit_name: str,
    split: str,
    max_samples: int,
    require_full_coreml: bool,
    min_cpu_spearman: float,
    max_spearman_drop: float,
    max_mean_abs_delta: float,
) -> tuple[bool, dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    cpu_session, cpu_opt_mode = create_cpu_session_with_fallback(artifact_path)
    hybrid_providers: list[object] = [_coreml_provider_with_options(compute_unit_name)]
    if not require_full_coreml:
        hybrid_providers.append("CPUExecutionProvider")
    hybrid_session = make_onnx_session(
        artifact_path=artifact_path,
        providers=hybrid_providers,
        require_full_coreml=require_full_coreml,
    )

    pairs, dataset_name = load_stsb_pairs(split=split, max_samples=max_samples)
    unique_texts = sorted({a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    emb_cpu = encode_texts_with_session(
        cpu_session, tokenizer, unique_texts, seq_len, matryoshka_dim
    )
    emb_hybrid = encode_texts_with_session(
        hybrid_session, tokenizer, unique_texts, seq_len, matryoshka_dim
    )
    text_idx = {t: i for i, t in enumerate(unique_texts)}

    gold = np.asarray([float(lbl) for _, _, lbl in pairs], dtype=np.float64)
    scores_cpu = np.asarray(
        [
            float(np.dot(emb_cpu[text_idx[a]], emb_cpu[text_idx[b]]))
            for a, b, _ in pairs
        ],
        dtype=np.float64,
    )
    scores_hybrid = np.asarray(
        [
            float(np.dot(emb_hybrid[text_idx[a]], emb_hybrid[text_idx[b]]))
            for a, b, _ in pairs
        ],
        dtype=np.float64,
    )
    score_delta = np.abs(scores_cpu - scores_hybrid)

    cpu_spearman = _spearman_corr(scores_cpu, gold)
    cpu_pearson = _pearson_corr(scores_cpu, gold)
    hybrid_spearman = _spearman_corr(scores_hybrid, gold)
    hybrid_pearson = _pearson_corr(scores_hybrid, gold)

    parity_spearman = _spearman_corr(scores_cpu, scores_hybrid)
    parity_pearson = _pearson_corr(scores_cpu, scores_hybrid)
    spearman_drop = float(cpu_spearman - hybrid_spearman)
    mean_abs_delta = float(np.mean(score_delta))

    cpu_spearman_ok = cpu_spearman >= min_cpu_spearman
    spearman_drop_ok = spearman_drop <= max_spearman_drop
    mean_abs_delta_ok = mean_abs_delta <= max_mean_abs_delta
    ok = bool(cpu_spearman_ok and spearman_drop_ok and mean_abs_delta_ok)

    report = {
        "status": "pass" if ok else "fail",
        "artifact": str(artifact_path),
        "seq_len": seq_len,
        "dataset": {
            "name": dataset_name,
            "split": split,
            "pair_count": len(pairs),
            "unique_text_count": len(unique_texts),
            "labels_normalized_to_0_1": True,
        },
        "cpu_graph_optimization_mode": cpu_opt_mode,
        "providers_cpu": cpu_session.get_providers(),
        "providers_hybrid": hybrid_session.get_providers(),
        "cpu": {
            "sts_spearman": float(cpu_spearman),
            "sts_pearson": float(cpu_pearson),
            "sim_mean": float(np.mean(scores_cpu)),
            "sim_std": float(np.std(scores_cpu)),
        },
        "hybrid": {
            "sts_spearman": float(hybrid_spearman),
            "sts_pearson": float(hybrid_pearson),
            "sim_mean": float(np.mean(scores_hybrid)),
            "sim_std": float(np.std(scores_hybrid)),
        },
        "parity": {
            "cpu_vs_hybrid_spearman": float(parity_spearman),
            "cpu_vs_hybrid_pearson": float(parity_pearson),
            "mean_abs_pair_delta": mean_abs_delta,
            "p95_abs_pair_delta": float(np.percentile(score_delta, 95)),
            "max_abs_pair_delta": float(np.max(score_delta)),
            "spearman_drop": spearman_drop,
        },
        "quality_gate": {
            "require_full_coreml": require_full_coreml,
            "min_cpu_spearman": min_cpu_spearman,
            "max_spearman_drop": max_spearman_drop,
            "max_mean_abs_delta": max_mean_abs_delta,
            "cpu_spearman_ok": cpu_spearman_ok,
            "spearman_drop_ok": spearman_drop_ok,
            "mean_abs_delta_ok": mean_abs_delta_ok,
        },
    }
    return ok, report


# ---------- Optional direct coremltools conversion path ----------


def load_hf_model(model_id: str, revision: Optional[str]):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_flash_attn"):
        model.config.use_flash_attn = False
    model.requires_grad_(False)
    model.eval()
    return model


def convert_to_coreml(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
) -> None:
    import coremltools as ct
    import torch
    from transformers import AutoTokenizer

    model = load_hf_model(model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    example = tokenize_fixed(
        tokenizer, ["search_query: hello world"], seq_len, dtype=np.int32
    )

    class TorchWrapper(torch.nn.Module):
        def __init__(self, encoder_model):
            super().__init__()
            self.encoder_model = encoder_model

        def forward(self, input_ids, attention_mask, token_type_ids):  # type: ignore[override]
            out = self.encoder_model(
                input_ids=input_ids.long(),
                attention_mask=attention_mask.long(),
                token_type_ids=token_type_ids.long(),
                return_dict=True,
            )
            return out.last_hidden_state

    wrapper = TorchWrapper(model).eval()

    with torch.no_grad():
        trace_inputs = (
            torch.from_numpy(example["input_ids"]),
            torch.from_numpy(example["attention_mask"]),
            torch.from_numpy(example["token_type_ids"]),
        )
        traced = torch.jit.trace(
            wrapper,
            trace_inputs,
            strict=False,
            check_trace=False,
        )
        traced = torch.jit.freeze(traced)

    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=example["input_ids"].shape,
                dtype=np.int32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=example["attention_mask"].shape,
                dtype=np.int32,
            ),
            ct.TensorType(
                name="token_type_ids",
                shape=example["token_type_ids"].shape,
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name="last_hidden_state")],
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(artifact_path))
    print(f"Saved Core ML package: {artifact_path}")


def _compute_unit_to_coremltools(value: str):
    import coremltools as ct

    mapping = {
        "cpu_only": "CPU_ONLY",
        "cpu_and_gpu": "CPU_AND_GPU",
        "cpu_and_ne": "CPU_AND_NE",
        "all": "ALL",
    }
    return getattr(ct.ComputeUnit, mapping[value])


def smoke_test_coreml(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
    matryoshka_dim: int,
    compute_unit_name: str,
) -> None:
    import coremltools as ct
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    inputs = tokenize_fixed(tokenizer, DEFAULT_PREFIXED_TEXTS, seq_len, dtype=np.int32)

    compute_unit = _compute_unit_to_coremltools(compute_unit_name)
    mlmodel = ct.models.MLModel(str(artifact_path), compute_units=compute_unit)
    outputs = mlmodel.predict(inputs)

    if "last_hidden_state" in outputs:
        hidden = outputs["last_hidden_state"]
    else:
        hidden = next(v for v in outputs.values() if isinstance(v, np.ndarray))

    embeddings = mean_pool(hidden, inputs["attention_mask"])
    if 0 < matryoshka_dim < embeddings.shape[1]:
        embeddings = embeddings[:, :matryoshka_dim]
    embeddings = l2_normalize(embeddings)

    print("Backend: coreml-convert")
    print(f"Compute units: {compute_unit_name}")
    print(f"Input shape: input_ids={inputs['input_ids'].shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    if embeddings.shape[0] >= 2:
        cosine = float(np.dot(embeddings[0], embeddings[1]))
        print(f"Cosine(doc, query): {cosine:.6f}")
    print("First embedding[0:8]:", np.array2string(embeddings[0, :8], precision=5))


def analyze_onnx_graph(artifact_path: Path) -> dict:
    import onnx

    model = onnx.load(str(artifact_path))
    op_counts = Counter(node.op_type for node in model.graph.node)
    zero_dim_value_infos: list[dict] = []
    total_nodes = len(model.graph.node)

    for vi in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
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
            zero_dim_value_infos.append({"name": vi.name, "shape": dims})

    return {
        "total_nodes": total_nodes,
        "op_counts_top20": dict(op_counts.most_common(20)),
        "zero_dim_value_infos": zero_dim_value_infos[:20],
        "zero_dim_count": len(zero_dim_value_infos),
    }


def run_export_static(
    repo_root: Path,
    model_id: str,
    revision: Optional[str],
    seq_len: int,
    static_output: Path,
    static_raw_output: Path,
    skip_simplify: bool,
    output_kind: str,
    output_matryoshka_dim: int,
) -> dict:
    script_path = (
        repo_root
        / "packages"
        / "ane-infer"
        / "src"
        / "ane"
        / "infer"
        / "nomic"
        / "export_static.py"
    )
    cmd = [
        sys.executable,
        str(script_path),
        "--model-id",
        model_id,
        "--seq-len",
        str(seq_len),
        "--output",
        str(static_output),
        "--raw-output",
        str(static_raw_output),
        "--output-kind",
        output_kind,
        "--matryoshka-dim",
        str(output_matryoshka_dim),
        "--json",
    ]
    if revision:
        cmd.extend(["--revision", revision])
    if skip_simplify:
        cmd.append("--skip-simplify")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Static export failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def run_analyze(
    artifact_path: Path,
    compute_unit_name: str,
    graph_only: bool,
) -> dict:
    report = {
        "status": "ok",
        "artifact": str(artifact_path),
        "graph": analyze_onnx_graph(artifact_path),
    }
    if not graph_only:
        report["coreml_probe"] = probe_coreml_strict_coverage(
            artifact_path, compute_unit_name
        )
    return report


def run_validate_strict(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
    matryoshka_dim: int,
    compute_unit_name: str,
) -> tuple[bool, dict]:
    try:
        smoke_test_onnx_coreml(
            model_id=model_id,
            revision=revision,
            artifact_path=artifact_path,
            seq_len=seq_len,
            matryoshka_dim=matryoshka_dim,
            compute_unit_name=compute_unit_name,
            require_full_coreml=True,
        )
        return True, {
            "status": "pass",
            "strict_coreml_only": True,
            "artifact": str(artifact_path),
            "coreml_probe": {
                "strict_session_ok": True,
                "supported_nodes": 1,
                "total_nodes": 1,
                "coverage_pct": 100.0,
            },
        }
    except Exception as exc:  # noqa: BLE001
        probe = parse_coreml_coverage_from_text(str(exc))
        if "coverage_pct" not in probe:
            probe = probe_coreml_strict_coverage(
                artifact_path=artifact_path,
                compute_unit_name=compute_unit_name,
            )
        return False, {
            "status": "fail",
            "strict_coreml_only": False,
            "artifact": str(artifact_path),
            "error": str(exc),
            "coreml_probe": probe,
        }


def run_gate(
    model_id: str,
    revision: Optional[str],
    artifact_path: Path,
    seq_len: int,
    matryoshka_dim: int,
    compute_unit_name: str,
    batch_size: int,
    warmup: int,
    iters: int,
    min_coverage_pct: float,
    min_speedup: float,
) -> tuple[bool, dict]:
    strict_ok, strict_report = run_validate_strict(
        model_id=model_id,
        revision=revision,
        artifact_path=artifact_path,
        seq_len=seq_len,
        matryoshka_dim=matryoshka_dim,
        compute_unit_name=compute_unit_name,
    )
    if strict_ok:
        return True, {
            "status": "pass",
            "mode": "strict_coreml_only",
            "gate_thresholds": {
                "min_coverage_pct": min_coverage_pct,
                "min_speedup": min_speedup,
            },
            "strict": strict_report,
        }

    analyze_report = run_analyze(
        artifact_path=artifact_path,
        compute_unit_name=compute_unit_name,
        graph_only=False,
    )
    bench_report = benchmark_hybrid_vs_cpu(
        model_id=model_id,
        revision=revision,
        artifact_path=artifact_path,
        seq_len=seq_len,
        matryoshka_dim=matryoshka_dim,
        compute_unit_name=compute_unit_name,
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
    )

    coverage_pct = float(
        analyze_report.get("coreml_probe", {}).get("coverage_pct", 0.0)
    )
    speedup = float(bench_report.get("speedup_vs_cpu", 0.0))
    coverage_ok = coverage_pct >= min_coverage_pct
    speedup_ok = speedup >= min_speedup
    hybrid_ok = coverage_ok and speedup_ok

    return hybrid_ok, {
        "status": "pass" if hybrid_ok else "fail",
        "mode": "hybrid_threshold_gate",
        "gate_thresholds": {
            "min_coverage_pct": min_coverage_pct,
            "min_speedup": min_speedup,
        },
        "checks": {
            "coverage_pct": coverage_pct,
            "coverage_ok": coverage_ok,
            "speedup_vs_cpu": speedup,
            "speedup_ok": speedup_ok,
        },
        "strict": strict_report,
        "analyze": analyze_report,
        "bench": bench_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nomic-embed-text-v1.5 on ANE-focused backends."
    )
    parser.add_argument(
        "--backend",
        default="onnx-coreml",
        choices=("onnx-coreml", "coreml-convert"),
        help="Backend mode. Default uses ONNX Runtime + CoreMLExecutionProvider.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision (branch/tag/commit) to pin.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Fixed token length used for conversion/smoke test.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="Output/input artifact path (.onnx for onnx-coreml, .mlpackage for coreml-convert).",
    )
    parser.add_argument(
        "--static-artifact",
        type=Path,
        default=DEFAULT_STATIC_ARTIFACT,
        help="Sanitized static ONNX artifact path for export-static-onnx.",
    )
    parser.add_argument(
        "--static-raw-artifact",
        type=Path,
        default=DEFAULT_STATIC_RAW_ARTIFACT,
        help="Raw static ONNX artifact path before sanitation.",
    )
    parser.add_argument(
        "--onnx-filename",
        default="onnx/model_fp16.onnx",
        help="Filename inside HF repo used by onnx-coreml backend.",
    )
    parser.add_argument(
        "--matryoshka-dim",
        type=int,
        default=512,
        help="Optional embedding truncation dimension after mean pooling.",
    )
    parser.add_argument(
        "--compute-unit",
        default="cpu_and_ne",
        choices=("cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"),
        help="Compute unit preference.",
    )
    parser.add_argument(
        "--require-full-coreml",
        action="store_true",
        help="Fail if ONNX Runtime would fall back to CPUExecutionProvider.",
    )
    parser.add_argument(
        "--bench-batch-size",
        type=int,
        default=DEFAULT_BENCH_BATCH_SIZE,
        help=f"Batch size for benchmark/gate (default {DEFAULT_BENCH_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--bench-warmup",
        type=int,
        default=DEFAULT_BENCH_WARMUP,
        help=f"Warmup iterations for benchmark/gate (default {DEFAULT_BENCH_WARMUP}).",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=DEFAULT_BENCH_ITERS,
        help=f"Timed iterations for benchmark/gate (default {DEFAULT_BENCH_ITERS}).",
    )
    parser.add_argument(
        "--quality-topk",
        type=int,
        default=DEFAULT_QUALITY_TOPK,
        help=f"Top-k neighbor overlap for eval-quality (default {DEFAULT_QUALITY_TOPK}).",
    )
    parser.add_argument(
        "--quality-min-spearman",
        type=float,
        default=DEFAULT_QUALITY_MIN_SPEARMAN,
        help=(
            "Minimum CPU-vs-hybrid pair-score Spearman for eval-quality "
            f"(default {DEFAULT_QUALITY_MIN_SPEARMAN})."
        ),
    )
    parser.add_argument(
        "--quality-max-auc-drop",
        type=float,
        default=DEFAULT_QUALITY_MAX_AUC_DROP,
        help=(
            "Maximum allowed AUC drop (cpu_auc-hybrid_auc) for eval-quality "
            f"(default {DEFAULT_QUALITY_MAX_AUC_DROP})."
        ),
    )
    parser.add_argument(
        "--quality-require-full-coreml",
        action="store_true",
        help=(
            "For eval-quality: disable CPU fallback and require a strict CoreML "
            "session for the hybrid path."
        ),
    )
    parser.add_argument(
        "--stsb-split",
        default=DEFAULT_STSB_SPLIT,
        help=f"Dataset split for eval-stsb (default {DEFAULT_STSB_SPLIT}).",
    )
    parser.add_argument(
        "--stsb-max-samples",
        type=int,
        default=DEFAULT_STSB_MAX_SAMPLES,
        help=(
            "Limit sample count for eval-stsb; 0 means full split "
            f"(default {DEFAULT_STSB_MAX_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--stsb-min-cpu-spearman",
        type=float,
        default=DEFAULT_STSB_MIN_CPU_SPEARMAN,
        help=(
            "Minimum CPU Spearman against STS-B labels for eval-stsb "
            f"(default {DEFAULT_STSB_MIN_CPU_SPEARMAN})."
        ),
    )
    parser.add_argument(
        "--stsb-max-spearman-drop",
        type=float,
        default=DEFAULT_STSB_MAX_SPEARMAN_DROP,
        help=(
            "Maximum allowed Spearman drop (cpu-hybrid) for eval-stsb "
            f"(default {DEFAULT_STSB_MAX_SPEARMAN_DROP})."
        ),
    )
    parser.add_argument(
        "--stsb-max-mean-abs-delta",
        type=float,
        default=DEFAULT_STSB_MAX_MEAN_ABS_DELTA,
        help=(
            "Maximum mean absolute pair-score delta between CPU and hybrid for "
            f"eval-stsb (default {DEFAULT_STSB_MAX_MEAN_ABS_DELTA})."
        ),
    )
    parser.add_argument(
        "--stsb-require-full-coreml",
        action="store_true",
        help=(
            "For eval-stsb: disable CPU fallback and require a strict CoreML "
            "session for the hybrid path."
        ),
    )
    parser.add_argument(
        "--min-coverage-pct",
        type=float,
        default=DEFAULT_GATE_MIN_COVERAGE_PCT,
        help=f"Minimum CoreML coverage percentage for hybrid gate (default {DEFAULT_GATE_MIN_COVERAGE_PCT}).",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=DEFAULT_GATE_MIN_SPEEDUP,
        help=f"Minimum hybrid-vs-CPU speedup for hybrid gate (default {DEFAULT_GATE_MIN_SPEEDUP}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Directory for default JSON reports (default {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional explicit JSON report output path.",
    )
    parser.add_argument(
        "--skip-simplify",
        action="store_true",
        help="Skip ONNX simplification during export-static-onnx.",
    )
    parser.add_argument(
        "--static-output-kind",
        default=DEFAULT_STATIC_OUTPUT_KIND,
        choices=("hidden", "embedding"),
        help=(
            "For export-static-onnx: exported output type. "
            f"Default {DEFAULT_STATIC_OUTPUT_KIND}."
        ),
    )
    parser.add_argument(
        "--static-output-matryoshka-dim",
        type=int,
        default=DEFAULT_STATIC_OUTPUT_MATRYOSHKA_DIM,
        help=(
            "For export-static-onnx when --static-output-kind embedding: optional "
            f"dim truncation (default {DEFAULT_STATIC_OUTPUT_MATRYOSHKA_DIM})."
        ),
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="For analyze: skip strict CoreML probing and do graph analysis only.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Force artifact preparation in 'run' mode even if present.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "convert", help="Prepare artifact (download ONNX or convert Core ML)."
    )
    sub.add_parser("smoke", aliases=["smoke-test"], help="Run embedding smoke test.")
    sub.add_parser("run", help="Prepare artifact if needed, then smoke test.")
    sub.add_parser(
        "export-static-onnx",
        help="Export fixed-shape ONNX with static RoPE tables and sanitation pass.",
    )
    sub.add_parser(
        "analyze", help="Analyze ONNX graph and CoreML strict partition coverage."
    )
    sub.add_parser(
        "validate-strict",
        help="Validate strict no-fallback CoreML run (fails if any CPU fallback required).",
    )
    sub.add_parser("bench", help="Benchmark hybrid (CoreML+CPU) vs CPU-only execution.")
    sub.add_parser(
        "eval-quality",
        help=(
            "Evaluate semantic quality and CPU-vs-hybrid embedding parity on a "
            "built-in labeled pair set."
        ),
    )
    sub.add_parser(
        "eval-stsb",
        help=(
            "Evaluate quality on GLUE STS-B and compare CPU vs hybrid/CoreML "
            "quality and parity."
        ),
    )
    sub.add_parser(
        "gate",
        help="Pass if strict CoreML-only succeeds, else enforce hybrid thresholds.",
    )
    return parser.parse_args()


def infer_repo_root() -> Path:
    # .../packages/ane-infer/src/ane/infer/nomic/pipeline.py -> repo root
    return Path(__file__).resolve().parents[6]


def main() -> int:
    args = parse_args()
    repo_root = infer_repo_root()
    ensure_repo_local_cache(repo_root)

    artifact = args.artifact
    if not artifact.is_absolute():
        artifact = (repo_root / artifact).resolve()

    static_artifact = args.static_artifact
    if not static_artifact.is_absolute():
        static_artifact = (repo_root / static_artifact).resolve()

    static_raw_artifact = args.static_raw_artifact
    if not static_raw_artifact.is_absolute():
        static_raw_artifact = (repo_root / static_raw_artifact).resolve()

    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = (repo_root / report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    json_out = args.json_out
    if json_out is not None and not json_out.is_absolute():
        json_out = (repo_root / json_out).resolve()

    def default_report_path(name: str) -> Path:
        return report_dir / f"{name}.json"

    def require_onnx_backend(command_name: str) -> None:
        if args.backend != "onnx-coreml":
            raise ValueError(
                f"'{command_name}' currently supports only --backend onnx-coreml"
            )

    def do_convert() -> None:
        if args.backend == "onnx-coreml":
            prepare_onnx_artifact(
                args.model_id,
                args.revision,
                args.onnx_filename,
                artifact,
            )
        else:
            convert_to_coreml(
                args.model_id,
                args.revision,
                artifact,
                args.seq_len,
            )

    def do_smoke() -> None:
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact}")
        if args.backend == "onnx-coreml":
            smoke_test_onnx_coreml(
                args.model_id,
                args.revision,
                artifact,
                args.seq_len,
                args.matryoshka_dim,
                args.compute_unit,
                args.require_full_coreml,
            )
        else:
            smoke_test_coreml(
                args.model_id,
                args.revision,
                artifact,
                args.seq_len,
                args.matryoshka_dim,
                args.compute_unit,
            )

    if args.command == "convert":
        do_convert()
        return 0

    if args.command in {"smoke", "smoke-test"}:
        do_smoke()
        return 0

    if args.command == "run":
        if args.force_convert or not artifact.exists():
            do_convert()
        do_smoke()
        return 0

    if args.command == "export-static-onnx":
        require_onnx_backend(args.command)
        report = run_export_static(
            repo_root=repo_root,
            model_id=args.model_id,
            revision=args.revision,
            seq_len=args.seq_len,
            static_output=static_artifact,
            static_raw_output=static_raw_artifact,
            skip_simplify=args.skip_simplify,
            output_kind=args.static_output_kind,
            output_matryoshka_dim=args.static_output_matryoshka_dim,
        )
        output_path = json_out or default_report_path("export_static")
        write_json(output_path, report)
        print(f"Exported static ONNX: {static_artifact}")
        print(f"Report: {output_path}")
        return 0

    if args.command == "analyze":
        require_onnx_backend(args.command)
        report = run_analyze(
            artifact_path=artifact,
            compute_unit_name=args.compute_unit,
            graph_only=args.graph_only,
        )
        output_path = json_out or default_report_path("analyze")
        write_json(output_path, report)
        print(f"Analyze complete: {output_path}")
        if "coreml_probe" in report:
            probe = report["coreml_probe"]
            coverage = probe.get("coverage_pct")
            if coverage is not None:
                print(f"CoreML coverage: {coverage:.2f}%")
        return 0

    if args.command == "validate-strict":
        require_onnx_backend(args.command)
        ok, report = run_validate_strict(
            model_id=args.model_id,
            revision=args.revision,
            artifact_path=artifact,
            seq_len=args.seq_len,
            matryoshka_dim=args.matryoshka_dim,
            compute_unit_name=args.compute_unit,
        )
        output_path = json_out or default_report_path("validate_strict")
        write_json(output_path, report)
        print(f"Strict validation report: {output_path}")
        return 0 if ok else 1

    if args.command == "bench":
        require_onnx_backend(args.command)
        report = benchmark_hybrid_vs_cpu(
            model_id=args.model_id,
            revision=args.revision,
            artifact_path=artifact,
            seq_len=args.seq_len,
            matryoshka_dim=args.matryoshka_dim,
            compute_unit_name=args.compute_unit,
            batch_size=args.bench_batch_size,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
        )
        output_path = json_out or default_report_path("bench")
        write_json(output_path, report)
        print(f"Bench complete: {output_path}")
        print(
            "CPU mean={:.2f}ms | Hybrid mean={:.2f}ms | speedup={:.3f}x".format(
                report["cpu"]["mean_ms"],
                report["hybrid"]["mean_ms"],
                report["speedup_vs_cpu"],
            )
        )
        return 0

    if args.command == "eval-quality":
        require_onnx_backend(args.command)
        ok, report = run_eval_quality(
            model_id=args.model_id,
            revision=args.revision,
            artifact_path=artifact,
            seq_len=args.seq_len,
            matryoshka_dim=args.matryoshka_dim,
            compute_unit_name=args.compute_unit,
            topk=args.quality_topk,
            require_full_coreml=args.quality_require_full_coreml,
            min_spearman=args.quality_min_spearman,
            max_auc_drop=args.quality_max_auc_drop,
        )
        output_path = json_out or default_report_path("quality")
        write_json(output_path, report)
        print(f"Quality report: {output_path}")

        cpu_auc = report.get("cpu", {}).get("auc")
        hybrid_auc = report.get("hybrid", {}).get("auc")
        auc_drop = report.get("quality_gate", {}).get("auc_drop")
        spearman = report.get("parity", {}).get("pair_score_spearman")
        top1 = report.get("parity", {}).get("top1_neighbor_agreement")
        topk_overlap = report.get("parity", {}).get("topk_neighbor_overlap")
        print(
            "AUC cpu={:.4f} hybrid={:.4f} drop={:.4f} | spearman={:.5f} | top1={:.3f} topk={:.3f}".format(
                float(cpu_auc) if cpu_auc is not None else float("nan"),
                float(hybrid_auc) if hybrid_auc is not None else float("nan"),
                float(auc_drop) if auc_drop is not None else float("nan"),
                float(spearman) if spearman is not None else float("nan"),
                float(top1) if top1 is not None else float("nan"),
                float(topk_overlap) if topk_overlap is not None else float("nan"),
            )
        )
        return 0 if ok else 1

    if args.command == "eval-stsb":
        require_onnx_backend(args.command)
        ok, report = run_eval_stsb(
            model_id=args.model_id,
            revision=args.revision,
            artifact_path=artifact,
            seq_len=args.seq_len,
            matryoshka_dim=args.matryoshka_dim,
            compute_unit_name=args.compute_unit,
            split=args.stsb_split,
            max_samples=args.stsb_max_samples,
            require_full_coreml=args.stsb_require_full_coreml,
            min_cpu_spearman=args.stsb_min_cpu_spearman,
            max_spearman_drop=args.stsb_max_spearman_drop,
            max_mean_abs_delta=args.stsb_max_mean_abs_delta,
        )
        output_path = json_out or default_report_path("stsb")
        write_json(output_path, report)
        print(f"STS-B report: {output_path}")
        print(
            (
                "STS-B Spearman cpu={:.5f} hybrid={:.5f} drop={:.5f} | "
                "cpu_vs_hybrid={:.5f} | mean_abs_delta={:.6f}"
            ).format(
                report["cpu"]["sts_spearman"],
                report["hybrid"]["sts_spearman"],
                report["parity"]["spearman_drop"],
                report["parity"]["cpu_vs_hybrid_spearman"],
                report["parity"]["mean_abs_pair_delta"],
            )
        )
        return 0 if ok else 1

    if args.command == "gate":
        require_onnx_backend(args.command)
        ok, report = run_gate(
            model_id=args.model_id,
            revision=args.revision,
            artifact_path=artifact,
            seq_len=args.seq_len,
            matryoshka_dim=args.matryoshka_dim,
            compute_unit_name=args.compute_unit,
            batch_size=args.bench_batch_size,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            min_coverage_pct=args.min_coverage_pct,
            min_speedup=args.min_speedup,
        )
        output_path = json_out or default_report_path("gate")
        write_json(output_path, report)
        print(f"Gate report: {output_path}")
        return 0 if ok else 1

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
