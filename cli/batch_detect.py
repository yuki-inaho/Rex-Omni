#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch detection CLI for Rex-Omni (detection task only).

Reads a JSON config (see cli/configs/config.schema.json) describing the model
settings, dataset directory + naming pattern, categories, and output path, then
runs detection over every matching image with a tqdm progress bar and writes a
single results JSON.

Usage:
    CUDA_VISIBLE_DEVICES=0 python cli/batch_detect.py --config cli/configs/ccw90_leaf.json
    python cli/batch_detect.py --config <cfg> --limit 5      # dry run on first 5 images

The output JSON is consumed by cli/overlay_detections.py to draw the boxes.
"""

import argparse
import glob
import json
import os
import sys
import time

from PIL import Image
from tqdm import tqdm

from rex_omni import RexOmniWrapper

# Defaults mirror RexOmniWrapper / app.py so behaviour matches the Gradio demo.
MODEL_DEFAULTS = {
    "model_path": "models/Rex-Omni",
    "backend": "transformers",
    "attn_implementation": "sdpa",
    "min_pixels": 16 * 28 * 28,
    "max_pixels": 2560 * 28 * 28,
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_p": 0.05,
    "top_k": 1,
    "repetition_penalty": 1.05,
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Optional schema validation (only if jsonschema is installed).
    schema_path = os.path.join(os.path.dirname(__file__), "configs", "config.schema.json")
    try:
        import jsonschema  # type: ignore

        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                jsonschema.validate(cfg, json.load(f))
    except ImportError:
        pass  # validation is best-effort

    # Minimal manual checks (always run).
    if "dataset" not in cfg or "image_dir" not in cfg["dataset"]:
        raise ValueError("config.dataset.image_dir is required")
    if not cfg.get("categories"):
        raise ValueError("config.categories must be a non-empty list")
    if "output" not in cfg or "results_json" not in cfg["output"]:
        raise ValueError("config.output.results_json is required")
    if cfg.get("task", "detection") != "detection":
        raise ValueError("this CLI only supports task='detection'")
    return cfg


def build_wrapper(model_cfg):
    m = {**MODEL_DEFAULTS, **(model_cfg or {})}
    kwargs = dict(
        model_path=m["model_path"],
        backend=m["backend"],
        min_pixels=m["min_pixels"],
        max_pixels=m["max_pixels"],
        max_tokens=m["max_tokens"],
        temperature=m["temperature"],
        top_p=m["top_p"],
        top_k=m["top_k"],
        repetition_penalty=m["repetition_penalty"],
    )
    # attn_implementation only applies to the transformers backend (same as app.py).
    if m["backend"] == "transformers":
        kwargs["attn_implementation"] = m["attn_implementation"]
    print(f"[batch_detect] loading model: {m['model_path']} "
          f"(backend={m['backend']}, attn={m.get('attn_implementation', 'n/a')})")
    return RexOmniWrapper(**kwargs)


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run_one_chunk(rex, paths, categories):
    """Run detection on a list of image paths; return list of (entry_dict)."""
    images = [Image.open(p).convert("RGB") for p in paths]
    results = rex.inference(images=images, task="detection", categories=categories)
    entries = []
    for p, res in zip(paths, results):
        preds = res.get("extracted_predictions", {})
        entries.append({
            "file": os.path.basename(p),
            "image_size": list(res.get("image_size", [])),
            "predictions": preds,
            "num_detections": sum(len(v) for v in preds.values()),
            "inference_time": res.get("inference_time"),
            "raw_output": res.get("raw_output"),
            "success": bool(res.get("success", False)),
        })
    return entries


def main():
    ap = argparse.ArgumentParser(description="Rex-Omni batch detection")
    ap.add_argument("--config", required=True, help="path to JSON config")
    ap.add_argument("--limit", type=int, default=None, help="process only first N images")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing results JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    categories = cfg["categories"]
    image_dir = cfg["dataset"]["image_dir"]
    pattern = cfg["dataset"].get("pattern", "*.jpg")
    batch_size = int(cfg.get("batch_size", 4))
    out_path = cfg["output"]["results_json"]
    include_raw = cfg["output"].get("include_raw_output", True)

    if os.path.exists(out_path) and not args.overwrite:
        print(f"[batch_detect] {out_path} exists; use --overwrite to replace. Aborting.")
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"[batch_detect] no images matched {os.path.join(image_dir, pattern)}")
        sys.exit(1)
    print(f"[batch_detect] {len(paths)} images | categories={categories} | batch_size={batch_size}")

    rex = build_wrapper(cfg.get("model"))

    entries = []
    t0 = time.time()
    with tqdm(total=len(paths), unit="img") as pbar:
        for chunk in chunked(paths, batch_size):
            try:
                entries.extend(run_one_chunk(rex, chunk, categories))
            except Exception as e:  # noqa: BLE001 - retry the chunk one-by-one
                tqdm.write(f"[batch_detect] chunk failed ({e}); retrying individually")
                for p in chunk:
                    try:
                        entries.extend(run_one_chunk(rex, [p], categories))
                    except Exception as e2:  # noqa: BLE001
                        tqdm.write(f"[batch_detect] FAILED {os.path.basename(p)}: {e2}")
                        entries.append({
                            "file": os.path.basename(p), "image_size": [],
                            "predictions": {}, "num_detections": 0,
                            "inference_time": None, "raw_output": None,
                            "success": False, "error": str(e2),
                        })
            pbar.update(len(chunk))
    elapsed = time.time() - t0

    if not include_raw:
        for e in entries:
            e.pop("raw_output", None)

    total_det = sum(e["num_detections"] for e in entries)
    n_fail = sum(1 for e in entries if not e["success"])
    out = {
        "meta": {
            "task": "detection",
            "categories": categories,
            "image_dir": image_dir,
            "pattern": pattern,
            "model_path": (cfg.get("model") or {}).get("model_path", MODEL_DEFAULTS["model_path"]),
            "num_images": len(entries),
            "num_failed": n_fail,
            "total_detections": total_det,
            "elapsed_sec": round(elapsed, 2),
        },
        "results": entries,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[batch_detect] done: {len(entries)} images, {total_det} detections, "
          f"{n_fail} failed, {elapsed:.1f}s")
    print(f"[batch_detect] results -> {out_path}")


if __name__ == "__main__":
    main()
