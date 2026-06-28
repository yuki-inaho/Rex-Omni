#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overlay Rex-Omni detection results onto the original images.

Reads the results JSON produced by cli/batch_detect.py and draws the bounding
boxes (reusing rex_omni.RexOmniVisualize, the same renderer as the Gradio demo),
writing one visualization image per input with a tqdm progress bar.

Usage:
    python cli/overlay_detections.py --results <out>/results.json
    python cli/overlay_detections.py --results <out>/results.json \
        --out-dir <out>/overlays --colors '{"leaf": "#00FF00"}' --draw-width 4
"""

import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

from rex_omni import RexOmniVisualize


def hex_to_rgb(s):
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"invalid hex color: {s!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def parse_colors(spec):
    """spec is a JSON string mapping class name -> #RRGGBB, or None."""
    if not spec:
        return None
    mapping = json.loads(spec)
    return {k: hex_to_rgb(v) for k, v in mapping.items()}


def main():
    ap = argparse.ArgumentParser(description="Overlay Rex-Omni detections")
    ap.add_argument("--results", required=True, help="results.json from batch_detect.py")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <results dir>/overlays)")
    ap.add_argument("--image-dir", default=None,
                    help="override source image dir (default: meta.image_dir)")
    ap.add_argument("--font-size", type=int, default=20)
    ap.add_argument("--draw-width", type=int, default=5)
    ap.add_argument("--no-labels", action="store_true", help="hide category labels")
    ap.add_argument("--colors", default=None,
                    help='JSON map class->hex, e.g. \'{"leaf":"#00FF00"}\'')
    ap.add_argument("--limit", type=int, default=None, help="only first N entries")
    args = ap.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    entries = data.get("results", [])
    if args.limit:
        entries = entries[:args.limit]

    image_dir = args.image_dir or meta.get("image_dir")
    if not image_dir:
        raise SystemExit("image_dir not found in results meta; pass --image-dir")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.results)), "overlays")
    os.makedirs(out_dir, exist_ok=True)

    custom_colors = parse_colors(args.colors)
    show_labels = not args.no_labels

    n_ok, n_skip = 0, 0
    for e in tqdm(entries, unit="img"):
        fname = e["file"]
        src = os.path.join(image_dir, fname)
        if not e.get("success", False) or not os.path.exists(src):
            n_skip += 1
            continue
        image = Image.open(src).convert("RGB")
        vis = RexOmniVisualize(
            image=image,
            predictions=e.get("predictions", {}),
            font_size=args.font_size,
            draw_width=args.draw_width,
            show_labels=show_labels,
            custom_colors=custom_colors,
        )
        stem = os.path.splitext(fname)[0]
        vis.save(os.path.join(out_dir, f"{stem}_vis.jpg"))
        n_ok += 1

    print(f"[overlay] wrote {n_ok} images ({n_skip} skipped) -> {out_dir}")


if __name__ == "__main__":
    main()
