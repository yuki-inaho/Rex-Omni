#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal verification script for the Rex-Omni uv environment.

Runs the documented detection example. Uses flash_attention_2 if available,
otherwise falls back to sdpa so the transformers backend works without a
compiled flash-attn wheel.
"""

import os
import torch
from PIL import Image
from rex_omni import RexOmniVisualize, RexOmniWrapper

# Choose attention implementation based on availability.
try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
except Exception:
    attn_impl = "sdpa"

print(f"[verify] torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[verify] device: {torch.cuda.get_device_name(0)}")
print(f"[verify] attn_implementation: {attn_impl}")

model_path = os.environ.get("REX_MODEL_PATH", "IDEA-Research/Rex-Omni")

rex_model = RexOmniWrapper(
    model_path=model_path,
    backend="transformers",
    attn_implementation=attn_impl,
    max_tokens=4096,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

image_path = "tutorials/detection_example/test_images/cafe.jpg"
image = Image.open(image_path).convert("RGB")

categories = [
    "man", "woman", "yellow flower", "sofa", "robot-shope light",
    "blanket", "microwave", "laptop", "cup", "white chair", "lamp",
]

results = rex_model.inference(images=image, task="detection", categories=categories)
result = results[0]

if result["success"]:
    predictions = result["extracted_predictions"]
    n = sum(len(v) for v in predictions.values())
    print(f"[verify] SUCCESS: {n} objects across {len(predictions)} categories")
    for cat, preds in predictions.items():
        print(f"  - {cat}: {len(preds)}")
    vis_image = RexOmniVisualize(
        image=image, predictions=predictions,
        font_size=20, draw_width=5, show_labels=True,
    )
    out = "tutorials/detection_example/test_images/cafe_visualize.jpg"
    vis_image.save(out)
    print(f"[verify] Visualization saved to: {out}")
else:
    print(f"[verify] Inference FAILED: {result['error']}")
    raise SystemExit(1)
