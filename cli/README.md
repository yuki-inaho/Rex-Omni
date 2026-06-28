# Rex-Omni バッチ検出 CLI

Gradio デモ（`app.py`）と同じデータフロー・同じ可視化関数（`rex_omni.RexOmniVisualize`）を再利用した、
**detection タスク限定**のバッチ処理 CLI です。

- `batch_detect.py` … JSON 設定を入力に、データセット内の画像を一括検出し、結果を JSON 出力（tqdm 進捗付き）
- `overlay_detections.py` … 結果 JSON を元画像にオーバーレイして可視化画像を出力

新規依存はありません（`tqdm` は導入済み。`jsonschema` があれば設定を自動検証、無くても動作）。

## 1. 設定 JSON

スキーマ: [`configs/config.schema.json`](configs/config.schema.json) ／ 実体例: [`configs/ccw90_leaf.json`](configs/ccw90_leaf.json)

```jsonc
{
  "model": {                       // 任意。RexOmniWrapper 設定（既定あり）
    "model_path": "models/Rex-Omni",
    "backend": "transformers",
    "attn_implementation": "sdpa", // このマシンでは sdpa 必須（flash-attn 不可）
    "min_pixels": 12544,
    "max_pixels": 2007040,
    "max_tokens": 4096,
    "temperature": 0.0, "top_p": 0.05, "top_k": 1, "repetition_penalty": 1.05
  },
  "dataset": {
    "image_dir": "/abs/path/to/images",
    "pattern": "*.jpg"             // 命名規則（glob）
  },
  "task": "detection",            // detection のみ
  "categories": ["leaf"],         // 1個以上。複数クラス可（例: ["leaf","fruit"]）
  "batch_size": 4,                // GPU 同時推論枚数
  "output": {
    "results_json": "/abs/path/to/out/results.json",
    "include_raw_output": true     // モデル生テキストを保存（トレーサビリティ）
  }
}
```

## 2. 実行

事前に GPU 空きを確認（共有環境のため）:
```bash
nvidia-smi --query-gpu=memory.free --format=csv
```

検出（先頭5枚で試走 → 本実行）:
```bash
cd /home/kasm-user/Desktop/Rex-Omni
CUDA_VISIBLE_DEVICES=0 .venv/bin/python cli/batch_detect.py --config cli/configs/ccw90_leaf.json --limit 5
CUDA_VISIBLE_DEVICES=0 .venv/bin/python cli/batch_detect.py --config cli/configs/ccw90_leaf.json --overwrite
```

オーバーレイ可視化:
```bash
.venv/bin/python cli/overlay_detections.py \
  --results /home/kasm-user/Desktop/ccw90_images_Jun28_2026/rexomni_leaf_Jun28_2026/results.json \
  --colors '{"leaf":"#00FF00"}'
# 既定で results.json と同階層の overlays/ に <stem>_vis.jpg を出力
```

`overlay_detections.py` の主なオプション: `--out-dir` `--image-dir`（元画像移動時）
`--font-size`(20) `--draw-width`(5) `--no-labels` `--colors`（クラス→HEX, 任意）`--limit`。

## 3. 出力 results.json の構造

```jsonc
{
  "meta": { "task":"detection","categories":["leaf"],"image_dir":"...","pattern":"*.jpg",
            "model_path":"...","num_images":501,"num_failed":0,
            "total_detections":1234,"elapsed_sec":123.4 },
  "results": [
    { "file":"00000500.jpg", "image_size":[600,800],
      "predictions": { "leaf": [ {"type":"box","coords":[x0,y0,x1,y1]}, ... ] },
      "num_detections": 12, "inference_time": 0.83, "raw_output":"...", "success": true }
  ]
}
```

- `predictions` は `RexOmniVisualize` にそのまま渡せる形（`extracted_predictions`）。
- 座標 `coords` は**元画像の絶対ピクセル**（[x0,y0,x1,y1]）。
- Rex-Omni は検出スコア（confidence）を出さないため score フィールドはありません。

## 4. 注意

- transformers バックエンドは `attn_implementation="sdpa"` を指定すること（このマシンは flash-attn 不可）。
- `batch_size` を上げると VRAM 消費が増えます（単枚実効 ~8GB / 32GB。4 は安全圏）。
- 1枚の推論失敗で全体を止めず、`success:false` として記録します。
