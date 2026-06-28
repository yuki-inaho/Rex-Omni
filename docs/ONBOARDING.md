# LLMオンボーディングサマリー

> 新任LLMエージェントが本リポジトリ（Rex-Omni / uv運用）に参加する際の初期資料です。
> 本ドキュメントは `IDEA-Research/Rex-Omni` をフォークした `yuki-inaho/Rex-Omni` を、
> このマシン上で **uv 環境**で動かすための実運用情報を記載しています。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** Rex-Omni（IDEA-Research）。**3BパラメータのマルチモーダルLLM（Qwen2.5-VL 3Bベース）**で、物体検出をはじめ各種ビジョン認識タスクを「next-token / next-point 予測」として統一的に解く。
- **最終成果物:** このマシンの **uv 仮想環境**で Rex-Omni を ①推論（detection/pointing/visual prompting/keypoint/OCR/GUI grounding）、②Gradioデモ、③評価・ファインチューニング、まで実行できる状態。
- **ビジネス背景・価値:** 言語駆動の汎用検出基盤。SAM連携でセグメンテーション、grounding data engine で自動アノテーション等に展開可能（`applications/` 参照）。
- **現時点の進捗サマリ（2026-06-28）:**
  - uv 環境構築済み（Python 3.10 / torch 2.7.0+cu128）
  - モデル `IDEA-Research/Rex-Omni`（7.6GB）を `models/Rex-Omni/` にダウンロード済み
  - 検出スモークテスト（`verify_setup.py`）成功：cafe.jpg で 30物体 / 11カテゴリ
  - Gradioデモ（`app.py`）起動 → Playwright で E2E テスト成功（検出可視化まで確認）
  - 上記の修正一式を `yuki-inaho/Rex-Omni` の `master` に push 済み

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ラインです。

- **transformers バックエンドでは `attn_implementation="sdpa"` を必須指定**。このマシンは **glibc 2.31 / nvcc無し**のため flash-attn（prebuilt wheel は GLIBC_2.32 要求）が動かない。`RexOmniWrapper` の既定は `flash_attention_2` なので、上書きしないとロード時に落ちる。
- **torch は CUDA 12.8 系（`cu128`）を使用**。GPU が **RTX PRO 4500 Blackwell** で、Blackwell は CUDA 12.8+ が必要。
- **gradio は 4.44.1、fastapi は `0.112.4` にピン**。新しい fastapi/starlette だと gradio 4.44.1 の `TemplateResponse` が壊れ、全ページ HTTP 500（`TypeError: unhashable type: 'dict'`）→ `demo.launch()` が落ちる。
- **Python は 3.10**（README の conda 環境に準拠）。
- **`models/`（重み 7.6GB）はコミットしない**（`.gitignore` 済み。`huggingface-cli download` で取得）。
- **GPUは共有環境**。他プロセス（例: SeedVR）が占有中は **VRAMが空くまで待機**してから起動する。
- **ライセンス**: 本体 IDEA License 1.0、ベースの Qwen は Qwen RESEARCH LICENSE。再配布・商用判断は人間に確認。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | （TBD：未作成） | 公式リサーチリポジトリのため社内要求定義は無し |
| 要件定義書 | `README.md` / [論文 arXiv:2510.12798](https://arxiv.org/abs/2510.12798) / [公式サイト](https://rex-omni.github.io/) | インストール手順・API・対応タスク・モデル仕様 |
| セットアップ手順 | `docs/ONBOARDING.md`（本書）/ `~/Desktop/report_RexOmni_uv_setup_Jun28_2026.md` | uv環境構築・モデルDL・ハマりどころ |
| WBS / 進捗 | （TBD） | 上記「進捗サマリ」を暫定の正とする |
| テスト資産 | `verify_setup.py`（検出スモーク）/ `test_gradio_playwright.py`（Gradio E2E）/ `tutorials/`（各タスクのサンプル/notebook）/ `evaluation/README.md` | 動作確認・回帰確認 |
| 既知課題リスト | 本書「2. クリティカルな要求・制約」 | flash-attn非対応・fastapi pin・GPU共有 |
| ファインチューニング | `finetuning/README.md` | SFT / GRPO 手順 |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- 推論スクリプトの作成・実行（detection 以外の pointing/OCR/keypoint 等も含む）
- Gradioデモの起動・Playwright(`playwright-cli`)によるUIテスト
- 依存関係の更新・uv環境の再構築、`requirements.txt` の調整
- ドキュメント整備、検証結果の報告、軽微なバグ修正

### 任せないタスク
- モデル重みの再配布やライセンス可否の最終判断（人間に確認）
- 本番／外部公開デプロイの承認（`share=True` での外部公開含む）
- GPU占有が長時間に及ぶジョブを、他プロセス稼働中に無断で開始すること
- upstream（`IDEA-Research/Rex-Omni`）への push（push 先は `fork` = `yuki-inaho/Rex-Omni` のみ）

## 5. インタラクション方針
- **回答スタイル:** 日本語、見出し＋箇条書き中心。結論を先に。
- **回答手順:** 前提・現状確認 → 論点 → 提案/実施 → 検証結果、の順。単純作業や単純調査は Sonnet サブエージェントに委譲し、判断・統括は上位モデルが行う。
- **禁止事項・注意:** 未確定事項を断定しない（TBDは明示）。破壊的操作・外部公開は事前確認。実測せずにVRAM等の数値を推測で書かない。
- **秘匿情報の扱い:** トークン/認証情報をログや成果物に残さない。git の push 先・認証は `yuki-inaho` アカウント（SSH）。

## 6. 試行タスク（オンボーディング演習）
1. `verify_setup.py` を実行し、cafe.jpg で 30物体前後が検出され可視化が保存されることを確認する。
2. `app.py` を `--attn_implementation sdpa` で起動し、`playwright-cli` でブラウザから画像アップロード→Run Inference→可視化表示まで通す。
3. detection 以外のタスク（例: `task="pointing"` か `task="ocr_box"`）を1件、`tutorials/` のサンプルを参考に実行してみる。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 制約・既知課題が増えたら本書「2.」と該当節を更新し、根拠（再現コマンド/エラー文）を併記。
- **TBDの扱い:** 埋まっていない項目は `（TBD）` と明示し、勝手に確定させない。
- **レビュー/承認フロー:** コード変更は commit 前に動作確認（最低限 `verify_setup.py`）。push 先は `fork`（`yuki-inaho/Rex-Omni`）。`master` 直 push 運用（個人フォークのため）。
- **その他の運用ルール:** 大容量成果物（`models/`、`playwright_artifacts/`、`.playwright-cli/`）はコミットしない。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:**
  - リポジトリ: `~/Desktop/Rex-Omni`（remote: `fork` → `git@github.com:yuki-inaho/Rex-Omni.git`、`origin` → upstream IDEA-Research）
  - `rex_omni/`（パッケージ本体: `wrapper.py`/`tasks.py`/`parser.py`/`utils.py`）、`app.py`（Gradio）、`tutorials/`、`evaluation/`、`finetuning/`、`applications/`
  - モデル: `models/Rex-Omni/`（gitignore対象）
- **代表的なコマンド:**
  ```bash
  # 環境（再構築する場合）
  uv venv --python 3.10
  uv pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install -r requirements.txt    # fastapi==0.112.4 ピン込み（flash-attn/vllm は別途・任意）
  uv pip install -e . --no-deps

  # モデルDL
  HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/huggingface-cli download IDEA-Research/Rex-Omni --local-dir models/Rex-Omni

  # 検出スモークテスト
  REX_MODEL_PATH="$(pwd)/models/Rex-Omni" CUDA_VISIBLE_DEVICES=0 .venv/bin/python verify_setup.py

  # Gradio デモ起動（sdpa 必須）
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python app.py \
    --model_path "$(pwd)/models/Rex-Omni" --backend transformers \
    --attn_implementation sdpa --server_name 127.0.0.1 --server_port 6121

  # Playwright で UI テスト（別ターミナル）
  playwright-cli open http://127.0.0.1:6121 --browser=chromium
  ```
- **依存ライブラリ（主要・実績値）:** Python 3.10.6 / torch 2.7.0+cu128 / torchvision 0.22.0+cu128 / transformers 4.51.3 / qwen-vl-utils 0.0.14 / accelerate 1.10.1 / gradio 4.44.1 / **fastapi 0.112.4（pin）** / starlette 0.38.6 / pydantic 2.10.6。flash-attn=このマシンでは不可（sdpa使用）、vllm=transformers運用では不要（遅延import）。
- **GPU / VRAM 消費（実測 2026-06-28、RTX PRO 4500 Blackwell, 総VRAM ~32GB）:**

  | フェーズ | torch allocated | torch reserved | nvidia-smi used |
  |---|---|---|---|
  | モデルロード後 | 7,161 MiB | 7,830 MiB | 8,094 MiB |
  | 推論ピーク（detection, cafe.jpg, 11カテゴリ） | 7,730 MiB（peak） | 7,940 MiB（peak） | 8,264 MiB |

  - **実効VRAM消費は約 8 GB**（bf16, sdpa, この解像度）。**最低 12GB 以上のGPUを推奨**。
  - 入力解像度（`--max_pixels`、既定 `2560*28*28`）や生成トークン数（`--max_tokens`）を上げるとVRAMは増える。大画像・大量カテゴリ時は余裕を見ること。
  - GPUは共有のため、起動前に `nvidia-smi --query-gpu=memory.free --format=csv` で空きを確認する。
- **連絡先/責任者:** yoshikawa@inaho.co（yuki-inaho）

> ※本テンプレートは必要に応じて拡張・縮退して構いません。記入済みドキュメントはバージョン管理してください。
