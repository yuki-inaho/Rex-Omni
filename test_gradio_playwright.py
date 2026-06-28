#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Playwright end-to-end test for the Rex-Omni Gradio app (app.py).

Prereq: the Gradio app must already be running, e.g.
    .venv/bin/python app.py --model_path "$(pwd)/models/Rex-Omni" \
        --backend transformers --attn_implementation sdpa \
        --server_name 127.0.0.1 --server_port 6121

Then run:
    .venv/bin/python test_gradio_playwright.py

Env vars:
    REX_APP_URL   target URL (default http://127.0.0.1:6121)
    REX_SHOT_DIR  screenshot output dir (default ./playwright_artifacts)

Exit code 0 = page loaded + detection produced output; 1 = failure.
"""

import os
import sys
import pathlib
from playwright.sync_api import sync_playwright, expect

URL = os.environ.get("REX_APP_URL", "http://127.0.0.1:6121")
SHOT_DIR = pathlib.Path(os.environ.get("REX_SHOT_DIR", "playwright_artifacts"))
SHOT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE = pathlib.Path("tutorials/detection_example/test_images/cafe.jpg").resolve()


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.set_default_timeout(30_000)

        print(f"[pw] navigating to {URL}")
        page.goto(URL, wait_until="networkidle")

        # 1) Assert the app rendered.
        expect(page.get_by_text("Rex Omni: Detect Anything Demo")).to_be_visible()
        print("[pw] PASS: title rendered")
        page.screenshot(path=str(SHOT_DIR / "01_loaded.png"), full_page=True)

        # 2) Upload the test image into the input image component.
        file_inputs = page.locator('input[type="file"]')
        file_inputs.first.set_input_files(str(IMAGE))
        print(f"[pw] uploaded {IMAGE.name}")

        # 3) Fill the categories textbox.
        try:
            cats = page.get_by_label("Categories")
            cats.fill("man, woman, cup, laptop, sofa, lamp")
        except Exception as e:
            print(f"[pw] WARN: could not fill categories via label: {e}")

        page.wait_for_timeout(1000)
        page.screenshot(path=str(SHOT_DIR / "02_configured.png"), full_page=True)

        # 4) Click Run Inference.
        page.get_by_role("button", name="Run Inference").click()
        print("[pw] clicked Run Inference; waiting for model output...")

        # 5) Wait for the raw model output textbox to populate (inference can be slow).
        ok = False
        try:
            output = page.get_by_label("Model Raw Output")
            expect(output).not_to_have_value("", timeout=180_000)
            val = output.input_value()
            print(f"[pw] model output length: {len(val)} chars")
            ok = len(val.strip()) > 0
        except Exception as e:
            print(f"[pw] WARN: output textbox check failed: {e}")
            # Fallback: check a result <img> appeared.
            try:
                page.locator(".image-container img, [data-testid='image'] img").first.wait_for(
                    timeout=180_000
                )
                ok = True
            except Exception as e2:
                print(f"[pw] result image check failed: {e2}")

        page.wait_for_timeout(1500)
        page.screenshot(path=str(SHOT_DIR / "03_result.png"), full_page=True)
        browser.close()

        if ok:
            print(f"[pw] SUCCESS: detection flow completed. Screenshots in {SHOT_DIR}/")
            return 0
        print("[pw] FAILURE: no model output detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
