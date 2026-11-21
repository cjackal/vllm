# SPDX-License-Identifier: Apache-2.0
"""GPU checks for Kimi-K3/K3.1 video input (branch `kimi-k3-video`).

Start the server from the branch with the Python frontend
(`VLLM_USE_RUST_FRONTEND` unset or 0), then:

    python k3_video_gpu_check.py --base-url http://localhost:8000/v1 \
        --out vllm.json

Point `--base-url` at an SGLang server and diff the two JSON files: the
per-case `added_tokens` must match when sampling, resizing and the prompt
format agree.
"""

import argparse
import base64
import io
import json
import math
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from openai import BadRequestError, OpenAI
from PIL import Image
from transformers import AutoTokenizer

# (name, BGR)
COLORS = [
    ("red", (0, 0, 255)),
    ("green", (0, 255, 0)),
    ("blue", (255, 0, 0)),
    ("yellow", (0, 255, 255)),
    ("white", (255, 255, 255)),
    ("purple", (255, 0, 255)),
    ("orange", (0, 165, 255)),
    ("cyan", (255, 255, 0)),
]


def encode_video(
    frame_fn: Callable[[int], np.ndarray], num_frames: int, fps: float
) -> tuple[str, dict]:
    first = frame_fn(0)
    height, width = first.shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        writer = cv2.VideoWriter(
            f.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for i in range(num_frames):
            writer.write(first if i == 0 else frame_fn(i))
        writer.release()
        data = Path(f.name).read_bytes()
    url = "data:video/mp4;base64," + base64.b64encode(data).decode()
    meta = {"width": width, "height": height, "frames": num_frames, "fps": fps}
    return url, meta


def color_digit_frames(size: tuple[int, int], fps: float, seconds: float):
    width, height = size

    def frame(i: int) -> np.ndarray:
        idx = int(i / (fps * seconds)) % len(COLORS)
        img = np.full((height, width, 3), COLORS[idx][1], np.uint8)
        cv2.putText(
            img,
            str(idx + 1),
            (width // 3, int(height * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            height / 120,
            (0, 0, 0),
            max(2, height // 60),
        )
        return img

    return frame


def moving_square_frames(size: tuple[int, int], num_frames: int):
    width, height = size
    side = height // 4

    def frame(i: int) -> np.ndarray:
        img = np.zeros((height, width, 3), np.uint8)
        x = int((width - side) * i / max(num_frames - 1, 1))
        img[height // 2 - side // 2 : height // 2 + side // 2, x : x + side] = 255
        return img

    return frame


def image_url(bgr: tuple[int, int, int]) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (448, 448), bgr[::-1]).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def expected_video_block(meta: dict, cfg: dict, tokenizer) -> dict | None:
    """Embeddings and total tokens of the branch's block for this video.

    Returns None without vLLM. The block is the header, each chunk's
    timestamp and pads, and `<|media_end|>`.
    """
    try:
        from vllm.transformers_utils.processors.kimi_k3 import (
            format_kimi_k3_timestamp,
            get_kimi_k3_frame_resize,
            kimi_k3_timestamp_ms,
            plan_kimi_k3_video,
        )
    except ImportError:
        return None
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    indices = plan_kimi_k3_video(meta["frames"], fps, width, height, cfg)
    chunk = cfg["temporal_merge_kernel_size"]
    chunk_tokens = get_kimi_k3_frame_resize(width, height, cfg)["num_tokens"]

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    block = count(f"<|media_begin|>video {width}x{height}<|media_content|>")
    block += count("<|media_end|>")
    for start in range(0, len(indices), chunk):
        ms = kimi_k3_timestamp_ms(indices[start], fps)
        block += count(format_kimi_k3_timestamp(ms)) + chunk_tokens
    return {
        "frames": len(indices),
        "chunks": math.ceil(len(indices) / chunk),
        "embeds": math.ceil(len(indices) / chunk) * chunk_tokens,
        "block": block,
    }


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
        self.model = args.model
        self.extra_body = (
            {} if args.keep_thinking else {"chat_template_kwargs": {"thinking": False}}
        )

    def chat(self, content: list[dict], max_tokens: int = 512):
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
            extra_body=self.extra_body,
        )
        return response, time.time() - start


def video_part(url: str) -> dict:
    return {"type": "video_url", "video_url": {"url": url}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="moonshotai/Kimi-K3")
    parser.add_argument("--out", default="k3_video_check.json")
    parser.add_argument("--keep-thinking", action="store_true")
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    cfg_path = hf_hub_download(args.model, "preprocessor_config.json")
    cfg = json.loads(Path(cfg_path).read_text())["media_proc_cfg"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    runner = Runner(args)

    order_url, order_meta = encode_video(
        color_digit_frames((640, 480), 30.0, 1.0), 8 * 30, 30.0
    )
    stamp_url, stamp_meta = encode_video(
        color_digit_frames((640, 480), 30.0, 2.0), 16 * 30, 30.0
    )
    cases: list[tuple[str, list[dict], list[dict], str]] = [
        (
            "order",
            [video_part(order_url)],
            [order_meta],
            (
                "List the background color and the digit of each part of the "
                "video, in order."
            ),
        ),
        (
            "timestamp",
            [video_part(stamp_url)],
            [stamp_meta],
            (
                "At what second does the digit 5 first appear? Answer with a "
                "number. (expected: 8)"
            ),
        ),
        (
            "mixed_order",
            [
                {"type": "image_url", "image_url": {"url": image_url((0, 0, 255))}},
                video_part(order_url),
                {"type": "image_url", "image_url": {"url": image_url((255, 0, 0))}},
            ],
            [order_meta],
            "Describe the first image, the video and the last image, in order.",
        ),
        (
            "two_videos",
            [video_part(order_url), video_part(stamp_url)],
            [order_meta, stamp_meta],
            "How long is each video, and how do they differ?",
        ),
    ]
    if not args.skip_large:
        n = 60 * 30
        url_1080, meta_1080 = encode_video(
            moving_square_frames((1920, 1080), n), n, 30.0
        )
        cases.append(
            (
                "1080p_60s_cap",
                [video_part(url_1080)],
                [meta_1080],
                "Which way does the white square move? (expected: left to right)",
            )
        )
        n = 10 * 60 * 30
        url_tiny, meta_tiny = encode_video(moving_square_frames((32, 32), n), n, 30.0)
        # 4 embeddings per chunk cannot show the motion; this case checks the
        # frame cap (1024 frames, 256 chunks) and that the capped frames still
        # carry timestamps up to the end of the video.
        cases.append(
            (
                "tiny_10min_frame_cap",
                [video_part(url_tiny)],
                [meta_tiny],
                "How long is the video? Answer in minutes. (expected: about 10)",
            )
        )

    results = {}
    for name, parts, metas, question in cases:
        base, _ = runner.chat([{"type": "text", "text": question}], max_tokens=1)
        try:
            text = {"type": "text", "text": question}
            response, seconds = runner.chat([*parts, text])
        except BadRequestError as e:
            results[name] = {"error": str(e)}
            print(f"[{name}] 400: {e}")
            continue
        assert response.usage is not None and base.usage is not None
        added = response.usage.prompt_tokens - base.usage.prompt_tokens
        expected = [expected_video_block(meta, cfg, tokenizer) for meta in metas]
        has_images = any(part["type"] == "image_url" for part in parts)
        extra = (
            None
            if has_images or None in expected
            else added - sum(block["block"] for block in expected if block)
        )
        results[name] = {
            "added_tokens": added,
            "expected": expected,
            # Separators between parts (one newline token per part for K3).
            "extra_tokens": extra,
            "seconds": round(seconds, 2),
            "answer": response.choices[0].message.content,
        }
        print(f"[{name}] {json.dumps(results[name], ensure_ascii=False)}")

    # Repeating a request should hit the prefix cache (with
    # --enable-prompt-tokens-details the server reports cached tokens).
    content = [video_part(order_url), {"type": "text", "text": "Describe the video."}]
    first, t1 = runner.chat(content, max_tokens=1)
    second, t2 = runner.chat(content, max_tokens=1)
    details = getattr(second.usage, "prompt_tokens_details", None)
    results["repeat"] = {
        "first_seconds": round(t1, 2),
        "second_seconds": round(t2, 2),
        "cached_tokens": getattr(details, "cached_tokens", None),
    }
    print(f"[repeat] {results['repeat']}")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
