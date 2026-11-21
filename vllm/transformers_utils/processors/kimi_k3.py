# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.tokenizers.hf import HfTokenizer
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    navit_resize_image,
)

# Frame cap when the checkpoint sets no `max_num_frames_each_video`. The patch
# budget alone lets a tiny-resolution video keep ~160K frames; per-frame
# decode, resize and chunk work makes that a host-side amplification. 1024
# frames (128 s at 8 fps) only binds below 240p.
_DEFAULT_MAX_NUM_FRAMES = 1024


def _merge_size(media_proc_cfg: Mapping[str, Any]) -> int:
    merge = media_proc_cfg["merge_kernel_size"]
    return int(merge[0] if isinstance(merge, (list, tuple)) else merge)


def supports_kimi_k3_video(media_proc_cfg: Mapping[str, Any]) -> bool:
    """Whether the checkpoint's `media_proc_cfg` carries a usable video budget.

    Each 4-frame chunk is resized with `in_patch_limit_each_frame`; the fused
    processor caps a chunk's frames at `in_patch_limit_video / frames`, so the
    two only agree when the video budget covers a full chunk.
    """
    limit = media_proc_cfg.get("in_patch_limit_video")
    each_frame = media_proc_cfg.get("in_patch_limit_each_frame")
    chunk = media_proc_cfg.get("temporal_merge_kernel_size")
    return (
        limit is not None
        and each_frame is not None
        and chunk is not None
        and media_proc_cfg.get("sample_fps") is not None
        and media_proc_cfg.get("timestamp_mode") == "hh:mm:ss.fff"
        and limit >= chunk * each_frame
    )


def get_kimi_k3_frame_resize(
    width: int, height: int, media_proc_cfg: Mapping[str, Any]
) -> dict[str, int]:
    return navit_resize_image(
        width,
        height,
        media_proc_cfg["patch_size"],
        _merge_size(media_proc_cfg),
        media_proc_cfg["in_patch_limit_each_frame"],
        media_proc_cfg["patch_limit_on_one_side"],
        media_proc_cfg["fixed_output_tokens"],
    )


def get_kimi_k3_max_video_tokens(media_proc_cfg: Mapping[str, Any]) -> int:
    """Upper bound on the embeddings of one video, used for the encoder budget."""
    merge = _merge_size(media_proc_cfg)
    chunk = media_proc_cfg["temporal_merge_kernel_size"]
    return media_proc_cfg["in_patch_limit_video"] // (chunk * merge * merge)


def get_kimi_k3_max_video_frames(
    width: int, height: int, media_proc_cfg: Mapping[str, Any]
) -> int:
    """Most frames a video with this frame size keeps.

    Like SGLang, a video over `in_patch_limit_video` keeps the frame size and
    samples fewer frames. Frames are counted with their padded patch grid, and
    the count also keeps the rounded-up last chunk within
    `get_kimi_k3_max_video_tokens`, so no video exceeds the profiled budget.
    """
    if not supports_kimi_k3_video(media_proc_cfg):
        raise ValueError("This Kimi-K3 checkpoint does not support video input")
    resize = get_kimi_k3_frame_resize(width, height, media_proc_cfg)
    patch = media_proc_cfg["patch_size"]
    frame_patches = ((resize["new_width"] + resize["pad_width"]) // patch) * (
        (resize["new_height"] + resize["pad_height"]) // patch
    )
    chunk = media_proc_cfg["temporal_merge_kernel_size"]
    max_chunks = get_kimi_k3_max_video_tokens(media_proc_cfg) // resize["num_tokens"]
    max_frames = min(
        media_proc_cfg["in_patch_limit_video"] // frame_patches, max_chunks * chunk
    )
    frame_cap = media_proc_cfg.get("max_num_frames_each_video")
    max_frames = min(
        max_frames, _DEFAULT_MAX_NUM_FRAMES if frame_cap is None else frame_cap
    )
    if max_frames < 1:
        raise ValueError(f"Video frames of {width}x{height} exceed the video budget")
    return max_frames


def plan_kimi_k3_video(
    num_frames: int,
    fps: float,
    width: int,
    height: int,
    media_proc_cfg: Mapping[str, Any],
) -> list[int]:
    """Indices of the source frames Kimi-K3 samples from a video.

    Frames are sampled at `min(sample_fps, fps)`, up to
    `get_kimi_k3_max_video_frames`.
    """
    if num_frames <= 0 or not fps > 0:
        raise ValueError(f"Invalid video: {num_frames} frames at {fps} fps")
    max_frames = get_kimi_k3_max_video_frames(width, height, media_proc_cfg)
    sample_fps = min(media_proc_cfg["sample_fps"], fps)
    count = min(max(round(num_frames * sample_fps / fps), 1), max_frames)
    return np.linspace(0, num_frames - 1, count).round().astype(int).tolist()


def kimi_k3_timestamp_ms(frame_index: int, fps: float) -> int:
    return round(frame_index * 1_000_000 / fps) // 1000


def format_kimi_k3_timestamp(ms: int) -> str:
    hours, minutes, secs = ms // 3_600_000, ms // 60_000 % 60, ms // 1000 % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms % 1000:03d}"


def _frame_to_hwc(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3D video frame, got shape {frame.shape}")
    if frame.shape[-1] not in (1, 3, 4):
        frame = frame.transpose(1, 2, 0)
    return frame


def _frame_to_pil(frame: Any) -> Image.Image:
    frame = _frame_to_hwc(frame)
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video frames, got {frame.dtype}")
    if frame.shape[-1] == 1:
        frame = frame[..., 0]
    return Image.fromarray(frame).convert("RGB")


class KimiK3Processor(ProcessorMixin):
    """HF-style processor wrapper for the Kimi-K3 model.

    K3 exposes the standard `image` and `video` modalities. The underlying
    checkpoint image processor works on `{"type": "image", "image": PIL}` and
    `{"type": "video_chunk", "video_chunk": [PIL, ...]}` media dicts, so this
    wrapper adapts images and sampled video chunks into that shape before
    delegating to `preprocess`.

    Text is only tokenized here; the single placeholder per image or video is
    expanded into its media block by the model's `_get_prompt_updates` on the
    vLLM side.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: object | list[object] | None = None,
        videos: Sequence[tuple[Any, Mapping[str, Any]]] | None = None,
        return_tensors: str | TensorType | None = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            medias: list[dict[str, Any]] = [
                {"type": "image", "image": image} for image in images
            ]
            mm_inputs = self.image_processor.preprocess(
                medias,
                return_tensors=return_tensors,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
            )
            mm_inputs["image_sizes"] = [list(media["image"].size) for media in medias]
        else:
            mm_inputs = {}

        video_inputs = (
            self._preprocess_videos(videos, return_tensors, do_rescale, do_normalize)
            if videos
            else {}
        )

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text)
        else:
            text_inputs = {}

        return BatchFeature(
            data={**text_inputs, **mm_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def _preprocess_videos(
        self,
        videos: Sequence[tuple[Any, Mapping[str, Any]]],
        return_tensors: str | TensorType | None,
        do_rescale: bool,
        do_normalize: bool,
    ) -> dict[str, torch.Tensor]:
        """Sample each video into timestamped chunks of up to 4 frames.

        The frames to keep are planned over the source video, so a loader that
        already decoded exactly those frames keeps all of them; any other
        input is subsampled uniformly down to the planned count.
        """
        cfg = self.image_processor.media_proc_cfg
        chunk_size = cfg["temporal_merge_kernel_size"]
        medias: list[dict[str, Any]] = []
        timestamps: list[int] = []
        sizes: list[list[int]] = []
        num_chunks: list[int] = []

        for frames, metadata in videos:
            num_decoded = len(frames)
            if num_decoded == 0:
                raise ValueError("Video input has no frames")
            # A missing fps fails the plan with a request error.
            fps = float(metadata.get("fps") or 0)
            indices = list(metadata.get("frames_indices") or range(num_decoded))
            if len(indices) != num_decoded:
                raise ValueError(
                    f"Video metadata lists {len(indices)} frame indices for "
                    f"{num_decoded} frames"
                )
            height, width = _frame_to_hwc(frames[0]).shape[:2]
            count = len(
                plan_kimi_k3_video(
                    int(metadata.get("total_num_frames", num_decoded)),
                    fps,
                    width,
                    height,
                    cfg,
                )
            )
            positions = (
                np.linspace(0, num_decoded - 1, count).round().astype(int).tolist()
                if num_decoded > count
                else list(range(num_decoded))
            )

            chunks = [
                positions[start : start + chunk_size]
                for start in range(0, len(positions), chunk_size)
            ]
            for chunk in chunks:
                medias.append(
                    {
                        "type": "video_chunk",
                        "video_chunk": [_frame_to_pil(frames[i]) for i in chunk],
                    }
                )
                timestamps.append(kimi_k3_timestamp_ms(indices[chunk[0]], fps))
            sizes.append([width, height])
            num_chunks.append(len(chunks))

        outputs = self.image_processor.preprocess(
            medias,
            return_tensors=return_tensors,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
        )
        return {
            "video_pixel_values": outputs["pixel_values"],
            "video_grid_thws": outputs["grid_thws"],
            "video_chunk_timestamps": torch.tensor(timestamps, dtype=torch.int64),
            "video_sizes": torch.tensor(sizes, dtype=torch.int64),
            "video_num_chunks": torch.tensor(num_chunks, dtype=torch.int64),
        }
