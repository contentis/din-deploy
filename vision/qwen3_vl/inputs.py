# SPDX-License-Identifier: Apache-2.0
"""Bounded image/video preparation. Budgets count merged LLM visual tokens."""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass(frozen=True)
class VideoSegment:
    path: str | Path
    start: float = 0.0
    end: float | None = None
    fps: float = 2.0
    max_frames: int = 32


def resize_shape(width, height, tokens, factor=32):
    if min(width, height, tokens, factor) < 1:
        raise ValueError("Dimensions and visual token budget must be positive")
    scale = min(1.0, math.sqrt(tokens * factor**2 / (width * height)))
    w = min(tokens, max(1, round(width * scale / factor)))
    h = min(tokens, max(1, round(height * scale / factor)))
    while w * h > tokens:
        if w >= h:
            w -= 1
        else:
            h -= 1
    return w * factor, h * factor


def read_segment(segment, max_frames, max_pixels=None):
    """Stream-decode a bounded segment; retain only sampled frames, with actual PTS.

    At most max_frames decoded images are retained; no whole-video array is made.
    Missing duration is acceptable only when an explicit segment end is supplied.
    """
    import av

    if not math.isfinite(segment.start) or segment.start < 0 or not math.isfinite(segment.fps) or segment.fps <= 0:
        raise ValueError("Video start must be nonnegative and fps must be positive and finite")
    if max_frames < 1 or segment.max_frames < 1:
        raise ValueError("max_frames must be positive")
    with av.open(str(segment.path)) as container:
        if not container.streams.video:
            raise ValueError("Input has no video stream")
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else container.duration / av.time_base
            if container.duration is not None
            else None
        )
        end = segment.end if segment.end is not None else duration
        if end is None or not math.isfinite(end) or end <= segment.start:
            raise ValueError("Provide a finite video end after start (or a video with valid duration)")
        if duration is not None:
            end = min(end, duration)
        if end <= segment.start:
            raise ValueError("Video segment starts after the video ends")
        count = min(max_frames, segment.max_frames, max(1, math.ceil((end - segment.start) * segment.fps)))
        targets = np.linspace(segment.start, end, count, endpoint=False)
        container.seek(int((segment.start + origin) / float(stream.time_base)), stream=stream, backward=True)
        images, times = [], []
        index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            seconds = float(frame.pts * stream.time_base) - origin
            if seconds >= end or index == len(targets):
                break
            if seconds + 1e-8 < targets[index]:
                continue
            image = frame.to_image().convert("RGB")
            if max_pixels is not None:
                # Shrink retained samples before collecting the segment, so a
                # short token budget cannot retain gigabytes of full-size RGB.
                scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
                if scale < 1:
                    image = image.resize(
                        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.BICUBIC,
                    )
            images.append(image)
            times.append(seconds)
            # A source frame can satisfy several target times, but is retained once.
            while index < len(targets) and targets[index] <= seconds + 1e-8:
                index += 1
        if not images:
            raise ValueError("Video segment contains no decodable timestamped frames")
        return images, times


def prepare_inputs(processor, prompt, images=(), videos=(), max_visual_tokens=256, max_image_tokens=None):
    """One request with any number of images/segments, followed by the text prompt.

    Equal budget per media item; unused quota is intentionally not redistributed.
    Video frames are uniformly subsampled before per-frame spatial resizing.
    """
    if isinstance(images, (str, Path, Image.Image)):
        images = [images]
    if isinstance(videos, VideoSegment):
        videos = [videos]
    count = len(images) + len(videos)
    if max_visual_tokens < 1 or (count and max_visual_tokens < count):
        raise ValueError("Visual budget must allow at least one token per media item")
    if max_image_tokens is not None and max_image_tokens < 1:
        raise ValueError("max_image_tokens must be positive")
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    temporal = processor.video_processor.temporal_patch_size
    base, remainder = divmod(max_visual_tokens, max(1, count))
    budgets = [base + (i < remainder) for i in range(count)]
    content, resized_images, resized_videos, metadata, media_report = [], [], [], [], []
    for i, source in enumerate(images):
        if isinstance(source, Image.Image):
            image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        budget = min(budgets[i], max_image_tokens or budgets[i])
        size = resize_shape(*image.size, budget, factor)
        resized_images.append(image.resize(size, Image.Resampling.BICUBIC))
        content.append({"type": "image"})
        media_report.append({"type": "image", "size": size, "visual_tokens": size[0] * size[1] // factor**2})
    for i, segment in enumerate(videos):
        budget = budgets[len(images) + i]
        frames, times = read_segment(
            segment,
            min(segment.max_frames, budget * temporal),
            max_pixels=min(budget, max_image_tokens or budget) * factor**2,
        )
        groups = math.ceil(len(frames) / temporal)
        per_frame = min(budget // groups, max_image_tokens or budget)
        size = resize_shape(*frames[0].size, per_frame, factor)
        array = np.stack([np.array(frame.resize(size, Image.Resampling.BICUBIC)) for frame in frames])
        resized_videos.append(array)
        # Processor timestamps = frames_indices / fps. Use seconds as indices and
        # fps=1 to preserve variable-frame-rate timestamps instead of guessing FPS.
        metadata.append({"fps": 1.0, "frames_indices": times, "total_num_frames": len(frames)})
        content.append({"type": "video"})
        media_report.append(
            {
                "type": "video",
                "size": size,
                "timestamps": times,
                "visual_tokens": groups * size[0] * size[1] // factor**2,
            }
        )
    content.append({"type": "text", "text": prompt})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    kwargs = {"text": [text], "return_tensors": "pt"}
    if resized_images:
        kwargs.update(images=resized_images, images_kwargs={"do_resize": False})
    if resized_videos:
        kwargs.update(
            videos=resized_videos,
            videos_kwargs={"do_resize": False, "do_sample_frames": False, "video_metadata": metadata},
        )
    inputs = processor(**kwargs)
    actual = sum(
        int(inputs[name].prod(-1).sum()) // 4 for name in ("image_grid_thw", "video_grid_thw") if name in inputs
    )
    if actual > max_visual_tokens:
        raise RuntimeError(f"Processor exceeded the visual budget: {actual} > {max_visual_tokens}")
    return inputs, {"visual_tokens": actual, "max_visual_tokens": max_visual_tokens, "media": media_report}


def position_ids(inputs, merge_size=2):
    """Weight-free Qwen3-VL multimodal positions; independently checked against HF."""
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("This runtime accepts batch size one")
    types = inputs.get("mm_token_type_ids", torch.zeros_like(ids))[0].tolist()
    grids = {
        1: iter(inputs.get("image_grid_thw", []).tolist() if "image_grid_thw" in inputs else []),
        2: iter([[1, h, w] for t, h, w in inputs.get("video_grid_thw", []) for _ in range(int(t))]),
    }
    result, current = [], 0
    for kind, group in itertools.groupby(types):
        length = len(list(group))
        if kind == 0:
            result.append(torch.arange(current, current + length).repeat(3, 1))
            current += length
        else:
            t, h, w = (int(x) for x in next(grids[kind]))
            h, w = h // merge_size, w // merge_size
            if t * h * w != length:
                raise ValueError("Visual placeholder count does not match the grid")
            result.append(
                torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")).reshape(
                    3, -1
                )
                + current
            )
            current += max(h, w)
    positions = torch.cat(result, dim=1)[:, None]
    return positions, int(positions.max()) + 1 - ids.shape[1]


def text_rotary(positions, config, dtype):
    head = config["head_dim"]
    rope = config.get("rope_parameters") or config["rope_scaling"]
    theta = rope.get("rope_theta", config.get("rope_theta", 5000000.0))
    inv = 1.0 / (theta ** (torch.arange(0, head, 2, device=positions.device).float() / head))
    frequencies = positions[..., None].float() * inv
    interleaved = frequencies[0].clone()
    for axis in (1, 2):
        interleaved[..., axis : rope["mrope_section"][axis] * 3 : 3] = frequencies[
            axis, ..., axis : rope["mrope_section"][axis] * 3 : 3
        ]
    emb = torch.cat([interleaved, interleaved], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
