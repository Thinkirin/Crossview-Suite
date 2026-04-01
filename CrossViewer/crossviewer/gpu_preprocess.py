import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Torch-side copy of Qwen2VLImageProcessor.smart_resize."""
    if max(height, width) / max(min(height, width), 1) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / max(min(height, width), 1)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_tensor(img, device):
    if torch.is_tensor(img):
        t = img
    else:
        import numpy as np
        if hasattr(img, "size") and hasattr(img, "mode"):
            img = np.array(img)
        t = torch.from_numpy(img)
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    if t.dtype != torch.uint8:
        t = t.to(torch.uint8)
    return t.to(device, non_blocking=True)


def qwen_vl_preprocess_torch(
    images: List,
    image_processor,
    device,
    debug: bool = False,
    debug_prefix: str = "",
    debug_per_item: bool = False,
    debug_max: int = 0,
    debug_every: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU version of Qwen2VLImageProcessor._preprocess + preprocess for images.
    Returns:
        pixel_values: [sum_patches, patch_dim] float32
        image_grid_thw: [num_images, 3] int64
    """
    do_resize = getattr(image_processor, "do_resize", True)
    do_rescale = getattr(image_processor, "do_rescale", True)
    do_normalize = getattr(image_processor, "do_normalize", True)
    rescale_factor = getattr(image_processor, "rescale_factor", None)
    if rescale_factor is None:
        rescale_factor = 1.0 / 255
    rescale_factor = float(rescale_factor)
    image_mean_val = getattr(image_processor, "image_mean", None)
    if image_mean_val is None:
        image_mean_val = [0.0, 0.0, 0.0]
    image_std_val = getattr(image_processor, "image_std", None)
    if image_std_val is None:
        image_std_val = [1.0, 1.0, 1.0]
    image_mean = torch.tensor(image_mean_val, device=device).view(1, 3, 1, 1)
    image_std = torch.tensor(image_std_val, device=device).view(1, 3, 1, 1)
    patch_size = int(getattr(image_processor, "patch_size", None) or 14)
    temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", None) or 2)
    merge_size = int(getattr(image_processor, "merge_size", None) or 2)
    min_pixels = getattr(image_processor, "min_pixels", None)
    if min_pixels is None:
        min_pixels = 56 * 56
    min_pixels = int(min_pixels)
    max_pixels = getattr(image_processor, "max_pixels", None)
    if max_pixels is None:
        max_pixels = 28 * 28 * 1280
    max_pixels = int(max_pixels)

    pixel_values_list = []
    grid_thw_list = []

    for i, img in enumerate(images):
        t0 = time.perf_counter() if debug and debug_per_item else None
        t = _to_tensor(img, device=device)
        h, w = int(t.shape[0]), int(t.shape[1])
        if do_resize:
            new_h, new_w = smart_resize(h, w, factor=patch_size * merge_size, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            new_h, new_w = h, w

        x = t.permute(2, 0, 1).unsqueeze(0).float()
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        if do_rescale:
            x = x * rescale_factor
        if do_normalize:
            x = (x - image_mean) / image_std

        patches = x  # [1,C,H,W]
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = temporal_patch_size - (patches.shape[0] % temporal_patch_size)
            patches = torch.cat([patches, patches[-1:].repeat(repeats, 1, 1, 1)], dim=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h = new_h // patch_size
        grid_w = new_w // patch_size

        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

        pixel_values_list.append(flatten)
        grid_thw_list.append(torch.tensor([grid_t, grid_h, grid_w], device=device, dtype=torch.int64))
        if debug and debug_per_item and (debug_max <= 0 or i < debug_max) and (debug_every <= 1 or i % debug_every == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            print(
                f"{debug_prefix}image[{i}] preprocess {h}x{w}->{new_h}x{new_w} "
                f"grid=({grid_t},{grid_h},{grid_w}) time={dt:.3f}s",
                flush=True,
            )

    pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
    image_grid_thw = torch.stack(grid_thw_list, dim=0) if grid_thw_list else None
    return pixel_values, image_grid_thw


def resize_image_mask_torch(
    images: List,
    masks: List,
    mask_ids: List[int],
    patch_size: int = 14,
    max_tokens: int = 32,
    debug: bool = False,
    debug_prefix: str = "",
    debug_max: int = 0,
    debug_every: int = 1,
    max_resize_pixels: Optional[int] = None,
):
    """
    GPU version of PixelRefer resize_image_mask (per-mask crop+resize).
    Returns: resize_images, resize_masks, mask_nums, box_params
    """
    resize_images = []
    resize_masks = []
    mask_nums = []
    box_params = []

    for i, mask in enumerate(masks):
        t0 = time.perf_counter() if debug else None
        img = images[mask_ids[i]]
        t_img = _to_tensor(img, device=img.device) if not torch.is_tensor(img) else img
        if t_img.dtype != torch.uint8:
            t_img = t_img.to(torch.uint8)
        h, w = int(t_img.shape[0]), int(t_img.shape[1])

        if torch.is_tensor(mask):
            t_mask = mask.to(t_img.device)
        else:
            t_mask = torch.from_numpy(mask).to(t_img.device)
        if t_mask.ndim != 2:
            t_mask = t_mask.squeeze()
        mask_bool = t_mask > 0.5
        fallback = False
        if not mask_bool.any():
            bbox = (0, 0, max(0, h - 1), max(0, w - 1))
            crop_img = t_img
            crop_mask = torch.zeros((h, w), device=t_img.device, dtype=torch.float32)
            fallback = True
        else:
            ys, xs = torch.where(mask_bool)
            min_row = int(ys.min().item())
            max_row = int(ys.max().item())
            min_col = int(xs.min().item())
            max_col = int(xs.max().item())

            pad = patch_size * 2
            top = max(0, min_row - pad)
            left = max(0, min_col - pad)
            bottom = min(h - 1, max_row + pad)
            right = min(w - 1, max_col + pad)
            bbox = (top, left, bottom, right)

            crop_img = t_img[top:bottom, left:right, :]
            crop_mask = t_mask[top:bottom, left:right]

        mask_h = int(crop_mask.shape[0])
        mask_w = int(crop_mask.shape[1])
        if mask_h <= 0 or mask_w <= 0 or crop_img.numel() == 0:
            bbox = (0, 0, max(0, h - 1), max(0, w - 1))
            crop_img = t_img
            crop_mask = torch.zeros((h, w), device=t_img.device, dtype=torch.float32)
            mask_h = int(crop_mask.shape[0])
            mask_w = int(crop_mask.shape[1])
            fallback = True
        mask_sum = float(mask_bool.sum().item()) if mask_bool.any() else 1.0
        if mask_sum <= 0:
            mask_sum = 1.0

        scale_rate = math.ceil(math.sqrt(196 * max_tokens / mask_sum))
        if scale_rate == 1:
            if (mask_sum / 196) > 100:
                scale_rate = math.sqrt((mask_sum / 196) / 100)
                scale_rate = 1 / scale_rate

        resize_h = int(math.ceil((mask_h * scale_rate) / patch_size) * patch_size)
        resize_w = int(math.ceil((mask_w * scale_rate) / patch_size) * patch_size)
        resize_h = max(resize_h, patch_size)
        resize_w = max(resize_w, patch_size)
        clamped = False
        if max_resize_pixels is not None and resize_h * resize_w > max_resize_pixels:
            scale = math.sqrt(max_resize_pixels / (resize_h * resize_w))
            resize_h = max(patch_size, int(math.floor(resize_h * scale / patch_size)) * patch_size)
            resize_w = max(patch_size, int(math.floor(resize_w * scale / patch_size)) * patch_size)
            clamped = True

        img_f = crop_img.permute(2, 0, 1).unsqueeze(0).float()
        img_f = F.interpolate(img_f, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
        resize_img = img_f[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8)

        mask_f = crop_mask.unsqueeze(0).unsqueeze(0).float()
        resize_mask = F.interpolate(
            mask_f,
            size=(resize_h // patch_size, resize_w // patch_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        mask_nums.append(min(max_tokens, int(resize_mask.sum().item())))
        resize_images.append(resize_img)
        resize_masks.append(resize_mask)
        box_params.append((bbox, h, w))
        if debug and (debug_max <= 0 or i < debug_max) and (clamped or debug_every <= 1 or i % debug_every == 0):
            if t_img.device.type == "cuda":
                torch.cuda.synchronize(t_img.device)
            dt = time.perf_counter() - t0
            clamp_msg = " clamped" if clamped else ""
            print(
                f"{debug_prefix}mask[{i}] view={mask_ids[i]} "
                f"bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) "
                f"resize=({resize_h},{resize_w}){clamp_msg} tokens={mask_nums[-1]} time={dt:.3f}s",
                flush=True,
            )

    return resize_images, resize_masks, mask_nums, box_params
