import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy_rgb(image):
    if isinstance(image, np.ndarray):
        return image
    try:
        import PIL.Image
        if isinstance(image, PIL.Image.Image):
            return np.array(image)
    except Exception:
        pass
    if torch.is_tensor(image):
        img = image.detach().cpu()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = img.numpy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    raise TypeError(f"Unsupported image type for numpy conversion: {type(image)}")


def resize_image_mask(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    mask_ids: List[int],
    patch_size: int = 14,
    max_tokens: int = 32,
):
    resize_images = []
    resize_masks = []
    mask_nums = []
    box_params = []
    for i, mask in enumerate(masks):
        image = images[mask_ids[i]]
        h, w = image.shape[:2]

        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        rows, cols = np.where(mask == 1)
        if rows.size == 0 or cols.size == 0:
            # Empty mask: fall back to full image box and zero mask
            bbox = (0, 0, h - 1, w - 1)
            cropping_img = image
            cropping_mask = np.zeros((h, w), dtype=np.float32)
        else:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            bbox = (
                max(0, min_row - patch_size * 2),
                max(0, min_col - patch_size * 2),
                min(h - 1, max_row + patch_size * 2),
                min(w - 1, max_col + patch_size * 2),
            )
            cropping_img = image[bbox[0]: bbox[2], bbox[1]: bbox[3], :]
            cropping_mask = mask[bbox[0]: bbox[2], bbox[1]: bbox[3]]

        mask_h = cropping_mask.shape[0]
        mask_w = cropping_mask.shape[1]
        mask_sum = float(np.sum(mask)) if mask is not None else 0.0

        scale_rate = math.ceil(math.sqrt(196 * max_tokens / max(mask_sum, 1.0)))
        if scale_rate == 1:
            if (mask_sum / 196) > 100:
                scale_rate = math.sqrt((mask_sum / 196) / 100)
                scale_rate = 1 / scale_rate

        resize_h = math.ceil((mask_h * scale_rate) / patch_size) * patch_size
        resize_w = math.ceil((mask_w * scale_rate) / patch_size) * patch_size
        resize_h = max(resize_h, patch_size)
        resize_w = max(resize_w, patch_size)

        resize_img = cv2.resize(cropping_img, (resize_w, resize_h))
        cropping_mask_t = torch.from_numpy(cropping_mask).float()
        resize_mask = F.interpolate(
            cropping_mask_t[None, None],
            size=(resize_h // patch_size, resize_w // patch_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mask_nums.append(min(max_tokens, int(resize_mask.sum().item())))

        resize_images.append(resize_img)
        resize_masks.append(resize_mask)
        box_params.append((bbox, h, w))

    return resize_images, resize_masks, mask_nums, box_params


def prepare_additional_inputs(
    images,
    masks_per_view,
    processor,
    patch_size: int = 14,
    max_tokens: int = 32,
    return_mask_nums: bool = False,
):
    if masks_per_view is None or processor is None:
        if return_mask_nums:
            return None, None, None, None, None
        return None, None, None, None

    images_np = [_to_numpy_rgb(img) for img in images]

    flat_masks = []
    mask_ids = []
    for view_idx, masks in enumerate(masks_per_view):
        for mask in masks:
            flat_masks.append(mask)
            mask_ids.append(view_idx)

    if len(flat_masks) == 0:
        return None, None, None, None

    resize_images, resize_masks, mask_nums, box_params = resize_image_mask(
        images_np,
        flat_masks,
        mask_ids,
        patch_size=patch_size,
        max_tokens=max_tokens,
    )

    dummy_text = [""] * len(resize_images)
    processed = processor(text=dummy_text, images=resize_images, return_tensors="pt")

    additional_masks = []
    additional_box_params = []
    additional_mask_nums = []
    offset = 0
    for masks in masks_per_view:
        count = len(masks)
        additional_masks.append(resize_masks[offset:offset + count])
        additional_box_params.append(box_params[offset:offset + count])
        additional_mask_nums.append(mask_nums[offset:offset + count])
        offset += count

    if return_mask_nums:
        return (
            processed["pixel_values"],
            processed["image_grid_thw"],
            additional_masks,
            additional_box_params,
            additional_mask_nums,
        )

    return (
        processed["pixel_values"],
        processed["image_grid_thw"],
        additional_masks,
        additional_box_params,
    )
