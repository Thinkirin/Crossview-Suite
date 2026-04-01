"""
Core modules for CrossViewer:
- MaskPooling: Extract features from masked regions
- ART: Adaptive Region Tokenizer
- OCVA: Object-Centric Cross-View Aligner
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskPooling(nn.Module):
    """
    Pool features from masked regions using weighted average
    Supports dynamic resolution
    """

    def __init__(self):
        super().__init__()

    def forward(self, features, mask):
        """
        Args:
            features: [H, W, D] - feature map
            mask: [H_mask, W_mask] - binary mask (can be different resolution)

        Returns:
            pooled: [D] - pooled features
        """
        H, W, D = features.shape

        if mask.shape != (H, W):
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_resized = F.interpolate(
                mask_tensor,
                size=(H, W),
                mode='nearest'
            ).squeeze().to(features.device)
        else:
            if isinstance(mask, np.ndarray):
                mask_resized = torch.from_numpy(mask).float().to(features.device)
            else:
                mask_resized = mask.float()

        mask_resized = (mask_resized > 0.5).float()
        mask_expanded = mask_resized.unsqueeze(-1)
        masked_features = features * mask_expanded
        pooled = masked_features.sum(dim=(0, 1)) / (mask_resized.sum() + 1e-8)

        return pooled


class ART(nn.Module):
    """
    Adaptive Region Tokenizer
    Extracts multiple tokens from masked region using k-means style clustering
    Supports dynamic resolution and variable number of objects per sample
    """

    def __init__(self, hidden_size, num_tokens=10, use_position_encoding=True, patch_size=14,
                 debug_nan=False, is_main_process=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.use_position_encoding = use_position_encoding
        self.patch_size = patch_size
        self.debug_nan = bool(debug_nan)
        self.is_main_process = bool(is_main_process)
        self._nan_reported = False

        if use_position_encoding:
            self.pos_encoder = nn.Linear(2, hidden_size)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def _nan_check(self, name, tensor):
        if not self.debug_nan or not self.is_main_process or self._nan_reported:
            return False
        if tensor is None:
            return False
        if isinstance(tensor, (list, tuple)):
            for i, t in enumerate(tensor):
                if self._nan_check(f"{name}[{i}]", t):
                    return True
            return False
        if not torch.is_tensor(tensor):
            return False
        if torch.isfinite(tensor).all():
            return False
        print(
            f"[nan] art stage={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}",
            flush=True
        )
        self._nan_reported = True
        return True

    def _tensor_stats(self, tensor):
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return None
        t = tensor.float()
        finite = torch.isfinite(t)
        if finite.any():
            t_min = float(t[finite].min().item())
            t_max = float(t[finite].max().item())
        else:
            t_min = float("nan")
            t_max = float("nan")
        return t_min, t_max

    def generate_position_grid(self, h, w, device, dtype=None):
        """Generate 2D position grid"""
        y_grid = torch.linspace(0, 1, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
        x_grid = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, w).expand(h, w)
        pos_grid = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        return pos_grid

    def generate_position_tensor(self, h, w, box_xy, raw_h, raw_w, device, dtype=None):
        """Generate position tensor aligned with PixelRefer full implementation.

        box_xy is (top, left, bottom, right) in raw image coordinates.
        """
        box_top, box_left, box_bottom, box_right = box_xy
        box_w = box_right - box_left
        box_h = box_bottom - box_top

        x_tensor = torch.arange(w, device=device, dtype=torch.float32) / max(w, 1)
        x_tensor = x_tensor.repeat(h, 1)
        x_tensor = (x_tensor * box_w + box_left) / max(raw_w - 1, 1)

        y_tensor = torch.arange(h, device=device, dtype=torch.float32) / max(h, 1)
        y_tensor = y_tensor.unsqueeze(1).repeat(1, w)
        y_tensor = (y_tensor * box_h + box_top) / max(raw_h - 1, 1)

        pos = torch.stack([x_tensor, y_tensor], dim=2).clamp_(0.0, 1.0)
        if dtype is not None and pos.dtype != dtype:
            pos = pos.to(dtype)
        return pos

    def kmeans_fast(self, tokens, num_clusters=10, num_iterations=5):
        """PixelRefer-style k-means for token selection."""
        n, d = tokens.shape
        centroids = tokens[torch.randperm(n)[:num_clusters]]

        for _ in range(num_iterations):
            tokens_expand = tokens.unsqueeze(1)  # [n, 1, d]
            centroids_expand = centroids.unsqueeze(0)  # [1, num_clusters, d]
            distances = torch.sum((tokens_expand - centroids_expand) ** 2, dim=2)  # [n, num_clusters]
            labels = torch.argmin(distances, dim=1)  # [n]
            new_centroids = torch.stack([
                tokens[labels == i].mean(dim=0) if tokens[labels == i].size(0) > 0 else centroids[i]
                for i in range(num_clusters)
            ])
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids

    def mask_pooling_pixelrefer(self, feat, mask, mask_token_num):
        """
        PixelRefer-style mask pooling: upsample features to mask resolution,
        extract masked tokens, and k-means compress.
        """
        if not feat.shape[-2:] == mask.shape[-2:]:
            feat = F.interpolate(feat, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        if feat.device != mask.device:
            mask = mask.to(feat.device)
        mask = (mask > 0).to(mask.dtype)
        mask = mask.permute(1, 0, 2, 3)  # [1,1,H,W]

        mask_emb = feat * mask
        valid = torch.any(mask_emb != 0, dim=(0, 1))
        mask_emb = mask_emb[:, :, valid]
        if mask_emb.numel() == 0:
            return mask_emb.new_zeros((0, feat.shape[1]))
        mask_embedding = mask_emb[0].permute(1, 0)  # [N, C]

        if len(mask_embedding) > mask_token_num:
            mask_embedding = self.kmeans_fast(mask_embedding, num_clusters=mask_token_num)
        return mask_embedding

    def forward_pixelrefer(self, feature_maps, masks_list, box_params_list, return_valid=False, return_token_mask=False):
        """
        PixelRefer Full-style ART: use cropped-image features + resized masks + bbox params.

        Args:
            feature_maps: List[Tensor] - [N][H, W, D] cropped image features (one per mask)
            masks_list: List[Tensor/np.ndarray] - resized masks aligned to crop grid
            box_params_list: List[Tuple] - (bbox, raw_h, raw_w) per mask
            return_valid: whether to return valid mask flags

        Returns:
            tokens: [K, num_tokens, D]
            valid: [K] (optional)
        """
        self._nan_reported = False
        if len(masks_list) == 0:
            device = self.proj[0].weight.device
            empty_tokens = torch.zeros((0, 0, self.hidden_size), device=device)
            empty_token_mask = torch.zeros((0, 0), device=device, dtype=torch.bool)
            if return_valid:
                if return_token_mask:
                    return empty_tokens, torch.zeros((0,), dtype=torch.bool, device=device), empty_token_mask
                return empty_tokens, torch.zeros((0,), dtype=torch.bool, device=device)
            if return_token_mask:
                return empty_tokens, empty_token_mask
            return empty_tokens

        obj_tokens_list = []
        valid_list = []
        token_masks = []
        for feat_map, mask, box_param in zip(feature_maps, masks_list, box_params_list):
            self._nan_check("forward_pixelrefer/feat_map", feat_map)
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).float().to(feat_map.device)
            else:
                mask_tensor = mask.float().to(feat_map.device)

            mask_bool = mask_tensor > 0.5
            has_fg = bool(mask_bool.any().item())
            valid_list.append(has_fg)

            box_xy, raw_h, raw_w = box_param
            feat = feat_map
            if feat.dim() == 3:
                h, w, _ = feat.shape
            else:
                raise ValueError(f"Unexpected feature map shape: {feat.shape}")

            if self.use_position_encoding:
                pos_grid = self.generate_position_tensor(
                    h,
                    w,
                    box_xy,
                    raw_h,
                    raw_w,
                    device=feat.device,
                    dtype=self.pos_encoder.weight.dtype,
                )
                pos_emb = self.pos_encoder(pos_grid).to(feat.dtype)
                if self.debug_nan and self.is_main_process and not self._nan_reported:
                    if (not torch.isfinite(pos_grid).all()) or (not torch.isfinite(pos_emb).all()) or (not torch.isfinite(feat).all()):
                        pg_stats = self._tensor_stats(pos_grid)
                        pe_stats = self._tensor_stats(pos_emb)
                        f_stats = self._tensor_stats(feat)
                        print(
                            f"[nan] art pos grid/emb "
                            f"box={box_xy} raw=({raw_h},{raw_w}) feat_hw=({h},{w}) "
                            f"pos_grid_minmax={pg_stats} pos_emb_minmax={pe_stats} feat_minmax={f_stats}",
                            flush=True,
                        )
                        self._nan_reported = True
                feat = feat + pos_emb
                self._nan_check("forward_pixelrefer/feat_with_pos", feat)

            feat = feat.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
            mask_in = mask_tensor.unsqueeze(0).unsqueeze(0)
            masked_tokens = self.mask_pooling_pixelrefer(feat, mask_in, self.num_tokens)  # [N, D] (variable)
            self._nan_check("forward_pixelrefer/masked_tokens", masked_tokens)

            if masked_tokens.dtype != feat_map.dtype:
                masked_tokens = masked_tokens.to(feat_map.dtype)

            if masked_tokens.numel() > 0:
                proj_dtype = self.proj[0].weight.dtype
                if masked_tokens.dtype != proj_dtype:
                    masked_tokens = masked_tokens.to(proj_dtype)
                masked_tokens = self.proj(masked_tokens)
                if masked_tokens.dtype != feat_map.dtype:
                    masked_tokens = masked_tokens.to(feat_map.dtype)
                self._nan_check("forward_pixelrefer/proj_tokens", masked_tokens)

            obj_tokens_list.append(masked_tokens)
            token_masks.append(torch.ones(masked_tokens.shape[0], device=feat_map.device, dtype=torch.bool))

        max_n = max((t.shape[0] for t in obj_tokens_list), default=0)
        K = len(obj_tokens_list)
        D = self.hidden_size
        obj_tokens = torch.zeros((K, max_n, D), device=feature_maps[0].device, dtype=feature_maps[0].dtype)
        obj_token_mask = torch.zeros((K, max_n), device=feature_maps[0].device, dtype=torch.bool)
        for i, t in enumerate(obj_tokens_list):
            n = t.shape[0]
            if n == 0:
                continue
            obj_tokens[i, :n] = t
            obj_token_mask[i, :n] = True

        if return_valid:
            valid_tensor = torch.tensor(valid_list, dtype=torch.bool, device=obj_tokens.device)
            if return_token_mask:
                return obj_tokens, valid_tensor, obj_token_mask
            return obj_tokens, valid_tensor
        if return_token_mask:
            return obj_tokens, obj_token_mask
        return obj_tokens

    def kmeans_sampling(self, tokens, k):
        """
        Fast k-means style sampling to get k representative tokens

        Args:
            tokens: [N, D] - all tokens in masked region
            k: number of tokens to sample

        Returns:
            sampled: [k, D]
        """
        orig_dtype = tokens.dtype
        if tokens.dtype in (torch.bfloat16, torch.float16):
            tokens = tokens.float()

        N, D = tokens.shape

        if N <= k:
            padding = torch.zeros(k - N, D, device=tokens.device, dtype=tokens.dtype)
            out = torch.cat([tokens, padding], dim=0)
            return out.to(orig_dtype) if out.dtype != orig_dtype else out

        indices = torch.randperm(N, device=tokens.device)[:k]
        centroids = tokens[indices]

        for _ in range(3):
            dists = torch.cdist(tokens.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)  # [N, k]
            assignments = dists.argmin(dim=1)  # [N]
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    centroids[i] = tokens[mask].mean(dim=0)

        return centroids.to(orig_dtype) if centroids.dtype != orig_dtype else centroids

    def forward(self, feature_maps, masks_list, return_valid=False, return_token_mask=False):
        """
        Process batch with dynamic resolution

        Args:
            feature_maps: List of [H_i, W_i, D] (length B) - feature maps with possibly different sizes
            masks_list: List of [K_i, H_orig, W_orig] (length B) - masks in original resolution

        Returns:
            object_tokens: List of [K_i, num_tokens, D] (length B)
        """
        batch_tokens = []
        batch_valid = []
        batch_token_masks = []

        self._nan_reported = False
        for feat_map, masks in zip(feature_maps, masks_list):
            H, W, D = feat_map.shape
            self._nan_check("forward/feat_map", feat_map)

            # Convert masks to tensor if needed
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks).float()
            elif not isinstance(masks, torch.Tensor):
                raise ValueError(f"Unsupported mask type: {type(masks)}")

            K = masks.shape[0]

            if K == 0:
                empty_tokens = torch.zeros((0, 0, D), device=feat_map.device, dtype=feat_map.dtype)
                empty_token_mask = torch.zeros((0, 0), device=feat_map.device, dtype=torch.bool)
                batch_tokens.append(empty_tokens)
                batch_token_masks.append(empty_token_mask)
                if return_valid:
                    batch_valid.append(torch.zeros((0,), dtype=torch.bool, device=feat_map.device))
                continue

            obj_tokens_list = []
            obj_token_masks = []

            valid_list = []
            for k in range(K):
                mask = masks[k]  # [H_orig, W_orig]
                mask_tensor = mask.float().to(feat_map.device)
                mask_bool = mask_tensor > 0.5
                has_fg = bool(mask_bool.any().item())
                valid_list.append(has_fg)

                if not has_fg:
                    obj_tokens_list.append(torch.zeros((0, D), device=feat_map.device, dtype=feat_map.dtype))
                    obj_token_masks.append(torch.zeros((0,), device=feat_map.device, dtype=torch.bool))
                    continue

                ys, xs = torch.where(mask_bool)
                min_row = int(ys.min().item())
                max_row = int(ys.max().item())
                min_col = int(xs.min().item())
                max_col = int(xs.max().item())

                pad = self.patch_size * 2
                raw_h, raw_w = mask_tensor.shape
                top = max(0, min_row - pad)
                left = max(0, min_col - pad)
                bottom = min(raw_h - 1, max_row + pad)
                right = min(raw_w - 1, max_col + pad)
                box_xy = (top, left, bottom, right)

                crop_mask = mask_tensor[top:bottom + 1, left:right + 1]
                mask_h = crop_mask.shape[0]
                mask_w = crop_mask.shape[1]

                mask_sum = float(mask_bool.sum().item())
                scale_rate = math.ceil(math.sqrt(196 * self.num_tokens / max(mask_sum, 1.0)))
                if scale_rate == 1:
                    if (mask_sum / 196) > 100:
                        scale_rate = math.sqrt((mask_sum / 196) / 100)
                        scale_rate = 1 / scale_rate

                resize_h = math.ceil((mask_h * scale_rate) / self.patch_size) * self.patch_size
                resize_w = math.ceil((mask_w * scale_rate) / self.patch_size) * self.patch_size
                resize_h = max(resize_h, self.patch_size)
                resize_w = max(resize_w, self.patch_size)

                mask_resized = F.interpolate(
                    crop_mask.unsqueeze(0).unsqueeze(0),
                    size=(resize_h // self.patch_size, resize_w // self.patch_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

                # Crop feature map to bbox and resize to mask grid
                scale_y = H / max(raw_h, 1)
                scale_x = W / max(raw_w, 1)
                feat_top = int(top * scale_y)
                feat_left = int(left * scale_x)
                feat_bottom = int((bottom + 1) * scale_y)
                feat_right = int((right + 1) * scale_x)
                feat_top = max(0, min(feat_top, H - 1))
                feat_left = max(0, min(feat_left, W - 1))
                feat_bottom = max(feat_top + 1, min(feat_bottom, H))
                feat_right = max(feat_left + 1, min(feat_right, W))

                feat_crop = feat_map[feat_top:feat_bottom, feat_left:feat_right, :]
                if feat_crop.numel() == 0:
                    feat_crop = feat_map

                feat = feat_crop.permute(2, 0, 1).unsqueeze(0)  # [1, D, h, w]
                feat = F.interpolate(
                    feat,
                    size=mask_resized.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                if self.use_position_encoding:
                    pos_grid = self.generate_position_tensor(
                        mask_resized.shape[0],
                        mask_resized.shape[1],
                        box_xy,
                        raw_h,
                        raw_w,
                        device=feat_map.device,
                        dtype=self.pos_encoder.weight.dtype
                    )
                    pos_emb = self.pos_encoder(pos_grid).permute(2, 0, 1).unsqueeze(0)
                    if pos_emb.dtype != feat.dtype:
                        pos_emb = pos_emb.to(feat.dtype)
                    feat = feat + pos_emb
                    self._nan_check("forward/feat_with_pos", feat)

                mask_in = mask_resized.unsqueeze(0).unsqueeze(0)
                masked_tokens = self.mask_pooling_pixelrefer(feat, mask_in, self.num_tokens)  # [N, D] (variable)
                self._nan_check("forward/masked_tokens", masked_tokens)

                if masked_tokens.dtype != feat_map.dtype:
                    masked_tokens = masked_tokens.to(feat_map.dtype)

                if masked_tokens.numel() > 0:
                    proj_dtype = self.proj[0].weight.dtype
                    if masked_tokens.dtype != proj_dtype:
                        masked_tokens = masked_tokens.to(proj_dtype)
                    masked_tokens = self.proj(masked_tokens)
                    if masked_tokens.dtype != feat_map.dtype:
                        masked_tokens = masked_tokens.to(feat_map.dtype)
                    self._nan_check("forward/proj_tokens", masked_tokens)

                obj_tokens_list.append(masked_tokens)
                obj_token_masks.append(torch.ones(masked_tokens.shape[0], device=feat_map.device, dtype=torch.bool))

            # Pad to max tokens per view
            max_n = max((t.shape[0] for t in obj_tokens_list), default=0)
            obj_tokens = torch.zeros((K, max_n, D), device=feat_map.device, dtype=feat_map.dtype)
            obj_token_mask = torch.zeros((K, max_n), device=feat_map.device, dtype=torch.bool)
            for i, t in enumerate(obj_tokens_list):
                n = t.shape[0]
                if n == 0:
                    continue
                obj_tokens[i, :n] = t
                obj_token_mask[i, :n] = True

            batch_tokens.append(obj_tokens)
            batch_token_masks.append(obj_token_mask)
            if return_valid:
                batch_valid.append(torch.tensor(valid_list, dtype=torch.bool, device=feat_map.device))

        if return_valid:
            if return_token_mask:
                return batch_tokens, batch_valid, batch_token_masks
            return batch_tokens, batch_valid
        if return_token_mask:
            return batch_tokens, batch_token_masks
        return batch_tokens


class GlobalMultiViewFusion(nn.Module):
    """
    Global Multi-View Fusion using Self-Attention
    All views interact simultaneously in a single attention operation
    """

    def __init__(self, hidden_size, num_heads=8, max_views=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_views = max_views

        self.view_pos_encoding = nn.Embedding(max_views, hidden_size)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, tokens_list, valid_list=None, token_masks_list=None):
        """
        Args:
            tokens_list: List[Tensor] - [num_views][K_i, N, D]
            valid_list: List[Tensor] - [num_views][K_i] - valid mask

        Returns:
            dict with:
                - global_feature: [B, D] - global fused feature for VQA
                - view_features: List[Tensor] - [num_views][D] - per-view features
                - attention_weights: Tensor - attention visualization
        """
        num_views = len(tokens_list)
        device = tokens_list[0].device

        all_tokens = []
        view_indices = []
        token_counts = []

        for v_idx, tokens in enumerate(tokens_list):
            K, N, D = tokens.shape
            tokens_flat = tokens.view(K * N, D)
            if token_masks_list is not None:
                mask = token_masks_list[v_idx].view(K * N)
                if mask.any():
                    tokens_flat = tokens_flat[mask]
                    num_tokens = tokens_flat.shape[0]
                else:
                    num_tokens = 0
            else:
                num_tokens = tokens_flat.shape[0]

            if num_tokens == 0:
                token_counts.append(0)
                continue

            view_pos = self.view_pos_encoding(
                torch.tensor([v_idx], device=device, dtype=torch.long)
            ).expand(num_tokens, -1)

            if tokens_flat.dtype != view_pos.dtype:
                view_pos = view_pos.to(tokens_flat.dtype)

            tokens_with_pos = tokens_flat + view_pos
            all_tokens.append(tokens_with_pos)
            view_indices.extend([v_idx] * num_tokens)
            token_counts.append(num_tokens)

        if len(all_tokens) == 0:
            zero = torch.zeros(tokens_list[0].shape[-1], device=device, dtype=tokens_list[0].dtype)
            view_features = [zero.clone() for _ in range(num_views)]
            return {
                'global_feature': zero,
                'view_features': view_features,
                'attention_weights': None
            }

        all_tokens = torch.cat(all_tokens, dim=0).unsqueeze(0)  # [1, total, D]

        attn_dtype = self.global_attn.in_proj_weight.dtype
        if all_tokens.dtype != attn_dtype:
            all_tokens = all_tokens.to(attn_dtype)

        attn_out, attn_weights = self.global_attn(
            query=all_tokens,
            key=all_tokens,
            value=all_tokens
        )

        tokens_updated = self.layer_norm1(all_tokens + attn_out)
        ffn_out = self.ffn(tokens_updated)
        tokens_updated = self.layer_norm2(tokens_updated + ffn_out)

        tokens_updated = tokens_updated.squeeze(0)  # [total, D]

        view_features = []
        start = 0
        for count in token_counts:
            view_tokens = tokens_updated[start:start + count]
            if count > 0:
                view_feat = view_tokens.mean(dim=0)
            else:
                view_feat = torch.zeros(tokens_updated.shape[-1], device=tokens_updated.device, dtype=tokens_updated.dtype)
            view_features.append(view_feat)
            start += count

        global_feature = torch.stack(view_features, dim=0).mean(dim=0)

        return {
            'global_feature': global_feature,
            'view_features': view_features,
            'attention_weights': attn_weights
        }


class OCVA(nn.Module):
    """
    Object-Centric Cross-View Aligner
    Combines Cross-Attention (Scheme A) and Contrastive Learning (Scheme B)

    Returns both fused features (for LLM) and contrastive embeddings (for InfoNCE loss)
    """

    def __init__(self, hidden_size, num_heads=8, contrast_dim=256, attn_fp32=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.contrast_dim = contrast_dim
        self.attn_fp32 = bool(attn_fp32)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.contrast_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, contrast_dim)
        )

    def pad_tokens(self, tokens_list, valid_list=None, token_mask_list=None):
        """
        Pad variable-length tokens to same K and N

        Args:
            tokens_list: List of [K_i, N_i, D]
            token_mask_list: Optional List of [K_i, N_i] (True=valid token)

        Returns:
            padded: [B, max_K, max_N, D]
            valid_mask: [B, max_K] - indicates which objects are valid
            token_mask: [B, max_K, max_N] - indicates which tokens are valid
        """
        B = len(tokens_list)
        max_K = max(tokens.shape[0] for tokens in tokens_list)
        max_N = max(tokens.shape[1] for tokens in tokens_list)
        D = tokens_list[0].shape[2]

        padded = torch.zeros(B, max_K, max_N, D, device=tokens_list[0].device, dtype=tokens_list[0].dtype)
        valid_mask = torch.zeros(B, max_K, dtype=torch.bool, device=tokens_list[0].device)
        token_mask = torch.zeros(B, max_K, max_N, dtype=torch.bool, device=tokens_list[0].device)

        for i, tokens in enumerate(tokens_list):
            K_i = tokens.shape[0]
            N_i = tokens.shape[1]
            padded[i, :K_i, :N_i] = tokens
            if valid_list is None:
                valid_mask[i, :K_i] = True
            else:
                valid_mask[i, :K_i] = valid_list[i].to(valid_mask.device)
            if token_mask_list is None:
                token_mask[i, :K_i, :N_i] = True
            else:
                token_mask[i, :K_i, :N_i] = token_mask_list[i].to(token_mask.device)

        return padded, valid_mask, token_mask

    def forward(
        self,
        tokens_A_list,
        tokens_B_list,
        valid_A_list=None,
        valid_B_list=None,
        token_mask_A_list=None,
        token_mask_B_list=None,
    ):
        """
        Args:
            tokens_A_list: List of [K_i, num_tokens, D] - Ego view tokens (length B)
            tokens_B_list: List of [K_i, num_tokens, D] - Exo view tokens (length B)

        Returns:
            dict with:
                - fused_features: [B, D] - Global pooled features
                - fused_object_features: [B, K, D] - Per-object fused features
                - ego_embeddings: [B, max_K, contrast_dim] - For InfoNCE loss
                - exo_embeddings: [B, max_K, contrast_dim] - For InfoNCE loss
                - valid_mask: [B, max_K] - Valid object mask
        """
        tokens_A, valid_mask_A, token_mask_A = self.pad_tokens(
            tokens_A_list, valid_A_list, token_mask_A_list
        )
        tokens_B, valid_mask_B, token_mask_B = self.pad_tokens(
            tokens_B_list, valid_B_list, token_mask_B_list
        )

        token_any_A = token_mask_A.any(dim=2)
        token_any_B = token_mask_B.any(dim=2)
        valid_mask = valid_mask_A & valid_mask_B & token_any_A & token_any_B

        B, K, N, D = tokens_A.shape

        if K == 0 or N == 0:
            device = tokens_A.device
            fused_features = torch.zeros(B, D, device=device)
            fused_object_features = torch.zeros(B, K, D, device=device)
            emb_A = torch.zeros(B, K, self.contrast_dim, device=device)
            emb_B = torch.zeros(B, K, self.contrast_dim, device=device)
            valid_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
            return {
                'fused_features': fused_features,
                'fused_object_features': fused_object_features,
                'ego_embeddings': emb_A,
                'exo_embeddings': emb_B,
                'valid_mask': valid_mask
            }

        tokens_A_flat = tokens_A.view(B * K, N, D)
        tokens_B_flat = tokens_B.view(B * K, N, D)
        token_mask_A_flat = token_mask_A.view(B * K, N)
        token_mask_B_flat = token_mask_B.view(B * K, N)

        # Ensure dtype matches attention weights (important for bf16 + DeepSpeed)
        orig_dtype = tokens_A_flat.dtype
        if self.attn_fp32:
            tokens_A_flat = tokens_A_flat.float()
            tokens_B_flat = tokens_B_flat.float()
        else:
            attn_dtype = self.cross_attn.in_proj_weight.dtype
            if tokens_A_flat.dtype != attn_dtype:
                tokens_A_flat = tokens_A_flat.to(attn_dtype)
            if tokens_B_flat.dtype != attn_dtype:
                tokens_B_flat = tokens_B_flat.to(attn_dtype)

        token_mask_B_flat_attn = token_mask_B_flat
        key_all_masked = ~token_mask_B_flat_attn.any(dim=1)
        if key_all_masked.any():
            token_mask_B_flat_attn = token_mask_B_flat_attn.clone()
            token_mask_B_flat_attn[key_all_masked, 0] = True
            tokens_B_flat = tokens_B_flat.clone()
            tokens_B_flat[key_all_masked, 0, :] = 0
        attn_out, attn_weights = self.cross_attn(
            query=tokens_A_flat,
            key=tokens_B_flat,
            value=tokens_B_flat,
            key_padding_mask=~token_mask_B_flat_attn
        )

        tokens_A_flat = self.layer_norm(tokens_A_flat + attn_out)
        ffn_out = self.ffn(tokens_A_flat)
        fused_tokens = self.layer_norm(tokens_A_flat + ffn_out)

        if self.attn_fp32 and fused_tokens.dtype != orig_dtype:
            fused_tokens = fused_tokens.to(orig_dtype)
        fused_tokens = fused_tokens.view(B, K, N, D)

        token_mask_f = token_mask_A.float().unsqueeze(-1)
        denom = token_mask_f.sum(dim=2).clamp(min=1.0)
        fused_pooled = (fused_tokens * token_mask_f).sum(dim=2) / denom
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()
        fused_global = (fused_pooled * valid_mask_expanded).sum(dim=1) / (valid_mask_expanded.sum(dim=1) + 1e-8)

        fusion_dtype = self.fusion[0].weight.dtype
        if fused_pooled.dtype != fusion_dtype:
            fused_pooled = fused_pooled.to(fusion_dtype)
        fused_object_features = self.fusion(fused_pooled)
        if fused_global.dtype != fusion_dtype:
            fused_global = fused_global.to(fusion_dtype)
        fused_features = self.fusion(fused_global)

        pooled_A = (tokens_A * token_mask_A.float().unsqueeze(-1)).sum(dim=2) / (token_mask_A.float().sum(dim=2).clamp(min=1.0).unsqueeze(-1))
        pooled_B = (tokens_B * token_mask_B.float().unsqueeze(-1)).sum(dim=2) / (token_mask_B.float().sum(dim=2).clamp(min=1.0).unsqueeze(-1))

        contrast_dtype = self.contrast_proj[0].weight.dtype
        if pooled_A.dtype != contrast_dtype:
            pooled_A = pooled_A.to(contrast_dtype)
        if pooled_B.dtype != contrast_dtype:
            pooled_B = pooled_B.to(contrast_dtype)
        emb_A = self.contrast_proj(pooled_A)
        emb_B = self.contrast_proj(pooled_B)

        emb_A = F.normalize(emb_A, dim=-1)
        emb_B = F.normalize(emb_B, dim=-1)

        return {
            'fused_features': fused_features,
            'fused_object_features': fused_object_features,
            'fused_tokens': fused_tokens,
            'fused_token_mask': token_mask_A,
            'ego_embeddings': emb_A,
            'exo_embeddings': emb_B,
            'valid_mask': valid_mask
        }


if __name__ == "__main__":
    # Test modules with dynamic resolution
    print("Testing MaskPooling with dynamic resolution...")
    mask_pooling = MaskPooling()
    features = torch.randn(14, 14, 768)
    mask = np.random.randint(0, 2, (448, 448)).astype(np.float32)
    pooled = mask_pooling(features, mask)
    print(f"✓ MaskPooling: feat={features.shape}, mask={mask.shape} -> {pooled.shape}")

    print("\nTesting ART with variable objects and dynamic resolution...")
    art = ART(hidden_size=768, num_tokens=10)

    # Batch with 2 samples, different number of objects
    feat_maps = [
        torch.randn(14, 14, 768),  # Sample 1: 14x14
        torch.randn(14, 14, 768),  # Sample 2: 14x14 (could be different)
    ]

    masks_list = [
        torch.randint(0, 2, (3, 448, 448)).float(),  # Sample 1: 3 objects
        torch.randint(0, 2, (2, 448, 448)).float(),  # Sample 2: 2 objects
    ]

    obj_tokens = art(feat_maps, masks_list)
    print(f"✓ ART:")
    for i, tokens in enumerate(obj_tokens):
        print(f"  Sample {i}: {tokens.shape}")

    print("\nTesting OCVA...")
    ocva = OCVA(hidden_size=768)

    # Different K per sample
    tokens_A_list = [
        torch.randn(3, 10, 768),  # Sample 1: 3 objects
        torch.randn(2, 10, 768),  # Sample 2: 2 objects
    ]
    tokens_B_list = [
        torch.randn(3, 10, 768),
        torch.randn(2, 10, 768),
    ]

    output = ocva(tokens_A_list, tokens_B_list)
    print(f"✓ CrossView:")
    print(f"  Fused features: {output['fused_features'].shape}")
    print(f"  Ego embeddings: {output['ego_embeddings'].shape}")
    print(f"  Exo embeddings: {output['exo_embeddings'].shape}")
    print(f"  Valid mask: {output['valid_mask'].shape}")
    print(f"  Valid mask content:\n{output['valid_mask']}")
