"""
Ablation modules for CrossViewer.

MeanPoolART:
    drop-in replacement for ART that skips scale-adaptive crop and k-means,
    using plain mask mean-pooling instead.

IdentityOCVA:
    drop-in replacement for OCVA that skips cross-attention
    entirely.  ART tokens are mean-pooled per object and returned in the same
    dict format, so all downstream code (VQA, InfoNCE, region injection) works
    unchanged.  InfoNCE embeddings are zeroed out so the contrastive loss is
    effectively disabled.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MeanPoolART(nn.Module):
    """
    Ablation: w/o ART.
    Replaces scale-adaptive crop + k-means with a single mean-pool over the
    masked region.  Output shape is [K, 1, D] (one token per object) instead
    of [K, num_tokens, D], which is still compatible with OCVA
    and the rest of the pipeline.
    """

    def __init__(self, hidden_size, num_tokens=10, use_position_encoding=True,
                 patch_size=14, debug_nan=False, is_main_process=True):
        super().__init__()
        self.hidden_size = hidden_size
        # num_tokens / patch_size kept for API compatibility, not used
        self.num_tokens = num_tokens
        self.patch_size = patch_size

        # Same projection as ART so weight shapes match if loading a checkpoint
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # pos_encoder stub — kept so freeze_pos_encoder in model.py doesn't crash
        if use_position_encoding:
            self.pos_encoder = nn.Linear(2, hidden_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_pool_mask(self, feat_map, mask_tensor):
        """
        Args:
            feat_map:    [H, W, D]  (float, on device)
            mask_tensor: [H_m, W_m] (float, on device)
        Returns:
            token: [1, D]
        """
        H, W, D = feat_map.shape
        if mask_tensor.shape != (H, W):
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='nearest'
            ).squeeze()
        mask_bin = (mask_tensor > 0.5).float()
        denom = mask_bin.sum().clamp(min=1.0)
        pooled = (feat_map * mask_bin.unsqueeze(-1)).sum(dim=(0, 1)) / denom  # [D]
        return pooled.unsqueeze(0)  # [1, D]

    # ------------------------------------------------------------------
    # forward — mirrors ART.forward signature
    # ------------------------------------------------------------------

    def forward(self, feature_maps, masks_list, return_valid=False, return_token_mask=False):
        """
        Args:
            feature_maps: List[[H_i, W_i, D]]
            masks_list:   List[[K_i, H_orig, W_orig]]
        Returns:
            batch_tokens: List[[K_i, 1, D]]
            (optionally) batch_valid, batch_token_masks
        """
        batch_tokens = []
        batch_valid = []
        batch_token_masks = []

        for feat_map, masks in zip(feature_maps, masks_list):
            H, W, D = feat_map.shape

            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks).float()
            K = masks.shape[0]

            if K == 0:
                batch_tokens.append(torch.zeros((0, 1, D), device=feat_map.device, dtype=feat_map.dtype))
                batch_token_masks.append(torch.zeros((0, 1), device=feat_map.device, dtype=torch.bool))
                if return_valid:
                    batch_valid.append(torch.zeros((0,), dtype=torch.bool, device=feat_map.device))
                continue

            obj_tokens_list = []
            valid_list = []

            for k in range(K):
                mask_tensor = masks[k].float().to(feat_map.device)
                has_fg = bool((mask_tensor > 0.5).any().item())
                valid_list.append(has_fg)

                token = self._mean_pool_mask(feat_map, mask_tensor)  # [1, D]

                proj_dtype = self.proj[0].weight.dtype
                token = token.to(proj_dtype)
                token = self.proj(token).to(feat_map.dtype)
                obj_tokens_list.append(token)

            obj_tokens = torch.stack(obj_tokens_list, dim=0)  # [K, 1, D]
            obj_token_mask = torch.ones((K, 1), device=feat_map.device, dtype=torch.bool)

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

    # ------------------------------------------------------------------
    # forward_pixelrefer — mirrors ART.forward_pixelrefer signature
    # ------------------------------------------------------------------

    def forward_pixelrefer(self, feature_maps, masks_list, box_params_list,
                           return_valid=False, return_token_mask=False):
        """
        Args:
            feature_maps:    List[[H_i, W_i, D]]  cropped image features
            masks_list:      List[Tensor/np.ndarray]
            box_params_list: List[(bbox, raw_h, raw_w)]
        Returns:
            tokens: [K, 1, D]
        """
        if len(masks_list) == 0:
            device = self.proj[0].weight.device
            empty = torch.zeros((0, 1, self.hidden_size), device=device)
            empty_mask = torch.zeros((0, 1), device=device, dtype=torch.bool)
            if return_valid:
                if return_token_mask:
                    return empty, torch.zeros((0,), dtype=torch.bool, device=device), empty_mask
                return empty, torch.zeros((0,), dtype=torch.bool, device=device)
            if return_token_mask:
                return empty, empty_mask
            return empty

        obj_tokens_list = []
        valid_list = []

        for feat_map, mask, _ in zip(feature_maps, masks_list, box_params_list):
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).float().to(feat_map.device)
            else:
                mask_tensor = mask.float().to(feat_map.device)

            has_fg = bool((mask_tensor > 0.5).any().item())
            valid_list.append(has_fg)

            token = self._mean_pool_mask(feat_map, mask_tensor)  # [1, D]

            proj_dtype = self.proj[0].weight.dtype
            token = token.to(proj_dtype)
            token = self.proj(token).to(feat_map.dtype)
            obj_tokens_list.append(token)

        K = len(obj_tokens_list)
        D = self.hidden_size
        device = feature_maps[0].device
        dtype = feature_maps[0].dtype

        obj_tokens = torch.stack(obj_tokens_list, dim=0)  # [K, 1, D]
        obj_token_mask = torch.ones((K, 1), device=device, dtype=torch.bool)

        if return_valid:
            valid_tensor = torch.tensor(valid_list, dtype=torch.bool, device=device)
            if return_token_mask:
                return obj_tokens, valid_tensor, obj_token_mask
            return obj_tokens, valid_tensor
        if return_token_mask:
            return obj_tokens, obj_token_mask
        return obj_tokens


class IdentityOCVA(nn.Module):
    """
    Ablation: w/o CrossView Attention.
    Drop-in replacement for OCVA.  Cross-attention is removed;
    each view's ART tokens are mean-pooled independently and passed straight
    through a fusion MLP (same architecture as the original fusion layer).
    Output dict keys are identical to OCVA so no other code
    needs to change.  InfoNCE embeddings are zeroed out (contrastive loss
    becomes 0).
    """

    def __init__(self, hidden_size, num_heads=8, contrast_dim=256, attn_fp32=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.contrast_dim = contrast_dim

        # Keep the same fusion + contrast_proj as OCVA so
        # checkpoint key names stay compatible if needed.
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
        # Stubs so model code that references these attributes doesn't crash.
        self.cross_attn = None
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    # ------------------------------------------------------------------
    # pad_tokens — identical to OCVA.pad_tokens
    # ------------------------------------------------------------------

    def pad_tokens(self, tokens_list, valid_list=None, token_mask_list=None):
        B = len(tokens_list)
        max_K = max(t.shape[0] for t in tokens_list)
        max_N = max(t.shape[1] for t in tokens_list)
        D = tokens_list[0].shape[2]
        device = tokens_list[0].device
        dtype = tokens_list[0].dtype

        padded = torch.zeros(B, max_K, max_N, D, device=device, dtype=dtype)
        valid_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)
        token_mask = torch.zeros(B, max_K, max_N, dtype=torch.bool, device=device)

        for i, tokens in enumerate(tokens_list):
            K_i, N_i = tokens.shape[0], tokens.shape[1]
            padded[i, :K_i, :N_i] = tokens
            if valid_list is None:
                valid_mask[i, :K_i] = True
            else:
                valid_mask[i, :K_i] = valid_list[i].to(device)
            if token_mask_list is None:
                token_mask[i, :K_i, :N_i] = True
            else:
                token_mask[i, :K_i, :N_i] = token_mask_list[i].to(device)

        return padded, valid_mask, token_mask

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens_A_list,
        tokens_B_list,
        valid_A_list=None,
        valid_B_list=None,
        token_mask_A_list=None,
        token_mask_B_list=None,
    ):
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
        device = tokens_A.device

        if K == 0 or N == 0:
            return {
                'fused_features': torch.zeros(B, D, device=device),
                'fused_object_features': torch.zeros(B, K, D, device=device),
                'fused_tokens': tokens_A,
                'fused_token_mask': token_mask_A,
                'ego_embeddings': torch.zeros(B, K, self.contrast_dim, device=device),
                'exo_embeddings': torch.zeros(B, K, self.contrast_dim, device=device),
                'valid_mask': torch.zeros(B, K, dtype=torch.bool, device=device),
            }

        # Mean-pool each view's tokens per object independently, then average.
        # tokens_B is already reordered to match tokens_A per object (done in model.py).
        # This keeps both views' information while removing cross-attention's
        # selective cross-view interaction.
        token_mask_f_A = token_mask_A.float().unsqueeze(-1)                          # [B, K, N, 1]
        pooled_A = (tokens_A * token_mask_f_A).sum(dim=2) / token_mask_f_A.sum(dim=2).clamp(min=1.0)  # [B, K, D]

        token_mask_f_B = token_mask_B.float().unsqueeze(-1)
        pooled_B = (tokens_B * token_mask_f_B).sum(dim=2) / token_mask_f_B.sum(dim=2).clamp(min=1.0)  # [B, K, D]

        pooled_AB = (pooled_A + pooled_B) / 2                                        # [B, K, D]

        # Apply fusion MLP
        fusion_dtype = self.fusion[0].weight.dtype
        fused_object_features = self.fusion(pooled_AB.to(fusion_dtype))              # [B, K, D]

        # Global feature: valid-masked average over objects
        valid_exp = valid_mask.unsqueeze(-1).float()
        fused_global = (fused_object_features * valid_exp).sum(dim=1) / (valid_exp.sum(dim=1) + 1e-8)
        fused_features = fused_global                                                 # [B, D]

        # fused_tokens: expand pooled_AB back to [B, K, N, D] for region injection
        fused_tokens = fused_object_features.unsqueeze(2).expand(B, K, N, D).to(tokens_A.dtype)

        # InfoNCE embeddings: zero out — contrastive loss will be 0
        ego_embeddings = torch.zeros(B, K, self.contrast_dim, device=device, dtype=tokens_A.dtype)
        exo_embeddings = torch.zeros(B, K, self.contrast_dim, device=device, dtype=tokens_A.dtype)

        return {
            'fused_features': fused_features,
            'fused_object_features': fused_object_features,
            'fused_tokens': fused_tokens,
            'fused_token_mask': token_mask_A,
            'ego_embeddings': ego_embeddings,
            'exo_embeddings': exo_embeddings,
            'valid_mask': valid_mask,
        }
