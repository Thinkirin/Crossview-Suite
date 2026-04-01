"""
CrossViewer: CrossViewer Model
End-to-end multi-view reasoning model with VLM
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .modules import OCVA, GlobalMultiViewFusion
from .modules_ablation import MeanPoolART
from .gpu_preprocess import qwen_vl_preprocess_torch, resize_image_mask_torch
from .losses import CombinedLoss


class CrossViewerModelNoART(nn.Module):
    """CrossViewer ablation that replaces ART with mean-pooled region tokens."""

    def __init__(self,
                 vision_encoder_path="Qwen/Qwen3-VL-8B-Instruct",
                 freeze_vision_encoder=True,
                 freeze_llm=True,
                 use_lora=False,
                 num_object_tokens=10,
                 num_cross_attn_heads=8,
                 contrast_dim=256,
                 temperature=0.07,
                 infonce_weight=1.0,
                 vqa_weight=0.5,
                 triplet_weight=0.0,
                 attn_implementation=None,
                 load_device=None,
                 low_cpu_mem_usage=True,
                 use_global_attention=False,
                 use_consistency_constraint=False,
                 use_supcon=True,
                 consistency_weight=0.1,
                 match_mode="gt",
                 region_source="fused",
                 pixelrefer_mode="full",
                 preprocess_on_gpu=False,
                 region_placeholder="<region>",
                 mask_num=32,
                 debug_pixelrefer=False,
                 debug_pixelrefer_max_steps=1,
                 debug_pixelrefer_max_masks=50,
                 debug_pixelrefer_every=1,
                 debug_pixelrefer_per_mask_preprocess=False,
                 debug_match=False,
                 debug_max_steps=1,
                 debug_nan=False,
                 freeze_pos_encoder=False,
                 unfreeze_lm_head=False):
        super().__init__()

        print("=" * 80)
        print("Initializing CrossViewerModel (ablation: w/o ART, using MeanPoolART)...")
        print("=" * 80)

        self._is_main_process = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0

        print("\n[1/5] Loading Qwen3-VL model...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": None
        }
        self._loaded_to_device = False
        if load_device is not None:
            model_kwargs["device_map"] = {"": int(load_device)}
            model_kwargs["low_cpu_mem_usage"] = bool(low_cpu_mem_usage)
            self._loaded_to_device = True
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        try:
            self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                vision_encoder_path,
                **model_kwargs
            )
        except Exception as e:
            if attn_implementation:
                msg = str(e)
                if "attn_implementation" in msg or "flash_attention_2" in msg:
                    print(f"  ⚠️  attn_implementation={attn_implementation} not available, falling back to default attention")
                    model_kwargs.pop("attn_implementation", None)
                    self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        vision_encoder_path,
                        **model_kwargs
                    )
                else:
                    raise
            else:
                raise
        self.processor = AutoProcessor.from_pretrained(vision_encoder_path)

        self.vision_encoder_module = self.qwen_model.visual
        self.llm = self.qwen_model.model

        if freeze_vision_encoder:
            for param in self.vision_encoder_module.parameters():
                param.requires_grad = False
            self.vision_encoder_module.eval()
            print("  ✓ Vision encoder frozen")

        if freeze_llm:
            self._freeze_llm_params()
            self.llm.eval()
            print("  ✓ LLM frozen")

        if use_lora:
            print("  ⚠ LoRA not implemented yet, using frozen LLM")

        hidden_size = self.qwen_model.config.vision_config.out_hidden_size
        llm_hidden_size = self.qwen_model.config.text_config.hidden_size

        print("\n[2/5] Initializing ART...")
        self.art = MeanPoolART(
            hidden_size=hidden_size,
            num_tokens=num_object_tokens,
            use_position_encoding=True,
            debug_nan=debug_nan,
            is_main_process=self._is_main_process,
        )
        if freeze_pos_encoder and hasattr(self.art, "pos_encoder"):
            for p in self.art.pos_encoder.parameters():
                p.requires_grad = False

        print("\n[3/5] Initializing Object-Centric Cross-View Aligner (OCVA)...")
        self.ocva = OCVA(
            hidden_size=hidden_size,
            num_heads=num_cross_attn_heads,
            contrast_dim=contrast_dim
        )

        self.use_global_attention = use_global_attention
        self.pixelrefer_mode = (pixelrefer_mode or "full").lower()
        self.preprocess_on_gpu = bool(preprocess_on_gpu)
        self.region_placeholder = region_placeholder
        self.region_token = "<REGION>"
        self.mask_num = int(mask_num)
        self.debug_pixelrefer = bool(debug_pixelrefer)
        self.debug_pixelrefer_max_steps = int(debug_pixelrefer_max_steps)
        self.debug_pixelrefer_max_masks = int(debug_pixelrefer_max_masks)
        self.debug_pixelrefer_every = max(1, int(debug_pixelrefer_every))
        self.debug_pixelrefer_per_mask_preprocess = bool(debug_pixelrefer_per_mask_preprocess)
        self._debug_pixelrefer_calls = 0
        if use_global_attention:
            print("\n[3.5/5] Initializing Global Multi-View Fusion...")
            self.global_fusion = GlobalMultiViewFusion(
                hidden_size=hidden_size,
                num_heads=num_cross_attn_heads,
                max_views=8
            )
            print("  ✓ Global attention enabled")
        else:
            self.global_fusion = None
            print("\n[3.5/5] Global attention disabled")

        print("\n[4/5] Initializing LLM Adapter...")
        self.llm_adapter = nn.Sequential(
            nn.Linear(hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            vision_encoder_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.region_token = "<REGION>"
        if self.region_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.region_token], special_tokens=True)
            self.qwen_model.resize_token_embeddings(len(self.tokenizer))
            if freeze_llm:
                self._freeze_llm_params()
        if unfreeze_lm_head and hasattr(self.qwen_model, "lm_head") and self.qwen_model.lm_head is not None:
            for param in self.qwen_model.lm_head.parameters():
                param.requires_grad = True
            print("  ✓ LLM lm_head unfrozen")
        self.region_token_id = self.tokenizer.convert_tokens_to_ids(self.region_token)
        self.region_token_default_num = num_object_tokens
        self.match_mode = match_mode
        self.region_source = region_source
        self.debug_match = bool(debug_match)
        self.debug_max_steps = int(debug_max_steps)
        self._debug_calls = 0
        self._debug_active = False
        self.debug_nan = bool(debug_nan)
        self._nan_reported = False

        print("\n[5/5] Initializing Loss...")
        self.criterion = CombinedLoss(
            temperature=temperature,
            info_nce_weight=infonce_weight,
            triplet_weight=triplet_weight,
            use_supcon=use_supcon
        )

        self.infonce_weight = infonce_weight
        self.vqa_weight = vqa_weight
        self.num_object_tokens = num_object_tokens
        self.use_consistency_constraint = use_consistency_constraint
        self.consistency_weight = consistency_weight

        if use_consistency_constraint:
            print(f"\n  ✓ Consistency constraint enabled (weight={consistency_weight})")
        else:
            print("\n  ✓ Consistency constraint disabled")

        print("\n" + "=" * 80)
        print("✓ CrossViewer Model initialized successfully")
        print("=" * 80 + "\n")

    def _freeze_llm_params(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        if hasattr(self.qwen_model, "lm_head"):
            for param in self.qwen_model.lm_head.parameters():
                param.requires_grad = False
        try:
            embed = self.qwen_model.get_input_embeddings()
            if embed is not None:
                for param in embed.parameters():
                    param.requires_grad = False
        except Exception:
            pass

    def _expand_region_tokens(self, text, region_token_counts):
        if not region_token_counts:
            return text
        expanded = text
        for count in region_token_counts:
            count = max(0, int(count))
            expanded = expanded.replace(
                self.region_placeholder,
                "[" + (self.region_token * count) + "]",
                1
            )
        return expanded

    def _infer_region_token_counts(self, region_refs, mask_nums_per_view, num_placeholders):
        if not region_refs or not mask_nums_per_view:
            return [self.mask_num] * num_placeholders
        counts = []
        for view_idx, obj_idx in region_refs:
            count = self.mask_num
            try:
                view_idx = int(view_idx)
                obj_idx = int(obj_idx)
            except Exception:
                counts.append(count)
                continue
            if 0 <= view_idx < len(mask_nums_per_view):
                view_counts = mask_nums_per_view[view_idx]
                if 0 <= obj_idx < len(view_counts):
                    try:
                        count = int(view_counts[obj_idx])
                    except Exception:
                        count = self.mask_num
            counts.append(count)
        if len(counts) < num_placeholders:
            counts.extend([self.mask_num] * (num_placeholders - len(counts)))
        return counts[:num_placeholders]

    def _preprocess_images_gpu(self, raw_images, device, debug=False, debug_prefix=""):
        if raw_images is None:
            return None, None
        return qwen_vl_preprocess_torch(
            raw_images,
            self.processor.image_processor,
            device,
            debug=debug,
            debug_prefix=debug_prefix,
            debug_per_item=debug,
            debug_max=self.debug_pixelrefer_max_masks,
            debug_every=self.debug_pixelrefer_every,
        )

    def _prepare_pixelrefer_gpu_full(self, raw_images, masks_per_view, device, debug=False, debug_prefix=""):
        if raw_images is None or masks_per_view is None:
            return None, None, None, None, None

        images_t = []
        for img in raw_images:
            if img is None:
                return None, None, None, None, None
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
            images_t.append(t.to(device, non_blocking=True))

        flat_masks = []
        mask_ids = []
        for v_idx, masks in enumerate(masks_per_view):
            for m in masks:
                flat_masks.append(m)
                mask_ids.append(v_idx)

        if len(flat_masks) == 0:
            return None, None, None, None, None

        max_pixels = getattr(self.processor.image_processor, "max_pixels", None)
        if max_pixels is None:
            max_pixels = 28 * 28 * 1280
        max_resize_pixels = int(max_pixels)

        resize_images, resize_masks, mask_nums, box_params = resize_image_mask_torch(
            images_t,
            flat_masks,
            mask_ids,
            patch_size=14,
            max_tokens=self.mask_num,
            debug=debug,
            debug_prefix=debug_prefix,
            debug_max=self.debug_pixelrefer_max_masks,
            debug_every=self.debug_pixelrefer_every,
            max_resize_pixels=max_resize_pixels,
        )

        additional_pixel_values, additional_grid_thw = qwen_vl_preprocess_torch(
            resize_images,
            self.processor.image_processor,
            device,
            debug=debug and self.debug_pixelrefer_per_mask_preprocess,
            debug_prefix=debug_prefix + "crop_",
            debug_per_item=debug and self.debug_pixelrefer_per_mask_preprocess,
            debug_max=self.debug_pixelrefer_max_masks,
            debug_every=self.debug_pixelrefer_every,
        )

        additional_masks = [[] for _ in range(len(masks_per_view))]
        additional_box_params = [[] for _ in range(len(masks_per_view))]
        additional_mask_nums = [[] for _ in range(len(masks_per_view))]
        for m, box, num, v_idx in zip(resize_masks, box_params, mask_nums, mask_ids):
            additional_masks[v_idx].append(m)
            additional_box_params[v_idx].append(box)
            additional_mask_nums[v_idx].append(num)

        return additional_pixel_values, additional_grid_thw, additional_masks, additional_box_params, additional_mask_nums

    def encode_images(self, pixel_values, grid_thw):
        """
        Encode images using Qwen3-VL vision encoder

        Args:
            pixel_values: Tensor from processor, shape depends on input
            grid_thw: Tensor [N, 3] (t, h, w for each image)

        Returns:
            feature_maps: List of [H_i, W_i, D]
        """
        # Move to device (NO processor call here - done in Dataset!)
        device = next(self.vision_encoder_module.parameters()).device
        pixel_values = pixel_values.to(device)
        grid_thw = grid_thw.to(device)

        with torch.set_grad_enabled(self.training and self.vision_encoder_module.training):
            outputs = self.vision_encoder_module(pixel_values, grid_thw=grid_thw)

        features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

        # Split features into per-image feature maps
        spatial_merge = self.qwen_model.config.vision_config.spatial_merge_size
        feature_maps = []
        start_idx = 0

        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            h_merged = int(h // spatial_merge)
            w_merged = int(w // spatial_merge)
            num_patches = int(t * h_merged * w_merged)

            img_features = features[start_idx:start_idx + num_patches]
            D = img_features.shape[-1]
            img_features = img_features.view(int(t), h_merged, w_merged, D)[0]

            feature_maps.append(img_features)
            start_idx += num_patches

        return feature_maps

    def _compute_consistency_loss(self, all_embeddings_per_sample):
        """
        Compute consistency constraint loss

        Args:
            all_embeddings_per_sample: List[List[Tensor]]
                Each element is a list of embeddings for one sample
                embeddings shape: [K, contrast_dim]

        Returns:
            consistency_loss: scalar tensor
        """
        total_loss = 0.0
        total_objects = 0

        for sample_embeddings in all_embeddings_per_sample:
            if len(sample_embeddings) == 0:
                continue

            # Stack all embeddings: [num_pairs*2, K, contrast_dim]
            stacked_embs = torch.stack(sample_embeddings, dim=0)  # [N, K, D]
            N, K, D = stacked_embs.shape

            # For each object k, compute consistency loss
            for k in range(K):
                # Get all embeddings for object k: [N, D]
                obj_k_embs = stacked_embs[:, k, :]  # [N, contrast_dim]

                # Compute center
                center_k = obj_k_embs.mean(dim=0)  # [contrast_dim]

                # Compute distance to center
                # Use cosine distance since embeddings are L2-normalized
                # cosine_dist = 1 - cosine_similarity
                obj_k_embs_norm = torch.nn.functional.normalize(obj_k_embs, dim=-1)
                center_k_norm = torch.nn.functional.normalize(center_k.unsqueeze(0), dim=-1)
                cosine_sim = torch.sum(obj_k_embs_norm * center_k_norm, dim=-1)  # [N]
                cosine_dist = 1 - cosine_sim  # [N]

                # Sum distances
                total_loss += cosine_dist.sum()
                total_objects += N

        if total_objects > 0:
            consistency_loss = total_loss / total_objects
        else:
            consistency_loss = torch.tensor(0.0, device=stacked_embs.device if len(all_embeddings_per_sample) > 0 else torch.device('cpu'))

        return consistency_loss

    def _debug_log(self, msg):
        if self._debug_active:
            print(msg)

    def _nan_check(self, name, tensor):
        if not self.debug_nan or not self._is_main_process or self._nan_reported:
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
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = tensor.device
        print(f"[nan] stage={name} shape={shape} dtype={dtype} device={device}", flush=True)
        self._nan_reported = True
        return True

    def _pool_object_tokens(self, tokens, token_mask=None):
        """
        Mean-pool tokens per object with optional token mask.
        """
        if tokens is None:
            return None
        if tokens.numel() == 0:
            return tokens.new_zeros((tokens.shape[0], tokens.shape[-1]))
        if token_mask is None:
            return tokens.mean(dim=1)
        mask_f = token_mask.to(tokens.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (tokens * mask_f).sum(dim=1) / denom

    def _compute_object_embeddings(self, tokens, token_mask=None):
        """
        Compute normalized contrastive embeddings for each object.
        """
        pooled = self._pool_object_tokens(tokens, token_mask)
        if pooled is None:
            return None
        proj_dtype = self.ocva.contrast_proj[0].weight.dtype
        if pooled.dtype != proj_dtype:
            pooled = pooled.to(proj_dtype)
        emb = self.ocva.contrast_proj(pooled)
        emb = F.normalize(emb, dim=-1)
        return emb

    def _match_objects(self, tokens_A, tokens_B, valid_A=None, valid_B=None,
                       token_mask_A=None, token_mask_B=None, mode="greedy"):
        """
        Match objects between two views using contrastive embeddings.

        Returns:
            match_ab: list length K_A_obj mapping A obj -> B obj (or -1)
            match_ba: list length K_B_obj mapping B obj -> A obj (or -1)
        """
        if tokens_A is None or tokens_B is None:
            return [], []

        K_A = tokens_A.shape[0]
        K_B = tokens_B.shape[0]
        K_A_obj = max(K_A - 1, 0)
        K_B_obj = max(K_B - 1, 0)
        if K_A_obj == 0 or K_B_obj == 0:
            return [-1] * K_A_obj, [-1] * K_B_obj

        tokens_A_obj = tokens_A[:K_A_obj]
        tokens_B_obj = tokens_B[:K_B_obj]
        token_mask_A_obj = token_mask_A[:K_A_obj] if token_mask_A is not None else None
        token_mask_B_obj = token_mask_B[:K_B_obj] if token_mask_B is not None else None

        valid_A_obj = (
            valid_A[:K_A_obj]
            if valid_A is not None
            else torch.ones(K_A_obj, device=tokens_A.device, dtype=torch.bool)
        )
        valid_B_obj = (
            valid_B[:K_B_obj]
            if valid_B is not None
            else torch.ones(K_B_obj, device=tokens_B.device, dtype=torch.bool)
        )

        if token_mask_A_obj is not None:
            valid_A_obj = valid_A_obj & token_mask_A_obj.any(dim=1)
        if token_mask_B_obj is not None:
            valid_B_obj = valid_B_obj & token_mask_B_obj.any(dim=1)

        a_idx = valid_A_obj.nonzero(as_tuple=False).squeeze(-1)
        b_idx = valid_B_obj.nonzero(as_tuple=False).squeeze(-1)
        if a_idx.numel() == 0 or b_idx.numel() == 0:
            return [-1] * K_A_obj, [-1] * K_B_obj

        emb_A = self._compute_object_embeddings(tokens_A_obj, token_mask_A_obj)
        emb_B = self._compute_object_embeddings(tokens_B_obj, token_mask_B_obj)
        if emb_A is None or emb_B is None:
            return [-1] * K_A_obj, [-1] * K_B_obj

        emb_A = emb_A[a_idx]
        emb_B = emb_B[b_idx]
        sim = emb_A @ emb_B.transpose(0, 1)

        match_pairs = []
        mode = (mode or "greedy").lower()
        if mode == "hungarian":
            try:
                from scipy.optimize import linear_sum_assignment
                sim_cpu = sim.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-sim_cpu)
                match_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
            except Exception:
                mode = "greedy"

        if mode == "greedy":
            num_a = sim.shape[0]
            num_b = sim.shape[1]
            flat = sim.reshape(-1)
            _, order = torch.sort(flat, descending=True)
            matched_a = set()
            matched_b = set()
            for idx in order.tolist():
                ai = idx // num_b
                bi = idx % num_b
                if ai in matched_a or bi in matched_b:
                    continue
                matched_a.add(ai)
                matched_b.add(bi)
                match_pairs.append((ai, bi))
                if len(matched_a) >= min(num_a, num_b):
                    break

        match_ab = [-1] * K_A_obj
        match_ba = [-1] * K_B_obj
        for ai, bi in match_pairs:
            a_orig = int(a_idx[ai].item())
            b_orig = int(b_idx[bi].item())
            match_ab[a_orig] = b_orig
            match_ba[b_orig] = a_orig

        return match_ab, match_ba

    def _reorder_tokens_by_match(self, tokens, valid, token_mask, match_idx, keep_global=True):
        """
        Reorder tokens to align with matched indices. Unmatched slots are zeroed.
        """
        K, N, D = tokens.shape
        K_obj = max(K - 1, 0) if keep_global else K

        out_tokens = tokens.new_zeros((K, N, D))
        out_valid = valid.new_zeros((K,), dtype=torch.bool) if valid is not None else None
        out_mask = token_mask.new_zeros((K, N), dtype=torch.bool) if token_mask is not None else None

        for i in range(K_obj):
            j = match_idx[i] if i < len(match_idx) else -1
            if j is None or j < 0 or j >= K_obj:
                if out_mask is not None and N > 0:
                    out_mask[i, 0] = True
                continue
            out_tokens[i] = tokens[j]
            if out_valid is not None:
                out_valid[i] = valid[j]
            if out_mask is not None:
                out_mask[i] = token_mask[j]

        if keep_global and K > 0:
            out_tokens[K - 1] = tokens[K - 1]
            if out_valid is not None:
                out_valid[K - 1] = valid[K - 1]
            if out_mask is not None:
                out_mask[K - 1] = token_mask[K - 1]

        return out_tokens, out_valid, out_mask

    def _build_region_embeddings_from_fused(self, fused_A, fused_B, region_refs, region_token_counts):
        """
        Build region embeddings from fused per-object features.
        """
        if not region_refs:
            return None
        if fused_A is None and fused_B is None:
            return None

        region_chunks = []
        default_count = 1
        feat_dim = fused_A.shape[-1] if fused_A is not None else fused_B.shape[-1]
        device = fused_A.device if fused_A is not None else fused_B.device
        dtype = fused_A.dtype if fused_A is not None else fused_B.dtype

        for idx, ref in enumerate(region_refs):
            if not isinstance(ref, (list, tuple)) or len(ref) < 2:
                continue
            view_idx, obj_idx = ref[0], ref[1]
            try:
                view_idx = int(view_idx)
                obj_idx = int(obj_idx)
            except Exception:
                continue

            count = default_count
            if region_token_counts is not None and idx < len(region_token_counts):
                try:
                    count = int(region_token_counts[idx])
                except Exception:
                    count = default_count
            count = max(0, count)

            if count == 0:
                region_chunks.append(torch.zeros((0, feat_dim), device=device, dtype=dtype))
                continue

            if view_idx == 0:
                src = fused_A
            elif view_idx == 1:
                src = fused_B
            else:
                src = None

            if src is None or obj_idx < 0 or obj_idx >= src.shape[0]:
                region_chunks.append(torch.zeros((count, feat_dim), device=device, dtype=dtype))
                continue

            vec = src[obj_idx].unsqueeze(0)
            if count == 1:
                region_chunks.append(vec)
            else:
                region_chunks.append(vec.expand(count, -1))

        if not region_chunks:
            return None
        return torch.cat(region_chunks, dim=0)

    def _build_region_embeddings(self, sample_tokens, sample_token_masks, region_refs, region_token_counts):
        """
        Build region embeddings (vision-space) aligned to <REGION> tokens.
        """
        if not region_refs:
            return None
        region_chunks = []
        default_count = int(self.region_token_default_num)
        for idx, ref in enumerate(region_refs):
            if not isinstance(ref, (list, tuple)) or len(ref) < 2:
                continue
            view_idx, obj_idx = ref[0], ref[1]
            try:
                view_idx = int(view_idx)
                obj_idx = int(obj_idx)
            except Exception:
                continue

            count = default_count
            if region_token_counts is not None and idx < len(region_token_counts):
                try:
                    count = int(region_token_counts[idx])
                except Exception:
                    count = default_count
            count = max(0, count)

            if view_idx < 0 or view_idx >= len(sample_tokens):
                if count > 0:
                    region_chunks.append(
                        torch.zeros((count, sample_tokens[0].shape[-1]), device=sample_tokens[0].device)
                    )
                continue

            tokens_view = sample_tokens[view_idx]
            if obj_idx < 0 or obj_idx >= tokens_view.shape[0]:
                if count > 0:
                    region_chunks.append(
                        torch.zeros((count, tokens_view.shape[-1]), device=tokens_view.device, dtype=tokens_view.dtype)
                    )
                continue

            token_vecs = tokens_view[obj_idx]  # [N, D]
            if sample_token_masks is not None:
                try:
                    token_mask = sample_token_masks[view_idx][obj_idx]
                    token_vecs = token_vecs[token_mask]
                except Exception:
                    pass

            if count == 0:
                region_chunks.append(token_vecs.new_zeros((0, token_vecs.shape[-1])))
                continue
            if token_vecs.shape[0] >= count:
                region_chunks.append(token_vecs[:count])
            else:
                pad = token_vecs.new_zeros((count - token_vecs.shape[0], token_vecs.shape[-1]))
                region_chunks.append(torch.cat([token_vecs, pad], dim=0))

        if not region_chunks:
            return None
        return torch.cat(region_chunks, dim=0)

    def _inject_region_embeddings(self, input_ids, text_embeds, region_embeds_per_sample):
        """
        Replace <REGION> token embeddings with region features.
        """
        if region_embeds_per_sample is None or self.region_token_id is None:
            return text_embeds
        if input_ids is None or text_embeds is None:
            return text_embeds
        for i in range(input_ids.shape[0]):
            if i >= len(region_embeds_per_sample):
                continue
            region_embeds = region_embeds_per_sample[i]
            if region_embeds is None:
                continue
            positions = (input_ids[i] == self.region_token_id).nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                continue
            n_tokens = positions.numel()
            if self._debug_active and i == 0:
                self._debug_log(
                    f"[debug] region_tokens_in_text={int(n_tokens)} region_embeds={int(region_embeds.shape[0])}"
                )
            if region_embeds.shape[0] < n_tokens:
                pad = region_embeds.new_zeros((n_tokens - region_embeds.shape[0], region_embeds.shape[1]))
                region_embeds = torch.cat([region_embeds, pad], dim=0)
            elif region_embeds.shape[0] > n_tokens:
                region_embeds = region_embeds[:n_tokens]
            if region_embeds.dtype != text_embeds.dtype:
                region_embeds = region_embeds.to(text_embeds.dtype)
            text_embeds[i, positions] = region_embeds
        return text_embeds

    def forward(self, pixel_values, image_grid_thw, masks=None, raw_images=None, questions=None, answers=None,
                compute_infonce=None, cutoff_len=None, use_all_views_for_vqa=False,
                use_all_views_for_infonce=False, target_indices=None, object_ids=None,
                region_refs=None, region_token_counts=None, match_mode=None, region_source=None,
                use_retrieval_inference=False, retrieval_topk=1,
                retrieval_source_view=0, retrieval_target_view=1,
                additional_pixel_values=None, additional_grid_thw=None,
                additional_masks=None, additional_box_params=None):
        """
        End-to-end forward pass with multi-view support

        Args:
            pixel_values: List of Tensors [B][sum_patches, D] (preprocessed in Dataset)
            image_grid_thw: List of Tensors [B][num_views, 3]
            masks: Optional List[List[np.ndarray]] - [B][num_views][K_i, H, W]
                   Used during training with Relations dataset
            questions: Optional List[str] - VQA questions [B] (one question per sample)
            answers: Optional List[int/str] - answer labels [B] for training
            compute_infonce: Optional List[bool] - whether to compute InfoNCE for each sample [B]
            target_indices: Optional List[int] - target object index per sample (0..K or K=global)
            object_ids: Optional List[List[str]] - object identity tokens per sample (length K, no global)
            region_refs: Optional List[List[Tuple[int,int]]] - per-sample list of (view_idx, obj_idx) for <REGION>
            region_token_counts: Optional List[List[int]] - per-sample token counts for each <REGION>
            match_mode: Optional[str] - "gt"|"greedy"|"hungarian"|"none" object alignment mode
            region_source: Optional[str] - "fused" or "art" region embedding source
            use_retrieval_inference: If True, infer target index via retrieval when target_indices is None
            retrieval_topk: Top-k candidates to keep for retrieval inference
            retrieval_source_view: Source view index for retrieval (query)
            retrieval_target_view: Target view index for retrieval (candidates)
            additional_pixel_values: Optional List[Tensor] cropped images for PixelRefer ART (Full)
            additional_grid_thw: Optional List[Tensor] grid metadata for cropped images
            additional_masks: Optional List[List[List[Tensor]]] resized masks per view
            additional_box_params: Optional List[List[List[Tuple]]] bbox params per view

        Returns:
            dict with:
                - loss: total loss (if training)
                - infonce_loss: contrastive loss
                - vqa_loss: VQA loss
                - answer_logits: predicted answer logits
                - embeddings: contrastive embeddings
        """
        self._nan_reported = False
        # Optional GPU preprocessing for main images
        if self.preprocess_on_gpu:
            need_preprocess = pixel_values is None or (
                isinstance(pixel_values, list) and all(v is None for v in pixel_values)
            )
            if need_preprocess:
                if raw_images is None or (isinstance(raw_images, list) and all(r is None for r in raw_images)):
                    raise ValueError("raw_images required for GPU preprocessing but not provided.")
                device = next(self.parameters()).device
                pixel_values = []
                image_grid_thw = []
                limit_steps = self.debug_pixelrefer_max_steps
                allow_debug = limit_steps <= 0 or self._debug_pixelrefer_calls < limit_steps
                debug_pre = self.debug_pixelrefer and self._is_main_process and allow_debug
                for b_idx, imgs in enumerate(raw_images):
                    prefix = f"[gpu_pre] b{b_idx} " if debug_pre else ""
                    pv, grid = self._preprocess_images_gpu(imgs, device, debug=debug_pre, debug_prefix=prefix)
                    pixel_values.append(pv)
                    image_grid_thw.append(grid)

        # Get batch size and per-sample num_views (can vary)
        B = len(pixel_values)
        num_views_list = [g.shape[0] for g in image_grid_thw]
        match_mode = (match_mode or self.match_mode or "gt").lower()
        region_source = (region_source or self.region_source or "fused").lower()
        self._debug_active = self.debug_match and self._debug_calls < max(self.debug_max_steps, 1)

        # 1. Flatten all pixel_values and grid_thw for batch encoding
        # pixel_values: [B][num_views_i, ...] -> [sum(num_views_i), ...]
        flat_pixel_values = torch.cat(pixel_values, dim=0)  # [sum(num_views_i), ...]
        flat_grid_thw = torch.cat(image_grid_thw, dim=0)    # [sum(num_views_i), 3]
        self._nan_check("flat_pixel_values", flat_pixel_values)

        # Extract visual features (NO processor call - already preprocessed!)
        feature_maps = self.encode_images(flat_pixel_values, flat_grid_thw)  # List of [H_i, W_i, D]
        self._nan_check("feature_maps", feature_maps)

        # PixelRefer Full path: use cropped images for ART if provided
        additional_mask_nums_gpu = None
        if self.pixelrefer_mode == "gpu_full":
            if raw_images is None or masks is None:
                if self._debug_active:
                    self._debug_log("[debug] gpu_full requested but raw_images/masks missing; fallback to mask-based")
            else:
                device = flat_pixel_values.device
                additional_pixel_values = []
                additional_grid_thw = []
                additional_masks = []
                additional_box_params = []
                additional_mask_nums_gpu = []
                limit_steps = self.debug_pixelrefer_max_steps
                allow_debug = limit_steps <= 0 or self._debug_pixelrefer_calls < limit_steps
                debug_pf = self.debug_pixelrefer and self._is_main_process and allow_debug
                for b_idx in range(B):
                    prefix = f"[gpu_pf] b{b_idx} " if debug_pf else ""
                    apv, ag, am, ab, amn = self._prepare_pixelrefer_gpu_full(
                        raw_images[b_idx],
                        masks[b_idx],
                        device,
                        debug=debug_pf,
                        debug_prefix=prefix,
                    )
                    additional_pixel_values.append(apv)
                    additional_grid_thw.append(ag)
                    additional_masks.append(am)
                    additional_box_params.append(ab)
                    additional_mask_nums_gpu.append(amn)
                if any(v is None for v in additional_pixel_values):
                    additional_pixel_values = None
                    additional_grid_thw = None
                    additional_masks = None
                    additional_box_params = None
                    additional_mask_nums_gpu = None

        use_pixelrefer = (
            additional_pixel_values is not None
            and additional_grid_thw is not None
            and additional_masks is not None
            and additional_box_params is not None
            and all(v is not None for v in additional_pixel_values)
            and all(v is not None for v in additional_grid_thw)
        )
        if self.pixelrefer_mode in ("gpu", "art", "mask"):
            use_pixelrefer = False

        if use_pixelrefer:
            flat_additional_pixel_values = torch.cat(additional_pixel_values, dim=0)
            flat_additional_grid_thw = torch.cat(additional_grid_thw, dim=0)
            additional_feature_maps = self.encode_images(
                flat_additional_pixel_values, flat_additional_grid_thw
            )  # List of [H_i, W_i, D] per mask crop
            self._nan_check("additional_feature_maps", additional_feature_maps)

            object_tokens_per_sample = []
            object_valid_per_sample = []
            object_token_masks_per_sample = []
            add_offset = 0
            for b_idx in range(B):
                sample_masks_per_view = additional_masks[b_idx]
                sample_box_params = additional_box_params[b_idx]
                total_masks = sum(len(v) for v in sample_masks_per_view)
                sample_feats = additional_feature_maps[add_offset:add_offset + total_masks]
                add_offset += total_masks

                view_tokens = []
                view_valid = []
                view_token_masks = []
                feat_idx = 0
                for v_idx, view_masks in enumerate(sample_masks_per_view):
                    count = len(view_masks)
                    feats_view = sample_feats[feat_idx:feat_idx + count]
                    feat_idx += count
                    box_view = sample_box_params[v_idx] if sample_box_params is not None else [((0, 0, 0, 0), 1, 1)] * count
                    tokens, valid, token_mask = self.art.forward_pixelrefer(
                        feats_view, view_masks, box_view, return_valid=True, return_token_mask=True
                    )
                    view_tokens.append(tokens)
                    view_valid.append(valid)
                    view_token_masks.append(token_mask)
                object_tokens_per_sample.append(view_tokens)
                object_valid_per_sample.append(view_valid)
                object_token_masks_per_sample.append(view_token_masks)
        else:
            # 2. Flatten masks similarly: [B][num_views][K_i, H, W] -> [B*num_views][K_i, H, W]
            if masks is not None:
                flat_masks = []
                for sample_masks in masks:
                    flat_masks.extend(sample_masks)
                object_tokens_flat, object_valid_flat, object_token_masks_flat = self.art(
                    feature_maps, flat_masks, return_valid=True, return_token_mask=True
                )  # List of [K_i, N_i, D], List of [K_i], List of [K_i, N_i]
            else:
                # No masks: treat each view as a single "object" with sampled spatial tokens
                object_tokens_flat = []
                object_valid_flat = []
                object_token_masks_flat = []
                for feat_map in feature_maps:
                    H, W, D = feat_map.shape
                    tokens = feat_map.view(H * W, D).unsqueeze(0)  # [1, H*W, D]
                    num_tokens = min(self.num_object_tokens, H * W)
                    indices = torch.linspace(0, H * W - 1, num_tokens, device=feat_map.device).long()
                    tokens = tokens[:, indices, :]  # [1, num_tokens, D]
                    object_tokens_flat.append(tokens)  # [1, num_tokens, D]
                    object_valid_flat.append(torch.ones(1, dtype=torch.bool, device=feat_map.device))
                    object_token_masks_flat.append(torch.ones(1, num_tokens, dtype=torch.bool, device=feat_map.device))

            # 3. Reshape tokens back to [B][num_views_i]
            # object_tokens_flat: [sum(num_views_i)] -> object_tokens_per_sample: [B][num_views_i]
            object_tokens_per_sample = []
            object_valid_per_sample = []
            object_token_masks_per_sample = []
            offset = 0
            for nv in num_views_list:
                sample_tokens = object_tokens_flat[offset:offset + nv]
                object_tokens_per_sample.append(sample_tokens)
                sample_valid = object_valid_flat[offset:offset + nv]
                object_valid_per_sample.append(sample_valid)
                sample_token_masks = object_token_masks_flat[offset:offset + nv]
                object_token_masks_per_sample.append(sample_token_masks)
                offset += nv
        self._nan_check("object_tokens_per_sample", object_tokens_per_sample)

        # Track per-sample global token index (last token after appending global mask)
        global_indices = []
        for sample_tokens in object_tokens_per_sample:
            if len(sample_tokens) == 0:
                global_indices.append(0)
            else:
                global_indices.append(sample_tokens[0].shape[0] - 1)

        view_lengths_per_sample = [
            [tokens.shape[0] for tokens in sample_tokens]
            for sample_tokens in object_tokens_per_sample
        ]

        match_ab_list = [None] * B
        match_ba_list = [None] * B
        if match_mode in ("greedy", "hungarian"):
            for b_idx, (sample_tokens, sample_valid, sample_token_masks) in enumerate(
                zip(object_tokens_per_sample, object_valid_per_sample, object_token_masks_per_sample)
            ):
                if len(sample_tokens) < 2:
                    continue
                tokens_A = sample_tokens[0]
                tokens_B = sample_tokens[1]
                valid_A = sample_valid[0] if sample_valid is not None else None
                valid_B = sample_valid[1] if sample_valid is not None else None
                token_mask_A = sample_token_masks[0] if sample_token_masks is not None else None
                token_mask_B = sample_token_masks[1] if sample_token_masks is not None else None
                match_ab, match_ba = self._match_objects(
                    tokens_A,
                    tokens_B,
                    valid_A=valid_A,
                    valid_B=valid_B,
                    token_mask_A=token_mask_A,
                    token_mask_B=token_mask_B,
                    mode=match_mode,
                )
                match_ab_list[b_idx] = match_ab
                match_ba_list[b_idx] = match_ba
            if self._debug_active and B > 0:
                match_ab = match_ab_list[0]
                match_ba = match_ba_list[0]
                view_lens = view_lengths_per_sample[0] if view_lengths_per_sample else []
                matched = sum(1 for m in match_ab if m is not None and m >= 0) if match_ab is not None else 0
                msg = (
                    f"[debug] match_mode={match_mode} view_lens={view_lens} "
                    f"match_ab_len={len(match_ab) if match_ab is not None else 0} "
                    f"matched={matched}"
                )
                self._debug_log(msg)
                try:
                    tokens_A = object_tokens_per_sample[0][0]
                    tokens_B = object_tokens_per_sample[0][1]
                    valid_A = object_valid_per_sample[0][0]
                    valid_B = object_valid_per_sample[0][1]
                    mask_A = object_token_masks_per_sample[0][0]
                    mask_B = object_token_masks_per_sample[0][1]
                    tokens_A_obj = tokens_A[:-1]
                    tokens_B_obj = tokens_B[:-1]
                    mask_A_obj = mask_A[:-1] if mask_A is not None else None
                    mask_B_obj = mask_B[:-1] if mask_B is not None else None
                    valid_A_obj = valid_A[:-1] if valid_A is not None else None
                    valid_B_obj = valid_B[:-1] if valid_B is not None else None

                    emb_A = self._compute_object_embeddings(tokens_A_obj, mask_A_obj)
                    emb_B = self._compute_object_embeddings(tokens_B_obj, mask_B_obj)
                    if emb_A is not None and emb_B is not None:
                        if valid_A_obj is not None:
                            emb_A = emb_A[valid_A_obj]
                        if valid_B_obj is not None:
                            emb_B = emb_B[valid_B_obj]
                        if emb_A.numel() > 0 and emb_B.numel() > 0:
                            sim = emb_A @ emb_B.transpose(0, 1)
                            sim_min = float(sim.min().item())
                            sim_mean = float(sim.mean().item())
                            sim_max = float(sim.max().item())
                            top1_vals = sim.max(dim=1).values
                            top1_mean = float(top1_vals.mean().item()) if top1_vals.numel() > 0 else 0.0
                            self._debug_log(
                                f"[debug] sim min/mean/max={sim_min:.4f}/{sim_mean:.4f}/{sim_max:.4f} "
                                f"top1_mean={top1_mean:.4f}"
                            )
                except Exception as exc:
                    self._debug_log(f"[debug] sim stats failed: {exc}")

        def _masked_mean(tokens, mask):
            if mask is None:
                return tokens.mean(dim=1)
            mask_f = mask.to(tokens.dtype).unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            return (tokens * mask_f).sum(dim=1) / denom

        # Retrieval-based target selection (inference only)
        retrieval_indices = None
        retrieval_scores = None
        if (not self.training) and masks is not None and use_retrieval_inference:
            retrieval_indices = []
            retrieval_scores = []
            inferred_targets = []
            for b_idx, (sample_tokens, sample_valid, global_idx) in enumerate(
                zip(object_tokens_per_sample, object_valid_per_sample, global_indices)
            ):
                if len(sample_tokens) <= max(retrieval_source_view, retrieval_target_view):
                    inferred_targets.append(global_idx)
                    retrieval_indices.append(None)
                    retrieval_scores.append(None)
                    continue

                tokens_src = sample_tokens[retrieval_source_view]
                tokens_tgt = sample_tokens[retrieval_target_view]
                valid_src = sample_valid[retrieval_source_view]
                valid_tgt = sample_valid[retrieval_target_view]

                if tokens_src.shape[0] <= 1 or tokens_tgt.shape[0] <= 1:
                    inferred_targets.append(global_idx)
                    retrieval_indices.append(None)
                    retrieval_scores.append(None)
                    continue

                tokens_src_obj = tokens_src[:-1]
                tokens_tgt_obj = tokens_tgt[:-1]
                valid_src_obj = valid_src[:-1] if valid_src is not None else None
                valid_tgt_obj = valid_tgt[:-1] if valid_tgt is not None else None

                if match_mode in ("greedy", "hungarian") and match_ab_list[b_idx] is not None:
                    src_idx = None
                    if target_indices is not None:
                        try:
                            ti = target_indices[b_idx]
                        except Exception:
                            ti = None
                        if ti is not None and 0 <= ti < tokens_src_obj.shape[0]:
                            if valid_src_obj is None or valid_src_obj[ti]:
                                src_idx = int(ti)

                    if src_idx is None:
                        if valid_src_obj is not None and valid_src_obj.any():
                            src_idx = int(valid_src_obj.nonzero(as_tuple=False)[0].item())
                        else:
                            src_idx = 0

                    match_ab = match_ab_list[b_idx]
                    if src_idx < len(match_ab) and match_ab[src_idx] >= 0:
                        inferred_targets.append(int(src_idx))
                    else:
                        inferred_targets.append(global_idx)
                    retrieval_indices.append(None)
                    retrieval_scores.append(None)
                    continue

                if tokens_src_obj.shape[0] == 0 or tokens_tgt_obj.shape[0] == 0:
                    inferred_targets.append(global_idx)
                    retrieval_indices.append(None)
                    retrieval_scores.append(None)
                    continue

                token_mask_src = object_token_masks_per_sample[b_idx][retrieval_source_view] if object_token_masks_per_sample is not None else None
                token_mask_tgt = object_token_masks_per_sample[b_idx][retrieval_target_view] if object_token_masks_per_sample is not None else None
                token_mask_src_obj = token_mask_src[:-1] if token_mask_src is not None else None
                token_mask_tgt_obj = token_mask_tgt[:-1] if token_mask_tgt is not None else None

                pooled_src = _masked_mean(tokens_src_obj, token_mask_src_obj)
                pooled_tgt = _masked_mean(tokens_tgt_obj, token_mask_tgt_obj)

                proj_dtype = self.ocva.contrast_proj[0].weight.dtype
                if pooled_src.dtype != proj_dtype:
                    pooled_src = pooled_src.to(proj_dtype)
                if pooled_tgt.dtype != proj_dtype:
                    pooled_tgt = pooled_tgt.to(proj_dtype)

                emb_src = F.normalize(self.ocva.contrast_proj(pooled_src), dim=-1)
                emb_tgt = F.normalize(self.ocva.contrast_proj(pooled_tgt), dim=-1)

                sim = torch.matmul(emb_src, emb_tgt.transpose(0, 1))  # [K_src, K_tgt]
                if valid_tgt_obj is not None and valid_tgt_obj.numel() > 0:
                    sim = sim.masked_fill(~valid_tgt_obj.unsqueeze(0), -1e9)

                src_idx = None
                if target_indices is not None:
                    try:
                        ti = target_indices[b_idx]
                    except Exception:
                        ti = None
                    if ti is not None and 0 <= ti < tokens_src_obj.shape[0]:
                        if valid_src_obj is None or valid_src_obj[ti]:
                            src_idx = int(ti)

                if src_idx is None:
                    if valid_src_obj is not None and valid_src_obj.any():
                        src_idx = int(valid_src_obj.nonzero(as_tuple=False)[0].item())
                    else:
                        src_idx = 0

                sim_src = sim[src_idx]
                topk = min(int(retrieval_topk), sim_src.numel()) if retrieval_topk else 0
                if topk <= 0:
                    inferred_targets.append(global_idx)
                    retrieval_indices.append(None)
                    retrieval_scores.append(None)
                    continue

                vals, idxs = torch.topk(sim_src, k=topk, largest=True)
                inferred_targets.append(int(idxs[0].item()))
                retrieval_indices.append(idxs.detach().cpu())
                retrieval_scores.append(vals.detach().cpu())

            if target_indices is None:
                target_indices = inferred_targets

        # 4. Object-Centric Cross-View Aligner (batch-wise to handle variable K)
        # Base path: use first 2 views; optional multi-view paths handled later
        tokens_A_list = []
        tokens_B_list = []
        valid_A_list = []
        valid_B_list = []
        token_mask_A_list = []
        token_mask_B_list = []
        tokens_A_list_rev = []
        tokens_B_list_rev = []
        valid_A_list_rev = []
        valid_B_list_rev = []
        token_mask_A_list_rev = []
        token_mask_B_list_rev = []
        for b_idx, (sample_tokens, sample_valid, sample_token_masks) in enumerate(
            zip(object_tokens_per_sample, object_valid_per_sample, object_token_masks_per_sample)
        ):
            if len(sample_tokens) < 2:
                raise ValueError(f"Sample has fewer than 2 views: {len(sample_tokens)}")
            tokens_A = sample_tokens[0]
            tokens_B = sample_tokens[1]
            valid_A = sample_valid[0] if sample_valid is not None else None
            valid_B = sample_valid[1] if sample_valid is not None else None
            token_mask_A = sample_token_masks[0] if sample_token_masks is not None else None
            token_mask_B = sample_token_masks[1] if sample_token_masks is not None else None

            if match_mode in ("greedy", "hungarian"):
                match_ab = match_ab_list[b_idx] or []
                match_ba = match_ba_list[b_idx] or []
                tokens_B_aligned, valid_B_aligned, token_mask_B_aligned = self._reorder_tokens_by_match(
                    tokens_B, valid_B, token_mask_B, match_ab
                )
                tokens_A_aligned, valid_A_aligned, token_mask_A_aligned = self._reorder_tokens_by_match(
                    tokens_A, valid_A, token_mask_A, match_ba
                )
            else:
                tokens_B_aligned, valid_B_aligned, token_mask_B_aligned = tokens_B, valid_B, token_mask_B
                tokens_A_aligned, valid_A_aligned, token_mask_A_aligned = tokens_A, valid_A, token_mask_A

            # A -> B (aligned to A order)
            tokens_A_list.append(tokens_A)
            tokens_B_list.append(tokens_B_aligned)
            valid_A_list.append(valid_A)
            valid_B_list.append(valid_B_aligned)
            token_mask_A_list.append(token_mask_A)
            token_mask_B_list.append(token_mask_B_aligned)

            # B -> A (aligned to B order)
            tokens_B_list_rev.append(tokens_B)
            tokens_A_list_rev.append(tokens_A_aligned)
            valid_B_list_rev.append(valid_B)
            valid_A_list_rev.append(valid_A_aligned)
            token_mask_B_list_rev.append(token_mask_B)
            token_mask_A_list_rev.append(token_mask_A_aligned)

        ocva_output = self.ocva(
            tokens_A_list, tokens_B_list, valid_A_list, valid_B_list, token_mask_A_list, token_mask_B_list
        )
        ocva_output_rev = None
        if region_source == "fused" and region_refs is not None:
            need_reverse = False
            for refs in region_refs:
                if not refs:
                    continue
                for ref in refs:
                    if isinstance(ref, (list, tuple)) and len(ref) >= 2:
                        try:
                            if int(ref[0]) == 1:
                                need_reverse = True
                                break
                        except Exception:
                            continue
                if need_reverse:
                    break
            if need_reverse:
                ocva_output_rev = self.ocva(
                    tokens_B_list_rev,
                    tokens_A_list_rev,
                    valid_B_list_rev,
                    valid_A_list_rev,
                    token_mask_B_list_rev,
                    token_mask_A_list_rev,
                )
        ego_emb = ocva_output['ego_embeddings']      # [B, max_K, contrast_dim]
        exo_emb = ocva_output['exo_embeddings']      # [B, max_K, contrast_dim]
        fused_features = ocva_output['fused_features']  # [B, D]
        fused_object_features = ocva_output['fused_object_features']  # [B, max_K, D]
        valid_mask = ocva_output['valid_mask']       # [B, max_K]
        self._nan_check("ego_emb", ego_emb)
        self._nan_check("exo_emb", exo_emb)
        self._nan_check("fused_features", fused_features)
        self._nan_check("fused_object_features", fused_object_features)
        if self._debug_active and B > 0:
            msg = (
                f"[debug] fused_object_features={tuple(fused_object_features.shape)} "
                f"fused_features={tuple(fused_features.shape)} "
                f"ego_emb={tuple(ego_emb.shape)} valid_sum={int(valid_mask.sum().item())}"
            )
            self._debug_log(msg)
            if ocva_output_rev is not None:
                rev_shape = tuple(ocva_output_rev['fused_object_features'].shape)
                self._debug_log(f"[debug] fused_object_features_rev={rev_shape}")
            try:
                view_lens = view_lengths_per_sample[0] if view_lengths_per_sample else []
                k_a = view_lens[0] if view_lens else fused_object_features.shape[1]
                fused_obj = fused_object_features[0][: max(k_a - 1, 0)]
                if fused_obj.numel() > 0:
                    norms = fused_obj.norm(dim=-1)
                    self._debug_log(
                        f"[debug] fused_obj_norm min/mean/max="
                        f"{float(norms.min().item()):.4f}/{float(norms.mean().item()):.4f}/{float(norms.max().item()):.4f}"
                    )
            except Exception as exc:
                self._debug_log(f"[debug] fused norm stats failed: {exc}")

        global_idx_tensor = torch.tensor(
            global_indices,
            device=valid_mask.device,
            dtype=torch.long
        )
        if valid_mask is not None and valid_mask.numel() > 0:
            valid_mask_infonce = valid_mask.clone()
            valid_mask_infonce[torch.arange(B, device=valid_mask.device), global_idx_tensor] = False
        else:
            valid_mask_infonce = valid_mask

        label_tensor = None
        if object_ids is not None and valid_mask is not None:
            if isinstance(object_ids, torch.Tensor):
                label_tensor = object_ids.to(valid_mask.device)
            else:
                label_tensor = torch.full(
                    (B, valid_mask.shape[1]),
                    -1,
                    device=valid_mask.device,
                    dtype=torch.long
                )
                label_map = {}
                next_label = 0
                for b_idx, ids in enumerate(object_ids):
                    if ids is None:
                        continue
                    for k_idx, obj_id in enumerate(ids):
                        if k_idx >= label_tensor.shape[1]:
                            break
                        if obj_id not in label_map:
                            label_map[obj_id] = next_label
                            next_label += 1
                        label_tensor[b_idx, k_idx] = label_map[obj_id]

        # Optional: use global multi-view fusion for VQA (overrides use_all_views_for_vqa)
        global_token_override = None
        if self.use_global_attention and questions is not None:
            # Use global self-attention to fuse all views
            fused_features_all = []
            for sample_tokens, sample_token_masks in zip(object_tokens_per_sample, object_token_masks_per_sample):
                if len(sample_tokens) < 2:
                    raise ValueError(f"Sample has fewer than 2 views: {len(sample_tokens)}")

                # Global fusion across all views
                global_output = self.global_fusion(sample_tokens, token_masks_list=sample_token_masks)
                fused_features_all.append(global_output['global_feature'])

            fused_features = torch.stack(fused_features_all, dim=0)  # [B, D]
            global_token_override = fused_features

        # Fallback: use all available views for VQA by averaging pairwise fusions
        elif use_all_views_for_vqa and questions is not None:
            fused_features_all = []
            for i, sample_tokens in enumerate(object_tokens_per_sample):
                if len(sample_tokens) < 2:
                    raise ValueError(f"Sample has fewer than 2 views: {len(sample_tokens)}")
                if len(sample_tokens) == 2:
                    fused_features_all.append(fused_features[i])
                    continue

                # Average fusion across all view pairs (both directions)
                pair_feats = []
                num_views = len(sample_tokens)
                for a in range(num_views):
                    for b in range(a + 1, num_views):
                        tokens_a = sample_tokens[a]
                        tokens_b = sample_tokens[b]
                        valid_a = object_valid_per_sample[i][a]
                        valid_b = object_valid_per_sample[i][b]
                        mask_a = object_token_masks_per_sample[i][a]
                        mask_b = object_token_masks_per_sample[i][b]

                        if match_mode in ("greedy", "hungarian"):
                            match_ab, match_ba = self._match_objects(
                                tokens_a, tokens_b,
                                valid_A=valid_a, valid_B=valid_b,
                                token_mask_A=mask_a, token_mask_B=mask_b,
                                mode=match_mode,
                            )
                            tokens_b_aligned, valid_b_aligned, mask_b_aligned = self._reorder_tokens_by_match(
                                tokens_b, valid_b, mask_b, match_ab
                            )
                            tokens_a_aligned, valid_a_aligned, mask_a_aligned = self._reorder_tokens_by_match(
                                tokens_a, valid_a, mask_a, match_ba
                            )
                        else:
                            tokens_b_aligned, valid_b_aligned, mask_b_aligned = tokens_b, valid_b, mask_b
                            tokens_a_aligned, valid_a_aligned, mask_a_aligned = tokens_a, valid_a, mask_a

                        out_ab = self.ocva(
                            [tokens_a],
                            [tokens_b_aligned],
                            [valid_a],
                            [valid_b_aligned],
                            [mask_a],
                            [mask_b_aligned],
                        )
                        pair_feats.append(out_ab['fused_features'][0])
                        out_ba = self.ocva(
                            [tokens_b],
                            [tokens_a_aligned],
                            [valid_b],
                            [valid_a_aligned],
                            [mask_b],
                            [mask_a_aligned],
                        )
                        pair_feats.append(out_ba['fused_features'][0])

                fused_features_all.append(torch.stack(pair_feats, dim=0).mean(dim=0))

            fused_features = torch.stack(fused_features_all, dim=0)
            global_token_override = fused_features

        # 5. Compute InfoNCE loss (if training with masks and compute_infonce=True)
        infonce_loss = None
        infonce_acc = None
        debug_n_valid = None
        if masks is not None and self.training:
            # Multi-view InfoNCE: average across all view pairs
            if use_all_views_for_infonce:
                per_sample_losses = []
                per_sample_accs = []
                per_sample_weights = []
                total_valid = 0
                # Collect all embeddings for consistency constraint
                all_embeddings_per_sample = [] if self.use_consistency_constraint else None

                for s_idx, sample_tokens in enumerate(object_tokens_per_sample):
                    if compute_infonce is not None and not compute_infonce[s_idx]:
                        continue
                    if len(sample_tokens) < 2:
                        continue

                    sample_valid = object_valid_per_sample[s_idx]
                    pair_losses = []
                    pair_accs = []
                    sample_embeddings = [] if self.use_consistency_constraint else None
                    num_views = len(sample_tokens)
                    for a in range(num_views):
                        for b in range(a + 1, num_views):
                            if a == 0 and b == 1:
                                emb_A = ego_emb[s_idx:s_idx + 1]
                                emb_B = exo_emb[s_idx:s_idx + 1]
                                vmask = valid_mask_infonce[s_idx:s_idx + 1]
                            else:
                                tokens_a = sample_tokens[a]
                                tokens_b = sample_tokens[b]
                                valid_a = sample_valid[a]
                                valid_b = sample_valid[b]
                                mask_a = object_token_masks_per_sample[s_idx][a]
                                mask_b = object_token_masks_per_sample[s_idx][b]

                                if match_mode in ("greedy", "hungarian"):
                                    match_ab, _ = self._match_objects(
                                        tokens_a, tokens_b,
                                        valid_A=valid_a, valid_B=valid_b,
                                        token_mask_A=mask_a, token_mask_B=mask_b,
                                        mode=match_mode,
                                    )
                                    tokens_b, valid_b, mask_b = self._reorder_tokens_by_match(
                                        tokens_b, valid_b, mask_b, match_ab
                                    )

                                pair_out = self.ocva(
                                    [tokens_a],
                                    [tokens_b],
                                    [valid_a],
                                    [valid_b],
                                    [mask_a],
                                    [mask_b],
                                )
                                emb_A = pair_out["ego_embeddings"]
                                emb_B = pair_out["exo_embeddings"]
                                vmask = pair_out["valid_mask"]
                                if vmask is not None and vmask.numel() > 0:
                                    vmask = vmask.clone()
                                    vmask[:, -1] = False

                            # Collect embeddings for consistency constraint
                            if self.use_consistency_constraint:
                                sample_embeddings.append(emb_A[0])  # [K, contrast_dim]
                                sample_embeddings.append(emb_B[0])  # [K, contrast_dim]

                            if vmask is not None:
                                total_valid += int(vmask.sum().item())
                            label_slice = label_tensor[s_idx:s_idx + 1] if label_tensor is not None else None
                            inf_loss, inf_dict = self.criterion(emb_A, emb_B, vmask, labels=label_slice)
                            pair_losses.append(inf_loss)
                            pair_accs.append(inf_dict.get("accuracy", 0.0))

                    if pair_losses:
                        per_sample_losses.append(torch.stack(pair_losses).mean())
                        per_sample_accs.append(
                            float(torch.tensor(pair_accs, device=fused_features.device).mean().item())
                        )
                        # Mildly upweight samples with more view pairs
                        per_sample_weights.append(float(len(pair_losses)) ** 0.5)

                        # Save embeddings for consistency constraint
                        if self.use_consistency_constraint and sample_embeddings:
                            all_embeddings_per_sample.append(sample_embeddings)

                if per_sample_losses:
                    losses = torch.stack(per_sample_losses)
                    weights = torch.tensor(
                        per_sample_weights,
                        device=fused_features.device,
                        dtype=losses.dtype
                    )
                    infonce_loss = (losses * weights).sum() / weights.sum()
                    infonce_loss = infonce_loss * self.infonce_weight
                    accs = torch.tensor(per_sample_accs, device=fused_features.device, dtype=weights.dtype)
                    infonce_acc = float((accs * weights).sum().item() / weights.sum().item())
                    debug_n_valid = total_valid
                else:
                    infonce_loss = torch.tensor(0.0, device=fused_features.device)
                    infonce_acc = 0.0
            else:
                # If compute_infonce is provided, only compute for specified samples
                if compute_infonce is not None:
                    # Filter samples where compute_infonce=True
                    compute_indices = [i for i, flag in enumerate(compute_infonce) if flag]
                    if len(compute_indices) > 0:
                        ego_emb_filtered = ego_emb[compute_indices]
                        exo_emb_filtered = exo_emb[compute_indices]
                        valid_mask_filtered = valid_mask_infonce[compute_indices]
                        debug_n_valid = int(valid_mask_filtered.sum().item()) if valid_mask_filtered is not None else None
                        label_filtered = label_tensor[compute_indices] if label_tensor is not None else None
                        infonce_loss, infonce_dict = self.criterion(
                            ego_emb_filtered, exo_emb_filtered, valid_mask_filtered, labels=label_filtered
                        )
                        infonce_loss = infonce_loss * self.infonce_weight
                        infonce_acc = infonce_dict.get('accuracy')
                    else:
                        infonce_loss = torch.tensor(0.0, device=fused_features.device)
                        infonce_acc = 0.0
                else:
                    # Compute for all samples (backward compatibility)
                    debug_n_valid = int(valid_mask_infonce.sum().item()) if valid_mask_infonce is not None else None
                    infonce_loss, infonce_dict = self.criterion(
                        ego_emb, exo_emb, valid_mask_infonce, labels=label_tensor
                    )
                    infonce_loss = infonce_loss * self.infonce_weight
                    infonce_acc = infonce_dict.get('accuracy')
        else:
            infonce_loss = torch.tensor(0.0, device=fused_features.device)
            infonce_acc = 0.0
            all_embeddings_per_sample = None

        # 5.5. Compute consistency constraint (if enabled and embeddings collected)
        consistency_loss = None
        if self.use_consistency_constraint and all_embeddings_per_sample and len(all_embeddings_per_sample) > 0:
            consistency_loss = self._compute_consistency_loss(all_embeddings_per_sample)
            consistency_loss = consistency_loss * self.consistency_weight
        else:
            consistency_loss = torch.tensor(0.0, device=fused_features.device)

        if self.debug_pixelrefer and self._is_main_process:
            if self.debug_pixelrefer_max_steps <= 0 or self._debug_pixelrefer_calls < self.debug_pixelrefer_max_steps:
                self._debug_pixelrefer_calls += 1

        # Region embeddings for PixelRefer-style <REGION> tokens (optional)
        if questions is not None and region_refs is not None:
            need_counts = (
                region_token_counts is None
                or (isinstance(region_token_counts, (list, tuple)) and all(r is None for r in region_token_counts))
            )
            if need_counts and additional_mask_nums_gpu is not None:
                region_token_counts = []
                for b_idx in range(B):
                    refs = region_refs[b_idx] if isinstance(region_refs, (list, tuple)) and b_idx < len(region_refs) else None
                    num_placeholders = questions[b_idx].count(self.region_placeholder) if questions and b_idx < len(questions) else 0
                    if num_placeholders > 0:
                        counts = self._infer_region_token_counts(refs, additional_mask_nums_gpu[b_idx], num_placeholders)
                    else:
                        counts = []
                    region_token_counts.append(counts)
            if region_token_counts is not None and questions is not None:
                new_questions = []
                for q, counts in zip(questions, region_token_counts):
                    if q is None or self.region_placeholder not in q:
                        new_questions.append(q)
                    else:
                        new_questions.append(self._expand_region_tokens(q, counts))
                questions = new_questions

        region_embeds_per_sample = None
        if questions is not None and region_refs is not None:
            region_embeds_per_sample = []
            for b_idx, sample_tokens in enumerate(object_tokens_per_sample):
                refs = None
                counts = None
                if isinstance(region_refs, (list, tuple)) and b_idx < len(region_refs):
                    refs = region_refs[b_idx]
                if isinstance(region_token_counts, (list, tuple)) and b_idx < len(region_token_counts):
                    counts = region_token_counts[b_idx]
                if region_source == "fused":
                    k_a = None
                    k_b = None
                    if view_lengths_per_sample and b_idx < len(view_lengths_per_sample):
                        view_lens = view_lengths_per_sample[b_idx]
                        if len(view_lens) > 0:
                            k_a = view_lens[0]
                        if len(view_lens) > 1:
                            k_b = view_lens[1]
                    if k_a is None:
                        k_a = fused_object_features.shape[1]
                    fused_tokens_a = None
                    fused_mask_a = None
                    if "fused_tokens" in ocva_output:
                        fused_tokens_a = ocva_output["fused_tokens"][b_idx][:k_a]
                        fused_mask_a = ocva_output.get("fused_token_mask")
                        if fused_mask_a is not None:
                            fused_mask_a = fused_mask_a[b_idx][:k_a]
                    fused_tokens_b = None
                    fused_mask_b = None
                    if ocva_output_rev is not None and k_b is not None and "fused_tokens" in ocva_output_rev:
                        fused_tokens_b = ocva_output_rev["fused_tokens"][b_idx][:k_b]
                        fused_mask_b = ocva_output_rev.get("fused_token_mask")
                        if fused_mask_b is not None:
                            fused_mask_b = fused_mask_b[b_idx][:k_b]

                    if fused_tokens_a is not None:
                        tokens_list = [fused_tokens_a]
                        mask_list = None
                        if fused_mask_a is not None and (fused_tokens_b is None or fused_mask_b is not None):
                            mask_list = [fused_mask_a]
                        if fused_tokens_b is not None:
                            tokens_list.append(fused_tokens_b)
                            if mask_list is not None:
                                mask_list.append(fused_mask_b)
                        region_feats = self._build_region_embeddings(
                            tokens_list,
                            mask_list,
                            refs,
                            counts,
                        )
                    else:
                        fused_a = fused_object_features[b_idx][:k_a]
                        fused_b = None
                        if ocva_output_rev is not None and k_b is not None:
                            fused_b = ocva_output_rev["fused_object_features"][b_idx][:k_b]
                        region_feats = self._build_region_embeddings_from_fused(
                            fused_a,
                            fused_b,
                            refs,
                            counts,
                        )
                else:
                    region_feats = self._build_region_embeddings(
                        sample_tokens,
                        object_token_masks_per_sample[b_idx] if object_token_masks_per_sample is not None else None,
                        refs,
                        counts
                    )
                if region_feats is None:
                    region_embeds_per_sample.append(None)
                    continue
                if self._debug_active and b_idx == 0:
                    ref_preview = refs[:4] if refs else refs
                    count_preview = counts[:4] if counts else counts
                    msg = (
                        f"[debug] region_source={region_source} refs={ref_preview} "
                        f"counts={count_preview} region_feats={tuple(region_feats.shape)}"
                    )
                    self._debug_log(msg)
                    try:
                        self._debug_log(
                            f"[debug] region_feats min/mean/max="
                            f"{float(region_feats.min().item()):.4f}/"
                            f"{float(region_feats.mean().item()):.4f}/"
                            f"{float(region_feats.max().item()):.4f}"
                        )
                    except Exception as exc:
                        self._debug_log(f"[debug] region stats failed: {exc}")
                proj_dtype = self.llm_adapter[0].weight.dtype
                if region_feats.dtype != proj_dtype:
                    region_feats = region_feats.to(proj_dtype)
                region_llm = self.llm_adapter(region_feats)
                region_embeds_per_sample.append(region_llm)

        # 6. VQA via LLM with Teacher Forcing (if questions provided)
        vqa_loss = None
        vqa_acc = None
        answer_logits = None

        if questions is not None:
            # Select visual tokens for LLM: [global, target]
            adapter_dtype = self.llm_adapter[0].weight.dtype
            if fused_object_features.dtype != adapter_dtype:
                fused_object_features = fused_object_features.to(adapter_dtype)

            use_target_indices = target_indices is not None
            if use_target_indices and not isinstance(target_indices, torch.Tensor):
                use_target_indices = all(t is not None for t in target_indices)

            if use_target_indices:
                target_idx_tensor = torch.tensor(
                    target_indices,
                    device=fused_object_features.device,
                    dtype=torch.long
                )
                if target_idx_tensor.ndim == 0:
                    target_idx_tensor = target_idx_tensor.unsqueeze(0)
                if target_idx_tensor.shape[0] != B:
                    raise ValueError(
                        f"target_indices length {target_idx_tensor.shape[0]} does not match batch size {B}"
                    )
                max_k = fused_object_features.shape[1]
                safe_target_idx = torch.where(
                    (target_idx_tensor >= 0) & (target_idx_tensor < max_k),
                    target_idx_tensor,
                    global_idx_tensor
                )
                gather_idx = torch.stack([global_idx_tensor, safe_target_idx], dim=1)  # [B, 2]
                gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, fused_object_features.size(-1))
                visual_tokens = fused_object_features.gather(dim=1, index=gather_idx)  # [B, 2, D]
            else:
                if fused_features.dtype != adapter_dtype:
                    fused_features = fused_features.to(adapter_dtype)
                visual_tokens = fused_features.unsqueeze(1)  # [B, 1, D]

            if global_token_override is not None and visual_tokens.shape[1] >= 1:
                global_token_override = global_token_override.to(visual_tokens.dtype)
                visual_tokens[:, 0, :] = global_token_override

            visual_embeds = self.llm_adapter(visual_tokens)  # [B, V, llm_hidden_size]

            # Questions are now List[str], one per sample (no more nesting)
            all_questions = questions  # [B]
            all_answers = answers if answers is not None else [None] * B

            # No need to replicate visual embeddings - one question per sample
            visual_embeds_expanded = visual_embeds  # [B, V, hidden]

            # Convert answers to strings
            if all_answers[0] is not None:
                answer_strings = [str(ans) if ans is not None else "" for ans in all_answers]
            else:
                answer_strings = None

            # Tokenize questions and answers together for Teacher Forcing
            embed_layer = self.qwen_model.get_input_embeddings()

            total_q = len(all_questions)

            if answer_strings is not None and self.training:
                # TEACHER FORCING: Input = [visual] + question + answer
                # Tokenize question + answer together
                full_texts = [q + " " + a for q, a in zip(all_questions, answer_strings)]
                full_inputs = self.tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True if cutoff_len else False,
                    max_length=cutoff_len,
                    return_tensors="pt"
                ).to(visual_embeds_expanded.device)

                # Also tokenize just questions to know where answer starts
                question_inputs = self.tokenizer(
                    all_questions,
                    padding=True,
                    truncation=True if cutoff_len else False,
                    max_length=cutoff_len,
                    return_tensors="pt"
                ).to(visual_embeds_expanded.device)

                # Get embeddings for full text
                text_embeds = embed_layer(full_inputs['input_ids'])  # [total_q, seq_len, hidden]
                text_embeds = self._inject_region_embeddings(
                    full_inputs['input_ids'],
                    text_embeds,
                    region_embeds_per_sample
                )

                # Ensure dtype match
                visual_embeds_expanded = visual_embeds_expanded.to(text_embeds.dtype)
                visual_token_count = visual_embeds_expanded.shape[1]

                # Concatenate: [visual tokens] + question + answer
                inputs_embeds = torch.cat([visual_embeds_expanded, text_embeds], dim=1)

                # Attention mask
                visual_attn = torch.ones(total_q, visual_token_count, device=visual_embeds_expanded.device)
                attention_mask = torch.cat([visual_attn, full_inputs['attention_mask']], dim=1)

                llm_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask
                )

                lm_logits = self.qwen_model.lm_head(llm_outputs.last_hidden_state)  # [total_q, 1+seq_len, vocab_size]

                # Construct labels aligned with visual tokens + text
                text_len = full_inputs['input_ids'].shape[1]
                labels = torch.full(
                    (total_q, visual_token_count + text_len),
                    -100,
                    device=full_inputs['input_ids'].device,
                    dtype=full_inputs['input_ids'].dtype
                )
                labels[:, visual_token_count:] = full_inputs['input_ids']

                # Mask out question tokens (set to -100)
                for i in range(total_q):
                    q_len = question_inputs['attention_mask'][i].sum().item()
                    labels[i, visual_token_count:visual_token_count + q_len] = -100

                # Shift for next-token prediction
                shift_logits = lm_logits[:, :-1, :].contiguous()  # [total_q, seq_len, vocab_size]
                shift_labels = labels[:, 1:].contiguous()  # [total_q, seq_len]

                # Flatten and compute loss
                vqa_loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100  # Ignore masked positions
                )
                vqa_loss = vqa_loss * self.vqa_weight

                # For monitoring (use last non-pad question token position)
                q_lens = question_inputs['attention_mask'].sum(dim=1)
                pos = visual_token_count + q_lens - 1
                pos = torch.clamp(pos, min=0, max=lm_logits.size(1) - 1).long()
                answer_logits = lm_logits[torch.arange(total_q, device=lm_logits.device), pos]
                # Compute VQA accuracy on first answer token
                with torch.no_grad():
                    total = 0
                    correct = 0
                    for i in range(total_q):
                        q_len = int(q_lens[i].item())
                        target_pos = visual_token_count + q_len - 1
                        if target_pos >= shift_logits.size(1):
                            continue
                        target_id = full_inputs['input_ids'][i, q_len]
                        pred_id = shift_logits[i, target_pos].argmax(dim=-1)
                        if target_id == pred_id:
                            correct += 1
                        total += 1
                    vqa_acc = (correct / total) if total > 0 else 0.0

            else:
                # Inference mode: only input [visual] + question
                question_inputs = self.tokenizer(
                    all_questions,
                    padding=True,
                    truncation=True if cutoff_len else False,
                    max_length=cutoff_len,
                    return_tensors="pt"
                ).to(visual_embeds_expanded.device)

                question_embeds = embed_layer(question_inputs['input_ids'])
                question_embeds = self._inject_region_embeddings(
                    question_inputs['input_ids'],
                    question_embeds,
                    region_embeds_per_sample
                )
                visual_embeds_expanded = visual_embeds_expanded.to(question_embeds.dtype)
                visual_token_count = visual_embeds_expanded.shape[1]

                inputs_embeds = torch.cat([visual_embeds_expanded, question_embeds], dim=1)

                visual_attn = torch.ones(total_q, visual_token_count, device=visual_embeds_expanded.device)
                attention_mask = torch.cat([visual_attn, question_inputs['attention_mask']], dim=1)

                llm_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask
                )

                lm_logits = self.qwen_model.lm_head(llm_outputs.last_hidden_state)
                q_lens = question_inputs['attention_mask'].sum(dim=1)
                pos = visual_token_count + q_lens - 1
                pos = torch.clamp(pos, min=0, max=lm_logits.size(1) - 1).long()
                answer_logits = lm_logits[torch.arange(total_q, device=lm_logits.device), pos]
                vqa_loss = torch.tensor(0.0, device=fused_features.device)
                vqa_acc = None

        # Total loss
        total_loss = infonce_loss + (vqa_loss if vqa_loss is not None else 0) + consistency_loss

        if self._debug_active:
            self._debug_calls += 1
            self._debug_active = False

        return {
            'loss': total_loss,
            'infonce_loss': infonce_loss,
            'vqa_loss': vqa_loss,
            'consistency_loss': consistency_loss,
            'answer_logits': answer_logits,
            'vqa_acc': vqa_acc,
            'infonce_acc': infonce_acc,
            'embeddings': {
                'ego': ego_emb,
                'exo': exo_emb,
                'fused': fused_features,
                'fused_objects': fused_object_features
            },
            'retrieval': {
                'indices': retrieval_indices,
                'scores': retrieval_scores
            } if retrieval_indices is not None else None,
            'valid_mask': valid_mask,
            'debug_n_valid': debug_n_valid
        }

    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'=' * 80}")
        print(f"Model Parameters:")
        print(f"  Total: {total_params / 1e6:.2f}M")
        print(f"  Trainable: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.1f}%)")
        print(f"  Frozen: {(total_params - trainable_params) / 1e6:.2f}M")
        print(f"{'=' * 80}\n")

        # Breakdown
        print("Trainable components:")
        for name, module in [
            ('ART', self.art),
            ('OCVA', self.ocva),
            ('LLM Adapter', self.llm_adapter)
        ]:
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name}: {params / 1e6:.2f}M")

        return total_params, trainable_params


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    print("Testing CrossViewerModelNoART...")

    model = CrossViewerModelNoART(
        vision_encoder_path="Qwen/Qwen3-VL-8B-Instruct",
        freeze_vision_encoder=True,
        freeze_llm=True
    )
    model.count_parameters()
    img1 = Image.new('RGB', (448, 448), color='red')
    img2 = Image.new('RGB', (448, 448), color='blue')
    images = [img1, img2]
    masks = [
        np.random.randint(0, 2, (3, 448, 448)).astype(np.float32),
        np.random.randint(0, 2, (3, 448, 448)).astype(np.float32),
    ]

    print("\n" + "=" * 80)
    print("Testing forward pass (with masks, no VQA)...")
    print("=" * 80)
    model.train()
    output = model(images, masks=masks)

    print(f"\n✓ Outputs:")
    print(f"  Total Loss: {output['loss'].item():.4f}")
    print(f"  InfoNCE Loss: {output['infonce_loss'].item():.4f}")
    print(f"  Ego embeddings: {output['embeddings']['ego'].shape}")
    print(f"  Exo embeddings: {output['embeddings']['exo'].shape}")
    print(f"  Fused features: {output['embeddings']['fused'].shape}")

    print("\n" + "=" * 80)
    print("Testing forward pass (with masks + VQA)...")
    print("=" * 80)
    question = "Which object appears in both views?"
    answer = torch.tensor([0])

    output = model(images, masks=masks, question=question, answer=answer)

    print(f"\n✓ Outputs:")
    print(f"  Total Loss: {output['loss'].item():.4f}")
    print(f"  InfoNCE Loss: {output['infonce_loss'].item():.4f}")
    print(f"  VQA Loss: {output['vqa_loss'].item():.4f}")
    print(f"  Answer logits: {output['answer_logits']}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
