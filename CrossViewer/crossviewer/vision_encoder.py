"""
Vision Encoder Wrapper for Qwen3-VL
"""
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np


class Qwen3VLVisionEncoder(nn.Module):
    """Qwen3-VL vision wrapper for image feature extraction."""

    def __init__(self, model_path="Qwen/Qwen3-VL-8B-Instruct", freeze=True):
        super().__init__()

        print(f"Loading Qwen3-VL from {model_path}...")

        self.full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None
        )

        self.visual = self.full_model.visual
        self.processor = AutoProcessor.from_pretrained(model_path)

        if freeze:
            self.visual.requires_grad_(False)
            self.visual.eval()
            print("✓ Vision encoder frozen")

        self.hidden_size = self.full_model.config.vision_config.out_hidden_size

        print(f"✓ Vision encoder loaded (hidden_size={self.hidden_size})")

    def preprocess_images(self, images):
        """
        Preprocess images using Qwen3-VL processor

        Args:
            images: List of PIL Images, numpy arrays, or torch tensors

        Returns:
            Dict with 'pixel_values' and 'image_grid_thw'
        """
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            elif isinstance(img, np.ndarray):
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            elif isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    img_np = (img_np * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        dummy_text = [""] * len(pil_images)
        processed = self.processor(text=dummy_text, images=pil_images, return_tensors="pt")
        return processed

    def forward(self, images, output_hidden_states=True):
        """
        Forward pass through vision encoder

        Args:
            images: List of PIL Images OR dict with 'pixel_values' and 'image_grid_thw'
            output_hidden_states: Whether to return all hidden states

        Returns:
            features: [total_patches, D] where D is hidden_size
            grid_thw: [num_images, 3] grid information for reshaping
        """
        if isinstance(images, dict) and 'pixel_values' in images:
            pixel_values = images['pixel_values']
            grid_thw = images['image_grid_thw']
        else:
            processed = self.preprocess_images(images)
            pixel_values = processed['pixel_values']
            grid_thw = processed['image_grid_thw']

        device = next(self.visual.parameters()).device
        pixel_values = pixel_values.to(device)
        grid_thw = grid_thw.to(device)

        with torch.set_grad_enabled(self.training and not self.visual.training):
            outputs = self.visual(
                pixel_values,
                grid_thw=grid_thw,
                output_hidden_states=output_hidden_states
            )

        features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

        return features, grid_thw

    def get_feature_maps(self, images):
        """
        Get 2D feature maps for mask pooling (batch processing)

        Args:
            images: List of PIL Images or dict with preprocessed data

        Returns:
            List of feature maps, each [H_i, W_i, D] (can have different sizes due to dynamic resolution)
        """
        features, grid_thw = self.forward(images)  # [total_patches, D], [B, 3]

        spatial_merge = self.full_model.config.vision_config.spatial_merge_size

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

    def to(self, device):
        """Override to method to handle bfloat16"""
        self.visual = self.visual.to(device)
        return self


if __name__ == "__main__":
    print("Testing Qwen3VLVisionEncoder...")
    encoder = Qwen3VLVisionEncoder()

    from PIL import Image
    img1 = Image.new('RGB', (448, 448), color='red')
    img2 = Image.new('RGB', (448, 448), color='blue')

    images = [img1, img2]

    print("\n✓ Testing preprocessing:")
    processed = encoder.preprocess_images(images)
    print(f"  Pixel values shape: {processed['pixel_values'].shape}")
    print(f"  Grid THW: {processed['image_grid_thw']}")

    features, grid_thw = encoder(images)
    print(f"\n✓ Forward pass:")
    print(f"  Features shape: {features.shape}")
    print(f"  Grid THW: {grid_thw}")

    total_patches_expected = sum([int(t*h*w) for t,h,w in grid_thw.tolist()])
    print(f"  Total patches from grid_thw: {total_patches_expected}")
    print(f"  Actual features: {features.shape[0]}")

    feature_maps = encoder.get_feature_maps(images)
    print(f"\n✓ Feature maps:")
    for i, fm in enumerate(feature_maps):
        print(f"  Image {i}: {fm.shape}")
