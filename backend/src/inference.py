import os
import torch
import numpy as np
from PIL import Image

from .model import EDSR
from .data import np2Tensor, set_channel


class EDSRInference:
    """
    Simplified EDSR inference wrapper for image denoising/super-resolution
    Optimized for CPU deployment
    """

    def __init__(self, model_path, scale=4, device='cpu'):
        """
        Initialize EDSR model for inference

        Args:
            model_path: Path to the pretrained .pt file
            scale: Upscaling factor (2, 3, or 4)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.scale = scale

        # Create model (EDSR-baseline parameters)
        self.model = EDSR(
            n_resblocks=16,
            n_feats=64,
            scale=scale,
            n_colors=3,
            rgb_range=255,
            res_scale=1.0
        )

        # Load pretrained weights
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)

        # Optimize for CPU inference
        if device == 'cpu':
            torch.set_num_threads(4)

        print(f"EDSR model loaded successfully on {device}")

    def preprocess(self, image_path):
        """
        Load and preprocess image for EDSR

        Args:
            image_path: Path to input image

        Returns:
            Tensor ready for model input
        """
        # Load image and close file handle immediately (important for Windows)
        with Image.open(image_path) as img:
            img_np = np.array(img.convert('RGB')).astype(np.float32)

        # Convert to tensor (C, H, W)
        img_tensor = np2Tensor(img_np, rgb_range=255)[0]

        # Add batch dimension (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)

    def postprocess(self, output_tensor):
        """
        Convert model output tensor to PIL Image

        Args:
            output_tensor: Model output (1, C, H, W)

        Returns:
            PIL Image
        """
        # Remove batch dimension and move to CPU
        output = output_tensor.squeeze(0).cpu()

        # Clamp to valid range and convert to numpy
        output = output.clamp(0, 255).round()
        output_np = output.byte().permute(1, 2, 0).numpy()

        # Convert to PIL Image
        return Image.fromarray(output_np, mode='RGB')

    @torch.no_grad()
    def infer(self, image_path, output_path=None):
        """
        Run inference on an image

        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)

        Returns:
            PIL Image of the processed result
        """
        # Preprocess
        input_tensor = self.preprocess(image_path)

        # Run model
        output_tensor = self.model(input_tensor)

        # Postprocess
        output_image = self.postprocess(output_tensor)

        # Save if output path provided
        if output_path:
            output_image.save(output_path)
            print(f"Saved result to {output_path}")

        return output_image

    @torch.no_grad()
    def infer_from_pil(self, pil_image, output_path=None):
        """
        Run inference on a PIL Image directly

        Args:
            pil_image: PIL Image object
            output_path: Path to save output (optional)

        Returns:
            PIL Image of the processed result
        """
        # Convert PIL to numpy
        img_np = np.array(pil_image.convert('RGB')).astype(np.float32)

        # Convert to tensor
        img_tensor = np2Tensor(img_np, rgb_range=255)[0]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Run model
        output_tensor = self.model(img_tensor)

        # Postprocess
        output_image = self.postprocess(output_tensor)

        # Save if output path provided
        if output_path:
            output_image.save(output_path)

        return output_image


# Singleton instance for the API
_model_instance = None

def get_model(model_path='models/edsr_baseline_x4-6b446fab.pt', scale=4, device='cpu'):
    """
    Get or create model instance (singleton pattern)
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = EDSRInference(model_path, scale, device)
    return _model_instance
