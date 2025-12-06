import os
import numpy as np
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from .metrics import calculate_all_metrics


class RealESRGANInference:
    """Real-ESRGAN inference wrapper for super-resolution"""

    def __init__(self, model_path, scale=4, device='cpu'):
        self.device = torch.device(device)
        self.scale = scale
        self.face_enhancer = None

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=(device == 'cuda'),
            device=self.device
        )

        print(f"Real-ESRGAN model loaded successfully on {device}")

    def _init_face_enhancer(self):
        """Initialize GFPGAN face enhancer (lazy loading)"""
        if self.face_enhancer is None:
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=self.scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler
            )
            print("GFPGAN face enhancer loaded successfully")

    @torch.no_grad()
    def infer_from_pil(self, pil_image, face_enhance=False, calculate_metrics=False, reference_image=None):
        """
        Run inference on a PIL Image

        Args:
            pil_image: PIL Image object
            face_enhance: Whether to use GFPGAN face enhancement (default: False)
            calculate_metrics: Whether to calculate quality metrics (default: False)
            reference_image: Reference image for metrics (PIL Image or path).
                           If None and calculate_metrics=True, uses input image

        Returns:
            If calculate_metrics is False: PIL Image of the processed result
            If calculate_metrics is True: tuple of (PIL Image, metrics dict)
        """
        img_rgb = np.array(pil_image.convert('RGB'))
        img_bgr = img_rgb[:, :, ::-1].copy()

        if face_enhance:
            self._init_face_enhancer()
            _, _, output_bgr = self.face_enhancer.enhance(
                img_bgr, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output_bgr, _ = self.upsampler.enhance(img_bgr, outscale=self.scale)

        output_rgb = output_bgr[:, :, ::-1]
        output_image = Image.fromarray(output_rgb)

        # Calculate metrics if requested
        if calculate_metrics:
            # Determine reference image
            if reference_image is None:
                # Use input image as reference
                ref_img = pil_image.convert('RGB')
            elif isinstance(reference_image, str):
                # Load from path
                ref_img = Image.open(reference_image).convert('RGB')
            else:
                # Assume it's already a PIL Image
                ref_img = reference_image.convert('RGB')

            # Resize output to match reference for fair comparison
            if output_image.size != ref_img.size:
                output_resized = output_image.resize(ref_img.size, Image.LANCZOS)
            else:
                output_resized = output_image

            # Calculate metrics
            metrics = calculate_all_metrics(ref_img, output_resized)
            return output_image, metrics

        return output_image


_realesrgan_instance = None

def get_realesrgan_model(model_path='models/RealESRGAN_x4plus.pth', scale=4, device='cpu'):
    """Get or create Real-ESRGAN model instance (singleton pattern)"""
    global _realesrgan_instance
    if _realesrgan_instance is None:
        _realesrgan_instance = RealESRGANInference(model_path, scale, device)
    return _realesrgan_instance
