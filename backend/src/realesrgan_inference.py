import os
import numpy as np
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


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
    def infer_from_pil(self, pil_image, face_enhance=False):
        """Run inference on a PIL Image"""
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
        return Image.fromarray(output_rgb)


_realesrgan_instance = None

def get_realesrgan_model(model_path='models/RealESRGAN_x4plus.pth', scale=4, device='cpu'):
    """Get or create Real-ESRGAN model instance (singleton pattern)"""
    global _realesrgan_instance
    if _realesrgan_instance is None:
        _realesrgan_instance = RealESRGANInference(model_path, scale, device)
    return _realesrgan_instance
