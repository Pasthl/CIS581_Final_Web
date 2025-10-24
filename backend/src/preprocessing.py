"""
Image Preprocessing Module
"""

import cv2
import numpy as np
from PIL import Image
import os


class ImagePreprocessor:
    """Simple image preprocessor"""

    def __init__(self):
        self.history = []

    def load_image(self, image_path):
        """Load image from path or PIL Image"""
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, Image.Image):
            img = np.array(image_path.convert('RGB'))
        else:
            raise TypeError("Input must be file path or PIL Image")
        return img.astype(np.uint8)

    def remove_jpeg_artifacts(self, img, strength='medium'):
        """Remove JPEG compression artifacts"""
        params = {
            'light': (3, 30, 30),
            'medium': (5, 50, 50),
            'strong': (7, 75, 75)
        }
        d, sigmaColor, sigmaSpace = params.get(strength, params['medium'])
        img_clean = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        self.history.append(f"Remove JPEG artifacts ({strength})")
        return img_clean

    def enhance_contrast(self, img, method='clahe', clip_limit=2.5):
        """Enhance contrast using CLAHE or histogram equalization"""
        if method == 'clahe':
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            self.history.append(f"CLAHE (clip={clip_limit})")
        elif method == 'histogram':
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            self.history.append("Histogram equalization")
        else:
            raise ValueError(f"Unknown method: {method}")
        return img_enhanced

    def denoise(self, img, strength='medium'):
        """Non-local means denoising"""
        params = {
            'light': (5, 5, 7, 21),
            'medium': (10, 10, 7, 21),
            'strong': (15, 15, 7, 21)
        }
        h, hColor, templateWindowSize, searchWindowSize = params.get(strength, params['medium'])
        img_denoised = cv2.fastNlMeansDenoisingColored(
            img, None, h=h, hColor=hColor,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize
        )
        self.history.append(f"Denoise ({strength})")
        return img_denoised

    def adjust_gamma(self, img, gamma=1.0):
        """Gamma correction (< 1.0: brighten, > 1.0: darken)"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        img_gamma = cv2.LUT(img, table)
        self.history.append(f"Gamma correction ({gamma:.2f})")
        return img_gamma

    def to_pil(self, img):
        """Convert to PIL Image"""
        return Image.fromarray(img.astype(np.uint8))

    def get_history(self):
        """Get processing history"""
        return self.history

    def reset_history(self):
        """Reset history"""
        self.history = []


def preprocess_pipeline_basic(image_path, output_path=None):
    """Basic preprocessing: light artifact removal + contrast enhancement"""
    preprocessor = ImagePreprocessor()
    img = preprocessor.load_image(image_path)
    img = preprocessor.remove_jpeg_artifacts(img, strength='light')
    img = preprocessor.enhance_contrast(img, method='clahe', clip_limit=2.0)
    result = preprocessor.to_pil(img)

    if output_path:
        result.save(output_path)

    print("\nProcessing steps:")
    for step in preprocessor.get_history():
        print(f"  - {step}")

    return result


def preprocess_pipeline_aggressive(image_path, output_path=None):
    """Aggressive preprocessing for low-quality images"""
    preprocessor = ImagePreprocessor()
    img = preprocessor.load_image(image_path)
    img = preprocessor.adjust_gamma(img, gamma=0.9)
    img = preprocessor.remove_jpeg_artifacts(img, strength='medium')
    img = preprocessor.enhance_contrast(img, method='clahe', clip_limit=3.0)
    img = preprocessor.denoise(img, strength='medium')
    result = preprocessor.to_pil(img)

    if output_path:
        result.save(output_path)

    print("\nProcessing steps:")
    for step in preprocessor.get_history():
        print(f"  - {step}")

    return result


def preprocess_pipeline_custom(
    image_path,
    output_path=None,
    remove_artifacts=True,
    artifact_strength='medium',
    enhance_contrast=True,
    contrast_method='clahe',
    contrast_clip=2.5,
    denoise=False,
    denoise_strength='medium',
    gamma=None
):
    """Custom preprocessing pipeline"""
    preprocessor = ImagePreprocessor()
    img = preprocessor.load_image(image_path)

    if gamma is not None:
        img = preprocessor.adjust_gamma(img, gamma=gamma)

    if remove_artifacts:
        img = preprocessor.remove_jpeg_artifacts(img, strength=artifact_strength)

    if enhance_contrast:
        img = preprocessor.enhance_contrast(img, method=contrast_method, clip_limit=contrast_clip)

    if denoise:
        img = preprocessor.denoise(img, strength=denoise_strength)

    result = preprocessor.to_pil(img)

    if output_path:
        result.save(output_path)

    print("\nProcessing steps:")
    for step in preprocessor.get_history():
        print(f"  - {step}")

    return result
