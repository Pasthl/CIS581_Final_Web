"""
Image Degradation Module

Provides functions to degrade high-quality images for evaluation purposes.
Used to simulate low-quality inputs for testing super-resolution models.
"""

import numpy as np
from PIL import Image, ImageFilter
import cv2


def downscale_image(image, scale_factor=4):
    """
    Downscale an image by a given factor

    Args:
        image: PIL Image or numpy array
        scale_factor: Factor to downscale by (e.g., 4 means 1/4 size)

    Returns:
        PIL Image at reduced resolution
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    original_size = image.size
    new_size = (original_size[0] // scale_factor, original_size[1] // scale_factor)

    # Use LANCZOS for high-quality downscaling
    downscaled = image.resize(new_size, Image.LANCZOS)

    return downscaled


def add_gaussian_noise(image, sigma=10):
    """
    Add Gaussian noise to an image

    Args:
        image: PIL Image
        sigma: Standard deviation of noise (0-50, default: 10)

    Returns:
        PIL Image with added noise
    """
    img_array = np.array(image).astype(np.float32)

    # Generate Gaussian noise
    noise = np.random.normal(0, sigma, img_array.shape)

    # Add noise and clip to valid range
    noisy = img_array + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)


def add_blur(image, blur_type='gaussian', kernel_size=5):
    """
    Add blur to an image

    Args:
        image: PIL Image
        blur_type: Type of blur ('gaussian', 'box', 'motion')
        kernel_size: Size of blur kernel

    Returns:
        PIL Image with added blur
    """
    if blur_type == 'gaussian':
        # Gaussian blur
        radius = kernel_size / 2
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif blur_type == 'box':
        # Box blur
        blurred = image.filter(ImageFilter.BoxBlur(radius=kernel_size//2))
    elif blur_type == 'motion':
        # Motion blur using OpenCV
        img_array = np.array(image)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Apply motion blur
        blurred_array = cv2.filter2D(img_array, -1, kernel)
        blurred = Image.fromarray(blurred_array.astype(np.uint8))
    else:
        blurred = image

    return blurred


def jpeg_compression(image, quality=50):
    """
    Apply JPEG compression artifacts

    Args:
        image: PIL Image
        quality: JPEG quality (0-100, lower means more artifacts)

    Returns:
        PIL Image with compression artifacts
    """
    from io import BytesIO

    # Save to bytes with JPEG compression
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    # Load back the compressed image
    compressed = Image.open(buffer).convert('RGB')

    return compressed


def degrade_for_evaluation(image, degradation_type='light', scale=4, **kwargs):
    """
    Apply degradation to a high-quality image for evaluation

    Args:
        image: PIL Image (high-quality ground truth)
        degradation_type: Type of degradation to apply
            - 'light': Light degradation with noise and blur, NO downscaling (default)
            - 'medium': Moderate degradation with noise, blur, and 2x downscaling
            - 'heavy': Heavy degradation with noise, blur, compression, and 4x downscaling
            - 'bicubic': Simple bicubic downscaling only
            - 'blur_downscale': Add blur then downscale
            - 'noise_downscale': Add noise then downscale
            - 'jpeg_downscale': JPEG compression then downscale
            - 'realistic': Combination of blur, noise, and compression
        scale: Downscaling factor (default: 4)
        **kwargs: Additional parameters for specific degradation types

    Returns:
        PIL Image (degraded low-quality image)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if degradation_type == 'light':
        # Light degradation: blur + noise, NO downscaling
        blur_kernel = kwargs.get('blur_kernel', 3)
        noise_sigma = kwargs.get('noise_sigma', 8)

        degraded = add_blur(image, blur_type='gaussian', kernel_size=blur_kernel)
        degraded = add_gaussian_noise(degraded, sigma=noise_sigma)

    elif degradation_type == 'medium':
        # Medium degradation: blur + noise + light compression + 2x downscale
        blur_kernel = kwargs.get('blur_kernel', 4)
        noise_sigma = kwargs.get('noise_sigma', 10)
        jpeg_quality = kwargs.get('jpeg_quality', 75)

        degraded = add_blur(image, blur_type='gaussian', kernel_size=blur_kernel)
        degraded = add_gaussian_noise(degraded, sigma=noise_sigma)
        degraded = jpeg_compression(degraded, quality=jpeg_quality)
        degraded = downscale_image(degraded, scale_factor=2)  # Only 2x downscale

    elif degradation_type == 'heavy':
        # Heavy degradation: blur + noise + compression + 4x downscale
        blur_kernel = kwargs.get('blur_kernel', 5)
        noise_sigma = kwargs.get('noise_sigma', 12)
        jpeg_quality = kwargs.get('jpeg_quality', 60)

        degraded = add_blur(image, blur_type='gaussian', kernel_size=blur_kernel)
        degraded = add_gaussian_noise(degraded, sigma=noise_sigma)
        degraded = jpeg_compression(degraded, quality=jpeg_quality)
        degraded = downscale_image(degraded, scale_factor=4)

    elif degradation_type == 'bicubic':
        # Simple bicubic downscaling
        degraded = downscale_image(image, scale_factor=scale)

    elif degradation_type == 'blur_downscale':
        # Add blur then downscale
        blur_kernel = kwargs.get('blur_kernel', 5)
        blurred = add_blur(image, blur_type='gaussian', kernel_size=blur_kernel)
        degraded = downscale_image(blurred, scale_factor=scale)

    elif degradation_type == 'noise_downscale':
        # Add noise then downscale
        noise_sigma = kwargs.get('noise_sigma', 10)
        noisy = add_gaussian_noise(image, sigma=noise_sigma)
        degraded = downscale_image(noisy, scale_factor=scale)

    elif degradation_type == 'jpeg_downscale':
        # JPEG compression then downscale
        jpeg_quality = kwargs.get('jpeg_quality', 50)
        compressed = jpeg_compression(image, quality=jpeg_quality)
        degraded = downscale_image(compressed, scale_factor=scale)

    elif degradation_type == 'realistic':
        # Realistic degradation: blur + noise + compression + downscale
        blur_kernel = kwargs.get('blur_kernel', 3)
        noise_sigma = kwargs.get('noise_sigma', 5)
        jpeg_quality = kwargs.get('jpeg_quality', 70)

        # Apply degradations in sequence
        degraded = add_blur(image, blur_type='gaussian', kernel_size=blur_kernel)
        degraded = add_gaussian_noise(degraded, sigma=noise_sigma)
        degraded = jpeg_compression(degraded, quality=jpeg_quality)
        degraded = downscale_image(degraded, scale_factor=scale)

    else:
        # Default to light degradation
        degraded = add_blur(image, blur_type='gaussian', kernel_size=3)
        degraded = add_gaussian_noise(degraded, sigma=8)

    return degraded


def create_lr_hr_pair(ground_truth_path, scale=4, degradation_type='bicubic', **kwargs):
    """
    Create a low-resolution / high-resolution image pair for evaluation

    Args:
        ground_truth_path: Path to high-quality ground truth image
        scale: Downscaling factor
        degradation_type: Type of degradation to apply
        **kwargs: Additional degradation parameters

    Returns:
        Tuple of (lr_image, hr_image) as PIL Images
    """
    # Load ground truth
    hr_image = Image.open(ground_truth_path).convert('RGB')

    # Create degraded low-resolution version
    lr_image = degrade_for_evaluation(
        hr_image,
        degradation_type=degradation_type,
        scale=scale,
        **kwargs
    )

    return lr_image, hr_image
