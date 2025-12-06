"""
Image Quality Assessment Metrics

Provides metrics for evaluating image super-resolution results:
- PSNR (Peak Signal-to-Noise Ratio) - Full-reference
- SSIM (Structural Similarity Index) - Full-reference
- NIQE (Natural Image Quality Evaluator) - No-reference
- LPIPS (Learned Perceptual Image Patch Similarity) - Full-reference
"""

import numpy as np
from PIL import Image
import torch

# Lazy-load pyiqa models to avoid startup overhead
_niqe_model = None
_lpips_model = None


def calculate_psnr(img1, img2, max_value=255.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images

    Args:
        img1: First image (numpy array, torch tensor, or PIL Image)
        img2: Second image (numpy array, torch tensor, or PIL Image)
        max_value: Maximum possible pixel value (default: 255 for 8-bit images)

    Returns:
        PSNR value in dB (higher is better)
    """
    # Convert to numpy arrays
    arr1 = _to_numpy(img1)
    arr2 = _to_numpy(img2)

    # Ensure same shape
    if arr1.shape != arr2.shape:
        raise ValueError(f"Images must have the same shape. Got {arr1.shape} and {arr2.shape}")

    # Calculate MSE
    mse = np.mean((arr1.astype(np.float64) - arr2.astype(np.float64)) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1, img2, max_value=255.0, window_size=11):
    """
    Calculate Structural Similarity Index (SSIM) between two images

    Args:
        img1: First image (numpy array, torch tensor, or PIL Image)
        img2: Second image (numpy array, torch tensor, or PIL Image)
        max_value: Maximum possible pixel value (default: 255)
        window_size: Size of the Gaussian window (default: 11)

    Returns:
        SSIM value between -1 and 1 (higher is better, 1 means identical)
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        # Convert to numpy arrays
        arr1 = _to_numpy(img1)
        arr2 = _to_numpy(img2)

        # Ensure same shape
        if arr1.shape != arr2.shape:
            raise ValueError(f"Images must have the same shape. Got {arr1.shape} and {arr2.shape}")

        # For color images, calculate SSIM per channel and average
        if len(arr1.shape) == 3:
            channel_ssim = []
            for i in range(arr1.shape[2]):
                ssim_val = ssim(
                    arr1[:, :, i],
                    arr2[:, :, i],
                    data_range=max_value,
                    win_size=window_size
                )
                channel_ssim.append(ssim_val)
            return float(np.mean(channel_ssim))
        else:
            # Grayscale image
            return float(ssim(arr1, arr2, data_range=max_value, win_size=window_size))

    except ImportError:
        # Fallback to simplified SSIM if scikit-image is not available
        return _calculate_ssim_simple(img1, img2, max_value)


def _calculate_ssim_simple(img1, img2, max_value=255.0):
    """
    Simplified SSIM calculation without external dependencies
    This is a basic approximation of SSIM
    """
    # Convert to numpy arrays
    arr1 = _to_numpy(img1).astype(np.float64)
    arr2 = _to_numpy(img2).astype(np.float64)

    # Constants
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    # Calculate means
    mu1 = arr1.mean()
    mu2 = arr2.mean()

    # Calculate variances and covariance
    sigma1_sq = ((arr1 - mu1) ** 2).mean()
    sigma2_sq = ((arr2 - mu2) ** 2).mean()
    sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()

    # Calculate SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim)


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE) between two images

    Args:
        img1: First image (numpy array, torch tensor, or PIL Image)
        img2: Second image (numpy array, torch tensor, or PIL Image)

    Returns:
        MSE value (lower is better)
    """
    # Convert to numpy arrays
    arr1 = _to_numpy(img1)
    arr2 = _to_numpy(img2)

    # Ensure same shape
    if arr1.shape != arr2.shape:
        raise ValueError(f"Images must have the same shape. Got {arr1.shape} and {arr2.shape}")

    # Calculate MSE
    mse = np.mean((arr1.astype(np.float64) - arr2.astype(np.float64)) ** 2)
    return float(mse)


def calculate_mae(img1, img2):
    """
    Calculate Mean Absolute Error (MAE) between two images

    Args:
        img1: First image (numpy array, torch tensor, or PIL Image)
        img2: Second image (numpy array, torch tensor, or PIL Image)

    Returns:
        MAE value (lower is better)
    """
    # Convert to numpy arrays
    arr1 = _to_numpy(img1)
    arr2 = _to_numpy(img2)

    # Ensure same shape
    if arr1.shape != arr2.shape:
        raise ValueError(f"Images must have the same shape. Got {arr1.shape} and {arr2.shape}")

    # Calculate MAE
    mae = np.mean(np.abs(arr1.astype(np.float64) - arr2.astype(np.float64)))
    return float(mae)


def calculate_niqe(img):
    """
    Calculate NIQE (Natural Image Quality Evaluator) score

    NIQE is a no-reference image quality metric that evaluates the "naturalness"
    of an image without requiring a reference image.

    Args:
        img: Image to evaluate (numpy array, torch tensor, or PIL Image)

    Returns:
        NIQE score (lower is better, typical range: 0-100)
    """
    global _niqe_model

    try:
        import pyiqa

        # Lazy-load NIQE model
        if _niqe_model is None:
            _niqe_model = pyiqa.create_metric('niqe', device='cpu')
            print("[Metrics] NIQE model loaded")

        # Convert to PIL Image if needed
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img.astype(np.uint8))
        elif isinstance(img, torch.Tensor):
            img_np = _to_numpy(img)
            img_pil = Image.fromarray(img_np.astype(np.uint8))
        elif isinstance(img, Image.Image):
            img_pil = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Convert PIL to tensor for pyiqa (expects tensor in [0, 1] range)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        # Calculate NIQE score
        score = _niqe_model(img_tensor)
        return float(score.item())

    except ImportError:
        print("[Warning] pyiqa not installed. Install with: pip install pyiqa")
        return None
    except Exception as e:
        print(f"[Warning] NIQE calculation failed: {e}")
        return None


def calculate_lpips(img1, img2):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity)

    LPIPS is a perceptual similarity metric using deep features.
    It better correlates with human perception than pixel-based metrics.

    Args:
        img1: Reference image (numpy array, torch tensor, or PIL Image)
        img2: Distorted image (numpy array, torch tensor, or PIL Image)

    Returns:
        LPIPS score (lower is better, range: 0-1)
    """
    global _lpips_model

    try:
        import pyiqa

        # Lazy-load LPIPS model
        if _lpips_model is None:
            _lpips_model = pyiqa.create_metric('lpips', device='cpu')
            print("[Metrics] LPIPS model loaded")

        # Convert both images to tensors
        def to_tensor(img):
            if isinstance(img, np.ndarray):
                img_pil = Image.fromarray(img.astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                img_np = _to_numpy(img)
                img_pil = Image.fromarray(img_np.astype(np.uint8))
            elif isinstance(img, Image.Image):
                img_pil = img
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # Convert to tensor in [0, 1] range
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        img1_tensor = to_tensor(img1)
        img2_tensor = to_tensor(img2)

        # Ensure same shape
        if img1_tensor.shape != img2_tensor.shape:
            raise ValueError(f"Images must have same shape. Got {img1_tensor.shape} and {img2_tensor.shape}")

        # Calculate LPIPS score
        score = _lpips_model(img1_tensor, img2_tensor)
        return float(score.item())

    except ImportError:
        print("[Warning] pyiqa not installed. Install with: pip install pyiqa")
        return None
    except Exception as e:
        print(f"[Warning] LPIPS calculation failed: {e}")
        return None


def calculate_all_metrics(img1, img2, max_value=255.0):
    """
    Calculate all quality metrics between two images

    Args:
        img1: Reference/ground truth image (numpy array, torch tensor, or PIL Image)
        img2: Distorted/processed image (numpy array, torch tensor, or PIL Image)
        max_value: Maximum possible pixel value (default: 255)

    Returns:
        Dictionary with all metrics:
        - psnr: Peak Signal-to-Noise Ratio (higher is better)
        - ssim: Structural Similarity Index (higher is better, 0-1)
        - niqe: Natural Image Quality Evaluator (lower is better)
        - lpips: Learned Perceptual Similarity (lower is better, 0-1)
    """
    metrics = {
        'psnr': calculate_psnr(img1, img2, max_value),
        'ssim': calculate_ssim(img1, img2, max_value),
    }

    # Calculate NIQE on the processed image (no-reference metric)
    niqe_score = calculate_niqe(img2)
    if niqe_score is not None:
        metrics['niqe'] = niqe_score

    # Calculate LPIPS (perceptual similarity)
    lpips_score = calculate_lpips(img1, img2)
    if lpips_score is not None:
        metrics['lpips'] = lpips_score

    return metrics


def _to_numpy(img):
    """
    Convert various image formats to numpy array

    Args:
        img: PIL Image, numpy array, or torch tensor

    Returns:
        Numpy array
    """
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, Image.Image):
        return np.array(img)
    elif isinstance(img, torch.Tensor):
        # Handle torch tensors (C, H, W) or (B, C, H, W)
        if img.dim() == 4:
            img = img.squeeze(0)  # Remove batch dimension
        if img.dim() == 3:
            # Convert (C, H, W) to (H, W, C)
            img = img.permute(1, 2, 0)
        return img.cpu().numpy()
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def compare_images(original_path, processed_path, reference_path=None):
    """
    Compare images and calculate quality metrics

    Args:
        original_path: Path to original/input image
        processed_path: Path to processed/output image
        reference_path: Optional path to high-quality reference image
                       If not provided, uses original_path as reference

    Returns:
        Dictionary with comparison results
    """
    # Load images
    original_img = Image.open(original_path).convert('RGB')
    processed_img = Image.open(processed_path).convert('RGB')

    if reference_path:
        reference_img = Image.open(reference_path).convert('RGB')
    else:
        reference_img = original_img

    # Resize processed image to match reference if needed
    if processed_img.size != reference_img.size:
        print(f"Warning: Resizing processed image from {processed_img.size} to {reference_img.size}")
        processed_img = processed_img.resize(reference_img.size, Image.LANCZOS)

    # Calculate metrics
    metrics = calculate_all_metrics(reference_img, processed_img)

    return {
        'original_size': original_img.size,
        'processed_size': processed_img.size,
        'reference_size': reference_img.size,
        'metrics': metrics
    }
