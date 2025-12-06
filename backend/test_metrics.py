"""
Test script for image quality metrics

Demonstrates how to use PSNR, SSIM, and other quality metrics
to evaluate super-resolution results.
"""

import os
from PIL import Image
from src.metrics import calculate_all_metrics, calculate_psnr, calculate_ssim


def test_basic_metrics():
    """Test basic metric calculations"""
    print("=" * 60)
    print("Testing Image Quality Metrics")
    print("=" * 60)

    # Create two test images
    # In practice, you would load real images
    img1 = Image.new('RGB', (100, 100), color='red')
    img2 = Image.new('RGB', (100, 100), color='red')  # Identical
    img3 = Image.new('RGB', (100, 100), color='blue')  # Different

    print("\n1. Testing identical images:")
    metrics = calculate_all_metrics(img1, img2)
    print(f"   PSNR: {metrics['psnr']:.2f} dB (should be inf)")
    print(f"   SSIM: {metrics['ssim']:.4f} (should be 1.0)")
    print(f"   MSE:  {metrics['mse']:.2f} (should be 0.0)")
    print(f"   MAE:  {metrics['mae']:.2f} (should be 0.0)")

    print("\n2. Testing different images:")
    metrics = calculate_all_metrics(img1, img3)
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   MSE:  {metrics['mse']:.2f}")
    print(f"   MAE:  {metrics['mae']:.2f}")


def test_model_with_metrics():
    """Test model inference with metrics calculation"""
    print("\n" + "=" * 60)
    print("Testing Model Inference with Metrics")
    print("=" * 60)

    # Check if test images exist
    test_image_path = 'test_images/sample.png'

    if not os.path.exists(test_image_path):
        print(f"\nTest image not found at: {test_image_path}")
        print("Creating a dummy test image...")
        os.makedirs('test_images', exist_ok=True)
        img = Image.new('RGB', (64, 64), color='red')
        img.save(test_image_path)

    print(f"\nLoading test image: {test_image_path}")

    try:
        from src import get_model

        # Load EDSR model
        model_path = os.path.join('models', 'edsr_baseline_x4-6b446fab.pt')
        if not os.path.exists(model_path):
            print(f"\nModel not found at: {model_path}")
            print("Please download the model first.")
            return

        print("Loading EDSR model...")
        model = get_model(model_path=model_path, scale=4, device='cpu')

        # Run inference with metrics
        print("\nRunning inference with quality metrics...")
        output_image, metrics = model.infer(
            test_image_path,
            calculate_metrics=True  # Enable metrics calculation
        )

        print("\nQuality Metrics (compared to input):")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  MSE:  {metrics['mse']:.2f}")
        print(f"  MAE:  {metrics['mae']:.2f}")

        print(f"\nInput size:  {Image.open(test_image_path).size}")
        print(f"Output size: {output_image.size}")

        # Save output
        output_path = 'test_images/output_with_metrics.png'
        output_image.save(output_path)
        print(f"\nSaved output to: {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def test_with_reference_image():
    """Test metrics calculation with a separate reference image"""
    print("\n" + "=" * 60)
    print("Testing with Custom Reference Image")
    print("=" * 60)

    # This is useful when you have a high-quality ground truth image
    # to compare against

    print("\nUsage example:")
    print("""
    from src import get_model

    model = get_model(model_path='models/edsr_baseline_x4-6b446fab.pt')

    # Compare output with a high-quality reference image
    output_img, metrics = model.infer(
        'low_quality.png',
        calculate_metrics=True,
        reference_image='high_quality_ground_truth.png'
    )

    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    """)


def interpret_metrics():
    """Print interpretation guide for metrics"""
    print("\n" + "=" * 60)
    print("Metrics Interpretation Guide")
    print("=" * 60)

    print("""
1. PSNR (Peak Signal-to-Noise Ratio)
   - Measured in dB (decibels)
   - Higher is better
   - Typical values:
     * > 40 dB: Excellent quality
     * 30-40 dB: Good quality
     * 20-30 dB: Acceptable quality
     * < 20 dB: Poor quality
   - Inf: Images are identical

2. SSIM (Structural Similarity Index)
   - Range: -1 to 1
   - Higher is better
   - Typical values:
     * > 0.95: Excellent similarity
     * 0.90-0.95: Good similarity
     * 0.80-0.90: Fair similarity
     * < 0.80: Poor similarity
   - 1.0: Images are identical

3. MSE (Mean Squared Error)
   - Lower is better
   - 0: Images are identical
   - Sensitive to brightness/contrast differences

4. MAE (Mean Absolute Error)
   - Lower is better
   - 0: Images are identical
   - Less sensitive to outliers than MSE
    """)


if __name__ == '__main__':
    print("\nImage Quality Assessment Metrics Demo")
    print("This script demonstrates the usage of PSNR, SSIM, and other metrics\n")

    # Test 1: Basic metrics
    test_basic_metrics()

    # Test 2: Model with metrics
    test_model_with_metrics()

    # Test 3: Reference image example
    test_with_reference_image()

    # Interpretation guide
    interpret_metrics()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
