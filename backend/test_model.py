"""
Test script to verify EDSR model loading and inference
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def test_model_loading():
    """Test if model loads correctly"""
    print("=" * 60)
    print("Testing EDSR Model Loading")
    print("=" * 60)

    try:
        from src import get_model

        model_path = 'models/edsr_baseline_x4-6b446fab.pt'

        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

        print("\nLoading EDSR model...")
        model = get_model(model_path=model_path, scale=4, device='cpu')

        print("✓ Model loaded successfully!")
        print(f"  Device: {next(model.model.parameters()).device}")
        print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")

        return True

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference with a dummy image"""
    print("\n" + "=" * 60)
    print("Testing EDSR Inference")
    print("=" * 60)

    try:
        from src import get_model

        # Create a small test image
        print("\nCreating test image (64x64)...")
        test_img = Image.new('RGB', (64, 64), color='red')
        test_path = 'storage/uploads/test_input.png'
        output_path = 'storage/outputs/test_output.png'

        os.makedirs('storage/uploads', exist_ok=True)
        os.makedirs('storage/outputs', exist_ok=True)

        test_img.save(test_path)
        print(f"✓ Test image saved: {test_path}")

        # Run inference
        print("\nRunning inference...")
        model = get_model(model_path='models/edsr_baseline_x4-6b446fab.pt')

        import time
        start = time.time()
        result = model.infer(test_path, output_path)
        elapsed = time.time() - start

        print(f"✓ Inference completed in {elapsed:.2f}s")
        print(f"  Input size: {test_img.size}")
        print(f"  Output size: {result.size}")
        print(f"  Scale factor: {result.size[0] // test_img.size[0]}x")
        print(f"  Output saved: {output_path}")

        # Cleanup
        os.remove(test_path)
        os.remove(output_path)
        print("\n✓ Test files cleaned up")

        return True

    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test if all dependencies are installed"""
    print("\n" + "=" * 60)
    print("Testing Dependencies")
    print("=" * 60)

    dependencies = {
        'torch': None,
        'torchvision': None,
        'PIL': 'Pillow',
        'numpy': None,
        'flask': 'Flask',
        'flask_cors': 'flask-cors'
    }

    all_ok = True
    for module, package in dependencies.items():
        try:
            if module == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')

            pkg_name = package if package else module
            print(f"✓ {pkg_name:20s} version {version}")

        except ImportError:
            pkg_name = package if package else module
            print(f"❌ {pkg_name:20s} NOT INSTALLED")
            all_ok = False

    return all_ok


if __name__ == '__main__':
    print("\n🧪 EDSR Backend Test Suite\n")

    # Test dependencies
    deps_ok = test_dependencies()

    if not deps_ok:
        print("\n⚠️  Some dependencies are missing. Install them with:")
        print("    pip install -r requirements.txt")
        sys.exit(1)

    # Test model loading
    model_ok = test_model_loading()

    if not model_ok:
        print("\n❌ Model loading failed!")
        sys.exit(1)

    # Test inference
    inference_ok = test_inference()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Dependencies: {'✓ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Model Loading: {'✓ PASS' if model_ok else '❌ FAIL'}")
    print(f"Inference: {'✓ PASS' if inference_ok else '❌ FAIL'}")

    if deps_ok and model_ok and inference_ok:
        print("\n🎉 All tests passed! Backend is ready to use.")
        print("\nTo start the server, run:")
        print("    python app.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
