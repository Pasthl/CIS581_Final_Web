"""
Quick test script to verify the backend is working
Run this before starting the full API server
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("Pixel Revival Backend - Quick Test")
print("=" * 70)

# Check Python version
print(f"\n[OK] Python version: {sys.version.split()[0]}")

# Check if we're in the right directory
if not os.path.exists('app.py'):
    print("\n[ERROR] Please run this script from the backend/ directory")
    print("   cd backend && python quick_test.py")
    sys.exit(1)

print("[OK] Running from backend directory")

# Check if model file exists
model_path = 'models/edsr_baseline_x4-6b446fab.pt'
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"[OK] Model file found: {size_mb:.2f} MB")
else:
    print(f"\n[ERROR] Model file not found: {model_path}")
    print("   Please copy the EDSR weights to backend/models/")
    sys.exit(1)

# Try importing key dependencies
print("\nChecking dependencies...")
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError:
    print("[ERROR] PyTorch not installed")
    print("   pip install torch")
    sys.exit(1)

try:
    import flask
    print(f"[OK] Flask {flask.__version__}")
except ImportError:
    print("[ERROR] Flask not installed")
    print("   pip install flask")
    sys.exit(1)

try:
    from PIL import Image
    print(f"[OK] Pillow installed")
except ImportError:
    print("[ERROR] Pillow not installed")
    print("   pip install Pillow")
    sys.exit(1)

# Try importing the model
print("\nLoading EDSR model...")
try:
    from src import get_model
    model = get_model(model_path=model_path, scale=4, device='cpu')
    print("[OK] EDSR model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {str(e)}")
    sys.exit(1)

# Create a tiny test
print("\nRunning quick inference test...")
try:
    from PIL import Image
    import numpy as np

    # Create 32x32 test image
    test_img = Image.new('RGB', (32, 32), color=(100, 150, 200))

    # Convert to numpy
    img_np = np.array(test_img).astype(np.float32)

    # Run inference
    from src.data import np2Tensor
    import torch

    img_tensor = np2Tensor(img_np, rgb_range=255)[0].unsqueeze(0)

    with torch.no_grad():
        output = model.model(img_tensor)

    print(f"[OK] Test inference successful!")
    print(f"  Input: {img_tensor.shape} -> Output: {output.shape}")
    print(f"  Scale: {output.shape[-1] // img_tensor.shape[-1]}x")

except Exception as e:
    print(f"[ERROR] Inference test failed: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 70)
print("SUCCESS! All checks passed! Backend is ready.")
print("=" * 70)
print("\nTo start the server:")
print("  python app.py")
print("\nThe API will be available at:")
print("  http://localhost:5000")
print("=" * 70)
