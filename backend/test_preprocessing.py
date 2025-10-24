"""
Preprocessing Test Script
Tests preprocessing effects on images in testimage folder
"""

import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from preprocessing import preprocess_pipeline_custom
from PIL import Image
import matplotlib.pyplot as plt


def process_testimage_folder():
    """
    Process all images in testimage folder and show comparison
    """
    testimage_dir = Path(current_dir) / 'testimage'

    if not testimage_dir.exists():
        print(f"Error: testimage folder not found")
        return

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(testimage_dir.glob(ext))

    # Remove duplicates
    image_files = list(set(image_files))

    if not image_files:
        print(f"\nNo images found in testimage folder")
        print("Please add test images (.jpg, .png, etc.)")
        return

    print("\n" + "=" * 60)
    print(f"Found {len(image_files)} image(s)")
    print("=" * 60)

    # Process each image
    all_results = []

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")

        try:
            original = Image.open(img_path)

            # Lightweight preprocessing: subtle contrast enhancement
            result = preprocess_pipeline_custom(
                str(img_path),
                None,
                remove_artifacts=False,
                enhance_contrast=True,
                contrast_method='clahe',
                contrast_clip=1.5,  # Reduced from 2.0 for more natural effect
                denoise=False,
                gamma=None
            )

            all_results.append({
                'name': img_path.name,
                'original': original,
                'processed': result
            })

            print(f"   Done")

        except Exception as e:
            print(f"   Error: {e}")
            continue

    if not all_results:
        print("\nNo images processed")
        return

    print("\n" + "=" * 60)
    print("Showing comparison...")
    print("=" * 60)

    show_comparison(all_results)


def show_comparison(results):
    """Show before/after comparison"""
    num_images = len(results)

    # 2 columns: Original, Preprocessed
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    fig.suptitle('Preprocessing Comparison', fontsize=16, fontweight='bold')

    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for row_idx, result in enumerate(results):
        # Original
        axes[row_idx, 0].imshow(result['original'])
        if row_idx == 0:
            axes[row_idx, 0].set_title('Original', fontsize=14, fontweight='bold')
        axes[row_idx, 0].set_ylabel(result['name'], fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Preprocessed
        axes[row_idx, 1].imshow(result['processed'])
        if row_idx == 0:
            axes[row_idx, 1].set_title('Preprocessed (Contrast+1.5)', fontsize=14, fontweight='bold')
        axes[row_idx, 1].axis('off')

    plt.tight_layout()
    print("\nClose window to exit")
    plt.show()
    print("Done!")


if __name__ == "__main__":
    try:
        process_testimage_folder()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
