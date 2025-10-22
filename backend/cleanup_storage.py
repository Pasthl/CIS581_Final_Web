"""
Manual cleanup script for storage folders
Run this if you want to clear all temporary files
"""

import os
import glob

def cleanup_storage():
    """Remove all files from uploads and outputs folders"""
    folders = ['storage/uploads', 'storage/outputs']

    total_deleted = 0
    for folder in folders:
        pattern = os.path.join(folder, '*')
        files = glob.glob(pattern)

        for file_path in files:
            if os.path.isfile(file_path) and not file_path.endswith('.gitkeep'):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    total_deleted += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"\nTotal files deleted: {total_deleted}")

if __name__ == '__main__':
    print("Cleaning up storage folders...")
    cleanup_storage()
    print("Done!")
