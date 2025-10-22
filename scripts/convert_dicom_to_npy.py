#!/usr/bin/env python3
"""
Batch Convert DICOM Files to NPY Format

Converts DICOM images to normalized NPY arrays for faster loading during training.
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'MriWizard'))

from MriWizard.io.dicom_loader import LoadDICOM


def convert():
    parser = argparse.ArgumentParser(description='Convert DICOM files to NPY format')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing DICOM files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for NPY files')
    parser.add_argument('--recursive', action='store_true', help='Search recursively for DICOM files')
    parser.add_argument('--normalize', action='store_true', default=True, help='Normalize images to [0, 1]')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DICOM to NPY Conversion")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Recursive search: {args.recursive}")
    print(f"Normalize: {args.normalize}")
    print("="*60)

    # Find DICOM files
    print("\nSearching for DICOM files...")
    if args.recursive:
        dicom_files = list(input_dir.rglob("*.dcm"))
    else:
        dicom_files = list(input_dir.glob("*.dcm"))

    print(f"Found {len(dicom_files)} DICOM files")

    if len(dicom_files) == 0:
        print("No DICOM files found. Exiting.")
        return

    # Load DICOM loader
    loader = LoadDICOM(convert_to_kspace=False)

    # Convert files
    print("\nConverting files...")
    successful = 0
    failed = 0
    failed_files = []

    for dcm_path in tqdm(dicom_files, desc="Converting"):
        try:
            # Load DICOM
            record = loader.load(str(dcm_path))
            image = record['image']

            # Create output path (preserve directory structure)
            rel_path = dcm_path.relative_to(input_dir)
            npy_path = output_dir / rel_path.with_suffix('.npy')
            npy_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as NPY
            np.save(npy_path, image)
            successful += 1

        except Exception as e:
            failed += 1
            failed_files.append((str(dcm_path), str(e)))
            tqdm.write(f"Error converting {dcm_path.name}: {e}")

    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Successful: {successful}/{len(dicom_files)}")
    print(f"Failed: {failed}/{len(dicom_files)}")

    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files[:10]:  # Show first 10
            print(f"  - {Path(file_path).name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print(f"\n✓ Conversion complete!")
    print(f"NPY files saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    convert()
