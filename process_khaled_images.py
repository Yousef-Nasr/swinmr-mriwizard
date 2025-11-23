#!/usr/bin/env python3
"""
Process all files in khaled folder, convert to DICOM, run SwinMR model, and save results.

This script:
1. Scans all files in the khaled folder
2. Converts image files (PNG, JPG, etc.) to DICOM format
3. Runs test_dicom.py on all DICOMs
4. Saves all results to khaled_results_ai folder
"""

import subprocess
import sys
from pathlib import Path
import shutil
import json
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid
import numpy as np
from PIL import Image
import datetime

def convert_image_to_dicom(image_path, output_path):
    """
    Convert an image file (PNG, JPG, etc.) to DICOM format.
    
    Args:
        image_path: Path to image file
        output_path: Path to save DICOM file
    """
    try:
        # Load image
        if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img = Image.open(image_path)
            
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            
            img_array = np.array(img, dtype=np.uint16)
        elif image_path.suffix.lower() == '.dcm':
            # Already DICOM, just copy
            shutil.copy(image_path, output_path)
            return True
        else:
            print(f"  ⚠ Unsupported format: {image_path.suffix}")
            return False
        
        # Create DICOM file
        file_meta = FileDataset(
            filename=str(output_path),
            dataset={},
            file_meta={},
            preamble=b"\0" * 128
        )
        
        # Set required DICOM attributes
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Create dataset
        ds = FileDataset(
            filename=str(output_path),
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )
        
        # Set patient info
        ds.PatientName = "Khaled^Patient"
        ds.PatientID = image_path.stem[:12]
        ds.PatientBirthDate = "19700101"
        ds.PatientSex = "O"
        
        # Set study/series info
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        
        # Set image info
        ds.Modality = "CT"
        ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        ds.SeriesDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.SeriesTime = datetime.datetime.now().strftime('%H%M%S')
        
        # Image size
        height, width = img_array.shape
        ds.Rows = height
        ds.Columns = width
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        # Set pixel data
        ds.PixelData = img_array.tobytes()
        
        # Save DICOM
        ds.save_as(str(output_path), write_like_original=False)
        return True
        
    except Exception as e:
        print(f"  ✗ Error converting {image_path.name}: {e}")
        return False


def process_khaled_folder():
    """
    Main processing function.
    """
    # Setup paths
    khaled_path = Path("khaled")
    dicom_dir = khaled_path / "DICOM"
    convert_dir = khaled_path / "converted_dcm"
    results_dir = Path("khaled_results_ai_2")
    
    # Create output directories
    convert_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing Khaled Images")
    print(f"{'='*70}\n")
    
    # Step 1: Copy DICOM files and add .dcm extension
    print("Step 1: Preparing DICOM files...")
    print(f"{'='*70}")
    
    converted_count = 0
    total_files = 0
    
    # Process DICOM files recursively (they don't have extensions)
    if dicom_dir.exists():
        # Recursively find all files in DICOM directory
        all_dicom_files = [f for f in dicom_dir.rglob("*") if f.is_file()]
        total_files += len(all_dicom_files)
        
        for dicom_file in all_dicom_files:
            try:
                # Copy DICOM files and add .dcm extension
                # Preserve directory structure in name
                relative_path = dicom_file.relative_to(dicom_dir)
                output_filename = str(relative_path).replace("\\", "_").replace("/", "_") + ".dcm"
                output_path = convert_dir / output_filename
                
                shutil.copy(dicom_file, output_path)
                converted_count += 1
                print(f"✓ Copied: {relative_path} → {output_path.name}")
            except Exception as e:
                print(f"✗ Error copying {dicom_file.name}: {e}")
    
    # Process any image files in khaled folder root
    for file_path in khaled_path.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            total_files += 1
            output_path = convert_dir / f"{file_path.stem}.dcm"
            print(f"Converting: {file_path.name}...")
            if convert_image_to_dicom(file_path, output_path):
                converted_count += 1
    
    print(f"\n✓ Prepared {converted_count}/{total_files} DICOM files\n")
    
    # Step 2: Run test_dicom.py on all converted DICOMs
    print("Step 2: Running SwinMR model on all DICOMs...")
    print(f"{'='*70}\n")
    
    # Find checkpoint file
    checkpoint_path = None
    for ckpt in Path(".").glob("**/*.pth"):
        if "best" in ckpt.name or "latest" in ckpt.name:
            checkpoint_path = str(ckpt)
            break
    
    if not checkpoint_path:
        for ckpt in Path(".").glob("**/*.pth"):
            checkpoint_path = str(ckpt)
            break
    
    if not checkpoint_path:
        print("✗ No checkpoint file (.pth) found!")
        return
    
    checkpoint_path = "checkpoints_epoch_26_new.pth"
    print(f"Using checkpoint: {checkpoint_path}\n")
    # Run test_dicom.py
    cmd = [
        sys.executable,
        "test_dicom.py",
        "--image-dir", str(convert_dir),
        "--checkpoint", checkpoint_path,
        "--output-dir", str(results_dir),
        "--recursive",
        "--device", "cuda"
    ]
    
    print(f"Running command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Processing complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running test_dicom.py: {e}")
        return
    
    # Step 3: Summary
    print(f"\n{'='*70}")
    print(f"Processing Complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {results_dir.absolute()}")
    print(f"Converted DICOMs: {convert_dir.absolute()}")
    
    # Count results
    result_files = list(results_dir.glob("**/*"))
    print(f"Total result files: {len(result_files)}")
    print(f"\n✓ All done!")


if __name__ == '__main__':
    try:
        process_khaled_folder()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

