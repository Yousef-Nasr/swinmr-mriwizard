#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly
"""

import sys
from pathlib import Path

print("Testing imports...")
print("="*60)

try:
    print("\n1. Testing MriWizard imports...")
    sys.path.insert(0, str(Path(__file__).parent / 'MriWizard'))
    from MriWizard.core.pipeline import Pipeline
    from MriWizard.io.dicom_loader import LoadDICOM
    from MriWizard.degrade.noise import AddGaussianNoiseKspace
    print("   ✓ MriWizard imports successful")
except Exception as e:
    print(f"   ✗ MriWizard imports failed: {e}")
    sys.exit(1)

try:
    print("\n2. Testing data module imports...")
    from data.pipeline_builder import build_degradation_pipeline
    from data.dataloader import HybridMRIDataset
    print("   ✓ Data module imports successful")
except Exception as e:
    print(f"   ✗ Data module imports failed: {e}")
    sys.exit(1)

try:
    print("\n3. Testing utils imports...")
    from utils.config_loader import load_config, load_degradation_config
    print("   ✓ Utils imports successful")
except Exception as e:
    print(f"   ✗ Utils imports failed: {e}")
    sys.exit(1)

try:
    print("\n4. Testing model imports...")
    from models.model_swinmr import ModelSwinMR
    print("   ✓ Model imports successful")
except Exception as e:
    print(f"   ✗ Model imports failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All imports successful!")
print("="*60)
print("\nYou can now run:")
print("  python scripts/validate_setup.py --config configs/train_config.json")
