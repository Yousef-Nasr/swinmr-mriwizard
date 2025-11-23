#!/usr/bin/env python3
"""
Test Validation Degradation Reproducibility

This script verifies that validation degradation is reproducible across runs.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader import HybridMRIDataset


def test_reproducibility(config_path, data_dir, num_samples=5):
    """
    Test that the same degradation is applied consistently.
    
    Args:
        config_path: Path to validation degradation config
        data_dir: Directory with validation data
        num_samples: Number of samples to test
    """
    print("=" * 70)
    print("Testing Validation Degradation Reproducibility")
    print("=" * 70)
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\nConfig: {config_path}")
    print(f"Deterministic: {config.get('execution', {}).get('deterministic', False)}")
    print(f"Seed: {config.get('execution', {}).get('seed', None)}")
    
    # Create dataset twice
    print("\n[1/3] Creating first dataset instance...")
    dataset1 = HybridMRIDataset(
        data_dir=data_dir,
        degradation_config=config_path,
        patch_size=None,
        cache_dir="./cache_test1",
        use_augmentation=False,
        return_metadata=True
    )
    
    print("[2/3] Creating second dataset instance...")
    dataset2 = HybridMRIDataset(
        data_dir=data_dir,
        degradation_config=config_path,
        patch_size=None,
        cache_dir="./cache_test2",
        use_augmentation=False,
        return_metadata=True
    )
    
    # Test samples
    print(f"\n[3/3] Testing {num_samples} samples...")
    print("-" * 70)
    
    all_identical = True
    
    for idx in range(min(num_samples, len(dataset1))):
        # Load same sample from both datasets
        input1, target1, meta1 = dataset1[idx]
        input2, target2, meta2 = dataset2[idx]
        
        # Convert to numpy for comparison
        input1_np = input1.numpy()
        input2_np = input2.numpy()
        target1_np = target1.numpy()
        target2_np = target2.numpy()
        
        # Check if targets are identical (they should be)
        target_identical = np.allclose(target1_np, target2_np, rtol=1e-5, atol=1e-8)
        
        # Check if degraded inputs are identical (they should be in deterministic mode)
        input_identical = np.allclose(input1_np, input2_np, rtol=1e-5, atol=1e-8)
        
        # Calculate difference
        if input_identical:
            max_diff = 0.0
        else:
            max_diff = np.abs(input1_np - input2_np).max()
        
        status = "✓ PASS" if input_identical else "✗ FAIL"
        print(f"Sample {idx}: {status} (max diff: {max_diff:.2e})")
        
        if not input_identical:
            all_identical = False
            print(f"  Target identical: {target_identical}")
            print(f"  Input shape: {input1_np.shape}")
            print(f"  Input1 range: [{input1_np.min():.4f}, {input1_np.max():.4f}]")
            print(f"  Input2 range: [{input2_np.min():.4f}, {input2_np.max():.4f}]")
    
    print("-" * 70)
    
    if all_identical:
        print("\n✓ SUCCESS: All samples are reproducible!")
        print("Validation degradation is working correctly.")
        return True
    else:
        print("\n✗ FAILURE: Some samples differ between runs!")
        print("Check that deterministic=true and seed is set in config.")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test validation degradation reproducibility"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/degradation_modrate_val.json",
        help="Path to validation degradation config"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../S1",
        help="Path to validation data directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"✗ Config file not found: {args.config}")
        sys.exit(1)
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"✗ Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Run test
    success = test_reproducibility(args.config, args.data_dir, args.num_samples)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

