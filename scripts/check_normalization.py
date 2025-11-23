"""
Script to check normalization consistency between input (degraded) and target images
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader import HybridMRIDataset
from torch.utils.data import DataLoader
import numpy as np

def check_normalization(data_dir, degradation_config, num_samples=10):
    """
    Check normalization statistics for input and target images
    
    Args:
        data_dir: Directory with data
        degradation_config: Path to degradation config
        num_samples: Number of samples to check
    """
    print("="*60)
    print("Normalization Consistency Check")
    print("="*60)
    
    # Create dataset
    dataset = HybridMRIDataset(
        data_dir=data_dir,
        degradation_config=degradation_config,
        patch_size=None,
        use_augmentation=False,
        return_metadata=True
    )
    
    print(f"\nChecking {num_samples} samples from {data_dir}")
    print(f"Using degradation config: {degradation_config}\n")
    
    # Statistics collectors
    input_mins = []
    input_maxs = []
    input_means = []
    input_stds = []
    
    target_mins = []
    target_maxs = []
    target_means = []
    target_stds = []
    
    # Check samples
    for i in range(min(num_samples, len(dataset))):
        input_img, target_img, metadata = dataset[i]
        
        # Convert to numpy
        input_np = input_img.numpy()
        target_np = target_img.numpy()
        
        # Collect statistics
        input_mins.append(input_np.min())
        input_maxs.append(input_np.max())
        input_means.append(input_np.mean())
        input_stds.append(input_np.std())
        
        target_mins.append(target_np.min())
        target_maxs.append(target_np.max())
        target_means.append(target_np.mean())
        target_stds.append(target_np.std())
        
        print(f"Sample {i}:")
        print(f"  Input  (degraded): min={input_np.min():.4f}, max={input_np.max():.4f}, "
              f"mean={input_np.mean():.4f}, std={input_np.std():.4f}")
        print(f"  Target (GT):       min={target_np.min():.4f}, max={target_np.max():.4f}, "
              f"mean={target_np.mean():.4f}, std={target_np.std():.4f}")
        print(f"  Source: {Path(metadata['source']).name}")
        print()
    
    # Print summary statistics
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print("\nInput (Degraded) Images:")
    print(f"  Min range:  [{np.min(input_mins):.4f}, {np.max(input_mins):.4f}]")
    print(f"  Max range:  [{np.min(input_maxs):.4f}, {np.max(input_maxs):.4f}]")
    print(f"  Mean range: [{np.min(input_means):.4f}, {np.max(input_means):.4f}]")
    print(f"  Std range:  [{np.min(input_stds):.4f}, {np.max(input_stds):.4f}]")
    
    print("\nTarget (Ground Truth) Images:")
    print(f"  Min range:  [{np.min(target_mins):.4f}, {np.max(target_mins):.4f}]")
    print(f"  Max range:  [{np.min(target_maxs):.4f}, {np.max(target_maxs):.4f}]")
    print(f"  Mean range: [{np.min(target_means):.4f}, {np.max(target_means):.4f}]")
    print(f"  Std range:  [{np.min(target_stds):.4f}, {np.max(target_stds):.4f}]")
    
    # Check consistency
    print("\n" + "="*60)
    print("CONSISTENCY CHECK")
    print("="*60)
    
    # Check if ranges are similar
    input_max_avg = np.mean(input_maxs)
    target_max_avg = np.mean(target_maxs)
    max_diff = abs(input_max_avg - target_max_avg)
    
    print(f"\nAverage max value:")
    print(f"  Input:  {input_max_avg:.4f}")
    print(f"  Target: {target_max_avg:.4f}")
    print(f"  Difference: {max_diff:.4f}")
    
    if max_diff > 0.1:
        print("\n⚠️  WARNING: Input and target have different normalization!")
        print("   This can cause training issues.")
    else:
        print("\n✓ Input and target normalization appears consistent.")
    
    # Check if values are in [0, 1] range
    all_input_in_range = all(0 <= m <= 1 for m in input_mins) and all(0 <= m <= 1 for m in input_maxs)
    all_target_in_range = all(0 <= m <= 1 for m in target_mins) and all(0 <= m <= 1 for m in target_maxs)
    
    print(f"\nValue range check:")
    print(f"  Input in [0, 1]:  {all_input_in_range}")
    print(f"  Target in [0, 1]: {all_target_in_range}")
    
    if not (all_input_in_range and all_target_in_range):
        print("\n⚠️  WARNING: Some values are outside [0, 1] range!")
    else:
        print("\n✓ All values are in [0, 1] range.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check normalization consistency')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--config', type=str, required=True, help='Degradation config')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to check')
    
    args = parser.parse_args()
    
    check_normalization(args.data_dir, args.config, args.num_samples)

