"""
Model Profiling Script

Profiles the SwinMR model to measure:
- FLOPs (Floating Point Operations)
- Parameter count
- Memory usage per batch size
- Inference speed
- GPU memory requirements

Usage:
    python scripts/profile_model.py --config configs/train_config.json
    python scripts/profile_model.py --config configs/train_config.json --batch-sizes 1,2,4,8
    python scripts/profile_model.py --config configs/train_config.json --output profile_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_swinmr import ModelSwinMR
from utils.config_loader import load_config


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def measure_flops(model, input_shape, device='cpu'):
    """
    Measure FLOPs using torch profiler or fvcore.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to run on

    Returns:
        FLOPs count (or None if tools not available)
    """
    try:
        # Try using fvcore (more accurate)
        from fvcore.nn import FlopCountAnalysis, parameter_count

        model = model.to(device)
        input_tensor = torch.randn(input_shape).to(device)

        flops = FlopCountAnalysis(model, input_tensor)
        total_flops = flops.total()

        return total_flops

    except ImportError:
        print("  ⚠ fvcore not installed. Install for FLOPs counting: pip install fvcore")

        # Try using thop as fallback
        try:
            from thop import profile, clever_format

            model = model.to(device)
            input_tensor = torch.randn(input_shape).to(device)

            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
            return flops

        except ImportError:
            print("  ⚠ thop not installed. Install for FLOPs counting: pip install thop")
            return None


def measure_inference_time(model, input_shape, device='cpu', num_runs=100, warmup_runs=10):
    """
    Measure inference time.

    Args:
        model: PyTorch model
        input_shape: Input shape (B, C, H, W)
        device: Device to run on
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Synchronize if using GPU
    if device != 'cpu':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_tensor)

            if device != 'cpu':
                torch.cuda.synchronize()

            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'fps': float(input_shape[0] / (np.mean(times) / 1000))  # Images per second
    }


def measure_memory_usage(model, input_shape, device='cuda'):
    """
    Measure GPU memory usage.

    Args:
        model: PyTorch model
        input_shape: Input shape (B, C, H, W)
        device: Device to run on

    Returns:
        Dictionary with memory statistics
    """
    if device == 'cpu' or not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    model = model.to(device)
    model.eval()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure before inference
    mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    # Run inference
    input_tensor = torch.randn(input_shape).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    # Measure after inference
    mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Clean up
    del input_tensor, output
    torch.cuda.empty_cache()

    return {
        'before_mb': float(mem_before),
        'after_mb': float(mem_after),
        'peak_mb': float(mem_peak),
        'activation_mb': float(mem_peak - mem_before)
    }


def profile_batch_sizes(model, input_size, batch_sizes, device='cuda'):
    """
    Profile model with different batch sizes.

    Args:
        model: PyTorch model
        input_size: Input image size (H, W)
        batch_sizes: List of batch sizes to test
        device: Device to run on

    Returns:
        Dictionary with results for each batch size
    """
    results = {}

    for batch_size in batch_sizes:
        print(f"\n  Testing batch size: {batch_size}")
        input_shape = (batch_size, 1, input_size[0], input_size[1])

        try:
            # Measure inference time
            timing = measure_inference_time(model, input_shape, device, num_runs=50, warmup_runs=5)
            print(f"    ✓ Inference time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
            print(f"    ✓ Throughput: {timing['fps']:.2f} images/sec")

            # Measure memory (GPU only)
            memory = None
            if device != 'cpu' and torch.cuda.is_available():
                memory = measure_memory_usage(model, input_shape, device)
                if 'error' not in memory:
                    print(f"    ✓ GPU memory: {memory['peak_mb']:.2f} MB (peak)")
                    print(f"    ✓ Activation memory: {memory['activation_mb']:.2f} MB")

            results[batch_size] = {
                'input_shape': list(input_shape),
                'timing': timing,
                'memory': memory
            }

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"    ✗ Out of memory!")
                results[batch_size] = {'error': 'OOM'}
                torch.cuda.empty_cache()
                break
            else:
                raise e

    return results


def main():
    parser = argparse.ArgumentParser(description='Profile SwinMR model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16',
                       help='Comma-separated batch sizes to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--num-runs', type=int, default=50,
                       help='Number of inference runs for timing')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'

    print("="*80)
    print(" "*30 + "MODEL PROFILING")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  - Config file: {args.config}")
    print(f"  - Model: {config['model']['net_type']}")
    print(f"  - Image size: {config['model']['img_size']}")
    print(f"  - Device: {device}")
    print(f"  - Batch sizes: {batch_sizes}")

    # Create model
    print("\n[1/5] Creating model...")

    model_opt = {
        'netG': config['model'].copy(),
        'path': {
            'models': config['paths']['checkpoints'],
            'log': config['paths']['logs'],
            'samples': config['paths']['samples']
        },
        'gpu_ids': config['system']['gpu_ids'],
        'is_train': False,
        'dist': config['system'].get('distributed', False),
        'train': {
            'freeze_patch_embedding': False,
            'E_decay': 0
        },
        'datasets': {
            'train': {'batch_size': 1}
        }
    }

    model_wrapper = ModelSwinMR(model_opt)
    model = model_wrapper.netG  # Get the actual network

    print(f"  ✓ Model created")

    # Count parameters
    print("\n[2/5] Analyzing model architecture...")
    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)

    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    print(f"  ✓ Model size: {model_size_mb:.2f} MB")

    # Measure FLOPs
    print("\n[3/5] Measuring FLOPs...")
    img_size = config['model']['img_size']
    input_shape = (1, 1, img_size, img_size)

    flops = measure_flops(model, input_shape, device)
    if flops is not None:
        gflops = flops / 1e9
        print(f"  ✓ FLOPs: {flops:,}")
        print(f"  ✓ GFLOPs: {gflops:.2f}")
        print(f"  ✓ FLOPs/param: {flops/total_params:.2f}")
    else:
        print(f"  ⚠ FLOPs measurement skipped (install fvcore or thop)")
        gflops = None

    # Profile single image
    print("\n[4/5] Profiling single image inference...")
    single_timing = measure_inference_time(model, input_shape, device, num_runs=args.num_runs)
    print(f"  ✓ Mean time: {single_timing['mean_ms']:.2f} ± {single_timing['std_ms']:.2f} ms")
    print(f"  ✓ Median time: {single_timing['median_ms']:.2f} ms")
    print(f"  ✓ Min/Max: {single_timing['min_ms']:.2f} / {single_timing['max_ms']:.2f} ms")

    if device != 'cpu' and torch.cuda.is_available():
        single_memory = measure_memory_usage(model, input_shape, device)
        if 'error' not in single_memory:
            print(f"  ✓ GPU memory (peak): {single_memory['peak_mb']:.2f} MB")

    # Profile different batch sizes
    print("\n[5/5] Profiling batch sizes...")
    batch_results = profile_batch_sizes(model, (img_size, img_size), batch_sizes, device)

    # Compile results
    results = {
        'config': {
            'model': config['model']['net_type'],
            'image_size': img_size,
            'device': device,
            'config_file': args.config
        },
        'architecture': {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(model_size_mb),
            'flops': int(flops) if flops is not None else None,
            'gflops': float(gflops) if gflops is not None else None
        },
        'single_image': {
            'input_shape': list(input_shape),
            'timing': single_timing,
            'memory': single_memory if device != 'cpu' and torch.cuda.is_available() else None
        },
        'batch_profiling': batch_results
    }

    # Print summary
    print("\n" + "="*80)
    print(" "*30 + "PROFILING SUMMARY")
    print("="*80)

    print("\nModel Architecture:")
    print(f"  - Parameters: {total_params:,} ({model_size_mb:.2f} MB)")
    if gflops:
        print(f"  - GFLOPs: {gflops:.2f}")

    print("\nSingle Image Performance:")
    print(f"  - Inference time: {single_timing['mean_ms']:.2f} ms")
    print(f"  - Throughput: {single_timing['fps']:.2f} images/sec")

    if device != 'cpu' and torch.cuda.is_available():
        print("\nBatch Size Recommendations:")
        for bs, data in batch_results.items():
            if 'error' not in data:
                mem_mb = data['memory']['peak_mb'] if data['memory'] else 0
                throughput = data['timing']['fps']
                print(f"  - Batch {bs}: {throughput:.1f} img/s, {mem_mb:.0f} MB GPU memory")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {args.output}")

    print("\n" + "="*80)
    print(" "*25 + "✓ PROFILING COMPLETE")
    print("="*80)

    return results


if __name__ == '__main__':
    main()
