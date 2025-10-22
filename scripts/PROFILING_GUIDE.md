# Model Profiling Guide

## Overview

The `profile_model.py` script provides comprehensive profiling of the SwinMR model, measuring:

- **FLOPs** (Floating Point Operations) - Computational complexity
- **Parameter Count** - Model size and memory footprint
- **Inference Speed** - Latency and throughput
- **GPU Memory Usage** - Memory requirements per batch size
- **Batch Size Analysis** - Performance scaling with batch size

## Installation

Install optional profiling tools for FLOPs measurement:

```bash
# Option 1: fvcore (recommended, more accurate)
pip install fvcore

# Option 2: thop (alternative)
pip install thop

# For LPIPS (if profiling perceptual metrics)
pip install lpips
```

## Basic Usage

### 1. Profile with Default Settings

```bash
python scripts/profile_model.py --config configs/train_config.json
```

**Output:**
- Parameter count and model size
- FLOPs and GFLOPs
- Inference time statistics
- Memory usage per batch size (1, 2, 4, 8, 16)

### 2. Custom Batch Sizes

```bash
python scripts/profile_model.py \
    --config configs/train_config.json \
    --batch-sizes 1,4,8,16,32
```

### 3. CPU Profiling

```bash
python scripts/profile_model.py \
    --config configs/train_config.json \
    --device cpu
```

### 4. Save Results to JSON

```bash
python scripts/profile_model.py \
    --config configs/train_config.json \
    --output results/profile_results.json
```

### 5. Custom Number of Runs

```bash
python scripts/profile_model.py \
    --config configs/train_config.json \
    --num-runs 100  # More runs = more accurate timing
```

## Output Explanation

### Console Output

```
================================================================================
                          MODEL PROFILING
================================================================================

Configuration:
  - Config file: configs/train_config.json
  - Model: swinir
  - Image size: 256
  - Device: cuda
  - Batch sizes: [1, 2, 4, 8, 16]

[1/5] Creating model...
  ✓ Model created

[2/5] Analyzing model architecture...
  ✓ Total parameters: 10,234,567
  ✓ Trainable parameters: 10,234,567
  ✓ Model size: 39.05 MB

[3/5] Measuring FLOPs...
  ✓ FLOPs: 45,678,901,234
  ✓ GFLOPs: 45.68
  ✓ FLOPs/param: 4,463.21

[4/5] Profiling single image inference...
  ✓ Mean time: 123.45 ± 5.67 ms
  ✓ Median time: 122.30 ms
  ✓ Min/Max: 115.20 / 145.80 ms
  ✓ GPU memory (peak): 512.34 MB

[5/5] Profiling batch sizes...

  Testing batch size: 1
    ✓ Inference time: 123.45 ± 5.67 ms
    ✓ Throughput: 8.10 images/sec
    ✓ GPU memory: 512.34 MB (peak)
    ✓ Activation memory: 256.78 MB

  Testing batch size: 2
    ✓ Inference time: 215.67 ± 8.90 ms
    ✓ Throughput: 9.27 images/sec
    ✓ GPU memory: 768.90 MB (peak)
    ✓ Activation memory: 513.56 MB

  [...]

================================================================================
                          PROFILING SUMMARY
================================================================================

Model Architecture:
  - Parameters: 10,234,567 (39.05 MB)
  - GFLOPs: 45.68

Single Image Performance:
  - Inference time: 123.45 ms
  - Throughput: 8.10 images/sec

Batch Size Recommendations:
  - Batch 1: 8.1 img/s, 512 MB GPU memory
  - Batch 2: 9.3 img/s, 769 MB GPU memory
  - Batch 4: 10.5 img/s, 1280 MB GPU memory
  - Batch 8: 11.2 img/s, 2305 MB GPU memory

✓ Results saved to: results/profile_results.json
```

### JSON Output Structure

```json
{
  "config": {
    "model": "swinir",
    "image_size": 256,
    "device": "cuda",
    "config_file": "configs/train_config.json"
  },
  "architecture": {
    "total_parameters": 10234567,
    "trainable_parameters": 10234567,
    "model_size_mb": 39.05,
    "flops": 45678901234,
    "gflops": 45.68
  },
  "single_image": {
    "input_shape": [1, 1, 256, 256],
    "timing": {
      "mean_ms": 123.45,
      "std_ms": 5.67,
      "min_ms": 115.20,
      "max_ms": 145.80,
      "median_ms": 122.30,
      "fps": 8.10
    },
    "memory": {
      "before_mb": 255.56,
      "after_mb": 255.56,
      "peak_mb": 512.34,
      "activation_mb": 256.78
    }
  },
  "batch_profiling": {
    "1": { ... },
    "2": { ... },
    "4": { ... }
  }
}
```

## Metrics Explained

### 1. FLOPs (Floating Point Operations)

**What it measures:** Computational complexity of one forward pass

**Interpretation:**
- **GFLOPs < 10**: Lightweight model
- **GFLOPs 10-50**: Medium model (SwinMR is here)
- **GFLOPs > 50**: Heavy model

**Use cases:**
- Compare model efficiency
- Estimate computational requirements
- Optimize for edge devices

### 2. Parameters

**What it measures:** Total number of trainable weights

**Interpretation:**
- Affects model capacity and memory footprint
- More parameters ≠ better performance (risk of overfitting)

**SwinMR typical values:**
- **Depths [6,6,6,6,6,6]**: ~10-15M parameters
- **Embed_dim 180**: ~10M parameters

### 3. Inference Time

**What it measures:** Time to process one batch (milliseconds)

**Interpretation:**
- **< 50ms**: Real-time capable
- **50-200ms**: Near real-time (SwinMR 256x256)
- **> 200ms**: Slower, batch processing

**Affected by:**
- Image size (256 vs 512)
- Batch size
- GPU/CPU
- Model depth

### 4. GPU Memory

**What it measures:** VRAM used during inference

**Components:**
- **Model weights**: Fixed (e.g., 39 MB for 10M params)
- **Activations**: Scales with batch size and image size
- **Peak memory**: Maximum during forward pass

**Interpretation:**
- **< 2GB**: Fits on small GPUs
- **2-8GB**: Medium GPUs (GTX 1080, RTX 2060)
- **> 8GB**: Requires high-end GPUs

### 5. Throughput (images/sec)

**What it measures:** Number of images processed per second

**Interpretation:**
- Higher = better efficiency
- Usually increases with batch size (up to a point)
- Limited by GPU memory

## Batch Size Recommendations

### How to Choose Batch Size

1. **GPU Memory Constraint**
   - Check profiling results for max batch size that fits
   - Leave 20% headroom for gradient computation during training

2. **Throughput vs Latency**
   - Larger batches = higher throughput
   - Smaller batches = lower latency

3. **Training Considerations**
   - Larger batches = more stable gradients, faster training
   - Smaller batches = better generalization (debated)

### Example Decision Matrix

**For RTX 3090 (24GB VRAM):**
```
Image Size: 256x256
- Batch 1:  512 MB  →  Use for inference/testing
- Batch 4:  1.3 GB  →  Good for training
- Batch 8:  2.3 GB  →  Optimal for training
- Batch 16: 4.5 GB  →  Maximum throughput
- Batch 32: OOM     →  Too large
```

**For RTX 2060 (6GB VRAM):**
```
Image Size: 256x256
- Batch 1:  512 MB  →  Safe
- Batch 2:  769 MB  →  Good for training
- Batch 4:  1.3 GB  →  Optimal for training
- Batch 8:  2.3 GB  →  Risk of OOM with gradients
```

## Advanced Usage

### Profile Different Model Configurations

```bash
# Small model
python scripts/profile_model.py \
    --config configs/train_config_small.json \
    --output results/profile_small.json

# Large model
python scripts/profile_model.py \
    --config configs/train_config_large.json \
    --output results/profile_large.json
```

### Compare Models

```python
import json

with open('results/profile_small.json') as f:
    small = json.load(f)

with open('results/profile_large.json') as f:
    large = json.load(f)

print(f"Small model: {small['architecture']['gflops']:.2f} GFLOPs")
print(f"Large model: {large['architecture']['gflops']:.2f} GFLOPs")
print(f"Speedup: {small['single_image']['timing']['fps'] / large['single_image']['timing']['fps']:.2f}x")
```

## Troubleshooting

### Issue: "fvcore not installed"

**Solution:**
```bash
pip install fvcore
# Or alternative:
pip install thop
```

### Issue: "CUDA out of memory"

**Solution:**
- Reduce batch size
- Use `--device cpu` for profiling
- Close other applications using GPU

### Issue: Large variance in timing

**Solution:**
- Increase `--num-runs` (e.g., 100 or 200)
- Ensure no other processes are using GPU
- Use consistent GPU clock speeds

### Issue: Results seem inaccurate

**Checklist:**
- ✓ GPU is not being used by other processes
- ✓ Sufficient warmup runs (default: 10)
- ✓ Enough measurement runs (default: 50)
- ✓ Consistent power mode (not power-saving)

## Integration with Training

Use profiling results to configure training:

```python
# Based on profiling results showing:
# - Batch 8: 2.3 GB GPU memory
# - Your GPU: 8 GB total
# - Training needs ~2x memory for gradients

# In train_config.json:
{
  "data": {
    "batch_size": 4  // Safe: 4 * 2.3GB / 8 * 2 = 2.3 GB ✓
  }
}
```

## Performance Optimization Tips

1. **Reduce Image Size**
   - 512→256: ~4x faster, 1/4 memory

2. **Mixed Precision Training**
   - Enable `mixed_precision: true` in config
   - ~2x faster, ~1/2 memory

3. **Gradient Checkpointing**
   - Trade compute for memory
   - Slower but fits larger batches

4. **Model Architecture**
   - Reduce depths: [6,6,6,6,6,6] → [6,6,6,6]
   - Reduce embed_dim: 180 → 128

## See Also

- `validate_setup.py` - Validate complete training setup
- `train.py` - Main training script
- `test.py` - Evaluation script
- Configuration guide in `configs/README.md`
