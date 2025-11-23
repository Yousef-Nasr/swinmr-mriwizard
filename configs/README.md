# Configuration Documentation

This directory contains JSON configuration files for training and degradation.

## Configuration Files

### 1. `train_config.json` - Main Training Configuration

The main configuration file that controls all aspects of training.

#### Structure

```json
{
  "experiment_name": "string",  // Name of your experiment
  "paths": { ... },              // Output paths
  "data": { ... },               // Data loading settings
  "model": { ... },              // Model architecture
  "training": { ... },           // Training settings
  "evaluation": { ... },         // Evaluation settings
  "logging": { ... },            // Logging settings
  "system": { ... }              // System settings
}
```

#### Detailed Parameters

##### `paths`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `root` | string | Root directory for outputs | `./results` |
| `checkpoints` | string | Checkpoint save directory | `./results/checkpoints` |
| `logs` | string | Log directory for TensorBoard | `./results/logs` |
| `samples` | string | Sample output directory | `./results/samples` |

##### `data`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `train_dir` | string | Training data directory | **Required** |
| `val_dir` | string | Validation data directory | **Required** |
| `test_dir` | string | Test data directory | Optional |
| `degradation_config` | string | Path to degradation JSON config | **Required** |
| `val_degradation_config` | string | Path to validation degradation config (for reproducible validation) | Optional |
| `batch_size` | int | Training batch size | `8` |
| `val_batch_size` | int | Validation batch size | `8` |
| `num_workers` | int | DataLoader workers | `4` |
| `patch_size` | int/null | Extract patches of this size (null = full image) | `256` |
| `use_patches` | bool | Whether to use patch extraction | `true` |
| `cache_dicom_conversions` | bool | Cache DICOM→NPY conversions | `true` |
| `prefetch_factor` | int | Prefetch factor for DataLoader | `2` |

##### `data.augmentation`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `random_flip` | bool | Enable random flipping | `true` |
| `flip_probability` | float | Probability of flip (0-1) | `0.5` |

##### `model`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `net_type` | string | Network type (`swinir`) | `swinir` |
| `upscale` | int | Upscale factor (1 = reconstruction only) | `1` |
| `in_chans` | int | Input channels (1 for grayscale MRI) | `1` |
| `img_size` | int | Image size for model | `256` |
| `window_size` | int | Window size for Swin Transformer | `8` |
| `img_range` | float | Image value range | `1.0` |
| `depths` | array[int] | Depths of each stage | `[6,6,6,6,6,6]` |
| `embed_dim` | int | Embedding dimension | `180` |
| `num_heads` | array[int] | Attention heads per stage | `[6,6,6,6,6,6]` |
| `mlp_ratio` | int | MLP expansion ratio | `2` |
| `upsampler` | string | Upsampler type (empty for reconstruction) | `""` |
| `resi_connection` | string | Residual connection type | `1conv` |

##### `training`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `epochs` | int | Total training epochs | `100` |
| `print_freq` | int | Print frequency (batches) | `100` |
| `save_freq` | int | Save frequency (batches) | `1000` |
| `val_freq` | int | Validation frequency (epochs) | `1` |

##### `training.loss.components`

Multi-component loss function:

**Spatial Loss (Charbonnier)**:
```json
"spatial": {
  "type": "charbonnier",
  "weight": 1.0,      // Loss weight
  "eps": 1e-3         // Smoothing parameter
}
```

**Frequency Loss (FFT)**:
```json
"frequency": {
  "type": "fft_mse",
  "weight": 0.1       // Lower weight for frequency domain
}
```

**Perceptual Loss (VGG19)**:
```json
"perceptual": {
  "type": "vgg19",
  "weight": 0.0025,                         // Small weight
  "layers": [2, 7, 16, 25, 34],            // VGG19 layers to use
  "layer_weights": [0.1, 0.1, 1.0, 1.0, 1.0]  // Per-layer weights
}
```

##### `training.optimizer`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `type` | string | Optimizer type (`adam`) | `adam` |
| `lr` | float | Learning rate | `2e-4` |
| `betas` | array[float] | Adam beta parameters | `[0.9, 0.999]` |
| `weight_decay` | float | Weight decay (L2 regularization) | `0` |

##### `training.optimizer.scheduler`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `type` | string | Scheduler type (`multistep`) | `multistep` |
| `milestones` | array[int] | Steps to reduce LR | `[25000, 50000, 75000]` |
| `gamma` | float | LR decay factor | `0.5` |

##### `training.early_stopping`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | bool | Enable early stopping | `true` |
| `patience` | int | Epochs to wait before stopping | `10` |
| `metric` | string | Metric to monitor (`val_ssim`, `val_loss`) | `val_ssim` |
| `mode` | string | `max` (for SSIM) or `min` (for loss) | `max` |
| `min_delta` | float | Minimum change to qualify as improvement | `0.0001` |

##### `training.gradient`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `clip_enabled` | bool | Enable gradient clipping | `true` |
| `clip_value` | float | Max gradient norm | `1.0` |

---

### 2. `degradation_all.json` - Comprehensive Degradation Configuration

Full configuration with all undersampling patterns enabled.

#### Structure

```json
{
  "name": "string",           // Descriptive name
  "description": "string",    // Description
  "noise": { ... },           // Gaussian k-space noise
  "undersampling": { ... },   // Undersampling patterns
  "artifacts": { ... },       // Optional artifacts
  "execution": { ... }        // Execution settings
}
```

#### Noise Configuration

```json
"noise": {
  "enabled": true,              // Enable/disable noise
  "type": "gaussian_kspace",    // Noise type
  "mean": 0.0,                  // Mean value
  "std_range": [0.005, 0.05],   // Random std sampling range
  "relative": true,             // Relative to signal amplitude
  "reference": "std"            // Reference metric: "std", "rms", "p99", "max"
}
```

- **`std_range`**: Sample noise std uniformly from this range each time
- **`relative=true`**: Noise level is relative to signal strength
- **`reference="std"`**: Use signal standard deviation as reference

#### Undersampling Configuration

```json
"undersampling": {
  "strategy": "one_of",  // or "apply_all"
  "patterns": [
    {
      "name": "pattern_type",
      "enabled": true,
      "weight": 1.0,
      "params": { ... }
    }
  ]
}
```

**Strategies**:
- **`one_of`**: Randomly choose ONE pattern per sample (weighted sampling)
- **`apply_all`**: Apply ALL enabled patterns sequentially

**Available Patterns**:

##### 1. Random Undersampling

```json
{
  "name": "random",
  "enabled": true,
  "weight": 1.0,
  "params": {
    "prob_range": [0.3, 0.7],    // Keep probability range
    "center_fraction": 0.08,     // Fully-sampled center fraction
    "axis": -2                   // Phase-encode axis
  }
}
```

##### 2. Uniform Undersampling

```json
{
  "name": "uniform",
  "enabled": true,
  "weight": 1.0,
  "params": {
    "R_range": [2, 8],           // Acceleration factor range
    "center_fraction": 0.08,
    "axis": -2
  }
}
```

##### 3. Partial Fourier

```json
{
  "name": "partial_fourier",
  "enabled": true,
  "weight": 1.0,
  "params": {
    "fraction_ranges": [[0.6, 0.9], [0.6, 0.9]],  // Per-axis fraction ranges
    "directions": ["+", "+"]                       // Direction per axis
  }
}
```

##### 4. K-max Undersampling

```json
{
  "name": "kmax",
  "enabled": true,
  "weight": 1.0,
  "params": {
    "fraction_ranges": [[0.7, 0.95], [0.7, 0.95]]  // Per-axis retention fraction
  }
}
```

##### 5. Elliptical Undersampling

```json
{
  "name": "elliptical",
  "enabled": true,
  "weight": 1.0,
  "params": {
    "radii_ranges": [[0.7, 0.95], [0.7, 0.95]]  // Per-axis radii fraction
  }
}
```

#### Artifacts Configuration (Optional)

All artifacts are disabled by default in `degradation_all.json`. Enable as needed:

##### Motion Artifacts

```json
"motion": {
  "enabled": false,
  "probability": 0.3,         // Apply to 30% of samples
  "params": {
    "degrees_range": [5, 15],       // Rotation range (degrees)
    "translation_range": [5, 15],   // Translation range (pixels)
    "num_transforms": 2             // Number of motion events
  }
}
```

##### Ghosting

```json
"ghosting": {
  "enabled": false,
  "probability": 0.2,
  "params": {
    "num_ghosts_range": [2, 4],
    "intensity_range": [0.05, 0.25]
  }
}
```

##### Spike Noise

```json
"spike": {
  "enabled": false,
  "probability": 0.15,
  "params": {
    "num_spikes_range": [1, 5],
    "relative_amp_range": [1.0, 3.0]
  }
}
```

##### Gibbs Ringing

```json
"gibbs": {
  "enabled": false,
  "probability": 0.2,
  "params": {
    "fraction_range": [0.6, 0.9],
    "axes": [-1, -2]  // NEW: Can specify multiple axes for random selection!
  }
}
```

**NEW Feature**: When `axes` contains multiple values (e.g., `[-1, -2]`), the pipeline will **randomly choose ONE axis** each time the degradation is applied. This increases training diversity.

**Examples**:
- `"axes": [-1]` - Always apply to axis -1 (readout direction)
- `"axes": [-2]` - Always apply to axis -2 (phase encoding direction)
- `"axes": [-1, -2]` - Randomly choose -1 OR -2 for each sample

#### Execution Settings

```json
"execution": {
  "apply_order": ["noise", "undersampling", "artifacts"],  // Application order
  "seed": null,                 // Random seed (null = random)
  "deterministic": false        // Deterministic degradation
}
```

---

### 3. `degradation_minimal.json` - Minimal Configuration for Testing

Simplified configuration with only one undersampling pattern. Use this for:
- Testing your pipeline
- Quick training experiments
- Debugging

```json
{
  "name": "minimal_degradation",
  "noise": {
    "enabled": true,
    "std_range": [0.01, 0.02]  // Lower noise for testing
  },
  "undersampling": {
    "strategy": "one_of",
    "patterns": [
      {
        "name": "random",
        "enabled": true,
        "weight": 1.0,
        "params": {
          "prob_range": [0.5, 0.6]  // Lighter undersampling
        }
      }
    ]
  },
  "artifacts": {}
}
```

---

## Creating Custom Configurations

### Example 1: Light Degradation for Fine-tuning

```json
{
  "name": "light_degradation",
  "noise": {
    "enabled": true,
    "std_range": [0.001, 0.01]  // Very light noise
  },
  "undersampling": {
    "strategy": "one_of",
    "patterns": [
      {
        "name": "uniform",
        "enabled": true,
        "params": {
          "R_range": [2, 4],      // Low acceleration factors
          "center_fraction": 0.12  // More center preserved
        }
      }
    ]
  }
}
```

### Example 2: Heavy Degradation for Robustness

```json
{
  "name": "heavy_degradation",
  "noise": {
    "enabled": true,
    "std_range": [0.05, 0.15]  // High noise
  },
  "undersampling": {
    "strategy": "one_of",
    "patterns": [
      {
        "name": "random",
        "params": {
          "prob_range": [0.15, 0.35]  // High acceleration
        }
      }
    ]
  },
  "artifacts": {
    "motion": {
      "enabled": true,
      "probability": 0.5,  // Apply to 50% of samples
      "params": {
        "degrees_range": [10, 30],
        "translation_range": [10, 30]
      }
    }
  }
}
```

### Example 3: Single Pattern for Controlled Experiments

```json
{
  "name": "uniform_only",
  "noise": {
    "enabled": false  // No noise
  },
  "undersampling": {
    "strategy": "one_of",
    "patterns": [
      {
        "name": "uniform",
        "enabled": true,
        "params": {
          "R_range": [4, 4],  // Fixed R=4
          "center_fraction": 0.08
        }
      }
    ]
  }
}
```

---

## Command-Line Config Overrides

Override any config parameter from command line:

```bash
# Override data paths
python training/train.py \
    --config configs/train_config.json \
    --data.train_dir ./my_data/train \
    --data.val_dir ./my_data/val

# Override training parameters
python training/train.py \
    --config configs/train_config.json \
    --training.epochs 50 \
    --training.optimizer.lr 1e-4 \
    --data.batch_size 16

# Use different degradation config
python training/train.py \
    --config configs/train_config.json \
    --data.degradation_config configs/degradation_minimal.json
```

---

## Best Practices

1. **Start with Minimal**: Test pipeline with `degradation_minimal.json` first
2. **Gradual Complexity**: Add degradations incrementally
3. **Monitor Metrics**: Watch PSNR/SSIM to ensure training is effective
4. **Version Control**: Keep different configs for different experiments
5. **Document Changes**: Use descriptive `name` and `description` fields

---

## Reproducible Validation Degradation

For consistent validation metrics across epochs, use separate degradation configs for training and validation:

```json
{
  "data": {
    "degradation_config": "./configs/degradation_modrate.json",
    "val_degradation_config": "./configs/degradation_modrate_val.json"
  }
}
```

**Key differences in validation config:**
- `augmentation.enabled`: false (no random augmentation)
- `execution.seed`: 42 (fixed seed)
- `execution.deterministic`: true (reproducible degradation)

**Benefits:**
- ✓ Same validation degradation every epoch
- ✓ Fair model comparison
- ✓ Reliable early stopping
- ✓ Reproducible results

See [VALIDATION_DEGRADATION.md](VALIDATION_DEGRADATION.md) for detailed documentation.

**Test reproducibility:**
```bash
python scripts/test_validation_reproducibility.py \
    --config configs/degradation_modrate_val.json \
    --data-dir ../S1
```

---

## Validation

Always validate your configs before training:

```bash
python scripts/validate_setup.py --config configs/train_config.json
```

This will check:
- ✓ All required fields present
- ✓ Valid parameter types and ranges
- ✓ Degradation config loads correctly
- ✓ Pipeline builds successfully

---

**Last Updated**: November 2025
