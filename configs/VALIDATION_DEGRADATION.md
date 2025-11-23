# Validation Degradation Configuration

## Overview

This document explains how to configure **reproducible validation degradation** to ensure consistent metrics across training epochs.

## Problem

During training, validation metrics can fluctuate due to random degradation being applied differently each epoch. This makes it difficult to:
- Track true model improvement
- Compare models fairly
- Detect overfitting reliably

## Solution

Use **separate degradation configs** for training and validation:
- **Training**: Random degradation (seed=null, deterministic=false)
- **Validation**: Fixed degradation (seed=42, deterministic=true)

## Configuration

### 1. Training Config (`train_config.json`)

Add a separate validation degradation config:

```json
{
  "data": {
    "degradation_config": "./configs/degradation_modrate.json",
    "val_degradation_config": "./configs/degradation_modrate_val.json"
  }
}
```

### 2. Validation Degradation Config

Create a validation-specific config with these key settings:

```json
{
  "augmentation": {
    "enabled": false
  },
  "execution": {
    "seed": 42,
    "deterministic": true
  }
}
```

**Key differences from training config:**
- `augmentation.enabled`: false (no random flips/rotations)
- `execution.seed`: 42 (or any fixed number)
- `execution.deterministic`: true (enables reproducible mode)

## Available Validation Configs

| Config | Description |
|--------|-------------|
| `degradation_modrate_val.json` | Moderate degradation with fixed seed |
| `degradation_all_val.json` | Comprehensive degradation with fixed seed |

## How It Works

1. **Initialization**: When `deterministic=true` and `seed` is set, the pipeline sets global random seeds
2. **Per-Sample Seeding**: Each validation sample gets a deterministic seed: `base_seed + sample_idx`
3. **Reproducibility**: Same sample always gets same degradation across epochs

## Example Usage

```bash
# Training with separate validation degradation
python training/train.py --config configs/train_config.json
```

The validation will use fixed degradation while training uses random degradation.

## Benefits

✅ **Consistent Metrics**: Same validation degradation every epoch  
✅ **Fair Comparison**: Compare models on identical validation conditions  
✅ **Reliable Early Stopping**: Detect true overfitting, not random fluctuation  
✅ **Reproducible Results**: Same validation metrics across runs  

## Creating Custom Validation Configs

To create your own validation config:

1. Copy an existing training degradation config
2. Set `augmentation.enabled` to `false`
3. Set `execution.seed` to a fixed value (e.g., 42)
4. Set `execution.deterministic` to `true`
5. Reference it in `train_config.json` as `val_degradation_config`

## Notes

- Training degradation should remain random for better generalization
- Validation degradation should be fixed for consistent evaluation
- The same validation images will get the same degradation every epoch
- Different validation images get different (but consistent) degradation

