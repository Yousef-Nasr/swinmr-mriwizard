# Validation Reproducibility Implementation

## Summary

Implemented a configuration system to ensure **reproducible validation degradation** across training epochs. This allows for consistent validation metrics and fair model comparison.

## Problem Solved

Previously, validation data was degraded randomly each epoch, causing:
- ❌ Fluctuating validation metrics due to random degradation
- ❌ Difficulty tracking true model improvement
- ❌ Unreliable early stopping
- ❌ Non-reproducible validation results

## Solution

Created a system to use **separate degradation configs** for training vs validation:
- **Training**: Random degradation (for better generalization)
- **Validation**: Fixed degradation (for consistent evaluation)

## Changes Made

### 1. New Configuration Files

#### `configs/degradation_modrate_val.json`
- Moderate degradation with fixed seed (42)
- Augmentation disabled
- Deterministic mode enabled

#### `configs/degradation_all_val.json`
- Comprehensive degradation with fixed seed (42)
- Augmentation disabled
- Deterministic mode enabled

### 2. Updated Training Config

**File**: `configs/train_config.json`

Added new parameter:
```json
{
  "data": {
    "val_degradation_config": "./degradation_modrate_val.json"
  }
}
```

### 3. Updated Training Script

**File**: `training/train.py`

Modified dataset creation to use separate validation config:
```python
# Use separate validation degradation config if provided
val_degradation_config_path = config['data'].get('val_degradation_config', degradation_config_path)
logger.info(f"Using validation degradation config: {val_degradation_config_path}")

val_dataset = HybridMRIDataset(
    data_dir=config['data']['val_dir'],
    degradation_config=val_degradation_config_path,  # Uses separate config
    ...
)
```

### 4. Updated Pipeline Builder

**File**: `data/pipeline_builder.py`

Added seed setting for deterministic mode:
```python
# Set random seed if specified for deterministic validation
execution_config = degradation_config.get("execution", {})
seed = execution_config.get("seed", None)
deterministic = execution_config.get("deterministic", False)

if seed is not None and deterministic:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 5. Updated Dataloader

**File**: `data/dataloader.py`

Added per-sample deterministic seeding:
```python
# Store deterministic settings
self.deterministic = execution_config.get("deterministic", False)
self.base_seed = execution_config.get("seed", None)

# In __getitem__:
if self.deterministic and self.base_seed is not None:
    # Use idx to ensure each sample gets consistent but different degradation
    sample_seed = self.base_seed + idx
    np.random.seed(sample_seed)
    random.seed(sample_seed)
    torch.manual_seed(sample_seed)
```

### 6. Documentation

#### `configs/VALIDATION_DEGRADATION.md`
- Complete guide on reproducible validation
- Configuration examples
- Benefits and use cases

#### `configs/README.md`
- Added `val_degradation_config` parameter documentation
- Added reproducible validation section
- Updated best practices

### 7. Test Script

**File**: `scripts/test_validation_reproducibility.py`

Utility to verify degradation reproducibility:
```bash
python scripts/test_validation_reproducibility.py \
    --config configs/degradation_modrate_val.json \
    --data-dir ../S1 \
    --num-samples 5
```

## How It Works

### Training Mode (Random Degradation)
1. Uses `degradation_config` (e.g., `degradation_modrate.json`)
2. `deterministic=false`, `seed=null`
3. Different degradation every epoch
4. Better generalization

### Validation Mode (Fixed Degradation)
1. Uses `val_degradation_config` (e.g., `degradation_modrate_val.json`)
2. `deterministic=true`, `seed=42`
3. Each sample gets: `seed = base_seed + sample_idx`
4. Same degradation every epoch for each sample
5. Consistent metrics

## Usage

### Basic Training
```bash
python training/train.py --config configs/train_config.json
```

The script automatically uses:
- `degradation_modrate.json` for training (random)
- `degradation_modrate_val.json` for validation (fixed)

### Test Reproducibility
```bash
python scripts/test_validation_reproducibility.py \
    --config configs/degradation_modrate_val.json \
    --data-dir ../S1
```

Expected output:
```
Sample 0: ✓ PASS (max diff: 0.00e+00)
Sample 1: ✓ PASS (max diff: 0.00e+00)
Sample 2: ✓ PASS (max diff: 0.00e+00)
...
✓ SUCCESS: All samples are reproducible!
```

## Configuration Structure

### Validation Degradation Config Template
```json
{
  "name": "your_degradation_validation",
  "noise": { ... },
  "undersampling": { ... },
  "artifacts": { ... },
  
  "augmentation": {
    "enabled": false  // CRITICAL: Disable for validation
  },
  
  "execution": {
    "apply_order": ["noise", "undersampling", "artifacts"],
    "seed": 42,           // CRITICAL: Fixed seed
    "deterministic": true // CRITICAL: Enable deterministic mode
  }
}
```

## Benefits

✅ **Consistent Metrics**: Same validation degradation every epoch  
✅ **Fair Comparison**: Compare models on identical validation conditions  
✅ **Reliable Early Stopping**: Detect true overfitting, not random fluctuation  
✅ **Reproducible Results**: Same validation metrics across runs  
✅ **Better Training**: Random training degradation for generalization  
✅ **Better Evaluation**: Fixed validation degradation for consistency  

## Backward Compatibility

- If `val_degradation_config` is not specified, falls back to `degradation_config`
- Existing configs continue to work without changes
- Opt-in feature for reproducible validation

## Files Modified

1. `configs/train_config.json` - Added `val_degradation_config`
2. `training/train.py` - Use separate validation config
3. `data/pipeline_builder.py` - Seed setting for deterministic mode
4. `data/dataloader.py` - Per-sample deterministic seeding
5. `configs/README.md` - Documentation updates

## Files Created

1. `configs/degradation_modrate_val.json` - Validation config
2. `configs/degradation_all_val.json` - Validation config
3. `configs/VALIDATION_DEGRADATION.md` - Detailed guide
4. `scripts/test_validation_reproducibility.py` - Test utility
5. `VALIDATION_REPRODUCIBILITY_CHANGES.md` - This file

## Testing

Run the reproducibility test to verify:
```bash
python scripts/test_validation_reproducibility.py \
    --config configs/degradation_modrate_val.json \
    --data-dir ../S1 \
    --num-samples 10
```

All samples should show `✓ PASS` with `max diff: 0.00e+00`.

## Notes

- Training degradation remains random for better generalization
- Validation degradation is fixed for consistent evaluation
- Each validation sample gets a unique but deterministic seed
- Seeds are reproducible across runs and epochs
- No performance impact on training speed

---

**Implementation Date**: November 2025  
**Status**: ✅ Complete and Tested

