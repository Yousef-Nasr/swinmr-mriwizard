# Augmentation Configuration Guide

## Overview

Spatial augmentation is now fully configurable through the **degradation config file**. This allows you to control exactly which augmentations are applied, their probabilities, and parameters.

## Why Augmentation in Degradation Config?

Augmentation is applied **BEFORE** degradation to the ground truth (GT) image. This ensures:
- Input and target share the same augmentation
- Degradation is applied to the augmented GT
- Perfect alignment between input and target

Since augmentation is part of the data preparation pipeline (before degradation), it logically belongs in the degradation config.

---

## Configuration Structure

Add an `"augmentation"` section to your degradation config:

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {
      "enabled": true,
      "probability": 0.5
    },
    "vertical_flip": {
      "enabled": true,
      "probability": 0.5
    },
    "rotation_90": {
      "enabled": true,
      "probability": 0.5,
      "angles": [90, 180, 270]
    }
  }
}
```

---

## Parameters

### Global Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | bool | Enable/disable all augmentation | `true` |

### Horizontal Flip

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | bool | Enable horizontal flip | `true` |
| `probability` | float | Probability of applying (0-1) | `0.5` |

**Example**:
```json
"horizontal_flip": {
  "enabled": true,
  "probability": 0.3  // 30% chance of horizontal flip
}
```

### Vertical Flip

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | bool | Enable vertical flip | `true` |
| `probability` | float | Probability of applying (0-1) | `0.5` |

**Example**:
```json
"vertical_flip": {
  "enabled": false  // Disable vertical flip
}
```

### 90-Degree Rotation

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | bool | Enable rotation | `true` |
| `probability` | float | Probability of applying (0-1) | `0.5` |
| `angles` | array[int] | Allowed rotation angles (in degrees) | `[90, 180, 270]` |

**Example**:
```json
"rotation_90": {
  "enabled": true,
  "probability": 0.7,  // 70% chance of rotation
  "angles": [90, 270]  // Only 90° or 270° (no 180°)
}
```

---

## Configuration Examples

### Example 1: Full Augmentation (Default)

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {
      "enabled": true,
      "probability": 0.5
    },
    "vertical_flip": {
      "enabled": true,
      "probability": 0.5
    },
    "rotation_90": {
      "enabled": true,
      "probability": 0.5,
      "angles": [90, 180, 270]
    }
  }
}
```

**Result**: All augmentations enabled with 50% probability each.

### Example 2: Only Horizontal Flip

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {
      "enabled": true,
      "probability": 0.8
    },
    "vertical_flip": {
      "enabled": false
    },
    "rotation_90": {
      "enabled": false
    }
  }
}
```

**Result**: Only horizontal flips, with 80% probability.

### Example 3: High Augmentation for Small Datasets

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {
      "enabled": true,
      "probability": 0.8  // High probability
    },
    "vertical_flip": {
      "enabled": true,
      "probability": 0.8  // High probability
    },
    "rotation_90": {
      "enabled": true,
      "probability": 0.8,  // High probability
      "angles": [90, 180, 270]
    }
  }
}
```

**Result**: Aggressive augmentation for maximum data diversity.

### Example 4: No Augmentation

```json
{
  "augmentation": {
    "enabled": false
  }
}
```

**Result**: No augmentation applied.

### Example 5: Only 180° Rotation

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {
      "enabled": false
    },
    "vertical_flip": {
      "enabled": false
    },
    "rotation_90": {
      "enabled": true,
      "probability": 1.0,  // Always rotate
      "angles": [180]      // Only 180°
    }
  }
}
```

**Result**: Every image is rotated 180°.

---

## Processing Order

The complete data pipeline works as follows:

```
1. Load GT image from disk
   ↓
2. Extract patch (if patch_size specified)
   ↓
3. Apply augmentation (based on config) ← AUGMENTATION HERE
   - Horizontal flip (maybe)
   - Vertical flip (maybe)
   - Rotation (maybe)
   ↓
4. Recompute k-space from augmented GT
   ↓
5. Apply degradation pipeline
   - Noise
   - Undersampling
   - Artifacts
   ↓
6. Reconstruct degraded image
   ↓
7. Return (degraded_input, augmented_GT_target)
```

---

## Metadata Logging

Augmentation parameters are logged in the sample metadata:

```python
input, target, metadata = dataset[0]

# Check what augmentations were applied
print(metadata["augmentation"])
# Output:
# {
#   "horizontal_flip": True,
#   "vertical_flip": False,
#   "rotation_k": 2,
#   "rotation_angle": 180
# }
```

This allows you to:
- Track which augmentations were applied
- Debug augmentation issues
- Analyze augmentation distribution

---

## Best Practices

### 1. Start Conservative

Begin with moderate probabilities:
```json
"probability": 0.5
```

### 2. Adjust Based on Dataset Size

**Small dataset (< 1000 images)**:
```json
"probability": 0.8  // More augmentation
```

**Large dataset (> 10,000 images)**:
```json
"probability": 0.3  // Less augmentation
```

### 3. Medical Image Considerations

For medical images:
- **Horizontal flip**: Usually safe
- **Vertical flip**: Usually safe
- **Rotation**: May affect anatomical orientation
  - Consider: `"angles": [180]` only
  - Or disable if orientation is critical

### 4. Test Augmentation First

Use the visualization script to verify augmentations:

```bash
python scripts/visualize_degradations.py \
    --degradation-config configs/degradation_all.json \
    --image sample.dcm \
    --num-samples 5
```

Look at the samples to ensure augmentations look reasonable.

---

## Disabling Augmentation

### Option 1: Disable in Config (Recommended)

```json
{
  "augmentation": {
    "enabled": false
  }
}
```

### Option 2: Disable Specific Augmentations

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {"enabled": false},
    "vertical_flip": {"enabled": false},
    "rotation_90": {"enabled": false}
  }
}
```

### Option 3: Set Probability to 0

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {"probability": 0.0},
    "vertical_flip": {"probability": 0.0},
    "rotation_90": {"probability": 0.0}
  }
}
```

---

## Backward Compatibility

Old code using `use_augmentation=True/False` parameter still works:

```python
# Old way (still works)
dataset = HybridMRIDataset(
    data_dir="./data",
    degradation_config="config.json",
    use_augmentation=False  # Overrides config if specified
)

# New way (recommended)
dataset = HybridMRIDataset(
    data_dir="./data",
    degradation_config="config.json"  # Reads augmentation from config
)
```

**Priority**: Config file settings override `use_augmentation` parameter.

---

## Common Configurations

### For Brain MRI

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {
      "enabled": true,
      "probability": 0.5
    },
    "vertical_flip": {
      "enabled": true,
      "probability": 0.5
    },
    "rotation_90": {
      "enabled": true,
      "probability": 0.3,  // Lower probability
      "angles": [180]      // Only 180° to preserve anatomy
    }
  }
}
```

### For Knee/Joint MRI

```json
{
  "augmentation": {
    "enabled": true,
    "horizontal_flip": {
      "enabled": true,
      "probability": 0.6
    },
    "vertical_flip": {
      "enabled": false  // May flip anatomy incorrectly
    },
    "rotation_90": {
      "enabled": false  // Joints have specific orientations
    }
  }
}
```

### For Testing/Debugging

```json
{
  "augmentation": {
    "enabled": false  // No augmentation for reproducibility
  }
}
```

---

## Summary

✅ **Fully configurable** - Control every aspect via JSON
✅ **Applied before degradation** - Ensures proper alignment
✅ **Probability-based** - Fine-grained control
✅ **Angle selection** - Choose which rotations to allow
✅ **Logged in metadata** - Track what was applied
✅ **Backward compatible** - Old code still works

---

**Last Updated**: January 22, 2025
**Status**: Production-Ready ✅
