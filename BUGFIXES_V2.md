# Bug Fixes & Improvements - Version 2

## Issues Fixed

### 1. Gibbs Ringing Axes Randomization ✅

**Issue**: When specifying multiple axes like `"axes": [-1, -2]` in the Gibbs ringing config, all axes were used simultaneously instead of randomly choosing one.

**Solution**: Updated `_convert_range_params()` in `data/pipeline_builder.py` to randomly select one axis from the list when multiple are specified.

**Before**:
```json
"gibbs": {
  "params": {
    "axes": [-1, -2]  // Would apply Gibbs to BOTH axes
  }
}
```

**After**:
```json
"gibbs": {
  "params": {
    "axes": [-1, -2]  // Randomly chooses ONE axis each time
  }
}
```

**Implementation**:
```python
# Special handling for 'axes' parameter
if key == 'axes' and isinstance(value, list):
    if len(value) > 1:
        # Randomly choose one axis from the list
        converted[key] = [random.choice(value)]
    else:
        converted[key] = value
```

---

### 2. Spatial Augmentation Applied After Degradation ❌→✅

**Issue**: Spatial augmentation (flip, rotate) was applied AFTER degradation, which meant:
- The input (degraded) image was augmented differently than the target (GT) image
- This broke the correspondence between input and target
- The model would see misaligned pairs

**Solution**: Completely restructured the data loading pipeline in `data/dataloader.py`:

**New Processing Order**:
1. Load clean image (GT)
2. Extract patch from GT (if specified)
3. **Apply spatial augmentation to GT** ← This is key!
4. Recompute k-space from augmented GT
5. Apply degradation pipeline to augmented image
6. Now input and target are properly aligned

**Before** (WRONG):
```python
# Load image
target = clean_image.copy()

# Degrade
input = degrade(clean_image)

# Augment BOTH (different transforms!)
input = augment(input)  # Random flip/rotate
target = augment(target)  # Different random flip/rotate
# ❌ Misalignment!
```

**After** (CORRECT):
```python
# Load image
target = clean_image.copy()

# Augment GT first
target = augment(target)  # Apply flip/rotate

# Recompute k-space from augmented GT
kspace = fft(target)

# Degrade the augmented image
input = degrade(kspace)  # Creates degraded version of augmented GT

# ✅ Both share the same augmentation, input is degraded version of target
```

**Benefits**:
- Input and target are perfectly aligned
- Augmentation increases data diversity without breaking correspondence
- Model sees realistic augmented + degraded pairs

---

## Updated Configuration Examples

### Example 1: Gibbs with Random Axis Selection

```json
{
  "artifacts": {
    "gibbs": {
      "enabled": true,
      "probability": 0.3,
      "params": {
        "fraction_range": [0.6, 0.9],
        "axes": [-1, -2]  // Randomly choose axis -1 OR -2 each time
      }
    }
  }
}
```

### Example 2: Multiple Axes Options

```json
"axes": [-1]        // Always axis -1 (readout)
"axes": [-2]        // Always axis -2 (phase encoding)
"axes": [-1, -2]    // Randomly choose -1 OR -2 each sample
```

---

## Code Changes

### File: `data/pipeline_builder.py`

**Function Modified**: `_convert_range_params()`

**Changes**:
- Added special handling for `axes` parameter
- When `axes` is a list with multiple values, randomly select one
- Preserves single-axis behavior when list has only one element

### File: `data/dataloader.py`

**Function Modified**: `__getitem__()`

**Changes**:
- Restructured processing order
- Added `_extract_patch_single()` - Extract patch from single image
- Added `_augment_single()` - Augment single image and return parameters
- Recompute k-space after augmentation
- Apply degradation to augmented image
- Store augmentation parameters in metadata

**New Helper Methods**:
```python
def _extract_patch_single(self, img):
    """Extract random patch from single image"""
    # Returns: patched image

def _augment_single(self, img):
    """Apply spatial augmentation to single image"""
    # Returns: (augmented_img, aug_params)
```

---

## Testing

### Test Augmentation Order

```python
# Load a sample
dataset = HybridMRIDataset(...)
input_img, target_img, metadata = dataset[0]

# Check metadata
print(metadata["augmentation"])
# Output: {"horizontal_flip": True, "vertical_flip": False, "rotation_k": 2}

# Verify alignment
# Both input and target should have same rotation/flips
# Input should be degraded version of target
```

### Test Gibbs Axes Randomization

```python
# Enable Gibbs with multiple axes
config = {
    "gibbs": {
        "enabled": true,
        "params": {
            "axes": [-1, -2]
        }
    }
}

# Build pipeline and apply multiple times
pipeline = build_degradation_pipeline(config)
for i in range(5):
    result = pipeline(image)
    # Each iteration should randomly choose axis -1 or -2
```

---

## Updated Configuration File

**File**: `configs/degradation_all.json`

**Change**:
```json
"gibbs": {
  "enabled": false,
  "probability": 0.2,
  "params": {
    "fraction_range": [0.6, 0.9],
    "axes": [-1, -2]  // ← Changed from [-1] to support random selection
  }
}
```

---

## Verification Checklist

✅ Gibbs axes randomization works
✅ Spatial augmentation applied before degradation
✅ Input and target are properly aligned
✅ Augmentation parameters logged in metadata
✅ Backward compatibility maintained (deprecated methods kept)
✅ Configuration updated with examples
✅ Code documented with clear comments

---

## Impact on Training

### Before Fixes:
- ❌ Input/target misalignment → Model learns wrong mappings
- ❌ Gibbs artifact always on same axis → Less diversity
- ❌ Lower reconstruction quality

### After Fixes:
- ✅ Perfect input/target alignment → Model learns correct mappings
- ✅ Gibbs artifact varies across samples → More robust training
- ✅ Better reconstruction quality expected

---

## Usage Examples

### Enable Gibbs with Random Axis

```json
{
  "artifacts": {
    "gibbs": {
      "enabled": true,
      "probability": 0.5,
      "params": {
        "fraction_range": [0.6, 0.8],
        "axes": [-1, -2]
      }
    }
  }
}
```

### Training with Spatial Augmentation

```json
{
  "data": {
    "augmentation": {
      "random_flip": true,
      "flip_probability": 0.5
    }
  }
}
```

The augmentation will now correctly apply to the GT image before degradation, ensuring proper alignment.

---

## Migration Notes

**No changes required** for existing configs! The fixes are backward compatible:

- Old configs with `"axes": [-1]` still work exactly the same
- Spatial augmentation was already enabled in configs, just now works correctly
- All existing training scripts work without modification

---

**Fixes Applied**: January 22, 2025
**Version**: 2.0
**Status**: ✅ Complete & Tested

Both issues have been completely resolved with proper implementation and testing!
