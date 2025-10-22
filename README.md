# SwinMR + MriWizard: Vendor-Agnostic MRI Reconstruction

A production-ready deep learning system for MRI acceleration and enhancement that works directly with clinical DICOM images. This project integrates the SwinMR architecture (Swin Transformer for Fast MRI) with the MriWizard k-space simulation framework to enable training on realistic degraded MRI data.

## 🎯 Key Features

- **Vendor-Agnostic**: Works directly with DICOM files from any scanner manufacturer
- **Hybrid Data Loading**: Supports NPY (fast path), DICOM, PNG, and JPG formats with automatic caching
- **Realistic K-space Simulation**: 5 undersampling patterns + Gaussian noise + optional artifacts
- **On-the-fly Degradation**: Different degradations each epoch for robust training
- **Modular Configuration**: Separate, composable configs for training and degradation
- **Production-Ready**: Comprehensive validation, logging, checkpointing, and early stopping

## 📊 Supported Degradation Patterns

### Undersampling Patterns
1. **Random Undersampling**: Probabilistic k-space line retention
2. **Uniform Undersampling**: Every R-th line with center preservation
3. **Partial Fourier**: Asymmetric k-space acquisition
4. **K-max Undersampling**: Central box cropping (all directions)
5. **Elliptical Undersampling**: Radial masking

### Additional Artifacts (Optional)
- **Gaussian Noise**: Complex k-space noise
- **Motion Artifacts**: Rigid body motion simulation
- **Ghosting**: N/2 ghosting artifacts
- **Spike Noise**: RF interference
- **Gibbs Ringing**: High-frequency truncation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd swinmr_mriwizard

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

#### Option A: Use NPY Files (Recommended for Speed)

Convert DICOM to NPY format:

```bash
python scripts/convert_dicom_to_npy.py \
    --input-dir /path/to/dicom/files \
    --output-dir ./data/train_npy \
    --recursive
```

#### Option B: Use DICOM Directly

The dataloader will automatically convert and cache DICOM files on first load.

Organize your data:
```
data/
├── train_npy/      # Training images
│   ├── image_001.npy
│   ├── image_002.npy
│   └── ...
├── val_npy/        # Validation images
│   └── ...
└── test_npy/       # Test images
    └── ...
```

### 3. Validate Setup

Before training, validate that everything is configured correctly:

```bash
python scripts/validate_setup.py --config configs/train_config.json
```

This checks:
- ✓ Configuration files load correctly
- ✓ Degradation pipeline builds successfully
- ✓ Dataloader works with your data
- ✓ Model initializes properly
- ✓ Forward pass and loss computation work

### 4. Visualize Degradations

See what degradations look like on your data:

```bash
python scripts/visualize_degradations.py \
    --degradation-config configs/degradation_all.json \
    --image data/train_npy/sample.npy \
    --output degradation_preview.png \
    --num-samples 3
```

### 5. Train

Start training:

```bash
python training/train.py --config configs/train_config.json --gpu 0
```

Monitor with TensorBoard:

```bash
tensorboard --logdir results/swinmr_mriwizard_hybrid/logs
```

### 6. Evaluate

Test your trained model:

```bash
python evaluation/test.py \
    --config configs/train_config.json \
    --checkpoint results/swinmr_mriwizard_hybrid/checkpoints/best.pth \
    --save-images
```

## ⚙️ Configuration

### Main Training Config (`configs/train_config.json`)

```json
{
  "experiment_name": "swinmr_mriwizard_hybrid",

  "data": {
    "train_dir": "./data/train_npy",
    "val_dir": "./data/val_npy",
    "degradation_config": "./configs/degradation_all.json",  // Reference to degradation config
    "batch_size": 8,
    "patch_size": 256,
    "use_patches": true
  },

  "model": {
    "net_type": "swinir",
    "window_size": 8,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6]
  },

  "training": {
    "epochs": 100,
    "loss": {
      "spatial": {"type": "charbonnier", "weight": 1.0},
      "frequency": {"type": "fft_mse", "weight": 0.1},
      "perceptual": {"type": "vgg19", "weight": 0.0025}
    },
    "optimizer": {
      "type": "adam",
      "lr": 2e-4,
      "scheduler": {
        "type": "multistep",
        "milestones": [25000, 50000, 75000],
        "gamma": 0.5
      }
    },
    "early_stopping": {
      "enabled": true,
      "patience": 10,
      "metric": "val_ssim"
    }
  }
}
```

### Degradation Config (`configs/degradation_all.json`)

```json
{
  "name": "comprehensive_degradation",

  "noise": {
    "enabled": true,
    "std_range": [0.005, 0.05],  // Random sampling from this range
    "relative": true,
    "reference": "std"
  },

  "undersampling": {
    "strategy": "one_of",  // Randomly choose ONE pattern per sample
    "patterns": [
      {
        "name": "random",
        "enabled": true,
        "weight": 1.0,  // Sampling weight
        "params": {
          "prob_range": [0.3, 0.7],
          "center_fraction": 0.08
        }
      },
      {
        "name": "uniform",
        "enabled": true,
        "weight": 1.0,
        "params": {
          "R_range": [2, 8],
          "center_fraction": 0.08
        }
      }
      // ... more patterns
    ]
  },

  "artifacts": {
    "motion": {
      "enabled": false,  // Disable optional artifacts
      "probability": 0.3,
      "params": { ... }
    }
  }
}
```

### Customizing Degradations

Create your own degradation config:

```bash
cp configs/degradation_all.json configs/my_degradation.json
# Edit my_degradation.json to your needs
```

Then reference it in your training config:

```json
{
  "data": {
    "degradation_config": "./configs/my_degradation.json"
  }
}
```

**Common Customizations:**

- **Testing Pipeline**: Use `degradation_minimal.json` (single undersampling pattern)
- **Light Degradation**: Reduce `std_range` and use higher `prob_range`/`R_range`
- **Heavy Degradation**: Increase noise, enable artifacts
- **Specific Pattern**: Disable all but one undersampling pattern

## 📁 Project Structure

```
swinmr_mriwizard/
├── configs/                      # Configuration files
│   ├── train_config.json        # Main training config
│   ├── degradation_all.json     # Comprehensive degradation
│   ├── degradation_minimal.json # Minimal (for testing)
│   └── README.md                # Config documentation
│
├── data/                         # Data loading
│   ├── dataloader.py            # Hybrid MRI dataset
│   └── pipeline_builder.py      # Build MriWizard pipeline from config
│
├── models/                       # SwinMR architecture
│   ├── model_swinmr.py          # Main model class
│   ├── network_swinmr.py        # Swin Transformer network
│   ├── basicblock.py            # Building blocks
│   └── loss.py                  # Loss functions
│
├── training/                     # Training scripts
│   └── train.py                 # Main training script
│
├── evaluation/                   # Evaluation
│   └── test.py                  # Model testing
│
├── utils/                        # Utilities
│   ├── config_loader.py         # Config loading & validation
│   ├── checkpoint.py            # Checkpointing
│   ├── logger.py                # Logging setup
│   └── metrics.py               # PSNR, SSIM, LPIPS
│
├── MriWizard/               # MriWizard library
│   ├── core/                    # Core functionality
│   ├── io/                      # DICOM/Image loaders
│   ├── degrade/                 # Degradation transforms
│   └── reconstruct/             # FFT reconstruction
│
├── scripts/                      # Helper scripts
│   ├── validate_setup.py        # Pre-training validation
│   ├── visualize_degradations.py # Visualize degradations
│   └── convert_dicom_to_npy.py  # Batch DICOM conversion
│
├── results/                      # Output directory
│   └── <experiment_name>/
│       ├── checkpoints/         # Model checkpoints
│       ├── logs/                # TensorBoard logs
│       └── samples/             # Sample outputs
│
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## 🎓 Training Tips

### 1. Start Small

Use `degradation_minimal.json` to quickly test your pipeline:

```bash
python training/train.py \
    --config configs/train_config.json \
    --data.degradation_config configs/degradation_minimal.json \
    --training.epochs 5
```

### 2. Monitor Training

Watch for:
- **PSNR**: Should increase (target >35 dB for diagnostic quality)
- **SSIM**: Should increase (target >0.90)
- **Loss**: Should decrease steadily

### 3. Adjust Learning Rate

If training is unstable:
```json
"optimizer": {
  "lr": 1e-4  // Reduce from 2e-4
}
```

### 4. Use Early Stopping

Prevent overfitting:
```json
"early_stopping": {
  "enabled": true,
  "patience": 10,
  "metric": "val_ssim"
}
```

### 5. Patch Size

- **Larger patches (320×320)**: Better context, more memory
- **Smaller patches (128×128)**: Faster training, less memory
- **Full images (null)**: Best for validation/testing

## 📊 Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-wise accuracy (higher is better)
  - >30 dB: Acceptable
  - >35 dB: Good
  - >40 dB: Excellent

- **SSIM (Structural Similarity Index)**: Measures perceptual quality (higher is better)
  - >0.85: Acceptable
  - >0.90: Good
  - >0.95: Excellent

- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity (lower is better)

## 🔧 Troubleshooting

### Issue: "No valid files found"

**Solution**: Check your data directory path and ensure files have correct extensions (.npy, .dcm, .png, .jpg)

```bash
ls data/train_npy/*.npy  # Should list files
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size: `"batch_size": 4`
2. Reduce patch size: `"patch_size": 128`
3. Use gradient accumulation
4. Use mixed precision training

### Issue: "Loss is NaN"

**Solutions**:
1. Reduce learning rate: `"lr": 1e-4`
2. Enable gradient clipping (already enabled by default)
3. Check your data for NaN values

### Issue: "Training is very slow"

**Solutions**:
1. Pre-convert DICOM to NPY: `scripts/convert_dicom_to_npy.py`
2. Increase `num_workers`: `"num_workers": 8`
3. Use smaller `patch_size` during training

### Issue: "Degradation visualization looks wrong"

**Solution**: Check your degradation config parameters. Use `degradation_minimal.json` as a starting point.

## 🚀 Advanced Usage

### Command-Line Config Overrides

Override config parameters from command line:

```bash
python training/train.py \
    --config configs/train_config.json \
    --data.batch_size 16 \
    --training.epochs 50 \
    --training.optimizer.lr 1e-4
```

### Resume Training

Resume from a checkpoint:

```bash
python training/train.py \
    --config configs/train_config.json \
    --resume results/swinmr_mriwizard_hybrid/checkpoints/epoch_50.pth
```

### Multi-GPU Training

Edit config (not fully implemented yet, infrastructure in place):

```json
"system": {
  "gpu_ids": [0, 1, 2, 3],
  "distributed": true
}
```

## 📚 References

1. **SwinMR**: Huang, J., Fang, Y., Wu, Y., et al. (2022). "Swin transformer for fast MRI." *Neurocomputing*, 493, 281-304.

2. **AIRS Medical**: Vendor-agnostic MRI reconstruction methodology

3. **MriWizard**: Custom k-space simulation framework for DICOM-based training

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{huang2022swin,
  title={Swin transformer for fast MRI},
  author={Huang, Jiahao and Fang, Yinzhe and Wu, Ying and Wu, Huanjun and Gao, Zhifan and Li, Yang and Del Ser, Javier and Xia, Jun and Yang, Guang},
  journal={Neurocomputing},
  volume={493},
  pages={281--304},
  year={2022},
  publisher={Elsevier}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional undersampling patterns
- [ ] Multi-coil support (parallel imaging)
- [ ] 3D volume processing
- [ ] Contextual pathway integration (scan parameters)
- [ ] Real-time inference optimization

## 📄 License

See LICENSE file for details.

## 💬 Support

For questions or issues:

1. Run validation: `python scripts/validate_setup.py --config configs/train_config.json`
2. Check configuration documentation: `configs/README.md`
3. Review troubleshooting section above

---

**Version**: 1.0
**Last Updated**: January 2025
**Status**: Production-Ready ✅
