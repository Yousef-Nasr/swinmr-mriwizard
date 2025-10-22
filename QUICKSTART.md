# Quick Start Guide - SwinMR + MriWizard

Get up and running in 5 minutes!

## 1. Installation (1 minute)

```bash
cd swinmr_mriwizard
pip install -r requirements.txt
```

## 2. Prepare Sample Data (2 minutes)

### Option A: Convert DICOM to NPY

```bash
python scripts/convert_dicom_to_npy.py \
    --input-dir /path/to/your/dicom/files \
    --output-dir ./data/train_npy \
    --recursive
```

### Option B: Use Existing NPY Files

Simply place your `.npy` files in:
- `./data/train_npy/` - Training images
- `./data/val_npy/` - Validation images

**Minimum requirement**: At least 1 file in each directory to test the pipeline.

## 3. Validate Setup (1 minute)

```bash
python scripts/validate_setup.py --config configs/train_config.json
```

You should see:
```
✓ ALL VALIDATION CHECKS PASSED!
```

## 4. Visualize Degradations (Optional, 1 minute)

```bash
python scripts/visualize_degradations.py \
    --degradation-config configs/degradation_minimal.json \
    --image data/train_npy/sample.npy \
    --output preview.png
```

Open `preview.png` to see what degradations look like.

## 5. Start Training (1 minute to start)

```bash
python training/train.py --config configs/train_config.json --gpu 0
```

**Monitor with TensorBoard**:
```bash
# In another terminal
tensorboard --logdir results/swinmr_mriwizard_hybrid/logs
```

Open browser to `http://localhost:6006`

## 6. Quick Test Training (Optional)

For a quick test with minimal settings:

```bash
python training/train.py \
    --config configs/train_config.json \
    --data.degradation_config configs/degradation_minimal.json \
    --training.epochs 5 \
    --data.batch_size 4
```

## Expected Training Progress

You should see output like:

```
Loading config from: configs/train_config.json
Using device: cuda:0
Creating datasets...
Train dataset: 150 samples
Val dataset: 30 samples
Creating model...
Model parameters: 11,735,104
Starting training...

Epoch 0 [0/19] Loss: 0.082345
Epoch 0 [10/19] Loss: 0.071234
...
Validation Epoch 0 - Loss: 0.065432, PSNR: 28.45 dB, SSIM: 0.8234
✓ Saved best checkpoint (SSIM: 0.8234)
```

## Troubleshooting

### "No valid files found"

Check your data directory:
```bash
ls -la data/train_npy/
```

Ensure files have `.npy`, `.dcm`, `.png`, or `.jpg` extensions.

### "CUDA out of memory"

Reduce batch size and patch size:

```bash
python training/train.py \
    --config configs/train_config.json \
    --data.batch_size 2 \
    --data.patch_size 128
```

### "Loss is NaN"

Reduce learning rate:

```bash
python training/train.py \
    --config configs/train_config.json \
    --training.optimizer.lr 1e-4
```

## Next Steps

1. **Read the full README**: `README.md`
2. **Customize degradations**: Edit `configs/degradation_all.json`
3. **Adjust training params**: Edit `configs/train_config.json`
4. **Monitor training**: Watch TensorBoard
5. **Evaluate model**: `python evaluation/test.py --config configs/train_config.json --checkpoint results/swinmr_mriwizard_hybrid/checkpoints/best.pth`

## Useful Commands

### Resume Training
```bash
python training/train.py \
    --config configs/train_config.json \
    --resume results/swinmr_mriwizard_hybrid/checkpoints/epoch_50.pth
```

### Test Model
```bash
python evaluation/test.py \
    --config configs/train_config.json \
    --checkpoint results/swinmr_mriwizard_hybrid/checkpoints/best.pth \
    --save-images
```

### Override Config from Command Line
```bash
python training/train.py \
    --config configs/train_config.json \
    --data.batch_size 16 \
    --training.epochs 100 \
    --training.optimizer.lr 1e-4
```

---

**Need Help?**
- Run validation: `python scripts/validate_setup.py --config configs/train_config.json`
- Check main README: `README.md`
- Review config docs: `configs/README.md`

**Happy Training! 🚀**
