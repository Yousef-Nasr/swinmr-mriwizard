#!/usr/bin/env python3
"""
Validate Setup Before Training

Checks:
- Configuration files load correctly
- Degradation pipeline builds successfully
- Dataloader works with sample data
- Model initializes and can perform forward pass
- Loss computation works
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader import HybridMRIDataset
from data.pipeline_builder import build_degradation_pipeline, print_pipeline_info
from models.model_swinmr import ModelSwinMR
from utils.config_loader import load_config, load_degradation_config, validate_config
import torch


def validate():
    parser = argparse.ArgumentParser(description='Validate setup before training')
    parser.add_argument('--config', type=str, required=True, help='Path to training config')
    args = parser.parse_args()

    print("="*70)
    print(" "*20 + "SwinMR + MriWizard Setup Validation")
    print("="*70)

    # 1. Load configurations
    print("\n[1/7] Loading configurations...")
    try:
        config = load_config(args.config)
        print(f"  ✓ Main config loaded: {args.config}")
        print(f"  ✓ Experiment: {config['experiment_name']}")
    except Exception as e:
        print(f"  ✗ Error loading main config: {e}")
        return False

    try:
        degradation_config_path = config['data']['degradation_config']
        deg_config = load_degradation_config(degradation_config_path)
        print(f"  ✓ Degradation config loaded: {degradation_config_path}")
        print(f"  ✓ Degradation name: {deg_config.get('name', 'unnamed')}")
    except Exception as e:
        print(f"  ✗ Error loading degradation config: {e}")
        return False

    # 2. Validate configurations
    print("\n[2/7] Validating configurations...")
    try:
        validate_config(config, 'training')
        print("  ✓ Training config is valid")
    except Exception as e:
        print(f"  ⚠ Warning in training config: {e}")

    try:
        validate_config(deg_config, 'degradation')
        print("  ✓ Degradation config is valid")
    except Exception as e:
        print(f"  ✗ Error in degradation config: {e}")
        return False

    # 3. Test degradation pipeline
    print("\n[3/7] Building degradation pipeline...")
    try:
        pipeline = build_degradation_pipeline(deg_config)
        print(f"  ✓ Pipeline built successfully")
        print_pipeline_info(pipeline)
    except Exception as e:
        print(f"  ✗ Error building pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Test dataloader
    print("\n[4/7] Testing dataloader...")
    try:
        # Check if train directory has files
        train_dir = Path(config['data']['train_dir'])
        if not train_dir.exists():
            print(f"  ⚠ Warning: Train directory does not exist: {train_dir}")
            print(f"  Skipping dataloader test")
        else:
            dataset = HybridMRIDataset(
                data_dir=str(train_dir),
                degradation_config=degradation_config_path,
                patch_size=config['data'].get('patch_size'),
                cache_dir='./cache_test',
                use_augmentation=False,
                return_metadata=True
            )
            print(f"  ✓ Dataloader created: {len(dataset)} files found")

            if len(dataset) > 0:
                # Load one sample
                try:
                    input_img, target_img, metadata = dataset[0]
                    print(f"  ✓ Sample loaded successfully")
                    print(f"    - Input shape: {input_img.shape}")
                    print(f"    - Target shape: {target_img.shape}")
                    print(f"    - Metadata keys: {list(metadata.keys())}")
                    print(f"    - Applied degradations: {len(metadata.get('applied', []))}")
                except Exception as e:
                    print(f"  ✗ Error loading sample: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print(f"  ⚠ Warning: No files found in {train_dir}")
                print(f"  Continuing validation...")

    except Exception as e:
        print(f"  ✗ Error creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Test model creation
    print("\n[5/7] Creating model...")
    try:
        # Prepare model options (SwinMR model expects specific structure)
        # The model config needs to be under 'netG' key
        model_opt = {
            'netG': config['model'].copy(),  # Network architecture config
            'path': {
                'models': config['paths']['checkpoints'],
                'log': config['paths']['logs'],
                'samples': config['paths']['samples']
            },
            'gpu_ids': config['system']['gpu_ids'],
            'is_train': False,  # Validation mode
            'dist': config['system'].get('distributed', False),  # Distributed training flag
            'train': {
                'freeze_patch_embedding': config.get('training', {}).get('freeze_patch_embedding', False),
                'E_decay': 0  # No EMA during validation
            },
            'datasets': {
                'train': {
                    'batch_size': config.get('data', {}).get('batch_size', 1)
                }
            }
        }

        model = ModelSwinMR(model_opt)
        # Count parameters from the network (model.netG is the actual nn.Module)
        num_params = sum(p.numel() for p in model.netG.parameters())
        print(f"  ✓ Model created successfully")
        print(f"    - Total parameters: {num_params:,}")
        print(f"    - Model type: {config['model'].get('net_type', 'swinir')}")
    except Exception as e:
        print(f"  ✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Test forward pass
    print("\n[6/7] Testing model forward pass...")
    try:
        # Create dummy data (low quality input and high quality target)
        dummy_L = torch.randn(1, 1, 256, 256)  # Low quality (degraded)
        dummy_H = torch.randn(1, 1, 256, 256)  # High quality (ground truth)

        # Feed data to model (SwinMR uses feed_data/test pattern)
        data = {'L': dummy_L, 'H': dummy_H}
        model.feed_data(data, need_H=True)

        # Run inference
        model.netG.eval()
        with torch.no_grad():
            model.test()

        # Get output
        output = model.E  # model.E is the enhanced/reconstructed output

        print(f"  ✓ Forward pass successful")
        print(f"    - Input shape: {dummy_L.shape}")
        print(f"    - Output shape: {output.shape}")
    except Exception as e:
        print(f"  ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 7. Test loss computation
    print("\n[7/7] Testing loss computation...")
    try:
        # Add all required loss parameters to training config
        model_opt['train']['G_lossfn_type'] = 'charbonnier'  # Use Charbonnier loss
        model_opt['train']['G_lossfn_weight'] = 1.0  # Loss weight
        model_opt['train']['G_charbonnier_eps'] = 1e-3  # Charbonnier epsilon
        model_opt['train']['alpha'] = 1.0  # Spatial loss weight
        model_opt['train']['beta'] = 0.1   # Frequency loss weight
        model_opt['train']['gamma'] = 0.0025  # Perceptual loss weight

        # Re-assign updated training options
        model.opt_train = model_opt['train']

        # Initialize loss functions (needed for total_loss)
        model.define_loss()

        # Compute total loss
        with torch.no_grad():
            loss = model.total_loss()

        print(f"  ✓ Loss computation successful")
        print(f"    - Total loss: {loss.item():.6f}")
        print(f"    - Image loss: {model.loss_image.item():.6f}")
        print(f"    - Frequency loss: {model.loss_freq.item():.6f}")
        print(f"    - Perceptual loss: {model.loss_perc.item():.6f}")
    except Exception as e:
        print(f"  ✗ Error computing loss: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 8. Test metrics computation
    print("\n[8/9] Testing metrics computation...")
    try:
        from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips

        # Use the output and target from forward pass
        # Convert to numpy for metrics computation
        pred_np = output.detach().cpu()
        target_np = dummy_H.detach().cpu()

        # Test PSNR
        psnr_value = calculate_psnr(pred_np, target_np, data_range=1.0)
        print(f"  ✓ PSNR computed: {psnr_value:.2f} dB")

        # Test SSIM
        ssim_value = calculate_ssim(pred_np, target_np, data_range=1.0)
        print(f"  ✓ SSIM computed: {ssim_value:.4f}")

        # Test LPIPS (may fail if not installed)
        try:
            lpips_value = calculate_lpips(pred_np, target_np)
            print(f"  ✓ LPIPS computed: {lpips_value:.4f}")
        except ImportError:
            print(f"  ⚠ LPIPS skipped (package not installed - run: pip install lpips)")
        except Exception as e:
            print(f"  ⚠ LPIPS error: {e}")

        print(f"  ✓ Metrics computation successful")
    except Exception as e:
        print(f"  ✗ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 9. Summary
    print("\n[9/9] Validation Summary")
    print("="*70)
    print("\n  Configuration:")
    print(f"    - Experiment: {config['experiment_name']}")
    print(f"    - Model: {config['model']['net_type']}")
    print(f"    - Parameters: {num_params:,}")
    print(f"    - Input channels: {config['model']['in_chans']}")
    print(f"    - Output channels: {config['model']['out_chans']}")
    print(f"    - Window size: {config['model']['window_size']}")
    print(f"    - Embed dim: {config['model']['embed_dim']}")

    print("\n  Dataset:")
    print(f"    - Training dir: {config['data']['train_dir']}")
    print(f"    - Files found: {len(dataset)} files")
    print(f"    - Batch size: {config['data']['batch_size']}")
    print(f"    - Patch size: {config['data']['patch_size']}")

    print("\n  Degradation Pipeline:")
    print(f"    - Transforms: {len(pipeline.steps)}")
    print(f"    - Config: {config['data']['degradation_config']}")

    print("\n" + "="*70)
    print(" "*20 + "✓ ALL VALIDATION CHECKS PASSED!")
    print("="*70)
    print("="*70)
    print("\nYour setup is ready for training!")
    print(f"\nTo start training, run:")
    print(f"  python training/train.py --config {args.config}")
    print("="*70)

    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
