"""Example: On-the-fly augmentation with PyTorch DataLoader."""

from MriWizard.core.pipeline import Pipeline
from MriWizard.io.image_loader import LoadImage
from MriWizard.degrade.noise import AddGaussianNoiseKspace
from MriWizard.degrade.undersample import RandomUndersample
from MriWizard.degrade.kmax import KmaxUndersample
from MriWizard.reconstruct.fft_recon import IFFTReconstruct
from MriWizard.datasets.dataset import MriWizardDataset

def main():
    # Define degradation pipeline
    pipeline = Pipeline([
        AddGaussianNoiseKspace(std=(0.005, 0.05)),
        RandomUndersample(prob_range=(0.3, 0.8), axis=-2),
        KmaxUndersample(fraction_ranges=((0.6, 0.95), (0.6, 0.95))),
        IFFTReconstruct(normalize=True),
    ])
    
    # Create dataset (replace with your image paths)
    ds = MriWizardDataset(
        paths=["path/to/images/*.png"],
        loader=LoadImage(convert_to_kspace=True),
        pipeline=pipeline,
        return_target=True
    )
    
    print(f"Dataset size: {len(ds)}")
    
    # Get a sample
    if len(ds) > 0:
        input_img, target_img, context = ds[0]
        print(f"Input shape: {input_img.shape}")
        print(f"Target shape: {target_img.shape}")
        print(f"Applied degradations: {len(context['applied'])}")
        for deg in context['applied']:
            print(f"  - {deg['transform']}")
    
    # Use with PyTorch DataLoader
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            ds,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Increase for parallel loading
        )
        
        print(f"\nDataLoader ready with {len(dataloader)} batches")
        
        # Example: iterate through one batch
        for batch_idx, (inputs, targets, contexts) in enumerate(dataloader):
            print(f"Batch {batch_idx}: inputs={inputs.shape}, targets={targets.shape}")
            break
            
    except ImportError:
        print("\nPyTorch not available. Install with: pip install torch")

if __name__ == "__main__":
    main()

