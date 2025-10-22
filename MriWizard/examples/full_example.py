"""Comprehensive example showcasing all MriWizard features."""

import numpy as np
from MriWizard.core.pipeline import Pipeline
from MriWizard.io.raw_loader import LoadRawKspace
from MriWizard.io.dicom_loader import LoadDICOM
from MriWizard.io.image_loader import LoadImage
from MriWizard.degrade.noise import AddGaussianNoiseKspace
from MriWizard.degrade.undersample import UniformUndersample, RandomUndersample
from MriWizard.degrade.kmax import KmaxUndersample
from MriWizard.degrade.elliptical import EllipticalUndersample
from MriWizard.degrade.partial_fourier import PartialFourier
from MriWizard.degrade.combine import ApplyAll, RandomSubset, OneOf
from MriWizard.degrade.motion import RandomMotionKspace
from MriWizard.degrade.ghosting import RandomGhostingKspace
from MriWizard.degrade.spike import RandomSpikeKspace
from MriWizard.degrade.biasfield import RandomBiasFieldImage
from MriWizard.degrade.gibbs import RandomGibbsRinging
from MriWizard.degrade.blur import RandomGaussianBlurImage
from MriWizard.reconstruct.fft_recon import IFFTReconstruct
from MriWizard.datasets.dataset import MriWizardDataset
from MriWizard.datasets.pairing import build_dataset

def demo_all_degradations():
    """Demonstrate each degradation type individually."""
    
    print("=" * 60)
    print("MriWizard - All Degradation Types")
    print("=" * 60)
    
    # Create synthetic k-space data for demonstration
    print("\n1. Creating synthetic k-space data...")
    kspace = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)
    kspace = kspace.astype(np.complex64)
    
    base_record = {
        "kspace": kspace,
        "image": None,
        "mask": None,
        "metadata": {"applied": []}
    }
    
    # Test each degradation
    degradations = [
        ("Gaussian Noise", AddGaussianNoiseKspace(sigma=0.05)),
        ("Uniform Undersampling", UniformUndersample(R=4, axis=-2)),
        ("Random Undersampling", RandomUndersample(prob=0.5, axis=-2)),
        ("Kmax Undersampling", KmaxUndersample(fractions=(0.8, 0.7))),
        ("Elliptical Undersampling", EllipticalUndersample(radii_fractions=(0.8, 0.7))),
        ("Partial Fourier", PartialFourier(fractions=(0.75, 1.0), directions=("+", None))),
    ]
    
    for name, transform in degradations:
        print(f"\n2. Testing {name}...")
        record = {**base_record, "kspace": kspace.copy(), "metadata": {"applied": []}}
        record = transform(record)
        print(f"   Applied: {record['metadata']['applied'][-1]}")
        print(f"   K-space shape: {record['kspace'].shape}")
        if record['mask'] is not None:
            print(f"   Mask non-zero fraction: {np.mean(record['mask'] > 0):.2%}")
    
    print("\n" + "=" * 60)

def demo_artifact_transforms():
    """Demonstrate artifact-specific transforms."""
    
    print("\n" + "=" * 60)
    print("MriWizard - Artifact Transforms")
    print("=" * 60)
    
    # Create synthetic k-space data
    print("\n1. Creating synthetic k-space data...")
    kspace = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)
    kspace = kspace.astype(np.complex64)
    
    base_record = {
        "kspace": kspace,
        "image": None,
        "mask": None,
        "metadata": {"applied": []}
    }
    
    # Test artifact transforms
    artifacts = [
        ("Motion Artifacts", RandomMotionKspace(
            translation_px_range=(1.0, 5.0),
            num_segments_range=(3, 6)
        )),
        ("Ghosting", RandomGhostingKspace(
            num_ghosts_range=(2, 4),
            intensity_range=(0.1, 0.3)
        )),
        ("Spike Noise", RandomSpikeKspace(
            num_spikes_range=(2, 5),
            relative_amp_range=(5.0, 15.0)
        )),
        ("Bias Field", RandomBiasFieldImage(
            coeff_range=(0.2, 0.5),
            sigma_frac_range=(0.1, 0.3)
        )),
        ("Gibbs Ringing", RandomGibbsRinging(
            fraction_range=(0.6, 0.85)
        )),
        ("Gaussian Blur", RandomGaussianBlurImage(
            sigma_px_range=(0.5, 2.0)
        )),
    ]
    
    for name, transform in artifacts:
        print(f"\n2. Testing {name}...")
        record = {**base_record, "kspace": kspace.copy(), "metadata": {"applied": []}}
        record = transform(record)
        print(f"   Applied: {record['metadata']['applied'][-1]}")
    
    print("\n" + "=" * 60)

def demo_oneof_combinator():
    """Demonstrate OneOf combinator for random transform selection."""
    
    print("\n" + "=" * 60)
    print("MriWizard - OneOf Combinator")
    print("=" * 60)
    
    # Create synthetic k-space
    kspace = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)
    kspace = kspace.astype(np.complex64)
    
    # Define artifact pool
    artifacts = [
        RandomMotionKspace(translation_px_range=(1.0, 5.0)),
        RandomGhostingKspace(intensity_range=(0.1, 0.3)),
        RandomSpikeKspace(num_spikes_range=(2, 5)),
        RandomBiasFieldImage(coeff_range=(0.2, 0.5)),
    ]
    
    # OneOf with equal probabilities
    print("\n1. OneOf with equal probabilities:")
    oneof_equal = OneOf(artifacts)
    
    for i in range(5):
        record = {
            "kspace": kspace.copy(),
            "image": None,
            "mask": None,
            "metadata": {"applied": []}
        }
        record = oneof_equal(record)
        applied = record['metadata']['applied'][-1]['transform']
        print(f"   Sample {i+1}: {applied}")
    
    # OneOf with custom probabilities
    print("\n2. OneOf with custom probabilities (70% motion, 20% ghosting, 5% spike, 5% bias):")
    oneof_weighted = OneOf(artifacts, probs=[0.7, 0.2, 0.05, 0.05])
    
    for i in range(5):
        record = {
            "kspace": kspace.copy(),
            "image": None,
            "mask": None,
            "metadata": {"applied": []}
        }
        record = oneof_weighted(record)
        applied = record['metadata']['applied'][-1]['transform']
        print(f"   Sample {i+1}: {applied}")
    
    # Use in pipeline
    print("\n3. OneOf in a full pipeline:")
    pipeline = Pipeline([
        RandomUndersample(prob_range=(0.3, 0.5)),
        OneOf([
            RandomMotionKspace(),
            RandomGhostingKspace(),
            RandomSpikeKspace(),
        ]),
        AddGaussianNoiseKspace(sigma_range=(0.01, 0.03)),
        IFFTReconstruct(normalize=True),
    ])
    
    record = {
        "kspace": kspace.copy(),
        "image": None,
        "mask": None,
        "metadata": {"applied": []}
    }
    record = pipeline(record)
    
    applied = [d['transform'] for d in record['metadata']['applied']]
    print(f"   Pipeline applied: {' -> '.join(applied)}")
    
    print("\n" + "=" * 60)

def demo_combined_pipeline():
    """Demonstrate combining multiple degradations."""
    
    print("\n" + "=" * 60)
    print("MriWizard - Combined Pipeline")
    print("=" * 60)
    
    # Create synthetic k-space
    kspace = np.random.randn(256, 256) + 1j * np.random.randn(256, 256)
    kspace = kspace.astype(np.complex64)
    
    # Build comprehensive pipeline
    pipeline = Pipeline([
        AddGaussianNoiseKspace(sigma_range=(0.01, 0.05)),
        RandomUndersample(prob_range=(0.3, 0.7), axis=-2),
        KmaxUndersample(fraction_ranges=((0.6, 0.9), (0.6, 0.9))),
        EllipticalUndersample(radii_ranges=((0.7, 0.9), (0.7, 0.9))),
        PartialFourier(fraction_ranges=((0.65, 0.85), (1.0, 1.0)),
                      directions=("+", None)),
        IFFTReconstruct(normalize=True),
    ])
    
    record = {
        "kspace": kspace,
        "image": None,
        "mask": None,
        "metadata": {"applied": []}
    }
    
    print("\n1. Applying full degradation pipeline...")
    record = pipeline(record)
    
    print(f"\n2. Results:")
    print(f"   Image shape: {record['image'].shape}")
    print(f"   Image range: [{record['image'].min():.3f}, {record['image'].max():.3f}]")
    print(f"   Number of degradations applied: {len(record['metadata']['applied'])}")
    print(f"\n3. Applied degradations:")
    for i, deg in enumerate(record['metadata']['applied'], 1):
        print(f"   {i}. {deg['transform']}")
    
    print("\n" + "=" * 60)

def demo_random_subset():
    """Demonstrate random subset of degradations."""
    
    print("\n" + "=" * 60)
    print("MriWizard - Random Degradation Subset")
    print("=" * 60)
    
    # Create synthetic k-space
    kspace = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)
    kspace = kspace.astype(np.complex64)
    
    # Define degradation pool
    degradations = [
        AddGaussianNoiseKspace(sigma_range=(0.01, 0.05)),
        UniformUndersample(R_range=(2, 4), axis=-2),
        RandomUndersample(prob_range=(0.3, 0.7), axis=-2),
        KmaxUndersample(fraction_ranges=((0.6, 0.9), (0.6, 0.9))),
        EllipticalUndersample(radii_ranges=((0.7, 0.9), (0.7, 0.9))),
    ]
    
    # Pipeline with random subset
    pipeline = Pipeline([
        RandomSubset(degradations, min_k=2, max_k=4),
        IFFTReconstruct(normalize=True),
    ])
    
    print("\n1. Running 5 random samples...")
    for i in range(5):
        record = {
            "kspace": kspace.copy(),
            "image": None,
            "mask": None,
            "metadata": {"applied": []}
        }
        record = pipeline(record)
        applied = [d['transform'] for d in record['metadata']['applied'][:-1]]  # Exclude IFFTReconstruct
        print(f"   Sample {i+1}: {len(applied)} degradations - {', '.join(applied)}")
    
    print("\n" + "=" * 60)

def demo_different_inputs():
    """Demonstrate loading different input types."""
    
    print("\n" + "=" * 60)
    print("MriWizard - Different Input Types")
    print("=" * 60)
    
    print("\n1. Loaders available:")
    print("   - LoadRawKspace: .h5, .mat, .npy, .npz files")
    print("   - LoadDICOM: DICOM files (.dcm)")
    print("   - LoadImage: .jpg, .png, .tiff, etc.")
    
    print("\n2. All loaders produce standardized records:")
    print("   {")
    print("     'kspace': np.ndarray | None,")
    print("     'image': np.ndarray | None,")
    print("     'mask': None,")
    print("     'metadata': dict")
    print("   }")
    
    print("\n3. Image loader can convert to k-space:")
    loader = LoadImage(convert_to_kspace=True, grayscale=True)
    print(f"   LoadImage(convert_to_kspace=True) ready")
    
    print("\n" + "=" * 60)

def demo_dataset_modes():
    """Demonstrate both on-the-fly and offline modes."""
    
    print("\n" + "=" * 60)
    print("MriWizard - Dataset Modes")
    print("=" * 60)
    
    print("\n1. On-the-fly mode (for training):")
    print("   - Zero disk usage")
    print("   - Different degradation each epoch")
    print("   - PyTorch DataLoader compatible")
    print("""
   ds = MriWizardDataset(
       paths=["data/*.png"],
       loader=LoadImage(convert_to_kspace=True),
       pipeline=pipeline
   )
   loader = DataLoader(ds, batch_size=8)
   """)
    
    print("\n2. Offline mode (pre-generate):")
    print("   - Fast training (no degradation overhead)")
    print("   - Reproducible datasets")
    print("   - Format: .npy (per-sample) or .npz (batched)")
    print("""
   build_dataset(
       input_paths=["data/*.h5"],
       output_dir="processed/",
       loader=LoadRawKspace(),
       pipeline=pipeline,
       format="npz",
       shard_size=1024
   )
   """)
    
    print("\n" + "=" * 60)

def main():
    """Run all demonstrations."""
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  MriWizard - Comprehensive Feature Demonstration  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    demo_all_degradations()
    demo_artifact_transforms()
    demo_oneof_combinator()
    demo_combined_pipeline()
    demo_random_subset()
    demo_different_inputs()
    demo_dataset_modes()
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Demonstration Complete!  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("║" + "  See examples/ for practical usage scripts  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

if __name__ == "__main__":
    main()

