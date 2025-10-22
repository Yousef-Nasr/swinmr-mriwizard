"""Example: Build offline dataset with pre-generated degraded pairs."""

from MriWizard.core.pipeline import Pipeline
from MriWizard.io.raw_loader import LoadRawKspace
from MriWizard.io.image_loader import LoadImage
from MriWizard.degrade.noise import AddGaussianNoiseKspace
from MriWizard.degrade.undersample import UniformUndersample, RandomUndersample
from MriWizard.degrade.kmax import KmaxUndersample
from MriWizard.degrade.elliptical import EllipticalUndersample
from MriWizard.degrade.partial_fourier import PartialFourier
from MriWizard.degrade.combine import ApplyAll, RandomSubset
from MriWizard.reconstruct.fft_recon import IFFTReconstruct
from MriWizard.datasets.pairing import build_dataset

def example_raw_kspace():
    """Process raw k-space data."""
    
    # Define pipeline with multiple degradations
    pipeline = Pipeline([
        AddGaussianNoiseKspace(sigma_range=(0.005, 0.05)),
        UniformUndersample(R_range=(2, 4), axis=-2),
        RandomUndersample(prob_range=(0.3, 0.8), axis=-2),
        KmaxUndersample(fraction_ranges=((0.7, 0.9), (0.7, 0.9), (1.0, 1.0))),
        EllipticalUndersample(radii_ranges=((0.6, 0.9), (0.6, 0.9), (0.9, 1.0))),
        PartialFourier(fraction_ranges=((0.6, 0.85), (1.0, 1.0), (1.0, 1.0)),
                      directions=("+", None, None)),
        IFFTReconstruct(normalize=True),
    ])
    
    # Build dataset
    build_dataset(
        input_paths=["path/to/raw/*.h5", "path/to/raw/*.npy"],
        output_dir="output/processed_raw",
        loader=LoadRawKspace(),
        pipeline=pipeline,
        format="npz",
        shard_size=1024,
        n_samples=None  # Process all files
    )

def example_images():
    """Process standard images."""
    
    # Define pipeline
    pipeline = Pipeline([
        AddGaussianNoiseKspace(sigma_range=(0.01, 0.1)),
        RandomUndersample(prob_range=(0.4, 0.9), axis=-2),
        KmaxUndersample(fraction_ranges=((0.6, 0.95), (0.6, 0.95))),
        IFFTReconstruct(normalize=True),
    ])
    
    # Build dataset
    build_dataset(
        input_paths=["path/to/images/*.png", "path/to/images/*.jpg"],
        output_dir="output/processed_images",
        loader=LoadImage(convert_to_kspace=True),
        pipeline=pipeline,
        format="npy",  # Save as individual .npy files
        n_samples=1000
    )

def example_random_degradations():
    """Use random subset of degradations."""
    
    # Define available degradations
    degradations = [
        AddGaussianNoiseKspace(sigma_range=(0.005, 0.05)),
        UniformUndersample(R_range=(2, 4), axis=-2),
        RandomUndersample(prob_range=(0.3, 0.8), axis=-2),
        KmaxUndersample(fraction_ranges=((0.6, 0.9), (0.6, 0.9))),
        EllipticalUndersample(radii_ranges=((0.6, 0.9), (0.6, 0.9))),
        PartialFourier(fraction_ranges=((0.6, 0.85), (1.0, 1.0))),
    ]
    
    # Pipeline that randomly selects 2-4 degradations per sample
    pipeline = Pipeline([
        RandomSubset(degradations, min_k=2, max_k=4),
        IFFTReconstruct(normalize=True),
    ])
    
    build_dataset(
        input_paths=["path/to/data/*.h5"],
        output_dir="output/random_degradations",
        loader=LoadRawKspace(),
        pipeline=pipeline,
        format="npz",
        shard_size=512
    )

if __name__ == "__main__":
    print("Example 1: Processing raw k-space data")
    print("Uncomment the desired example to run:\n")
    
    # Uncomment the example you want to run:
    # example_raw_kspace()
    # example_images()
    # example_random_degradations()
    
    print("Edit the file paths and uncomment an example to run.")

