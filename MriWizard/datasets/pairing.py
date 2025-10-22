"""Offline dataset builder that pre-generates degraded pairs."""

import numpy as np
from pathlib import Path
from typing import List, Union
from glob import glob
from tqdm import tqdm
from MriWizard.core.base import Loader
from MriWizard.core.pipeline import Pipeline
from MriWizard.io.writer import save_sample, NpzSharder

def build_dataset(input_paths: Union[List[str], str],
                  output_dir: str,
                  loader: Loader,
                  pipeline: Pipeline,
                  format: Union[str, None] = "npz",
                  shard_size: int = 1024,
                  n_samples: Union[int, None] = None) -> None:
    """
    Build a dataset by applying pipeline to input files and saving results.
    
    Args:
        input_paths: List of file paths or glob patterns
        output_dir: Directory to save processed data
        loader: Loader instance to read files
        pipeline: Pipeline to apply degradations
        format: Output format - "npy" for per-sample, "npz" for sharded, None for no export
        shard_size: Number of samples per shard (for npz format)
        n_samples: Maximum number of samples to generate (None = all)
    """
    # Expand paths
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    file_paths = []
    for path in input_paths:
        if "*" in path or "?" in path:
            # Glob pattern
            file_paths.extend(sorted(glob(path)))
        else:
            file_paths.append(path)
    
    if len(file_paths) == 0:
        raise ValueError(f"No files found for paths: {input_paths}")
    
    # Limit number of samples if requested
    if n_samples is not None:
        file_paths = file_paths[:n_samples]
    
    print(f"Processing {len(file_paths)} files...")
    
    # Initialize writer if needed
    if format == "npz":
        sharder = NpzSharder(output_dir, shard_size)
    elif format == "npy":
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    elif format is None:
        print("Warning: format=None, no data will be saved to disk")
        return
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Process files
    for idx, path in enumerate(tqdm(file_paths, desc="Building dataset")):
        # Load file
        record = loader.load(path)
        
        # Store original target before degradation
        if record["image"] is not None:
            target = record["image"].copy()
        else:
            # Reconstruct target from clean k-space
            from MriWizard.reconstruct.fft_recon import IFFTReconstruct
            recon = IFFTReconstruct()
            target_record = recon({
                "kspace": record["kspace"].copy(),
                "image": None,
                "mask": None,
                "metadata": {"applied": []}
            })
            target = target_record["image"]
        
        # Apply pipeline (degradations + reconstruction)
        record = pipeline(record)
        
        # Get degraded input
        if record["image"] is None:
            raise ValueError("Pipeline must produce an image (use IFFTReconstruct)")
        
        input_img = record["image"]
        context = record["metadata"]
        
        # Save based on format
        if format == "npy":
            sample_id = f"sample_{idx:06d}"
            save_sample(output_dir, sample_id, input_img, target, context)
        elif format == "npz":
            sharder.add(input_img, target, context)
    
    # Finalize if using sharder
    if format == "npz":
        sharder.finalize()
        print(f"Dataset saved to {output_dir} ({sharder.current_shard_idx} shards)")
    else:
        print(f"Dataset saved to {output_dir}")

