"""On-the-fly dataset for training with degradation pipeline."""

import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
from glob import glob
from MriWizard.core.base import Loader, Record
from MriWizard.core.pipeline import Pipeline

# Try to import torch for optional PyTorch compatibility
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchDataset = object

class MriWizardDataset(TorchDataset if TORCH_AVAILABLE else object):
    """Dataset that applies degradation pipeline on-the-fly during training."""
    
    def __init__(self, paths: Union[List[str], str], loader: Loader, pipeline: Pipeline,
                 return_target: bool = True):
        """
        Initialize dataset.
        
        Args:
            paths: List of file paths or glob patterns
            loader: Loader instance to read files
            pipeline: Pipeline to apply degradations
            return_target: If True, return (input, target) pairs; if False, just input
        """
        self.loader = loader
        self.pipeline = pipeline
        self.return_target = return_target
        
        # Expand paths
        if isinstance(paths, str):
            paths = [paths]
        
        self.file_paths = []
        for path in paths:
            if "*" in path or "?" in path:
                # Glob pattern
                self.file_paths.extend(sorted(glob(path)))
            else:
                self.file_paths.append(path)
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No files found for paths: {paths}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, np.ndarray, dict], np.ndarray]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            If return_target=True: (input, target, context)
            If return_target=False: input only
        """
        # Load file
        path = self.file_paths[idx]
        record = self.loader.load(path)
        
        # Store original target before degradation
        if self.return_target:
            # If we have an image, use it as target
            if record["image"] is not None:
                target = record["image"].copy()
            else:
                # Reconstruct target from clean k-space
                from MriWizard.reconstruct.fft_recon import IFFTReconstruct
                recon = IFFTReconstruct()
                target_record = recon({"kspace": record["kspace"].copy(), 
                                      "image": None, 
                                      "mask": None,
                                      "metadata": {"applied": []}})
                target = target_record["image"]
        
        # Apply pipeline (degradations + reconstruction)
        record = self.pipeline(record)
        
        # Get degraded input
        if record["image"] is None:
            raise ValueError("Pipeline must produce an image (use IFFTReconstruct)")
        
        input_img = record["image"]
        context = record["metadata"]
        
        if self.return_target:
            return input_img, target, context
        else:
            return input_img

