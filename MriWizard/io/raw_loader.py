"""Loader for raw k-space data in various formats (.h5, .mat, .npy)."""

import numpy as np
from pathlib import Path
from typing import Union
from MriWizard.core.base import Record
from MriWizard.core.utils import to_complex64

class LoadRawKspace:
    """Load raw k-space data from .h5, .mat, or .npy files."""
    
    def __init__(self, key: Union[str, None] = None):
        """
        Initialize loader.
        
        Args:
            key: Key/dataset name for .h5/.mat files (if None, auto-detect)
        """
        self.key = key
    
    def load(self, path: str) -> Record:
        """
        Load k-space from file.
        
        Args:
            path: Path to file
            
        Returns:
            Record with kspace, image=None, and metadata
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == ".npy":
            kspace = np.load(str(path))
            metadata = {}
        elif suffix == ".npz":
            data = np.load(str(path))
            if self.key and self.key in data:
                kspace = data[self.key]
            else:
                # Try common keys
                for k in ["kspace", "data", "arr_0"]:
                    if k in data:
                        kspace = data[k]
                        break
                else:
                    kspace = data[list(data.keys())[0]]
            metadata = {}
        elif suffix == ".h5" or suffix == ".hdf5":
            import h5py
            with h5py.File(str(path), "r") as f:
                if self.key:
                    kspace = f[self.key][:]
                else:
                    # Try common keys
                    for k in ["kspace", "data", "reconstruction_rss"]:
                        if k in f:
                            kspace = f[k][:]
                            break
                    else:
                        # Use first dataset
                        kspace = f[list(f.keys())[0]][:]
                
                # Extract metadata from attributes
                metadata = dict(f.attrs) if hasattr(f, "attrs") else {}
        elif suffix == ".mat":
            from scipy.io import loadmat
            data = loadmat(str(path))
            if self.key and self.key in data:
                kspace = data[self.key]
            else:
                # Try common keys, skip matlab metadata keys
                for k in ["kspace", "data"]:
                    if k in data:
                        kspace = data[k]
                        break
                else:
                    # Get first non-metadata key
                    for k in data.keys():
                        if not k.startswith("__"):
                            kspace = data[k]
                            break
            metadata = {}
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Ensure complex64
        kspace = to_complex64(kspace)
        
        return {
            "kspace": kspace,
            "image": None,
            "mask": None,
            "metadata": {"source": str(path), **metadata, "applied": []}
        }
    
    def __call__(self, record: Record) -> Record:
        """Allow use as a transform in pipeline."""
        return record

