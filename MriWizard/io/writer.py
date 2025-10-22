"""Writers for saving processed data as .npy or .npz files."""

import numpy as np
from pathlib import Path
from typing import Dict, Any

def save_sample(output_dir: str, sample_id: str, input_data: np.ndarray, 
                target_data: np.ndarray, context: Dict[str, Any]) -> None:
    """
    Save a single sample as separate .npy files with a .json context.
    
    Args:
        output_dir: Directory to save files
        sample_id: Unique identifier for this sample
        input_data: Degraded input image
        target_data: Ground truth target image
        context: Metadata and degradation parameters
    """
    import json
    
    output_dir = Path(output_dir)
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(str(sample_dir / "input.npy"), input_data)
    np.save(str(sample_dir / "target.npy"), target_data)
    
    with open(sample_dir / "context.json", "w") as f:
        json.dump(context, f, indent=2, default=str)

class NpzSharder:
    """Write batches of samples to sharded .npz files."""
    
    def __init__(self, output_dir: str, shard_size: int = 1024):
        """
        Initialize sharder.
        
        Args:
            output_dir: Directory to save shards
            shard_size: Number of samples per shard
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        
        self.current_shard = []
        self.current_shard_idx = 0
        self.sample_count = 0
    
    def add(self, input_data: np.ndarray, target_data: np.ndarray, 
            context: Dict[str, Any]) -> None:
        """
        Add a sample to the current shard.
        
        Args:
            input_data: Degraded input image
            target_data: Ground truth target image
            context: Metadata and degradation parameters
        """
        self.current_shard.append({
            "input": input_data,
            "target": target_data,
            "context": context
        })
        self.sample_count += 1
        
        if len(self.current_shard) >= self.shard_size:
            self._flush()
    
    def _flush(self) -> None:
        """Write current shard to disk and reset."""
        if not self.current_shard:
            return
        
        # Stack arrays
        inputs = np.stack([s["input"] for s in self.current_shard])
        targets = np.stack([s["target"] for s in self.current_shard])
        contexts = [s["context"] for s in self.current_shard]
        
        # Save shard
        shard_name = f"data_{self.current_shard_idx:05d}.npz"
        np.savez_compressed(
            str(self.output_dir / shard_name),
            inputs=inputs,
            targets=targets,
            contexts=contexts
        )
        
        self.current_shard = []
        self.current_shard_idx += 1
    
    def finalize(self) -> None:
        """Flush remaining samples and close."""
        self._flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

