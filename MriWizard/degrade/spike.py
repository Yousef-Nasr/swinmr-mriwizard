"""Spike artifact simulation via random k-space corruption.

Following TorchIO's approach:
- Spike positions are sampled as fractional coordinates in [0, 1]^3
- Positions are mapped to k-space indices relative to k-space center
- Spikes are added to k-space maximum * intensity factor
- Single-sided spikes (not symmetric) for realistic artifacts
"""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, ifft2c, to_complex64, to_float32

class RandomSpikeKspace:
    """Inject sparse high-amplitude spikes in k-space (TorchIO approach).
    
    This implementation follows TorchIO's spike artifact simulation:
    1. Sample spike positions as fractional coordinates [0, 1]
    2. Map positions to k-space indices relative to center
    3. Add spikes with intensity = kspace_max * intensity_factor
    4. Single-sided (asymmetric) for realistic artifacts
    
    Reference: TorchIO RandomSpike transform
    """
    
    def __init__(self,
                 num_spikes_range: Union[Tuple[int, int], None] = None,
                 relative_amp_range: Union[Tuple[float, float], None] = None):
        """
        Initialize spike artifact transform.
        
        Args:
            num_spikes_range: Range for number of spikes to add (min, max)
                            TorchIO default: (1, 1) to (1, 10)
            relative_amp_range: Range for spike intensity relative to k-space max (min, max)
                               TorchIO default: (1, 3)
                               Higher values = stronger artifacts
        
        Following TorchIO semantics:
        - Spike positions sampled uniformly in normalized space [0, 1]
        - Intensity is ratio to k-space spectrum maximum
        - Single-sided spikes (realistic RF interference)
        """
        if num_spikes_range is None:
            num_spikes_range = (1, 5)
        if relative_amp_range is None:
            relative_amp_range = (1.0, 3.0)
        
        self.num_spikes_range = num_spikes_range
        self.relative_amp_range = relative_amp_range
    
    def __call__(self, record: Record) -> Record:
        """
        Apply spike artifacts to k-space following TorchIO approach.
        
        Args:
            record: Input record
            
        Returns:
            Record with spike-corrupted k-space
        """
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"].astype(np.complex64)
        
        # Sample number of spikes
        num_spikes = np.random.randint(
            self.num_spikes_range[0],
            self.num_spikes_range[1] + 1
        )
        
        if num_spikes == 0:
            return record
        
        # Sample single intensity factor for all spikes (TorchIO approach)
        intensity_factor = float(np.random.uniform(
            self.relative_amp_range[0],
            self.relative_amp_range[1]
        ))
        
        # Sample spike positions as fractional coordinates [0, 1]
        # TorchIO: spikes_positions = torch.rand(num_spikes, 3).numpy()
        spike_positions_normalized = np.random.rand(num_spikes, kspace.ndim).astype(np.float32)
        
        # Get k-space shape and center
        shape = np.array(kspace.shape)
        mid_shape = shape // 2  # Center of k-space
        
        # Convert normalized positions [0, 1] to k-space indices
        # TorchIO: indices = np.floor(spikes_positions * shape).astype(int)
        indices = np.floor(spike_positions_normalized * shape).astype(int)
        
        # Compute artifact intensity (max of spectrum * intensity_factor)
        # TorchIO: artifact = spectrum.cpu().numpy().max() * intensity_factor
        kspace_max = float(np.max(np.abs(kspace)))
        artifact_value = kspace_max * intensity_factor
        
        # Add spikes to k-space
        spike_kspace = kspace.copy()
        spike_locations = []
        
        for index in indices:
            # Map to k-space relative to center (TorchIO approach)
            # diff = index - mid_shape
            # i, j, k = mid_shape + diff  (which simplifies to just index)
            diff = index - mid_shape
            spike_idx = tuple(mid_shape + diff)
            
            # Add spike (real positive value, single-sided)
            # TorchIO: spectrum[i, j, k] += artifact
            spike_kspace[spike_idx] = spike_kspace[spike_idx] + np.complex64(artifact_value)
            
            spike_locations.append([int(x) for x in spike_idx])
        
        # Update record
        record["kspace"] = spike_kspace
        record["metadata"]["applied"].append({
            "transform": "RandomSpikeKspace",
            "num_spikes": int(num_spikes),
            "intensity_factor": float(intensity_factor),
            "spike_positions_normalized": spike_positions_normalized.tolist(),
            "spike_locations": spike_locations,
            "relative_amplitudes": [float(intensity_factor)] * num_spikes  # All same in TorchIO
        })
        
        return record

