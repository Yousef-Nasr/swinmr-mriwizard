"""Ghosting artifact simulation via periodic k-space line mixing."""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, to_complex64

class RandomGhostingKspace:
    """Simulate N/2 ghosting artifacts via periodic line replication."""
    
    def __init__(self,
                 num_ghosts_range: Union[Tuple[int, int], None] = None,
                 intensity_range: Union[Tuple[float, float], None] = None,
                 axis: int = -2):
        """
        Initialize ghosting artifact transform.
        
        Args:
            num_ghosts_range: Range for number of ghost reflections (min, max)
            intensity_range: Range for ghost intensity relative to original (min, max)
            axis: Axis along which ghosting occurs (default: -2, phase-encode)
        
        Ghosting is simulated by adding weighted copies of alternating k-space lines.
        """
        if num_ghosts_range is None:
            num_ghosts_range = (2, 4)
        if intensity_range is None:
            intensity_range = (0.05, 0.25)
        
        self.num_ghosts_range = num_ghosts_range
        self.intensity_range = intensity_range
        self.axis = axis
    
    def __call__(self, record: Record) -> Record:
        """
        Apply ghosting artifacts to k-space.
        
        Args:
            record: Input record
            
        Returns:
            Record with ghosting-corrupted k-space
        """
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"].astype(np.complex64)
        
        # Sample parameters
        num_ghosts = np.random.randint(
            self.num_ghosts_range[0],
            self.num_ghosts_range[1] + 1
        )
        intensity = np.random.uniform(
            self.intensity_range[0],
            self.intensity_range[1]
        )
        
        # Create ghosting pattern
        # N/2 ghost: every other line has different phase/amplitude
        axis_size = kspace.shape[self.axis]
        
        # Create alternating pattern
        pattern = np.ones(axis_size, dtype=np.float32)
        
        # Alternate lines get phase shift and intensity change
        # Period = 2 for N/2 ghost, can generalize to num_ghosts
        period = num_ghosts
        
        for i in range(axis_size):
            if i % period != 0:
                pattern[i] = 1.0 + intensity * np.random.uniform(0.5, 1.5)
        
        # Apply pattern along axis
        # Reshape pattern to broadcast correctly
        pattern_shape = [1] * kspace.ndim
        pattern_shape[self.axis] = axis_size
        pattern_broadcast = pattern.reshape(pattern_shape)
        
        ghosted_kspace = kspace * pattern_broadcast
        
        # Add phase modulation for more realistic ghosting
        # Phase shifts alternate between 0 and π
        phase_pattern = np.zeros(axis_size, dtype=np.float32)
        for i in range(axis_size):
            if i % period != 0:
                phase_pattern[i] = np.random.uniform(-np.pi, np.pi) * intensity
        
        phase_pattern_broadcast = phase_pattern.reshape(pattern_shape)
        phase_shift = np.exp(1j * phase_pattern_broadcast).astype(np.complex64)
        
        ghosted_kspace = ghosted_kspace * phase_shift
        
        # Update record
        record["kspace"] = ghosted_kspace
        record["metadata"]["applied"].append({
            "transform": "RandomGhostingKspace",
            "num_ghosts": int(num_ghosts),
            "intensity": float(intensity),
            "axis": int(self.axis)
        })
        
        return record

