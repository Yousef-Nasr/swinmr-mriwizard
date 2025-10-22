"""Combine multiple degradations in flexible ways."""

import numpy as np
from typing import List
from MriWizard.core.base import Transform, Record

class ApplyAll:
    """Apply all provided transforms sequentially."""
    
    def __init__(self, transforms: List[Transform]):
        """
        Initialize.
        
        Args:
            transforms: List of transforms to apply in order
        """
        self.transforms = transforms
    
    def __call__(self, record: Record) -> Record:
        """
        Apply all transforms.
        
        Args:
            record: Input record
            
        Returns:
            Record after all transforms
        """
        for transform in self.transforms:
            record = transform(record)
        return record

class RandomSubset:
    """Randomly sample and apply a subset of transforms."""
    
    def __init__(self, transforms: List[Transform], min_k: int = 1, max_k: int = None):
        """
        Initialize.
        
        Args:
            transforms: List of available transforms
            min_k: Minimum number of transforms to apply
            max_k: Maximum number of transforms to apply (if None, use len(transforms))
        """
        self.transforms = transforms
        self.min_k = min_k
        self.max_k = max_k if max_k is not None else len(transforms)
        
        if self.min_k > len(transforms):
            raise ValueError(f"min_k ({min_k}) cannot exceed number of transforms ({len(transforms)})")
        if self.max_k > len(transforms):
            raise ValueError(f"max_k ({max_k}) cannot exceed number of transforms ({len(transforms)})")
    
    def __call__(self, record: Record) -> Record:
        """
        Apply random subset of transforms.
        
        Args:
            record: Input record
            
        Returns:
            Record after selected transforms
        """
        # Sample number of transforms to apply
        k = np.random.randint(self.min_k, self.max_k + 1)
        
        # Randomly select k transforms without replacement
        selected = np.random.choice(len(self.transforms), size=k, replace=False)
        
        # Apply selected transforms
        for idx in selected:
            record = self.transforms[idx](record)
        
        return record

class OneOf:
    """Select and apply exactly one transform from a list."""
    
    def __init__(self, transforms: List[Transform], probs: List[float] = None):
        """
        Initialize.
        
        Args:
            transforms: List of transforms to choose from
            probs: Optional probability weights for each transform (must sum to 1.0)
                  If None, all transforms have equal probability
        """
        if not transforms:
            raise ValueError("transforms list cannot be empty")
        
        self.transforms = transforms
        
        if probs is not None:
            if len(probs) != len(transforms):
                raise ValueError(f"Length of probs ({len(probs)}) must match transforms ({len(transforms)})")
            
            probs_array = np.array(probs, dtype=np.float32)
            if not np.isclose(probs_array.sum(), 1.0):
                raise ValueError(f"Probabilities must sum to 1.0, got {probs_array.sum()}")
            
            self.probs = probs_array
        else:
            # Equal probability for all transforms
            self.probs = np.ones(len(transforms), dtype=np.float32) / len(transforms)
    
    def __call__(self, record: Record) -> Record:
        """
        Select and apply one transform.
        
        Args:
            record: Input record
            
        Returns:
            Record after selected transform
        """
        # Select one transform based on probabilities
        idx = np.random.choice(len(self.transforms), p=self.probs)
        
        # Apply selected transform
        return self.transforms[idx](record)

