"""Pipeline orchestration for MriWizard."""

from typing import List
from MriWizard.core.base import Transform, Record

class Pipeline:
    """Execute a sequence of transforms on a record."""
    
    def __init__(self, steps: List[Transform]):
        """
        Initialize pipeline with a list of transforms.
        
        Args:
            steps: List of transforms to apply sequentially
        """
        self.steps = steps
    
    def __call__(self, record: Record) -> Record:
        """
        Apply all steps sequentially to the record.
        
        Args:
            record: Input record
            
        Returns:
            Transformed record
        """
        for step in self.steps:
            record = step(record)
        return record

