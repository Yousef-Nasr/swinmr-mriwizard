"""Identity transform - no operation placeholder for degradation pipelines."""

from MriWizard.core.base import Record


class Identity:
    """No-operation transform that returns the record unchanged.
    
    Useful as a placeholder in degradation pipelines, especially with OneOf
    to allow skipping degradation entirely.
    
    Example:
        # Skip degradation with 30% probability
        degrader = OneOf([
            AddGaussianNoiseKspace(std=0.1),
            RandomUndersample(factor=2),
            Identity()  # 33% chance to skip all degradation
        ])
    """
    
    def __init__(self):
        """Initialize identity transform (no parameters needed)."""
        pass
    
    def __call__(self, record: Record) -> Record:
        """
        Return record unchanged.
        
        Args:
            record: Input record
            
        Returns:
            Same record, completely unchanged
        """
        # Optionally log this transform was applied
        record["metadata"]["applied"].append({
            "transform": "Identity"
        })
        return record
