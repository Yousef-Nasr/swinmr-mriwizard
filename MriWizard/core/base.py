"""Base classes and protocols for MriWizard."""

from typing import Protocol, Dict, Any

Record = Dict[str, Any]

class Transform(Protocol):
    """Protocol for transforms that modify records."""
    
    def __call__(self, record: Record) -> Record:
        """Apply transform to a record and return modified record."""
        ...

class Loader(Protocol):
    """Protocol for loaders that read files and return records."""
    
    def load(self, path: str) -> Record:
        """Load a file and return a standardized record."""
        ...

# Simple registry for custom transforms
REGISTRY: Dict[str, type] = {}

def register(name: str):
    """Decorator to register a transform class in the global registry."""
    def _wrap(cls):
        REGISTRY[name] = cls
        return cls
    return _wrap

