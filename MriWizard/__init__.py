"""
MriWizard - A modular data preparation framework for MRI image reconstruction and restoration tasks.
"""

__version__ = "0.1.0"

from MriWizard.core.pipeline import Pipeline
from MriWizard.core.base import Transform, Loader, REGISTRY, register

# Import commonly used modules for convenience
from MriWizard import degrade
from MriWizard import preprocess
from MriWizard import spatial
from MriWizard import io
from MriWizard import reconstruct

__all__ = [
    "Pipeline",
    "Transform",
    "Loader",
    "REGISTRY",
    "register",
    "degrade",
    "preprocess",
    "spatial",
    "io",
    "reconstruct",
]

