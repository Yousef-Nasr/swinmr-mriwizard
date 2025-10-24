"""
Pipeline Builder Module

Dynamically builds MriWizard degradation pipeline from JSON configuration.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add MriWizard to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'MriWizard'))

from MriWizard.core.pipeline import Pipeline
from MriWizard.degrade.noise import AddGaussianNoiseKspace
from MriWizard.degrade.undersample import RandomUndersample, UniformUndersample
from MriWizard.degrade.partial_fourier import PartialFourier
from MriWizard.degrade.kmax import KmaxUndersample
from MriWizard.degrade.elliptical import EllipticalUndersample
from MriWizard.degrade.combine import OneOf, ApplyAll
from MriWizard.degrade.identity import Identity
from MriWizard.reconstruct.fft_recon import IFFTReconstruct

# Optional artifacts (may not all be present)
try:
    from MriWizard.degrade.motion import RandomMotionKspace
except ImportError:
    RandomMotionKspace = None

try:
    from MriWizard.degrade.ghosting import RandomGhostingKspace
except ImportError:
    RandomGhostingKspace = None

try:
    from MriWizard.degrade.spike import RandomSpikeKspace
except ImportError:
    RandomSpikeKspace = None

try:
    from MriWizard.degrade.gibbs import RandomGibbsRinging
except ImportError:
    RandomGibbsRinging = None

try:
    from MriWizard.degrade.biasfield import RandomBiasFieldImage
except ImportError:
    RandomBiasFieldImage = None

try:
    from MriWizard.degrade.blur import RandomGaussianBlurImage
except ImportError:
    RandomGaussianBlurImage = None

try:
    from MriWizard.degrade.gamma import RandomGamma
except ImportError:
    RandomGamma = None


# Mapping from config names to MriWizard classes
DEGRADATION_MAP = {
    "identity": Identity,
    "random": RandomUndersample,
    "uniform": UniformUndersample,
    "partial_fourier": PartialFourier,
    "kmax": KmaxUndersample,
    "elliptical": EllipticalUndersample,
    "gaussian_kspace": AddGaussianNoiseKspace,
    "motion": RandomMotionKspace,
    "ghosting": RandomGhostingKspace,
    "spike": RandomSpikeKspace,
    "gibbs": RandomGibbsRinging,
    "biasfield": RandomBiasFieldImage,
    "blur": RandomGaussianBlurImage,
    "gamma": RandomGamma
}


def build_degradation_pipeline(degradation_config: Dict[str, Any]) -> Pipeline:
    """
    Build MriWizard pipeline from degradation config

    Args:
        degradation_config: Loaded from degradation_*.json

    Returns:
        Pipeline object ready for use

    Example:
        >>> config = load_degradation_config('configs/degradation_all.json')
        >>> pipeline = build_degradation_pipeline(config)
        >>> degraded_record = pipeline(record)
    """
    # Check if this is a top-level one_of strategy (multiple pipelines to choose from)
    top_level_strategy = degradation_config.get("strategy", None)
    
    if top_level_strategy == "one_of":
        # Handle multiple complete pipelines with OneOf
        patterns = degradation_config.get("patterns", [])
        pipeline_transforms = []
        weights = []
        
        for pattern in patterns:
            if not pattern.get("enabled", True):
                continue
            
            # Each pattern is a complete pipeline config
            pipeline_config = pattern.get("pipeline", {})
            weight = pattern.get("weight", 1.0)
            
            # Build the complete pipeline for this pattern
            transforms = _build_complete_pipeline(pipeline_config)
            
            if transforms:
                # Wrap all transforms in ApplyAll to keep them together
                pipeline_transforms.append(ApplyAll(transforms))
                weights.append(weight)
        
        if pipeline_transforms:
            # Normalize weights to probabilities
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            # Return OneOf that selects between complete pipelines
            return Pipeline([OneOf(pipeline_transforms, probs=probs)])
        
        return Pipeline([IFFTReconstruct(normalize=True)])
    
    else:
        # Single pipeline (legacy format)
        transforms = _build_complete_pipeline(degradation_config)
        return Pipeline(transforms)


def _build_complete_pipeline(degradation_config: Dict[str, Any]) -> List:
    """Build a complete degradation pipeline from config"""
    transforms = []
    
    # Get execution order
    apply_order = degradation_config.get("execution", {}).get("apply_order", ["noise", "undersampling", "artifacts"])

    for step in apply_order:
        if step == "noise":
            # 1. Add noise (if enabled)
            noise_transform = _build_noise(degradation_config.get("noise", {}))
            if noise_transform:
                transforms.append(noise_transform)

        elif step == "undersampling":
            # 2. Add undersampling
            us_transform = _build_undersampling(degradation_config.get("undersampling", {}))
            if us_transform:
                transforms.append(us_transform)

        elif step == "artifacts":
            # 3. Add artifacts
            artifact_transforms = _build_artifacts(degradation_config.get("artifacts", {}))
            transforms.extend(artifact_transforms)

    # 4. Add reconstruction
    transforms.append(IFFTReconstruct(normalize=True))
    
    return transforms


def _build_noise(noise_config: Dict[str, Any]):
    """Build noise transform from config - supports one_of strategy"""
    if not noise_config:
        return None

    # Check for one_of strategy (new config format)
    strategy = noise_config.get("strategy", "single")

    if strategy == "one_of":
        # Build OneOf with multiple noise patterns
        patterns = noise_config.get("patterns", [])
        noise_transforms = []
        weights = []

        for pattern in patterns:
            if not pattern.get("enabled", True):
                continue

            name = pattern["name"]
            params = _convert_range_params(pattern.get("params", {}))
            weight = pattern.get("weight", 1.0)

            # Get degradation class
            degrade_class = DEGRADATION_MAP.get(name)
            if degrade_class is None:
                print(f"Warning: Unknown noise type '{name}', skipping")
                continue

            # Create transform
            if name == "identity":
                noise_transforms.append(Identity())
            else:
                noise_transforms.append(degrade_class(**params))

            weights.append(weight)

        if noise_transforms:
            return OneOf(noise_transforms, probs=weights)
        return None

    else:
        # Legacy single noise format
        if not noise_config.get("enabled", False):
            return None

        params = {
            "mean": noise_config.get("mean", 0.0),
            "std": tuple(noise_config["std_range"]),
            "relative": noise_config.get("relative", True),
            "reference": noise_config.get("reference", "std")
        }

        return AddGaussianNoiseKspace(**params)


def _build_undersampling(us_config: Dict[str, Any]):
    """Build undersampling transform from config"""
    if not us_config:
        return None

    strategy = us_config.get("strategy", "one_of")
    patterns = us_config.get("patterns", [])

    # Filter enabled patterns
    enabled_patterns = [p for p in patterns if p.get("enabled", True)]

    if not enabled_patterns:
        return None

    if strategy == "one_of":
        # Build OneOf combinator with all enabled patterns
        pattern_transforms = []
        weights = []

        for pattern in enabled_patterns:
            transform = _build_pattern(pattern)
            if transform:
                pattern_transforms.append(transform)
                weights.append(pattern.get("weight", 1.0))

        if not pattern_transforms:
            return None

        # Normalize weights to probabilities
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        return OneOf(pattern_transforms, probs=probs)

    elif strategy == "apply_all":
        # Build ApplyAll combinator
        pattern_transforms = []

        for pattern in enabled_patterns:
            transform = _build_pattern(pattern)
            if transform:
                pattern_transforms.append(transform)

        if not pattern_transforms:
            return None

        return ApplyAll(pattern_transforms)

    else:
        raise ValueError(f"Unknown undersampling strategy: {strategy}")


def _build_pattern(pattern_config: Dict[str, Any]):
    """Build individual undersampling pattern from config"""
    pattern_name = pattern_config["name"]
    params = pattern_config.get("params", {})

    if pattern_name not in DEGRADATION_MAP:
        print(f"Warning: Unknown pattern type '{pattern_name}', skipping")
        return None

    pattern_class = DEGRADATION_MAP[pattern_name]

    if pattern_class is None:
        print(f"Warning: Pattern '{pattern_name}' not available in MriWizard, skipping")
        return None

    try:
        # Convert range parameters to tuples if needed
        params = _convert_range_params(params)
        return pattern_class(**params)
    except Exception as e:
        print(f"Error creating pattern '{pattern_name}': {e}")
        return None


def _build_artifacts(artifacts_config: Dict[str, Any]) -> List:
    """Build artifact transforms from config - supports one_of strategy"""
    if not artifacts_config:
        return []

    # Check for one_of strategy (new config format)
    strategy = artifacts_config.get("strategy", "individual")

    if strategy == "one_of":
        # Build OneOf with multiple artifact patterns
        patterns = artifacts_config.get("patterns", [])
        artifact_transforms = []
        weights = []

        for pattern in patterns:
            if not pattern.get("enabled", True):
                continue

            name = pattern["name"]
            params = _convert_range_params(pattern.get("params", {}))
            weight = pattern.get("weight", 1.0)

            # Get degradation class
            degrade_class = DEGRADATION_MAP.get(name)
            if degrade_class is None:
                print(f"Warning: Unknown artifact type '{name}', skipping")
                continue

            # Create transform
            if name == "identity":
                artifact_transforms.append(Identity())
            else:
                artifact_transforms.append(degrade_class(**params))

            weights.append(weight)

        if artifact_transforms:
            return [OneOf(artifact_transforms, probs=weights)]
        return []

    else:
        # Legacy individual artifact format
        transforms = []

        for artifact_name, artifact_cfg in artifacts_config.items():
            # Skip strategy key
            if artifact_name == "strategy":
                continue

            if not artifact_cfg.get("enabled", False):
                continue

            if artifact_name not in DEGRADATION_MAP:
                print(f"Warning: Unknown artifact type '{artifact_name}', skipping")
                continue

            artifact_class = DEGRADATION_MAP[artifact_name]

            if artifact_class is None:
                print(f"Warning: Artifact '{artifact_name}' not available in MriWizard, skipping")
                continue

            params = artifact_cfg.get("params", {})
            prob = artifact_cfg.get("probability", 1.0)

            try:
                # Convert range parameters
                params = _convert_range_params(params)
                artifact_transform = artifact_class(**params)

                # Wrap in probability if < 1.0
                if prob < 1.0:
                    # Use OneOf with identity for probability
                    transforms.append(
                        OneOf([artifact_transform, Identity()], probs=[prob, 1 - prob])
                    )
                else:
                    transforms.append(artifact_transform)

            except Exception as e:
                print(f"Error creating artifact '{artifact_name}': {e}")

        return transforms


def _convert_range_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert range parameters from lists to tuples

    MriWizard expects tuples for range parameters like:
    - prob_range, R_range, std_range, etc.

    Special handling:
    - 'axes': If list of multiple values (e.g., [-1, -2]), randomly choose one
    """
    import random

    converted = {}

    for key, value in params.items():
        # Special handling for 'axes' parameter (Gibbs artifact expects tuple)
        if key == 'axes' and isinstance(value, list):
            if len(value) > 1:
                # Randomly choose one axis from the list and wrap in tuple
                converted[key] = (random.choice(value),)
            else:
                # Single value - wrap in tuple
                converted[key] = (value[0],) if len(value) == 1 else tuple(value)
        # Special handling for 'axis' parameter (undersample expects single integer)
        elif key == 'axis' and isinstance(value, list):
            if len(value) > 1:
                # Randomly choose one axis from the list
                converted[key] = random.choice(value)
            else:
                # Single value - extract integer from list
                converted[key] = value[0] if len(value) == 1 else value
        # Parameters that should be tuples (noise std, mean, etc.)
        elif key in ['std', 'mean', 'sigma'] and isinstance(value, list):
            # Convert list to tuple for range parameters
            converted[key] = tuple(value)
        elif key.endswith('_range') or key.endswith('_ranges'):
            if isinstance(value, list):
                # Convert nested lists to nested tuples
                if value and isinstance(value[0], list):
                    converted[key] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
                else:
                    converted[key] = tuple(value)
            else:
                converted[key] = value
        else:
            converted[key] = value

    return converted


def print_pipeline_info(pipeline: Pipeline):
    """
    Print information about the pipeline

    Args:
        pipeline: Pipeline object
    """
    print(f"Pipeline with {len(pipeline.steps)} transforms:")
    for i, transform in enumerate(pipeline.steps, 1):
        transform_name = transform.__class__.__name__
        print(f"  {i}. {transform_name}")


if __name__ == '__main__':
    # Test pipeline building
    import json
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = json.load(f)

        print("="*60)
        print(f"Building pipeline from: {sys.argv[1]}")
        print("="*60)

        pipeline = build_degradation_pipeline(config)
        print_pipeline_info(pipeline)

    else:
        print("Usage: python pipeline_builder.py <degradation_config.json>")
