"""
Configuration Loading and Validation Module

Handles loading and validating both main training config and degradation config.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load main training configuration from JSON file

    Args:
        config_path: Path to training config JSON file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required sections
    required_sections = ['experiment_name', 'paths', 'data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate paths exist for data directories (create if needed)
    if 'train_dir' in config['data']:
        Path(config['data']['train_dir']).mkdir(parents=True, exist_ok=True)
    if 'val_dir' in config['data']:
        Path(config['data']['val_dir']).mkdir(parents=True, exist_ok=True)

    # Make paths absolute if relative
    if 'degradation_config' in config['data']:
        deg_config_path = Path(config['data']['degradation_config'])
        if not deg_config_path.is_absolute():
            # Make relative to the training config file
            deg_config_path = config_path.parent / deg_config_path
            config['data']['degradation_config'] = str(deg_config_path)

    return config


def load_degradation_config(config_path: str) -> Dict[str, Any]:
    """
    Load degradation configuration from JSON file

    Args:
        config_path: Path to degradation config JSON file

    Returns:
        Dictionary containing degradation configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Degradation config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required sections
    if 'name' not in config:
        config['name'] = 'unnamed_degradation'

    # Validate noise config if enabled
    if config.get('noise', {}).get('enabled', False):
        noise_cfg = config['noise']
        if 'std_range' not in noise_cfg:
            raise ValueError("Noise config missing 'std_range'")
        if len(noise_cfg['std_range']) != 2:
            raise ValueError("'std_range' must be a 2-element list [min, max]")

    # Validate undersampling config
    if 'undersampling' in config:
        us_cfg = config['undersampling']
        if 'patterns' not in us_cfg:
            raise ValueError("Undersampling config missing 'patterns'")
        if 'strategy' not in us_cfg:
            us_cfg['strategy'] = 'one_of'  # Default strategy

        # Validate each pattern
        for pattern in us_cfg['patterns']:
            if 'name' not in pattern:
                raise ValueError("Undersampling pattern missing 'name'")
            if 'params' not in pattern:
                raise ValueError(f"Undersampling pattern '{pattern['name']}' missing 'params'")

    return config


def validate_config(config: Dict[str, Any], config_type: str = 'training') -> bool:
    """
    Validate configuration dictionary

    Args:
        config: Configuration dictionary
        config_type: Type of config ('training' or 'degradation')

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    if config_type == 'training':
        # Check data paths exist
        if 'train_dir' in config.get('data', {}):
            train_dir = Path(config['data']['train_dir'])
            if not train_dir.exists():
                print(f"Warning: Training directory does not exist: {train_dir}")

        # Check degradation config exists
        if 'degradation_config' in config.get('data', {}):
            deg_path = Path(config['data']['degradation_config'])
            if not deg_path.exists():
                raise FileNotFoundError(f"Degradation config not found: {deg_path}")

        # Validate model parameters
        model_cfg = config.get('model', {})
        if model_cfg.get('in_chans', 1) != 1:
            print("Warning: Model expects 1 input channel for MRI reconstruction")

        # Validate training parameters
        train_cfg = config.get('training', {})
        if train_cfg.get('epochs', 0) <= 0:
            raise ValueError("Training epochs must be > 0")

    elif config_type == 'degradation':
        # Validate degradation config structure
        if 'noise' in config and config['noise'].get('enabled', False):
            std_range = config['noise'].get('std_range')
            if std_range and std_range[0] >= std_range[1]:
                raise ValueError("Noise std_range[0] must be < std_range[1]")

        # Validate undersampling patterns
        if 'undersampling' in config:
            patterns = config['undersampling'].get('patterns', [])
            enabled_patterns = [p for p in patterns if p.get('enabled', True)]
            if not enabled_patterns:
                raise ValueError("At least one undersampling pattern must be enabled")

    return True


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries (deep merge)

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def parse_cli_overrides(args: list) -> Dict[str, Any]:
    """
    Parse command-line override arguments

    Format: --data.batch_size 16 --training.epochs 50

    Args:
        args: List of command-line arguments

    Returns:
        Dictionary of overrides
    """
    overrides = {}

    i = 0
    while i < len(args):
        if args[i].startswith('--') and i + 1 < len(args):
            key = args[i][2:]  # Remove '--'
            value = args[i + 1]

            # Parse nested keys (e.g., 'data.batch_size')
            keys = key.split('.')
            current = overrides
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Try to convert value to appropriate type
            try:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string

            current[keys[-1]] = value
            i += 2
        else:
            i += 1

    return overrides


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration

    Args:
        config: Configuration dictionary
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(' ' * indent + f"{key}: {value}")


if __name__ == '__main__':
    # Test loading configs
    import sys

    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
        print("="*60)
        print("Training Configuration:")
        print("="*60)
        print_config(config)

        if 'degradation_config' in config.get('data', {}):
            print("\n" + "="*60)
            print("Degradation Configuration:")
            print("="*60)
            deg_config = load_degradation_config(config['data']['degradation_config'])
            print_config(deg_config)
    else:
        print("Usage: python config_loader.py <config_path>")
