"""Checkpoint management for model saving and loading."""

import os
import torch
from pathlib import Path
from typing import Union, Dict, Any, Optional


class CheckpointManager:
    """Manage model checkpoints with saving and loading."""
    
    def __init__(self, checkpoint_dir: Union[str, Path], keep_last_n: int = 3):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_history = []
    
    def save(self, state_dict: Dict[str, Any], step: int, is_best: bool = False,
            model_name: str = 'model') -> str:
        """
        Save model checkpoint.
        
        Args:
            state_dict: Dictionary containing model state and metadata
            step: Current training step
            is_best: If True, also save as best model
            model_name: Name for checkpoint file
            
        Returns:
            Path to saved checkpoint
        """
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'{model_name}_step_{step:06d}.pt'
        torch.save(state_dict, checkpoint_path)
        self.checkpoint_history.append(str(checkpoint_path))
        
        # Save as best if specified
        if is_best:
            best_path = self.checkpoint_dir / f'{model_name}_best.pt'
            torch.save(state_dict, best_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def save_full(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 step: int, epoch: int, metrics: Dict[str, float] = None,
                 is_best: bool = False, model_name: str = 'model') -> str:
        """
        Save complete training state including model, optimizer, and scheduler.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            step: Current training step
            epoch: Current epoch
            metrics: Dictionary of evaluation metrics
            is_best: If True, also save as best model
            model_name: Name for checkpoint file
            
        Returns:
            Path to saved checkpoint
        """
        state_dict = {
            'model': model.state_dict() if hasattr(model, 'state_dict') else model,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'step': step,
            'epoch': epoch,
            'metrics': metrics or {}
        }
        
        return self.save(state_dict, step, is_best, model_name)
    
    def load(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded state dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path, map_location='cpu')
    
    def load_to_model(self, model: torch.nn.Module, checkpoint_path: Union[str, Path],
                     strict: bool = True):
        """
        Load model weights from checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint
            strict: If True, require exact match of keys
        """
        checkpoint = self.load(checkpoint_path)
        
        # Extract model state from checkpoint
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            model_state = checkpoint
        
        model.load_state_dict(model_state, strict=strict)
    
    def load_full(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 checkpoint_path: Union[str, Path],
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 strict: bool = True) -> Dict[str, Any]:
        """
        Load complete training state.
        
        Args:
            model: Model to load
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint
            scheduler: Optional scheduler to load
            strict: If True, require exact match of model keys
            
        Returns:
            Dictionary with 'step', 'epoch', 'metrics', etc.
        """
        checkpoint = self.load(checkpoint_path)
        
        # Load model
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=strict)
        
        # Load optimizer
        if 'optimizer' in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Load scheduler
        if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        # Return metadata
        return {
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }
    
    def get_best_checkpoint(self, model_name: str = 'model') -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / f'{model_name}_best.pt'
        return str(best_path) if best_path.exists() else None
    
    def get_latest_checkpoint(self, model_name: str = 'model') -> Optional[str]:
        """Get path to most recent checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob(f'{model_name}_step_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        return str(checkpoints[-1]) if checkpoints else None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only recent ones."""
        if len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if Path(old_checkpoint).exists():
                os.remove(old_checkpoint)


class EarlyStopping:
    """Early stopping callback based on validation metric."""

    def __init__(self, patience: int = 5, verbose: bool = True,
                 mode: str = 'max', delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of checks without improvement to stop
            verbose: If True, print messages
            mode: 'max' to maximize metric, 'min' to minimize
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'max':
            self.is_better = lambda new, old: new > old + delta
        else:
            self.is_better = lambda new, old: new < old - delta

    def __call__(self, metric: float) -> bool:
        """
        Check if should stop training.

        Args:
            metric: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
        elif self.is_better(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Metric improved to {metric:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} checks")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Training stopped")

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# Backward compatibility: function-based API
def save_checkpoint(path: str, model, optimizer=None, scheduler=None,
                   epoch: int = 0, step: int = 0, metrics: Dict[str, float] = None):
    """
    Save checkpoint (backward compatibility function).

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        metrics: Dictionary of metrics
    """
    state_dict = {
        'model': model.state_dict() if hasattr(model, 'state_dict') else model,
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {}
    }

    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()

    torch.save(state_dict, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, strict: bool = True) -> Dict[str, Any]:
    """
    Load checkpoint (backward compatibility function).

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        strict: If True, require exact match of keys

    Returns:
        Dictionary with 'epoch', 'step', 'metrics', etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location='cpu')

    # Load model
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    # Load optimizer
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    # Load scheduler
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")

    print(f"Checkpoint loaded from {path}")

    # Return metadata
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'metrics': checkpoint.get('metrics', {})
    }
