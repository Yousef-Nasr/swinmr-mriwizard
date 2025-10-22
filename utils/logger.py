"""Training logger with TensorBoard and file logging."""

import os
import logging
from pathlib import Path
from typing import Union, Dict, Any
from datetime import datetime


class TrainingLogger:
    """Logger for training with file and TensorBoard support."""
    
    def __init__(self, log_dir: Union[str, Path], experiment_name: str = None):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment for subfolder
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # File logging
        self.log_file = self.exp_dir / 'training.log'
        self._setup_file_logger()
        
        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(str(self.exp_dir / 'tensorboard'))
            self.logger.info(f"TensorBoard writer initialized at {self.exp_dir / 'tensorboard'}")
        except ImportError:
            self.tb_writer = None
            self.logger.warning("TensorBoard not available. Install: pip install tensorboard")
    
    def _setup_file_logger(self):
        """Setup file-based logging."""
        self.logger = logging.getLogger('SwinMR_Training')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log scalar to TensorBoard and console.
        
        Args:
            tag: Metric name
            value: Metric value
            step: Training step/epoch
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tags_values: Dict[str, float], step: int):
        """
        Log multiple scalars.
        
        Args:
            tags_values: Dictionary of {tag: value}
            step: Training step/epoch
        """
        if self.tb_writer is not None:
            for tag, value in tags_values.items():
                self.tb_writer.add_scalar(tag, value, step)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.info("=" * 80)
        self.info("TRAINING CONFIGURATION")
        self.info("=" * 80)
        
        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    self.info("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                else:
                    self.info("  " * indent + f"{key}: {value}")
        
        print_dict(config)
        self.info("=" * 80)
    
    def log_training_summary(self, epoch: int, step: int, metrics: Dict[str, float], 
                            learning_rate: float):
        """
        Log training summary for current step.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics {name: value}
            learning_rate: Current learning rate
        """
        msg = f"Epoch {epoch} | Step {step} | LR {learning_rate:.2e} | "
        msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(msg)
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Learning_Rate', learning_rate, step)
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f'Training/{k}', v, step)
    
    def log_validation_summary(self, epoch: int, step: int, metrics: Dict[str, float]):
        """
        Log validation summary.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics {name: value}
        """
        msg = f"[VAL] Epoch {epoch} | Step {step} | "
        msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(msg)
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f'Validation/{k}', v, step)
    
    def log_model_info(self, model_name: str, num_params: int, trainable_params: int):
        """
        Log model information.
        
        Args:
            model_name: Name of model
            num_params: Total parameters
            trainable_params: Trainable parameters
        """
        self.info("=" * 80)
        self.info(f"Model: {model_name}")
        self.info(f"Total Parameters: {num_params:,}")
        self.info(f"Trainable Parameters: {trainable_params:,}")
        self.info("=" * 80)
    
    def flush(self):
        """Flush TensorBoard writer."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
    
    def close(self):
        """Close logger."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        self.info("Logger closed")


class MetricsTracker:
    """Track and compute running statistics of metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
    
    def update(self, **kwargs):
        """
        Update metrics.
        
        Args:
            **kwargs: metric_name=value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(float(value))
    
    def get_means(self) -> Dict[str, float]:
        """
        Get mean values of all metrics.
        
        Returns:
            Dictionary of {name: mean}
        """
        return {name: sum(values) / len(values) 
                for name, values in self.metrics.items() if len(values) > 0}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
    
    def get_last(self, name: str) -> float:
        """Get last value of a metric."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]
        return 0.0


# Backward compatibility function
def setup_logger(log_dir: Union[str, Path], experiment_name: str = None) -> TrainingLogger:
    """
    Setup logger (backward compatibility function).

    Args:
        log_dir: Directory to save logs
        experiment_name: Name of experiment

    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(log_dir, experiment_name)
