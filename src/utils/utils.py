"""
Utility functions for WRF diffusion model
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import psutil
import GPUtil


def setup_logging(log_file: str, level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024**3,  # GB
        'vms': memory_info.vms / 1024**3,  # GB
        'percent': process.memory_percent()
    }


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage"""
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = {}
        
        for i, gpu in enumerate(gpus):
            gpu_info[f'gpu_{i}'] = {
                'name': gpu.name,
                'memory_used': gpu.memoryUsed / 1024,  # GB
                'memory_total': gpu.memoryTotal / 1024,  # GB
                'memory_util': gpu.memoryUtil * 100,  # %
                'load': gpu.load * 100,  # %
                'temperature': gpu.temperature  # °C
            }
        
        return gpu_info
    except Exception as e:
        logging.warning(f"Failed to get GPU info: {e}")
        return {}


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
        'disk_usage': psutil.disk_usage('/').total / 1024**3,  # GB
        'python_version': os.sys.version,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


def save_training_plot(metrics: Dict[str, List[float]], save_path: str) -> None:
    """Save training metrics plot"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(metrics['train_losses'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Validation loss
    if metrics['val_losses']:
        axes[0, 1].plot(metrics['val_losses'])
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(metrics['learning_rates'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Memory usage
    if 'memory_usage' in metrics:
        memory_data = metrics['memory_usage']
        epochs = list(range(len(memory_data)))
        axes[1, 1].plot(epochs, [m['rss'] for m in memory_data], label='RSS')
        axes[1, 1].plot(epochs, [m['vms'] for m in memory_data], label='VMS')
        axes[1, 1].set_title('Memory Usage')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_sample_visualization(samples: Dict[str, torch.Tensor], save_path: str) -> None:
    """Save sample visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input
    input_data = samples['input'][0, 0, -1].cpu().numpy()  # First batch, first channel, last time step
    im1 = axes[0].imshow(input_data, cmap='viridis')
    axes[0].set_title('Input')
    plt.colorbar(im1, ax=axes[0])
    
    # Prediction
    pred_data = samples['prediction'][0, 0, -1].cpu().numpy()
    im2 = axes[1].imshow(pred_data, cmap='viridis')
    axes[1].set_title('Prediction')
    plt.colorbar(im2, ax=axes[1])
    
    # Target
    target_data = samples['target'][0, 0, -1].cpu().numpy()
    im3 = axes[2].imshow(target_data, cmap='viridis')
    axes[2].set_title('Target')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    # Move to CPU for numpy operations
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Remove NaN values
    mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    if len(pred_flat) == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    
    # Calculate metrics
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    # R² score
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2)
    }


def save_config(config, save_path: str) -> None:
    """Save configuration to file"""
    config_dict = {}
    
    # Convert dataclass to dict
    for key, value in config.__dict__.items():
        if hasattr(value, '__dict__'):
            config_dict[key] = value.__dict__
        else:
            config_dict[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_directory_structure(base_path: str) -> None:
    """Create directory structure for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'checkpoints',
        'logs',
        'results',
        'scripts',
        'src/models',
        'src/data',
        'src/training',
        'src/utils',
        'src/config'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)


def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements"""
    requirements = {
        'python_version': os.sys.version_info >= (3, 8),
        'torch_available': torch.__version__ is not None,
        'cuda_available': torch.cuda.is_available(),
        'gpu_memory': False,
        'disk_space': False,
        'ram': False
    }
    
    # Check GPU memory
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            requirements['gpu_memory'] = gpu_memory >= 16  # Minimum 16GB
        except:
            requirements['gpu_memory'] = False
    
    # Check disk space
    disk_usage = psutil.disk_usage('/')
    requirements['disk_space'] = disk_usage.free / 1024**3 >= 100  # Minimum 100GB
    
    # Check RAM
    memory = psutil.virtual_memory()
    requirements['ram'] = memory.total / 1024**3 >= 32  # Minimum 32GB
    
    return requirements


def monitor_resources(interval: int = 60) -> None:
    """Monitor system resources"""
    import time
    
    logging.info("Starting resource monitoring...")
    
    try:
        while True:
            # Get resource usage
            memory = get_memory_usage()
            gpu_memory = get_gpu_memory_usage()
            
            # Log resources
            logging.info(f"Memory: RSS={memory['rss']:.2f}GB, VMS={memory['vms']:.2f}GB, Percent={memory['percent']:.1f}%")
            
            for gpu_id, gpu_info in gpu_memory.items():
                logging.info(f"{gpu_id}: {gpu_info['memory_used']:.2f}GB/{gpu_info['memory_total']:.2f}GB "
                           f"({gpu_info['memory_util']:.1f}%) Load={gpu_info['load']:.1f}% "
                           f"Temp={gpu_info['temperature']:.0f}°C")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("Resource monitoring stopped")


def cleanup_memory() -> None:
    """Clean up memory"""
    import gc
    
    # Garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logging.info("Memory cleanup completed")


def get_file_size(file_path: str) -> float:
    """Get file size in GB"""
    try:
        return os.path.getsize(file_path) / 1024**3
    except:
        return 0.0


def validate_netcdf_file(file_path: str) -> bool:
    """Validate NetCDF file"""
    try:
        import netCDF4 as nc
        
        # Try to open file
        ds = nc.Dataset(file_path, 'r')
        
        # Check basic structure
        if 'Time' not in ds.dimensions:
            return False
        
        # Close file
        ds.close()
        
        return True
    except Exception as e:
        logging.error(f"Error validating {file_path}: {e}")
        return False


def create_batch_generator(data_loader: torch.utils.data.Dataataloader, 
                          batch_size: int, 
                          shuffle: bool = True):
    """Create batch generator for custom batching"""
    while True:
        if shuffle:
            indices = torch.randperm(len(data_loader.dataset))
        else:
            indices = torch.arange(len(data_loader.dataset))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield [data_loader.dataset[idx] for idx in batch_indices]


class EarlyStopping:
    """Early stopping implementation"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if early stopping should occur"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop