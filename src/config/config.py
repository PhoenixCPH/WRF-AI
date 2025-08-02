"""
Configuration file for WRF Diffusion Model
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class DataConfig:
    """Data processing configuration"""
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    file_pattern: str = "wrfout_d01_*.nc"
    
    # Data dimensions
    time_steps: int = 720
    spatial_dims: Tuple[int, int] = (64, 64)
    vertical_levels: int = 45
    
    # Key variables for training
    key_variables: List[str] = field(default_factory=lambda: [
        'AOD_OUT', 'o3', 'h2o2', 'no', 'no2', 'so2', 'co', 'hcho',
        'T', 'U', 'V', 'W', 'PH', 'PHB', 'P', 'PB', 'QVAPOR'
    ])
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 8
    prefetch_factor: int = 4
    
    # Memory management
    memory_limit_gb: float = 64.0
    chunk_size: int = 100
    cache_size: int = 1000


@dataclass
class ModelConfig:
    """Diffusion model configuration"""
    # Model architecture
    model_type: str = "unet_3d"
    in_channels: int = 17  # Number of key variables
    out_channels: int = 17
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    attention_levels: List[int] = field(default_factory=lambda: [2, 3])
    
    # Diffusion process
    num_steps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Time embedding
    time_embed_dim: int = 256
    
    # Conditioning
    use_conditioning: bool = True
    conditioning_dim: int = 64


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    
    # Optimizer
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    
    # GPU optimization
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 4
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # Distributed training
    distributed: bool = True
    world_size: int = 1
    rank: int = 0


@dataclass
class SystemConfig:
    """System configuration"""
    # GPU settings
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Memory settings
    gpu_memory_fraction: float = 0.9
    pin_memory: bool = True
    
    # Parallel processing
    num_workers: int = 8
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/training.log"
    
    # Monitoring
    monitor_gpu: bool = True
    monitor_memory: bool = True


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Project settings
    project_name: str = "wrf_diffusion"
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set random seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        # Create directories
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.log_file.rsplit('/', 1)[0], exist_ok=True)
        os.makedirs(self.data.processed_dir, exist_ok=True)
        
        # Validate GPU availability
        if self.system.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.system.device = "cpu"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


# Default configuration
default_config = Config()