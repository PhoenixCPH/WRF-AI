"""
Data preprocessing pipeline for WRF NetCDF files
"""
import os
import glob
import numpy as np
import xarray as xr
import dask.array as da
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..config.config import Config


class WRFDataProcessor:
    """Efficient data processor for WRF NetCDF files"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU device
        self.device = torch.device(config.system.device)
        
        # Cache for processed data
        self.data_cache = {}
        
    def load_netcdf_file(self, file_path: str) -> xr.Dataset:
        """Load NetCDF file with Dask for efficient memory usage"""
        try:
            # Use chunks for efficient loading
            chunks = {
                'Time': self.config.data.chunk_size,
                'bottom_top': self.config.data.vertical_levels,
                'south_north': self.config.data.spatial_dims[0],
                'west_east': self.config.data.spatial_dims[1]
            }
            
            ds = xr.open_dataset(file_path, chunks=chunks)
            return ds
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def extract_key_variables(self, ds: xr.Dataset) -> Dict[str, np.ndarray]:
        """Extract key variables from dataset"""
        variables = {}
        
        for var_name in self.config.data.key_variables:
            if var_name in ds.variables:
                var_data = ds[var_name].data
                
                # Convert to numpy if it's a dask array
                if isinstance(var_data, da.Array):
                    var_data = var_data.compute()
                
                # Handle missing values
                if np.ma.is_masked(var_data):
                    var_data = var_data.filled(0.0)
                
                variables[var_name] = var_data
            else:
                self.logger.warning(f"Variable {var_name} not found in dataset")
        
        return variables
    
    def normalize_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize data using variable-specific scaling"""
        # Basic normalization - can be enhanced with variable-specific stats
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        if std > 0:
            normalized = (data - mean) / std
        else:
            normalized = data - mean
        
        return np.nan_to_num(normalized, 0.0)
    
    def preprocess_single_file(self, file_path: str) -> torch.Tensor:
        """Preprocess a single WRF file"""
        try:
            # Load NetCDF file
            ds = self.load_netcdf_file(file_path)
            
            # Extract key variables
            variables = self.extract_key_variables(ds)
            
            # Stack variables into a single tensor
            processed_data = []
            for var_name in self.config.data.key_variables:
                if var_name in variables:
                    var_data = variables[var_name]
                    
                    # Normalize
                    normalized = self.normalize_data(var_data, var_name)
                    
                    # Convert to tensor and move to GPU
                    tensor = torch.from_numpy(normalized).float().to(self.device)
                    processed_data.append(tensor)
            
            # Stack along channel dimension
            if processed_data:
                combined = torch.stack(processed_data, dim=0)  # [channels, time, height, width]
                return combined
            else:
                raise ValueError("No valid variables found")
                
        except Exception as e:
            self.logger.error(f"Error preprocessing {file_path}: {e}")
            raise
    
    def create_data_pairs(self, file_list: List[str]) -> List[Tuple[str, str]]:
        """Create input-output file pairs for sequential prediction"""
        file_pairs = []
        
        # Sort files by date
        sorted_files = sorted(file_list)
        
        # Create consecutive pairs
        for i in range(len(sorted_files) - 1):
            file_pairs.append((sorted_files[i], sorted_files[i + 1]))
        
        return file_pairs
    
    def get_file_list(self) -> List[str]:
        """Get list of WRF files"""
        pattern = os.path.join(self.config.data.data_dir, self.config.data.file_pattern)
        files = glob.glob(pattern)
        return sorted(files)


class WRFDataset(Dataset):
    """PyTorch dataset for WRF data"""
    
    def __init__(self, file_pairs: List[Tuple[str, str]], processor: WRFDataProcessor, 
                 config: Config, cache_size: int = 100):
        self.file_pairs = file_pairs
        self.processor = processor
        self.config = config
        self.cache = {}
        self.cache_size = cache_size
        
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input-output pair"""
        input_file, output_file = self.file_pairs[idx]
        
        # Check cache
        cache_key = f"{input_file}_{output_file}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Process input file
            input_data = self.processor.preprocess_single_file(input_file)
            
            # Process output file
            output_data = self.processor.preprocess_single_file(output_file)
            
            # Ensure same shape
            if input_data.shape != output_data.shape:
                min_time = min(input_data.shape[1], output_data.shape[1])
                input_data = input_data[:, :min_time, :, :]
                output_data = output_data[:, :min_time, :, :]
            
            # Cache result
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = (input_data, output_data)
            
            return input_data, output_data
            
        except Exception as e:
            logging.error(f"Error processing pair {idx}: {e}")
            # Return dummy data
            dummy_shape = (len(self.config.data.key_variables), 
                          self.config.data.time_steps,
                          *self.config.data.spatial_dims)
            dummy_input = torch.zeros(dummy_shape, device=self.processor.device)
            dummy_output = torch.zeros(dummy_shape, device=self.processor.device)
            return dummy_input, dummy_output


class WRFDataLoader:
    """Efficient data loader for WRF data with GPU acceleration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = WRFDataProcessor(config)
        self.logger = logging.getLogger(__name__)
        
    def create_dataloader(self, file_pairs: List[Tuple[str, str]], 
                         shuffle: bool = True) -> DataLoader:
        """Create DataLoader with optimized settings"""
        dataset = WRFDataset(file_pairs, self.processor, self.config)
        
        # Create dataloader with GPU optimization
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.system.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=True,
            drop_last=True
        )
        
        return dataloader
    
    def get_train_val_loaders(self, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders"""
        # Get file list and create pairs
        file_list = self.processor.get_file_list()
        file_pairs = self.processor.create_data_pairs(file_list)
        
        # Split into train and validation
        n_pairs = len(file_pairs)
        n_val = int(n_pairs * val_split)
        n_train = n_pairs - n_val
        
        train_pairs = file_pairs[:n_train]
        val_pairs = file_pairs[n_train:]
        
        self.logger.info(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")
        
        # Create dataloaders
        train_loader = self.create_dataloader(train_pairs, shuffle=True)
        val_loader = self.create_dataloader(val_pairs, shuffle=False)
        
        return train_loader, val_loader
    
    def preprocess_all_data(self, output_dir: str) -> None:
        """Preprocess all data and save to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        file_list = self.processor.get_file_list()
        self.logger.info(f"Preprocessing {len(file_list)} files...")
        
        for i, file_path in enumerate(file_list):
            try:
                # Process file
                processed_data = self.processor.preprocess_single_file(file_path)
                
                # Save processed data
                filename = os.path.basename(file_path).replace('.nc', '_processed.pt')
                output_path = os.path.join(output_dir, filename)
                
                torch.save(processed_data.cpu(), output_path)
                
                if i % 10 == 0:
                    self.logger.info(f"Processed {i+1}/{len(file_list)} files")
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        self.logger.info("Data preprocessing completed!")


# Utility functions
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
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024**3,  # GB
        'vms': memory_info.vms / 1024**3,  # GB
        'percent': process.memory_percent()
    }


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return {
            f'gpu_{i}': {
                'allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved(i) / 1024**3,  # GB
                'max_allocated': torch.cuda.max_memory_allocated(i) / 1024**3  # GB
            }
            for i in range(torch.cuda.device_count())
        }
    return {}