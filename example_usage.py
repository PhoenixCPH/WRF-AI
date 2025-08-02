#!/usr/bin/env python3
"""
Example usage of WRF diffusion model
"""
import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config
from models.diffusion_model import WRFDiffusionModel
from utils.utils import setup_logging

def setup_example():
    """Setup example configuration"""
    config = Config()
    
    # Use smaller dimensions for example
    config.data.spatial_dims = (32, 32)
    config.data.vertical_levels = 10
    config.data.time_steps = 100
    
    # Use smaller model for example
    config.model.channels = [32, 64, 128]
    config.model.num_steps = 100
    
    # Use smaller batch size
    config.data.batch_size = 2
    
    return config

def create_synthetic_data(config):
    """Create synthetic WRF-like data for demonstration"""
    batch_size = config.data.batch_size
    channels = config.model.in_channels
    time_steps = config.data.time_steps
    height, width = config.data.spatial_dims
    levels = config.data.vertical_levels
    
    # Create synthetic data with realistic patterns
    data = torch.zeros(batch_size, channels, time_steps, height, width)
    
    for b in range(batch_size):
        for c in range(channels):
            for t in range(time_steps):
                # Create spatial patterns
                x = torch.linspace(0, 4*np.pi, width)
                y = torch.linspace(0, 4*np.pi, height)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                # Add time evolution
                time_factor = torch.sin(2 * np.pi * t / time_steps)
                
                # Create different patterns for different channels
                if c < 5:  # Meteorological variables
                    pattern = torch.sin(X + time_factor) * torch.cos(Y + time_factor)
                elif c < 10:  # Chemical variables
                    pattern = torch.sin(X + Y + time_factor) * torch.exp(-0.1 * (X**2 + Y**2))
                else:  # Optical variables
                    pattern = torch.cos(X - Y + time_factor) * torch.sin(0.5 * X + 0.5 * Y)
                
                # Add noise
                noise = 0.1 * torch.randn(height, width)
                
                data[b, c, t] = pattern + noise
    
    return data

def main():
    """Run example usage"""
    print("=== WRF Diffusion Model Example ===")
    
    # Setup logging
    setup_logging("logs/example.log", "INFO")
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = setup_example()
    logger.info("Configuration created")
    
    # Create model
    logger.info("Creating model...")
    model = WRFDiffusionModel(config)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create synthetic data
    logger.info("Creating synthetic data...")
    data = create_synthetic_data(config)
    print(f"Created synthetic data with shape: {data.shape}")
    
    # Test diffusion process
    logger.info("Testing diffusion process...")
    
    # Forward diffusion
    t = torch.randint(0, config.model.num_steps, (config.data.batch_size,))
    noisy_data, noise = model.diffusion.forward(data, t)
    
    print(f"Forward diffusion successful")
    print(f"  - Original data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  - Noisy data range: [{noisy_data.min():.3f}, {noisy_data.max():.3f}]")
    
    # Reverse diffusion (prediction)
    logger.info("Testing reverse diffusion...")
    with torch.no_grad():
        predicted_noise = model.unet(noisy_data, t)
        predicted_data = model.diffusion.reverse_process(noisy_data, t)
    
    print(f"Reverse diffusion successful")
    print(f"  - Predicted data range: [{predicted_data.min():.3f}, {predicted_data.max():.3f}]")
    
    # Test next step prediction
    logger.info("Testing next step prediction...")
    with torch.no_grad():
        next_step_prediction = model.predict_next_step(data, num_steps=20)
    
    print(f"Next step prediction successful")
    print(f"  - Prediction shape: {next_step_prediction.shape}")
    print(f"  - Prediction range: [{next_step_prediction.min():.3f}, {next_step_prediction.max():.3f}]")
    
    # Calculate some metrics
    logger.info("Calculating metrics...")
    
    # MSE between original and predicted
    mse = torch.mean((data - next_step_prediction) ** 2)
    print(f"MSE between original and prediction: {mse.item():.6f}")
    
    # Noise prediction accuracy
    noise_mse = torch.mean((noise - predicted_noise) ** 2)
    print(f"Noise prediction MSE: {noise_mse.item():.6f}")
    
    # Save example outputs
    logger.info("Saving example outputs...")
    
    os.makedirs("examples", exist_ok=True)
    
    # Save data
    torch.save({
        'original_data': data,
        'noisy_data': noisy_data,
        'predicted_data': predicted_data,
        'next_step_prediction': next_step_prediction,
        'noise': noise,
        'predicted_noise': predicted_noise
    }, "examples/example_data.pt")
    
    # Save configuration
    import json
    config_dict = {}
    for key, value in config.__dict__.items():
        if hasattr(value, '__dict__'):
            config_dict[key] = value.__dict__
        else:
            config_dict[key] = value
    
    with open("examples/example_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("Example completed successfully!")
    print("Check the 'examples' directory for saved outputs.")
    
    # Print usage instructions
    print("\n=== Usage Instructions ===")
    print("1. To preprocess data:")
    print("   python main.py preprocess --data_dir /path/to/wrf/data --output_dir outputs")
    print()
    print("2. To train the model:")
    print("   python main.py train --data_dir /path/to/processed/data --output_dir outputs")
    print()
    print("3. To evaluate the model:")
    print("   python main.py evaluate --model_path /path/to/model.pth --data_dir /path/to/test/data")
    print()
    print("4. To check system info:")
    print("   python main.py info")

if __name__ == "__main__":
    main()