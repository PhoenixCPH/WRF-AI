#!/usr/bin/env python3
"""
Test script to verify WRF diffusion model setup
"""
import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        from config.config import Config
        print("✓ Config import successful")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from models.diffusion_model import WRFDiffusionModel
        print("✓ Model import successful")
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False
    
    try:
        from data.preprocessing import WRFDataProcessor, WRFDataLoader
        print("✓ Data processing import successful")
    except Exception as e:
        print(f"✗ Data processing import failed: {e}")
        return False
    
    try:
        from training.trainer import WRFTrainer
        print("✓ Training import successful")
    except Exception as e:
        print(f"✗ Training import failed: {e}")
        return False
    
    try:
        from utils.utils import get_system_info, check_system_requirements
        print("✓ Utils import successful")
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        config = Config()
        print("✓ Configuration created successfully")
        print(f"  - Model channels: {config.model.channels}")
        print(f"  - Data batch size: {config.data.batch_size}")
        print(f"  - Training epochs: {config.training.epochs}")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def test_model():
    """Test model creation"""
    print("\nTesting model...")
    
    try:
        config = Config()
        model = WRFDiffusionModel(config)
        print("✓ Model created successfully")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        channels = config.model.in_channels
        time_steps = 10
        height, width = config.data.spatial_dims
        
        x = torch.randn(batch_size, channels, time_steps, height, width)
        t = torch.randint(0, config.model.num_steps, (batch_size,))
        
        with torch.no_grad():
            loss = model(x, t)
        
        print(f"✓ Forward pass successful, loss: {loss.item():.6f}")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_system_info():
    """Test system information"""
    print("\nTesting system information...")
    
    try:
        from utils.utils import get_system_info, check_system_requirements
        
        system_info = get_system_info()
        print("✓ System info retrieved")
        for key, value in system_info.items():
            print(f"  - {key}: {value}")
        
        requirements = check_system_requirements()
        print("✓ System requirements checked")
        for req_name, req_met in requirements.items():
            status = "✓" if req_met else "✗"
            print(f"  {status} {req_name}: {req_met}")
        
        return True
    except Exception as e:
        print(f"✗ System info test failed: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available, version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")
        
        # Test GPU tensor
        try:
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.mm(x, y)
            print("✓ GPU tensor operations successful")
            return True
        except Exception as e:
            print(f"✗ GPU operations failed: {e}")
            return False
    else:
        print("✗ CUDA not available")
        return False

def main():
    """Run all tests"""
    print("=== WRF Diffusion Model Setup Test ===")
    
    tests = [
        test_imports,
        test_config,
        test_model,
        test_system_info,
        test_gpu
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Setup is complete.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    exit(main())