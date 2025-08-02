# WRF Diffusion Model

A deep learning model for predicting WRF (Weather Research and Forecasting) data using diffusion models.

## Overview

This project implements a 3D U-Net based diffusion model for predicting future WRF-Chem model outputs based on previous inputs. The model is designed to handle the complex multi-variable nature of WRF data including meteorological, chemical, and radiation variables.

## Features

- **3D U-Net Architecture**: Custom 3D convolutional neural network for spatio-temporal data
- **Diffusion Model**: State-of-the-art generative modeling approach
- **GPU Optimization**: Optimized for NVIDIA A800 GPUs with mixed precision training
- **Memory Efficient**: Handles large 14TB datasets efficiently
- **Multi-variable Support**: Processes 17 key WRF variables simultaneously
- **Time-aware Modeling**: Sequential prediction with proper temporal dependencies

## Data Structure

The model processes WRF NetCDF files with the following structure:
- **Dimensions**: 720 time steps × 64 × 64 spatial grid × 45 vertical levels
- **Key Variables**: 17 selected variables including meteorological and chemical species
- **File Size**: Individual files are several GB each, total dataset ~14TB

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure CUDA 11.8 is installed and compatible with your system

## Usage

### Data Preprocessing

```bash
python scripts/preprocessing/preprocess_data.py \
    --data_dir /path/to/wrf/data \
    --output_dir /path/to/processed/data \
    --num_workers 8 \
    --validate_files
```

### Training

```bash
python scripts/training/train.py \
    --data_dir /path/to/processed/data \
    --epochs 1000 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --device cuda
```

### Evaluation

```bash
python scripts/evaluation/evaluate.py \
    --model_path /path/to/checkpoint.pth \
    --data_dir /path/to/test/data \
    --num_samples 50 \
    --save_visualizations
```

## Configuration

The project uses a configuration system with the following key components:

- **DataConfig**: Data loading and preprocessing parameters
- **ModelConfig**: Neural network architecture settings
- **TrainingConfig**: Training optimization parameters
- **SystemConfig**: System and GPU settings

## Model Architecture

### 3D U-Net
- **Encoder**: 4 levels with increasing channel dimensions
- **Attention Blocks**: Self-attention mechanisms at higher levels
- **Decoder**: Skip connections with upsampling
- **Time Embedding**: Sinusoidal time encoding for diffusion process

### Diffusion Process
- **Forward Process**: Gradual noise addition over 1000 steps
- **Reverse Process**: Learned denoising with U-Net
- **Sampling**: Iterative generation from noise

## Performance Optimization

### GPU Utilization
- Mixed precision training (FP16)
- Gradient accumulation
- Memory-efficient data loading
- Multi-GPU support

### Memory Management
- Chunked data processing
- Smart caching
- Periodic memory cleanup
- Out-of-core data handling

## Key Variables

The model focuses on 17 key WRF variables:
- **Meteorological**: T, U, V, W, PH, P, QVAPOR
- **Chemical**: O3, H2O2, NO, NO2, SO2, CO, HCHO
- **Optical**: AOD_OUT, AOD2D_OUT

## Monitoring

The system includes comprehensive monitoring:
- Training loss and metrics
- Memory usage tracking
- GPU utilization monitoring
- Automatic checkpointing
- Detailed logging

## Error Handling

The implementation includes robust error handling:
- Graceful degradation on missing data
- Automatic retry mechanisms
- Comprehensive logging
- Checkpoint recovery

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyTorch and modern deep learning tools
- Optimized for NVIDIA A800 GPUs
- Designed for large-scale climate and weather modeling