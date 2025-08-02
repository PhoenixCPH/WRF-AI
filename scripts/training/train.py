"""
Main training script for WRF diffusion model
"""
import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.config import Config
from training.trainer import WRFTrainer
from data.preprocessing import WRFDataLoader
from utils.utils import (
    setup_logging, get_system_info, check_system_requirements, 
    create_directory_structure, save_config, cleanup_memory
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train WRF diffusion model')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing WRF data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='unet_3d',
                       help='Model architecture')
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of diffusion steps')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       help='Mixed precision training (fp16/fp32)')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    # Preprocessing
    parser.add_argument('--preprocess_only', action='store_true',
                       help='Only preprocess data, do not train')
    parser.add_argument('--force_preprocess', action='store_true',
                       help='Force preprocessing even if processed data exists')
    
    # Validation
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default='logs/training.log',
                       help='Log file path')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting WRF diffusion model training")
    logger.info(f"Arguments: {args}")
    
    # Check system requirements
    logger.info("Checking system requirements...")
    requirements = check_system_requirements()
    
    for req_name, req_met in requirements.items():
        status = "✓" if req_met else "✗"
        logger.info(f"{status} {req_name}: {req_met}")
    
    if not all(requirements.values()):
        logger.warning("Some system requirements are not met")
    
    # Get system information
    system_info = get_system_info()
    logger.info(f"System info: {system_info}")
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = Config()
    
    # Override configuration with command line arguments
    config.training.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.model.model_type = args.model_type
    config.model.num_steps = args.num_steps
    config.system.device = args.device
    config.data.num_workers = args.num_workers
    config.training.mixed_precision = args.mixed_precision
    config.training.checkpoint_dir = args.checkpoint_dir
    config.system.log_level = args.log_level
    config.system.log_file = args.log_file
    
    # Update data directory
    if args.data_dir:
        config.data.data_dir = args.data_dir
    
    # Save configuration
    save_config(config, os.path.join(args.output_dir, 'config.json'))
    
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = WRFDataLoader(config)
        
        # Preprocess data if needed
        if args.preprocess_only or args.force_preprocess:
            logger.info("Preprocessing data...")
            processed_dir = os.path.join(args.output_dir, 'data', 'processed')
            data_loader.processor.preprocess_all_data(processed_dir)
            
            if args.preprocess_only:
                logger.info("Preprocessing completed. Exiting.")
                return
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = WRFTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Generate samples
        logger.info("Generating samples...")
        trainer.generate_samples(num_samples=10)
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_memory()


if __name__ == "__main__":
    main()