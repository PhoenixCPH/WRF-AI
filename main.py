#!/usr/bin/env python3
"""
Main entry point for WRF diffusion model
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config
from utils.utils import setup_logging, get_system_info, check_system_requirements


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='WRF Diffusion Model')
    
    # Main command
    parser.add_argument('command', choices=['train', 'preprocess', 'evaluate', 'info'],
                       help='Command to execute')
    
    # Global options
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing WRF data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default='logs/main.log',
                       help='Log file path')
    
    return parser.parse_known_args()


def main():
    """Main entry point"""
    args, unknown_args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"WRF Diffusion Model - {args.command}")
    logger.info(f"Arguments: {args}")
    
    if args.command == 'info':
        # Show system information
        print("=== System Information ===")
        
        system_info = get_system_info()
        for key, value in system_info.items():
            print(f"{key}: {value}")
        
        print("\n=== System Requirements ===")
        requirements = check_system_requirements()
        for req_name, req_met in requirements.items():
            status = "✓" if req_met else "✗"
            print(f"{status} {req_name}: {req_met}")
        
        return
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = Config()
    
    # Override configuration
    config.data.data_dir = args.data_dir
    config.system.device = args.device
    config.system.log_level = args.log_level
    config.system.log_file = args.log_file
    
    # Execute command
    if args.command == 'preprocess':
        # Run preprocessing
        from scripts.preprocessing.preprocess_data import main as preprocess_main
        
        # Build preprocessing arguments
        preprocess_args = [
            '--data_dir', args.data_dir,
            '--output_dir', os.path.join(args.output_dir, 'data', 'processed'),
            '--device', args.device,
            '--log_level', args.log_level,
            '--log_file', args.log_file.replace('.log', '_preprocess.log')
        ]
        
        # Add unknown arguments
        preprocess_args.extend(unknown_args)
        
        # Update sys.argv for subprocess
        sys.argv = ['preprocess_data.py'] + preprocess_args
        
        preprocess_main()
    
    elif args.command == 'train':
        # Run training
        from scripts.training.train import main as train_main
        
        # Build training arguments
        train_args = [
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--device', args.device,
            '--log_level', args.log_level,
            '--log_file', args.log_file.replace('.log', '_train.log')
        ]
        
        # Add unknown arguments
        train_args.extend(unknown_args)
        
        # Update sys.argv for subprocess
        sys.argv = ['train.py'] + train_args
        
        train_main()
    
    elif args.command == 'evaluate':
        # Run evaluation
        from scripts.evaluation.evaluate import main as evaluate_main
        
        # Build evaluation arguments
        evaluate_args = [
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--device', args.device,
            '--log_level', args.log_level,
            '--log_file', args.log_file.replace('.log', '_evaluate.log')
        ]
        
        # Add unknown arguments
        evaluate_args.extend(unknown_args)
        
        # Update sys.argv for subprocess
        sys.argv = ['evaluate.py'] + evaluate_args
        
        evaluate_main()
    
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()