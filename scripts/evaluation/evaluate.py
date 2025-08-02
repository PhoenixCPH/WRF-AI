"""
Evaluation script for WRF diffusion model
"""
import os
import sys
import argparse
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.config import Config
from models.diffusion_model import WRFDiffusionModel
from data.preprocessing import WRFDataLoader, setup_logging
from utils.utils import (
    calculate_metrics, save_sample_visualization, 
    save_training_plot, get_memory_usage, get_gpu_memory_usage
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate WRF diffusion model')
    
    # Model paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to configuration file')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing WRF data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Prediction parameters
    parser.add_argument('--prediction_steps', type=int, default=50,
                       help='Number of diffusion steps for prediction')
    parser.add_argument('--t_start', type=int, default=None,
                       help='Starting timestep for prediction')
    
    # Visualization
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--save_samples', action='store_true',
                       help='Save sample predictions')
    
    # Metrics
    parser.add_argument('--calculate_metrics', action='store_true',
                       help='Calculate evaluation metrics')
    parser.add_argument('--metrics_file', type=str, default='evaluation_metrics.json',
                       help='File to save metrics')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default='logs/evaluation.log',
                       help='Log file path')
    
    return parser.parse_args()


def load_model(model_path: str, config: Config, device: torch.device) -> WRFDiffusionModel:
    """Load trained model"""
    logger = logging.getLogger(__name__)
    
    # Initialize model
    model = WRFDiffusionModel(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def evaluate_model(model: WRFDiffusionModel, data_loader: torch.utils.data.DataLoader, 
                  config: Config, args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate model on test data"""
    logger = logging.getLogger(__name__)
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch_idx, (input_data, target_data) in enumerate(data_loader):
            if batch_idx >= args.num_samples // args.batch_size:
                break
            
            # Move to device
            input_data = input_data.to(config.system.device)
            target_data = target_data.to(config.system.device)
            
            # Generate prediction
            prediction = model.predict_next_step(
                input_data, 
                num_steps=args.prediction_steps
            )
            
            # Store results
            all_predictions.append(prediction.cpu())
            all_targets.append(target_data.cpu())
            
            # Calculate loss
            t = torch.randint(0, config.model.num_steps, (input_data.shape[0],), 
                            device=config.system.device)
            loss = model(target_data, t)
            all_losses.append(loss.item())
            
            logger.info(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
            
            # Clear cache
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = {
        'mean_loss': np.mean(all_losses),
        'std_loss': np.std(all_losses)
    }
    
    if args.calculate_metrics:
        # Calculate detailed metrics
        detailed_metrics = calculate_metrics(all_predictions, all_targets)
        metrics.update(detailed_metrics)
        
        logger.info(f"Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.6f}")
    
    return metrics, all_predictions, all_targets


def generate_predictions(model: WRFDiffusionModel, data_loader: torch.utils.data.DataLoader,
                        config: Config, args: argparse.Namespace) -> List[Dict[str, torch.Tensor]]:
    """Generate predictions for visualization"""
    logger = logging.getLogger(__name__)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (input_data, target_data) in enumerate(data_loader):
            if batch_idx >= 10:  # Limit to 10 batches for visualization
                break
            
            # Move to device
            input_data = input_data.to(config.system.device)
            target_data = target_data.to(config.system.device)
            
            # Generate prediction
            prediction = model.predict_next_step(
                input_data, 
                num_steps=args.prediction_steps
            )
            
            predictions.append({
                'input': input_data.cpu(),
                'prediction': prediction.cpu(),
                'target': target_data.cpu()
            })
            
            logger.info(f"Generated prediction batch {batch_idx}")
    
    return predictions


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting WRF diffusion model evaluation")
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    if args.config_path:
        logger.info(f"Loading configuration from {args.config_path}")
        config = Config.from_yaml(args.config_path)
    else:
        logger.info("Using default configuration")
        config = Config()
    
    # Override configuration with arguments
    config.data.data_dir = args.data_dir
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.system.device = args.device
    config.system.log_level = args.log_level
    config.system.log_file = args.log_file
    
    # Set device
    device = torch.device(config.system.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        logger.info("Loading model...")
        model = load_model(args.model_path, config, device)
        
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = WRFDataLoader(config)
        
        # Create test data loader
        _, test_loader = data_loader.get_train_val_loaders()
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics, all_predictions, all_targets = evaluate_model(
            model, test_loader, config, args
        )
        
        # Save metrics
        if args.calculate_metrics:
            metrics_path = os.path.join(args.output_dir, args.metrics_file)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
        
        # Generate visualizations
        if args.save_visualizations:
            logger.info("Generating visualizations...")
            
            # Generate sample predictions
            predictions = generate_predictions(model, test_loader, config, args)
            
            # Save visualizations
            for i, sample in enumerate(predictions):
                vis_path = os.path.join(args.output_dir, f'visualization_{i}.png')
                save_sample_visualization(sample, vis_path)
                
                logger.info(f"Visualization saved: {vis_path}")
        
        # Save samples
        if args.save_samples:
            logger.info("Saving samples...")
            
            samples_dir = os.path.join(args.output_dir, 'samples')
            os.makedirs(samples_dir, exist_ok=True)
            
            for i, sample in enumerate(predictions):
                sample_path = os.path.join(samples_dir, f'sample_{i}.pt')
                torch.save(sample, sample_path)
                
                logger.info(f"Sample saved: {sample_path}")
        
        # Create evaluation report
        report = {
            'model_path': args.model_path,
            'config_path': args.config_path,
            'evaluation_date': str(torch.datetime.now()),
            'metrics': metrics,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'prediction_steps': args.prediction_steps
        }
        
        report_path = os.path.join(args.output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()