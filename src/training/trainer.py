"""
Training pipeline for WRF diffusion model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import time
from tqdm import tqdm
import json

from ..models.diffusion_model import WRFDiffusionModel
from ..data.preprocessing import WRFDataLoader, setup_logging, get_memory_usage, get_gpu_memory_usage
from ..config.config import Config


class WRFTrainer:
    """Trainer for WRF diffusion model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # Setup logging
        setup_logging(config.system.log_file, config.system.log_level)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = WRFDiffusionModel(config).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision == "fp16" else None
        
        # Initialize data loader
        self.data_loader = WRFDataLoader(config)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
        self.logger.info("Trainer initialized successfully")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.training.optimizer == "AdamW":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=1e-6
            )
        elif self.config.training.scheduler == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.training.epochs
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (input_data, target_data) in enumerate(progress_bar):
            # Move data to device
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            
            # Sample timesteps
            batch_size = input_data.shape[0]
            t = torch.randint(0, self.config.model.num_steps, (batch_size,), device=self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    loss = self.model(target_data, t)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                loss = self.model(target_data, t)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log metrics
            if self.global_step % self.config.training.log_interval == 0:
                self._log_metrics(loss.item(), 'train')
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_data, target_data in tqdm(val_loader, desc="Validation"):
                # Move data to device
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                # Sample timesteps
                batch_size = input_data.shape[0]
                t = torch.randint(0, self.config.model.num_steps, (batch_size,), device=self.device)
                
                # Forward pass
                loss = self.model(target_data, t)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: Optional[int] = None) -> None:
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.training.epochs
        
        # Create data loaders
        train_loader, val_loader = self.data_loader.get_train_val_loaders()
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training batches: {len(train_loader)}")
        self.logger.info(f"Validation batches: {len(val_loader)}")
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validate epoch
                if epoch % self.config.training.eval_interval == 0:
                    val_loss = self.validate_epoch(val_loader)
                    self.val_losses.append(val_loss)
                    
                    self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint('best_model.pth')
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Save checkpoint
                if epoch % self.config.training.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                # Log memory usage
                if self.config.system.monitor_memory:
                    self._log_memory_usage()
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_checkpoint('interrupted_checkpoint.pth')
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.save_checkpoint('error_checkpoint.pth')
            raise
        
        self.logger.info("Training completed!")
        self.save_final_metrics()
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info("Checkpoint loaded successfully")
    
    def _log_metrics(self, loss: float, mode: str) -> None:
        """Log training metrics"""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(
            f"Step {self.global_step} | {mode.upper()} Loss: {loss:.6f} | LR: {current_lr:.2e}"
        )
    
    def _log_memory_usage(self) -> None:
        """Log memory usage"""
        try:
            # CPU memory
            cpu_memory = get_memory_usage()
            
            # GPU memory
            gpu_memory = get_gpu_memory_usage()
            
            self.logger.info(
                f"Memory - CPU: {cpu_memory['rss']:.2f}GB | "
                f"GPU: {gpu_memory}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {e}")
    
    def save_final_metrics(self) -> None:
        """Save final training metrics"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_loss': self.best_loss,
            'total_epochs': self.current_epoch,
            'total_steps': self.global_step
        }
        
        metrics_path = os.path.join(self.config.training.checkpoint_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved: {metrics_path}")
    
    def generate_samples(self, num_samples: int = 5) -> None:
        """Generate sample predictions"""
        self.model.eval()
        
        # Get validation data
        _, val_loader = self.data_loader.get_train_val_loaders()
        
        with torch.no_grad():
            for i, (input_data, target_data) in enumerate(val_loader):
                if i >= num_samples:
                    break
                
                # Move to device
                input_data = input_data.to(self.device)
                
                # Generate prediction
                prediction = self.model.predict_next_step(input_data)
                
                # Save samples
                sample_path = os.path.join(
                    self.config.training.checkpoint_dir, 
                    f'sample_{i}.pt'
                )
                torch.save({
                    'input': input_data.cpu(),
                    'prediction': prediction.cpu(),
                    'target': target_data.cpu()
                }, sample_path)
        
        self.logger.info(f"Generated {num_samples} samples")


def main():
    """Main training script"""
    # Load configuration
    config = Config()
    
    # Create trainer
    trainer = WRFTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()