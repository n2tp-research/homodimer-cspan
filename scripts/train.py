"""
Training script for CSPAN homodimerization prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import wandb
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from data.feature_extraction import ESM2FeatureExtractor
from models.cspan import CSPAN
from utils.losses import CombinedLoss
from utils.metrics import compute_all_metrics, MetricTracker


class Trainer:
    """Trainer class for CSPAN model."""
    
    def __init__(self, config_path: str, args):
        """Initialize trainer."""
        self.args = args
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize experiment tracking
        if self.config['experiment']['use_wandb'] and not args.no_wandb:
            self._init_wandb()
        
        # Create directories
        self.checkpoint_dir = Path(self.config['experiment']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_data()
        self._init_model()
        self._init_training()
        
        # Initialize metrics tracking
        self.metric_tracker = MetricTracker([
            'val_auprc', 'val_mcc', 'val_f1_optimal', 'val_balanced_accuracy'
        ])
        
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb.init(
            project=self.config['experiment']['wandb_project'],
            entity=self.config['experiment'].get('wandb_entity'),
            config=self.config,
            name=f"cspan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=['cspan', 'homodimer']
        )
        
    def _init_data(self):
        """Initialize data loaders and feature extractor."""
        print("Initializing data loaders...")
        
        # Get data loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            config_path=self.args.config,
            batch_size=self.config['training']['batch_size'],
            use_bucket_sampling=True
        )
        
        # Initialize feature extractor
        self.feature_extractor = ESM2FeatureExtractor(
            config_path=self.args.config,
            device=self.device,
            use_half_precision=self.config['training']['use_amp'],
            verbose=True
        )
        
        # Set custom collate function
        collate_fn = self.feature_extractor.get_collate_fn()
        self.train_loader.collate_fn = collate_fn
        self.val_loader.collate_fn = collate_fn
        self.test_loader.collate_fn = collate_fn
        
    def _init_model(self):
        """Initialize model."""
        print("Initializing model...")
        
        # Create model
        self.model = CSPAN(config_path=self.args.config)
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Load checkpoint if specified
        if self.args.resume:
            self._load_checkpoint(self.args.resume)
        
    def _init_training(self):
        """Initialize training components."""
        train_config = self.config['training']
        
        # Loss function
        self.criterion = CombinedLoss(
            focal_alpha=train_config['loss']['focal_loss_alpha'],
            focal_gamma=train_config['loss']['focal_loss_gamma'],
            aux_weight=train_config['loss']['auxiliary_loss_weight'],
            reg_weight=train_config['loss']['regularization_weight']
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            betas=(train_config['adam_beta1'], train_config['adam_beta2']),
            eps=train_config['adam_epsilon'],
            weight_decay=train_config['weight_decay']
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * train_config['max_epochs']
        warmup_steps = train_config['warmup_steps']
        
        # Cosine schedule with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision training
        self.use_amp = train_config['use_amp'] and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -np.inf
        self.patience_counter = 0
        
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Check if features need to be extracted
            if batch.get('needs_features', False):
                # Extract features in main process
                with torch.no_grad():
                    sequences = batch['sequences']
                    indices = batch['indices'].tolist()
                    
                    # Extract features
                    features_list = self.feature_extractor.extract_features(
                        sequences,
                        split='train',
                        indices=indices,
                        use_cache=True,
                        save_to_cache=True
                    )
                    
                    # Pad features
                    max_len = max(feat.shape[0] for feat in features_list)
                    padded_features = []
                    attention_masks = []
                    
                    for feat in features_list:
                        seq_len = feat.shape[0]
                        if seq_len < max_len:
                            padding = np.zeros((max_len - seq_len, feat.shape[1]))
                            feat = np.concatenate([feat, padding], axis=0)
                        
                        padded_features.append(feat)
                        
                        mask = np.ones(max_len)
                        mask[seq_len:] = 0
                        attention_masks.append(mask)
                    
                    # Convert to tensors
                    features = torch.tensor(np.stack(padded_features), dtype=torch.float32).to(self.device)
                    attention_mask = torch.tensor(np.stack(attention_masks), dtype=torch.float32).to(self.device)
            else:
                # Features already extracted
                features = batch['features'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
            
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(features, attention_mask)
                    losses = self.criterion(
                        outputs['logits'],
                        labels,
                        attention_mask=attention_mask
                    )
                    loss = losses['total_loss']
            else:
                outputs = self.model(features, attention_mask)
                losses = self.criterion(
                    outputs['logits'],
                    labels,
                    attention_mask=attention_mask
                )
                loss = losses['total_loss']
            
            # Gradient accumulation
            accumulation_steps = self.config['training']['gradient_accumulation_steps']
            loss = loss / accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * accumulation_steps
            all_predictions.extend(outputs['probabilities'].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.global_step % self.config['experiment']['log_interval'] == 0:
                log_dict = {
                    'train/loss': loss.item() * accumulation_steps,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                }
                
                for name, value in losses.items():
                    if name != 'total_loss':
                        log_dict[f'train/{name}'] = value.item()
                
                if self.config['experiment']['use_wandb'] and not self.args.no_wandb:
                    wandb.log(log_dict, step=self.global_step)
            
            # Clear cache periodically
            if batch_idx % self.config['hardware']['empty_cache_interval'] == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels)
        
        train_metrics = compute_all_metrics(all_labels, all_predictions)
        train_metrics['loss'] = epoch_loss
        
        return train_metrics
    
    @torch.no_grad()
    def validate(self, loader, split='val'):
        """Validate model."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        for batch in tqdm(loader, desc=f"Validating ({split})"):
            # Check if features need to be extracted
            if batch.get('needs_features', False):
                # Extract features in main process
                sequences = batch['sequences']
                indices = batch['indices'].tolist()
                
                # Extract features
                features_list = self.feature_extractor.extract_features(
                    sequences,
                    split=split if split != 'val' else 'valid',
                    indices=indices,
                    use_cache=True,
                    save_to_cache=False
                )
                
                # Pad features
                max_len = max(feat.shape[0] for feat in features_list)
                padded_features = []
                attention_masks = []
                
                for feat in features_list:
                    seq_len = feat.shape[0]
                    if seq_len < max_len:
                        padding = np.zeros((max_len - seq_len, feat.shape[1]))
                        feat = np.concatenate([feat, padding], axis=0)
                    
                    padded_features.append(feat)
                    
                    mask = np.ones(max_len)
                    mask[seq_len:] = 0
                    attention_masks.append(mask)
                
                # Convert to tensors
                features = torch.tensor(np.stack(padded_features), dtype=torch.float32).to(self.device)
                attention_mask = torch.tensor(np.stack(attention_masks), dtype=torch.float32).to(self.device)
            else:
                # Features already extracted
                features = batch['features'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
            
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(features, attention_mask)
            losses = self.criterion(
                outputs['logits'],
                labels,
                attention_mask=attention_mask
            )
            
            # Track metrics
            total_loss += losses['total_loss'].item()
            all_predictions.extend(outputs['probabilities'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels)
        
        metrics = compute_all_metrics(all_labels, all_predictions)
        metrics['loss'] = total_loss / len(loader)
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print("\nStarting training...")
        
        for epoch in range(self.start_epoch, self.config['training']['max_epochs']):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(self.val_loader, 'val')
            
            # Update metric tracker
            self.metric_tracker.update(
                {f'val_{k}': v for k, v in val_metrics.items()},
                epoch
            )
            
            # Print metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"AUPRC: {train_metrics['auprc']:.4f}, "
                  f"MCC: {train_metrics.get('mcc', 0):.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"AUPRC: {val_metrics['auprc']:.4f}, "
                  f"MCC: {val_metrics.get('mcc', 0):.4f}")
            
            # Log to wandb
            if self.config['experiment']['use_wandb'] and not self.args.no_wandb:
                log_dict = {}
                for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            log_dict[f'{split}/{name}'] = value
                
                wandb.log(log_dict, step=self.global_step)
            
            # Check for improvement
            primary_metric = self.config['evaluation']['primary_metric']
            current_metric = val_metrics[primary_metric]
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best {primary_metric}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                
                # Save regular checkpoint
                if epoch % self.config['experiment']['save_interval'] == 0:
                    self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\nTraining completed!")
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_metrics = self.validate(self.test_loader, 'test')
        
        print("\nTest Set Results:")
        for name, value in sorted(test_metrics.items()):
            if isinstance(value, float):
                print(f"{name:25s}: {value:.4f}")
        
        # Save results
        results = {
            'config': self.config,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_epoch': self.metric_tracker.best_epochs[f'val_{primary_metric}'],
            'total_epochs': epoch + 1
        }
        
        results_path = self.checkpoint_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Close feature extractor
        self.feature_extractor.close()
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Keep only best k checkpoints
        if not is_best:
            checkpoints = sorted(
                self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
                key=lambda x: int(x.stem.split('_')[-1])
            )
            
            keep_best_k = self.config['experiment']['keep_best_k']
            if len(checkpoints) > keep_best_k:
                for ckpt in checkpoints[:-keep_best_k]:
                    ckpt.unlink()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        print(f"Resumed from epoch {checkpoint['epoch']}")


def main():
    parser = argparse.ArgumentParser(description='Train CSPAN model')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Train model
    trainer = Trainer(args.config, args)
    trainer.train()


if __name__ == '__main__':
    main()