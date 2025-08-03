"""
Training script with pre-extracted features to avoid CUDA multiprocessing issues.
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
from scripts.train import Trainer


class FeatureAwareTrainer(Trainer):
    """Modified trainer that handles feature extraction in main process."""
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with feature extraction in main process."""
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
        """Validate model with feature extraction in main process."""
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
    parser.add_argument('--precompute-features', action='store_true',
                       help='Precompute all features before training')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Precompute features if requested
    if args.precompute_features:
        print("Precomputing features...")
        from data.feature_extraction import precompute_all_features
        precompute_all_features(args.config)
        print("Feature precomputation complete!")
    
    # Train model
    trainer = FeatureAwareTrainer(args.config, args)
    trainer.train()


if __name__ == '__main__':
    main()