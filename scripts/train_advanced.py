"""
Advanced training script with curriculum learning, auxiliary MLM, and hard negative mining.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import heapq
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import Trainer
from data.dataset import HomodimerDataset
from models.cspan import CSPAN
from utils.losses import CombinedLoss


class HardNegativeMiner:
    """Manages hard negative mining during training."""
    
    def __init__(self, config: Dict, num_negatives: int):
        self.config = config
        self.top_k_percent = config['top_k_percent']
        self.update_frequency = config['update_frequency']
        self.num_negatives = num_negatives
        
        # Store scores for each negative sample
        self.negative_scores = np.zeros(num_negatives)
        self.negative_indices = []
        self.hard_negative_pool = set()
        
    def update_scores(self, indices: List[int], scores: List[float], labels: List[int]):
        """Update scores for negative samples."""
        for idx, score, label in zip(indices, scores, labels):
            if label == 0:  # Negative sample
                self.negative_scores[idx] = score
                self.negative_indices.append(idx)
    
    def get_hard_negatives(self) -> List[int]:
        """Get indices of current hard negatives."""
        if len(self.negative_indices) == 0:
            return []
        
        # Get top k% hardest negatives (highest scores)
        k = int(len(self.negative_indices) * self.top_k_percent / 100)
        k = max(1, k)
        
        # Use heap to efficiently get top k
        top_k_indices = heapq.nlargest(
            k,
            self.negative_indices,
            key=lambda x: self.negative_scores[x]
        )
        
        self.hard_negative_pool = set(top_k_indices)
        return list(self.hard_negative_pool)
    
    def should_use_hard_negative(self, idx: int, label: int) -> bool:
        """Check if this sample should be included based on hard negative mining."""
        if label == 1:  # Always include positive samples
            return True
        
        # Include if it's a hard negative or randomly with some probability
        if idx in self.hard_negative_pool:
            return True
        
        # Include other negatives with reduced probability
        return np.random.random() < 0.3  # 30% chance for easy negatives


class CurriculumScheduler:
    """Manages curriculum learning schedule."""
    
    def __init__(self, config: Dict, dataset_size: int):
        self.config = config
        self.strategy = config['strategy']
        self.initial_fraction = config['initial_fraction']
        self.increment_fraction = config['increment_fraction']
        self.stage_epochs = config['stage_epochs']
        self.dataset_size = dataset_size
        
        # Track curriculum progress
        self.current_stage = 0
        self.current_fraction = self.initial_fraction
        self.sample_difficulties = None
        
    def initialize_difficulties(self, sequences: List[str], labels: List[int]):
        """Initialize sample difficulties based on strategy."""
        difficulties = []
        
        for seq, label in zip(sequences, labels):
            if self.strategy == "length":
                # Shorter sequences are easier
                difficulty = len(seq)
            elif self.strategy == "difficulty":
                # Would need model predictions, using random for now
                difficulty = np.random.random()
            elif self.strategy == "combined":
                # Combine length and other factors
                length_factor = len(seq) / 1000.0
                balance_factor = 0.5 if label == 1 else 1.0  # Positive samples are "easier"
                difficulty = length_factor * balance_factor
            else:
                difficulty = np.random.random()
            
            difficulties.append(difficulty)
        
        self.sample_difficulties = np.array(difficulties)
    
    def get_curriculum_indices(self, epoch: int) -> List[int]:
        """Get indices of samples to use in current curriculum stage."""
        # Update stage if needed
        if epoch > 0 and epoch % self.stage_epochs == 0:
            self.current_stage += 1
            self.current_fraction = min(1.0, self.current_fraction + self.increment_fraction)
        
        # Get samples based on current fraction
        num_samples = int(self.dataset_size * self.current_fraction)
        num_samples = min(num_samples, self.dataset_size)
        
        # Sort by difficulty and take easiest samples
        sorted_indices = np.argsort(self.sample_difficulties)
        selected_indices = sorted_indices[:num_samples]
        
        print(f"Curriculum Stage {self.current_stage}: Using {len(selected_indices)} samples ({self.current_fraction:.1%})")
        
        return selected_indices.tolist()


class AdvancedTrainer(Trainer):
    """Advanced trainer with curriculum learning, MLM, and hard negative mining."""
    
    def __init__(self, config_path: str, args):
        super().__init__(config_path, args)
        
        # Load advanced training config
        self.adv_config = self.config.get('advanced_training', {})
        
        # Initialize hard negative mining
        if self.adv_config.get('hard_negative_mining', {}).get('enabled', False):
            print("Initializing hard negative mining...")
            train_dataset = HomodimerDataset('train', config_path)
            num_negatives = len([l for l in train_dataset.labels if l == 0])
            self.hard_negative_miner = HardNegativeMiner(
                self.adv_config['hard_negative_mining'],
                num_negatives
            )
        else:
            self.hard_negative_miner = None
        
        # Initialize curriculum learning
        if self.adv_config.get('enhanced_curriculum', {}).get('enabled', False):
            print("Initializing curriculum learning...")
            train_dataset = HomodimerDataset('train', config_path)
            self.curriculum_scheduler = CurriculumScheduler(
                self.adv_config['enhanced_curriculum'],
                len(train_dataset)
            )
            self.curriculum_scheduler.initialize_difficulties(
                train_dataset.sequences,
                train_dataset.labels
            )
        else:
            self.curriculum_scheduler = None
        
        # Update loss function for auxiliary MLM if enabled
        if self.adv_config.get('auxiliary_mlm', {}).get('enabled', False):
            print("Enabling auxiliary MLM task...")
            self.use_mlm = True
            # Update loss function weights
            self.criterion.aux_weight = self.adv_config['auxiliary_mlm']['weight']
            self.criterion.mlm_loss.mask_prob = self.adv_config['auxiliary_mlm']['mask_prob']
        else:
            self.use_mlm = False
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with advanced features."""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_indices = []
        
        # Get curriculum indices if enabled
        if self.curriculum_scheduler is not None:
            curriculum_indices = set(self.curriculum_scheduler.get_curriculum_indices(epoch))
        else:
            curriculum_indices = None
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Apply curriculum learning filter
            if curriculum_indices is not None:
                # Filter batch based on curriculum
                mask = torch.tensor([
                    idx.item() in curriculum_indices 
                    for idx in batch['indices']
                ])
                
                if mask.sum() == 0:  # Skip if no samples selected
                    continue
                
                # Filter batch tensors
                if batch.get('needs_features', False):
                    # Extract features for filtered sequences
                    filtered_sequences = [batch['sequences'][i] for i, m in enumerate(mask) if m]
                    filtered_indices = [batch['indices'][i].item() for i, m in enumerate(mask) if m]
                    
                    with torch.no_grad():
                        features_list = self.feature_extractor.extract_features(
                            filtered_sequences,
                            split='train',
                            indices=filtered_indices,
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
                            
                            attn_mask = np.ones(max_len)
                            attn_mask[seq_len:] = 0
                            attention_masks.append(attn_mask)
                        
                        features = torch.tensor(np.stack(padded_features), dtype=torch.float32).to(self.device)
                        attention_mask = torch.tensor(np.stack(attention_masks), dtype=torch.float32).to(self.device)
                else:
                    features = batch['features'][mask].to(self.device)
                    attention_mask = batch['attention_mask'][mask].to(self.device)
                
                labels = batch['labels'][mask].to(self.device)
                indices = batch['indices'][mask]
            else:
                # No curriculum filtering
                if batch.get('needs_features', False):
                    # Extract features in main process
                    with torch.no_grad():
                        sequences = batch['sequences']
                        indices_list = batch['indices'].tolist()
                        
                        features_list = self.feature_extractor.extract_features(
                            sequences,
                            split='train',
                            indices=indices_list,
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
                            
                            attn_mask = np.ones(max_len)
                            attn_mask[seq_len:] = 0
                            attention_masks.append(attn_mask)
                        
                        features = torch.tensor(np.stack(padded_features), dtype=torch.float32).to(self.device)
                        attention_mask = torch.tensor(np.stack(attention_masks), dtype=torch.float32).to(self.device)
                else:
                    features = batch['features'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                
                labels = batch['labels'].to(self.device)
                indices = batch['indices']
            
            # Store original features for MLM if needed
            original_features = features.clone() if self.use_mlm else None
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(features, attention_mask)
                    
                    # Prepare loss inputs
                    loss_kwargs = {
                        'logits': outputs['logits'],
                        'targets': labels,
                        'attention_mask': attention_mask
                    }
                    
                    # Add MLM components if enabled
                    if self.use_mlm and original_features is not None:
                        # Get hidden states from cross-scale features
                        # We need to modify CSPAN to return these
                        loss_kwargs['hidden_states'] = outputs.get('hidden_states')
                        loss_kwargs['original_features'] = original_features
                    
                    # Add attention weights for regularization
                    if hasattr(outputs, 'motif_attention'):
                        loss_kwargs['attention_weights'] = {
                            'motif': outputs['motif_attention']
                        }
                    
                    losses = self.criterion(**loss_kwargs)
                    loss = losses['total_loss']
            else:
                outputs = self.model(features, attention_mask)
                
                loss_kwargs = {
                    'logits': outputs['logits'],
                    'targets': labels,
                    'attention_mask': attention_mask
                }
                
                if self.use_mlm and original_features is not None:
                    loss_kwargs['hidden_states'] = outputs.get('hidden_states')
                    loss_kwargs['original_features'] = original_features
                
                if hasattr(outputs, 'motif_attention'):
                    loss_kwargs['attention_weights'] = {
                        'motif': outputs['motif_attention']
                    }
                
                losses = self.criterion(**loss_kwargs)
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
            predictions = outputs['probabilities'].detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            indices_np = indices.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels_np)
            all_indices.extend(indices_np)
            
            # Update hard negative miner
            if self.hard_negative_miner is not None:
                self.hard_negative_miner.update_scores(
                    indices_np.tolist(),
                    predictions.tolist(),
                    labels_np.tolist()
                )
                
                # Update hard negative pool periodically
                if self.global_step % self.hard_negative_miner.update_frequency == 0:
                    hard_negatives = self.hard_negative_miner.get_hard_negatives()
                    print(f"\nUpdated hard negative pool: {len(hard_negatives)} samples")
            
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
                
                # Add component losses
                for name, value in losses.items():
                    if name != 'total_loss':
                        log_dict[f'train/{name}'] = value.item()
                
                # Add curriculum info
                if self.curriculum_scheduler is not None:
                    log_dict['train/curriculum_fraction'] = self.curriculum_scheduler.current_fraction
                    log_dict['train/curriculum_stage'] = self.curriculum_scheduler.current_stage
                
                if self.config['experiment']['use_wandb'] and not self.args.no_wandb:
                    import wandb
                    wandb.log(log_dict, step=self.global_step)
            
            # Clear cache periodically
            if batch_idx % self.config['hardware']['empty_cache_interval'] == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels)
        
        from utils.metrics import compute_all_metrics
        train_metrics = compute_all_metrics(all_labels, all_predictions)
        train_metrics['loss'] = epoch_loss
        
        return train_metrics


def main():
    parser = argparse.ArgumentParser(description='Train CSPAN with advanced features')
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
    
    # Advanced training flags
    parser.add_argument('--enable-curriculum', action='store_true',
                       help='Enable curriculum learning')
    parser.add_argument('--enable-mlm', action='store_true',
                       help='Enable auxiliary MLM task')
    parser.add_argument('--enable-hard-negative', action='store_true',
                       help='Enable hard negative mining')
    parser.add_argument('--enable-all', action='store_true',
                       help='Enable all advanced features')
    
    args = parser.parse_args()
    
    # Load and modify config based on flags
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable advanced features
    if args.enable_all:
        args.enable_curriculum = True
        args.enable_mlm = True
        args.enable_hard_negative = True
    
    if args.enable_curriculum:
        config['advanced_training']['enhanced_curriculum']['enabled'] = True
        print("✓ Curriculum learning enabled")
    
    if args.enable_mlm:
        config['advanced_training']['auxiliary_mlm']['enabled'] = True
        print("✓ Auxiliary MLM task enabled")
    
    if args.enable_hard_negative:
        config['advanced_training']['hard_negative_mining']['enabled'] = True
        print("✓ Hard negative mining enabled")
    
    # Save modified config
    temp_config = 'config_advanced_temp.yml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    args.config = temp_config
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Train model
    trainer = AdvancedTrainer(args.config, args)
    trainer.train()
    
    # Clean up temp config
    import os
    if os.path.exists(temp_config):
        os.remove(temp_config)


if __name__ == '__main__':
    main()