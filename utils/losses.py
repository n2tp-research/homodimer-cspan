"""
Loss functions for homodimerization prediction with class imbalance handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        pos_weight: Optional[float] = None
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class (0, 1)
            gamma: Focusing parameter (>= 0)
            reduction: Reduction method ('none', 'mean', 'sum')
            pos_weight: Alternative positive class weight
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (batch_size, 1) or (batch_size,)
            targets: Ground truth labels (batch_size,) with values in {0, 1}
        
        Returns:
            Loss value
        """
        # Ensure inputs are the right shape
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        
        # Convert to float
        targets = targets.float()
        
        # Calculate binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Calculate probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        # Apply positive weight if specified
        if self.pos_weight is not None:
            pos_weight_t = self.pos_weight * targets + (1 - targets)
            focal_weight = focal_weight * pos_weight_t
        
        # Calculate focal loss
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MaskedLanguageModelingLoss(nn.Module):
    """
    Auxiliary self-supervised loss for masked residue prediction.
    """
    
    def __init__(self, feature_dim: int = 1280, mask_prob: float = 0.15):
        """
        Initialize MLM loss.
        
        Args:
            feature_dim: Dimension of features to predict
            mask_prob: Probability of masking each position
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.mask_prob = mask_prob
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def create_mlm_masks(
        self,
        attention_mask: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create masks for MLM task.
        
        Args:
            attention_mask: Valid positions (batch, seq_len)
            special_tokens_mask: Positions to never mask (batch, seq_len)
        
        Returns:
            MLM mask (batch, seq_len) with True for masked positions
        """
        batch_size, seq_len = attention_mask.shape
        
        # Create probability matrix
        probability_matrix = torch.full(
            (batch_size, seq_len), self.mask_prob, device=attention_mask.device
        )
        
        # Don't mask special tokens or padding
        if special_tokens_mask is not None:
            probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)
        probability_matrix.masked_fill_(~attention_mask.bool(), value=0.0)
        
        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        return masked_indices
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_features: torch.Tensor,
        mlm_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MLM loss.
        
        Args:
            hidden_states: Model hidden states (batch, seq_len, hidden_dim)
            original_features: Original ESM2 features (batch, seq_len, feature_dim)
            mlm_mask: Positions to compute loss on (batch, seq_len)
            attention_mask: Valid positions (batch, seq_len)
        
        Returns:
            MLM loss value
        """
        # Project hidden states to feature dimension if needed
        if hidden_states.size(-1) != self.feature_dim:
            # Add projection layer if dimensions don't match
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(
                    hidden_states.size(-1), self.feature_dim
                ).to(hidden_states.device)
            hidden_states = self.projection(hidden_states)
        
        # Compute MSE loss only on masked positions
        loss = self.mse_loss(hidden_states, original_features)
        
        # Apply masks
        if attention_mask is not None:
            mlm_mask = mlm_mask & attention_mask.bool()
        
        # Average over masked positions
        masked_loss = loss[mlm_mask]
        
        if masked_loss.numel() > 0:
            return masked_loss.mean()
        else:
            return torch.tensor(0.0, device=loss.device)


class EntropyRegularization(nn.Module):
    """
    Entropy regularization for attention distributions.
    """
    
    def __init__(self, weight: float = 1e-3):
        """
        Initialize entropy regularization.
        
        Args:
            weight: Regularization weight
        """
        super().__init__()
        self.weight = weight
        
    def forward(self, attention_weights: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute entropy regularization.
        
        Args:
            attention_weights: Attention weights (batch, ..., seq_len)
            mask: Valid positions (batch, seq_len)
        
        Returns:
            Entropy regularization loss
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_weights = attention_weights + eps
        
        # Compute entropy
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention shape
            while mask.dim() < attention_weights.dim() - 1:
                mask = mask.unsqueeze(1)
            
            # Only compute entropy for valid positions
            valid_positions = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            entropy = entropy.sum(dim=-1) / valid_positions.squeeze(-1)
        
        # Average entropy across batch and heads
        entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
        
        return self.weight * entropy_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with focal loss, auxiliary MLM, and regularization.
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        aux_weight: float = 0.1,
        reg_weight: float = 1e-4,
        mask_prob: float = 0.15
    ):
        """
        Initialize combined loss.
        
        Args:
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            aux_weight: Weight for auxiliary MLM loss
            reg_weight: Weight for regularization terms
            mask_prob: Probability for MLM masking
        """
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.mlm_loss = MaskedLanguageModelingLoss(mask_prob=mask_prob)
        self.entropy_reg = EntropyRegularization(weight=reg_weight)
        
        self.aux_weight = aux_weight
        self.reg_weight = reg_weight
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        original_features: Optional[torch.Tensor] = None,
        attention_weights: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions (batch, 1)
            targets: Ground truth labels (batch,)
            hidden_states: Model hidden states for MLM (batch, seq_len, hidden_dim)
            original_features: Original ESM2 features (batch, seq_len, feature_dim)
            attention_weights: Dictionary of attention weights from different layers
            attention_mask: Valid positions (batch, seq_len)
        
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        
        # Primary focal loss
        focal_loss = self.focal_loss(logits, targets)
        losses['focal_loss'] = focal_loss
        total_loss = focal_loss
        
        # Auxiliary MLM loss
        if hidden_states is not None and original_features is not None:
            # Create MLM masks
            mlm_mask = self.mlm_loss.create_mlm_masks(attention_mask)
            
            # Compute MLM loss
            mlm_loss = self.mlm_loss(
                hidden_states, original_features, mlm_mask, attention_mask
            )
            losses['mlm_loss'] = mlm_loss
            total_loss = total_loss + self.aux_weight * mlm_loss
        
        # Entropy regularization on attention weights
        if attention_weights is not None and self.reg_weight > 0:
            entropy_losses = []
            
            for name, weights in attention_weights.items():
                if weights is not None:
                    entropy_loss = self.entropy_reg(weights, attention_mask)
                    entropy_losses.append(entropy_loss)
                    losses[f'entropy_{name}'] = entropy_loss
            
            if entropy_losses:
                total_entropy = sum(entropy_losses) / len(entropy_losses)
                losses['entropy_reg'] = total_entropy
                total_loss = total_loss + total_entropy
        
        losses['total_loss'] = total_loss
        
        return losses


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better calibration.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Label smoothing factor
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Predicted logits (batch, 1)
            targets: Ground truth labels (batch,)
        
        Returns:
            Loss value
        """
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        
        targets = targets.float()
        
        # Apply label smoothing
        smooth_targets = targets * self.confidence + (1 - targets) * self.smoothing
        
        # Compute BCE loss with smoothed targets
        loss = F.binary_cross_entropy_with_logits(inputs, smooth_targets)
        
        return loss


if __name__ == "__main__":
    # Test losses
    batch_size = 8
    seq_len = 100
    
    # Create dummy data
    logits = torch.randn(batch_size, 1)
    targets = torch.randint(0, 2, (batch_size,))
    hidden_states = torch.randn(batch_size, seq_len, 512)
    original_features = torch.randn(batch_size, seq_len, 1280)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 80:] = 0
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=0.1, gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss()
    losses = combined_loss(
        logits, targets,
        hidden_states=hidden_states,
        original_features=original_features,
        attention_mask=attention_mask
    )
    
    print("\nCombined losses:")
    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")