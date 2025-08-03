"""
Cross-Scale Protein Attention Network (CSPAN) for homodimerization prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import yaml
from einops import rearrange, reduce
import math

from models.layers import (
    LocalFeatureExtractor,
    RegionalFeatureExtractor,
    GlobalFeatureExtractor,
    CrossScaleAttention
)


class DimerizationMotifDiscovery(nn.Module):
    """Learnable motif discovery module for identifying dimerization patterns."""
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_motifs: int = 32,
        motif_dim: int = 512,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        """
        Initialize motif discovery module.
        
        Args:
            feature_dim: Input feature dimension
            num_motifs: Number of learnable motif queries
            motif_dim: Dimension of each motif
            temperature: Temperature for attention softmax
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_motifs = num_motifs
        self.motif_dim = motif_dim
        self.temperature = temperature
        
        # Learnable motif queries
        self.motif_queries = nn.Parameter(
            torch.randn(num_motifs, motif_dim) * 0.02
        )
        
        # Projections
        self.feature_proj = nn.Linear(feature_dim, motif_dim)
        self.motif_value_proj = nn.Linear(feature_dim, motif_dim)
        
        # Motif importance scoring
        self.importance_scorer = nn.Sequential(
            nn.Linear(motif_dim, motif_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(motif_dim // 2, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(motif_dim, feature_dim)
        
        # Normalization
        self.norm = nn.LayerNorm(motif_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features (batch, seq_len, feature_dim)
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Tuple of:
                - Aggregated motif features (batch, feature_dim)
                - Motif attention weights (batch, num_motifs, seq_len)
        """
        batch_size, seq_len, _ = features.shape
        
        # Project features
        projected_features = self.feature_proj(features)  # (batch, seq_len, motif_dim)
        value_features = self.motif_value_proj(features)  # (batch, seq_len, motif_dim)
        
        # Expand motif queries for batch
        motif_queries = self.motif_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch, num_motifs, motif_dim)
        
        # Compute attention scores between motifs and sequence positions
        attention_scores = torch.bmm(
            motif_queries,
            projected_features.transpose(1, 2)
        ) / (math.sqrt(self.motif_dim) * self.temperature)
        # (batch, num_motifs, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, self.num_motifs, -1)
            attention_scores = attention_scores.masked_fill(
                ~mask_expanded.bool(), float('-inf')
            )
        
        # Softmax over sequence positions
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate features for each motif
        motif_features = torch.bmm(
            attention_weights,
            value_features
        )  # (batch, num_motifs, motif_dim)
        
        # Apply normalization
        motif_features = self.norm(motif_features)
        
        # Compute motif importance scores
        importance_scores = self.importance_scorer(motif_features).squeeze(-1)
        importance_weights = F.sigmoid(importance_scores)  # (batch, num_motifs)
        
        # Weighted aggregation of motif features
        weighted_features = motif_features * importance_weights.unsqueeze(-1)
        aggregated_features = weighted_features.sum(dim=1)  # (batch, motif_dim)
        
        # Project back to original dimension
        output = self.output_proj(aggregated_features)  # (batch, feature_dim)
        
        return output, attention_weights


class HierarchicalSequenceAggregation(nn.Module):
    """Hierarchical aggregation with sliding windows."""
    
    def __init__(
        self,
        feature_dim: int = 512,
        window_size: int = 64,
        stride: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize hierarchical aggregation module.
        
        Args:
            feature_dim: Input feature dimension
            window_size: Size of sliding window
            stride: Stride for sliding window
            hidden_dim: Hidden dimension for aggregation
            dropout: Dropout rate
        """
        super().__init__()
        
        self.window_size = window_size
        self.stride = stride
        self.feature_dim = feature_dim
        
        # Window-level aggregation
        self.window_attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Cross-window aggregation
        self.cross_window_attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features (batch, seq_len, feature_dim)
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Aggregated features (batch, feature_dim)
        """
        batch_size, seq_len, _ = features.shape
        
        # Extract overlapping windows
        windows = []
        window_masks = []
        
        for start in range(0, seq_len - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = features[:, start:end, :]
            windows.append(window)
            
            if mask is not None:
                window_mask = mask[:, start:end]
                window_masks.append(window_mask)
        
        # Handle remaining sequence if any
        if len(windows) == 0 or (seq_len - len(windows) * self.stride) > 0:
            start = max(0, seq_len - self.window_size)
            window = features[:, start:, :]
            windows.append(window)
            
            if mask is not None:
                window_mask = mask[:, start:]
                window_masks.append(window_mask)
        
        # Stack windows
        num_windows = len(windows)
        stacked_windows = torch.stack(windows, dim=1)  # (batch, num_windows, window_size, feature_dim)
        
        if mask is not None:
            stacked_masks = torch.stack(window_masks, dim=1)  # (batch, num_windows, window_size)
        
        # Per-window aggregation with attention
        window_features = []
        
        for i in range(num_windows):
            window = stacked_windows[:, i, :, :]  # (batch, window_size, feature_dim)
            
            # Compute attention weights for positions in window
            attn_scores = self.window_attention(window).squeeze(-1)  # (batch, window_size)
            
            # Apply mask if available
            if mask is not None:
                window_mask = stacked_masks[:, i, :]  # (batch, window_size)
                attn_scores = attn_scores.masked_fill(~window_mask.bool(), float('-inf'))
            
            # Softmax and aggregate
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, window_size, 1)
            window_feat = (window * attn_weights).sum(dim=1)  # (batch, feature_dim)
            window_features.append(window_feat)
        
        # Stack window features
        window_features = torch.stack(window_features, dim=1)  # (batch, num_windows, feature_dim)
        
        # Cross-window attention
        cross_scores = self.cross_window_attention(window_features).squeeze(-1)  # (batch, num_windows)
        cross_weights = F.softmax(cross_scores, dim=-1).unsqueeze(-1)  # (batch, num_windows, 1)
        
        # Final aggregation
        aggregated = (window_features * cross_weights).sum(dim=1)  # (batch, feature_dim)
        output = self.output_proj(self.dropout(aggregated))
        
        return output


class CSPAN(nn.Module):
    """Cross-Scale Protein Attention Network for homodimerization prediction."""
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize CSPAN model from configuration."""
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']
        self.data_config = config['data']
        
        # Extract key dimensions
        self.esm_dim = self.model_config['esm_dim']
        self.hidden_dim = self.model_config['hidden_dim']
        self.num_motifs = self.model_config['num_motifs']
        self.dropout = self.model_config['dropout']
        
        # Multi-scale feature extractors
        self.local_extractor = LocalFeatureExtractor(
            in_channels=self.esm_dim,
            out_channels=self.model_config['local_features']['conv_filters'],
            kernel_sizes=self.model_config['local_features']['kernel_sizes'],
            dropout=self.dropout
        )
        
        self.regional_extractor = RegionalFeatureExtractor(
            in_channels=self.esm_dim,
            out_channels=self.model_config['regional_features']['conv_filters'],
            kernel_size=self.model_config['regional_features']['kernel_size'],
            dilation_rates=self.model_config['regional_features']['dilation_rates'],
            dropout=self.dropout
        )
        
        self.global_extractor = GlobalFeatureExtractor(
            in_channels=self.esm_dim,
            d_model=self.model_config['global_features']['d_model'],
            num_heads=self.model_config['global_features']['num_heads'],
            dropout=self.model_config['global_features']['attention_dropout']
        )
        
        # Cross-scale attention
        self.cross_scale_attention = CrossScaleAttention(
            local_dim=self.local_extractor.output_dim,
            regional_dim=self.regional_extractor.output_dim,
            global_dim=self.model_config['global_features']['d_model'],
            d_model=self.model_config['cross_scale']['d_model'],
            num_heads=self.model_config['num_heads'],
            dropout=self.dropout
        )
        
        # Motif discovery
        self.motif_discovery = DimerizationMotifDiscovery(
            feature_dim=self.model_config['cross_scale']['d_model'],
            num_motifs=self.num_motifs,
            motif_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # Hierarchical aggregation
        self.hierarchical_aggregation = HierarchicalSequenceAggregation(
            feature_dim=self.model_config['cross_scale']['d_model'],
            window_size=self.model_config['aggregation']['window_size'],
            stride=self.model_config['aggregation']['stride'],
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout
        )
        
        # Calculate final feature dimension
        cross_scale_dim = self.model_config['cross_scale']['d_model']
        motif_dim = cross_scale_dim
        hierarchical_dim = cross_scale_dim
        
        # Include standard deviation pooling if configured
        pool_multiplier = 3 if self.model_config['aggregation'].get('include_std_pool', True) else 2
        self.final_feature_dim = cross_scale_dim * pool_multiplier + motif_dim + hierarchical_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.final_feature_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: ESM2 features (batch, seq_len, esm_dim)
            attention_mask: Mask for valid positions (batch, seq_len)
        
        Returns:
            Dictionary containing:
                - logits: Raw prediction scores (batch, 1)
                - probabilities: Sigmoid probabilities (batch, 1)
                - motif_attention: Motif attention weights (batch, num_motifs, seq_len)
        """
        # Extract multi-scale features
        local_features = self.local_extractor(features, attention_mask)
        regional_features = self.regional_extractor(features, attention_mask)
        global_features = self.global_extractor(features, attention_mask)
        
        # Cross-scale attention
        cross_scale_features = self.cross_scale_attention(
            local_features, regional_features, global_features, attention_mask
        )
        
        # Motif discovery
        motif_features, motif_attention = self.motif_discovery(
            cross_scale_features, attention_mask
        )
        
        # Hierarchical aggregation
        hierarchical_features = self.hierarchical_aggregation(
            cross_scale_features, attention_mask
        )
        
        # Multiple pooling strategies on cross-scale features
        if attention_mask is not None:
            # Mask out padded positions
            masked_features = cross_scale_features * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1, keepdim=True)  # (batch, 1)
            
            # Average pooling
            avg_pool = masked_features.sum(dim=1) / lengths
            
            # Max pooling
            masked_features_for_max = masked_features.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float('-inf')
            )
            max_pool, _ = masked_features_for_max.max(dim=1)
        else:
            avg_pool = cross_scale_features.mean(dim=1)
            max_pool, _ = cross_scale_features.max(dim=1)
        
        # Standard deviation pooling (optional)
        if self.model_config['aggregation'].get('include_std_pool', True):
            if attention_mask is not None:
                # Compute std only on valid positions
                mean = avg_pool.unsqueeze(1)
                variance = ((masked_features - mean) ** 2).sum(dim=1) / lengths
                std_pool = torch.sqrt(variance + 1e-6)
            else:
                std_pool = cross_scale_features.std(dim=1)
            
            # Concatenate all features
            combined_features = torch.cat([
                avg_pool,
                max_pool,
                std_pool,
                motif_features,
                hierarchical_features
            ], dim=-1)
        else:
            # Concatenate without std pooling
            combined_features = torch.cat([
                avg_pool,
                max_pool,
                motif_features,
                hierarchical_features
            ], dim=-1)
        
        # Classification
        logits = self.classifier(combined_features)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'motif_attention': motif_attention,
            'hidden_states': cross_scale_features  # Return for MLM task
        }
    
    def predict(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simple prediction interface returning only probabilities."""
        output = self.forward(features, attention_mask)
        return output['probabilities']


if __name__ == "__main__":
    # Test the model
    import os
    
    # Create test config if needed
    if not os.path.exists("config.yml"):
        print("Config file not found. Please ensure config.yml exists.")
        exit(1)
    
    # Initialize model
    model = CSPAN("config.yml")
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 150
    esm_dim = 1280
    
    # Create dummy input
    features = torch.randn(batch_size, seq_len, esm_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[0, 120:] = 0  # Mask last 30 positions of first sequence
    
    # Forward pass
    output = model(features, mask)
    
    print(f"\nOutput shapes:")
    print(f"Logits: {output['logits'].shape}")
    print(f"Probabilities: {output['probabilities'].shape}")
    print(f"Motif attention: {output['motif_attention'].shape}")
    print(f"\nPredicted probabilities: {output['probabilities'].squeeze().tolist()}")