"""
Ablation study model variants for CSPAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import yaml

from models.layers import (
    LocalFeatureExtractor,
    RegionalFeatureExtractor,
    GlobalFeatureExtractor,
    CrossScaleAttention
)
from models.cspan import DimerizationMotifDiscovery, HierarchicalSequenceAggregation


class CSPANNoCross(nn.Module):
    """CSPAN variant without cross-scale attention."""
    
    def __init__(self, config_path: str = "config.yml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']
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
        
        # NO CROSS-SCALE ATTENTION - Direct concatenation instead
        combined_dim = (self.local_extractor.output_dim + 
                       self.regional_extractor.output_dim + 
                       self.model_config['global_features']['d_model'])
        
        # Project combined features
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        # Motif discovery (kept)
        self.motif_discovery = DimerizationMotifDiscovery(
            feature_dim=self.hidden_dim,
            num_motifs=self.num_motifs,
            motif_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # Hierarchical aggregation (kept)
        self.hierarchical_aggregation = HierarchicalSequenceAggregation(
            feature_dim=self.hidden_dim,
            window_size=self.model_config['aggregation']['window_size'],
            stride=self.model_config['aggregation']['stride'],
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout
        )
        
        # Calculate final feature dimension
        pool_multiplier = 3 if self.model_config['aggregation'].get('include_std_pool', True) else 2
        self.final_feature_dim = self.hidden_dim * pool_multiplier + self.hidden_dim + self.hidden_dim
        
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        local_features = self.local_extractor(features, attention_mask)
        regional_features = self.regional_extractor(features, attention_mask)
        global_features = self.global_extractor(features, attention_mask)
        
        # Direct concatenation instead of cross-scale attention
        combined_features = torch.cat([local_features, regional_features, global_features], dim=-1)
        
        # Project to common dimension
        projected_features = self.feature_projection(combined_features)
        
        # Motif discovery
        motif_features, motif_attention = self.motif_discovery(projected_features, attention_mask)
        
        # Hierarchical aggregation
        hierarchical_features = self.hierarchical_aggregation(projected_features, attention_mask)
        
        # Pooling
        if attention_mask is not None:
            masked_features = projected_features * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1, keepdim=True)
            avg_pool = masked_features.sum(dim=1) / lengths
            
            masked_features_for_max = masked_features.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float('-inf')
            )
            max_pool, _ = masked_features_for_max.max(dim=1)
        else:
            avg_pool = projected_features.mean(dim=1)
            max_pool, _ = projected_features.max(dim=1)
        
        if self.model_config['aggregation'].get('include_std_pool', True):
            if attention_mask is not None:
                mean = avg_pool.unsqueeze(1)
                variance = ((masked_features - mean) ** 2).sum(dim=1) / lengths
                std_pool = torch.sqrt(variance + 1e-6)
            else:
                std_pool = projected_features.std(dim=1)
            
            combined_features = torch.cat([avg_pool, max_pool, std_pool, motif_features, hierarchical_features], dim=-1)
        else:
            combined_features = torch.cat([avg_pool, max_pool, motif_features, hierarchical_features], dim=-1)
        
        # Classification
        logits = self.classifier(combined_features)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'motif_attention': motif_attention
        }


class CSPANNoMotif(nn.Module):
    """CSPAN variant without motif discovery module."""
    
    def __init__(self, config_path: str = "config.yml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']
        self.esm_dim = self.model_config['esm_dim']
        self.hidden_dim = self.model_config['hidden_dim']
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
        
        # Cross-scale attention (kept)
        self.cross_scale_attention = CrossScaleAttention(
            local_dim=self.local_extractor.output_dim,
            regional_dim=self.regional_extractor.output_dim,
            global_dim=self.model_config['global_features']['d_model'],
            d_model=self.model_config['cross_scale']['d_model'],
            num_heads=self.model_config['num_heads'],
            dropout=self.dropout
        )
        
        # NO MOTIF DISCOVERY - Skip this component
        
        # Hierarchical aggregation (kept)
        self.hierarchical_aggregation = HierarchicalSequenceAggregation(
            feature_dim=self.model_config['cross_scale']['d_model'],
            window_size=self.model_config['aggregation']['window_size'],
            stride=self.model_config['aggregation']['stride'],
            hidden_dim=self.hidden_dim // 2,
            dropout=self.dropout
        )
        
        # Calculate final feature dimension (no motif features)
        cross_scale_dim = self.model_config['cross_scale']['d_model']
        pool_multiplier = 3 if self.model_config['aggregation'].get('include_std_pool', True) else 2
        self.final_feature_dim = cross_scale_dim * pool_multiplier + cross_scale_dim
        
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        local_features = self.local_extractor(features, attention_mask)
        regional_features = self.regional_extractor(features, attention_mask)
        global_features = self.global_extractor(features, attention_mask)
        
        # Cross-scale attention
        cross_scale_features = self.cross_scale_attention(
            local_features, regional_features, global_features, attention_mask
        )
        
        # NO MOTIF DISCOVERY
        
        # Hierarchical aggregation
        hierarchical_features = self.hierarchical_aggregation(cross_scale_features, attention_mask)
        
        # Pooling
        if attention_mask is not None:
            masked_features = cross_scale_features * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1, keepdim=True)
            avg_pool = masked_features.sum(dim=1) / lengths
            
            masked_features_for_max = masked_features.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float('-inf')
            )
            max_pool, _ = masked_features_for_max.max(dim=1)
        else:
            avg_pool = cross_scale_features.mean(dim=1)
            max_pool, _ = cross_scale_features.max(dim=1)
        
        if self.model_config['aggregation'].get('include_std_pool', True):
            if attention_mask is not None:
                mean = avg_pool.unsqueeze(1)
                variance = ((masked_features - mean) ** 2).sum(dim=1) / lengths
                std_pool = torch.sqrt(variance + 1e-6)
            else:
                std_pool = cross_scale_features.std(dim=1)
            
            combined_features = torch.cat([avg_pool, max_pool, std_pool, hierarchical_features], dim=-1)
        else:
            combined_features = torch.cat([avg_pool, max_pool, hierarchical_features], dim=-1)
        
        # Classification
        logits = self.classifier(combined_features)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'motif_attention': None  # No motif attention
        }


class CSPANSingleScale(nn.Module):
    """CSPAN variant with only global self-attention (no multi-scale)."""
    
    def __init__(self, config_path: str = "config.yml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']
        self.esm_dim = self.model_config['esm_dim']
        self.hidden_dim = self.model_config['hidden_dim']
        self.dropout = self.model_config['dropout']
        
        # ONLY global feature extractor
        self.global_extractor = GlobalFeatureExtractor(
            in_channels=self.esm_dim,
            d_model=self.hidden_dim,
            num_heads=self.model_config['global_features']['num_heads'],
            dropout=self.model_config['global_features']['attention_dropout']
        )
        
        # Simple pooling-based aggregation
        pool_multiplier = 3 if self.model_config['aggregation'].get('include_std_pool', True) else 2
        self.final_feature_dim = self.hidden_dim * pool_multiplier
        
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Only global features
        global_features = self.global_extractor(features, attention_mask)
        
        # Pooling
        if attention_mask is not None:
            masked_features = global_features * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1, keepdim=True)
            avg_pool = masked_features.sum(dim=1) / lengths
            
            masked_features_for_max = masked_features.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float('-inf')
            )
            max_pool, _ = masked_features_for_max.max(dim=1)
        else:
            avg_pool = global_features.mean(dim=1)
            max_pool, _ = global_features.max(dim=1)
        
        if self.model_config['aggregation'].get('include_std_pool', True):
            if attention_mask is not None:
                mean = avg_pool.unsqueeze(1)
                variance = ((masked_features - mean) ** 2).sum(dim=1) / lengths
                std_pool = torch.sqrt(variance + 1e-6)
            else:
                std_pool = global_features.std(dim=1)
            
            combined_features = torch.cat([avg_pool, max_pool, std_pool], dim=-1)
        else:
            combined_features = torch.cat([avg_pool, max_pool], dim=-1)
        
        # Classification
        logits = self.classifier(combined_features)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'motif_attention': None
        }


class ESM2Baseline(nn.Module):
    """Simple ESM2 baseline with mean pooling + MLP."""
    
    def __init__(self, config_path: str = "config.yml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']
        self.esm_dim = self.model_config['esm_dim']
        self.hidden_dim = self.model_config['hidden_dim']
        self.dropout = self.model_config['dropout']
        
        # Simple MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.esm_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Simple mean pooling
        if attention_mask is not None:
            masked_features = features * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1, keepdim=True)
            pooled_features = masked_features.sum(dim=1) / lengths
        else:
            pooled_features = features.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'motif_attention': None
        }


# Model registry
ABLATION_MODELS = {
    'cspan-no-cross': CSPANNoCross,
    'cspan-no-motif': CSPANNoMotif,
    'cspan-single-scale': CSPANSingleScale,
    'esm2-baseline': ESM2Baseline
}


def get_ablation_model(model_name: str, config_path: str = "config.yml"):
    """Get ablation model by name."""
    if model_name not in ABLATION_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(ABLATION_MODELS.keys())}")
    
    return ABLATION_MODELS[model_name](config_path)