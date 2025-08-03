"""
Multi-scale feature extraction layers for CSPAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
from einops import rearrange, repeat


class LocalFeatureExtractor(nn.Module):
    """Extract local motif features using multi-kernel convolutions."""
    
    def __init__(
        self,
        in_channels: int = 1280,
        out_channels: int = 256,
        kernel_sizes: List[int] = [3, 5, 7, 9],
        dropout: float = 0.1
    ):
        """
        Initialize local feature extractor.
        
        Args:
            in_channels: Input feature dimension (ESM2 embedding size)
            out_channels: Output channels per kernel
            kernel_sizes: List of kernel sizes for multi-scale convolution
            dropout: Dropout rate
        """
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        
        # Create convolution layers for each kernel size
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=k,
                    padding=k//2  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.convs.append(conv)
        
        # Output projection
        self.output_dim = out_channels * len(kernel_sizes)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, seq_len, channels)
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Local features (batch, seq_len, output_dim)
        """
        # Transpose for conv1d: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply each convolution
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            conv_outputs.append(out)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch, total_channels, seq_len)
        
        # Transpose back: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return x


class RegionalFeatureExtractor(nn.Module):
    """Extract regional domain features using dilated convolutions."""
    
    def __init__(
        self,
        in_channels: int = 1280,
        out_channels: int = 256,
        kernel_size: int = 3,
        dilation_rates: List[int] = [2, 4, 8, 16],
        dropout: float = 0.1
    ):
        """
        Initialize regional feature extractor.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output channels per dilation
            kernel_size: Base kernel size
            dilation_rates: List of dilation rates
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dilation_rates = dilation_rates
        self.out_channels = out_channels
        
        # Create dilated convolution layers
        self.dilated_convs = nn.ModuleList()
        for d in dilation_rates:
            # Calculate padding for 'same' output
            padding = (kernel_size - 1) * d // 2
            
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.dilated_convs.append(conv)
        
        # Output projection
        self.output_dim = out_channels * len(dilation_rates)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, seq_len, channels)
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Regional features (batch, seq_len, output_dim)
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)
        
        # Apply each dilated convolution
        conv_outputs = []
        for conv in self.dilated_convs:
            out = conv(x)
            conv_outputs.append(out)
        
        # Concatenate all outputs
        x = torch.cat(conv_outputs, dim=1)
        
        # Transpose back
        x = x.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return x


class GlobalFeatureExtractor(nn.Module):
    """Extract global features using multi-head self-attention."""
    
    def __init__(
        self,
        in_channels: int = 1280,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Initialize global feature extractor.
        
        Args:
            in_channels: Input feature dimension
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_rope: Whether to use rotary position embeddings
            use_flash_attention: Whether to use Flash Attention (if available)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention
        
        # Check if Flash Attention is available
        self.flash_attention_available = False
        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attention_available = True
                print("Flash Attention is available and will be used")
            except ImportError:
                print("Flash Attention not available, using standard attention")
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Multi-head attention (only used if not using flash attention)
        if not self.flash_attention_available:
            self.attention = nn.MultiheadAttention(
                d_model,
                num_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            # For flash attention, we need separate Q, K, V projections
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.attn_dropout = dropout
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize RoPE if needed
        if use_rope:
            self.rope = RotaryPositionEmbedding(d_model // num_heads)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, seq_len, channels)
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Global features (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Self-attention
        if self.flash_attention_available and self.use_flash_attention:
            attn_output = self._flash_attention_forward(x, mask)
        else:
            # Standard attention path
            if mask is not None:
                # Convert to attention mask format (True = masked)
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            
            # Self-attention with optional RoPE
            if self.use_rope:
                # Apply RoPE to queries and keys
                q = k = v = x
                q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
                
                # Apply rotary embeddings
                q = self.rope(q, seq_len=seq_len)
                k = self.rope(k, seq_len=seq_len)
                
                # Reshape back
                q = rearrange(q, 'b h n d -> b n (h d)')
                k = rearrange(k, 'b h n d -> b n (h d)')
                
                # Manual attention computation
                attn_output = self._manual_attention(q, k, v, attn_mask)
            else:
                attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        # Apply mask to output
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return x
    
    def _flash_attention_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flash Attention forward pass."""
        from flash_attn import flash_attn_func
        
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', 
                       three=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = rearrange(q, 'b s h d -> b h s d')
            k = rearrange(k, 'b s h d -> b h s d')
            q = self.rope(q, seq_len=seq_len)
            k = self.rope(k, seq_len=seq_len)
            q = rearrange(q, 'b h s d -> b s h d')
            k = rearrange(k, 'b h s d -> b s h d')
        
        # Flash Attention
        # Note: flash_attn expects (batch, seqlen, nheads, headdim)
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            causal=False,
            return_attn_probs=False
        )
        
        # Reshape and project output
        attn_output = rearrange(attn_output, 'b s h d -> b s (h d)')
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Manual attention computation for RoPE compatibility."""
        batch_size, seq_len, d_model = q.shape
        
        # Reshape for attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout.p, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        
        return attn_output


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.dim = dim
        
        # Precompute sin/cos embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        sincos = torch.einsum('i,j->ij', position, inv_freq)
        
        self.register_buffer('sin', sincos.sin())
        self.register_buffer('cos', sincos.cos())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary position embedding.
        
        Args:
            x: Input tensor (batch, heads, seq_len, head_dim)
            seq_len: Sequence length
        
        Returns:
            Tensor with position embeddings applied
        """
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Split x into two halves
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        
        # Apply rotation
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return x_rot


class CrossScaleAttention(nn.Module):
    """Cross-scale attention mechanism for feature interaction."""
    
    def __init__(
        self,
        local_dim: int = 1024,
        regional_dim: int = 1024,
        global_dim: int = 512,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-scale attention.
        
        Args:
            local_dim: Local feature dimension
            regional_dim: Regional feature dimension
            global_dim: Global feature dimension
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Projections for queries, keys, and values
        self.q_proj = nn.Linear(global_dim, d_model)
        self.k_proj = nn.Linear(local_dim + regional_dim, d_model)
        self.v_proj = nn.Linear(local_dim + regional_dim, d_model)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        local_features: torch.Tensor,
        regional_features: torch.Tensor,
        global_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            local_features: Local features (batch, seq_len, local_dim)
            regional_features: Regional features (batch, seq_len, regional_dim)
            global_features: Global features (batch, seq_len, global_dim)
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Cross-scale features (batch, seq_len, d_model)
        """
        # Concatenate local and regional features
        multi_scale = torch.cat([local_features, regional_features], dim=-1)
        
        # Project to query, key, value
        q = self.q_proj(global_features)
        k = self.k_proj(multi_scale)
        v = self.v_proj(multi_scale)
        
        # Prepare attention mask
        if mask is not None:
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
        
        # Cross-scale attention
        attn_output, _ = self.attention(q, k, v, attn_mask=attn_mask)
        
        # Residual connection and normalization
        output = self.norm(global_features[:, :, :self.d_model] + self.dropout(attn_output))
        
        # Output projection
        output = self.output_proj(output)
        
        # Apply mask if provided
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        return output


if __name__ == "__main__":
    # Test the layers
    batch_size = 2
    seq_len = 100
    esm_dim = 1280
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, esm_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[0, 80:] = 0  # Mask last 20 positions of first sequence
    
    # Test local feature extractor
    local_extractor = LocalFeatureExtractor(esm_dim)
    local_features = local_extractor(x, mask)
    print(f"Local features shape: {local_features.shape}")
    
    # Test regional feature extractor
    regional_extractor = RegionalFeatureExtractor(esm_dim)
    regional_features = regional_extractor(x, mask)
    print(f"Regional features shape: {regional_features.shape}")
    
    # Test global feature extractor
    global_extractor = GlobalFeatureExtractor(esm_dim)
    global_features = global_extractor(x, mask)
    print(f"Global features shape: {global_features.shape}")
    
    # Test cross-scale attention
    cross_attention = CrossScaleAttention()
    cross_features = cross_attention(local_features, regional_features, global_features, mask)
    print(f"Cross-scale features shape: {cross_features.shape}")