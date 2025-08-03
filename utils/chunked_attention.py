"""
Efficient attention implementations for long sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange


def chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 512,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Compute attention in chunks to reduce memory usage.
    
    Args:
        query: Query tensor (batch, heads, seq_len, head_dim)
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        chunk_size: Size of chunks for computation
        mask: Attention mask (batch, seq_len) or (batch, heads, seq_len, seq_len)
        dropout_p: Dropout probability
        scale: Scale factor for attention scores
        
    Returns:
        Attention output (batch, heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Initialize output
    output = torch.zeros_like(query)
    
    # Process query chunks
    for i in range(0, seq_len, chunk_size):
        q_start = i
        q_end = min(i + chunk_size, seq_len)
        q_chunk = query[:, :, q_start:q_end]
        
        # Initialize chunk output
        chunk_output = torch.zeros_like(q_chunk)
        
        # Process key/value chunks
        for j in range(0, seq_len, chunk_size):
            k_start = j
            k_end = min(j + chunk_size, seq_len)
            
            k_chunk = key[:, :, k_start:k_end]
            v_chunk = value[:, :, k_start:k_end]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
            
            # Apply mask if provided
            if mask is not None:
                if mask.dim() == 2:
                    # Broadcast mask
                    chunk_mask = mask[:, None, q_start:q_end, None] * mask[:, None, None, k_start:k_end]
                    scores = scores.masked_fill(~chunk_mask.bool(), float('-inf'))
                elif mask.dim() == 4:
                    chunk_mask = mask[:, :, q_start:q_end, k_start:k_end]
                    scores = scores.masked_fill(~chunk_mask.bool(), float('-inf'))
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply dropout
            if dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
            # Apply attention to values
            chunk_output += torch.matmul(attn_weights, v_chunk)
        
        # Store chunk output
        output[:, :, q_start:q_end] = chunk_output
    
    return output


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention implementation with multiple strategies."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        attention_mode: str = 'auto'
    ):
        """
        Initialize memory-efficient attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            attention_mode: 'flash', 'chunked', 'standard', or 'auto'
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.attention_mode = attention_mode
        
        # Check available attention implementations
        self.flash_available = False
        try:
            from flash_attn import flash_attn_func
            self.flash_available = True
        except ImportError:
            pass
        
        # Projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Determine chunk size for chunked attention
        self.chunk_size = min(512, max_seq_len // 4)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with automatic attention mode selection.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', 
                       three=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        
        # Determine attention mode
        if self.attention_mode == 'auto':
            if self.flash_available and seq_len <= 2048:
                mode = 'flash'
            elif seq_len > 1024:
                mode = 'chunked'
            else:
                mode = 'standard'
        else:
            mode = self.attention_mode
        
        # Compute attention based on mode
        if mode == 'flash' and self.flash_available:
            attn_output = self._flash_attention(q, k, v, mask)
            attn_weights = None
        elif mode == 'chunked':
            attn_output = self._chunked_attention(q, k, v, mask)
            attn_weights = None
        else:
            attn_output, attn_weights = self._standard_attention(q, k, v, mask)
        
        # Reshape and project output
        attn_output = rearrange(attn_output, 'b s h d -> b s (h d)')
        output = self.out_proj(attn_output)
        
        if return_attention and attn_weights is None:
            # Recompute attention weights if needed
            q = rearrange(q, 'b s h d -> b h s d')
            k = rearrange(k, 'b s h d -> b h s d')
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None, None, :].bool(), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
        
        return output, attn_weights
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Flash Attention implementation."""
        from flash_attn import flash_attn_func
        
        # Flash attention requires fp16 or bf16
        dtype_orig = q.dtype
        if q.dtype not in [torch.float16, torch.bfloat16]:
            q = q.half()
            k = k.half()
            v = v.half()
        
        # Flash attention expects (batch, seqlen, nheads, headdim)
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
            return_attn_probs=False
        )
        
        # Convert back to original dtype if needed
        if dtype_orig not in [torch.float16, torch.bfloat16]:
            attn_output = attn_output.to(dtype_orig)
        
        return attn_output
    
    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Chunked attention for memory efficiency."""
        # Rearrange for chunked computation
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')
        
        # Compute chunked attention
        output = chunked_attention(
            q, k, v,
            chunk_size=self.chunk_size,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # Rearrange back
        output = rearrange(output, 'b h s d -> b s h d')
        
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard attention implementation."""
        # Rearrange for attention computation
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :].bool(), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Rearrange back
        output = rearrange(output, 'b h s d -> b s h d')
        
        return output, attn_weights


class LinearAttention(nn.Module):
    """Linear attention for O(n) complexity on very long sequences."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feature_map: str = 'elu'
    ):
        """
        Initialize linear attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            feature_map: Feature map type ('elu' or 'softmax')
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.feature_map = feature_map
        
        # Projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with linear attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d', 
                           three=3, h=self.num_heads)
        
        # Apply feature map
        if self.feature_map == 'elu':
            q = F.elu(q) + 1
            k = F.elu(k) + 1
        else:  # softmax approximation
            q = F.softmax(q / self.temperature, dim=-1)
            k = F.softmax(k / self.temperature, dim=-1)
        
        # Apply mask to keys
        if mask is not None:
            k = k * mask[:, None, :, None]
        
        # Linear attention: (Q @ K^T) @ V = Q @ (K^T @ V)
        # Compute K^T @ V first for O(n) complexity
        kv = torch.einsum('bhsd,bhse->bhde', k, v)
        
        # Then Q @ (K^T @ V)
        output = torch.einsum('bhsd,bhde->bhse', q, kv)
        
        # Normalize by key sum
        k_sum = k.sum(dim=-2, keepdim=True)
        output = output / (k_sum + 1e-6)
        
        # Reshape and project
        output = rearrange(output, 'b h s d -> b s (h d)')
        output = self.out_proj(output)
        
        return output


if __name__ == "__main__":
    # Test efficient attention implementations
    batch_size = 2
    seq_len = 2048
    d_model = 512
    num_heads = 8
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len)
    mask[0, seq_len//2:] = 0
    
    # Test memory efficient attention
    print("Testing Memory Efficient Attention...")
    attn = MemoryEfficientAttention(d_model, num_heads, attention_mode='auto')
    output, weights = attn(x, mask)
    print(f"Output shape: {output.shape}")
    
    # Test linear attention
    print("\nTesting Linear Attention...")
    linear_attn = LinearAttention(d_model, num_heads)
    output = linear_attn(x, mask)
    print(f"Output shape: {output.shape}")
    
    # Memory usage comparison
    print("\nMemory efficiency comparison:")
    print(f"Standard attention memory: O({seq_len}²) = O({seq_len**2:,})")
    print(f"Chunked attention memory: O({seq_len} × chunk_size) = O({seq_len * 512:,})")
    print(f"Linear attention memory: O({seq_len}) = O({seq_len:,})")