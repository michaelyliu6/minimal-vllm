"""
Custom attention implementation that uses our KV cache instead of HF's built-in caching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from kv_cache import KVCache

class CustomGPT2Attention(nn.Module):
    """
    Custom GPT-2 attention layer that uses our KV cache implementation.
    
    This replaces the standard GPT-2 attention with our own implementation
    that gives us full control over the KV cache.
    """
    
    def __init__(self, original_attention: nn.Module, layer_idx: int, kv_cache: KVCache):
        super().__init__()
        
        # Copy the original attention's parameters
        self.layer_idx = layer_idx
        self.kv_cache = kv_cache
        
        # Get configuration from original attention
        self.embed_dim = original_attention.embed_dim
        self.num_heads = original_attention.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        # Copy the weights from original attention
        self.c_attn = original_attention.c_attn  # Combined Q, K, V projection
        self.c_proj = original_attention.c_proj  # Output projection
        self.attn_dropout = original_attention.attn_dropout
        self.resid_dropout = original_attention.resid_dropout
        
        # Copy other attributes
        self.scale_attn_weights = getattr(original_attention, 'scale_attn_weights', True)
        self.is_cross_attention = getattr(original_attention, 'is_cross_attention', False)
        
        # For causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((1024, 1024), dtype=torch.bool)).view(1, 1, 1024, 1024),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
    
    def _attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        """
        # Calculate attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(value.size(-1))
        
        # Apply causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        """
        batch_size, seq_length = tensor.shape[:2]
        tensor = tensor.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_dim)
    
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge heads back to original shape.
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_length = tensor.shape[:2]
        return tensor.view(batch_size, seq_length, self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass using our custom KV cache.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            layer_past: Past key-value pairs (we'll ignore this and use our cache)
            attention_mask: Optional attention mask
            head_mask: Optional head mask (not implemented)
            encoder_hidden_states: For cross-attention (not implemented)
            encoder_attention_mask: For cross-attention (not implemented)
            use_cache: Whether to use cache
            output_attentions: Whether to return attention weights
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)
        
        # Split into heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        present = None
        if use_cache and hasattr(self, 'kv_cache') and self.kv_cache is not None:
            # Get current cache position
            cache_position = self.kv_cache.current_seq_len
            
            # Initialize cache on first use
            if cache_position == 0:
                # First time - allocate cache with a reasonable size
                max_seq_len = 1024  # Increased from 512
                self.kv_cache.allocate(
                    batch_size=batch_size,
                    seq_len=max_seq_len,
                    hidden_size=self.embed_dim,
                    num_heads=self.num_heads
                )
            
            # Store the NEW key/value in cache BEFORE concatenation
            # This ensures we store only the new tokens
            self.kv_cache.update(self.layer_idx, key, value, cache_position)
            
            # Update sequence length only for the first layer to avoid multiple updates
            if self.layer_idx == 0:
                self.kv_cache.update_seq_len(seq_length)
            
            # If we have cached KV pairs, retrieve and concatenate for attention
            if cache_position > 0 and self.layer_idx in self.kv_cache.cache:
                # Get cached key and value from position 0 to current cache position
                cached_key, cached_value = self.kv_cache.get(self.layer_idx, 0, cache_position)
                
                # Concatenate cached KV with new KV for attention computation
                key = torch.cat([cached_key, key], dim=2)  # dim=2 is sequence length
                value = torch.cat([cached_value, value], dim=2)
            
            # Return current key/value for HF compatibility
            present = (key, value)
        else:
            # Standard attention without caching
            if use_cache:
                present = (key, value)
        
        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        # Merge heads and project output
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs

def replace_attention_layers(model: nn.Module, kv_cache: KVCache) -> nn.Module:
    """
    Replace all GPT-2 attention layers with our custom implementation.
    
    Args:
        model: The HF GPT-2 model
        kv_cache: Our custom KV cache instance
    
    Returns:
        Modified model with custom attention layers
    """
    for layer_idx, layer in enumerate(model.transformer.h):
        # Replace the attention layer
        original_attention = layer.attn
        custom_attention = CustomGPT2Attention(original_attention, layer_idx, kv_cache)
        layer.attn = custom_attention
    
    return model 