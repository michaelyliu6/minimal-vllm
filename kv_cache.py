"""
Basic KV Cache implementation for storing and retrieving key-value pairs during inference.
"""

from typing import Dict, Optional, Tuple
import torch

class KVCache:
    """
    A simple KV cache that stores key-value pairs for each layer.
    This helps avoid recomputing attention for previously processed tokens.
    """
    
    def __init__(self, num_layers: int, device: str = "cuda"):
        self.num_layers = num_layers
        self.device = device
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.current_seq_len = 0
    
    def allocate(self, batch_size: int, seq_len: int, hidden_size: int, num_heads: int) -> None:
        """
        Allocate space in the cache for a new sequence.
        
        Args:
            batch_size: Number of sequences in the batch
            seq_len: Maximum sequence length
            hidden_size: Size of the hidden dimension
            num_heads: Number of attention heads
        """
        head_dim = hidden_size // num_heads
        
        # Initialize empty cache for each layer
        for layer_idx in range(self.num_layers):
            # Shape: (batch_size, num_heads, seq_len, head_dim)
            key = torch.zeros(
                (batch_size, num_heads, seq_len, head_dim),
                device=self.device
            )
            value = torch.zeros(
                (batch_size, num_heads, seq_len, head_dim),
                device=self.device
            )
            self.cache[layer_idx] = (key, value)
        
        self.current_seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int
    ) -> None:
        """
        Update the cache with new key-value pairs for a specific layer.
        
        Args:
            layer_idx: Index of the transformer layer
            key: New key tensor of shape (batch_size, num_heads, new_tokens, head_dim)
            value: New value tensor of shape (batch_size, num_heads, new_tokens, head_dim)
            start_pos: Starting position in the sequence to update
        """
        if layer_idx not in self.cache:
            raise ValueError(f"Layer {layer_idx} not found in cache")
        
        cache_key, cache_value = self.cache[layer_idx]
        
        # Update the cache at the specified position
        cache_key[:, :, start_pos:start_pos + key.size(2)] = key
        cache_value[:, :, start_pos:start_pos + value.size(2)] = value
        
        self.current_seq_len = max(self.current_seq_len, start_pos + key.size(2))
    
    def get(self, layer_idx: int, start_pos: int, end_pos: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve key-value pairs from the cache for a specific layer and position range.
        
        Args:
            layer_idx: Index of the transformer layer
            start_pos: Starting position in the sequence
            end_pos: Ending position in the sequence (exclusive). If None, returns all cached values.
        
        Returns:
            Tuple of (key, value) tensors for the specified range
        """
        if layer_idx not in self.cache:
            raise ValueError(f"Layer {layer_idx} not found in cache")
        
        cache_key, cache_value = self.cache[layer_idx]
        
        if end_pos is None:
            end_pos = self.current_seq_len
        
        return (
            cache_key[:, :, start_pos:end_pos],
            cache_value[:, :, start_pos:end_pos]
        )
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        self.current_seq_len = 0 