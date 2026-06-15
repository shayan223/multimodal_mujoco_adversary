"""
---
title: Lightweight 1D U-Net with Attention for Diffusion Models
summary: >
  Lightweight 1D UNet model with attention for Denoising Diffusion Probabilistic Models (DDPM) 
  optimized for 1D feature vectors with fast runtime
---

# Lightweight 1D U-Net with Attention

This is a lightweight 1D U-Net model specifically designed for processing 1D feature vectors
in diffusion models. It includes attention mechanisms while keeping the model relatively shallow
and fast for real-time applications.

Key features:
- Lightweight architecture optimized for 1D data
- Multi-head attention at key resolutions
- Fast runtime suitable for real-time applications
- Compatible with existing diffusion model interface
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
import torch.nn.functional as F


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Time step embeddings using sinusoidal position encoding
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ResidualBlock(nn.Module):
    """
    Lightweight residual block for 1D data with time conditioning
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float = 0.1):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act1 = Swish()
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = Swish()
        
        # Time embedding projection
        self.time_emb = nn.Linear(time_channels, out_channels)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Time embedding
        t_emb = self.time_emb(t)[:, :, None]  # [batch, out_channels, 1]
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + t_emb  # Add time embedding
        h = self.act1(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        
        # Shortcut connection
        shortcut = self.shortcut(x)
        return h + shortcut


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for 1D sequences
    """
    def __init__(self, n_channels: int, n_heads: int = 4, d_k: int = None):
        super().__init__()
        if d_k is None:
            d_k = n_channels // n_heads
            
        self.n_heads = n_heads
        self.d_k = d_k
        
        # Projections for Q, K, V
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        
        # Normalization
        self.norm = nn.GroupNorm(1, n_channels)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        batch_size, n_channels, length = x.shape
        
        # Reshape to [batch, length, channels]
        x_reshaped = x.permute(0, 2, 1)
        
        # Apply normalization
        x_norm = self.norm(x)
        x_norm_reshaped = x_norm.permute(0, 2, 1)
        
        # Get Q, K, V
        qkv = self.projection(x_norm_reshaped)
        qkv = qkv.view(batch_size, length, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Scaled dot-product attention
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        
        # Apply attention to values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch_size, length, self.n_heads * self.d_k)
        
        # Output projection
        res = self.output(res)
        
        # Add residual connection
        res = res + x_reshaped
        
        # Reshape back to [batch, channels, length]
        return res.permute(0, 2, 1)


class DownBlock(nn.Module):
    """
    Downsampling block with optional attention
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool = False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with skip connections and optional attention
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool = False):
        super().__init__()
        # Input has in_channels + out_channels due to skip connection
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.upsample = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        x = self.upsample(x)
        return x


class MiddleBlock(nn.Module):
    """
    Middle block with attention at the lowest resolution
    """
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UNet1D_V2(nn.Module):
    """
    Lightweight 1D U-Net with attention for diffusion models
    
    Simplified architecture that avoids complex skip connection handling
    """
    
    def __init__(self, 
                 input_channels: int = 1, 
                 n_channels: int = 32,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True),
                 n_blocks: int = 1):
        """
        Args:
            input_channels: Number of input channels
            n_channels: Base number of channels
            ch_mults: Channel multipliers for each resolution level
            is_attn: Whether to use attention at each resolution
            n_blocks: Number of residual blocks per resolution
        """
        super().__init__()
        
        n_resolutions = len(ch_mults)
        
        # Input projection
        self.input_proj = nn.Conv1d(input_channels, n_channels, kernel_size=1)
        
        # Time embedding
        self.time_emb = TimeEmbedding(n_channels * 4)
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        in_ch = n_channels
        
        for i in range(n_resolutions):
            out_ch = n_channels * ch_mults[i]
            
            # Create downsampling block
            down_block = nn.ModuleList()
            
            # Residual blocks
            for _ in range(n_blocks):
                down_block.append(ResidualBlock(in_ch, out_ch, n_channels * 4))
                in_ch = out_ch
            
            # Attention
            if is_attn[i]:
                down_block.append(AttentionBlock(out_ch))
            
            # Downsample
            if i < n_resolutions - 1:
                down_block.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            
            self.down_layers.append(down_block)
        
        # Middle block
        self.middle = MiddleBlock(in_ch, n_channels * 4)
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        
        for i in reversed(range(n_resolutions)):
            out_ch = n_channels * ch_mults[i]
            
            # Create upsampling block
            up_block = nn.ModuleList()
            
            # Upsample first (except for first iteration)
            if i < n_resolutions - 1:
                up_block.append(nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=2, padding=1))
            
            # Skip connection handling - need to account for doubled channels after concatenation
            skip_ch = out_ch  # This will be the skip connection channels
            total_ch = in_ch + skip_ch  # Total channels after concatenation
            
            # Residual blocks - input channels will be doubled due to skip connection
            for _ in range(n_blocks):
                up_block.append(ResidualBlock(total_ch, out_ch, n_channels * 4))
                total_ch = out_ch  # Update for next block
            
            # Attention
            if is_attn[i]:
                up_block.append(AttentionBlock(out_ch))
            
            # Channel reduction for next level
            if i > 0:
                next_ch = n_channels * ch_mults[i-1]
                up_block.append(nn.Conv1d(out_ch, next_ch, kernel_size=1))
                in_ch = next_ch
            
            self.up_layers.append(up_block)
        
        # Output projection
        self.output_proj = nn.Conv1d(in_ch, input_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Forward pass through the U-Net
        
        Args:
            x: Input tensor [batch, channels, length]
            t: Time tensor [batch]
        """
        # Get time embeddings
        t_emb = self.time_emb(t)
        
        # Input projection
        x = self.input_proj(x)
        
        # Store skip connections
        skip_connections = [x]
        
        # Downsampling path
        for down_block in self.down_layers:
            for layer in down_block:
                if isinstance(layer, (ResidualBlock, AttentionBlock)):
                    x = layer(x, t_emb)
                else:  # Downsampling conv
                    x = layer(x)
                    skip_connections.append(x)
        
        # Middle block
        x = self.middle(x, t_emb)
        
        # Upsampling path
        for up_block in self.up_layers:
            skip_added = False
            for layer in up_block:
                if isinstance(layer, nn.ConvTranspose1d):
                    x = layer(x)
                elif isinstance(layer, ResidualBlock):
                    # Add skip connection before residual block if not already added
                    if not skip_added and len(skip_connections) > 1:
                        skip = skip_connections.pop()
                        # Handle size mismatch
                        if skip.shape[2] != x.shape[2]:
                            skip = F.interpolate(skip, size=x.shape[2], mode='nearest')
                        x = torch.cat([x, skip], dim=1)
                        skip_added = True
                    x = layer(x, t_emb)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x, t_emb)
                else:  # Channel reduction conv
                    x = layer(x)
        
        # Final output projection
        x = self.output_proj(x)
        
        return x


# Alias for compatibility
UNet1D = UNet1D_V2
