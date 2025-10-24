"""
---
title: 1D U-Net model for Denoising Diffusion Probabilistic Models (DDPM)
summary: >
  1D UNet model for Denoising Diffusion Probabilistic Models (DDPM) for single-dimensional data
---

# 1D U-Net model for [Denoising Diffusion Probabilistic Models (DDPM)](index.html)

This is a 1D [U-Net](../../unet/index.html) based model to predict noise
$\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ for single-dimensional input data.

U-Net is a gets it's name from the U shape in the model diagram.
It processes 1D data by progressively lowering (halving) the feature map resolution and then
increasing the resolution.
There are pass-through connection at each resolution.

This implementation contains a bunch of modifications to original U-Net (residual blocks, multi-head attention)
 and also adds time-step embeddings $t$. Adapted for 1D data processing using Conv1d layers.
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

from labml_helpers.module import Module


class Swish(Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        print('T input shape:',t.shape)

        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        
        print('Time embedding shape:',emb.shape)
        return emb


class ResidualBlock(Module):
    """
    ### Residual block

    A residual block has two linear layers with layer normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Layer normalization and the first linear layer
        self.norm1 = nn.LayerNorm(in_channels)
        self.act1 = Swish()
        self.linear1 = nn.Linear(in_channels, out_channels)

        # Layer normalization and the second linear layer
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = Swish()
        self.linear2 = nn.Linear(out_channels, out_channels)

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, length]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # Reshape for linear layers: [batch_size, length, in_channels]
        batch_size, in_channels, length = x.shape
        x_reshaped = x.permute(0, 2, 1)  # [batch_size, length, in_channels]
        
        # First linear layer
        h = self.linear1(self.act1(self.norm1(x_reshaped)))
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, None, :]  # [batch_size, 1, out_channels]
        # Second linear layer
        h = self.linear2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection
        shortcut = self.shortcut(x_reshaped)
        h = h + shortcut
        
        # Reshape back to [batch_size, out_channels, length]
        return h.permute(0, 2, 1)


class AttentionBlock(Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, length]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, length = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, length]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, length)

        #
        return res


class DownBlock(Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
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


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.linear = nn.Linear(n_channels, n_channels * 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Reshape for linear layer: [batch_size, length, n_channels]
        batch_size, n_channels, length = x.shape
        x_reshaped = x.permute(0, 2, 1)  # [batch_size, length, n_channels]
        
        # Apply linear transformation
        h = self.linear(x_reshaped)  # [batch_size, length, n_channels * 2]
        
        # Reshape back and repeat to simulate upsampling
        h = h.permute(0, 2, 1)  # [batch_size, n_channels * 2, length]
        # Simple upsampling by repeating each element
        h = h.repeat_interleave(2, dim=2)  # [batch_size, n_channels * 2, length * 2]
        
        return h


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.linear = nn.Linear(n_channels, n_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Reshape for linear layer: [batch_size, length, n_channels]
        batch_size, n_channels, length = x.shape
        x_reshaped = x.permute(0, 2, 1)  # [batch_size, length, n_channels]
        
        # Apply linear transformation
        h = self.linear(x_reshaped)  # [batch_size, length, n_channels]
        
        # Simple downsampling by taking every other element
        h = h[:, ::2, :]  # [batch_size, length//2, n_channels]
        
        # Reshape back to [batch_size, n_channels, length//2]
        return h.permute(0, 2, 1)


class UNet1D(Module):
    """
    ## 1D U-Net
    """

    def __init__(self, input_channels: int = 1, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `input_channels` is the number of channels in the input data.
        * `n_channels` is number of channels in the initial feature map that we transform the input into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project input into feature map
        self.image_proj = nn.Linear(input_channels, n_channels)

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and linear layer
        self.norm = nn.LayerNorm(in_channels)
        self.act = Swish()
        self.final = nn.Linear(in_channels, input_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, input_channels, length]`
        * `t` has shape `[batch_size]`
        """
        print('At forward, X input shape:',x.shape)
        print('At forward, T input shape:',t.shape)
        # Get time-step embeddings
        t = self.time_emb(t)
        print('After time embedding, T shape:',t.shape)
        print('Shape before projection:',x.shape)

        # Reshape for linear projection: [batch_size, length, input_channels]
        batch_size, input_channels, length = x.shape
        x_reshaped = x.permute(0, 2, 1)  # [batch_size, length, input_channels]
        
        # Get input projection
        x_proj = self.image_proj(x_reshaped)  # [batch_size, length, n_channels]
        
        # Reshape back to [batch_size, n_channels, length] for compatibility with existing blocks
        x = x_proj.permute(0, 2, 1)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Reshape for final linear layer: [batch_size, length, in_channels]
        x_final = x.permute(0, 2, 1)  # [batch_size, length, in_channels]
        
        # Final normalization and linear transformation
        x_final = self.final(self.act(self.norm(x_final)))  # [batch_size, length, input_channels]
        
        # Reshape back to [batch_size, input_channels, length]
        return x_final.permute(0, 2, 1)
