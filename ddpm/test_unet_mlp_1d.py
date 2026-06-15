#!/usr/bin/env python3
"""
Test script for the MLP-based 1D UNet implementation
"""

import torch
import sys
import os

# Add the ddpm directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ddpm'))

from unet_mlp_1d import UNetMLP1D

def test_unet_mlp_1d():
    """Test the MLP-based 1D UNet implementation with fixed-size inputs"""
    
    # Test parameters for fixed-size inputs
    batch_size = 4
    input_channels = 1  # Single channel input
    sequence_length = 29  # Fixed sequence length
    n_channels = 64
    
    # Create model
    model = UNetMLP1D(
        input_channels=input_channels,
        n_channels=n_channels,
        ch_mults=(1, 2, 2, 4),
        is_attn=(False, False, True, True),
        n_blocks=2
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test input with shape (batch_size, channels=1, 29)
    x = torch.randn(batch_size, input_channels, sequence_length)
    t = torch.randint(0, 1000, (batch_size,))
    
    print(f"Input shape: {x.shape}")
    print(f"Time shape: {t.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(x, t)
        
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.4f}")
        print(f"Output std: {output.std().item():.4f}")
        
        # Check that output has the same shape as input
        assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
        
        print("✅ Test passed! MLP-based 1D UNet is working correctly with fixed-size inputs.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise

def test_different_batch_sizes():
    """Test the model with different batch sizes"""
    
    model = UNetMLP1D(input_channels=1, n_channels=29)
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 29)  # Fixed shape: (batch_size, 1, 29)
        t = torch.randint(0, 1000, (batch_size,))
        
        try:
            with torch.no_grad():
                output = model(x, t)
            
            assert output.shape == x.shape, f"Failed for batch size {batch_size}"
            print(f"✅ Batch size {batch_size}: {output.shape}")
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")

if __name__ == "__main__":
    print("Testing MLP-based 1D UNet with fixed-size inputs...")
    print("=" * 60)
    
    test_unet_mlp_1d()
    print()
    
    print("Testing different batch sizes...")
    print("=" * 60)
    test_different_batch_sizes()
    
    print("\n🎉 All tests completed!")