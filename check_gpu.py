#!/usr/bin/env python3
"""Quick GPU / CUDA diagnostics when PyTorch reports GPU unavailable."""

import os
import sys
import subprocess

def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
    except Exception as e:
        return type('R', (), {'returncode': -1, 'stdout': '', 'stderr': str(e)})()

print("=== 1. Python & PyTorch ===")
print("Python:", sys.executable, sys.version.split()[0])
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA (build):", repr(torch.version.cuda))
    print("cuDNN version:", getattr(torch.backends.cudnn, "version", lambda: None)())
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
except Exception as e:
    print("Error:", e)

print("\n=== 2. nvidia-smi (driver + GPU visible to OS) ===")
r = run("nvidia-smi")
print(r.stdout or r.stderr or "(no output)")
if r.returncode != 0:
    print("-> nvidia-smi failed; driver or GPU not visible to system.")

print("\n=== 3. LD_LIBRARY_PATH (can affect CUDA runtime) ===")
print(os.environ.get("LD_LIBRARY_PATH", "(not set)"))

print("\n=== 4. PyTorch CUDA runtime load test ===")
try:
    import torch
    # Force a small CUDA op to see real error
    x = torch.tensor([1.0]).cuda()
    print("-> Success: tensor created on GPU.")
except Exception as e:
    print("-> Error when using CUDA:", e)
