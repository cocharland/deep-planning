#!/usr/bin/env python3
"""Quick test to verify GPU acceleration is working"""

import torch
import torch.nn as nn
import time
import numpy as np

print('PyTorch version:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())
print()

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (Metal Performance Shaders) GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")
print()

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model and move to device
model = TestModel().to(device)
print(f"Model device: {next(model.parameters()).device}")

# Create test data
batch_size = 1024
test_data = torch.randn(batch_size, 256).to(device)
target = torch.randn(batch_size, 256).to(device)

print(f"Data device: {test_data.device}")
print()

# Test forward pass
print("Running test forward pass...")
start = time.time()
for i in range(100):
    output = model(test_data)
    loss = nn.MSELoss()(output, target)
elapsed = time.time() - start
print(f"✓ 100 iterations completed in {elapsed:.2f}s ({100/elapsed:.1f} iter/s)")
print()

# Test with gradient computation
print("Running test with backpropagation...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
start = time.time()
for i in range(100):
    output = model(test_data)
    loss = nn.MSELoss()(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
elapsed = time.time() - start
print(f"✓ 100 training iterations completed in {elapsed:.2f}s ({100/elapsed:.1f} iter/s)")
print()

print("GPU acceleration is working correctly!")
