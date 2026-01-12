#!/usr/bin/env python3
"""Debug script to test model training on small dataset"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Copy constants from main script
GRID_SIZE = 16
PATCH_SIZE = 4
EMBEDDING_DIM = 32

# Small test set
TRAINING_SAMPLES = 100
EPOCHS = 20
LEARNING_RATE = 0.001  # Start with smaller LR
BATCH_SIZE = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

# Copy model class
class SimpleViT(nn.Module):
    def __init__(self):
        super(SimpleViT, self).__init__()
        patch_dim = PATCH_SIZE * PATCH_SIZE * 4
        self.patch_embedding = nn.Linear(patch_dim, EMBEDDING_DIM)
        self.wq = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.wk = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.wv = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dense = nn.Linear(EMBEDDING_DIM, 256)
        self.output = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        nn.init.kaiming_normal_(self.patch_embedding.weight)
        nn.init.kaiming_normal_(self.dense.weight)
        nn.init.kaiming_normal_(self.output.weight)

    def extract_patches(self, input_grid):
        batch_size = input_grid.shape[0]
        patches_per_side = GRID_SIZE // PATCH_SIZE
        num_patches = patches_per_side * patches_per_side
        patch_dim = PATCH_SIZE * PATCH_SIZE * 4

        patches = torch.zeros(batch_size, num_patches, patch_dim, device=input_grid.device)

        patch_idx = 0
        for p_row in range(patches_per_side):
            for p_col in range(patches_per_side):
                row_start = p_row * PATCH_SIZE
                row_end = row_start + PATCH_SIZE
                col_start = p_col * PATCH_SIZE
                col_end = col_start + PATCH_SIZE

                patch = input_grid[:, row_start:row_end, col_start:col_end, :]
                patches[:, patch_idx, :] = patch.reshape(batch_size, -1)
                patch_idx += 1

        return patches

    def forward(self, input_grid):
        patches = self.extract_patches(input_grid)
        embedded = self.patch_embedding(patches)
        avg_embedding = embedded.mean(dim=1)
        dense_out = F.relu(self.dense(avg_embedding))
        output = torch.sigmoid(self.output(dense_out))
        return output

# Simple training data generator
def generate_simple_sample():
    """Generate simple test case - straight line path"""
    input_grid = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
    output_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Create a simple vertical path
    col = GRID_SIZE // 2
    start_row = 2
    end_row = GRID_SIZE - 3

    # Set channels: empty=1 everywhere
    input_grid[:, :, 3] = 1.0

    # Start position
    input_grid[start_row, col, 1] = 1.0
    input_grid[start_row, col, 3] = 0.0

    # Goal position
    input_grid[end_row, col, 2] = 1.0
    input_grid[end_row, col, 3] = 0.0

    # Path
    for row in range(start_row, end_row + 1):
        output_grid[row, col] = 1.0

    return {'input': input_grid, 'output': output_grid.flatten()}

# Generate test data
print("Generating test data...")
data = [generate_simple_sample() for _ in range(TRAINING_SAMPLES)]
inputs = torch.tensor(np.array([d['input'] for d in data]), dtype=torch.float32)
outputs = torch.tensor(np.array([d['output'] for d in data]), dtype=torch.float32)
print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")

# Check data
print(f"Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
print(f"Output mean: {outputs.mean():.3f}")
print()

# Test model
model = SimpleViT().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Testing forward pass...")
test_input = inputs[0:1].to(device)
test_output = model(test_input)
print(f"Model output shape: {test_output.shape}")
print(f"Model output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
print(f"Model output mean: {test_output.mean():.3f}")
print()

# Training loop with detailed logging
print("Training...")
for epoch in range(EPOCHS):
    epoch_loss = 0

    indices = torch.randperm(TRAINING_SAMPLES)

    for batch_start in range(0, TRAINING_SAMPLES, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TRAINING_SAMPLES)
        batch_indices = indices[batch_start:batch_end]

        batch_inputs = inputs[batch_indices].to(device)
        batch_targets = outputs[batch_indices].to(device)

        # Forward
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        if epoch == 0 and batch_start == 0:
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append((name, grad_norm))
            print("First batch gradients:")
            for name, norm in grad_norms:
                print(f"  {name}: {norm:.6f}")
            print()

        optimizer.step()

        epoch_loss += loss.item() * len(batch_indices)

    avg_loss = epoch_loss / TRAINING_SAMPLES

    # Test prediction
    with torch.no_grad():
        test_pred = model(inputs[0:1].to(device))
        test_target = outputs[0:1].to(device)
        test_loss = criterion(test_pred, test_target).item()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} - Loss: {avg_loss:.6f} - Test: {test_loss:.6f} - Pred range: [{test_pred.min():.3f}, {test_pred.max():.3f}]")

print("\nFinal test:")
with torch.no_grad():
    test_input = inputs[0:1].to(device)
    test_output = model(test_input).cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    target_output = outputs[0].numpy().reshape(GRID_SIZE, GRID_SIZE)

    print("Target path (first 8 rows):")
    print(target_output[:8, :8].astype(int))
    print("\nPredicted (first 8 rows):")
    print((test_output[:8, :8] > 0.5).astype(int))
