#!/usr/bin/env python3

"""
Fast Expert ViT Path Planning Model Training with GPU Acceleration

This optimized version uses PyTorch with MPS (Metal Performance Shaders) for GPU acceleration on macOS.

Usage: python3 train-expert-fast.py
"""

import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model hyperparameters (must match vit-planner.html)
GRID_SIZE = 16
PATCH_SIZE = 4
EMBEDDING_DIM = 32
NUM_PATCHES = (GRID_SIZE // PATCH_SIZE) ** 2  # 16 patches

# Training hyperparameters
TRAINING_SAMPLES = 5000000  # 5 million samples for comprehensive training
EPOCHS = 100  # Reduced epochs since we have much more data
LEARNING_RATE = 0.001  # Lower LR for stable training with Adam
BATCH_SIZE = 2048  # Much larger batches for GPU

# Device will be set in main()
DEVICE = None


class SimpleViT(nn.Module):
    """Simplified Vision Transformer with GPU acceleration"""

    def __init__(self):
        super(SimpleViT, self).__init__()

        # Patch embedding layer
        patch_dim = PATCH_SIZE * PATCH_SIZE * 4
        self.patch_embedding = nn.Linear(patch_dim, EMBEDDING_DIM)

        # Attention weights (not used in forward but kept for compatibility)
        self.wq = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.wk = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.wv = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

        # Dense and output layers
        self.dense = nn.Linear(EMBEDDING_DIM, 256)
        self.output = nn.Linear(256, GRID_SIZE * GRID_SIZE)

        # Initialize weights with He initialization
        nn.init.kaiming_normal_(self.patch_embedding.weight)
        nn.init.kaiming_normal_(self.dense.weight)
        nn.init.kaiming_normal_(self.output.weight)

    def extract_patches(self, input_grid):
        """Extract patches from input grid - works with batches"""
        batch_size = input_grid.shape[0]
        patches_per_side = GRID_SIZE // PATCH_SIZE
        num_patches = patches_per_side * patches_per_side
        patch_dim = PATCH_SIZE * PATCH_SIZE * 4

        patches = torch.zeros(batch_size, num_patches, patch_dim, device=input_grid.device)

        patch_idx = 0
        for p_row in range(patches_per_side):
            for p_col in range(patches_per_side):
                # Extract patch for all batches at once
                row_start = p_row * PATCH_SIZE
                row_end = row_start + PATCH_SIZE
                col_start = p_col * PATCH_SIZE
                col_end = col_start + PATCH_SIZE

                patch = input_grid[:, row_start:row_end, col_start:col_end, :]
                # Flatten the patch: [batch, patch_h, patch_w, channels] -> [batch, patch_dim]
                patches[:, patch_idx, :] = patch.reshape(batch_size, -1)
                patch_idx += 1

        return patches

    def forward(self, input_grid):
        """Forward pass - fully batched and GPU accelerated"""
        # Extract patches: [batch, num_patches, patch_dim]
        patches = self.extract_patches(input_grid)

        # Embed patches: [batch, num_patches, embedding_dim]
        embedded = self.patch_embedding(patches)

        # Average pooling over patches: [batch, embedding_dim]
        avg_embedding = embedded.mean(dim=1)

        # Dense layer with ReLU: [batch, 256]
        dense_out = F.relu(self.dense(avg_embedding))

        # Output layer with sigmoid: [batch, grid_size * grid_size]
        output = torch.sigmoid(self.output(dense_out))

        return output


def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(cell, grid):
    """Get valid neighboring cells"""
    neighbors = []
    row, col = cell
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if (0 <= new_row < GRID_SIZE and
            0 <= new_col < GRID_SIZE and
            grid[new_row, new_col] == 0):
            neighbors.append((new_row, new_col))

    return neighbors


def astar(grid, start, goal):
    """A* pathfinding algorithm"""
    open_set = [start]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        # Find node with lowest f_score
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.insert(0, current)
            return path

        open_set.remove(current)

        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    open_set.append(neighbor)

    return None


def generate_random_sample():
    """Generate a random training sample - returns NumPy arrays"""
    # Create empty grid
    grid_state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    # Add random walls (20-40% coverage)
    wall_density = 0.2 + random.random() * 0.2
    wall_mask = np.random.random((GRID_SIZE, GRID_SIZE)) < wall_density
    grid_state[wall_mask] = 1

    # Random start and goal
    empty_cells = np.argwhere(grid_state == 0)
    if len(empty_cells) < 2:
        return generate_random_sample()

    indices = np.random.choice(len(empty_cells), 2, replace=False)
    start = tuple(empty_cells[indices[0]])
    goal = tuple(empty_cells[indices[1]])

    grid_state[start] = 0
    grid_state[goal] = 0

    # Find path using A*
    path = astar(grid_state, start, goal)

    if not path:
        return generate_random_sample()

    # Create input tensor (4 channels: wall, start, goal, empty)
    input_grid = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)

    # Channel 0: walls
    input_grid[:, :, 0] = grid_state

    # Channel 1: start
    input_grid[start[0], start[1], 1] = 1

    # Channel 2: goal
    input_grid[goal[0], goal[1], 2] = 1

    # Channel 3: empty
    input_grid[:, :, 3] = 1 - input_grid[:, :, :3].sum(axis=2)

    # Create output tensor (binary: 1 for path, 0 otherwise)
    output_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for cell in path:
        output_grid[cell[0], cell[1]] = 1

    return {'input': input_grid, 'output': output_grid.flatten()}


def train():
    """Train the expert model with GPU acceleration"""
    print('=== Training Expert ViT Path Planning Model (GPU Accelerated) ===\n')
    print('Configuration:')
    print(f'  Device: {DEVICE}')
    print(f'  Grid size: {GRID_SIZE}x{GRID_SIZE}')
    print(f'  Patch size: {PATCH_SIZE}x{PATCH_SIZE}')
    print(f'  Embedding dim: {EMBEDDING_DIM}')
    print(f'  Training samples: {TRAINING_SAMPLES}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Learning rate: {LEARNING_RATE}')
    print(f'  Batch size: {BATCH_SIZE}\n')

    # Generate training data
    print('Generating training data...')
    training_data = []
    gen_start = time.time()

    for i in range(TRAINING_SAMPLES):
        training_data.append(generate_random_sample())
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - gen_start
            rate = (i + 1) / elapsed
            eta = (TRAINING_SAMPLES - i - 1) / rate
            print(f'  Generated {i + 1}/{TRAINING_SAMPLES} samples ({elapsed:.1f}s, ETA: {eta:.1f}s)')

    gen_time = time.time() - gen_start
    print(f'✓ Training data ready in {gen_time:.1f}s\n')

    # Initialize model and move to device
    print('Initializing model...')
    model = SimpleViT().to(DEVICE)
    print('✓ Model initialized\n')

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Convert training data to tensors
    print('Converting data to tensors...')
    inputs = torch.tensor(np.array([sample['input'] for sample in training_data]), dtype=torch.float32)
    outputs = torch.tensor(np.array([sample['output'] for sample in training_data]), dtype=torch.float32)
    print(f'✓ Data converted (inputs: {inputs.shape}, outputs: {outputs.shape})\n')

    # Training loop
    print('Training...')
    train_start = time.time()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0

        # Shuffle data each epoch
        indices = torch.randperm(TRAINING_SAMPLES)

        # Process in batches
        for batch_start in range(0, TRAINING_SAMPLES, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, TRAINING_SAMPLES)
            batch_indices = indices[batch_start:batch_end]

            # Move batch to device
            batch_inputs = inputs[batch_indices].to(DEVICE)
            batch_targets = outputs[batch_indices].to(DEVICE)

            # Forward pass
            predictions = model(batch_inputs)

            # Compute loss
            loss = criterion(predictions, batch_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * len(batch_indices)
            num_batches += 1

        avg_loss = epoch_loss / TRAINING_SAMPLES

        # Track improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            improvement = "↓"
        else:
            improvement = " "

        # More frequent logging at the start, then every 10
        if epoch < 5 or (epoch + 1) % 10 == 0:
            elapsed = time.time() - train_start
            samples_per_sec = (epoch + 1) * TRAINING_SAMPLES / elapsed
            print(f'  Epoch {epoch + 1:3d}/{EPOCHS} - Loss: {avg_loss:.6f} {improvement} - Time: {elapsed:.1f}s - {samples_per_sec:.0f} samples/s')

    total_time = time.time() - train_start
    print(f'\n✓ Training complete in {total_time:.1f}s\n')
    print(f'  Average speed: {TRAINING_SAMPLES * EPOCHS / total_time:.0f} samples/second')

    return model


def export_weights(model, filename):
    """Export model weights to JSON"""
    print('Exporting model weights...')

    # Move model to CPU for export
    model = model.cpu()

    # Extract weights from PyTorch model
    weights = {
        'patchEmbedding': model.patch_embedding.weight.detach().numpy().T.tolist(),  # Transpose for compatibility
        'attention': {
            'wq': model.wq.weight.detach().numpy().T.tolist(),
            'wk': model.wk.weight.detach().numpy().T.tolist(),
            'wv': model.wv.weight.detach().numpy().T.tolist()
        },
        'dense': model.dense.weight.detach().numpy().T.tolist(),  # Transpose for compatibility
        'output': model.output.weight.detach().numpy().T.tolist(),  # Transpose for compatibility
        'metadata': {
            'gridSize': GRID_SIZE,
            'patchSize': PATCH_SIZE,
            'embeddingDim': EMBEDDING_DIM,
            'trainingSamples': TRAINING_SAMPLES,
            'epochs': EPOCHS,
            'learningRate': LEARNING_RATE,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }
    }

    with open(filename, 'w') as f:
        json.dump(weights, f)

    import os
    file_size = os.path.getsize(filename) / 1024
    print(f'✓ Weights saved to {filename} ({file_size:.1f} KB)\n')


def main():
    """Main execution"""
    global DEVICE

    print('PyTorch version:', torch.__version__)
    print('NumPy version:', np.__version__)
    print('MPS available:', torch.backends.mps.is_available())
    print()

    # Set up device (use MPS on macOS if available, otherwise CPU)
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using Metal Performance Shaders (MPS) GPU acceleration")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA GPU acceleration")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")
    print()

    model = train()
    export_weights(model, 'expert-weights.json')

    print('=== Training Complete ===')
    print('Next steps:')
    print('1. The expert weights are saved in expert-weights.json')
    print('2. Run the embedding script to integrate into vit-planner.html')
    print('3. Users can instantly load the expert model in their browser')


if __name__ == '__main__':
    main()
