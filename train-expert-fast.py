#!/usr/bin/env python3

"""
Fast Expert ViT Path Planning Model Training with NumPy

This optimized version uses NumPy for vectorized operations, making it much faster.

Usage: python3 train-expert-fast.py
"""

import json
import random
import time
import numpy as np

# Model hyperparameters (must match vit-planner.html)
GRID_SIZE = 16
PATCH_SIZE = 4
EMBEDDING_DIM = 32
NUM_PATCHES = (GRID_SIZE // PATCH_SIZE) ** 2  # 16 patches

# Training hyperparameters
TRAINING_SAMPLES = 500
EPOCHS = 500
LEARNING_RATE = 0.05
BATCH_SIZE = 50  # Larger batches for better vectorization


class SimpleViT:
    """Simplified Vision Transformer with NumPy acceleration"""

    def __init__(self):
        # Initialize weights with He initialization
        scale_embed = np.sqrt(2.0 / (PATCH_SIZE * PATCH_SIZE * 4))
        self.patch_embedding = np.random.randn(PATCH_SIZE * PATCH_SIZE * 4, EMBEDDING_DIM) * scale_embed

        scale_attn = np.sqrt(2.0 / EMBEDDING_DIM)
        self.attention = {
            'wq': np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * scale_attn,
            'wk': np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * scale_attn,
            'wv': np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * scale_attn
        }

        scale_dense = np.sqrt(2.0 / EMBEDDING_DIM)
        self.dense = np.random.randn(EMBEDDING_DIM, 256) * scale_dense

        scale_output = np.sqrt(2.0 / 256)
        self.output = np.random.randn(256, GRID_SIZE * GRID_SIZE) * scale_output

    def extract_patches(self, input_grid):
        """Extract patches from input grid - vectorized"""
        patches = []
        patches_per_side = GRID_SIZE // PATCH_SIZE

        for p_row in range(patches_per_side):
            for p_col in range(patches_per_side):
                patch = []
                for i in range(PATCH_SIZE):
                    for j in range(PATCH_SIZE):
                        row = p_row * PATCH_SIZE + i
                        col = p_col * PATCH_SIZE + j
                        patch.extend(input_grid[row, col])
                patches.append(patch)

        return np.array(patches)

    def forward(self, input_grid):
        """Forward pass - vectorized with NumPy"""
        # Extract patches
        patches = self.extract_patches(input_grid)  # [num_patches, patch_dim]

        # Embed patches: patches @ patch_embedding
        embedded = patches @ self.patch_embedding  # [num_patches, embedding_dim]

        # Simple self-attention (averaged)
        avg_embedding = np.mean(embedded, axis=0)  # [embedding_dim]

        # Dense layer with ReLU
        dense_out = avg_embedding @ self.dense  # [256]
        dense_out = np.maximum(0, dense_out)  # ReLU

        # Output layer with sigmoid
        output = dense_out @ self.output  # [grid_size * grid_size]
        output = 1 / (1 + np.exp(-np.clip(output, -20, 20)))  # Sigmoid with clipping

        return output

    def forward_batch(self, input_batch):
        """Forward pass for a batch of inputs"""
        batch_size = len(input_batch)
        outputs = np.zeros((batch_size, GRID_SIZE * GRID_SIZE))

        for i, input_grid in enumerate(input_batch):
            outputs[i] = self.forward(input_grid)

        return outputs


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
    """Train the expert model with NumPy acceleration"""
    print('=== Training Expert ViT Path Planning Model (NumPy Accelerated) ===\n')
    print('Configuration:')
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
        if (i + 1) % 100 == 0:
            elapsed = time.time() - gen_start
            print(f'  Generated {i + 1}/{TRAINING_SAMPLES} samples ({elapsed:.1f}s)')

    gen_time = time.time() - gen_start
    print(f'✓ Training data ready in {gen_time:.1f}s\n')

    # Initialize model
    print('Initializing model...')
    model = SimpleViT()
    print('✓ Model initialized\n')

    # Training loop
    print('Training...')
    train_start = time.time()

    # Pre-extract all inputs and outputs for faster access
    inputs = np.array([sample['input'] for sample in training_data])
    outputs = np.array([sample['output'] for sample in training_data])

    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0

        # Shuffle data each epoch
        indices = np.random.permutation(TRAINING_SAMPLES)

        # Process in batches
        for batch_start in range(0, TRAINING_SAMPLES, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, TRAINING_SAMPLES)
            batch_indices = indices[batch_start:batch_end]
            batch_size = len(batch_indices)

            batch_inputs = inputs[batch_indices]
            batch_outputs = outputs[batch_indices]

            # Accumulate gradients over batch
            output_grad = np.zeros_like(model.output)
            dense_grad = np.zeros_like(model.dense)

            batch_loss = 0

            for i in range(batch_size):
                predicted = model.forward(batch_inputs[i])
                target = batch_outputs[i]

                # MSE loss
                loss = np.mean((predicted - target) ** 2)
                batch_loss += loss

                # Gradient
                gradient = 2 * (predicted - target) / len(predicted)

                # Accumulate gradients (simplified - only updating output and dense layers)
                # Get intermediate activations for backprop
                patches = model.extract_patches(batch_inputs[i])
                embedded = patches @ model.patch_embedding
                avg_embedding = np.mean(embedded, axis=0)
                dense_out = avg_embedding @ model.dense
                dense_out_relu = np.maximum(0, dense_out)

                # Output layer gradient
                output_grad += np.outer(dense_out_relu, gradient)

                # Dense layer gradient (through ReLU)
                dense_upstream = gradient @ model.output.T
                dense_upstream *= (dense_out > 0)  # ReLU derivative
                dense_grad += np.outer(avg_embedding, dense_upstream)

            # Update weights with averaged gradients
            model.output -= LEARNING_RATE * output_grad / batch_size
            model.dense -= LEARNING_RATE * dense_grad / batch_size * 0.1  # Lower LR for earlier layers

            epoch_loss += batch_loss
            num_batches += 1

        avg_loss = epoch_loss / TRAINING_SAMPLES

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - train_start
            samples_per_sec = (epoch + 1) * TRAINING_SAMPLES / elapsed
            print(f'  Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.6f} - Time: {elapsed:.1f}s - {samples_per_sec:.0f} samples/s')

    total_time = time.time() - train_start
    print(f'\n✓ Training complete in {total_time:.1f}s\n')
    print(f'  Average speed: {TRAINING_SAMPLES * EPOCHS / total_time:.0f} samples/second')

    return model


def export_weights(model, filename):
    """Export model weights to JSON"""
    print('Exporting model weights...')

    weights = {
        'patchEmbedding': model.patch_embedding.tolist(),
        'attention': {
            'wq': model.attention['wq'].tolist(),
            'wk': model.attention['wk'].tolist(),
            'wv': model.attention['wv'].tolist()
        },
        'dense': model.dense.tolist(),
        'output': model.output.tolist(),
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
    print('NumPy version:', np.__version__)
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
