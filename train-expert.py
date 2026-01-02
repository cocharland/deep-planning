#!/usr/bin/env python3

"""
Train Expert ViT Path Planning Model

This script trains a Vision Transformer model offline to learn path planning.
The trained weights are exported to a JSON file that can be loaded into the browser.

Usage: python3 train-expert.py
"""

import json
import random
import math
import time
from collections import deque

# Model hyperparameters (must match vit-planner.html)
GRID_SIZE = 16
PATCH_SIZE = 4
EMBEDDING_DIM = 32
NUM_PATCHES = (GRID_SIZE // PATCH_SIZE) ** 2  # 16 patches

# Training hyperparameters
TRAINING_SAMPLES = 500
EPOCHS = 500
LEARNING_RATE = 0.05
BATCH_SIZE = 10


def init_weights(input_dim, output_dim):
    """Initialize random weights with He initialization"""
    scale = math.sqrt(2.0 / input_dim)
    weights = []
    for i in range(input_dim):
        row = []
        for j in range(output_dim):
            row.append((random.random() - 0.5) * scale)
        weights.append(row)
    return weights


class SimpleViT:
    """Simplified Vision Transformer (matches browser implementation)"""

    def __init__(self):
        self.patch_embedding = init_weights(PATCH_SIZE * PATCH_SIZE * 4, EMBEDDING_DIM)
        self.attention = {
            'wq': init_weights(EMBEDDING_DIM, EMBEDDING_DIM),
            'wk': init_weights(EMBEDDING_DIM, EMBEDDING_DIM),
            'wv': init_weights(EMBEDDING_DIM, EMBEDDING_DIM)
        }
        self.dense = init_weights(EMBEDDING_DIM, 256)
        self.output = init_weights(256, GRID_SIZE * GRID_SIZE)

    def extract_patches(self, input_grid):
        """Extract patches from input grid"""
        patches = []
        patches_per_side = GRID_SIZE // PATCH_SIZE

        for p_row in range(patches_per_side):
            for p_col in range(patches_per_side):
                patch = []
                for i in range(PATCH_SIZE):
                    for j in range(PATCH_SIZE):
                        row = p_row * PATCH_SIZE + i
                        col = p_col * PATCH_SIZE + j
                        patch.extend(input_grid[row][col])
                patches.append(patch)

        return patches

    def forward(self, input_grid):
        """Forward pass through the network"""
        # Extract patches
        patches = self.extract_patches(input_grid)

        # Embed patches
        embedded = []
        for patch in patches:
            result = [0.0] * EMBEDDING_DIM
            for i in range(len(patch)):
                for j in range(EMBEDDING_DIM):
                    result[j] += patch[i] * self.patch_embedding[i][j]
            embedded.append(result)

        # Simple self-attention (averaged)
        avg_embedding = [0.0] * EMBEDDING_DIM
        for emb in embedded:
            for i in range(EMBEDDING_DIM):
                avg_embedding[i] += emb[i]
        for i in range(EMBEDDING_DIM):
            avg_embedding[i] /= len(embedded)

        # Dense layer
        dense_out = [0.0] * 256
        for i in range(EMBEDDING_DIM):
            for j in range(256):
                dense_out[j] += avg_embedding[i] * self.dense[i][j]

        # ReLU activation
        for i in range(len(dense_out)):
            dense_out[i] = max(0, dense_out[i])

        # Output layer
        output = [0.0] * (GRID_SIZE * GRID_SIZE)
        for i in range(256):
            for j in range(GRID_SIZE * GRID_SIZE):
                output[j] += dense_out[i] * self.output[i][j]

        # Sigmoid activation
        for i in range(len(output)):
            output[i] = 1 / (1 + math.exp(-max(-20, min(20, output[i]))))  # Clamp to prevent overflow

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
            grid[new_row][new_col] == 0):
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

    return None  # No path found


def generate_random_sample():
    """Generate a random training sample"""
    # Create empty grid
    grid_state = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

    # Add random walls (20-40% coverage)
    wall_density = 0.2 + random.random() * 0.2
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if random.random() < wall_density:
                grid_state[row][col] = 1

    # Random start and goal
    while True:
        start_row = random.randint(0, GRID_SIZE - 1)
        start_col = random.randint(0, GRID_SIZE - 1)
        if grid_state[start_row][start_col] == 0:
            break

    while True:
        goal_row = random.randint(0, GRID_SIZE - 1)
        goal_col = random.randint(0, GRID_SIZE - 1)
        if (grid_state[goal_row][goal_col] == 0 and
            (goal_row != start_row or goal_col != start_col)):
            break

    grid_state[start_row][start_col] = 0
    grid_state[goal_row][goal_col] = 0

    start = (start_row, start_col)
    goal = (goal_row, goal_col)

    # Find path using A*
    path = astar(grid_state, start, goal)

    if not path:
        return generate_random_sample()  # Try again if no path

    # Create input tensor (4 channels: wall, start, goal, empty)
    input_grid = []
    for row in range(GRID_SIZE):
        input_row = []
        for col in range(GRID_SIZE):
            is_wall = 1 if grid_state[row][col] == 1 else 0
            is_start = 1 if (row == start_row and col == start_col) else 0
            is_goal = 1 if (row == goal_row and col == goal_col) else 0
            is_empty = 1 if (is_wall == 0 and is_start == 0 and is_goal == 0) else 0
            input_row.append([is_wall, is_start, is_goal, is_empty])
        input_grid.append(input_row)

    # Create output tensor (binary: 1 for path, 0 otherwise)
    output_grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    for cell in path:
        output_grid[cell[0]][cell[1]] = 1

    return {'input': input_grid, 'output': output_grid}


def train():
    """Train the expert model"""
    print('=== Training Expert ViT Path Planning Model ===\n')
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
    for i in range(TRAINING_SAMPLES):
        training_data.append(generate_random_sample())
        if (i + 1) % 100 == 0:
            print(f'  Generated {i + 1}/{TRAINING_SAMPLES} samples')
    print('✓ Training data ready\n')

    # Initialize model
    print('Initializing model...')
    model = SimpleViT()
    print('✓ Model initialized\n')

    # Training loop
    print('Training...')
    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0

        # Process all samples
        for sample in training_data:
            predicted = model.forward(sample['input'])
            target = [cell for row in sample['output'] for cell in row]

            # MSE loss
            loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(predicted)
            total_loss += loss

            # Gradient descent
            gradient = [2 * (p - t) / len(predicted) for p, t in zip(predicted, target)]

            # Update output layer
            for i in range(256):
                for j in range(GRID_SIZE * GRID_SIZE):
                    model.output[i][j] -= LEARNING_RATE * gradient[j] * 0.5

            # Update dense layer
            for i in range(EMBEDDING_DIM):
                for j in range(256):
                    model.dense[i][j] -= LEARNING_RATE * gradient[j % GRID_SIZE] * 0.01

        avg_loss = total_loss / len(training_data)

        if (epoch + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f'  Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.6f} - Time: {elapsed:.1f}s')

    total_time = time.time() - start_time
    print(f'\n✓ Training complete in {total_time:.1f}s\n')

    return model


def export_weights(model, filename):
    """Export model weights to JSON"""
    print('Exporting model weights...')

    weights = {
        'patchEmbedding': model.patch_embedding,
        'attention': model.attention,
        'dense': model.dense,
        'output': model.output,
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
        json.dump(weights, f, indent=2)

    import os
    file_size = os.path.getsize(filename) / 1024
    print(f'✓ Weights saved to {filename} ({file_size:.1f} KB)\n')


def main():
    """Main execution"""
    model = train()
    export_weights(model, 'expert-weights.json')

    print('=== Training Complete ===')
    print('Next steps:')
    print('1. The expert weights are saved in expert-weights.json')
    print('2. These will be embedded into vit-planner.html')
    print('3. Users can instantly load the expert model in their browser')


if __name__ == '__main__':
    main()
