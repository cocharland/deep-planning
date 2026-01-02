#!/usr/bin/env node

/**
 * Train Expert ViT Path Planning Model
 *
 * This script trains a Vision Transformer model offline to learn path planning.
 * The trained weights are exported to a JSON file that can be loaded into the browser.
 *
 * Usage: node train-expert.js
 */

const fs = require('fs');

// Model hyperparameters (must match vit-planner.html)
const GRID_SIZE = 16;
const PATCH_SIZE = 4;
const EMBEDDING_DIM = 32;
const NUM_PATCHES = (GRID_SIZE / PATCH_SIZE) ** 2; // 16 patches

// Training hyperparameters
const TRAINING_SAMPLES = 1000;
const EPOCHS = 1000;
const LEARNING_RATE = 0.05;
const BATCH_SIZE = 10;

// Initialize random weights
function initWeights(input, output) {
    const weights = [];
    const scale = Math.sqrt(2.0 / input);
    for (let i = 0; i < input; i++) {
        weights[i] = [];
        for (let j = 0; j < output; j++) {
            weights[i][j] = (Math.random() - 0.5) * scale;
        }
    }
    return weights;
}

// Simple ViT model (matches browser implementation)
class SimpleViT {
    constructor() {
        this.patchEmbedding = initWeights(PATCH_SIZE * PATCH_SIZE * 4, EMBEDDING_DIM);
        this.attention = {
            wq: initWeights(EMBEDDING_DIM, EMBEDDING_DIM),
            wk: initWeights(EMBEDDING_DIM, EMBEDDING_DIM),
            wv: initWeights(EMBEDDING_DIM, EMBEDDING_DIM)
        };
        this.dense = initWeights(EMBEDDING_DIM, 256);
        this.output = initWeights(256, GRID_SIZE * GRID_SIZE);
    }

    extractPatches(input) {
        const patches = [];
        const patchesPerSide = GRID_SIZE / PATCH_SIZE;

        for (let pRow = 0; pRow < patchesPerSide; pRow++) {
            for (let pCol = 0; pCol < patchesPerSide; pCol++) {
                const patch = [];
                for (let i = 0; i < PATCH_SIZE; i++) {
                    for (let j = 0; j < PATCH_SIZE; j++) {
                        const row = pRow * PATCH_SIZE + i;
                        const col = pCol * PATCH_SIZE + j;
                        patch.push(...input[row][col]);
                    }
                }
                patches.push(patch);
            }
        }
        return patches;
    }

    forward(input) {
        // Extract patches
        const patches = this.extractPatches(input);

        // Embed patches
        const embedded = patches.map(patch => {
            let result = new Array(EMBEDDING_DIM).fill(0);
            for (let i = 0; i < patch.length; i++) {
                for (let j = 0; j < EMBEDDING_DIM; j++) {
                    result[j] += patch[i] * this.patchEmbedding[i][j];
                }
            }
            return result;
        });

        // Simple self-attention (averaged)
        const avgEmbedding = new Array(EMBEDDING_DIM).fill(0);
        for (const emb of embedded) {
            for (let i = 0; i < EMBEDDING_DIM; i++) {
                avgEmbedding[i] += emb[i];
            }
        }
        for (let i = 0; i < EMBEDDING_DIM; i++) {
            avgEmbedding[i] /= embedded.length;
        }

        // Dense layer
        const denseOut = new Array(256).fill(0);
        for (let i = 0; i < EMBEDDING_DIM; i++) {
            for (let j = 0; j < 256; j++) {
                denseOut[j] += avgEmbedding[i] * this.dense[i][j];
            }
        }

        // ReLU activation
        for (let i = 0; i < denseOut.length; i++) {
            denseOut[i] = Math.max(0, denseOut[i]);
        }

        // Output layer
        const output = new Array(GRID_SIZE * GRID_SIZE).fill(0);
        for (let i = 0; i < 256; i++) {
            for (let j = 0; j < GRID_SIZE * GRID_SIZE; j++) {
                output[j] += denseOut[i] * this.output[i][j];
            }
        }

        // Sigmoid activation
        for (let i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-output[i]));
        }

        return output;
    }
}

// A* algorithm for generating training data
function astar(grid, start, goal) {
    const openSet = [start];
    const cameFrom = new Map();
    const gScore = new Map();
    const fScore = new Map();

    gScore.set(`${start.row},${start.col}`, 0);
    fScore.set(`${start.row},${start.col}`, heuristic(start, goal));

    while (openSet.length > 0) {
        // Find node with lowest fScore
        let current = openSet[0];
        let currentIdx = 0;
        for (let i = 1; i < openSet.length; i++) {
            const key = `${openSet[i].row},${openSet[i].col}`;
            const currKey = `${current.row},${current.col}`;
            if (fScore.get(key) < fScore.get(currKey)) {
                current = openSet[i];
                currentIdx = i;
            }
        }

        if (current.row === goal.row && current.col === goal.col) {
            return reconstructPath(cameFrom, current);
        }

        openSet.splice(currentIdx, 1);

        // Check neighbors
        const neighbors = getNeighbors(current, grid);
        for (const neighbor of neighbors) {
            const key = `${neighbor.row},${neighbor.col}`;
            const currentKey = `${current.row},${current.col}`;
            const tentativeGScore = gScore.get(currentKey) + 1;

            if (!gScore.has(key) || tentativeGScore < gScore.get(key)) {
                cameFrom.set(key, current);
                gScore.set(key, tentativeGScore);
                fScore.set(key, tentativeGScore + heuristic(neighbor, goal));

                if (!openSet.some(n => n.row === neighbor.row && n.col === neighbor.col)) {
                    openSet.push(neighbor);
                }
            }
        }
    }

    return null; // No path found
}

function heuristic(a, b) {
    return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
}

function getNeighbors(cell, grid) {
    const neighbors = [];
    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    for (const [dr, dc] of dirs) {
        const newRow = cell.row + dr;
        const newCol = cell.col + dc;

        if (newRow >= 0 && newRow < GRID_SIZE &&
            newCol >= 0 && newCol < GRID_SIZE &&
            grid[newRow][newCol] === 0) {
            neighbors.push({ row: newRow, col: newCol });
        }
    }

    return neighbors;
}

function reconstructPath(cameFrom, current) {
    const path = [current];
    let key = `${current.row},${current.col}`;

    while (cameFrom.has(key)) {
        current = cameFrom.get(key);
        path.unshift(current);
        key = `${current.row},${current.col}`;
    }

    return path;
}

// Generate random training sample
function generateRandomSample() {
    // Create empty grid
    const gridState = Array(GRID_SIZE).fill(0).map(() => Array(GRID_SIZE).fill(0));

    // Add random walls (20-40% coverage)
    const wallDensity = 0.2 + Math.random() * 0.2;
    for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
            if (Math.random() < wallDensity) {
                gridState[row][col] = 1;
            }
        }
    }

    // Random start and goal
    let startRow, startCol, goalRow, goalCol;
    do {
        startRow = Math.floor(Math.random() * GRID_SIZE);
        startCol = Math.floor(Math.random() * GRID_SIZE);
    } while (gridState[startRow][startCol] === 1);

    do {
        goalRow = Math.floor(Math.random() * GRID_SIZE);
        goalCol = Math.floor(Math.random() * GRID_SIZE);
    } while (gridState[goalRow][goalCol] === 1 ||
             (goalRow === startRow && goalCol === startCol));

    gridState[startRow][startCol] = 0;
    gridState[goalRow][goalCol] = 0;

    const start = { row: startRow, col: startCol };
    const goal = { row: goalRow, col: goalCol };

    // Find path using A*
    const path = astar(gridState, start, goal);

    if (!path) {
        return generateRandomSample(); // Try again if no path
    }

    // Create input tensor (4 channels: wall, start, goal, empty)
    const input = [];
    for (let row = 0; row < GRID_SIZE; row++) {
        input[row] = [];
        for (let col = 0; col < GRID_SIZE; col++) {
            const isWall = gridState[row][col] === 1 ? 1 : 0;
            const isStart = (row === startRow && col === startCol) ? 1 : 0;
            const isGoal = (row === goalRow && col === goalCol) ? 1 : 0;
            const isEmpty = (isWall === 0 && isStart === 0 && isGoal === 0) ? 1 : 0;
            input[row][col] = [isWall, isStart, isGoal, isEmpty];
        }
    }

    // Create output tensor (binary: 1 for path, 0 otherwise)
    const output = Array(GRID_SIZE).fill(0).map(() => Array(GRID_SIZE).fill(0));
    for (const cell of path) {
        output[cell.row][cell.col] = 1;
    }

    return { input, output };
}

// Training function
function train() {
    console.log('=== Training Expert ViT Path Planning Model ===\n');
    console.log(`Configuration:`);
    console.log(`  Grid size: ${GRID_SIZE}x${GRID_SIZE}`);
    console.log(`  Patch size: ${PATCH_SIZE}x${PATCH_SIZE}`);
    console.log(`  Embedding dim: ${EMBEDDING_DIM}`);
    console.log(`  Training samples: ${TRAINING_SAMPLES}`);
    console.log(`  Epochs: ${EPOCHS}`);
    console.log(`  Learning rate: ${LEARNING_RATE}`);
    console.log(`  Batch size: ${BATCH_SIZE}\n`);

    // Generate training data
    console.log('Generating training data...');
    const trainingData = [];
    for (let i = 0; i < TRAINING_SAMPLES; i++) {
        trainingData.push(generateRandomSample());
        if ((i + 1) % 100 === 0) {
            console.log(`  Generated ${i + 1}/${TRAINING_SAMPLES} samples`);
        }
    }
    console.log('✓ Training data ready\n');

    // Initialize model
    console.log('Initializing model...');
    const model = new SimpleViT();
    console.log('✓ Model initialized\n');

    // Training loop
    console.log('Training...');
    const startTime = Date.now();

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        let totalLoss = 0;

        // Process in batches
        for (let batch = 0; batch < trainingData.length; batch += BATCH_SIZE) {
            const batchData = trainingData.slice(batch, batch + BATCH_SIZE);

            for (const sample of batchData) {
                const predicted = model.forward(sample.input);
                const target = sample.output.flat();

                // MSE loss
                let loss = 0;
                for (let i = 0; i < predicted.length; i++) {
                    loss += Math.pow(predicted[i] - target[i], 2);
                }
                loss /= predicted.length;
                totalLoss += loss;

                // Gradient descent
                const gradient = predicted.map((p, i) => 2 * (p - target[i]) / predicted.length);

                // Update output layer
                for (let i = 0; i < 256; i++) {
                    for (let j = 0; j < GRID_SIZE * GRID_SIZE; j++) {
                        model.output[i][j] -= LEARNING_RATE * gradient[j] * 0.5;
                    }
                }

                // Update dense layer
                for (let i = 0; i < EMBEDDING_DIM; i++) {
                    for (let j = 0; j < 256; j++) {
                        model.dense[i][j] -= LEARNING_RATE * gradient[j % GRID_SIZE] * 0.01;
                    }
                }
            }
        }

        const avgLoss = totalLoss / trainingData.length;

        if ((epoch + 1) % 50 === 0) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            console.log(`  Epoch ${epoch + 1}/${EPOCHS} - Loss: ${avgLoss.toFixed(6)} - Time: ${elapsed}s`);
        }
    }

    const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\n✓ Training complete in ${totalTime}s\n`);

    return model;
}

// Export weights to JSON
function exportWeights(model, filename) {
    console.log('Exporting model weights...');

    const weights = {
        patchEmbedding: model.patchEmbedding,
        attention: model.attention,
        dense: model.dense,
        output: model.output,
        metadata: {
            gridSize: GRID_SIZE,
            patchSize: PATCH_SIZE,
            embeddingDim: EMBEDDING_DIM,
            trainingSamples: TRAINING_SAMPLES,
            epochs: EPOCHS,
            learningRate: LEARNING_RATE,
            timestamp: new Date().toISOString()
        }
    };

    fs.writeFileSync(filename, JSON.stringify(weights, null, 2));

    const fileSize = (fs.statSync(filename).size / 1024).toFixed(1);
    console.log(`✓ Weights saved to ${filename} (${fileSize} KB)\n`);
}

// Main execution
function main() {
    const model = train();
    exportWeights(model, 'expert-weights.json');

    console.log('=== Training Complete ===');
    console.log('Next steps:');
    console.log('1. The expert weights are saved in expert-weights.json');
    console.log('2. These will be embedded into vit-planner.html');
    console.log('3. Users can instantly load the expert model in their browser');
}

main();
