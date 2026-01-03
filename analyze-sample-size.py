#!/usr/bin/env python3

"""
Analyze optimal sample size for ViT path planning model

This script calculates:
1. Problem space complexity (possible grid configurations)
2. Model capacity (number of parameters)
3. Recommended sample size based on ML best practices
"""

import math

# Model configuration
GRID_SIZE = 16
PATCH_SIZE = 4
EMBEDDING_DIM = 32

print("=" * 70)
print("ViT Path Planning Model: Sample Size Analysis")
print("=" * 70)
print()

# 1. Problem Space Analysis
print("1. PROBLEM SPACE COMPLEXITY")
print("-" * 70)

total_cells = GRID_SIZE * GRID_SIZE
print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {total_cells} cells")

# Theoretical maximum configurations (ignoring start/goal constraints)
# Each cell can be: wall, start, goal, or empty
# But realistically: 1 start, 1 goal, rest are wall or empty
# So approximately: C(256,2) * 2^254 for choosing start/goal and wall placement
# This is astronomically large, so let's consider practical space

print(f"\nTheoretical grid configurations (with walls):")
print(f"  - Each cell: wall or empty = 2^{total_cells}")
print(f"  - This equals: ~10^{int(total_cells * 0.301)} configurations (astronomical!)")

# More realistic: considering typical wall density
typical_wall_density = 0.3
expected_walls = int(total_cells * typical_wall_density)
print(f"\nRealistic configurations (30% wall density):")
print(f"  - ~{expected_walls} walls on average")
print(f"  - Start/goal positions: {total_cells} × {total_cells-1} = {total_cells * (total_cells-1):,} combinations")
print(f"  - Wall placements: C({total_cells}, {expected_walls}) ≈ {math.comb(min(total_cells, 50), min(expected_walls, 25)):,} (subset shown)")

print()

# 2. Model Capacity Analysis
print("2. MODEL CAPACITY")
print("-" * 70)

# Count parameters in our model
patch_dim = PATCH_SIZE * PATCH_SIZE * 4  # 4 channels
patch_embedding_params = patch_dim * EMBEDDING_DIM
attention_params = 3 * (EMBEDDING_DIM * EMBEDDING_DIM)  # wq, wk, wv
dense_params = EMBEDDING_DIM * 256
output_params = 256 * total_cells

total_params = patch_embedding_params + attention_params + dense_params + output_params

print(f"Layer-by-layer parameter count:")
print(f"  Patch Embedding: {patch_dim} × {EMBEDDING_DIM} = {patch_embedding_params:,} params")
print(f"  Attention (Q,K,V): 3 × ({EMBEDDING_DIM} × {EMBEDDING_DIM}) = {attention_params:,} params")
print(f"  Dense Layer: {EMBEDDING_DIM} × 256 = {dense_params:,} params")
print(f"  Output Layer: 256 × {total_cells} = {output_params:,} params")
print(f"\n  TOTAL PARAMETERS: {total_params:,}")

print()

# 3. Sample Size Recommendations
print("3. SAMPLE SIZE RECOMMENDATIONS")
print("-" * 70)

# Rule of thumb in ML: 10-50 samples per parameter for simple problems
# For neural networks: often 5-10x the number of parameters
# For our regression task, let's use different ratios

ratios = {
    "Absolute Minimum (overfitting risk)": 1,
    "Conservative (basic patterns)": 5,
    "Standard (good generalization)": 10,
    "Recommended (robust learning)": 20,
    "Ideal (excellent generalization)": 50,
}

print(f"Based on {total_params:,} parameters:\n")

for description, ratio in ratios.items():
    samples = ratio * total_params
    print(f"  {description:.<45} {samples:>8,} samples")

print()

# 4. Practical Considerations
print("4. PRACTICAL CONSIDERATIONS")
print("-" * 70)

# Path diversity analysis
avg_path_length = GRID_SIZE * 1.5  # Heuristic: diagonal is ~22.6, typical is ~24
print(f"Average path characteristics:")
print(f"  - Shortest possible path: {GRID_SIZE - 1 + GRID_SIZE - 1} cells (Manhattan distance)")
print(f"  - Typical path length: ~{int(avg_path_length)} cells")
print(f"  - Grid complexity increases with walls and obstacles")

print(f"\nDistinct path patterns to learn:")
print(f"  - Straight paths (no obstacles)")
print(f"  - Simple detours (single obstacle)")
print(f"  - Complex mazes (multiple obstacles)")
print(f"  - Corner cases (start/goal near edges)")
print(f"  - Unreachable scenarios (isolated regions)")

# Estimate unique path patterns
# Each unique wall configuration creates different paths
# With 30% wall density, approximately:
unique_patterns_estimate = 1000  # Conservative estimate for distinct learnable patterns
print(f"\n  Estimated unique patterns to learn: ~{unique_patterns_estimate:,}")

print()

# 5. Memory and Training Time Analysis
print("5. MEMORY & TRAINING TIME ESTIMATES")
print("-" * 70)

bytes_per_sample = total_cells * 4 * 4 + total_cells * 4  # input (4 channels, 4 bytes) + output
kb_per_sample = bytes_per_sample / 1024

print(f"Memory requirements:")
print(f"  - Per sample: ~{bytes_per_sample:,} bytes ({kb_per_sample:.1f} KB)")

for description, ratio in ratios.items():
    samples = ratio * total_params
    mb = samples * kb_per_sample / 1024
    print(f"  - {samples:>8,} samples: {mb:>8.1f} MB")

# Training time estimates (based on our benchmark: 2,086 samples/second for 500 epochs)
samples_per_sec = 2086
epochs = 500

print(f"\nTraining time estimates ({epochs} epochs at {samples_per_sec:,} samples/sec):")
for description, ratio in ratios.items():
    samples = ratio * total_params
    total_training_samples = samples * epochs
    seconds = total_training_samples / samples_per_sec
    minutes = seconds / 60
    hours = minutes / 60

    if hours > 1:
        time_str = f"{hours:.1f} hours"
    elif minutes > 1:
        time_str = f"{minutes:.1f} minutes"
    else:
        time_str = f"{seconds:.0f} seconds"

    print(f"  - {samples:>8,} samples: {time_str:>12}")

print()

# 6. Recommendations
print("6. FINAL RECOMMENDATIONS")
print("-" * 70)

# Calculate recommended sample size
# For our task: regression on paths, simplified ViT
# We want enough samples to learn path patterns but not overfit
recommended_min = 10 * total_params
recommended_max = 50 * total_params

print(f"For your ViT path planning model ({total_params:,} parameters):\n")
print(f"  Minimum (quick experiments):     {1 * total_params:>8,} samples")
print(f"  Good (basic path learning):      {5 * total_params:>8,} samples")
print(f"  Better (robust generalization):  {10 * total_params:>8,} samples ✓ RECOMMENDED")
print(f"  Best (production quality):       {20 * total_params:>8,} samples")
print(f"  Overkill (diminishing returns):  {50 * total_params:>8,} samples")

print(f"\nCurrent configuration:")
print(f"  You're training with: 500 samples")
print(f"  Ratio: {500 / total_params:.2f} samples per parameter")
print(f"  Status: {'⚠️  SEVERELY UNDERTRAINED' if 500 < recommended_min / 10 else '✓ Adequate for demos'}")

# Specific recommendations
ratio_current = 500 / total_params
if ratio_current < 1:
    print(f"\n  ⚠️  WARNING: With only {500} samples for {total_params:,} parameters,")
    print(f"     your model will likely memorize training data without generalizing!")
    print(f"\n  📊 RECOMMENDATION: Increase to at least {10 * total_params:,} samples")
    print(f"     This would take approximately {(10 * total_params * epochs / samples_per_sec / 60):.1f} minutes to train.")

print()

# 7. Optimization Suggestions
print("7. OPTIMIZATION SUGGESTIONS")
print("-" * 70)

print("To improve performance with current sample size:")
print("  1. Reduce model capacity (fewer parameters)")
print(f"     - Example: Reduce embedding dim from {EMBEDDING_DIM} to 16")
print(f"     - Example: Reduce dense layer from 256 to 128")
print("  2. Use data augmentation")
print("     - Rotate/flip grid configurations")
print("     - Swap start and goal positions")
print("  3. Add regularization")
print("     - L2 weight penalty")
print("     - Dropout layers")
print("  4. Increase training data (recommended)")
print(f"     - Scale up to {10 * total_params:,}+ samples")

print()
print("=" * 70)
