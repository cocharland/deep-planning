#!/usr/bin/env python3

"""
Calculate optimal model architecture for different sample sizes
"""

GRID_SIZE = 16
PATCH_SIZE = 4
total_cells = GRID_SIZE * GRID_SIZE
patch_dim = PATCH_SIZE * PATCH_SIZE * 4

def calculate_params(embedding_dim, dense_dim):
    """Calculate total parameters for given architecture"""
    patch_embedding = patch_dim * embedding_dim
    attention = 3 * (embedding_dim * embedding_dim)
    dense = embedding_dim * dense_dim
    output = dense_dim * total_cells
    total = patch_embedding + attention + dense + output
    return {
        'patch_embedding': patch_embedding,
        'attention': attention,
        'dense': dense,
        'output': output,
        'total': total
    }

print("=" * 80)
print("Optimal Architecture for Different Sample Sizes")
print("=" * 80)
print()

sample_sizes = [500, 5000, 50000, 100000, 500000]
target_ratio = 10  # Want at least 10 samples per parameter

print(f"Target: {target_ratio} samples per parameter for good generalization\n")

for samples in sample_sizes:
    max_params = samples // target_ratio

    print(f"Sample Size: {samples:,}")
    print(f"  Max parameters (at {target_ratio}:1 ratio): {max_params:,}")
    print()

    # Try different architectures
    configs = [
        (8, 64, "Tiny"),
        (16, 128, "Small"),
        (24, 192, "Medium"),
        (32, 256, "Large (current)"),
        (48, 384, "Very Large"),
        (64, 512, "Huge"),
    ]

    best_config = None
    for emb_dim, dense_dim, name in configs:
        params = calculate_params(emb_dim, dense_dim)
        total = params['total']
        ratio = samples / total

        fits = "✓" if total <= max_params else "✗"

        status = ""
        if ratio >= 20:
            status = "Excellent"
        elif ratio >= 10:
            status = "Good"
        elif ratio >= 5:
            status = "Okay"
        elif ratio >= 1:
            status = "Risky"
        else:
            status = "BAD"

        print(f"    {fits} {name:16} (emb={emb_dim:2}, dense={dense_dim:3}): "
              f"{total:>6,} params, ratio={ratio:>5.1f}:1 [{status}]")

        if total <= max_params and best_config is None:
            best_config = (emb_dim, dense_dim, name, total)

    if best_config:
        print(f"\n    → RECOMMENDED: {best_config[2]} with {best_config[3]:,} parameters")

    print()
    print("-" * 80)
    print()

# Specific recommendation for 500 samples
print("=" * 80)
print("SPECIFIC RECOMMENDATION FOR 500 SAMPLES")
print("=" * 80)
print()

samples = 500
max_params = samples // target_ratio

print(f"With {samples} samples, you should have ≤ {max_params} parameters\n")

# Find the best architecture
recommended = (8, 64)  # embedding_dim, dense_dim
params = calculate_params(*recommended)

print(f"Recommended Architecture:")
print(f"  Embedding Dim: {recommended[0]}")
print(f"  Dense Layer: {recommended[1]}")
print(f"\n  Total Parameters: {params['total']:,}")
print(f"  Samples per parameter: {samples / params['total']:.1f}:1")
print(f"\n  Breakdown:")
print(f"    Patch Embedding: {params['patch_embedding']:,}")
print(f"    Attention: {params['attention']:,}")
print(f"    Dense: {params['dense']:,}")
print(f"    Output: {params['output']:,}")

print("\n" + "=" * 80)
