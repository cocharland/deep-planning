# Training Recommendations for ViT Path Planner

## Current Situation

- **Model Parameters**: 78,848
- **Current Training**: 500 samples
- **Ratio**: 0.01 samples/parameter (⚠️ SEVERELY UNDERTRAINED)

## Problem

The output layer alone (256 cells × 256 dense = 65,536 params) is already too large for small datasets. This is fundamentally a **model capacity vs. data** mismatch.

---

## Recommended Solutions

### 🎯 Option 1: Increase Samples (Moderate - 20K samples)

**Best for**: Quick improvement with reasonable training time

```bash
# Edit train-expert-fast.py
TRAINING_SAMPLES = 20000  # Up from 500
EPOCHS = 200              # Down from 500 (less overfitting per sample)
```

**Expected**:
- Training time: ~32 minutes
- Ratio: 0.25 samples/parameter (still low, but 50x better!)
- Model should learn basic path patterns
- File size: ~30 MB

---

### 🚀 Option 2: Serious Training (50K samples)

**Best for**: Good generalization

```bash
# Edit train-expert-fast.py
TRAINING_SAMPLES = 50000
EPOCHS = 200
```

**Expected**:
- Training time: ~1.3 hours
- Ratio: 0.63 samples/parameter (getting reasonable)
- Model should generalize well to new grids
- File size: ~75 MB

---

### ⚡ Option 3: Production Quality (100K samples)

**Best for**: Excellent performance

```bash
# Edit train-expert-fast.py
TRAINING_SAMPLES = 100000
EPOCHS = 200
```

**Expected**:
- Training time: ~2.7 hours
- Ratio: 1.27 samples/parameter (acceptable)
- Strong generalization
- File size: ~150 MB

---

### 🎨 Option 4: Reduce Model Complexity

**Best for**: Keeping file size small

Modify the model architecture:

```python
# In train-expert-fast.py and vit-planner.html
EMBEDDING_DIM = 16    # Down from 32
# And in dense layer
self.dense = np.random.randn(EMBEDDING_DIM, 128) * scale_dense  # Down from 256
```

**New parameter count**: ~36,608 parameters
- With 20K samples: ratio = 0.55:1 (decent!)
- Training time: ~15 minutes for 20K samples
- File size: ~15 MB

---

### 💡 Option 5: Data Augmentation (Smart!)

**Best for**: Maximum sample diversity without extra data generation

Generate fewer unique grids but create variations:

```python
def augment_sample(sample):
    """Create 8 variations from one sample: original + 3 rotations + 4 flips"""
    variations = []

    # Original
    variations.append(sample)

    # 3 rotations (90°, 180°, 270°)
    for k in range(1, 4):
        rotated_input = np.rot90(sample['input'], k)
        rotated_output = np.rot90(sample['output'].reshape(16, 16), k).flatten()
        variations.append({'input': rotated_input, 'output': rotated_output})

    # 4 flips (horizontal, vertical, both, diagonal)
    flipped_h_input = np.fliplr(sample['input'])
    flipped_h_output = np.fliplr(sample['output'].reshape(16, 16)).flatten()
    variations.append({'input': flipped_h_input, 'output': flipped_h_output})

    # ... more flips

    return variations
```

With augmentation:
- Generate 5,000 unique grids
- Augment to 40,000 samples (8x multiplier)
- Training time: ~30 minutes for 200 epochs
- Effective ratio: 0.51:1

---

## My Recommendation: Combined Approach 🌟

For best results with reasonable time/size:

1. **Reduce model size slightly**:
   - EMBEDDING_DIM = 24 (down from 32)
   - Dense layer = 192 (down from 256)
   - New parameters: ~57,024

2. **Increase samples moderately**:
   - TRAINING_SAMPLES = 30000
   - EPOCHS = 200

3. **Add simple augmentation**:
   - Flip grids horizontally/vertically
   - Effective samples: ~90,000

**Result**:
- Training time: ~50 minutes
- Ratio: 1.58 samples/parameter ✓
- File size: ~45 MB
- Good generalization expected

---

## Quick Commands

### For 20K samples (Recommended minimum):
```bash
# 1. Edit train-expert-fast.py: Change TRAINING_SAMPLES to 20000, EPOCHS to 200
# 2. Run:
python3 train-expert-fast.py
python3 embed-weights.py
```

### For 50K samples (Recommended good):
```bash
# 1. Edit train-expert-fast.py: Change TRAINING_SAMPLES to 50000, EPOCHS to 200
# 2. Run:
python3 train-expert-fast.py
python3 embed-weights.py
```

### For 100K samples (Recommended best):
```bash
# 1. Edit train-expert-fast.py: Change TRAINING_SAMPLES to 100000, EPOCHS to 200
# 2. Run:
python3 train-expert-fast.py
python3 embed-weights.py
```

---

## File Size Considerations

The HTML file will grow based on sample count. Current (500 samples) = 1.6 MB.

Estimated sizes:
- 500 samples: ~1.7 MB
- 5,000 samples: ~3 MB
- 20,000 samples: ~10 MB
- 50,000 samples: ~25 MB
- 100,000 samples: ~50 MB

**Note**: Modern browsers handle 50 MB files easily. GitHub Pages has a 100 MB file limit.

---

## What Should You Do?

**For demo purposes**: Use **20,000 samples** with **200 epochs**
- Good enough to show path learning
- Reasonable file size (~10 MB)
- Fast training (~32 min)

**For impressive results**: Use **50,000-100,000 samples** with **200 epochs**
- Strong generalization
- Worth the training time for a quality demo
- Larger file but still acceptable
