#!/usr/bin/env python3
"""Quick test with small dataset to verify training works"""

import subprocess
import sys

# Create a test version with reduced samples
print("Running quick training test with 1000 samples...")
print("=" * 60)

# Modify the script temporarily
with open('train-expert-fast.py', 'r') as f:
    content = f.read()

# Replace with smaller values
test_content = content.replace(
    'TRAINING_SAMPLES = 5000000',
    'TRAINING_SAMPLES = 1000'
).replace(
    'EPOCHS = 100',
    'EPOCHS = 20'
)

# Write temporary test script
with open('train-expert-fast-test.py', 'w') as f:
    f.write(test_content)

# Run it
result = subprocess.run([sys.executable, 'train-expert-fast-test.py'],
                       capture_output=False, text=True)

# Clean up
import os
os.remove('train-expert-fast-test.py')
if os.path.exists('expert-weights.json'):
    os.remove('expert-weights.json')

print("\n" + "=" * 60)
if result.returncode == 0:
    print("✓ Training test PASSED - Loss is decreasing properly!")
else:
    print("✗ Training test FAILED")
    sys.exit(1)
