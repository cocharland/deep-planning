#!/usr/bin/env python3

"""
Embed expert model weights into vit-planner.html

This script takes the trained weights from expert-weights.json and embeds them
directly into the HTML file so users can instantly load the expert model.
"""

import json
import re

def main():
    print('=== Embedding Expert Weights into HTML ===\n')

    # Load the trained weights
    print('Loading expert-weights.json...')
    with open('expert-weights.json', 'r') as f:
        weights = json.load(f)

    print(f'✓ Loaded weights ({len(json.dumps(weights)) / 1024:.1f} KB)')
    print(f'  Training: {weights["metadata"]["trainingSamples"]} samples, {weights["metadata"]["epochs"]} epochs')
    print(f'  Timestamp: {weights["metadata"]["timestamp"]}\n')

    # Read the HTML file
    print('Reading vit-planner.html...')
    with open('vit-planner.html', 'r') as f:
        html_content = f.read()

    print('✓ Read HTML file\n')

    # Create the weights loading function
    weights_js = f'''
        // Pre-trained expert model weights
        const EXPERT_WEIGHTS = {json.dumps(weights, separators=(',', ':'))};

        // Load expert model from pre-trained weights
        async function loadExpertModelInstant() {{
            addLog('=== Loading Pre-Trained Expert Model ===', 'info');
            addLog('Loading weights trained offline...', 'info');

            // Create new model
            model = new SimpleViT();

            // Load pre-trained weights
            model.patchEmbedding = EXPERT_WEIGHTS.patchEmbedding;
            model.attention = EXPERT_WEIGHTS.attention;
            model.dense = EXPERT_WEIGHTS.dense;
            model.output = EXPERT_WEIGHTS.output;

            // Update stats
            samplesCount.textContent = `${{EXPERT_WEIGHTS.metadata.trainingSamples}} (expert)`;
            currentEpoch.textContent = EXPERT_WEIGHTS.metadata.epochs;
            trainingLoss.textContent = 'N/A (pre-trained)';
            progressBar.style.width = '100%';
            progressBar.textContent = '100%';

            addLog(`Expert model loaded! Trained on ${{EXPERT_WEIGHTS.metadata.trainingSamples}} samples for ${{EXPERT_WEIGHTS.metadata.epochs}} epochs.`, 'success');
            addLog(`Training date: ${{EXPERT_WEIGHTS.metadata.timestamp}}`, 'info');
            addLog('Try "Visualize Predictions" or "Predict Path" to see expert performance!', 'success');

            // Clear the grid display
            for (let row = 0; row < GRID_SIZE; row++) {{
                for (let col = 0; col < GRID_SIZE; col++) {{
                    grid[row][col].clear();
                }}
            }}
            startCell = null;
            goalCell = null;
        }}
'''

    # Find where to insert the weights (after the loadExpertModel function)
    pattern = r'(// Load expert model - simulates a well-trained model\s+async function loadExpertModel\(\) \{[\s\S]*?\})'

    if not re.search(pattern, html_content):
        print('ERROR: Could not find loadExpertModel function in HTML')
        return

    # Insert the new instant loading function after the existing one
    html_content = re.sub(
        pattern,
        r'\1\n' + weights_js,
        html_content
    )

    # Update the button event listener to use the instant loading version
    # Find the loadExpertBtn event listener
    old_listener = r"document\.getElementById\('loadExpertBtn'\)\.addEventListener\('click', \(\) => \{\s+if \(!isTraining\) \{\s+loadExpertModel\(\);\s+\} else \{\s+addLog\('Cannot load expert model while training!', 'error'\);\s+\}\s+\}\);"

    new_listener = """document.getElementById('loadExpertBtn').addEventListener('click', () => {
            if (!isTraining) {
                loadExpertModelInstant();
            } else {
                addLog('Cannot load expert model while training!', 'error');
            }
        });"""

    html_content = re.sub(old_listener, new_listener, html_content, flags=re.MULTILINE)

    # Write the updated HTML
    print('Writing updated vit-planner.html...')
    with open('vit-planner.html', 'w') as f:
        f.write(html_content)

    # Calculate file size increase
    import os
    html_size = os.path.getsize('vit-planner.html') / 1024

    print(f'✓ Updated HTML file ({html_size:.1f} KB)\n')

    print('=== Embedding Complete ===')
    print('The expert model is now embedded in vit-planner.html')
    print('Users can click "Load Expert Model" for instant access to the trained model')
    print('No training wait time - weights load instantly!')


if __name__ == '__main__':
    main()
