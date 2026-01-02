# Path Planning Algorithm Visualizations

Interactive visualizations of pathfinding algorithms, built entirely client-side for GitHub Pages.

## Demo

Visit the live demo: [https://YOUR-USERNAME.github.io/deep-planning/](https://YOUR-USERNAME.github.io/deep-planning/)

## Features

### Dijkstra's Algorithm
- Interactive grid-based visualization
- Set start and goal points by clicking
- Add walls to create obstacles
- Watch the algorithm explore the grid in real-time
- See the shortest path highlighted
- Smooth animations for visited nodes and final path

### A* (A-Star) Algorithm
- All Dijkstra features plus heuristic-guided search
- Multiple heuristic functions: Manhattan, Euclidean, Chebyshev
- Real-time statistics showing nodes explored, path length, and time taken
- Compare efficiency with Dijkstra by selecting "None" heuristic
- Faster pathfinding with significantly fewer explored nodes

### Vision Transformers & Transformer Architecture
- Comprehensive educational guide to transformers and ViT
- Interactive attention mechanism visualization
- Visual patch tokenization demonstration
- Multi-head attention explained with interactive components
- Detailed mathematical formulations and explanations
- Comparison of CNNs vs Vision Transformers
- Architecture diagrams and practical implementation tips
- Coverage of recent developments (DeiT, BEiT, MAE, Swin Transformer)

## How to Use

1. **Set Start Point**: Click "Set Start" button, then click on the grid where you want to start
2. **Set Goal Point**: Click "Set Goal" button, then click on the grid where you want to end
3. **Add Walls** (optional): Click "Toggle Walls" and click cells to create obstacles
4. **Run Algorithm**: Click "Run Dijkstra" to watch the algorithm find the shortest path
5. **Clear Path**: Remove the visualization while keeping your setup
6. **Reset Grid**: Start over with a clean grid

## Pages

1. **[Dijkstra's Algorithm](index.html)** - Interactive pathfinding visualization
2. **[A* Algorithm](astar.html)** - Heuristic-based pathfinding with multiple heuristic options
3. **[Transformers & ViT](transformers.html)** - Educational guide to transformer architecture and Vision Transformers

## Technologies

- Pure HTML5, CSS3, and JavaScript
- No external dependencies
- Fully client-side execution
- Responsive design
- Interactive visualizations and demos

## Algorithm Details

### Dijkstra's Algorithm
A graph search algorithm that finds the shortest path between nodes. It works by:
1. Starting at the source node with distance 0
2. Exploring all neighbors and updating their distances
3. Always visiting the unvisited node with the smallest distance
4. Continuing until the goal is reached or all reachable nodes are visited

**Time Complexity:** O((V + E) log V) where V is vertices and E is edges

**Pros:** Guarantees shortest path, works on any weighted graph
**Cons:** Explores many unnecessary nodes, no sense of direction

### A* (A-Star) Algorithm
An informed search algorithm that uses heuristics to guide exploration. It works by:
1. Maintaining a cost function: `f(n) = g(n) + h(n)`
   - `g(n)` = actual distance from start to node n
   - `h(n)` = estimated distance from node n to goal (heuristic)
2. Always expanding the node with the lowest f-score
3. Continuing until the goal is reached

**Time Complexity:** O((V + E) log V) in worst case, much faster in practice

**Pros:** Much faster than Dijkstra, explores fewer nodes, still guarantees shortest path (with admissible heuristic)
**Cons:** Requires a good heuristic function

**Heuristics Available:**
- **Manhattan Distance:** Best for grid-based movement (4 directions). Calculates `|x1 - x2| + |y1 - y2|`
- **Euclidean Distance:** Straight-line distance. Calculates `√((x1-x2)² + (y1-y2)²)`
- **Chebyshev Distance:** Best for 8-directional movement. Calculates `max(|x1-x2|, |y1-y2|)`
- **None:** Behaves exactly like Dijkstra's algorithm

### Transformers & Vision Transformers

**Transformer Architecture** introduced in "Attention Is All You Need" (2017), revolutionized deep learning by replacing recurrent/convolutional layers with self-attention mechanisms.

**Key Components:**
1. **Self-Attention Mechanism:** Allows each element to attend to all others
   - Formula: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
2. **Multi-Head Attention:** Multiple attention computations in parallel
3. **Positional Encoding:** Injects sequence position information
4. **Feed-Forward Networks:** Position-wise transformations

**Vision Transformers (ViT)** adapt transformers for images by:
1. Splitting images into patches (e.g., 16×16 pixels)
2. Linearly embedding flattened patches
3. Adding positional embeddings
4. Processing through transformer encoder
5. Using [CLS] token for classification

**Advantages:**
- Global receptive field from first layer
- Excellent long-range dependency modeling
- Highly scalable and parallelizable
- State-of-the-art results when pre-trained on large datasets

**Trade-offs:**
- Requires large datasets or pre-training
- Less inductive bias than CNNs
- Higher computational cost than CNNs

## Future Content

Coming soon:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Greedy Best-First Search
- Bidirectional Search

## Local Development

Simply open [index.html](index.html) in your browser. No build process or server required!

## License

MIT License - feel free to use and modify as you wish.
