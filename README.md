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

## How to Use

1. **Set Start Point**: Click "Set Start" button, then click on the grid where you want to start
2. **Set Goal Point**: Click "Set Goal" button, then click on the grid where you want to end
3. **Add Walls** (optional): Click "Toggle Walls" and click cells to create obstacles
4. **Run Algorithm**: Click "Run Dijkstra" to watch the algorithm find the shortest path
5. **Clear Path**: Remove the visualization while keeping your setup
6. **Reset Grid**: Start over with a clean grid

## Technologies

- Pure HTML5, CSS3, and JavaScript
- No external dependencies
- Fully client-side execution
- Responsive design

## Algorithm Details

**Dijkstra's Algorithm** is a graph search algorithm that finds the shortest path between nodes in a graph. It works by:
1. Starting at the source node with distance 0
2. Exploring all neighbors and updating their distances
3. Always visiting the unvisited node with the smallest distance
4. Continuing until the goal is reached or all reachable nodes are visited

Time Complexity: O((V + E) log V) where V is vertices and E is edges

## Future Algorithms

Coming soon:
- A* (A-star) Algorithm
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Greedy Best-First Search

## Local Development

Simply open [index.html](index.html) in your browser. No build process or server required!

## License

MIT License - feel free to use and modify as you wish.
