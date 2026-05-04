
from collections import deque

def bfs(graph, start, goal):
    # Queue for BFS: (current_node, path)
    queue = deque([(start, [start])])
    visited = set()

    print(f"Starting BFS Search from {start} to {goal}...\n")

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            print(f"Visited Node: {current}")

            # Add neighbors to the queue
            for neighbor in graph.get(current, {}):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None

# Graph definition from image
graph = {
    'a': {'b': 4, 'c': 3},
    'b': {'f': 5, 'e': 12},
    'c': {'e': 10, 'd': 7},
    'd': {'e': 2},
    'e': {'z': 5},
    'f': {'z': 16},
    'z': {}
}

# Running BFS
path = bfs(graph, 'a', 'z')

if path:
    print(f"\nBFS Path found: {' -> '.join(path)}")
else:
    print("\nNo path found using BFS.")
