
import heapq

def a_star(graph, heuristics, start, goal):
    # Priority queue stores (f_score, current_node, path, g_score)
    open_list = [(heuristics[start], start, [start], 0)]
    visited = {}

    print(f"Starting A* Search from {start} to {goal}...\n")

    while open_list:
        f, current, path, g = heapq.heappop(open_list)

        if current == goal:
            return path, g

        if current in visited and visited[current] <= g:
            continue
        
        visited[current] = g
        print(f"Expanding Node {current}: g={g}, h={heuristics[current]}, f={f}")

        for neighbor, weight in graph.get(current, {}).items():
            new_g = g + weight
            new_f = new_g + heuristics[neighbor]
            heapq.heappush(open_list, (new_f, neighbor, path + [neighbor], new_g))

    return None, float('inf')

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

# Heuristic values (orange numbers in image)
heuristics = {
    'a': 14,
    'b': 12,
    'c': 11,
    'd': 6,
    'e': 4,
    'f': 11,
    'z': 0
}

path, cost = a_star(graph, heuristics, 'a', 'z')

if path:
    print(f"\nOptimal Path: {' -> '.join(path)}")
    print(f"Total Cost: {cost}")
else:
    print("No path found.")
