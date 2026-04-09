# ==========================
# Drone Delivery Navigation
# ==========================

import heapq
from collections import deque

# --------- Graph Definition ---------
graph = {
    'A': {'B': 2, 'C': 5, 'D': 1},
    'B': {'A': 2, 'D': 2, 'E': 3},
    'C': {'A': 5, 'D': 2, 'F': 3},
    'D': {'A': 1, 'B': 2, 'C': 2, 'E': 1, 'F': 4},
    'E': {'B': 3, 'D': 1, 'G': 2},
    'F': {'C': 3, 'D': 4, 'G': 1},
    'G': {'E': 2, 'F': 1, 'H': 3},
    'H': {'G': 3}
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 6,
    'D': 4,
    'E': 2,
    'F': 2,
    'G': 1,
    'H': 0
}

start = 'A'
goal = 'H'

# ===================================
# Depth-First Search (DFS)
# ===================================
def dfs(graph, start, goal):
    visited = set()
    path = []
    found = [False]

    def dfs_helper(node, current_path):
        if found[0]:
            return

        visited.add(node)
        current_path.append(node)

        if node == goal:
            found[0] = True
            path.extend(current_path)
            return

        # Explore neighbours (sorted for determinism)
        for neighbour in sorted(graph[node].keys()):
            if neighbour not in visited:
                dfs_helper(neighbour, current_path)
                if found[0]:
                    return

        # Backtrack
        current_path.pop()

    dfs_helper(start, [])
    return path

# ===================================
# Breadth-First Search (BFS)
# ===================================
def bfs(graph, start, goal):
    visited = set([start])
    queue = deque()
    queue.append((start, [start]))

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path

        for neighbour in sorted(graph[node].keys()):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))

    return None  # No path found

# ===================================
# Uniform Cost Search (UCS)
# ===================================
def ucs(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    explored = set()

    while frontier:
        cost, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)

        if node == goal:
            return path, cost

        for neighbour, edge_cost in graph[node].items():
            if neighbour not in explored:
                new_cost = cost + edge_cost
                heapq.heappush(frontier, (new_cost, neighbour, path + [neighbour]))

    return None, float('inf')  # No path found

# ===================================
# A* Search
# ===================================
def a_star(graph, start, goal, heuristic):
    frontier = []
    heapq.heappush(frontier, (heuristic[start], 0, start, [start]))
    explored = set()

    while frontier:
        f, g, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)

        if node == goal:
            return path, g

        for neighbour, edge_cost in graph[node].items():
            if neighbour not in explored:
                new_g = g + edge_cost
                new_f = new_g + heuristic[neighbour]
                heapq.heappush(frontier, (new_f, new_g, neighbour, path + [neighbour]))

    return None, float('inf')  # No path found

# ===================================
# Run and Compare
# ===================================
if __name__ == "__main__":
    print("DFS Path:", dfs(graph, start, goal))

    print("BFS Path:", bfs(graph, start, goal))

    ucs_path, ucs_cost = ucs(graph, start, goal)
    print("UCS Path:", ucs_path, "| Cost:", ucs_cost)

    astar_path, astar_cost = a_star(graph, start, goal, heuristic)
    print("A* Path:", astar_path, "| Cost:", astar_cost)
