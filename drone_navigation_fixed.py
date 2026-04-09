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

def dfs(graph, start, goal):
    visited = set()
    nodes_expanded = [0]
    path = []
    found = [False]

    def dfs_helper(node, current_path):
        if found[0]:
            return

        visited.add(node)
        nodes_expanded[0] += 1
        current_path.append(node)

        if node == goal:
            found[0] = True
            path.extend(current_path)
            return

        for neighbour in sorted(graph[node].keys()):
            if neighbour not in visited:
                dfs_helper(neighbour, current_path)
                if found[0]:
                    return

        current_path.pop()

    dfs_helper(start, [])
    return path, nodes_expanded[0]


# ===================================
# Breadth-First Search (BFS)

def bfs(graph, start, goal):
    visited = set([start])
    queue = deque([(start, [start])])
    nodes_expanded = 0

    while queue:
        node, path = queue.popleft()
        nodes_expanded += 1

        if node == goal:
            return path, nodes_expanded

        for neighbour in sorted(graph[node].keys()):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))

    return None, nodes_expanded


# ===================================
# Uniform Cost Search (UCS)

def ucs(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    explored = set()
    nodes_expanded = 0

    while frontier:
        cost, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded

        for neighbour, edge_cost in graph[node].items():
            if neighbour not in explored:
                new_cost = cost + edge_cost
                heapq.heappush(frontier, (new_cost, neighbour, path + [neighbour]))

    return None, float('inf'), nodes_expanded


# ===================================
# A* Search

def a_star(graph, start, goal, heuristic):
    frontier = []
    heapq.heappush(frontier, (heuristic[start], 0, start, [start]))
    explored = set()
    nodes_expanded = 0

    while frontier:
        f, g, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            return path, g, nodes_expanded

        for neighbour, edge_cost in graph[node].items():
            if neighbour not in explored:
                new_g = g + edge_cost
                new_f = new_g + heuristic[neighbour]
                heapq.heappush(frontier, (new_f, new_g, neighbour, path + [neighbour]))

    return None, float('inf'), nodes_expanded


# ===================================
# UCS with Battery Constraint

def ucs_with_battery(graph, start, goal, max_cost):
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    explored = set()
    nodes_expanded = 0

    while frontier:
        cost, node, path = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded

        for neighbour, edge_cost in graph[node].items():
            if neighbour not in explored:
                new_cost = cost + edge_cost
                if new_cost <= max_cost:
                    heapq.heappush(frontier, (new_cost, neighbour, path + [neighbour]))

    return None, float('inf'), nodes_expanded


if __name__ == "__main__":

    print("=" * 60)
    print("       DRONE DELIVERY NAVIGATION - RESULTS")
    print("=" * 60)

    # --- DFS ---
    dfs_path, dfs_expanded = dfs(graph, start, goal)
    dfs_steps = len(dfs_path) - 1 if dfs_path else 0
    dfs_cost = sum(graph[dfs_path[i]][dfs_path[i+1]] for i in range(len(dfs_path)-1)) if dfs_path else 0
    print(f"\n DFS")
    print(f"   Path          : {' -> '.join(dfs_path)}")
    print(f"   Steps (hops)  : {dfs_steps}")
    print(f"   Total cost    : {dfs_cost}")
    print(f"   Nodes expanded: {dfs_expanded}")

    # --- BFS ---
    bfs_path, bfs_expanded = bfs(graph, start, goal)
    bfs_steps = len(bfs_path) - 1 if bfs_path else 0
    bfs_cost = sum(graph[bfs_path[i]][bfs_path[i+1]] for i in range(len(bfs_path)-1)) if bfs_path else 0
    print(f"\n BFS")
    print(f"   Path          : {' -> '.join(bfs_path)}")
    print(f"   Steps (hops)  : {bfs_steps}")
    print(f"   Total cost    : {bfs_cost}")
    print(f"   Nodes expanded: {bfs_expanded}")

    # --- UCS ---
    ucs_path, ucs_cost, ucs_expanded = ucs(graph, start, goal)
    ucs_steps = len(ucs_path) - 1 if ucs_path else 0
    print(f"\n UCS")
    print(f"   Path          : {' -> '.join(ucs_path)}")
    print(f"   Steps (hops)  : {ucs_steps}")
    print(f"   Total cost    : {ucs_cost}")
    print(f"   Nodes expanded: {ucs_expanded}")

    # --- A* ---
    astar_path, astar_cost, astar_expanded = a_star(graph, start, goal, heuristic)
    astar_steps = len(astar_path) - 1 if astar_path else 0
    print(f"\n A*")
    print(f"   Path          : {' -> '.join(astar_path)}")
    print(f"   Steps (hops)  : {astar_steps}")
    print(f"   Total cost    : {astar_cost}")
    print(f"   Nodes expanded: {astar_expanded}")

    # --- Comparison Table ---
    print(f"\n{'=' * 60}")
    print(f"{'COMPARISON TABLE':^60}")
    print(f"{'=' * 60}")
    print(f"{'Algorithm':<12} {'Path':<25} {'Hops':<6} {'Cost':<6} {'Expanded'}")
    print(f"{'-' * 60}")
    print(f"{'DFS':<12} {'->'.join(dfs_path):<25} {dfs_steps:<6} {dfs_cost:<6} {dfs_expanded}")
    print(f"{'BFS':<12} {'->'.join(bfs_path):<25} {bfs_steps:<6} {bfs_cost:<6} {bfs_expanded}")
    print(f"{'UCS':<12} {'->'.join(ucs_path):<25} {ucs_steps:<6} {ucs_cost:<6} {ucs_expanded}")
    print(f"{'A*':<12} {'->'.join(astar_path):<25} {astar_steps:<6} {astar_cost:<6} {astar_expanded}")

    print(f"\n{'=' * 60}")
    print(f"OPTIONAL: Battery-Constrained Search (max cost = 7)")
    print(f"{'=' * 60}")
    battery_path, battery_cost, battery_expanded = ucs_with_battery(graph, start, goal, max_cost=7)
    if battery_path:
        print(f"   Path found    : {' -> '.join(battery_path)}")
        print(f"   Total cost    : {battery_cost}")
        print(f"   Nodes expanded: {battery_expanded}")
    else:
        print("   No path found within battery limit of 7.")

    battery_path2, battery_cost2, battery_expanded2 = ucs_with_battery(graph, start, goal, max_cost=10)
    print(f"\n   Battery limit = 10:")
    if battery_path2:
        print(f"   Path found    : {' -> '.join(battery_path2)}")
        print(f"   Total cost    : {battery_cost2}")
        print(f"   Nodes expanded: {battery_expanded2}")
    else:
        print("   No path found within battery limit of 10.")
