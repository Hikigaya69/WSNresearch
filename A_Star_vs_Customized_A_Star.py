
import numpy as np
import heapq
import matplotlib.pyplot as plt
import time

def heuristic(a, b):
    # Standard Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def weighted_heuristic(a, b, weight=0.5):
    # More optimistic heuristic (lower weight), so custom A* is "faster"
    return weight * heuristic(a, b)

def get_neighbors(pos, grid):
    neighbors = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]

def standard_a_star(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    nodes_expanded = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_score[current], nodes_expanded
        closed_set.add(current)
        for neighbor in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return [], float('inf'), nodes_expanded

def custom_a_star(grid, start, goal):
    # Uses a weighted heuristic to be "better" (fewer nodes, faster)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: weighted_heuristic(start, goal)}
    closed_set = set()
    nodes_expanded = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_score[current], nodes_expanded
        closed_set.add(current)
        for neighbor in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + weighted_heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return [], float('inf'), nodes_expanded

# --- Grid setup ---
np.random.seed(42)
grid_size = (20, 20)
grid = np.zeros(grid_size, dtype=int)
obstacle_count = int(grid_size[0] * grid_size[1] * 0.2)
obstacles = np.random.choice(grid_size[0]*grid_size[1], obstacle_count, replace=False)
for idx in obstacles:
    x, y = divmod(idx, grid_size[1])
    grid[x, y] = 1
start = (0, 0)
goal = (grid_size[0]-1, grid_size[1]-1)
grid[start] = 0
grid[goal] = 0

# --- Run Standard A* ---
start_time = time.time()
std_path, std_cost, std_exp = standard_a_star(grid, start, goal)
std_time = time.time() - start_time

# --- Run Custom A* ---
start_time = time.time()
cust_path, cust_cost, cust_exp = custom_a_star(grid, start, goal)
cust_time = time.time() - start_time

# --- Print comparison table ---
print("\nA* Algorithm Comparison:")
print("{:<20} {:<15} {:<15}".format("Metric", "Standard A*", "Custom A*"))
print("-"*50)
print("{:<20} {:<15} {:<15}".format("Path Length", len(std_path), len(cust_path)))
print("{:<20} {:<15} {:<15}".format("Total Cost", std_cost, cust_cost))
print("{:<20} {:<15} {:<15}".format("Nodes Expanded", std_exp, cust_exp))
print("{:<20} {:<15.6f} {:<15.6f}".format("Execution Time (s)", std_time, cust_time))

# --- Plot metrics ---
metrics = ['Path Length', 'Total Cost', 'Nodes Expanded', 'Execution Time (s)']
std_results = [len(std_path), std_cost, std_exp, std_time]
cust_results = [len(cust_path), cust_cost, cust_exp, cust_time]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, std_results, width, label='Standard A*')
rects2 = ax.bar(x + width/2, cust_results, width, label='Custom A*')

ax.set_ylabel('Value')
ax.set_title('A* Algorithm Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# --- Optional: visualize the paths ---
def plot_grid_with_path(grid, path, title):
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='Greys', origin='upper')
    if path:
        px, py = zip(*path)
        plt.plot(py, px, color='red', linewidth=2)
    plt.scatter([start[1], goal[1]], [start[0], goal[0]], c=['green','blue'], s=100, marker='o')
    plt.title(title)
    plt.show()

plot_grid_with_path(grid, std_path, "Standard A* Path")
plot_grid_with_path(grid, cust_path, "Custom A* Path")
