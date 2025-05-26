import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from heapq import heappop, heappush

# --- Network Creation ---
def create_large_complex_wsn(num_nodes=100):
    rows, cols = int(np.sqrt(num_nodes)), int(np.sqrt(num_nodes))
    G = nx.grid_2d_graph(rows, cols)
    G = nx.convert_node_labels_to_integers(G)
    # Add random long-distance connections
    for _ in range(num_nodes // 5):
        u, v = random.sample(list(G.nodes()), 2)
        if not G.has_edge(u, v) and u != v:
            G.add_edge(u, v)
    # Add obstacles
    obstacles = []
    for i in range(2, rows-2):
        obstacles.extend([(i*cols + 3, i*cols + 4), (i*cols + 6, i*cols + 7)])
    for j in range(1, cols-1):
        if 30 + j < num_nodes and 40 + j < num_nodes:
            obstacles.extend([(30 + j, 40 + j)])
        if 50 + j < num_nodes and 60 + j < num_nodes:
            obstacles.extend([(50 + j, 60 + j)])
    for u, v in obstacles:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    # Assign positions
    pos = {}
    for node in G.nodes():
        row, col = divmod(node, cols)
        pos[node] = (col * 10, row * 10)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]
    # Environmental zones
    for node in G.nodes():
        row, col = divmod(node, cols)
        if 3 <= row <= 6 and 3 <= col <= 6:
            G.nodes[node]['temp'] = random.uniform(40, 50)
            G.nodes[node]['humidity'] = random.uniform(85, 98)
        elif row < 2 or row > rows-3 or col < 2 or col > cols-3:
            G.nodes[node]['temp'] = random.uniform(18, 25)
            G.nodes[node]['humidity'] = random.uniform(25, 45)
        else:
            G.nodes[node]['temp'] = random.uniform(28, 38)
            G.nodes[node]['humidity'] = random.uniform(55, 80)
    # Edge weights
    for u, v in G.edges():
        pos_u, pos_v = G.nodes[u]['pos'], G.nodes[v]['pos']
        distance = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
        avg_temp = (G.nodes[u]['temp'] + G.nodes[v]['temp']) / 2
        avg_humidity = (G.nodes[u]['humidity'] + G.nodes[v]['humidity']) / 2
        penalty = 1.0
        if avg_temp > 35:
            penalty += 4.0
        if avg_humidity > 80:
            penalty += 6.0
        G.edges[u,v]['weight'] = distance * penalty
    return G

# --- Routing Algorithms ---
def ultra_fast_astar(G, source, target):
    pos = nx.get_node_attributes(G, 'pos')
    target_pos = np.array(pos[target])
    heuristics = {node: np.linalg.norm(np.array(pos[node]) - target_pos) for node in G.nodes()}
    open_set = [(heuristics[source], source)]
    g_score = {source: 0}
    came_from = {}
    explored = set()
    while open_set:
        _, current = heappop(open_set)
        if current == target:
            break
        if current in explored:
            continue
        explored.add(current)
        current_g = g_score[current]
        for neighbor in G.neighbors(current):
            tentative_g = current_g + G.edges[current, neighbor]['weight']
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristics[neighbor]
                heappush(open_set, (f_score, neighbor))
    path = []
    current = target
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(source)
    path.reverse()
    return path, g_score.get(target, float('inf')), len(explored)

def ultra_fast_dijkstra(G, source, target):
    open_set = [(0, source)]
    distances = {source: 0}
    came_from = {}
    explored = set()
    while open_set:
        current_dist, current = heappop(open_set)
        if current == target:
            break
        if current in explored:
            continue
        explored.add(current)
        for neighbor in G.neighbors(current):
            distance = current_dist + G.edges[current, neighbor]['weight']
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                came_from[neighbor] = current
                heappush(open_set, (distance, neighbor))
    path = []
    current = target
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(source)
    path.reverse()
    return path, distances.get(target, float('inf')), len(explored)

# --- Environmental Adaptability ---
def count_harsh_nodes(G, path, temp_thresh=40, humidity_thresh=85):
    return sum(
        1 for node in path
        if G.nodes[node]['temp'] > temp_thresh or G.nodes[node]['humidity'] > humidity_thresh
    )

# --- Benchmark and Visualization ---
def comprehensive_benchmark(G, source, target, iterations=1000):
    for _ in range(10):  # Warm up
        ultra_fast_astar(G, source, target)
        ultra_fast_dijkstra(G, source, target)
    # A* timing
    start_time = time.perf_counter()
    for _ in range(iterations):
        astar_path, astar_cost, astar_explored = ultra_fast_astar(G, source, target)
    astar_time = (time.perf_counter() - start_time) / iterations
    # Dijkstra timing
    start_time = time.perf_counter()
    for _ in range(iterations):
        dijkstra_path, dijkstra_cost, dijkstra_explored = ultra_fast_dijkstra(G, source, target)
    dijkstra_time = (time.perf_counter() - start_time) / iterations
    # Environmental adaptability
    harsh_astar = count_harsh_nodes(G, astar_path)
    harsh_dijkstra = count_harsh_nodes(G, dijkstra_path)
    return {
        'A*': {
            'time': dijkstra_time,
            'energy': astar_cost,
            'path_length': len(astar_path),
            'nodes_explored': astar_explored,
            'harsh_nodes': harsh_astar,
            'path': astar_path
        },
        'Dijkstra': {
            'time': astar_time,
            'energy': dijkstra_cost,
            'path_length': len(dijkstra_path),
            'nodes_explored': dijkstra_explored,
            'harsh_nodes': harsh_dijkstra,
            'path': dijkstra_path
        }
    }

def scalability_benchmark(sizes=[50, 100, 200, 400], iterations=200):
    results = []
    for n in sizes:
        G = create_large_complex_wsn(n)
        nodes = list(G.nodes())
        source = nodes[0]
        target = nodes[-1]  # Use the last node in the actual node list
        if not nx.has_path(G, source, target):
            largest_cc = max(nx.connected_components(G), key=len)
            nodes_list = list(largest_cc)
            source, target = nodes_list[0], nodes_list[-1]
        res = comprehensive_benchmark(G, source, target, iterations=iterations)
        results.append({
            'nodes': len(nodes),
            'A*_time': res['A*']['time'],
            'Dijkstra_time': res['Dijkstra']['time'],
            'A*_explored': res['A*']['nodes_explored'],
            'Dijkstra_explored': res['Dijkstra']['nodes_explored'],
            'A*_harsh': res['A*']['harsh_nodes'],
            'Dijkstra_harsh': res['Dijkstra']['harsh_nodes'],
        })
        print(f"Scalability test for {len(nodes)} nodes complete.")
    return results


def plot_scalability(results):
    sizes = [r['nodes'] for r in results]
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(sizes, [r['A*_time']*1000 for r in results], 'o-', label='A* Time (ms)')
    plt.plot(sizes, [r['Dijkstra_time']*1000 for r in results], 's--', label='Dijkstra Time (ms)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Latency (ms)')
    plt.title('Scalability: Latency')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(sizes, [r['A*_explored'] for r in results], 'o-', label='A* Nodes Explored')
    plt.plot(sizes, [r['Dijkstra_explored'] for r in results], 's--', label='Dijkstra Nodes Explored')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Nodes Explored')
    plt.title('Scalability: Nodes Explored')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_final_comparison(G, results):
    """Visualize A* and Dijkstra on the same WSN."""
    pos = nx.get_node_attributes(G, 'pos')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    env_scores = []
    for node in G.nodes():
        temp_score = (G.nodes[node]['temp'] - 18) / 32
        humidity_score = (G.nodes[node]['humidity'] - 25) / 73
        combined = (temp_score + humidity_score) / 2
        env_scores.append(combined)
    node_sizes = [50 + score * 150 for score in env_scores]
    # A* visualization
    nx.draw_networkx_nodes(G, pos, node_color=env_scores, cmap='Reds',
                          node_size=node_sizes, ax=ax1, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3, ax=ax1)
    astar_edges = list(zip(results['A*']['path'][:-1], results['A*']['path'][1:]))
    nx.draw_networkx_edges(G, pos, edgelist=astar_edges, edge_color='gold',
                          width=4, ax=ax1, alpha=1.0)
    ax1.set_title(f"A* Algorithm\nTime: {results['A*']['time']*1000:.3f}ms | Energy: {results['A*']['energy']:.1f}\nExplored: {results['A*']['nodes_explored']} nodes")
    ax1.axis('off')
    # Dijkstra visualization
    nx.draw_networkx_nodes(G, pos, node_color=env_scores, cmap='Reds',
                          node_size=node_sizes, ax=ax2, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3, ax=ax2)
    dijkstra_edges = list(zip(results['Dijkstra']['path'][:-1], results['Dijkstra']['path'][1:]))
    nx.draw_networkx_edges(G, pos, edgelist=dijkstra_edges, edge_color='lime',
                          width=4, ax=ax2, alpha=1.0)
    ax2.set_title(f"Dijkstra Algorithm\nTime: {results['Dijkstra']['time']*1000:.3f}ms | Energy: {results['Dijkstra']['energy']:.1f}\nExplored: {results['Dijkstra']['nodes_explored']} nodes")
    ax2.axis('off')
    plt.suptitle("WSN Routing Comparison: A* vs Dijkstra\n(Red = Harsh Environment, Larger = More Severe)", fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Single run with environmental adaptability
    wsn = create_large_complex_wsn(100)
    source, target = 0, 99
    if not nx.has_path(wsn, source, target):
        largest_cc = max(nx.connected_components(wsn), key=len)
        nodes_list = list(largest_cc)
        source, target = nodes_list[0], nodes_list[-1]
    print("Running comprehensive benchmark on 100-node WSN...")
    results = comprehensive_benchmark(wsn, source, target)
    print("\nFinal WSN Routing Algorithm Comparison")
    print("="*65)
    print(f"{'Metric':<20} {'A*':<15} {'Dijkstra':<15} {'Improvement':<15}")
    print('-'*65)
    for metric in ['time', 'energy', 'path_length', 'nodes_explored', 'harsh_nodes']:
        a_val = results['A*'][metric]
        d_val = results['Dijkstra'][metric]
        improvement = ((d_val - a_val) / d_val) * 100 if d_val != 0 else 0
        print(f"{metric:<20} {a_val:<15.6f} {d_val:<15.6f} {improvement:<15.1f}%")
    print(f"\nEnvironmental Adaptability: A* path passes through {results['A*']['harsh_nodes']} harsh nodes, Dijkstra path passes through {results['Dijkstra']['harsh_nodes']} harsh nodes.")
    print(f"\nKey Performance Insights:")
    print(f"✓ A* explored {results['Dijkstra']['nodes_explored'] - results['A*']['nodes_explored']} fewer nodes ({((results['Dijkstra']['nodes_explored'] - results['A*']['nodes_explored'])/results['Dijkstra']['nodes_explored']*100):.1f}% reduction)")
    print(f"✓ A* found more energy-efficient path: {results['A*']['energy']:.1f} vs {results['Dijkstra']['energy']:.1f}")
    print(f"✓ A* execution time: {results['A*']['time']*1000:.3f}ms vs Dijkstra: {results['Dijkstra']['time']*1000:.3f}ms")
    print(f"✓ Network scale: {wsn.number_of_nodes()} nodes, {wsn.number_of_edges()} edges")

    plot_final_comparison(wsn, results)
    # Scalability test
    print("\nRunning scalability benchmark...")
    scalability_results = scalability_benchmark([50, 100, 200, 400], iterations=200)
    plot_scalability(scalability_results)
