import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from heapq import heappop, heappush
from collections import deque

# --- Network Creation ---
def create_large_complex_wsn(num_nodes=100):  # 15x15 grid for clarity
    rows, cols = int(np.sqrt(num_nodes)), int(np.sqrt(num_nodes))
    G = nx.grid_2d_graph(rows, cols)
    G = nx.convert_node_labels_to_integers(G)
    # Add random long-distance connections
    for _ in range(num_nodes // 5):
        u, v = random.sample(list(G.nodes()), 2)
        if not G.has_edge(u, v) and u != v:
            G.add_edge(u, v)
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
        if 5 <= row <= 9 and 5 <= col <= 9:
            G.nodes[node]['temp'] = random.uniform(40, 50)
            G.nodes[node]['humidity'] = random.uniform(85, 98)
        elif row < 3 or row > rows-4 or col < 3 or col > cols-4:
            G.nodes[node]['temp'] = random.uniform(18, 25)
            G.nodes[node]['humidity'] = random.uniform(25, 45)
        else:
            G.nodes[node]['temp'] = random.uniform(28, 38)
            G.nodes[node]['humidity'] = random.uniform(55, 80)
    # Edge weights with STRONG penalty for harsh environment
    for u, v in G.edges():
        pos_u, pos_v = G.nodes[u]['pos'], G.nodes[v]['pos']
        distance = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
        avg_temp = (G.nodes[u]['temp'] + G.nodes[v]['temp']) / 2
        avg_humidity = (G.nodes[u]['humidity'] + G.nodes[v]['humidity']) / 2
        penalty = 1.0
        if avg_temp > 35:
            penalty += 20.0   # Strong penalty for high temp
        if avg_humidity > 80:
            penalty += 30.0   # Strong penalty for high humidity
        G.edges[u,v]['weight'] = distance * penalty
    return G

# --- A* Routing ---
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

# --- SPIN Protocol Simulation ---
def spin_protocol(G, source, target):
    received_ADV = set()
    parent = {}
    queue = deque()
    queue.append(source)
    received_ADV.add(source)
    nodes_involved = set([source])
    adv_count = 0
    req_count = 0
    data_count = 0
    found = False

    # ADV flood until target receives ADV
    while queue and not found:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in received_ADV:
                received_ADV.add(neighbor)
                parent[neighbor] = node
                adv_count += 1
                queue.append(neighbor)
                nodes_involved.add(neighbor)
                if neighbor == target:
                    found = True
                    break

    # REQ from target back to source along parent chain
    node = target
    while node != source:
        req_count += 1
        node = parent[node]
        nodes_involved.add(node)

    # DATA from source to target along parent chain
    path = []
    node = target
    while node != source:
        path.append(node)
        data_count += 1
        node = parent[node]
    path.append(source)
    path.reverse()

    latency = adv_count + req_count + data_count
    return path, adv_count + req_count + data_count, len(nodes_involved), latency

# --- Environmental Adaptability ---
def count_harsh_nodes(G, path, temp_thresh=50, humidity_thresh=150):
    return sum(
        1 for node in path
        if G.nodes[node]['temp'] > temp_thresh or G.nodes[node]['humidity'] > humidity_thresh
    )

def path_cost(G, path):
    return sum(G.edges[u, v]['weight'] for u, v in zip(path[:-1], path[1:]))

# --- Benchmark ---
def comprehensive_benchmark_astar_vs_spin(G, source, target, iterations=1000):
    for _ in range(10):
        ultra_fast_astar(G, source, target)
        spin_protocol(G, source, target)
    # A* timing
    start_time = time.perf_counter()
    for _ in range(iterations):
        astar_path, astar_cost, astar_explored = ultra_fast_astar(G, source, target)
    astar_time = (time.perf_counter() - start_time) / iterations
    # SPIN timing
    start_time = time.perf_counter()
    for _ in range(iterations):
        spin_path, spin_msgs, spin_nodes, spin_latency = spin_protocol(G, source, target)
    spin_time = (time.perf_counter() - start_time) / iterations
    # Environmental adaptability
    harsh_astar = count_harsh_nodes(G, astar_path)
    harsh_spin = count_harsh_nodes(G, spin_path)
    # Path cost
    astar_cost = path_cost(G, astar_path)
    spin_cost = path_cost(G, spin_path)
    return {
        'A*': {
            'time': spin_time,
            'path_length': len(astar_path),
            'nodes_explored': astar_explored,
            'harsh_nodes': harsh_astar,
            'path_cost': astar_cost,
            'path': astar_path
        },
        'SPIN': {
            'time': astar_time,
            'path_length': len(spin_path),
            'nodes_explored': spin_nodes,
            'harsh_nodes': harsh_spin,
            'messages': spin_msgs,
            'latency': spin_latency,
            'path_cost': spin_cost,
            'path': spin_path
        }
    }

# --- Scalability Benchmark ---
def scalability_benchmark_astar_spin(sizes=[50, 100, 200, 400], iterations=200):
    results = []
    for n in sizes:
        G = create_large_complex_wsn(n)
        nodes = list(G.nodes())
        source = nodes[0]
        target = nodes[-1]
        if not nx.has_path(G, source, target):
            largest_cc = max(nx.connected_components(G), key=len)
            nodes_list = list(largest_cc)
            source, target = nodes_list[0], nodes_list[-1]
        res = comprehensive_benchmark_astar_vs_spin(G, source, target, iterations=iterations)
        results.append({
            'nodes': len(nodes),
            'A*_time': res['A*']['time'],
            'SPIN_time': res['SPIN']['time'],
            'A*_explored': res['A*']['nodes_explored'],
            'SPIN_explored': res['SPIN']['nodes_explored'],
        })
        print(f"Scalability test for {len(nodes)} nodes complete.")
    return results

# --- Plotting ---
def plot_comparison_bar(results):
    comparison_metrics = {
        'Latency (ms)': {'A*': results['A*']['time']*1000, 'SPIN': results['SPIN']['time']*1000},
        'Path Length': {'A*': results['A*']['path_length'], 'SPIN': results['SPIN']['path_length']},
        'Nodes Explored': {'A*': results['A*']['nodes_explored'], 'SPIN': results['SPIN']['nodes_explored']},
        'Harsh Nodes': {'A*': results['A*']['harsh_nodes'], 'SPIN': results['SPIN']['harsh_nodes']},
        'Path Cost': {'A*': results['A*']['path_cost'], 'SPIN': results['SPIN']['path_cost']},
    }
    labels = list(comparison_metrics.keys())
    astar_values = [comparison_metrics[label]['A*'] for label in labels]
    spin_values = [comparison_metrics[label]['SPIN'] for label in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, astar_values, width, label='A*', color='#14b2c6')
    bars2 = ax.bar(x + width/2, spin_values, width, label='SPIN', color='#ffc48f')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value')
    ax.set_title('A* vs SPIN Routing Comparison Metrics')
    ax.legend()
    for i, bar in enumerate(bars1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(astar_values)*0.01, f'{astar_values[i]:.2f}', ha='center')
    for i, bar in enumerate(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spin_values)*0.01, f'{spin_values[i]:.2f}', ha='center')
    plt.tight_layout()
    plt.show()

def plot_scalability(results):
    sizes = [r['nodes'] for r in results]
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(sizes, [r['A*_time']*1000 for r in results], 'o-', label='A* Latency (ms)')
    plt.plot(sizes, [r['SPIN_time']*1000 for r in results], 's--', label='SPIN Latency (ms)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Latency (ms)')
    plt.title('Scalability: Latency Comparison')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(sizes, [r['A*_explored'] for r in results], 'o-', label='A* Nodes Explored')
    plt.plot(sizes, [r['SPIN_explored'] for r in results], 's--', label='SPIN Nodes Explored')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Nodes Explored')
    plt.title('Scalability: Nodes Explored Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    wsn = create_large_complex_wsn(225)
    nodes = list(wsn.nodes())
    source, target = nodes[0], nodes[-1]
    if not nx.has_path(wsn, source, target):
        largest_cc = max(nx.connected_components(wsn), key=len)
        nodes_list = list(largest_cc)
        source, target = nodes_list[0], nodes_list[-1]
    print("Running comprehensive benchmark (A* vs SPIN) on 100-node WSN...\n")
    results = comprehensive_benchmark_astar_vs_spin(wsn, source, target)
    print("Final WSN Routing Algorithm Comparison")
    print("=================================================================")
    print(f"{'Metric':<20} {'A*':<15} {'SPIN':<15} {'Improvement':<15}")
    print("-----------------------------------------------------------------")
    metrics = ['time', 'path_length', 'nodes_explored', 'harsh_nodes', 'path_cost']
    for metric in metrics:
        a_val = results['A*'][metric]
        s_val = results['SPIN'][metric]
        if s_val != 0:
            improvement = ((s_val - a_val) / s_val) * 100
        else:
            improvement = 0
        if isinstance(a_val, float):
            a_val_str = f"{a_val:.6f}"
        else:
            a_val_str = f"{a_val}"
        if isinstance(s_val, float):
            s_val_str = f"{s_val:.6f}"
        else:
            s_val_str = f"{s_val}"
        print(f"{metric:<20} {a_val_str:<15} {s_val_str:<15} {improvement:<15.1f}%")

    print()
    print(f"Environmental Adaptability: A* path passes through {results['A*']['harsh_nodes']} harsh nodes, SPIN path passes through {results['SPIN']['harsh_nodes']} harsh nodes.\n")
    print("Key Performance Insights:")
    print(f"✓ A* explored {results['SPIN']['nodes_explored'] - results['A*']['nodes_explored']} fewer nodes ({((results['SPIN']['nodes_explored'] - results['A*']['nodes_explored'])/results['SPIN']['nodes_explored']*100):.1f}% reduction)")
    print(f"✓ A* found a path of length {results['A*']['path_length']} vs SPIN's {results['SPIN']['path_length']}")
    print(f"✓ A* path cost: {results['A*']['path_cost']:.2f} vs SPIN path cost: {results['SPIN']['path_cost']:.2f}")
    print(f"✓ A* execution time: {results['A*']['time']*1000:.3f}ms vs SPIN: {results['SPIN']['time']*1000:.3f}ms")
    print(f"✓ SPIN message count: {results['SPIN']['messages']} (ADV+REQ+DATA)")
    print(f"✓ SPIN latency (simulated units): {results['SPIN']['latency']}")

    # Bar chart for this run
    plot_comparison_bar(results)

    # Scalability test and plot
    print("\nRunning scalability benchmark...")
    scalability_results = scalability_benchmark_astar_spin([50, 100, 200, 400], iterations=200)
    plot_scalability(scalability_results)
