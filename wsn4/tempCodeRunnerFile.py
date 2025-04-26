import random
import math
import matplotlib.pyplot as plt

# Constants
SINK_NODE = (100, 100)  # Sink's position
NUM_NODES = 100  # Number of nodes
NODE_RANGE = 100  # Maximum range of nodes in X and Y
MAX_ENERGY = 1000  # Maximum energy for a node
ENERGY_PER_HOP = 50  # Energy consumed per hop
ENERGY_PER_TRANSMISSION = 10  # Energy consumed per transmission
RANGE = 30  # Communication range

# Node Class
class Node:
    def __init__(self, x, y, energy=MAX_ENERGY):
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True
        self.distance_to_sink = math.sqrt((self.x - SINK_NODE[0])**2 + (self.y - SINK_NODE[1])**2)

    def consume_energy(self):
        if self.energy > 0:
            self.energy -= ENERGY_PER_HOP
        if self.energy <= 0:
            self.alive = False

# Generate nodes randomly
def generate_nodes(num_nodes):
    nodes = []
    for _ in range(num_nodes):
        x = random.randint(0, NODE_RANGE)
        y = random.randint(0, NODE_RANGE)
        nodes.append(Node(x, y))
    return nodes

# Euclidean Distance
def euclidean_distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

# Dijkstra's Algorithm (Simplified for this example)
def dijkstra(nodes):
    # Create a graph with nodes and distances
    distances = {i: float('inf') for i in range(len(nodes))}
    distances[0] = 0  # Distance from the first node to itself is zero
    previous_nodes = [None] * len(nodes)
    nodes_to_visit = nodes.copy()
    
    while nodes_to_visit:
        # Get node with the smallest distance
        current_node = min(nodes_to_visit, key=lambda node: distances[nodes.index(node)])
        nodes_to_visit.remove(current_node)
        
        # Update distances for neighboring nodes
        for neighbor in nodes:
            if neighbor != current_node:
                dist = euclidean_distance(current_node, neighbor)
                new_dist = distances[nodes.index(current_node)] + dist
                if new_dist < distances[nodes.index(neighbor)]:
                    distances[nodes.index(neighbor)] = new_dist
                    previous_nodes[nodes.index(neighbor)] = current_node
    return distances

# A* Algorithm
def a_star(nodes):
    open_set = [nodes[0]]  # Starting node
    came_from = {}
    g_score = {i: float('inf') for i in range(len(nodes))}
    g_score[0] = 0
    f_score = {i: float('inf') for i in range(len(nodes))}
    f_score[0] = euclidean_distance(nodes[0], SINK_NODE)
    
    while open_set:
        current_node = min(open_set, key=lambda node: f_score[nodes.index(node)])
        open_set.remove(current_node)
        
        if current_node.x == SINK_NODE[0] and current_node.y == SINK_NODE[1]:
            break
        
        for neighbor in nodes:
            if neighbor != current_node:
                tentative_g_score = g_score[nodes.index(current_node)] + euclidean_distance(current_node, neighbor)
                if tentative_g_score < g_score[nodes.index(neighbor)]:
                    came_from[nodes.index(neighbor)] = current_node
                    g_score[nodes.index(neighbor)] = tentative_g_score
                    f_score[nodes.index(neighbor)] = g_score[nodes.index(neighbor)] + euclidean_distance(neighbor, SINK_NODE)
                    if neighbor not in open_set:
                        open_set.append(neighbor)
    return g_score

# Hybrid Shortest Path Algorithm (HybridSP)
def hybrid_sp(nodes):
    distances = dijkstra(nodes)  # Use Dijkstra for initial paths
    for node in nodes:
        if node.alive:
            node.consume_energy()
    return distances

# LEACH Algorithm (Low Energy Adaptive Clustering Hierarchy)
def leach(nodes):
    clusters = []
    for node in nodes:
        if node.alive and node.energy > ENERGY_PER_TRANSMISSION:
            cluster = {'center': node, 'members': []}
            for other_node in nodes:
                if other_node != node and euclidean_distance(node, other_node) < RANGE:
                    cluster['members'].append(other_node)
            clusters.append(cluster)
    for cluster in clusters:
        cluster['center'].consume_energy()
        for member in cluster['members']:
            member.consume_energy()
    return clusters

# Simulate and compare performance of all algorithms
def simulate():
    nodes = generate_nodes(NUM_NODES)
    algorithms = ['Dijkstra', 'A*', 'HybridSP', 'LEACH']
    alive_nodes = {algo: [] for algo in algorithms}
    time_to_sink = {algo: [] for algo in algorithms}

    rounds = 50  # Number of rounds in the simulation
    for round_num in range(rounds):
        for algo in algorithms:
            if algo == 'Dijkstra':
                distances = dijkstra(nodes)
            elif algo == 'A*':
                distances = a_star(nodes)
            elif algo == 'HybridSP':
                distances = hybrid_sp(nodes)
            elif algo == 'LEACH':
                clusters = leach(nodes)

            # Track the number of alive nodes and time taken to reach sink for each round
            alive_nodes[algo].append(sum(1 for node in nodes if node.alive))
            time_to_sink[algo].append(min(distances, key=distances.get) if distances else None)

            # Simulate node energy depletion
            for node in nodes:
                if not node.alive:
                    node.consume_energy()

    # Plot the results
    plot_results(alive_nodes, time_to_sink)

# Plot Comparison of Algorithms
def plot_results(alive_nodes, time_to_sink):
    fig, ax1 = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Alive Nodes Comparison
    for algo in alive_nodes:
        ax1[0].plot(alive_nodes[algo], label=algo)
    ax1[0].set_title('Alive Nodes Comparison')
    ax1[0].set_xlabel('Rounds')
    ax1[0].set_ylabel('Alive Nodes')
    ax1[0].legend()

    # Plot Time to Reach Sink Comparison
    for algo in time_to_sink:
        ax1[1].plot(time_to_sink[algo], label=algo)
    ax1[1].set_title('Time to Reach Sink Comparison')
    ax1[1].set_xlabel('Rounds')
    ax1[1].set_ylabel('Time to Sink')
    ax1[1].legend()

    plt.tight_layout()
    plt.show()

# Run the simulation
simulate()
import random
import math
import matplotlib.pyplot as plt

# Constants
SINK_NODE = (100, 100)  # Sink's position
NUM_NODES = 100  # Number of nodes
NODE_RANGE = 100  # Maximum range of nodes in X and Y
MAX_ENERGY = 1000  # Maximum energy for a node
ENERGY_PER_HOP = 50  # Energy consumed per hop
ENERGY_PER_TRANSMISSION = 10  # Energy consumed per transmission
RANGE = 30  # Communication range

# Node Class
class Node:
    def __init__(self, x, y, energy=MAX_ENERGY):
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True
        self.distance_to_sink = math.sqrt((self.x - SINK_NODE[0])**2 + (self.y - SINK_NODE[1])**2)

    def consume_energy(self):
        if self.energy > 0:
            self.energy -= ENERGY_PER_HOP
        if self.energy <= 0:
            self.alive = False

# Generate nodes randomly
def generate_nodes(num_nodes):
    nodes = []
    for _ in range(num_nodes):
        x = random.randint(0, NODE_RANGE)
        y = random.randint(0, NODE_RANGE)
        nodes.append(Node(x, y))
    return nodes

# Euclidean Distance
def euclidean_distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

# Dijkstra's Algorithm (Simplified for this example)
def dijkstra(nodes):
    # Create a graph with nodes and distances
    distances = {i: float('inf') for i in range(len(nodes))}
    distances[0] = 0  # Distance from the first node to itself is zero
    previous_nodes = [None] * len(nodes)
    nodes_to_visit = nodes.copy()
    
    while nodes_to_visit:
        # Get node with the smallest distance
        current_node = min(nodes_to_visit, key=lambda node: distances[nodes.index(node)])
        nodes_to_visit.remove(current_node)
        
        # Update distances for neighboring nodes
        for neighbor in nodes:
            if neighbor != current_node:
                dist = euclidean_distance(current_node, neighbor)
                new_dist = distances[nodes.index(current_node)] + dist
                if new_dist < distances[nodes.index(neighbor)]:
                    distances[nodes.index(neighbor)] = new_dist
                    previous_nodes[nodes.index(neighbor)] = current_node
    return distances

# A* Algorithm
def a_star(nodes):
    open_set = [nodes[0]]  # Starting node
    came_from = {}
    g_score = {i: float('inf') for i in range(len(nodes))}
    g_score[0] = 0
    f_score = {i: float('inf') for i in range(len(nodes))}
    f_score[0] = euclidean_distance(nodes[0], SINK_NODE)
    
    while open_set:
        current_node = min(open_set, key=lambda node: f_score[nodes.index(node)])
        open_set.remove(current_node)
        
        if current_node.x == SINK_NODE[0] and current_node.y == SINK_NODE[1]:
            break
        
        for neighbor in nodes:
            if neighbor != current_node:
                tentative_g_score = g_score[nodes.index(current_node)] + euclidean_distance(current_node, neighbor)
                if tentative_g_score < g_score[nodes.index(neighbor)]:
                    came_from[nodes.index(neighbor)] = current_node
                    g_score[nodes.index(neighbor)] = tentative_g_score
                    f_score[nodes.index(neighbor)] = g_score[nodes.index(neighbor)] + euclidean_distance(neighbor, SINK_NODE)
                    if neighbor not in open_set:
                        open_set.append(neighbor)
    return g_score

# Hybrid Shortest Path Algorithm (HybridSP)
def hybrid_sp(nodes):
    distances = dijkstra(nodes)  # Use Dijkstra for initial paths
    for node in nodes:
        if node.alive:
            node.consume_energy()
    return distances

# LEACH Algorithm (Low Energy Adaptive Clustering Hierarchy)
def leach(nodes):
    clusters = []
    for node in nodes:
        if node.alive and node.energy > ENERGY_PER_TRANSMISSION:
            cluster = {'center': node, 'members': []}
            for other_node in nodes:
                if other_node != node and euclidean_distance(node, other_node) < RANGE:
                    cluster['members'].append(other_node)
            clusters.append(cluster)
    for cluster in clusters:
        cluster['center'].consume_energy()
        for member in cluster['members']:
            member.consume_energy()
    return clusters

# Simulate and compare performance of all algorithms
def simulate():
    nodes = generate_nodes(NUM_NODES)
    algorithms = ['Dijkstra', 'A*', 'HybridSP', 'LEACH']
    alive_nodes = {algo: [] for algo in algorithms}
    time_to_sink = {algo: [] for algo in algorithms}

    rounds = 50  # Number of rounds in the simulation
    for round_num in range(rounds):
        for algo in algorithms:
            if algo == 'Dijkstra':
                distances = dijkstra(nodes)
            elif algo == 'A*':
                distances = a_star(nodes)
            elif algo == 'HybridSP':
                distances = hybrid_sp(nodes)
            elif algo == 'LEACH':
                clusters = leach(nodes)

            # Track the number of alive nodes and time taken to reach sink for each round
            alive_nodes[algo].append(sum(1 for node in nodes if node.alive))
            time_to_sink[algo].append(min(distances, key=distances.get) if distances else None)

            # Simulate node energy depletion
            for node in nodes:
                if not node.alive:
                    node.consume_energy()

    # Plot the results
    plot_results(alive_nodes, time_to_sink)

# Plot Comparison of Algorithms
def plot_results(alive_nodes, time_to_sink):
    fig, ax1 = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Alive Nodes Comparison
    for algo in alive_nodes:
        ax1[0].plot(alive_nodes[algo], label=algo)
    ax1[0].set_title('Alive Nodes Comparison')
    ax1[0].set_xlabel('Rounds')
    ax1[0].set_ylabel('Alive Nodes')
    ax1[0].legend()

    # Plot Time to Reach Sink Comparison
    for algo in time_to_sink:
        ax1[1].plot(time_to_sink[algo], label=algo)
    ax1[1].set_title('Time to Reach Sink Comparison')
    ax1[1].set_xlabel('Rounds')
    ax1[1].set_ylabel('Time to Sink')
    ax1[1].legend()

    plt.tight_layout()
    plt.show()

# Run the simulation
simulate()
