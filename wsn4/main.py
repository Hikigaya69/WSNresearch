import os
import random
import math
import csv
import heapq
import matplotlib.pyplot as plt
import pandas as pd

# 📋 Config
FIELD_X, FIELD_Y = 100, 100
SINK_X, SINK_Y = 50, 200
NUM_NODES = 40
INIT_ENERGY = 2
PACKET_SIZE = 4000
TEMP_RANGE = (20, 50)
HUMIDITY_RANGE = (30, 90)
FAIL_TEMP = 45
FAIL_HUM = 85
MAX_ROUNDS = 100
E_ELEC = 50e-9
E_AMP = 100e-12
E_DA = 5e-9

os.makedirs("csv", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# 📦 Node
class Node:
    def __init__(self, nid, x, y):
        self.id = nid
        self.x = x
        self.y = y
        self.energy = INIT_ENERGY
        self.dead = False
        self.temp = random.uniform(*TEMP_RANGE)
        self.hum = random.uniform(*HUMIDITY_RANGE)
        self.parent = None

    def dist(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def fail_env(self):
        if self.temp > FAIL_TEMP or self.hum > FAIL_HUM:
            if random.random() < 0.25:
                self.dead = True

    def tx(self, kbits, d):
        cost = E_ELEC * kbits + E_AMP * kbits * d**2
        self.energy -= cost
        if self.energy <= 0:
            self.dead = True
        return cost

    def rx(self, kbits):
        cost = (E_ELEC + E_DA) * kbits
        self.energy -= cost
        if self.energy <= 0:
            self.dead = True
        return cost

# 🌐 Init Network
def create_net():
    nodes = [Node(i, random.uniform(0, FIELD_X), random.uniform(0, FIELD_Y)) for i in range(NUM_NODES)]
    sink = Node("sink", SINK_X, SINK_Y)
    sink.energy = float('inf')
    return nodes, sink

# 🚨 Transmission Simulation
def simulate(chain, sink):
    used, reached = 0, 0
    for n in chain:
        n.fail_env()
        if n.dead or not n.parent:
            continue
        d = n.dist(n.parent)
        used += n.tx(PACKET_SIZE, d)
        if n.parent != sink:
            used += n.parent.rx(PACKET_SIZE)
        else:
            reached += 1
    return used, reached

# ⚙️ A* Algorithm
def a_star_with_energy(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []

    source = alive[0]
    g_score = {n: float('inf') for n in alive}
    g_score[source] = 0

    f_score = {n: float('inf') for n in alive}
    f_score[source] = source.dist(sink)

    came_from = {}

    open_set = [(f_score[source], source)]
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current.dist(sink) < 20:
            break

        for neighbor in alive:
            if neighbor == current or neighbor in visited:
                continue
            d = current.dist(neighbor)
            tx_cost = E_ELEC * PACKET_SIZE + E_AMP * PACKET_SIZE * d**2
            rx_cost = (E_ELEC + E_DA) * PACKET_SIZE
            cost = g_score[current] + tx_cost + rx_cost

            if cost < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = cost
                f_score[neighbor] = cost + neighbor.dist(sink)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    for node in alive:
        node.parent = came_from.get(node, None)
    source.parent = sink
    return alive

# 🧭 Dijkstra Algorithm
def dijkstra_with_energy(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    for node in alive:
        node.energy = INIT_ENERGY
    g_score = {n: float('inf') for n in alive}
    g_score[alive[0]] = 0
    open_set = [(0, alive[0])]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current.dist(sink) < 20:
            break
        for neighbor in alive:
            if neighbor == current:
                continue
            dist = current.dist(neighbor)
            tx_cost = current.tx(PACKET_SIZE, dist)
            rx_cost = neighbor.rx(PACKET_SIZE)
            total = g_score[current] + tx_cost + rx_cost
            if total < g_score[neighbor]:
                g_score[neighbor] = total
                neighbor.parent = current
                heapq.heappush(open_set, (total, neighbor))

    return alive

# 🧩 Hybrid A* + Dijkstra
def hybrid_a_star_dijkstra(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []

    source = alive[0]
    g_score = {n: float('inf') for n in alive}
    g_score[source] = 0

    f_score = {n: float('inf') for n in alive}
    f_score[source] = source.dist(sink)

    came_from = {}

    open_set = [(f_score[source], source)]
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current.dist(sink) < 20:
            break

        for neighbor in alive:
            if neighbor == current or neighbor in visited:
                continue
            d = current.dist(neighbor)
            tx_cost = E_ELEC * PACKET_SIZE + E_AMP * PACKET_SIZE * d**2
            rx_cost = (E_ELEC + E_DA) * PACKET_SIZE
            cost = g_score[current] + tx_cost + rx_cost

            if cost < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = cost
                f_score[neighbor] = cost + neighbor.dist(sink)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    for node in alive:
        node.parent = came_from.get(node, None)
    source.parent = sink
    return alive

# 🌿 LEACH Algorithm
def leach_with_energy(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    chs = sorted(alive, key=lambda n: n.energy, reverse=True)[:max(1, len(alive)//10)]
    for n in alive:
        closest_ch = min(chs, key=lambda ch: n.dist(ch))
        n.parent = closest_ch
    for ch in chs:
        ch.parent = sink
    return alive

# 🐍 Bellman-Ford Algorithm
def bellman_ford_with_energy(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    distance = {n: float('inf') for n in alive}
    parent = {n: None for n in alive}
    source = alive[0]
    distance[source] = 0

    for _ in range(len(alive) - 1):
        for node in alive:
            for neighbor in alive:
                if node == neighbor: continue
                d = node.dist(neighbor)
                cost = E_ELEC * PACKET_SIZE + E_AMP * PACKET_SIZE * d**2 + (E_ELEC + E_DA) * PACKET_SIZE
                if distance[node] + cost < distance[neighbor]:
                    distance[neighbor] = distance[node] + cost
                    parent[neighbor] = node

    for n in alive:
        n.parent = parent[n] if parent[n] else sink
    source.parent = sink
    return alive

# 🧪 Simulation Loop
protocols = {
    "AStar": a_star_with_energy,
    "Dijkstra": dijkstra_with_energy,
    "LEACH": leach_with_energy,
    "BellmanFord": bellman_ford_with_energy,
    "HybridAStarDijkstra": hybrid_a_star_dijkstra  # Adding hybrid protocol
}

def run():
    results = {}
    for name, algo in protocols.items():
        nodes, sink = create_net()
        stats = []
        for r in range(MAX_ROUNDS):
            chain = algo(nodes, sink)
            used, reached = simulate(chain, sink)
            alive = sum(1 for n in nodes if not n.dead)
            avg_dist = sum(n.dist(sink) for n in nodes if not n.dead) / max(1, alive)
            stats.append((r + 1, used, alive, reached, avg_dist))
            if alive == 0:
                break
        results[name] = stats
        with open(f"csv/{name}.csv", "w", newline='') as f:
            csv.writer(f).writerows([["Round", "EnergyUsed", "AliveNodes", "ReachedSink", "AvgDistToSink"]] + stats)
    return results

# 📈 Plot
def plot_compare(res):
    for idx, label in zip([1, 2, 3, 4], ["Energy", "Alive Nodes", "Reached Sink", "Avg Distance"]):
        plt.figure()
        for name, data in res.items():
            x = [r[0] for r in data]
            y = [r[idx] for r in data]
            plt.plot(x, y, label=name)
        plt.title(f"{label} Comparison")
        plt.xlabel("Rounds")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"graphs/{label.replace(' ', '_').lower()}.png")
    plt.show()

# 📊 Final Summary Table
def final_metrics_table(results):
    summary = []
    for algo, data in results.items():
        total_energy = sum(r[1] for r in data)
        total_alive = data[-1][2]
        total_reached = sum(r[3] for r in data)
        avg_dist = sum(r[4] for r in data) / len(data)
        summary.append([algo, round(total_energy, 6), total_alive, total_reached, round(avg_dist, 2)])
    df = pd.DataFrame(summary, columns=["Algorithm", "TotalEnergyUsed", "FinalAliveNodes", "TotalReachedSink", "AvgDistToSink"])
    df.sort_values("TotalEnergyUsed", inplace=True)
    print("\n📊 Final Metrics Table:")
    print(df.to_string(index=False))
    df.to_csv("csv/final_metrics_summary.csv", index=False)

# ▶️ Execute
res = run()
plot_compare(res)
final_metrics_table(res)
