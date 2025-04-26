# ✅ WSN Comparison: LEACH, Dijkstra, A*, HybridSP with Output Export
# ------------------------------------------------
# Generates CSVs, saves graphs to folder

import os
import random
import math
import csv
import heapq
import matplotlib.pyplot as plt

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

# 📁 Ensure output directories
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

# 🌐 Init Net

def create_net():
    nodes = [Node(i, random.uniform(0, FIELD_X), random.uniform(0, FIELD_Y)) for i in range(NUM_NODES)]
    sink = Node("sink", SINK_X, SINK_Y)
    sink.energy = float('inf')
    return nodes, sink

# 🚨 Transmission

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

# ⚙️ A* Algo

def a_star(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    open_set = [(0, alive[0])]
    came_from = {}
    g_score = {n: float('inf') for n in alive}
    g_score[alive[0]] = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current.dist(sink) < 20:
            break
        for neighbor in alive:
            if neighbor == current:
                continue
            temp_g = g_score[current] + current.dist(neighbor)
            if temp_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g
                f = temp_g + neighbor.dist(sink)
                heapq.heappush(open_set, (f, neighbor))

    for n in came_from:
        n.parent = came_from[n]
    alive[0].parent = sink
    return alive

# 🚀 LEACH

def leach(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    chs = random.sample(alive, max(1, len(alive)//10))
    for n in alive:
        closest = min(chs, key=lambda h: n.dist(h))
        n.parent = closest
    for ch in chs:
        ch.parent = sink
    return alive

# 🧠 Dijkstra (star pattern)

def dijkstra(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    for n in alive:
        n.parent = sink
    return alive

# 🧬 HybridSP

def hybrid(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    scored = sorted(alive, key=lambda n: (n.dist(sink)+1e-6)/(n.energy+1e-6))
    for i in range(len(scored)-1):
        scored[i].parent = scored[i+1]
    scored[-1].parent = sink
    return scored

# 🧪 Simulation Loop
protocols = {
    "LEACH": leach,
    "Dijkstra": dijkstra,
    "AStar": a_star,
    "HybridSP": hybrid
}

def run():
    results = {}
    for name, proto in protocols.items():
        nodes, sink = create_net()
        stat = []
        for r in range(MAX_ROUNDS):
            chain = proto(nodes, sink)
            used, reached = simulate(chain, sink)
            alive = sum(1 for n in nodes if not n.dead)
            avg_dist = sum(n.dist(sink) for n in nodes if not n.dead) / max(1, alive)
            stat.append((r+1, used, alive, reached, avg_dist))
            if alive == 0:
                break
        results[name] = stat
        with open(f"csv/{name}.csv", "w", newline='') as f:
            csv.writer(f).writerows([["Round", "EnergyUsed", "AliveNodes", "ReachedSink", "AvgDistToSink"]]+stat)
    return results

# 📈 Plot

def plot_compare(res):
    for metric_idx, label in zip([1, 2, 3, 4], ["Energy", "Alive Nodes", "Reached Sink", "Avg Distance"]):
        plt.figure()
        for name, data in res.items():
            x = [r[0] for r in data]
            y = [r[metric_idx] for r in data]
            plt.plot(x, y, label=name)
        plt.title(f"{label} Comparison")
        plt.xlabel("Rounds")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"graphs/{label.replace(' ', '_').lower()}.png")
    plt.show()

# ▶️ Execute
res = run()
plot_compare(res)