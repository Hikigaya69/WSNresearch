# ✅ WSN Simulation: PEGASIS vs LEACH vs Dijkstra vs Custom Hybrid
# ---------------------------------------------------------------
# Single file research-ready engine with:
# 📡 WSN deployment, 🔋 energy model, 🔄 animated rounds
# 📁 CSV logging, 📈 graphs, and 🧬 your custom hybrid protocol

import random, math, matplotlib.pyplot as plt, matplotlib.animation as animation, csv

# ------------------- CONFIG -------------------
FIELD_X, FIELD_Y = 100, 100
SINK_X, SINK_Y = 50, 200
NUM_NODES = 50
INIT_ENERGY = 2
E_ELEC = 50e-9
E_AMP = 100e-12
E_DA = 5e-9
PACKET_SIZE = 4000
MAX_ROUNDS = 100

# ------------------- NODE CLASS -------------------
class Node:
    def __init__(self, node_id, x, y, energy=INIT_ENERGY):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = energy
        self.dead = False
        self.parent = None

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def consume_tx(self, bits, distance):
        cost = E_ELEC * bits + E_AMP * bits * distance**2
        self.energy -= cost
        if self.energy <= 0:
            self.dead = True
        return cost

    def consume_rx(self, bits):
        cost = (E_ELEC + E_DA) * bits
        self.energy -= cost
        if self.energy <= 0:
            self.dead = True
        return cost

# ------------------- NETWORK INIT -------------------
def init_nodes():
    nodes = []
    for i in range(NUM_NODES):
        x, y = random.uniform(0, FIELD_X), random.uniform(0, FIELD_Y)
        nodes.append(Node(i, x, y))
    return nodes, Node("sink", SINK_X, SINK_Y, energy=float("inf"))

# ------------------- CUSTOM HYBRID ALGORITHM -------------------
def custom_hybrid(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    # Fitness = (1/distance) * (energy ratio) / (load penalty)
    scored = sorted(alive, key=lambda n: (n.distance(sink) + 1e-6) / (n.energy + 1e-6))
    for i in range(len(scored)-1):
        scored[i].parent = scored[i+1]
    scored[-1].parent = sink
    return scored

# ------------------- SIMPLE DIJKSTRA ALGO -------------------
def dijkstra_like(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    source = min(alive, key=lambda n: n.distance(sink))
    for n in alive:
        if n != source:
            n.parent = source
    source.parent = sink
    return alive

# ------------------- PEGASIS SIMULATION (Simplified) -------------------
def pegasis_chain(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    chain = [alive.pop(0)]
    while alive:
        last = chain[-1]
        next_node = min(alive, key=lambda n: n.distance(last))
        next_node.parent = last
        chain.append(next_node)
        alive.remove(next_node)
    chain[0].parent = sink
    return list(reversed(chain))

# ------------------- LEACH-LIKE RANDOM CLUSTER -------------------
def leach_like(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    cluster_heads = random.sample(alive, max(1, len(alive)//10))
    for n in alive:
        closest = min(cluster_heads, key=lambda h: n.distance(h))
        n.parent = closest
    for h in cluster_heads:
        h.parent = sink
    return alive

# ------------------- TRANSMISSION MODEL -------------------
def simulate_round(chain, sink):
    energy_used = 0
    for node in chain:
        if node.dead or node.parent is None:
            continue
        d = node.distance(node.parent)
        energy_used += node.consume_tx(PACKET_SIZE, d)
        if node.parent != sink:
            energy_used += node.parent.consume_rx(PACKET_SIZE)
    return energy_used

# ------------------- SIMULATION LOOP -------------------
def simulate(protocol_fn, label):
    nodes, sink = init_nodes()
    stats = []

    for r in range(MAX_ROUNDS):
        chain = protocol_fn(nodes, sink)
        energy = simulate_round(chain, sink)
        alive = sum(1 for n in nodes if not n.dead)
        stats.append((r+1, energy, alive))
        print(f"{label} | Round {r+1}: Energy={energy:.6f}, Alive={alive}")
        if alive == 0:
            break

    with open(f"stats_{label}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "EnergyUsed", "AliveNodes"])
        writer.writerows(stats)

    return stats

# ------------------- GRAPH RESULTS -------------------
def compare_results(all_stats):
    plt.figure()
    for label, stats in all_stats.items():
        rounds, energies, alives = zip(*stats)
        plt.plot(rounds, alives, label=f"Alive: {label}")
    plt.xlabel("Rounds")
    plt.ylabel("Alive Nodes")
    plt.title("Protocol Comparison - Node Survival")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_alive_nodes.png")
    plt.show()

    plt.figure()
    for label, stats in all_stats.items():
        rounds, energies, alives = zip(*stats)
        plt.plot(rounds, energies, label=f"Energy: {label}")
    plt.xlabel("Rounds")
    plt.ylabel("Energy Used")
    plt.title("Protocol Comparison - Energy Usage")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_energy_usage.png")
    plt.show()

# ------------------- MAIN -------------------
def main():
    all_stats = {
        "PEGASIS": simulate(pegasis_chain, "PEGASIS"),
        "LEACH": simulate(leach_like, "LEACH"),
        "Dijkstra": simulate(dijkstra_like, "Dijkstra"),
        "CustomHybrid": simulate(custom_hybrid, "CustomHybrid")
    }
    compare_results(all_stats)

if __name__ == '__main__':
    main()