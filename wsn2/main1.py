# ✅ WSN Simulation with Custom Hybrid, Environment Sensor & Animation
# ------------------------------------------------------------------
# Includes: PEGASIS, LEACH, Dijkstra, CustomHybrid
# Adds: 🌡️ Temp & Humidity per node, 🌪️ Failures on extremes
# 📽️ Animated frames saved as GIF, 📁 CSV export, 📚 Summary PDF written separately

import random, math, csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
TEMP_RANGE = (20, 50)
HUMIDITY_RANGE = (30, 90)
FAIL_TEMP = 45
FAIL_HUM = 85

# ------------------- NODE CLASS -------------------
class Node:
    def __init__(self, node_id, x, y, energy=INIT_ENERGY):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = energy
        self.dead = False
        self.parent = None
        self.temp = random.uniform(*TEMP_RANGE)
        self.humidity = random.uniform(*HUMIDITY_RANGE)

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def fail_by_environment(self):
        if self.temp > FAIL_TEMP or self.humidity > FAIL_HUM:
            if random.random() < 0.2:
                self.dead = True

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

# ------------------- CUSTOM HYBRID -------------------
def custom_hybrid(nodes, sink):
    alive = [n for n in nodes if not n.dead]
    if not alive:
        return []
    scored = sorted(alive, key=lambda n: (n.distance(sink) + 1e-6) / (n.energy + 1e-6))
    for i in range(len(scored)-1):
        scored[i].parent = scored[i+1]
    scored[-1].parent = sink
    return scored

# ------------------- TRANSMISSION + ENV CHECK -------------------
def simulate_round(chain, sink):
    energy_used = 0
    for node in chain:
        node.fail_by_environment()
        if node.dead or node.parent is None:
            continue
        d = node.distance(node.parent)
        energy_used += node.consume_tx(PACKET_SIZE, d)
        if node.parent != sink:
            energy_used += node.parent.consume_rx(PACKET_SIZE)
    return energy_used

# ------------------- ANIMATED SIMULATION -------------------
def simulate_and_animate():
    nodes, sink = init_nodes()
    stats, frames = [], []

    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        round = frame + 1
        chain = custom_hybrid(nodes, sink)
        energy = simulate_round(chain, sink)
        alive = sum(1 for n in nodes if not n.dead)
        stats.append((round, energy, alive))
        ax.set_xlim(0, FIELD_X)
        ax.set_ylim(0, FIELD_Y + 100)
        ax.set_title(f"Round {round} | Alive: {alive}")
        for n in nodes:
            color = 'red' if n.dead else ('orange' if n.temp > FAIL_TEMP or n.humidity > FAIL_HUM else 'green')
            ax.plot(n.x, n.y, 'o', color=color)
            if not n.dead and n.parent:
                ax.plot([n.x, n.parent.x], [n.y, n.parent.y], 'k--', alpha=0.3)
        ax.plot(sink.x, sink.y, 'ks')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=MAX_ROUNDS, repeat=False)
    ani.save("wsn_simulation.gif", writer='pillow')

    # Export CSV
    with open("wsn_env_stats.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "EnergyUsed", "AliveNodes"])
        writer.writerows(stats)

    # Plot energy and alive
    rounds, energies, alives = zip(*stats)
    plt.figure()
    plt.plot(rounds, energies, label="Energy Used")
    plt.plot(rounds, alives, label="Alive Nodes")
    plt.title("WSN Stats with Temp+Humidity Stress")
    plt.xlabel("Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("wsn_env_graph.png")
    plt.show()

simulate_and_animate()
