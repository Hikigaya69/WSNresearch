import random
import matplotlib.pyplot as plt
import pygad
import math

# -------------------- CONFIG --------------------
FIELD_X = 100
FIELD_Y = 100
SINK_X = 50
SINK_Y = 200
NUM_NODES = 100
E_INITIAL = 2
E_ELEC = 50e-9
E_AMP = 100e-12
E_DA = 5e-9
PACKET_SIZE = 4000
GA_POP_SIZE = 50
GA_NUM_GENERATIONS = 100
GA_MUTATION_RATE = 0.1

# -------------------- NODE --------------------
class Node:
    def __init__(self, node_id, x, y, energy):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = energy
        self.dead = False
        self.role = "normal"
        self.closest = None
        self.prev = None
        self.distance_to_sink = 0

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def consume_tx(self, bits, distance):
        cost = E_ELEC * bits + E_AMP * bits * (distance ** 2)
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

    def is_alive(self):
        return not self.dead

# -------------------- NETWORK --------------------
class Network:
    def __init__(self):
        self.nodes = []
        self.sink = Node("sink", SINK_X, SINK_Y, float("inf"))
        self.init_nodes()

    def init_nodes(self):
        for i in range(NUM_NODES):
            x = random.uniform(0, FIELD_X)
            y = random.uniform(0, FIELD_Y)
            node = Node(i, x, y, E_INITIAL)
            node.distance_to_sink = node.distance(self.sink)
            self.nodes.append(node)

    def get_alive_nodes(self):
        return [n for n in self.nodes if n.is_alive()]

# -------------------- FIXED: GA + Fitness --------------------
def fitness_func(ga_instance, solution, solution_idx):
    dist = 0
    nodes = ga_instance.nodes
    for i in range(len(solution) - 1):
        a = nodes[int(solution[i])]
        b = nodes[int(solution[i+1])]
        dist += a.distance(b)
    return -dist  # minimize distance

def run_ga(nodes):
    gene_space = list(range(len(nodes)))

    def on_start(ga_instance):
        ga_instance.nodes = nodes

    ga = pygad.GA(
        num_generations=GA_NUM_GENERATIONS,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=GA_POP_SIZE,
        num_genes=len(nodes),
        gene_space=gene_space,
        parent_selection_type="tournament",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=int(GA_MUTATION_RATE * 100),
        stop_criteria="reach_0.01",
        allow_duplicate_genes=False,
        on_start=on_start
    )

    ga.run()
    solution, _, _ = ga.best_solution()
    return list(map(int, solution))

# -------------------- PEGASIS --------------------
def form_pegasis_chain(network):
    nodes = [n for n in network.nodes if n.is_alive()]
    if len(nodes) <= 1:
        return []
    order = run_ga(nodes)
    return [nodes[i] for i in order]

# -------------------- TRANSMISSION --------------------
def transmit(chain, sink):
    energy_used = 0
    for i in range(len(chain) - 1):
        tx = chain[i]
        rx = chain[i + 1]
        energy_used += tx.consume_tx(PACKET_SIZE, tx.distance(rx))
        energy_used += rx.consume_rx(PACKET_SIZE)
    last = chain[-1]
    energy_used += last.consume_tx(PACKET_SIZE, last.distance(sink))
    return energy_used

# -------------------- VISUALIZATION --------------------
def plot_network(network, title="WSN Layout"):
    plt.figure()
    for node in network.nodes:
        color = 'red' if node.dead else 'blue'
        plt.plot(node.x, node.y, 'o', color=color)
        plt.text(node.x, node.y, str(node.id), fontsize=6)
    plt.plot(SINK_X, SINK_Y, 'ks', label="Sink")
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.show()

def plot_stats(energy, alive):
    plt.figure()
    plt.plot(range(len(energy)), energy, label='Energy Used')
    plt.plot(range(len(alive)), alive, label='Alive Nodes')
    plt.xlabel("Rounds")
    plt.ylabel("Value")
    plt.title("Simulation Statistics")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------- MAIN --------------------
def main():
    net = Network()
    energy_record = []
    alive_record = []

    for round in range(100):
        chain = form_pegasis_chain(net)
        if not chain:
            break
        used = transmit(chain, net.sink)
        energy_record.append(used)
        alive = len(net.get_alive_nodes())
        alive_record.append(alive)
        print(f"Round {round+1}: Energy Used = {used:.6f}, Alive = {alive}")

    plot_network(net)
    plot_stats(energy_record, alive_record)

if __name__ == '__main__':
    main()
