import random
from utils.helpers import total_chain_distance
from genetic.ga_optimizer import run_ga

def form_pegasis_chain(network):
    nodes = [n for n in network.nodes if n.is_alive()]
    if len(nodes) <= 1:
        return []
    best_order = run_ga(nodes)
    chain = [nodes[i] for i in best_order]
    return chain