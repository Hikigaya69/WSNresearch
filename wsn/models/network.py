import random
from models.node import Node
import config

class Network:
    def __init__(self):
        self.nodes = []
        self.sink = Node("sink", config.SINK_X, config.SINK_Y, float("inf"))
        self.init_nodes()

    def init_nodes(self):
        for i in range(config.NUM_NODES):
            x = random.uniform(0, config.FIELD_X)
            y = random.uniform(0, config.FIELD_Y)
            node = Node(i, x, y, config.E_INITIAL)
            node.distance_to_sink = node.distance(self.sink)
            self.nodes.append(node)

    def get_alive_nodes(self):
        return [n for n in self.nodes if n.is_alive()]
