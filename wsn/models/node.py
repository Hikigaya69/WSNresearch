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
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

    def consume_tx(self, bits, distance):
        from config import E_ELEC, E_AMP
        cost = E_ELEC * bits + E_AMP * bits * (distance ** 2)
        self.energy -= cost
        self.dead = self.energy <= 0
        return cost

    def consume_rx(self, bits):
        from config import E_ELEC, E_DA
        cost = (E_ELEC + E_DA) * bits
        self.energy -= cost
        self.dead = self.energy <= 0
        return cost

    def is_alive(self):
        return not self.dead
