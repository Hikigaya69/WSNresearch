import matplotlib.pyplot as plt
from config import SINK_X, SINK_Y

def plot_network(network, title="WSN Layout", show=True):
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
    if show:
        plt.show()