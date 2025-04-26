from models.network import Network
from protocols.pegasus_ga import form_pegasis_chain
from simulation.transmission import transmit
from visualization.plot_network import plot_network
from visualization.stats_plot import plot_stats

energy_record = []
alive_record = []

def main():
    net = Network()
    for round in range(100):
        chain = form_pegasis_chain(net)
        if len(chain) == 0:
            break
        used = transmit(chain, net.sink)
        energy_record.append(used)
        alive = len([n for n in net.nodes if n.is_alive()])
        alive_record.append(alive)
        print(f"Round {round+1}: Energy Used = {used:.5f}, Alive = {alive}")

    plot_network(net)
    plot_stats(energy_record, alive_record)

if __name__ == '__main__':
    main()