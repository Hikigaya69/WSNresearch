import matplotlib.pyplot as plt

def plot_stats(energy, alive):
    plt.figure()
    plt.plot(range(len(energy)), energy, label='Energy Used')
    plt.plot(range(len(alive)), alive, label='Alive Nodes')
    plt.xlabel("Rounds")
    plt.legend()
    plt.title("Simulation Stats")
    plt.grid(True)
    plt.show()