
def fitness_func(solution, nodes):
    dist = 0
    for i in range(len(solution) - 1):
        a = nodes[int(solution[i])]
        b = nodes[int(solution[i+1])]
        dist += a.distance(b)
    return -dist  # minimize distance
