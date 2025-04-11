import pygad
from genetic.fitness import fitness_func

solution = []

def run_ga(nodes):
    gene_space = list(range(len(nodes)))

    ga = pygad.GA(
        num_generations=100,
        num_parents_mating=10,
        fitness_func=lambda sol, idx: fitness_func(sol, nodes),
        sol_per_pop=20,
        num_genes=len(nodes),
        gene_space=gene_space,
        parent_selection_type="tournament",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        stop_criteria="reach_0.01",
        allow_duplicate_genes=False
    )

    ga.run()
    solution, _, _ = ga.best_solution()
    return list(map(int, solution))