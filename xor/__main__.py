from numpy.random import seed

from .genetic_algorithm import GeneticAlgorithm
from .utils import plot_results
from . import mutation

# Set random seed (for reproducibility)
random_seed = 1234
seed(random_seed)

# Initialize parameters
lower_bound = -30
upper_bound = 30
alpha = 0.5
num_iterations = 1000
population_size = 100
offspring_size = 20
mutation_rate = 0.05
mutation_type = mutation.uniform
recombination_type = "arithmetic"
selection_type = "genitor"

# Initialize genetic algorithm
ga = GeneticAlgorithm(lower_bound, upper_bound, alpha, num_iterations, population_size, offspring_size, mutation_rate,
                      mutation_type, recombination_type, selection_type)

# Run algorithm
solutions, max_fitness, mean_fitness, diversity_genotype, diversity_phenotype = ga.run()

# Print
plot_results(solutions[0], mean_fitness, max_fitness, diversity_genotype, diversity_phenotype)
print(f"Final weights: {solutions[0]}")
