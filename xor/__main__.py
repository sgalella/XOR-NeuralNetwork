import matplotlib.pyplot as plt
from numpy.random import seed

from .genetic_algorithm import GeneticAlgorithm
from .visualization import plot_fitness, plot_diversity
from . import mutation, recombination, selection

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
recombination_type = recombination.arithmetic
selection_type = selection.genitor

# Initialize genetic algorithm
ga = GeneticAlgorithm(lower_bound, upper_bound, alpha, num_iterations, population_size, offspring_size, mutation_rate,
                      mutation_type, recombination_type, selection_type)

# Run algorithm
solutions, max_fitness, mean_fitness, diversity_genotype, diversity_phenotype = ga.run()

# Print
plot_fitness(mean_fitness, max_fitness)
plot_diversity(diversity_genotype, diversity_phenotype)
plt.show()
