import matplotlib.pyplot as plt


def plot_results(solution, mean_fitness, max_fitness, diversity_genotype, diversity_phenotype):
    """
    Prints the algorithm convergence.

    Args:
        solution (np.array): Permutation of rows containing queens.
    """
    # Convergence figure
    plt.figure()
    plt.plot(range(len(mean_fitness)), mean_fitness, 'b')
    plt.plot(range(len(max_fitness)), max_fitness, 'r--')
    plt.legend(("mean fitness", "max fitness"))
    plt.xlabel('iterations')
    plt.ylabel('fitness')
    plt.title('Fitness through generations')
    plt.grid(alpha=0.3)
    plt.savefig('images/convergence.png')
    plt.show()
