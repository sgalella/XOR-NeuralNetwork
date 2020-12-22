import matplotlib.pyplot as plt


def plot_fitness(mean_fitness, max_fitness):
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
    plt.savefig('images/convergence.jpg')
    plt.draw()


def plot_diversity(diversity_genotype, diversity_phenotype):
    plt.figure()
    plt.bar(range(len(diversity_phenotype)), diversity_genotype, color='lime', alpha=0.5)
    plt.bar(range(len(diversity_phenotype)), diversity_phenotype, color='orange', alpha=0.5)
    plt.legend(("genotype diversity", "phenotype diversity"))
    plt.xlabel('iterations')
    plt.ylabel('diversity')
    plt.title('Diversity through generations')
    plt.grid(alpha=0.3)
    plt.savefig('images/diversity.jpg')
    plt.draw()
