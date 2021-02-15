import numpy as np


def arithmetic(individual1, individual2, alpha, gene1=None, gene2=None):
    """
    Creates a new individual by recombinating two parents using the
    Partially Mapped Crossover (PMX) method.

    Args:
        parent1 (np.array): First parent.
        parent2 (np.array): Second parent.

    Returns:
        new_individual1, new_individual2 (tuple): Recombined individuals.
    """
    # Copy parents
    parent1 = individual1.copy()
    parent2 = individual2.copy()

    # Initialize new individuals
    new_individual1 = parent1.copy()
    new_individual2 = parent2.copy()

    # Arithmetic recombination
    recombination_values = new_individual1 * alpha + new_individual2 * (1 - alpha)

    # Perform the pmx recombination
    # 1. Select two genes at random and copy segment to new individuals
    if gene1 is None or gene2 is None:
        gene1, gene2 = _choose_random_genes(individual1)
    new_individual1[gene1:gene2 + 1] = recombination_values[gene1:gene2 + 1]
    new_individual2[gene1:gene2 + 1] = recombination_values[gene1:gene2 + 1]

    return (new_individual1, new_individual2)


def _choose_random_genes(individual):
    """
    Selects two separate genes from individual.

    Args:
        individual (np.array): Genotype of individual.

    Returns:
        gene1, gene2 (tuple): Genes separated by at least another gene.
    """
    gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
    while gene2 - gene1 < 2:
        gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
    return (gene1, gene2)
