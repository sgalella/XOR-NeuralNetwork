import numpy as np


def uniform(individual, upper_bound, lower_bound, gene1=None):
    """
    Mutates indidividual by using the uniform method.

    Args:
        individual (np.array): Original individual.

    Returns:
        mutated_individual (np.array): Individual mutated.
    """
    mutated_individual = individual.copy()
    gene1 = np.random.randint(9)
    mutated_individual[gene1] = (upper_bound - lower_bound) * np.random.random() + lower_bound
    return mutated_individual
