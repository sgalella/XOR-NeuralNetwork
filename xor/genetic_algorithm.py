import numpy as np
from tqdm import tqdm


class MutationTypeError(Exception):
    """
    Mutation type does not exist.
    """
    def __init__(self, name):
        super().__init__(f"'{name}' mutation type does not exist.")


class RecombinationTypeError(Exception):
    """
    Recombination type does not exist.
    """
    def __init__(self, name):
        super().__init__(f"'{name}' recombination type does not exist.")


class SelectionTypeError(Exception):
    """
    Selection type does not exist.
    """
    def __init__(self, name):
        super().__init__(f"'{name}' selection type does not exist.")


class GeneticAlgorithm:
    """
    Genetic algorithm for TSP.
    """
    def __init__(self, lower_bound=-5, upper_bound=5, alpha=0.5, num_iterations=1000, population_size=100, offspring_size=20, mutation_rate=0.2,
                 mutation_type="uniform", recombination_type="arithmetic", selection_type="genitor"):
        """
        Initializes the algorithm.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self.recombination_type = recombination_type
        self.selection_type = selection_type
        assert self.offspring_size < self.population_size, "Population size has to be greater than the number of selected individuals"

    def __repr__(self):
        """
        Visualizes algorithm parameters when printing.
        """
        return (f"Iterations: {self.num_iterations}\n"
                f"Population size: {self.population_size}\n"
                f"Num selected: {self.num_selected}\n"
                f"Mutation rate: {self.mutation_rate}\n")

    def random_initial_population(self):
        """
        Generates random population of individuals.

        Returns:
            population (np.array): Population containg the different individuals.
        """
        # Initialize population
        population = (self.upper_bound - self.lower_bound) * np.random.random((self.population_size, 9)) + self.lower_bound

        return population

    def sigmoid(self, x):
        """
        Performs the sigmoid activation function on x.

        Args:
            x (np.array): Weighted value of neurons at a given layer.

        Returns:
            Sigmoid activation function function on x.
        """
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, x, individual):
        """
        Performs the forward pass of the network.

        Args:
            x (np.array): Input to the neural network.
            individual (np.array): Values of the weights of the network.

        Returns:
            y (np.array): Value of the output layer.
        """
        w11, w12, w21, w22, wy1, wy2, b1, b2, by = individual
        x1, x2 = x
        h1 = self.sigmoid(w11 * x1 + w12 * x2 + b1)
        h2 = self.sigmoid(w21 * x1 + w22 * x2 + b2)
        y = self.sigmoid(wy1 * h1 + wy2 * h2 + by)
        return y

    def compute_fitness(self, population):
        """
        Computes the fitness for each individual by calculating the accuracy of the network to emulate a XOR.

        Args:
            population (np.array): Population containg the different individuals.

        Returns:
            fitness_population (np.array): Fitness of the population.
        """
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 1, 1, 0])

        fitness_population = np.zeros([len(population), 1])

        for idx, individual in enumerate(population):
            fitness = 0
            for x, output in zip(inputs, outputs):
                fitness += abs(output - self.forward_pass(x, individual))
            fitness_population[idx] = np.exp(-fitness)

        return fitness_population.flatten()

    @staticmethod
    def choose_random_genes(individual):
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

    @staticmethod
    def recombination_arithmetic(individual1, individual2, alpha, gene1=None, gene2=None):
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
            gene1, gene2 = GeneticAlgorithm.choose_random_genes(individual1)
        new_individual1[gene1:gene2 + 1] = recombination_values[gene1:gene2 + 1]
        new_individual2[gene1:gene2 + 1] = recombination_values[gene1:gene2 + 1]

        return (new_individual1, new_individual2)

    @staticmethod
    def mutation_uniform(individual, upper_bound, lower_bound, gene1=None):
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

    @staticmethod
    def selection_genitor(fitness_population):
        """
        Selects population using the genitor method.

        Args:
            population (np.array): Population containg the different individuals.
            fitness_population (np.array): Fitness of the population.
        """
        survivors = np.argsort(fitness_population)
        return survivors[::-1]

    def generate_next_population(self, population, mutation, recombination, selection):
        """
        Generates the population for the next iteration.

        Args:
            population (np.array): Population containg the different individuals.

        Returns:
            next_population, fitness_population (tuple): Returns tuple containing the next population and its fitness
        """
        # Initialize new offspring
        offspring = (self.upper_bound - self.lower_bound) * np.random.random((self.offspring_size, 9)) + self.lower_bound

        # Recombinate best individuals
        for individual in range(0, self.offspring_size, 2):
            idx_parent1, idx_parent2 = np.random.choice(self.population_size, size=2, replace=False)
            new_individual1, new_individual2 = recombination(population[idx_parent1], population[idx_parent2], self.alpha)
            offspring[individual] = new_individual1
            offspring[individual + 1] = new_individual2

        # Add mutation
        for idx in range(len(population)):
            if np.random.random() < self.mutation_rate:
                individual_mutated = mutation(population[idx], self.upper_bound, self.lower_bound)
                population[idx] = individual_mutated

        # Group populations
        temporal_population = np.vstack((population, offspring))
        fitness_population = self.compute_fitness(temporal_population)

        # Select next generation with probability fitness / total_fitness
        survivors = selection(fitness_population)
        survivors = survivors[:self.population_size]

        return (temporal_population[survivors], fitness_population[survivors])

    def run(self):
        """
        Runs the algorithm.

        Returns:
            solutions, max_fitness, mean_fitness (tuple): Returns tuple containing the solutions the fitness mean and max along the iterations
        """
        # Initialize first population
        population = self.random_initial_population()

        # Initialize fitness variables
        mean_fitness = []
        max_fitness = []
        diversity_genotype = []
        diversity_phenotype = []

        # Initialize best_fitness
        best_fitness_all = 0

        # Choose mutation
        if self.mutation_type == "uniform":
            mutation = self.mutation_uniform
        else:
            raise MutationTypeError(self.mutation_type)

        # Choose recombination
        if self.recombination_type == "arithmetic":
            recombination = self.recombination_arithmetic
        else:
            raise RecombinationTypeError(self.recombination_type)

        # Choose selection
        if self.selection_type == "genitor":
            selection = self.selection_genitor
        else:
            raise SelectionTypeError(self.selection_type)

        # Iterate through generations
        for iteration in tqdm(range(self.num_iterations), ncols=75):
            population, fitness = self.generate_next_population(population, mutation, recombination, selection)

            # Save statistics iteration
            best_fitness_iteration = np.max(fitness)
            mean_fitness_iteration = np.mean(fitness)
            diversity_genotype_iteration = np.unique(population, axis=0).shape[0]
            diversity_phenotype_iteration = np.unique(fitness).shape[0]

            max_fitness.append(best_fitness_iteration)
            mean_fitness.append(mean_fitness_iteration)
            diversity_genotype.append(diversity_genotype_iteration)
            diversity_phenotype.append(diversity_phenotype_iteration)

            # Keep best individuals
            if best_fitness_iteration > best_fitness_all:
                solutions = []
                for best_individual in population[np.where(fitness == best_fitness_iteration)]:
                    if not any((best_individual == individual).all() for individual in solutions):
                        solutions.append(best_individual)
                best_fitness_all = best_fitness_iteration
            elif best_fitness_iteration == best_fitness_all:
                for best_individual in population[np.where(fitness == best_fitness_iteration)]:
                    if not any((best_individual == individual).all() for individual in solutions):
                        solutions.append(best_individual)

        return (np.asarray(solutions), max_fitness, mean_fitness, diversity_genotype, diversity_phenotype)
