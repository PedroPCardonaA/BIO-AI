import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from LinReg import LinReg
import matplotlib.pyplot as plt

class EA(ABC):
    """
    A base class for Evolutionary Algorithms. Defines essential
    methods for population initialization, selection, crossover,
    mutation, and elitism.
    """

    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate, max_generations):
        """
        Initialize the EA with the main hyperparameters and create an initial population.
        
        Args:
            population_size (int): Number of individuals in the population.
            chromosome_length (int): Size of each chromosome.
            mutation_rate (float): Probability of flipping a bit in mutation.
            crossover_rate (float): Probability of applying crossover.
            max_generations (int): Maximum number of generations to evolve.
        """
        super().__init__()
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.entropies = []
        self.fitnesses = []

        self.population = np.random.choice([0, 1], (population_size, chromosome_length))
        self.fitness = np.zeros(population_size)

    def _tournament_selection(self, tournament_size):
        """
        Select a parent by picking a subset of the population and returning
        the index of the fittest individual in that subset.
        """
        tournament_indices = np.random.choice(self.population_size, tournament_size)
        tournament_fitness = self.fitness[tournament_indices]
        return tournament_indices[np.argmax(tournament_fitness)]
    
    def _roulette_wheel_selection(self):
        """
        Select a parent based on fitness proportion. The higher the fitness,
        the higher the chance of being selected.
        """
        total_fitness = np.sum(self.fitness)
        selection_probabilities = self.fitness / total_fitness
        return np.random.choice(self.population_size, p=selection_probabilities)
    
    def _select_parents(self, selection_method=1, tournament_size=3):
        """
        Select two parents using either tournament selection or roulette wheel selection.
        
        Args:
            selection_method (int): 1 for tournament, otherwise roulette.
            tournament_size (int): Number of individuals for tournament selection.
            
        Returns:
            tuple: Indices of the two selected parents.
        """
        if selection_method == 1:
            parent1 = self._tournament_selection(tournament_size)
            parent2 = self._tournament_selection(tournament_size)
        else:
            parent1 = self._roulette_wheel_selection()
            parent2 = self._roulette_wheel_selection()
        return parent1, parent2
    
    def _single_point_crossover(self, parent1, parent2, crossover_posibility=0.7):
        """
        Perform single-point crossover at a random cut point in the chromosome.
        
        Args:
            parent1 (ndarray): First parent chromosome.
            parent2 (ndarray): Second parent chromosome.
            crossover_posibility (float): Chance to perform crossover.
            
        Returns:
            tuple: Two resulting offspring chromosomes.
        """
        if np.random.random() < crossover_posibility:
            crossover_point = np.random.randint(self.chromosome_length)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        return child1, child2
    
    def _uniform_crossover(self, parent1, parent2):
        """
        Perform uniform crossover, swapping bits according to a random mask.
        
        Args:
            parent1 (ndarray): First parent chromosome.
            parent2 (ndarray): Second parent chromosome.
            
        Returns:
            tuple: Two resulting offspring chromosomes.
        """
        if np.random.random() < self.crossover_rate:
            mask = np.random.choice([0, 1], self.chromosome_length)
            child1 = parent1 * mask + parent2 * (1 - mask)
            child2 = parent2 * mask + parent1 * (1 - mask)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        return child1, child2

    def _bit_flip_mutation(self, chromosome):
        """
        Mutate a chromosome by flipping bits with probability self.mutation_rate.
        
        Args:
            chromosome (ndarray): Chromosome to mutate.
            
        Returns:
            ndarray: Mutated chromosome.
        """
        mask = np.random.choice(
            [0, 1],
            self.chromosome_length,
            p=[1 - self.mutation_rate, self.mutation_rate]
        )
        for i in range(len(chromosome)):
            if mask[i] == 1:
                chromosome[i] = 1 - chromosome[i]  # flip 0 <-> 1
        return chromosome

    def _elitisim_selection(self, child_population, child_fitness):
        """
        Preserve the best individual from the old population if it
        is better than the best in the new population.
        """
        elite_index = np.argmax(self.fitness)
        elite_child_index = np.argmax(child_fitness)
        if child_fitness[elite_child_index] > self.fitness[elite_index]:
            self.population[elite_index] = child_population[elite_child_index]
            self.fitness[elite_index] = child_fitness[elite_child_index]

    def _entropy(self):
        """
        Calculate the sum of bit-level entropies across all genes in the population.
        """
        num_ones_per_gene = np.sum(self.population, axis=0)
        p = num_ones_per_gene / self.population_size
        p = np.clip(p, 1e-10, 1 - 1e-10)
        entropy_per_gene = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        total_entropy = np.sum(entropy_per_gene)

        return total_entropy


    
    def _plot_entropy(self):
        """
        Plot the entropy of the population over time.
        """
        plt.plot(self.entropies)
        plt.xlabel("Generation")
        plt.ylabel("Entropy")
        plt.savefig("entropy.png")

    def _plot_fitness(self):
        """
        Plot the fitness of the population over time.
        """
        if not self.fitnesses:
            print("No fitness data to plot.")
            return

        fitnesses_array = np.array(self.fitnesses)
        generations = fitnesses_array[:, 0]
        max_fitness = fitnesses_array[:, 1]
        avg_fitness = fitnesses_array[:, 2]
        min_fitness = fitnesses_array[:, 3]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, max_fitness, label='Max Fitness', linestyle='-', marker='o')
        plt.plot(generations, avg_fitness, label='Average Fitness', linestyle='--', marker='x')
        plt.plot(generations, min_fitness, label='Min Fitness', linestyle=':', marker='s')

        plt.title('Fitness Over Generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend()

        plt.grid(True)

        plt.savefig("fitness.png")





    def _crowding_replacement(self, 
                            parent1_index, parent2_index,
                            parent1, parent2,
                            child1, child2):
        """
        Perform crowding replacement to maintain diversity in the population.
        
        Args:
            parent1_index (int): Index of the first parent.
            parent2_index (int): Index of the second parent.
            parent1 (ndarray): First parent chromosome.
            parent2 (ndarray): Second parent chromosome.
            child1 (ndarray): First child chromosome.
            child2 (ndarray): Second child chromosome.
        """
        def hamming_distance(chromosome1, chromosome2):
            return np.sum(chromosome1 != chromosome2)
        
        dist_child1_parent1 = hamming_distance(child1, parent1)
        dist_child1_parent2 = hamming_distance(child1, parent2)
        dist_child2_parent1 = hamming_distance(child2, parent1)
        dist_child2_parent2 = hamming_distance(child2, parent2)

        # For child1
        if dist_child1_parent1 <= dist_child1_parent2:
            self.population[parent1_index] = child1
            self.fitness[parent1_index] = self._fitness_function(child1)
        else:
            self.population[parent2_index] = child1
            self.fitness[parent2_index] = self._fitness_function(child1)

        # For child2
        if dist_child2_parent1 <= dist_child2_parent2:
            self.population[parent1_index] = child2
            self.fitness[parent1_index] = self._fitness_function(child2)
        else:
            self.population[parent2_index] = child2
            self.fitness[parent2_index] = self._fitness_function(child2)



    @abstractmethod
    def _fitness_function(self, chromosome, **kwargs):
        """
        Calculate the fitness of a given chromosome. Must be implemented by subclasses.
        """
        pass
    
    def run(self, selection_method=1, tournament_size=3, crossover_method=1, crowding=False):
        """
        Main loop for running the EA. Repeats selection, crossover, mutation,
        and elitism over self.max_generations generations.
        
        Args:
            selection_method (int): 1 for tournament, otherwise roulette selection.
            tournament_size (int): Number of individuals per tournament.
            crossover_method (int): 1 for single-point, otherwise uniform crossover.
            
        Returns:
            ndarray: The best chromosome found after evolution.
        """
        for generation in range(self.max_generations):
            child_population = np.zeros((self.population_size, self.chromosome_length))
            child_fitness = np.zeros(self.population_size)

            for i in range(0, self.population_size, 2):
                parent1_index, parent2_index = self._select_parents(selection_method, tournament_size)
                parent1 = self.population[parent1_index]
                parent2 = self.population[parent2_index]

                if crossover_method == 1:
                    child1, child2 = self._single_point_crossover(parent1, parent2)
                else:
                    child1, child2 = self._uniform_crossover(parent1, parent2)

                child1 = self._bit_flip_mutation(child1)
                child2 = self._bit_flip_mutation(child2)

                child_population[i] = child1
                child_population[i + 1] = child2

                child_fitness[i] = self._fitness_function(child1)
                child_fitness[i + 1] = self._fitness_function(child2)
                if crowding:
                    self._crowding_replacement(parent1_index, parent2_index, 
                                            parent1, parent2,
                                            child1, child2)



            self._elitisim_selection(child_population, child_fitness)
            self.population = child_population
            self.fitness = child_fitness

            self.entropies.append(self._entropy())
            print(f"Generation {generation + 1}: {np.max(self.fitness)} | Entropy: {self.entropies[-1]} | Population size: {self.population_size}")
            self.fitnesses.append([
                generation, 
                float(np.max(self.fitness)), 
                float(np.mean(self.fitness)), 
                float(np.min(self.fitness))
            ])
        
        self._plot_entropy()
        self._plot_fitness()
        return self.population[np.argmax(self.fitness)]
    
    

class KnapsackProblem(EA):
    """
    A concrete EA subclass for the 0-1 Knapsack problem.
    """

    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate,
                 max_generations, weights, values, capacity):
        """
        Initialize the KnapsackProblem with data-specific parameters.
        
        Args:
            population_size (int): Size of the population.
            chromosome_length (int): Number of items in the knapsack.
            mutation_rate (float): Probability of mutation.
            crossover_rate (float): Probability of crossover.
            max_generations (int): Max number of iterations.
            weights (array-like): Weights of the items.
            values (array-like): Values of the items.
            capacity (int): Maximum weight capacity of the knapsack.
        """
        super().__init__(population_size, chromosome_length, mutation_rate, crossover_rate, max_generations)
        self.weights = weights
        self.values = values
        self.capacity = capacity

    def _fitness_function(self, chromosome):
        """
        Compute total value of the chosen items without exceeding capacity.
        """
        total_weight = np.sum(chromosome * self.weights)
        total_value = np.sum(chromosome * self.values)
        if total_weight > self.capacity:
            return 0
        return total_value
        


class FeatureSelection(EA):
    """
    A concrete EA subclass for feature selection tasks.
    Uses a LinReg model to evaluate chosen features.
    """
        
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate,
                 max_generations, X, y):
        """
        Initialize feature selection with dataset and labels.
        
        Args:
            population_size (int): Size of the population.
            chromosome_length (int): Number of features in the dataset.
            mutation_rate (float): Probability of mutation.
            crossover_rate (float): Probability of crossover.
            max_generations (int): Max number of generations to evolve.
            X (ndarray): Input features.
            y (ndarray): Target values/labels.
        """
        super().__init__(population_size, chromosome_length, mutation_rate, crossover_rate, max_generations)
        self.X = X
        self.y = y
        self.linreg = LinReg()

    def _fitness_function(self, chromosome):
        """
        Compute fitness by training a linear model on the selected features 
        and measuring performance (lower is better, so fitness is negative).
        """
        selected_columns = np.where(chromosome == 1)[0]
        if len(selected_columns) == 0:
            return 0
        return -self.linreg.get_fitness(self.X[:, selected_columns], self.y, rng=42)


def knapsack_run(crowding=False):
    """Run a Knapsack EA example using a CSV dataset."""
    df = pd.read_csv("data/knapPI_12_500_1000_82.csv")
    weights = df["w"].values
    values = df["p"].values

    kp = KnapsackProblem(100, len(weights), 1/len(weights), 0.7, 100, weights, values, 280785)
    kp.run(crowding=crowding)


def feature_selection_run(crowding=False):
    """Run a feature selection EA example using a text-based dataset."""
    df = pd.read_csv("data/dataset.txt", sep=",", header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    fs = FeatureSelection(100, X.shape[1], 1/X.shape[1], 0.7, 100, X, y)
    fs = fs.run(crowding=crowding)


if __name__ == "__main__":
    feature_selection_run(crowding=False)
    