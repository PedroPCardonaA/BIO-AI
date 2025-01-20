import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from itertools import combinations
import random

# Function to load dataset
def load_dataset(path):
    data = pd.read_csv(path, header=None)
    assert data.shape[1] == 102, "Unexpected number of columns in record"

    # Split features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Function to generate initial population
def generate_population(population_size, features_count):
    return [np.random.choice([True, False], size=features_count).tolist() for _ in range(population_size)]

# Tournament selection
def tournament_selection(population, fitness_values, tournament_size):
    selected = []
    for _ in range(len(population)):
        candidates = random.sample(range(len(population)), tournament_size)
        best_candidate = min(candidates, key=lambda idx: fitness_values[idx])
        selected.append(population[best_candidate])
    return selected

# Uniform crossover
def uniform_crossover(parent1, parent2):
    offspring1, offspring2 = [], []
    for p1, p2 in zip(parent1, parent2):
        if random.random() < 0.5:
            offspring1.append(p1)
            offspring2.append(p2)
        else:
            offspring1.append(p2)
            offspring2.append(p1)
    return offspring1, offspring2


# Flip bit mutation
def flip_bit_mutation(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = not solution[i]

# Elitism
def elitism(population, fitness_values, elitism_count):
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0])]
    return sorted_population[:elitism_count]

# Fitness function
def fitness_function(solution, X, y, cache):
    key = ''.join(['1' if bit else '0' for bit in solution])

    if key in cache:
        return cache[key]

    selected_indices = [idx for idx, bit in enumerate(solution) if bit]
    if not selected_indices:
        cache[key] = float('inf')
        return float('inf')

    X_selected = X[:, selected_indices]
    model = LinearRegression()
    model.fit(X_selected, y)
    y_pred = model.predict(X_selected)
    rmse = root_mean_squared_error(y, y_pred)

    cache[key] = rmse
    return rmse

def rmse_with_all_features(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return root_mean_squared_error(y, y_pred)

# Genetic Algorithm
def genetic_algorithm(X, y, generations, population_size, mutation_rate):
    features_count = X.shape[1]
    population = generate_population(population_size, features_count)
    cache = {}
    all_features_rmse = rmse_with_all_features(X, y)

    for generation in range(generations):
        fitnesses = [fitness_function(sol, X, y, cache) for sol in population]
        min_fitness = min(fitnesses)
        
        if generation % 10 == 0:
            print(f"Generation: {generation}, Min Fitness: {min_fitness:.6f}, All Features RMSE: {all_features_rmse:.6f}")
        
        parents = tournament_selection(population, fitnesses, tournament_size=2)
        offspring = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = uniform_crossover(parents[i], parents[i + 1])
                offspring.append(child1)
                offspring.append(child2)

        for child in offspring:
            flip_bit_mutation(child, mutation_rate)

        elite = elitism(population, fitnesses, population_size // 2)
        population = elite + offspring[:population_size - len(elite)]

# Main function
if __name__ == "__main__":
    X, y = load_dataset("data/dataset.txt")
    genetic_algorithm(X, y, generations=1000, population_size=100, mutation_rate=1/10)
