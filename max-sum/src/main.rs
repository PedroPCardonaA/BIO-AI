use rand::Rng;
use rayon::prelude::*;

fn main() {
    let population_size = 1_000;
    let features_count = 100;
    let lower_bound = 90;
    let mutation_rate = 0.01;
    let tournament_size = 100;
    let max_generations = 1000;

    let mut population = generate_population(population_size, features_count);
    let mut fitnest_value = *evaluate_population(&population)
        .par_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let mut generation = 0;

    while generation <= max_generations && fitnest_value < lower_bound as f64 {
        let selected_population = tournament_selection(&population, tournament_size);
        let new_population: Vec<Vec<bool>> = (0..selected_population.len())
            .into_par_iter()
            .flat_map(|i| {
                let parent1 = &selected_population[i];
                let parent2 = &selected_population[(i + 1) % selected_population.len()];
                let (child1, child2) = single_point_crossover(parent1, parent2);
                vec![
                    mutate(&child1, mutation_rate),
                    mutate(&child2, mutation_rate),
                ]
            })
            .collect();

        population = new_population;
        fitnest_value = *evaluate_population(&population)
            .par_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        generation += 1;

        println!("Generation: {}, Fitness: {}", generation, fitnest_value);
    }

    let best_individual = population
        .par_iter()
        .max_by(|a, b| fitness_function(a).partial_cmp(&fitness_function(b)).unwrap())
        .unwrap();
    println!("Best individual: {:?}", best_individual);
    println!("Fitness: {}", fitness_function(best_individual));
}

/// Generates an initial population for the genetic algorithm.
/// 
/// # Parameters
/// - `size`: The number of individuals in the population.
/// - `features_count`: The number of features (genes) for each individual.
/// 
/// # Returns
/// A vector containing the generated population, where each individual is a vector of boolean values.
fn generate_population(size: usize, features_count: usize) -> Vec<Vec<bool>> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| (0..features_count).map(|_| rng.gen_bool(0.5)).collect())
        .collect()
}

/// Calculates the fitness of an individual.
/// 
/// # Parameters
/// - `individual`: A reference to an individual represented as a vector of boolean values.
/// 
/// # Returns
/// The fitness value of the individual, calculated as the sum of its `true` values.
fn fitness_function(individual: &Vec<bool>) -> f64 {
    individual.iter().map(|&bit| if bit { 1.0 } else { 0.0 }).sum()
}

/// Evaluates the fitness of an entire population in parallel.
/// 
/// # Parameters
/// - `population`: A reference to the population, where each individual is a vector of boolean values.
/// 
/// # Returns
/// A vector of fitness values, one for each individual in the population.
fn evaluate_population(population: &Vec<Vec<bool>>) -> Vec<f64> {
    population
        .par_iter()
        .map(|individual| fitness_function(individual))
        .collect()
}

/// Performs tournament selection to choose individuals for the next generation.
/// 
/// # Parameters
/// - `population`: A reference to the current population.
/// - `tournament_size`: The number of individuals to participate in each tournament.
/// 
/// # Returns
/// A vector of selected individuals for the next generation.
fn tournament_selection(population: &Vec<Vec<bool>>, tournament_size: usize) -> Vec<Vec<bool>> {
    let population_arc = std::sync::Arc::new(population.clone());
    (0..population.len())
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut tournament = Vec::new();
            for _ in 0..tournament_size {
                let idx = rng.gen_range(0..population_arc.len());
                tournament.push(population_arc[idx].clone());
            }
            tournament
                .iter()
                .max_by(|a, b| fitness_function(a).partial_cmp(&fitness_function(b)).unwrap())
                .unwrap()
                .clone()
        })
        .collect()
}

/// Performs single-point crossover to generate two offspring from two parents.
/// 
/// # Parameters
/// - `parent1`: A reference to the first parent.
/// - `parent2`: A reference to the second parent.
/// 
/// # Returns
/// A tuple containing two offspring, each represented as a vector of boolean values.
fn single_point_crossover(parent1: &Vec<bool>, parent2: &Vec<bool>) -> (Vec<bool>, Vec<bool>) {
    let mut rng = rand::thread_rng();
    let crossover_point = rng.gen_range(0..parent1.len());
    let child1 = parent1[..crossover_point]
        .iter()
        .chain(parent2[crossover_point..].iter())
        .cloned()
        .collect();
    let child2 = parent2[..crossover_point]
        .iter()
        .chain(parent1[crossover_point..].iter())
        .cloned()
        .collect();
    (child1, child2)
}

/// Applies mutation to an individual by flipping its bits with a given probability.
/// 
/// # Parameters
/// - `individual`: A reference to the individual to mutate.
/// - `mutation_rate`: The probability of flipping each bit.
/// 
/// # Returns
/// A mutated individual, represented as a vector of boolean values.
fn mutate(individual: &Vec<bool>, mutation_rate: f64) -> Vec<bool> {
    let mut rng = rand::thread_rng();
    individual
        .iter()
        .map(|&gene| {
            if rng.gen::<f64>() < mutation_rate {
                !gene
            } else {
                gene
            }
        })
        .collect()
}
