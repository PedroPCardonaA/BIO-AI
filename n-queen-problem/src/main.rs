use rand::Rng;

fn main() {
    const POPULATION_SIZE: usize = 100;
    const FEATURES_COUNT: usize = 10;
    let mutation_rate = 0.15;
    let tournament_size = 10;
    let max_generations = 1000;
    let mut fitness_history = Vec::new();
    
    let mut population = generate_population(POPULATION_SIZE, FEATURES_COUNT);
    let mut generation = 0;
    while generation <= max_generations && fittest_value(&population, FEATURES_COUNT) < FEATURES_COUNT * (FEATURES_COUNT - 1) / 2 {
        let selected_population = tournament_selection(&population, tournament_size, FEATURES_COUNT);
        let new_population: Vec<Vec<usize>> = (0..POPULATION_SIZE / 2)
            .flat_map(|i| {
                let parent1 = &selected_population[i % selected_population.len()];
                let parent2 = &selected_population[(i + 1) % selected_population.len()];
                let (child1, child2) = uniform_crossover(parent1, parent2);
                vec![
                    random_resetting_mutation(&swap_mutation(&child1, mutation_rate), mutation_rate),
                    random_resetting_mutation(&swap_mutation(&child2, mutation_rate), mutation_rate),
                ]
            })
            .collect();

        
        let elitism_count = (0.05 * POPULATION_SIZE as f64) as usize;
        let mut sorted_population: Vec<Vec<usize>> = population.clone();
        sorted_population.sort_by(|a, b| fitness_function(b, FEATURES_COUNT).cmp(&fitness_function(a, FEATURES_COUNT)));
        let elite_individuals: Vec<Vec<usize>> = sorted_population.into_iter().take(elitism_count).collect();
        
        population = new_population;
        population.extend(elite_individuals);
            
        let fittest = fittest_value(&population, FEATURES_COUNT);
        fitness_history.push(fittest);

        if generation % 50 == 0 {
            println!("Generation: {}, Fittest: {}", generation, fittest_value(&population, FEATURES_COUNT));
        }

        generation += 1;
    }

    println!("Fitness: {}", fittest_value(&population, FEATURES_COUNT));
    print_board(&fittest_individual(&population, FEATURES_COUNT));
}

/// Generates a population of individuals.
/// 
/// # Parameters
/// - `population_size`: The number of individuals in the population.
/// - `features_count`: The number of features in each individual.
/// 
/// # Returns
/// A vector of vectors, where each inner vector represents an individual and contains the features of that individual.
fn generate_population(population_size: usize, features_count: usize) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    (0..population_size)
        .map(|_| {
            (0..features_count)
                .map(|_| rng.gen_range(0..features_count))
                .collect()
        })
        .collect()
}

/// The fitness function for the N-Queens problem.
/// 
/// # Parameters
/// - `individual`: The individual to evaluate.
/// - `features_count`: The number of features in each individual.
/// 
/// # Returns
/// The fitness of the individual.
fn fitness_function(individual: &Vec<usize>, features_count: usize) -> usize {
    let mut conflicts = 0;

    for i in 0..individual.len() {
        for j in i + 1..individual.len() {
            if individual[i] == individual[j]
                || (individual[i] as isize - individual[j] as isize).abs() == (j as isize - i as isize).abs()
            {
                conflicts += 1;
            }
        }
    }

    let max_non_conflicting = features_count * (features_count - 1) / 2;
    max_non_conflicting - conflicts
}

/// Returns the fittest individual in a population.
/// 
/// # Parameters
/// - `population`: A reference to the population.
/// - `features_count`: The number of features in each individual.
/// 
/// # Returns
/// The fittest individual in the population.
fn fittest_individual(population: &Vec<Vec<usize>>, features_count: usize) -> Vec<usize> {
    population
        .iter()
        .max_by(|a, b| fitness_function(a, features_count).partial_cmp(&fitness_function(b, features_count)).unwrap())
        .unwrap()
        .clone()
}

/// Returns the fittest value in a population.
/// 
/// # Parameters
/// - `population`: A reference to the population.
/// - `features_count`: The number of features in each individual.
/// 
/// # Returns
/// The fittest value in the population.
fn fittest_value(population: &Vec<Vec<usize>>, features_count: usize) -> usize {
    *evaluate_population(population, features_count)
        .iter()
        .max()
        .unwrap()
}

/// Evaluates the fitness of each individual in a population.
/// 
/// # Parameters
/// - `population`: A reference to the population.
/// - `features_count`: The number of features in each individual.
/// 
/// # Returns
/// A vector containing the fitness of each individual in the population.
fn evaluate_population(population: &Vec<Vec<usize>>, features_count: usize) -> Vec<usize> {
    population
        .iter()
        .map(|individual| fitness_function(individual, features_count))
        .collect()
}

/// Selects individuals from a population using tournament selection.
/// 
/// # Parameters
/// - `population`: A reference to the population.
/// - `tournament_size`: The number of individuals to participate in each tournament.
/// - `features_count`: The number of features in each individual.
/// 
/// # Returns
/// A vector of selected individuals for the next generation.
fn tournament_selection(
    population: &Vec<Vec<usize>>, 
    tournament_size: usize, 
    features_count: usize
) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    (0..population.len())
        .map(|_| {
            let mut tournament: Vec<&Vec<usize>> = (0..tournament_size)
                .map(|_| &population[rng.gen_range(0..population.len())])
                .collect();
            tournament.sort_by(|a, b| fitness_function(b, features_count).cmp(&fitness_function(a, features_count)));
            if rng.gen_bool(0.7) {
                tournament[0].clone()
            } else { 
                tournament[rng.gen_range(1..tournament.len())].clone()
            }
        })
        .collect()
}


/// Performs uniform crossover on two parents to produce two children.
/// 
/// # Parameters
/// - `parent1`: The first parent.
/// - `parent2`: The second parent.
/// 
/// # Returns
/// A tuple containing the two children.
/// The first element is the first child, and the second element is the second child.
fn uniform_crossover(parent1: &Vec<usize>, parent2: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut rng = rand::thread_rng();
    let mut child1 = Vec::new();
    let mut child2 = Vec::new();
    for i in 0..parent1.len() {
        if rng.gen_bool(0.5) {
            child1.push(parent1[i]);
            child2.push(parent2[i]);
        } else {
            child1.push(parent2[i]);
            child2.push(parent1[i]);
        }
    }
    (child1, child2)
}

/// Performs swap mutation on an individual.
/// 
/// # Parameters
/// - `individual`: The individual to mutate.
/// - `mutation_rate`: The probability that a gene will be mutated.
/// 
/// # Returns
/// The mutated individual.
fn swap_mutation(individual: &Vec<usize>, mutation_rate: f64) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut mutated_individual = individual.clone();
    for i in 0..mutated_individual.len() {
        if rng.gen_bool(mutation_rate) {
            let mutation_point = rng.gen_range(0..mutated_individual.len());
            mutated_individual.swap(i, mutation_point);
        }
    }
    mutated_individual
}

/// Performs random resetting mutation on an individual.
/// 
/// # Parameters
/// - `individual`: The individual to mutate.
/// - `mutation_rate`: The probability that a gene will be mutated.
/// 
/// # Returns
/// The mutated individual.
fn random_resetting_mutation(individual: &Vec<usize>, mutation_rate: f64) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut mutated_individual = individual.clone();
    for i in 0..mutated_individual.len() {
        if rng.gen_bool(mutation_rate) {
            mutated_individual[i] = rng.gen_range(0..mutated_individual.len());
        }
    }
    mutated_individual
}

/// Prints the board representation of an individual.
/// 
/// # Parameters
/// - `board`: A reference to the individual to print.
/// 
/// # Returns
/// The board representation of the individual.
fn print_board(board: &Vec<usize>) {
    for i in 0..board.len() {
        for j in 0..board.len() {
            if board[j] == i {
                print!("Q ");
            } else {
                print!("_ ");
            }
        }
        println!();
    }
    println!();
}


        