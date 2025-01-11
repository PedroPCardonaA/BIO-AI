use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use plotters::prelude::*;

fn main() {
    let population_size = 100;
    let features_count = 1000;
    let lower_bound = 900;
    let mutation_rate = 0.01;
    let tournament_size = 10;
    let max_generations = 1000;
    let mut fitness_landscapes = HashMap::new();
    let mut fiteness_history = Vec::new();
    let mut divesity_history = Vec::new();

    let mut population = generate_population(population_size, features_count);
    let mut fitnest_value = *evaluate_population(&population)
        .par_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let mut generation = 0;

    for individual in population.iter() {
        let fitness = fitness_function(individual);
        fitness_landscapes.insert(individual.clone(), fitness);
    }

    divesity_history.push(diversity_function(&population));

    while generation <= max_generations && fitnest_value < lower_bound as f64 {
        let selected_population = tournament_selection(&population, tournament_size);
        let new_population: Vec<Vec<bool>> = (0..population_size)
            .into_par_iter()
            .map(|i| {
                let parent1 = &selected_population[i % selected_population.len()];
                let parent2 = &selected_population[(i + 1) % selected_population.len()];
                let (child1, _) = uniform_crossover(parent1, parent2);
                mutate(&child1, mutation_rate)
            })
            .collect();


        population = new_population;

        for individual in population.iter() {
            let fitness = fitness_function(individual);
            fitness_landscapes.insert(individual.clone(), fitness);
        }

        generation += 1;

        if generation % 50 == 0 || generation == max_generations{
            fitnest_value = *evaluate_population(&population)
            .par_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
            println!("Generation: {}, Fitness: {}", generation, fitnest_value);
            fiteness_history.push(fitnest_value);
            divesity_history.push(diversity_function(&population));
            println!("Population: {:?}", population.len());
        }
        
    }

    let best_individual = population
        .par_iter()
        .max_by(|a, b| fitness_function(a).partial_cmp(&fitness_function(b)).unwrap())
        .unwrap();
    println!("Best individual: {:?}", best_individual);
    println!("Fitness: {}", fitness_function(best_individual));

    fitnest_value = *evaluate_population(&population)
            .par_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
    println!("Generation: {}, Fitness: {}", generation, fitnest_value);
    fiteness_history.push(fitnest_value);
    divesity_history.push(diversity_function(&population));
    println!("Population: {:?}", population.len());

    plot_graph(&fiteness_history, "fitness_history.png", "Fitnes Over Generation").unwrap();
    plot_graph(&divesity_history, "diversity_history.png", "Diversity over generation").unwrap();

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
    let feature_sum:f64 = individual.iter().map(|&bit| if bit { 1.0 } else { 0.0 }).sum();
    feature_sum
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

/// Applies uniform crossover to two parents to generate two children.
/// The crossover is performed by randomly selecting bits from the two parents with equal probability.
/// 
/// # Parameters
/// - `parent1`: A reference to the first parent.
/// - `parent2`: A reference to the second parent.
/// 
/// # Returns
/// A tuple containing the two children, each represented as a vector of boolean values.
fn uniform_crossover(parent1: &Vec<bool>, parent2: &Vec<bool>) -> (Vec<bool>, Vec<bool>) {
    let mut rng = rand::thread_rng();
    let child1: Vec<bool> = parent1.iter().zip(parent2.iter()).map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b }).collect();
    let child2: Vec<bool> = parent1.iter().zip(parent2.iter()).map(|(&a, &b)| if rng.gen_bool(0.5) { b } else { a }).collect();
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

/// Plots the fitness history of the genetic algorithm.
/// 
/// # Parameters
/// - `fitness_history`: A reference to a vector containing the best fitness value for each generation.
/// 
/// # Returns
/// A `Result` containing `Ok` if the plot was successfully generated, or an `Err` if an error occurred.
fn plot_graph(
    fitness_history: &Vec<f64>, 
    file_name: &str, 
    title: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    if fitness_history.is_empty() {
        return Err("Fitness history is empty, cannot generate plot.".into());
    }

    let max_fitness = *fitness_history
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let generations = fitness_history.len();

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..generations, 0.0..max_fitness)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            fitness_history.iter().enumerate().map(|(x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Best Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).draw()?;

    Ok(())
}


/// Calculates the diversity of a population.
///     
/// # Parameters
/// - `population`: A reference to the population, where each individual is a vector of boolean values.
/// 
/// # Returns
/// The diversity of the population, calculated as the sum of the Hamming distances between all pairs of individuals.
fn diversity_function(population: &Vec<Vec<bool>>) -> f64 {
    let mut diversity = 0.0;
    for i in 0..population.len() {
        for j in i + 1..population.len() {
            let distance = hamming_distance(&population[i], &population[j]);
            diversity += distance;
        }
    }
    diversity
}

/// Calculates the Hamming distance between two individuals.
/// 
/// # Parameters
/// - `individual1`: A reference to the first individual.
/// - `individual2`: A reference to the second individual.
/// 
/// # Returns
/// The Hamming distance between the two individuals, calculated as the number of differing bits.
fn hamming_distance(individual1: &Vec<bool>, individual2: &Vec<bool>) -> f64 {
    individual1
        .iter()
        .zip(individual2.iter())
        .map(|(&a, &b)| if a == b { 0.0 } else { 1.0 })
        .sum()
}
 
