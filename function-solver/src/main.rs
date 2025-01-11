use rand::Rng;
use rayon::prelude::*;
use plotters::prelude::*;

fn main(){
    let population_size = 500;
    let features_count = 3;
    let mutation_rate = 0.01;
    let tournament_size = 10;
    let max_generations = 1000;
    let mut fitness_history = Vec::new();
    let mut diversity_history = Vec::new();

    let mut population = generate_population(population_size, features_count);
    let mut generation = 0;
    diversity_history.push(diversity_function(&population));
    while generation <= max_generations {
        let selected_population = tournament_selection(&population, tournament_size);
        let new_population: Vec<Vec<f64>> = (0..population_size / 2)
            .into_par_iter()
            .flat_map(|i| {
                let parent1 = &selected_population[i % selected_population.len()];
                let parent2 = &selected_population[(i + 1) % selected_population.len()];
                let (child1, child2) = arithmetic_crossover(parent1, parent2);
                vec![mutate(&child1, mutation_rate), mutate(&child2, mutation_rate)]
            })
            .collect();

        let elitism_count = (0.05 * population_size as f64) as usize;
        let mut sorted_population: Vec<Vec<f64>> = population.clone();
        sorted_population.sort_by(|a, b| fitness_function(b).partial_cmp(&fitness_function(a)).unwrap());
        let elite_individuals: Vec<Vec<f64>> = sorted_population.into_iter().take(elitism_count).collect();
        
        population = new_population;
        population.splice(0..elitism_count, elite_individuals);
        let fittest = fittest_value(&population);
        fitness_history.push(fittest);
        if generation % 50 == 0 {
            println!("Generation: {}, Fittest: {}", generation, fittest);
            diversity_history.push(diversity_function(&population));
        }
        generation += 1;
    }
    

    plot_graph(&fitness_history, "fitness_history.png", "Fitness History").unwrap();
    plot_graph(&diversity_history, "diversity_history.png", "Diversity History").unwrap();

    println!("Fitness: {}", fittest_value(&population));

}

/// Generates a population of individuals.
/// 
/// # Parameters
/// - `population_size`: The number of individuals in the population.
/// - `features_count`: The number of features in each individual.
/// 
/// # Returns
/// A vector of vectors, where each inner vector represents an individual and contains the features of that individual.
fn generate_population(population_size: usize, features_count: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..population_size)
        .map(|_| {
            (0..features_count)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect()
        })
        .collect()
}

/// Evaluates the fitness of an individual.
/// 
/// # Parameters
/// - `individual`: A reference to a vector containing the features of the individual.
/// 
/// # Returns
/// The fitness of the individual.
fn fitness_function(individual: &Vec<f64>) -> f64 {
    (individual[0] - 1.0).powf(2.0) + (individual[1] + 2.0).powf(2.0) + (individual[2] - 5.0).powf(2.0) + 3.0
}

/// Returns the fittest value in a population.
/// 
/// # Parameters
/// - `population`: A reference to the population.
///
/// # Returns
/// The fittest value in the population.
fn fittest_value(population: &Vec<Vec<f64>>) -> f64 {
    *evaluate_population(population)
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

/// Evaluates the fitness of each individual in a population.
///
/// # Parameters
/// - `population`: A reference to the population.
/// 
/// # Returns
/// A vector containing the fitness of each individual in the population.
fn evaluate_population(population: &Vec<Vec<f64>>) -> Vec<f64> {
    population
        .iter()
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
fn tournament_selection(population: &Vec<Vec<f64>>, tournament_size: usize) -> Vec<Vec<f64>> {
    let population_arc = std::sync::Arc::new(population.clone());
    (0..population.len())
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let tournament: Vec<Vec<f64>> = (0..tournament_size)
                .map(|_| {
                    let index = rng.gen_range(0..population_arc.len());
                    population_arc[index].clone()
                })
                .collect();
            tournament
                .iter()
                .min_by(|a, b| fitness_function(a).partial_cmp(&fitness_function(b)).unwrap())
                .unwrap()
                .clone()
        })
        .collect()
}

/// Performs arithmetic crossover between two parents.
/// 
/// # Parameters
/// - `parent1`: A reference to the first parent.
/// - `parent2`: A reference to the second parent.
/// 
/// # Returns
/// A tuple containing the two children, each represented as a vector of numeric values.
fn arithmetic_crossover(parent1: &Vec<f64>, parent2: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let alpha = rng.gen_range(0.0..1.0);
    let child1: Vec<f64> = parent1.iter().zip(parent2.iter()).map(|(a, b)| alpha * a + (1.0 - alpha) * b).collect();
    let child2: Vec<f64> = parent1.iter().zip(parent2.iter()).map(|(a, b)| (1.0 - alpha) * a + alpha * b).collect();
    (child1, child2)
}


/// Mutates an individual by randomly changing some of its genes.
/// 
/// # Parameters
/// - `individual`: A reference to the individual to be mutated.
/// - `mutation_rate`: The probability that a gene will be mutated.
/// 
/// # Returns
/// The mutated individual.
fn mutate(individual: &Vec<f64>, mutation_rate: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    individual
        .iter()
        .map(|gene| {
            if rng.gen_bool(mutation_rate) {
                rng.gen_range(-10.0..10.0)
            } else {
                *gene
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
fn diversity_function(population: &Vec<Vec<f64>>) -> f64 {
    let mut diversity = 0.0;
    for i in 0..population.len() {
        for j in i + 1..population.len() {
            let distance = euclidean_distance(&population[i], &population[j]);
            diversity += distance;
        }
    }
    diversity
}

/// Calculate the Euclidean distance between two individuals.
/// 
/// # Parameters
/// - `individual1`: A reference to the first individual.
/// - `individual2`: A reference to the second individual.
/// 
/// # Returns
/// The Euclidean distance between the two individuals.
/// 
/// # Note
/// The Euclidean distance is calculated as the square root of the sum of the squared differences between the corresponding elements of the two individuals.
fn euclidean_distance(individual1: &Vec<f64>, individual2: &Vec<f64>) -> f64 {
    individual1
        .iter()
        .zip(individual2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
 