use csv::ReaderBuilder;
use plotters::prelude::*;
use rand::Rng;

/// Structure to represent an item in the knapsack problem
/// 
/// # Fields
/// - `weight`: The weight of the item
/// - `value`: The value of the item
#[derive(Debug, Clone)]
struct Item {
    weight: u32,
    value: u32,
}

/// Load a dataset from a CSV file
/// 
/// # Arguments
/// - `filename`: The name of the file to load
/// 
/// # Returns
/// A vector of `Item` structures
fn load_dataset(filename: &str) -> Vec<Item> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(filename)
        .expect("Unable to open file");

    reader
        .records()
        .map(|record| {
            let record = record.expect("Error reading record");
            let values: Vec<u32> = record
                .iter()
                .map(|v| v.parse().expect("Invalid number"))
                .collect();

            Item {
                weight: values[2], 
                value: values[1],
            }
        })
        .collect()
}

/// Evaluate the fitness of a solution
/// 
/// # Arguments
/// - `individual`: A vector of booleans representing the selected items
/// - `items`: A vector of `Item` structures
/// - `capacity`: The maximum capacity of the knapsack
/// 
/// # Returns
/// The fitness of the solution
fn fitness_function(individual: &Vec<bool>, items: &Vec<Item>, capacity: u32) -> i64 {
    let mut total_weight = 0;
    let mut total_value = 0;

    for (i, &selected) in individual.iter().enumerate() {
        if selected {
            total_weight += items[i].weight;
            total_value += items[i].value;
        }
    }

    if total_weight > capacity {
        let penalty = (total_weight - capacity) as i64 * -1;
        total_value as i64 + penalty
    } else {
        total_value as i64
    }
}

/// Generate a population of solutions
/// 
/// # Arguments
/// - `population_size`: The number of solutions to generate
/// - `features_count`: The number of items in the dataset
/// 
/// # Returns
/// A vector of solutions
fn generate_population(population_size: usize, features_count: usize) -> Vec<Vec<bool>> {
    let mut rng = rand::thread_rng();
    (0..population_size)
        .map(|_| (0..features_count).map(|_| rng.gen_bool(0.5)).collect())
        .collect()
}

/// Perform tournament selection
/// 
/// # Arguments
/// - `population`: A vector of solutions
/// - `fitness_values`: A vector of fitness values
/// - `tournament_size`: The number of individuals to select
/// 
/// # Returns
/// A vector of selected solutions
fn tournament_selection(
    population: &Vec<Vec<bool>>,
    fitness_values: &Vec<i64>,
    tournament_size: usize) -> Vec<Vec<bool>>{
        let mut rng = rand::thread_rng();
        (0..population.len())
            .map(|_|{
                let candidates: Vec<_> = (0..tournament_size)
                    .map(|_| rng.gen_range(0..population.len()))
                    .collect();
                let best_candidate = candidates
                    .iter()
                    .max_by_key(|&&idx| fitness_values[idx])
                    .unwrap();
                population[*best_candidate].clone()
            })
            .collect()
}


/// Perform roulette wheel selection
/// 
/// # Arguments
/// - `population`: A vector of solutions
/// - `fitness_values`: A vector of fitness values
/// - `selection_size`: The number of individuals to select
/// 
/// # Returns
/// A vector of selected solutions
fn roulette_wheel_selection(
    population: &Vec<Vec<bool>>,
    fitness_values: &Vec<i64>,
    selection_size: usize) -> Vec<Vec<bool>>{
        let total_fitness: i64 = fitness_values.iter().sum();
        let probabilities: Vec<f64> = fitness_values.iter().map(|&f| f as f64 / total_fitness as f64).collect();
        let mut rng = rand::thread_rng();
        (0..selection_size)
            .map(|_|{
                let mut acc = 0.0;
                let threshold = rng.gen::<f64>();
                for (i, &prob) in probabilities.iter().enumerate(){
                    acc += prob;
                    if acc >= threshold{
                        return population[i].clone();
                    }
                }
                population[probabilities.len()-1].clone()
            })
            .collect()
}

/// Perform single-point crossover
/// 
/// # Arguments
/// - `parent1`: The first parent
/// - `parent2`: The second parent
/// 
/// # Returns
/// A tuple containing the two offspring
fn single_point_crossover(parent1: &Vec<bool>, parent2: &Vec<bool>) -> (Vec<bool>, Vec<bool>){
    let mut rng = rand::thread_rng();
    let point = rng.gen_range(0..parent1.len());
    let mut offspring1 = parent1[..point].to_vec();
    offspring1.extend_from_slice(&parent2[point..]);
    let mut offspring2 = parent2[..point].to_vec();
    offspring2.extend_from_slice(&parent1[point..]);
    (offspring1, offspring2)
}

/// Perform mutation by flipping a bit
/// 
/// # Arguments
/// - `solution`: The solution to mutate
/// - `mutation_rate`: The probability of mutation
fn flip_bit_mutation(solution: &mut Vec<bool>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for bit in solution.iter_mut() {
        if rng.gen::<f64>() < mutation_rate {
            *bit = !*bit;
        }
    }
}

/// Perform elitism
/// 
/// # Arguments
/// - `population`: A vector of solutions
/// - `fitness_values`: A vector of fitness values
/// - `elitism_count`: The number of solutions to keep
/// 
/// # Returns
/// A vector of selected solutions
fn elitism(
    population: &mut Vec<Vec<bool>>,
    fitness_values: &Vec<i64>,
    elitism_count: usize,
) -> Vec<Vec<bool>> {
    let mut paired: Vec<_> = population.iter().zip(fitness_values).collect();
    paired.sort_by_key(|&(_, fitness)| -fitness);
    paired.into_iter().take(elitism_count).map(|(sol, _)| sol.clone()).collect()
}

/// Perform a genetic algorithm
/// 
/// # Arguments
/// - `items`: A vector of `Item` structures
/// - `capacity`: The maximum capacity of the knapsack
/// - `generations`: The number of generations to run
/// - `population_size`: The number of solutions in the population
/// - `mutation_rate`: The probability of mutation
fn genetic_algorithm(items: Vec<Item>, capacity: u32, generations: usize, population_size: usize, mutation_rate: f64){
    let num_items = items.len();
    let mut population = generate_population(population_size, num_items);
    let mut data = Vec::new();

    for generation in 0..generations{
        let fitnesses: Vec<i64> = population.iter().map(|sol| fitness_function(sol, &items, capacity)).collect();

        if generation % 10 == 0{
            let max_fitness = fitnesses.iter().max().unwrap();
            let avg_fitness: f64 = fitnesses.iter().map(|&f| f as f64).sum::<f64>() / fitnesses.len() as f64;
            println!("Generation {}: Max Fitness = {}, Avg Fitness = {}", generation, max_fitness, avg_fitness);
            data.push([generation as f64, *max_fitness as f64, avg_fitness, *fitnesses.iter().min().unwrap() as f64]);
        }

        let parents = roulette_wheel_selection(&population, &fitnesses, 5);

        let mut offspring = Vec::new();
        for pair in parents.chunks(2){
            if pair.len() == 2{
                let (offspring1, offspring2) = single_point_crossover(&pair[0], &pair[1]);
                offspring.push(offspring1);
                offspring.push(offspring2);
            }
        }
        for child in offspring.iter_mut(){
            flip_bit_mutation(child, mutation_rate);
        }

        population = elitism(&mut population, &fitnesses, population_size/2);
        population.extend(offspring.into_iter().take(population_size - population.len()));
    }
    plot_data(data);
}


fn plot_data(data: Vec<[f64; 4]>) {
    let generations: Vec<f64> = data.iter().map(|entry| entry[0]).collect();
    let max_fitness: Vec<f64> = data.iter().map(|entry| entry[1]).collect();
    let avg_fitness: Vec<f64> = data.iter().map(|entry| entry[2]).collect();
    let min_fitness: Vec<f64> = data.iter().map(|entry| entry[3]).collect();

    let root = BitMapBackend::new("fitness_plot_tournament.png", (1024, 768))
        .into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Fitness Over Generations with tournament selection.", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            generations[0]..*generations.last().unwrap(),
            min_fitness.iter().cloned().reduce(f64::min).unwrap()..max_fitness.iter().cloned().reduce(f64::max).unwrap(),
        )
        .unwrap();

    // Configure the axes
    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            generations.iter().zip(max_fitness.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))
        .unwrap()
        .label("Max Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            generations.iter().zip(avg_fitness.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))
        .unwrap()
        .label("Avg Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            generations.iter().zip(min_fitness.iter()).map(|(&x, &y)| (x, y)),
            &GREEN,
        ))
        .unwrap()
        .label("Min Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    println!("Plot saved as 'fitness_plot.png'");
}


/// Main function
fn main() {
    let items = load_dataset("data/knapPI_12_500_1000_82.csv");
    println!("Loaded {} items", items.len());
    let capacity = 280785;
    genetic_algorithm(items, capacity, 1000, 50, 0.01);
}