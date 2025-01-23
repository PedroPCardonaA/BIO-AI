use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use csv::ReaderBuilder;
use plotters::prelude::*;
use rand::Rng;

/// Structure to represent an item in the knapsack problem
#[derive(Debug, Clone)]
struct Item {
    weight: u32,
    value: u32,
}

/// Load a dataset from a CSV file
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
fn generate_population(population_size: usize, features_count: usize) -> Vec<Vec<bool>> {
    let mut rng = rand::thread_rng();
    (0..population_size)
        .map(|_| (0..features_count).map(|_| rng.gen_bool(0.5)).collect())
        .collect()
}

/// Perform tournament selection
fn tournament_selection(
    population: &Vec<Vec<bool>>,
    fitness_values: &Vec<i64>,
    tournament_size: usize
) -> Vec<Vec<bool>> {
    let mut rng = rand::thread_rng();
    (0..population.len())
        .map(|_| {
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
fn roulette_wheel_selection(
    population: &Vec<Vec<bool>>,
    fitness_values: &Vec<i64>,
    selection_size: usize
) -> Vec<Vec<bool>> {
    let total_fitness: i64 = fitness_values.iter().sum();
    // If total_fitness is 0 or negative, 
    // we can avoid division by zero or negative. 
    if total_fitness <= 0 {
        // Fallback: just return random solutions from the population
        let mut rng = rand::thread_rng();
        return (0..selection_size)
            .map(|_| {
                let idx = rng.gen_range(0..population.len());
                population[idx].clone()
            })
            .collect();
    }

    let probabilities: Vec<f64> = fitness_values
        .iter()
        .map(|&f| f as f64 / total_fitness as f64)
        .collect();

    let mut rng = rand::thread_rng();
    (0..selection_size)
        .map(|_| {
            let mut acc = 0.0;
            let threshold = rng.gen::<f64>();
            for (i, &prob) in probabilities.iter().enumerate() {
                acc += prob;
                if acc >= threshold {
                    return population[i].clone();
                }
            }
            population[probabilities.len() - 1].clone()
        })
        .collect()
}

/// Perform single-point crossover
fn single_point_crossover(parent1: &Vec<bool>, parent2: &Vec<bool>) -> (Vec<bool>, Vec<bool>) {
    let mut rng = rand::thread_rng();
    let point = rng.gen_range(0..parent1.len());
    let mut offspring1 = parent1[..point].to_vec();
    offspring1.extend_from_slice(&parent2[point..]);

    let mut offspring2 = parent2[..point].to_vec();
    offspring2.extend_from_slice(&parent1[point..]);

    (offspring1, offspring2)
}

/// Perform mutation by flipping a bit
fn flip_bit_mutation(solution: &mut Vec<bool>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for bit in solution.iter_mut() {
        if rng.gen::<f64>() < mutation_rate {
            *bit = !*bit;
        }
    }
}

/// Perform elitism
fn elitism(
    population: &mut Vec<Vec<bool>>,
    fitness_values: &Vec<i64>,
    elitism_count: usize,
) -> Vec<Vec<bool>> {
    let mut paired: Vec<_> = population.iter().zip(fitness_values).collect();
    // Sort in descending order of fitness
    paired.sort_by_key(|&(_, fitness)| -fitness);
    paired
        .into_iter()
        .take(elitism_count)
        .map(|(sol, _)| sol.clone())
        .collect()
}

/// A threshold at which you decide to stop the algorithm
static STOP_THRESHOLD: i64 = 290_000;

/// Perform the genetic algorithm in a thread, stopping early if any other thread found a solution
fn genetic_algorithm_parallel(
    items: Vec<Item>,
    capacity: u32,
    generations: usize,
    population_size: usize,
    mutation_rate: f64,
    stop_signal: Arc<AtomicBool>,
    thread_id: usize,
) {
    let num_items = items.len();
    let mut population = generate_population(population_size, num_items);
    let mut data = Vec::new();

    for generation in 0..generations {
        // If some other thread has already found a good-enough solution, stop immediately.
        if stop_signal.load(Ordering::Relaxed) {
            println!("[Thread {thread_id}] Detected stop signal at generation {generation}. Exiting...");
            break;
        }

        // Evaluate fitness
        let fitnesses: Vec<i64> = population
            .iter()
            .map(|sol| fitness_function(sol, &items, capacity))
            .collect();

        // Check if we already have a good-enough fitness in this population
        let max_fitness = *fitnesses.iter().max().unwrap_or(&0);
        if max_fitness >= STOP_THRESHOLD {
            println!(
                "[Thread {thread_id}] Found fitness {} ≥ {} at generation {}. Broadcasting stop...",
                max_fitness, STOP_THRESHOLD, generation
            );
            // Broadcast stop to every thread
            stop_signal.store(true, Ordering::Relaxed);
            break;
        }

        if generation % 10 == 0 {
            let avg_fitness: f64 = fitnesses.iter().map(|&f| f as f64).sum::<f64>() / fitnesses.len() as f64;
            let min_fitness = *fitnesses.iter().min().unwrap_or(&0);
            println!(
                "[Thread {thread_id}] Generation {}: Max Fitness = {}, Avg Fitness = {}, Min Fitness = {}",
                generation, max_fitness, avg_fitness, min_fitness
            );
            data.push([
                generation as f64,
                max_fitness as f64,
                avg_fitness,
                min_fitness as f64,
            ]);
        }

        // Selection (can swap to tournament_selection if desired)
        let parents = roulette_wheel_selection(&population, &fitnesses, 5);

        // Crossover
        let mut offspring = Vec::new();
        for pair in parents.chunks(2) {
            if pair.len() == 2 {
                let (offspring1, offspring2) = single_point_crossover(&pair[0], &pair[1]);
                offspring.push(offspring1);
                offspring.push(offspring2);
            }
        }

        // Mutation
        for child in offspring.iter_mut() {
            flip_bit_mutation(child, mutation_rate);
        }

        // Elitism
        population = elitism(&mut population, &fitnesses, population_size / 2);
        population.extend(offspring.into_iter().take(population_size - population.len()));
    }

    // Optionally plot data for each thread if desired:
    plot_data(data, thread_id);
}

fn plot_data(data: Vec<[f64; 4]>, thread_id: usize) {
    if data.is_empty() {
        return;
    }

    let generations: Vec<f64> = data.iter().map(|entry| entry[0]).collect();
    let max_fitness: Vec<f64> = data.iter().map(|entry| entry[1]).collect();
    let avg_fitness: Vec<f64> = data.iter().map(|entry| entry[2]).collect();
    let min_fitness: Vec<f64> = data.iter().map(|entry| entry[3]).collect();

    let plot_name = format!("fitness_plot_thread_{}.png", thread_id);
    let root = BitMapBackend::new(&plot_name, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Thread {thread_id}: Fitness Over Generations"), ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            generations[0]..*generations.last().unwrap_or(&0.0),
            min_fitness.iter().cloned().reduce(f64::min).unwrap_or(0.0)
                ..max_fitness.iter().cloned().reduce(f64::max).unwrap_or(1.0),
        )
        .unwrap();

    // Configure the axes
    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness")
        .draw()
        .unwrap();

    // Max fitness
    chart
        .draw_series(LineSeries::new(
            generations.iter().zip(max_fitness.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))
        .unwrap()
        .label("Max Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Avg fitness
    chart
        .draw_series(LineSeries::new(
            generations.iter().zip(avg_fitness.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))
        .unwrap()
        .label("Avg Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Min fitness
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

    println!("Thread {thread_id} plot saved as '{plot_name}'.");
}

/// Main function
fn main() {
    let items = load_dataset("data/knapPI_12_500_1000_82.csv");
    println!("Loaded {} items", items.len());
    let capacity = 280785;

    // The shared atomic bool to signal all threads to stop when set to true.
    let stop_signal = Arc::new(AtomicBool::new(false));

    // We will run N threads in parallel. Each runs the GA.
    let num_threads = 4;
    let generations = 1000;
    let population_size = 50;
    let mutation_rate = 0.01;

    let mut handles = Vec::with_capacity(num_threads);

    for i in 0..num_threads {
        let items_clone = items.clone();
        let stop_clone = Arc::clone(&stop_signal);

        let handle = thread::spawn(move || {
            genetic_algorithm_parallel(
                items_clone,
                capacity,
                generations,
                population_size,
                mutation_rate,
                stop_clone,
                i,
            );
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for h in handles {
        h.join().unwrap();
    }

    println!("All threads have finished (or were stopped early).");
}
