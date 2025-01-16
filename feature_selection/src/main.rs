use csv::Reader;
use ndarray::{Array2, Axis};
use rand::seq::SliceRandom;
use rand::Rng;

fn load_dataset(file_path: &str) -> (Array2<f64>, Vec<f64>) {
    let mut reader = Reader::from_path(file_path).expect("Failed to read file");
    let mut features = Vec::new();
    let mut targets = Vec::new();

    for result in reader.records() {
        let record = result.expect("Failed to parse record");
        let mut row: Vec<f64> = record
            .iter()
            .map(|x| x.parse::<f64>().unwrap())
            .collect();
        targets.push(row.pop().unwrap());
        features.push(row);
    }

    let num_rows = features.len();
    let num_cols = features[0].len();
    (
        Array2::from_shape_vec((num_rows, num_cols), features.into_iter().flatten().collect()).unwrap(),
        targets,
    )
}

fn initialize_population(pop_size: usize, num_features: usize) -> Vec<Vec<bool>> {
    (0..pop_size)
        .map(|_| (0..num_features).map(|_| rand::random::<bool>()).collect())
        .collect()
}

fn crossover(parent1: &[bool], parent2: &[bool]) -> (Vec<bool>, Vec<bool>) {
    let crossover_point = rand::thread_rng().gen_range(0..parent1.len());
    let child1 = parent1[0..crossover_point]
        .iter()
        .chain(&parent2[crossover_point..])
        .copied()
        .collect();
    let child2 = parent2[0..crossover_point]
        .iter()
        .chain(&parent1[crossover_point..])
        .copied()
        .collect();
    (child1, child2)
}

fn mutate(chromosome: &mut Vec<bool>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for gene in chromosome.iter_mut() {
        if rng.gen::<f64>() < mutation_rate {
            *gene = !*gene;
        }
    }
}

fn deterministic_crowding(
    parent1: &[bool],
    parent2: &[bool],
    offspring1: &[bool],
    offspring2: &[bool],
    fitness_fn: impl Fn(&[bool]) -> f64,
) -> (Vec<bool>, Vec<bool>) {
    let parent1_fitness = fitness_fn(parent1);
    let parent2_fitness = fitness_fn(parent2);
    let offspring1_fitness = fitness_fn(offspring1);
    let offspring2_fitness = fitness_fn(offspring2);

    if offspring1_fitness < parent1_fitness || offspring1_fitness < parent2_fitness {
        (offspring1.to_vec(), parent2.to_vec())
    } else {
        (parent1.to_vec(), offspring2.to_vec())
    }
}

fn tournament_selection<'a>(
    population: &'a [Vec<bool>],
    fitness_fn: &impl Fn(&[bool]) -> f64,
    tournament_size: usize,
) -> &'a Vec<bool> {
    let mut rng = rand::thread_rng();
    let tournament: Vec<&Vec<bool>> = population
        .choose_multiple(&mut rng, tournament_size)
        .collect();
    tournament.into_iter().min_by(|a, b| fitness_fn(a).partial_cmp(&fitness_fn(b)).unwrap()).unwrap()
}

fn calculate_rmse(chromosome: &[bool], features: &Array2<f64>, targets: &[f64]) -> f64 {
    let selected_features: Vec<usize> = chromosome
        .iter()
        .enumerate()
        .filter_map(|(i, &is_selected)| if is_selected { Some(i) } else { None })
        .collect();

    if selected_features.is_empty() {
        return f64::MAX; // Penalize empty feature selection
    }

    let selected_data: Array2<f64> = features.select(Axis(1), &selected_features);
    let predictions: Vec<f64> = selected_data
        .outer_iter()
        .map(|row| row.mean().unwrap_or(0.0))
        .collect();

    let mse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / targets.len() as f64;

    mse.sqrt()
}

fn run_ga(
    features: &Array2<f64>,
    targets: &[f64],
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
) {
    let mut population = initialize_population(population_size, features.ncols());
    let fitness_fn = |chromosome: &[bool]| calculate_rmse(chromosome, features, targets);

    for generation in 0..generations {
        let mut new_population = Vec::new();
        for _ in 0..(population_size / 2) {
            let parent1 = tournament_selection(&population, &fitness_fn, 3);
            let parent2 = tournament_selection(&population, &fitness_fn, 3);
            let (mut child1, mut child2) = crossover(parent1, parent2);
            mutate(&mut child1, mutation_rate);
            mutate(&mut child2, mutation_rate);

            let (survivor1, survivor2) = deterministic_crowding(parent1, parent2, &child1, &child2, &fitness_fn);
            new_population.push(survivor1);
            new_population.push(survivor2);
        }
        population = new_population;

        // Log progress and best fitness
        let best_fitness = population.iter().map(|c| fitness_fn(c)).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        println!("Generation {}: Best fitness = {:.4}", generation, best_fitness);
    }
}

fn main() {
    let file_path = "data/dataset.txt"; // Update with your dataset path
    let (features, targets) = load_dataset(file_path);
    let population_size = 100;
    let generations = 100;
    let mutation_rate = 1.0 / features.ncols() as f64;

    run_ga(&features, &targets, population_size, generations, mutation_rate);
}
