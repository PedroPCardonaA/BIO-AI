use rand::Rng;

const BITSTRING_SIZE: usize = 24;
const POPULATION_SIZE: usize = 100;
const GENERATIONS: usize = 1000;
const MUTATION_RATE: f64 = 0.01;
const TOURNAMENT_SIZE: usize = 5;

fn bitstring_to_real(bitstring: &Vec<bool>) -> f64 {
    let int_value: u32 = bitstring.iter().rev().enumerate().map(|(i, &b)| (b as u32) << i).sum();
    let max_value = (1 << BITSTRING_SIZE) - 1;
    (int_value as f64) * 128.0 / max_value as f64
}

fn fitness(bitstring: &Vec<bool>) -> f64 {
    let x = bitstring_to_real(bitstring);
    x.sin()
}

fn random_bitstring() -> Vec<bool> {
    let mut rng = rand::thread_rng();
    (0..BITSTRING_SIZE).map(|_| rng.gen_bool(0.5)).collect()
}

fn crossover(parent1: &Vec<bool>, parent2: &Vec<bool>) -> Vec<bool> {
    let mut rng = rand::thread_rng();
    let point = rng.gen_range(0..BITSTRING_SIZE);
    parent1[..point].iter().chain(&parent2[point..]).cloned().collect()
}

fn mutate(bitstring: &mut Vec<bool>) {
    let mut rng = rand::thread_rng();
    for bit in bitstring.iter_mut() {
        if rng.gen_bool(MUTATION_RATE) {
            *bit = !*bit;
        }
    }
}

// Tournament selection
fn tournament_selection(population: &Vec<Vec<bool>>, fitness_scores: &Vec<f64>) -> Vec<bool> {
    let mut rng = rand::thread_rng();
    let mut tournament: Vec<usize> = (0..TOURNAMENT_SIZE)
        .map(|_| rng.gen_range(0..POPULATION_SIZE))
        .collect();
    
    tournament.sort_by(|&a, &b| fitness_scores[b].partial_cmp(&fitness_scores[a]).unwrap());
    population[tournament[0]].clone()
}

fn main() {
    let mut rng = rand::thread_rng();

    // Initialize population
    let mut population: Vec<Vec<bool>> = (0..POPULATION_SIZE).map(|_| random_bitstring()).collect();

    for generation in 0..GENERATIONS {
        // Evaluate fitness
        let fitness_scores: Vec<f64> = population.iter().map(|bitstring| fitness(bitstring)).collect();

        // Print best solution
        let (best_index, best_fitness) = fitness_scores
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        println!("Generation {}: Best fitness = {}", generation, best_fitness);

        // Create new population using tournament selection
        let mut new_population = Vec::new();
        while new_population.len() < POPULATION_SIZE {
            let parent1 = tournament_selection(&population, &fitness_scores);
            let parent2 = tournament_selection(&population, &fitness_scores);
            let mut offspring = crossover(&parent1, &parent2);
            mutate(&mut offspring);
            new_population.push(offspring);
        }
        population = new_population;
    }

    // Evaluate final population
    let fitness_scores: Vec<f64> = population.iter().map(|bitstring| fitness(bitstring)).collect();
    let (best_index, best_fitness) = fitness_scores
        .iter()
        .enumerate()
        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    let best_bitstring = &population[best_index];
    println!(
        "Best solution: x = {}, sin(x) = {}",
        bitstring_to_real(best_bitstring),
        best_fitness
    );
}
