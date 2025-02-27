use rand::Rng;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run NSGA-II: population size 100, for 50 generations.
    let final_population = nsga_2(100, 2500);
    
    // Evaluate final population: get Pareto fronts and ranks.
    let (fronts, ranks) = non_dominated_sort(final_population.clone());
    
    println!("Final Pareto Fronts:");
    for (i, front) in fronts.iter().enumerate() {
        println!("Front {}: {:?}", i + 1, front);
    }
    
    // Plot the final population, highlighting individuals with rank 0 in red.
    plot_population(&final_population, &ranks)?;
    
    Ok(())
}

// --- Objective functions ---
// Here, we try to minimize both f1 and f2.
fn function_1(x: f64) -> f64 {
    x 
}

fn function_2(x: f64, y: f64) -> f64 {
    (1.0 + y) / x
}

// --- Population initialization ---
fn generate_population(size: usize) -> Vec<Vec<f64>> {
    let mut population = Vec::new();
    let mut rng = rand::rng();
    for _ in 0..size {
        let individual = vec![
            rng.random_range(0.1..1.0), // x value
            rng.random_range(0.0..5.0)  // y value
        ];
        population.push(individual);
    }
    population
}

// --- Variation operators ---
fn random_mutation(individual: &mut Vec<f64>) {
    let mut rng = rand::rng();
    let index = rng.random_range(0..individual.len());
    if index == 0 {
        individual[index] = rng.random_range(0.1..1.0);
    } else {
        individual[index] = rng.random_range(0.0..5.0);
    }
}

fn arithmetic_crossover(parent_1: &Vec<f64>, parent_2: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rng();
    let alpha = rng.random_range(0.0..1.0);
    let child_1 = vec![
        alpha * parent_1[0] + (1.0 - alpha) * parent_2[0],
        alpha * parent_1[1] + (1.0 - alpha) * parent_2[1],
    ];
    let child_2 = vec![
        alpha * parent_2[0] + (1.0 - alpha) * parent_1[0],
        alpha * parent_2[1] + (1.0 - alpha) * parent_1[1],
    ];
    (child_1, child_2)
}

// --- Dominance and sorting ---
/// Returns true if `individual_1` dominates `individual_2` (minimization).
fn dominates(individual_1: &Vec<f64>, individual_2: &Vec<f64>) -> bool {
    let f1_1 = function_1(individual_1[0]);
    let f2_1 = function_2(individual_1[0], individual_1[1]);
    let f1_2 = function_1(individual_2[0]);
    let f2_2 = function_2(individual_2[0], individual_2[1]);
    
    (f1_1 <= f1_2 && f2_1 <= f2_2) && (f1_1 < f1_2 || f2_1 < f2_2)
}

/// Non-dominated sorting that returns a tuple:
/// - A vector of fronts (each front is a Vec of individuals)
/// - A vector `rank` where rank[i] is the front number for the i-th individual.
fn non_dominated_sort(population: Vec<Vec<f64>>) -> (Vec<Vec<Vec<f64>>>, Vec<usize>) {
    let mut fronts: Vec<Vec<Vec<f64>>> = Vec::new();
    let n = population.len();
    let mut domination_counts = vec![0; n];
    let mut dominated_set = vec![Vec::new(); n];
    let mut rank = vec![0; n];

    for (i, p) in population.iter().enumerate() {
        for (j, q) in population.iter().enumerate() {
            if dominates(p, q) {
                dominated_set[i].push(j);
            } else if dominates(q, p) {
                domination_counts[i] += 1;
            }
        }
        if domination_counts[i] == 0 {
            rank[i] = 0;
        }
    }
    
    let mut current_front: Vec<Vec<f64>> = population.iter()
        .enumerate()
        .filter(|(i, _)| rank[*i] == 0)
        .map(|(_, ind)| ind.clone())
        .collect();
    
    let mut front_number = 0;
    while !current_front.is_empty() {
        fronts.push(current_front.clone());
        let mut next_front: Vec<Vec<f64>> = Vec::new();
        for (i, _p) in population.iter().enumerate() {
            if rank[i] == front_number {
                for &j in &dominated_set[i] {
                    domination_counts[j] -= 1;
                    if domination_counts[j] == 0 {
                        rank[j] = front_number + 1;
                        next_front.push(population[j].clone());
                    }
                }
            }
        }
        front_number += 1;
        current_front = next_front;
    }
    (fronts, rank)
}

// --- Crowding distance ---
/// Computes crowding distances for a given set of individuals (a front),
/// given their fitness values (each fitness is a Vec of objective values).
fn crowding_distance(population: &Vec<Vec<f64>>, fitness: &Vec<Vec<f64>>) -> Vec<f64> {
    let n = population.len();
    if n == 0 {
        return vec![];
    }
    let num_objectives = fitness[0].len();
    let mut distances = vec![0.0; n];
    for obj in 0..num_objectives {
        let mut sorted: Vec<(usize, f64)> = (0..n)
            .map(|i| (i, fitness[i][obj]))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances[sorted[0].0] = f64::INFINITY;
        distances[sorted[n - 1].0] = f64::INFINITY;
        let min_val = sorted[0].1;
        let max_val = sorted[n - 1].1;
        let range = max_val - min_val;
        if range == 0.0 {
            continue;
        }
        for i in 1..(n - 1) {
            distances[sorted[i].0] += (sorted[i + 1].1 - sorted[i - 1].1) / range;
        }
    }
    distances
}

/// Computes crowding distances for the entire population, front by front.
fn compute_crowding_distances(population: &Vec<Vec<f64>>, ranks: &Vec<usize>) -> Vec<f64> {
    // Compute fitness values for each individual.
    let fitness: Vec<Vec<f64>> = population
        .iter()
        .map(|ind| vec![function_1(ind[0]), function_2(ind[0], ind[1])])
        .collect();
    let mut distances = vec![0.0; population.len()];
    let max_rank = *ranks.iter().max().unwrap();
    for front in 0..=max_rank {
        let indices: Vec<usize> = ranks
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r == front)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let mut pop_front = Vec::new();
        let mut fit_front = Vec::new();
        for &i in &indices {
            pop_front.push(population[i].clone());
            fit_front.push(fitness[i].clone());
        }
        let cd = crowding_distance(&pop_front, &fit_front);
        for (j, &idx) in indices.iter().enumerate() {
            distances[idx] = cd[j];
        }
    }
    distances
}

// --- Tournament selection ---
/// Selects one individual based on binary tournament using rank (lower is better)
/// and, in case of tie, crowding distance (higher is better).
fn tournament_selection(
    population: &Vec<Vec<f64>>, 
    ranks: &Vec<usize>, 
    crowding: &Vec<f64>
) -> Vec<f64> {
    let mut rng = rand::rng();
    let i = rng.random_range(0..population.len());
    let j = rng.random_range(0..population.len());
    if ranks[i] < ranks[j] {
        population[i].clone()
    } else if ranks[i] > ranks[j] {
        population[j].clone()
    } else {
        if crowding[i] > crowding[j] {
            population[i].clone()
        } else {
            population[j].clone()
        }
    }
}

// --- NSGA-II Main Loop ---
fn nsga_2(pop_size: usize, generations: usize) -> Vec<Vec<f64>> {
    let mut population = generate_population(pop_size);
    let mut rng = rand::rng();
    for gen in 0..generations {
        // Evaluate current population: sort and assign crowding distances.
        let (_fronts, ranks) = non_dominated_sort(population.clone());
        let crowding = compute_crowding_distances(&population, &ranks);
        
        // Create offspring via tournament selection, crossover, and mutation.
        let mut offspring = Vec::new();
        while offspring.len() < pop_size {
            let parent1 = tournament_selection(&population, &ranks, &crowding);
            let parent2 = tournament_selection(&population, &ranks, &crowding);
            let (mut child1, mut child2) = arithmetic_crossover(&parent1, &parent2);
            if rng.random_bool(0.1) {
                random_mutation(&mut child1);
            }
            if rng.random_bool(0.1) {
                random_mutation(&mut child2);
            }
            offspring.push(child1);
            if offspring.len() < pop_size {
                offspring.push(child2);
            }
        }
        
        // Combine parent and offspring populations.
        let mut combined = population.clone();
        combined.extend(offspring);
        
        // Perform non-dominated sorting on the combined population.
        let (combined_fronts, _) = non_dominated_sort(combined.clone());
        let mut new_population = Vec::new();
        let mut i = 0;
        // Fill the new population front by front.
        while new_population.len() < pop_size {
            let front = &combined_fronts[i];
            if new_population.len() + front.len() <= pop_size {
                new_population.extend(front.clone());
            } else {
                // If the front must be partially included, sort it by crowding distance (descending)
                let front_fitness: Vec<Vec<f64>> = front
                    .iter()
                    .map(|ind| vec![function_1(ind[0]), function_2(ind[0], ind[1])])
                    .collect();
                let cd = crowding_distance(front, &front_fitness);
                let mut front_with_cd: Vec<(Vec<f64>, f64)> = front.iter().cloned().zip(cd.into_iter()).collect();
                front_with_cd.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let remaining = pop_size - new_population.len();
                for j in 0..remaining {
                    new_population.push(front_with_cd[j].0.clone());
                }
            }
            i += 1;
        }
        population = new_population;
        if gen % 100 == 0 {
            println!("Generation {}: Population size {}", gen, population.len());
        }
    }
    population
}

// --- Plotting ---
/// Plots all individuals in the objective space (x-axis: function_1, y-axis: function_2)
/// and highlights those with rank 0 (first Pareto front) in red.
fn plot_population(
    population: &Vec<Vec<f64>>,
    ranks: &Vec<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("population_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Population in Objective Space", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..1.0, 0.0..70.0)?;
    
    chart.configure_mesh().draw()?;
    
    for (i, individual) in population.iter().enumerate() {
        let x_val = function_1(individual[0]);
        let y_val = function_2(individual[0], individual[1]);
        let point_color = if ranks[i] == 0 { &RED } else { &BLUE };
        chart.draw_series(std::iter::once(Circle::new(
            (x_val, y_val),
            5,
            point_color.filled(),
        )))?;
    }
    
    root.present()?;
    println!("Plot saved to population_plot.png");
    Ok(())
}
