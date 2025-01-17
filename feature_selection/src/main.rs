use csv::ReaderBuilder;
use rand::Rng;
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use std::error::Error;

pub fn load_dataset(path: &str) -> Result<(DMatrix<f64>, DVector<f64>), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    // We'll store feature values (row-wise) in `data`.
    let mut data: Vec<f64> = Vec::new();
    // We'll store each target value in `targets`.
    let mut targets: Vec<f64> = Vec::new();

    for record_result in reader.records() {
        let record = record_result?;
        // Convert each field in the record to f64
        let values: Vec<f64> = record
            .iter()
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;

        // Expect exactly 102 columns: 101 features + 1 target
        assert_eq!(
            values.len(),
            102,
            "Unexpected number of columns in record"
        );

        // Last column is the target
        targets.push(values[101]);

        // First 101 columns are features
        data.extend_from_slice(&values[..101]);
    }

    // The number of rows in our dataset equals the number of targets
    let nrows = targets.len();
    let ncols = 101; // We know we have 101 features

    // Construct a DMatrix from `data`, which is stored row by row
    let features = DMatrix::from_row_slice(nrows, ncols, &data);

    // Construct a DVector for our target values
    let target_vector = DVector::from_row_slice(&targets);

    Ok((features, target_vector))
}


fn generate_population(population_size: usize, features_count: usize) -> Vec<Vec<bool>> {
    (0..population_size)
        .map(|_| {
            (0..features_count)
                .map(|_| rand::random())
                .collect()
        })
        .collect()
}

fn tournament_selection(
    population: &Vec<Vec<bool>>,
    fitness_values: &Vec<f64>,
    tournament_size: usize) -> Vec<Vec<bool>>{
        let mut rng = rand::thread_rng();
        (0..population.len())
            .map(|_|{
                let candidates: Vec<_> = (0..tournament_size)
                    .map(|_| rng.gen_range(0..population.len()))
                    .collect();
                let best_candidate = candidates
                    .iter()
                    .min_by(|&&idx1, &&idx2| fitness_values[idx1].partial_cmp(&fitness_values[idx2]).unwrap())
                    .unwrap();
                population[*best_candidate].clone()
            })
            .collect()
}

fn single_point_crossover(parent1: &Vec<bool>, parent2: &Vec<bool>) -> (Vec<bool>, Vec<bool>){
    let mut rng = rand::thread_rng();
    let point = rng.gen_range(0..parent1.len());
    let mut offspring1 = parent1[..point].to_vec();
    offspring1.extend_from_slice(&parent2[point..]);
    let mut offspring2 = parent2[..point].to_vec();
    offspring2.extend_from_slice(&parent1[point..]);
    (offspring1, offspring2)
}

fn flip_bit_mutation(solution: &mut Vec<bool>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for bit in solution.iter_mut() {
        if rng.gen::<f64>() < mutation_rate {
            *bit = !*bit;
        }
    }
}

fn elitism(
    population: &mut Vec<Vec<bool>>,
    fitness_values: &Vec<f64>,
    elitism_count: usize,
) -> Vec<Vec<bool>> {
    let mut paired: Vec<_> = population.iter().zip(fitness_values).collect();
    paired.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    paired.into_iter()
        .take(elitism_count)
        .map(|(sol, _)| sol.clone())
        .collect()
}

fn genetic_algorithm(X: &DMatrix<f64>, y: &DVector<f64>, generations: usize, population_size: usize, mutation_rate: f64){
    let features_count = X.ncols();
    let mut population = generate_population(population_size, features_count);
    let mut cache: HashMap<String, f64> = HashMap::new();

    for generation in 0..generations{
        let fitnesses: Vec<f64> = population.iter().map(|sol| fitness_function(sol, X, y, &mut cache)).collect();
        let min_fitness = fitnesses.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        println!("Generation: {}, Min Fitness: {}", generation, min_fitness);
        let parents = tournament_selection(&population, &fitnesses, 2);
        let mut offspring: Vec<Vec<bool>> = Vec::new();
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
}



pub fn fitness_function(
    solution: &[bool],
    X: &DMatrix<f64>,
    y: &DVector<f64>,
    cache: &mut std::collections::HashMap<String, f64>,
) -> f64 {
    // 1. Convert solution (Vec<bool>) to a unique string key like "1010011"
    let key = solution
        .iter()
        .map(|&bit| if bit { '1' } else { '0' })
        .collect::<String>();

    // 2. If we already computed fitness for this solution, return it immediately
    if let Some(&cached_fitness) = cache.get(&key) {
        return cached_fitness;
    }

    // 3. Gather selected feature indices from the boolean mask
    let selected_indices: Vec<usize> = solution
        .iter()
        .enumerate()
        .filter_map(|(col_idx, &bit)| if bit { Some(col_idx) } else { None })
        .collect();

    // Edge case: If no features are selected, return a large penalty
    if selected_indices.is_empty() {
        let penalty = f64::MAX;
        cache.insert(key, penalty);
        return penalty;
    }

    // 4. Build the submatrix of X that has only the selected columns
    let nrows = X.nrows();
    let ncols = selected_indices.len();

    // We'll place the data row-by-row in a Vec, then construct a DMatrix
    let mut data = Vec::with_capacity(nrows * ncols);
    for row in 0..nrows {
        for &col in &selected_indices {
            data.push(X[(row, col)]);
        }
    }
    let X_selected = DMatrix::from_row_slice(nrows, ncols, &data);

    // 5. Solve for coefficients via SVD-based least squares
    //    This handles tall (nrows >= ncols) or wide (nrows < ncols) matrices.
    let svd = X_selected.clone().svd(true, true);
    // Tolerance can be adjusted (1e-12, 1e-9, etc.) to handle near-singular matrices
    let coeffs = svd.solve(&y, 1e-12);

    // 6. Compute predictions and calculate RMSE
    let y_pred = X_selected * coeffs.unwrap();
    let residuals = &y_pred - y;
    let mse = residuals.dot(&residuals) / (nrows as f64);
    let rmse = mse.sqrt();

    // 7. Store in cache before returning
    cache.insert(key, rmse);

    rmse
}

fn main() {
    let (X, y) = load_dataset("data/dataset.txt").unwrap();
    genetic_algorithm(&X, &y, 100, 100, 0.01);
}
