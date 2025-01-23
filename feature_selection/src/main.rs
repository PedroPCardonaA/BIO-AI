use rand::prelude::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::error::Error;
use std::f64;
use std::fs::File;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
mod lin_reg;
use lin_reg::LinReg;
use csv::ReaderBuilder;


pub trait FitnessEvaluator: Send + Sync {
    fn fitness(&self, chromosome: &Array1<u8>) -> f64;
}

pub struct EASolver<F: FitnessEvaluator + 'static> {
    pub population_size: usize,
    pub chromosome_length: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub max_generations: usize,

    pub population: Array2<u8>,
    pub fitness: Arc<Mutex<Vec<f64>>>,

    pub entropies: Vec<f64>,
    pub fitnesses: Vec<(usize, f64, f64, f64)>,

    evaluator: Arc<F>,

    cache: Arc<Mutex<HashMap<Vec<u8>, f64>>>,
    stop_flag: Arc<AtomicBool>,
}

impl<F: FitnessEvaluator> EASolver<F> {
    pub fn new(
        population_size: usize,
        chromosome_length: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        max_generations: usize,
        evaluator: F,
    ) -> Self {
        let mut rng = thread_rng();

        let mut population = Array2::<u8>::zeros((population_size, chromosome_length));
        for i in 0..population_size {
            for j in 0..chromosome_length {
                if rng.gen_bool(0.5) {
                    population[[i, j]] = 1;
                }
            }
        }

        let fitness = Arc::new(Mutex::new(vec![0.0; population_size]));

        EASolver {
            population_size,
            chromosome_length,
            mutation_rate,
            crossover_rate,
            max_generations,
            population,
            fitness,
            entropies: vec![],
            fitnesses: vec![],
            evaluator: Arc::new(evaluator),
            // Use empty HashMap and new AtomicBool
            cache: Arc::new(Mutex::new(HashMap::new())),
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    fn evaluate_population_parallel(&mut self) {
        let num_threads = num_cpus::get();
    
        let chunk_size = (self.population_size + num_threads - 1) / num_threads;

        let mut thread_handles = Vec::new();

        for t in 0..num_threads {
            let stop_flag = self.stop_flag.clone();
            let cache = self.cache.clone();
            let fitness_arc = self.fitness.clone();
            let evaluator = self.evaluator.clone();

            let start = t * chunk_size;
            let end = (start + chunk_size).min(self.population_size);
    
            let population_slice = self.population.slice(ndarray::s![start..end, ..]).to_owned();
    
            let handle = thread::spawn(move || {
                for i_local in 0..(end - start) {
                    if stop_flag.load(Ordering::SeqCst) {
                        break;
                    }
                    let i_global = start + i_local;
                    let chromosome = population_slice.slice(ndarray::s![i_local, ..]).to_owned();
                    let key: Vec<u8> = chromosome.to_vec();
                    let mut cached_val = None;
                    {
                        let map = cache.lock().unwrap();
                        if let Some(&val) = map.get(&key) {
                            cached_val = Some(val);
                        }
                    }
                    let fitness_val = match cached_val {
                        Some(val) => val,
                        None => {
                            let val = evaluator.fitness(&chromosome);
                            let mut map = cache.lock().unwrap();
                            map.insert(key, val);
                            val
                        }
                    };
                    {
                        let mut fitness = fitness_arc.lock().unwrap();
                        fitness[i_global] = fitness_val;
                    }
                    if fitness_val > -0.125 {
                        stop_flag.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            });
    
            thread_handles.push(handle);
        }
    
        for handle in thread_handles {
            let _ = handle.join();
        }
    }

    pub fn run(
        &mut self,
        selection_method: usize,
        tournament_size: usize,
        crossover_method: usize,
        crowding: bool,
    ) -> Array1<u8> {
        self.evaluate_population_parallel();

        for generation in 0..self.max_generations {
            if self.stop_flag.load(Ordering::SeqCst) {
                println!("Stop flag triggered; ending at generation {}", generation);
                break;
            }
            let mut child_population =
                Array2::<u8>::zeros((self.population_size, self.chromosome_length));
            let mut child_fitness = vec![0.0; self.population_size];
            let mut i = 0;
            while i < self.population_size {
                let (p1_idx, p2_idx) = self.select_parents(selection_method, tournament_size);
                let parent1 = self.population.slice(ndarray::s![p1_idx, ..]).to_owned();
                let parent2 = self.population.slice(ndarray::s![p2_idx, ..]).to_owned();

                let (mut c1, mut c2) = if crossover_method == 1 {
                    self.single_point_crossover(&parent1, &parent2)
                } else {
                    self.uniform_crossover(&parent1, &parent2)
                };
                self.bit_flip_mutation(&mut c1);
                self.bit_flip_mutation(&mut c2);

                let c1_fit = self.get_or_compute_fitness(&c1);
                let c2_fit = self.get_or_compute_fitness(&c2);

                if crowding {
                    self.crowding_replacement(p1_idx, p2_idx, &c1, &c2);
                } else {
                    child_population.slice_mut(ndarray::s![i, ..]).assign(&c1);
                    child_fitness[i] = c1_fit;
                    if i + 1 < self.population_size {
                        child_population.slice_mut(ndarray::s![i + 1, ..]).assign(&c2);
                        child_fitness[i + 1] = c2_fit;
                    }
                }

                i += 2;
            }

            if !crowding {
                self.elitism_selection(&mut child_population, &mut child_fitness);

                self.population = child_population;
                self.fitness = Arc::new(Mutex::new(child_fitness));
            }
            if !crowding {
                self.evaluate_population_parallel();
            }

            let fitness = self.fitness.lock().unwrap();
            let max_f = fitness
                .iter()
                .cloned()
                .fold(f64::MIN, f64::max);
            let min_f = fitness
                .iter()
                .cloned()
                .fold(f64::MAX, f64::min);
            let avg_f = fitness.iter().sum::<f64>() / self.population_size as f64;
            let ent = self.entropy();
            self.entropies.push(ent);

            println!(
                "Generation {}: max_f = {:.4}, avg_f = {:.4}, entropy = {:.4}",
                generation, max_f, avg_f, ent
            );
            self.fitnesses.push((generation, max_f, avg_f, min_f));

            if self.stop_flag.load(Ordering::SeqCst) {
                println!("Stop flag triggered after generation {}", generation);
                break;
            }
        }

        let best_idx = {
            let fitness = self.fitness.lock().unwrap();
            fitness
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        };
        self.population.slice(ndarray::s![best_idx, ..]).to_owned()
    }

    fn get_or_compute_fitness(&self, chromosome: &Array1<u8>) -> f64 {
        let key = chromosome.to_vec();
        {
            let map = self.cache.lock().unwrap();
            if let Some(val) = map.get(&key) {
                return *val;
            }
        }
        let val = self.evaluator.fitness(chromosome);
        {
            let mut map = self.cache.lock().unwrap();
            map.insert(key, val);
        }
        val
    }

    fn tournament_selection(&self, tournament_size: usize) -> usize {
        let mut rng = thread_rng();
        let mut best_index = 0;
        let mut best_fitness = f64::MIN;
        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..self.population_size);
            let fitness = self.fitness.lock().unwrap();
            if fitness[idx] > best_fitness {
                best_fitness = fitness[idx];
                best_index = idx;
            }
        }
        best_index
    }

    fn roulette_wheel_selection(&self) -> usize {
        let mut rng = thread_rng();
        let total_fitness: f64 = self.fitness.lock().unwrap().iter().sum();
        if total_fitness <= 0.0 {
            return rng.gen_range(0..self.population_size);
        }
        let mut accum = 0.0;
        let pick = rng.gen_range(0.0..total_fitness);
        for (idx, fit) in self.fitness.lock().unwrap().iter().enumerate() {
            accum += fit;
            if accum >= pick {
                return idx;
            }
        }
        self.population_size - 1
    }

    fn select_parents(&self, selection_method: usize, tournament_size: usize) -> (usize, usize) {
        if selection_method == 1 {
            let p1 = self.tournament_selection(tournament_size);
            let p2 = self.tournament_selection(tournament_size);
            (p1, p2)
        } else {
            let p1 = self.roulette_wheel_selection();
            let p2 = self.roulette_wheel_selection();
            (p1, p2)
        }
    }

    fn single_point_crossover(&self, p1: &Array1<u8>, p2: &Array1<u8>) -> (Array1<u8>, Array1<u8>) {
        let mut rng = thread_rng();
        if rng.gen_bool(self.crossover_rate) {
            let crossover_point = rng.gen_range(0..self.chromosome_length);
            let mut child1 = p1.clone();
            let mut child2 = p2.clone();
            for i in crossover_point..self.chromosome_length {
                child1[i] = p2[i];
                child2[i] = p1[i];
            }
            (child1, child2)
        } else {
            (p1.clone(), p2.clone())
        }
    }

    fn uniform_crossover(&self, p1: &Array1<u8>, p2: &Array1<u8>) -> (Array1<u8>, Array1<u8>) {
        let mut rng = thread_rng();
        if rng.gen_bool(self.crossover_rate) {
            let mut child1 = Array1::<u8>::zeros(self.chromosome_length);
            let mut child2 = Array1::<u8>::zeros(self.chromosome_length);
            for i in 0..self.chromosome_length {
                if rng.gen_bool(0.5) {
                    child1[i] = p1[i];
                    child2[i] = p2[i];
                } else {
                    child1[i] = p2[i];
                    child2[i] = p1[i];
                }
            }
            (child1, child2)
        } else {
            (p1.clone(), p2.clone())
        }
    }

    fn bit_flip_mutation(&self, chromosome: &mut Array1<u8>) {
        let mut rng = thread_rng();
        for i in 0..self.chromosome_length {
            if rng.gen_bool(self.mutation_rate) {
                chromosome[i] = 1 - chromosome[i];
            }
        }
    }

    fn elitism_selection(
        &mut self,
        child_population: &mut Array2<u8>,
        child_fitness: &mut Vec<f64>,
    ) {
        let elite_idx = {
            let fitness = self.fitness.lock().unwrap();
            fitness
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        };
        let elite_child_idx = child_fitness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if child_fitness[elite_child_idx] > *self.fitness.lock().unwrap().get(elite_idx).unwrap() {
            self.population
                .slice_mut(ndarray::s![elite_idx, ..])
                .assign(&child_population.slice(ndarray::s![elite_child_idx, ..]));
            let mut fitness = self.fitness.lock().unwrap();
            fitness[elite_idx] = child_fitness[elite_child_idx];
        }
    }

    /// Crowding replacement strategy
    /// 
    /// This method replaces the parents with the children if the children are better
    /// than the parents and the children are more dissimilar to the parents than the
    /// parents are to each other.
    /// 
    /// The dissimilarity is measured as the Hamming distance between two chromosomes.
    ///
    /// # Arguments
    /// p1_index: usize - Index of the first parent in the population
    /// p2_index: usize - Index of the second parent in the population
    /// child1: &Array1<u8> - First child chromosome
    /// child2: &Array1<u8> - Second child chromosome
    /// 
    /// # Returns
    /// None
    fn crowding_replacement(
        &mut self,
        p1_index: usize,
        p2_index: usize,
        child1: &Array1<u8>,
        child2: &Array1<u8>,
    ) {
        let parent1 = self.population.slice(ndarray::s![p1_index, ..]);
        let parent2 = self.population.slice(ndarray::s![p2_index, ..]);

        let d11 = hamming_distance(child1, &parent1);
        let d12 = hamming_distance(child1, &parent2);
        let d21 = hamming_distance(child2, &parent1);
        let d22 = hamming_distance(child2, &parent2);

        let c1_fit = self.get_or_compute_fitness(child1);
        let c2_fit = self.get_or_compute_fitness(child2);

        if d11 <= d12 {
            self.population
                .slice_mut(ndarray::s![p1_index, ..])
                .assign(child1);
            {
                let mut fitness = self.fitness.lock().unwrap();
                fitness[p1_index] = c1_fit;
            }
        } else {
            self.population
                .slice_mut(ndarray::s![p2_index, ..])
                .assign(child1);
            {
                let mut fitness = self.fitness.lock().unwrap();
                fitness[p2_index] = c1_fit;
            }
        }

        if d21 <= d22 {
            self.population
                .slice_mut(ndarray::s![p1_index, ..])
                .assign(child2);
            {
                let mut fitness = self.fitness.lock().unwrap();
                fitness[p1_index] = c2_fit;
            }
        } else {
            self.population
                .slice_mut(ndarray::s![p2_index, ..])
                .assign(child2);
            {
                let mut fitness = self.fitness.lock().unwrap();
                fitness[p2_index] = c2_fit;
            }
        }
    }

    /// Calculate the entropy of the population
    /// 
    /// This method calculates the entropy of the population by calculating the entropy
    /// of each column in the population and summing them.
    /// 
    /// # Arguments
    /// None
    /// 
    /// # Returns
    /// f64 - The entropy of the population
    fn entropy(&self) -> f64 {
        let mut total_entropy = 0.0;
        for col in 0..self.chromosome_length {
            let mut num_ones = 0.0;
            for row in 0..self.population_size {
                if self.population[[row, col]] == 1 {
                    num_ones += 1.0;
                }
            }
            let p = num_ones / self.population_size as f64;
            let p_clamped = p.max(1e-10).min(1.0 - 1e-10);
            let e = -p_clamped * p_clamped.log2()
                - (1.0 - p_clamped) * (1.0 - p_clamped).log2();
            total_entropy += e;
        }
        total_entropy
    }
}

/// Calculate the Hamming distance between two chromosomes
/// 
/// This function calculates the Hamming distance between two chromosomes.
/// The Hamming distance is the number of positions at which the two
/// chromosomes differ.
/// 
/// # Arguments
/// a: &Array1<u8> - First chromosome
/// b: &Array1<u8> - Second chromosome
/// 
/// # Returns
/// usize - The Hamming distance between the two chromosomes
fn hamming_distance(a: &Array1<u8>, b: &ndarray::ArrayView1<u8>) -> usize {
    let mut dist = 0;
    for i in 0..a.len() {
        if a[i] != b[i] {
            dist += 1;
        }
    }
    dist
}

pub struct FeatureSelection {
    pub x: Array2<f64>,
    pub y: Array1<f64>,
    pub linreg: LinReg,
}

impl FitnessEvaluator for FeatureSelection {
    fn fitness(&self, chromosome: &Array1<u8>) -> f64 {
        let columns: Vec<usize> = chromosome
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b == 1 { Some(i) } else { None })
            .collect();
        if columns.is_empty() {
            return f64::MIN;
        }
        let selected_x = select_columns(&self.x, &columns);
        let rmse = self.linreg.get_fitness(&selected_x, &self.y, Some(42));
        -rmse
    }
}

/// Select columns from a 2D array
/// 
/// This function selects columns from a 2D array based on the indices
/// provided in the `cols` slice.
/// 
/// # Arguments
/// x: &Array2<f64> - The 2D array from which to select columns
/// cols: &[usize] - The indices of the columns to select
/// 
/// # Returns
/// Array2<f64> - The 2D array containing only the selected columns
fn select_columns(x: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    x.select(ndarray::Axis(1), cols)
}

/// Read data from a CSV file
/// 
/// This function reads data from a CSV file and returns the input
/// features and target values as separate arrays.
/// 
/// # Arguments
/// file_path: &str - The path to the CSV file
/// 
/// # Returns
/// Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> - A tuple containing the input features
fn read_data(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut x = Vec::new();
    let mut y = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|x| x.parse::<f64>().unwrap()).collect();
        y.push(row[row.len() - 1]);
        x.push(row[..row.len() - 1].to_vec());
    }

    let x_shape = (x.len(), x[0].len());
    let x_flat: Vec<f64> = x.into_iter().flatten().collect();
    Ok((Array2::from_shape_vec(x_shape, x_flat)?, Array1::from(y)))
}


fn main() {
    println!("Reading data...");
    let (x, y) = read_data("data/dataset.txt").expect("Failed to read dataset");

    let fs_eval = FeatureSelection {
        x,
        y,
        linreg: LinReg::new(),
    };


    let mut ea = EASolver::new(100, 101, 0.01, 0.7, 1000, fs_eval);

    let best_features = ea.run(1, 5, 1, false);

    println!("Best feature selection: {:?}", best_features);
}
