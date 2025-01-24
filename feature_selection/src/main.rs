use rand::prelude::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::error::Error;
use std::f64;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};
use std::thread;
use csv::ReaderBuilder;
use std::fs::File;
mod lin_reg;
use lin_reg::LinReg;


/// ------------------ FitnessEvaluator Trait ------------------ ///
pub trait FitnessEvaluator: Send + Sync {
    fn fitness(&self, chromosome: &Array1<u8>) -> f64;
}

/// ------------------ Example: FeatureSelection struct as Evaluator ------------------ ///
#[derive(Clone)]
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

        // If no features selected, punish heavily
        if columns.is_empty() {
            return f64::MIN;
        }

        let selected_x = select_columns(&self.x, &columns);
        let score = -1.0*self.linreg.get_fitness(&selected_x, &self.y, Some(42));
        score // e.g. -RMSE in real usage
    }
}

/// Helper: select columns
fn select_columns(x: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    x.select(ndarray::Axis(1), cols)
}

/// ------------------ The Parallel EA Solver (Island Model) ------------------ ///
pub struct EASolver<F: FitnessEvaluator + 'static> {
    pub population_size: usize,
    pub chromosome_length: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub max_generations: usize,

    // Each solver has its own population/fitness
    pub population: Array2<u8>,
    pub fitness: Vec<f64>,

    // Shared read/write cache of (chromosome -> fitness)
    cache: Arc<RwLock<HashMap<Vec<u8>, f64>>>,

    // Shared evaluation code
    evaluator: Arc<F>,

    // A shared stop flag
    stop_flag: Arc<AtomicBool>,

    // Optional: store a "goal" fitness threshold
    pub goal_fitness: f64,
}

impl<F: FitnessEvaluator> EASolver<F> {
    pub fn new(
        population_size: usize,
        chromosome_length: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        max_generations: usize,
        evaluator: Arc<F>,            // pass evaluator by Arc
        cache: Arc<RwLock<HashMap<Vec<u8>, f64>>>,
        stop_flag: Arc<AtomicBool>,
        goal_fitness: f64,
    ) -> Self {
        // Randomly initialize a population
        let mut rng = thread_rng();
        let mut population = Array2::<u8>::zeros((population_size, chromosome_length));
        for i in 0..population_size {
            for j in 0..chromosome_length {
                if rng.gen_bool(0.5) {
                    population[[i, j]] = 1;
                }
            }
        }
        let fitness = vec![0.0; population_size];

        EASolver {
            population_size,
            chromosome_length,
            mutation_rate,
            crossover_rate,
            max_generations,
            population,
            fitness,
            cache,
            evaluator,
            stop_flag,
            goal_fitness,
        }
    }

    /// Main evolutionary loop: runs until max_generations or the stop_flag is triggered.
    pub fn run(&mut self) -> Array1<u8> {
        // Evaluate the population before starting
        self.evaluate_population();

        for generation in 0..self.max_generations {
            // Stop if another thread or ourselves found a solution
            if self.stop_flag.load(Ordering::SeqCst) {
                println!("Thread: stopping early (stop_flag) at gen {}", generation);
                break;
            }

            // Create a child population
            let mut child_population = Array2::<u8>::zeros((self.population_size, self.chromosome_length));
            let mut child_fitness = vec![0.0; self.population_size];

            let mut i = 0;
            while i < self.population_size {
                let (p1_idx, p2_idx) = self.select_parents_tournament(5);
                let parent1 = self.population.slice(ndarray::s![p1_idx, ..]).to_owned();
                let parent2 = self.population.slice(ndarray::s![p2_idx, ..]).to_owned();

                let (mut c1, mut c2) = self.single_point_crossover(&parent1, &parent2);
                self.bit_flip_mutation(&mut c1);
                self.bit_flip_mutation(&mut c2);

                let c1_fit = self.get_or_compute_fitness(&c1);
                let c2_fit = self.get_or_compute_fitness(&c2);

                // Insert them into the child pop
                child_population.slice_mut(ndarray::s![i, ..]).assign(&c1);
                child_fitness[i] = c1_fit;

                if i + 1 < self.population_size {
                    child_population.slice_mut(ndarray::s![i + 1, ..]).assign(&c2);
                    child_fitness[i + 1] = c2_fit;
                }

                i += 2;
            }

            // Replace with new generation
            self.population = child_population;
            self.fitness = child_fitness;

            // Recompute or keep fitness. We'll just keep it from above
            // But if you'd like, you can call evaluate_population() again

            // Check if we reached the goal
            let (best_fitness, _) = self
                .fitness
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, &fit)| (fit, i))
                .unwrap();

            if best_fitness >= self.goal_fitness {
                println!(
                    "[Thread] Found a solution with fitness {:.6} at generation {}. Setting stop flag.",
                    best_fitness, generation
                );
                // Signal other threads to stop
                self.stop_flag.store(true, Ordering::SeqCst);
                break;
            }

            if generation % 3 == 0 {
                println!("[Thread] Generation {}: best_fitness = {:.6}", generation, best_fitness);
            }
        }

        // Return best solution
        let best_idx = self
            .fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        self.population.slice(ndarray::s![best_idx, ..]).to_owned()
    }

    /// Evaluate an entire population (e.g. at the start)
    fn evaluate_population(&mut self) {
        for i in 0..self.population_size {
            if self.stop_flag.load(Ordering::SeqCst) {
                break;
            }
            let chromosome = self.population.slice(ndarray::s![i, ..]).to_owned();
            let fit = self.get_or_compute_fitness(&chromosome);
            self.fitness[i] = fit;

            // If any meets the goal, set stop flag
            if fit >= self.goal_fitness {
                self.stop_flag.store(true, Ordering::SeqCst);
                break;
            }
        }
    }

    /// Returns the fitness if present in cache, otherwise computes and inserts it.
    fn get_or_compute_fitness(&self, chromosome: &Array1<u8>) -> f64 {
        let key = chromosome.to_vec();

        // 1) Try read-lock first
        {
            let map_read = self.cache.read().unwrap();
            if let Some(&val) = map_read.get(&key) {
                return val;
            }
        }

        // 2) Not found => compute
        let val = self.evaluator.fitness(chromosome);

        // 3) Write-lock to insert
        {
            let mut map_write = self.cache.write().unwrap();
            map_write.insert(key, val);
        }

        val
    }

    /// Example: Tournament selection
    fn select_parents_tournament(&self, t_size: usize) -> (usize, usize) {
        let p1 = self.tournament_selection(t_size);
        let p2 = self.tournament_selection(t_size);
        (p1, p2)
    }

    fn tournament_selection(&self, t_size: usize) -> usize {
        let mut rng = thread_rng();
        let mut best_index = 0;
        let mut best_fitness = f64::MIN;
        for _ in 0..t_size {
            let idx = rng.gen_range(0..self.population_size);
            if self.fitness[idx] > best_fitness {
                best_fitness = self.fitness[idx];
                best_index = idx;
            }
        }
        best_index
    }

    /// Single-point crossover
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

    /// Bit-flip mutation
    fn bit_flip_mutation(&self, chromosome: &mut Array1<u8>) {
        let mut rng = thread_rng();
        for i in 0..self.chromosome_length {
            if rng.gen_bool(self.mutation_rate) {
                chromosome[i] = 1 - chromosome[i];
            }
        }
    }
}

/// ------------------ CSV Reading Example ------------------ ///
fn read_data(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut x = Vec::new();
    let mut y = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|field| field.parse().unwrap()).collect();
        y.push(row[row.len() - 1]);
        x.push(row[..row.len() - 1].to_vec());
    }
    let x_shape = (x.len(), x[0].len());
    let x_flat: Vec<f64> = x.into_iter().flatten().collect();

    Ok((Array2::from_shape_vec(x_shape, x_flat)?, Array1::from(y)))
}

/// ------------------ MAIN: Spawn multiple threads, each with its own EA ------------------ ///
fn main() {
    // Read data
    let (x, y) = read_data("data/dataset.txt").expect("Failed to read dataset");

    // Build the shared FeatureSelection evaluator
    // (wrap in Arc so we can clone references in each thread)
    let fs_eval = Arc::new(FeatureSelection {
        x,
        y,
        linreg: LinReg::new(),
    });

    // Global shared cache (chromosome -> fitness). Use RwLock for concurrency
    let cache = Arc::new(RwLock::new(HashMap::new()));

    let stop_flag = Arc::new(AtomicBool::new(false));

    let population_size = 100;
    let chromosome_length = 101;
    let mutation_rate = 0.01;
    let crossover_rate = 0.7;
    let max_generations = 1000;
    let goal_fitness = -0.125; 

    // Number of threads (islands)
    let num_threads = 10;

    // Spawn multiple threads, each with its own EASolver instance
    let mut handles = Vec::new();
    for t_id in 0..num_threads {
        let evaluator_clone = fs_eval.clone();
        let cache_clone = cache.clone();
        let flag_clone = stop_flag.clone();

        let handle = thread::spawn(move || {
            // Create a local solver for this thread
            let mut solver = EASolver::new(
                population_size,
                chromosome_length,
                mutation_rate,
                crossover_rate,
                max_generations,
                evaluator_clone,
                cache_clone,
                flag_clone,
                goal_fitness,
            );
            println!("Thread {} starting run() ...", t_id);
            let best_chromosome = solver.run();
            println!("Thread {} finished. Best chromosome found = {:?}", t_id, best_chromosome);
            best_chromosome
        });

        handles.push(handle);
    }

    // Join all threads, collecting their best results
    for handle in handles {
        match handle.join() {
            Ok(best) => {
                println!("A thread returned best chromosome = {:?}", best);
            }
            Err(e) => {
                eprintln!("A thread panicked: {:?}", e);
            }
        }
    }

    println!("All threads completed.");
}
