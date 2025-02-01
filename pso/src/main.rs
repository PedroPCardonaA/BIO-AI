use std::f64::consts::PI;

use rand::Rng;

#[derive(Clone)]
pub struct PSO {
    pub pop_size: usize,
    pub chromosome_length: usize,
    pub max_gen: usize,
    pub best_individual: Vec<f64>,    
    pub best_fitness: f64,                
    pub population: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>, 
}

/// Particle Swarm Optimization (PSO) implementation.
///
/// # Attributes
/// - `pop_size`: Population size.
/// - `chromosome_length`: Length of each chromosome.
/// - `max_gen`: Maximum number of generations.
/// - `best_individual`: Best individual found.
/// - `best_fitness`: Fitness value of the best individual.
/// - `population`: Vector containing tuples of position, velocity, and personal best position.
///
/// # Methods
/// - `new(pop_size: usize, chromosome_length: usize, max_gen: usize) -> PSO`:
///   Creates a new PSO instance with the given population size, chromosome length, and maximum generations.
/// - `fitness(x: &Vec<f64>) -> f64`:
///   Calculates the fitness of a given chromosome.
/// - `run(&mut self)`:
///   Runs the PSO algorithm for the specified number of generations.
impl PSO {
    pub fn new(pop_size: usize, chromosome_length: usize, max_gen: usize) -> PSO {
        let mut rng = rand::rng();
        let mut population = Vec::with_capacity(pop_size);
        let mut best_individual = vec![0.0; chromosome_length];
        let mut best_fitness = std::f64::NEG_INFINITY;

        for _ in 0..pop_size {
            let x: Vec<f64> = (0..chromosome_length)
                .map(|_| rng.random_range(0.0..=1.0))
                .collect();
            let v: Vec<f64> = (0..chromosome_length)
                .map(|_| rng.random_range(-0.1..=0.1))
                .collect();
            let b = x.clone();
            let fit = PSO::fitness(&x);
            if fit > best_fitness {
                best_fitness = fit;
                best_individual = x.clone();
            }
            population.push((x, v, b));
        }

        PSO {
            pop_size,
            chromosome_length,
            max_gen,
            best_individual,
            best_fitness,
            population,
        }
    }

    pub fn fitness(x: &Vec<f64>) -> f64 {
        let mut sum = 0.0;
        for xi in x {
            let t = 10.0 * xi - 5.0;
            sum += t * t - 10.0 * (2.0 * PI * t).cos();
        }
        - (50.0 + sum)
    }

    pub fn run(&mut self) {
        let mut rng = rand::rng();
        let w = 0.7;
        let phi1 = 1.5;
        let phi2 = 1.5;

        for gen in 0..self.max_gen {
            for particle in self.population.iter_mut() {
                let (ref mut x, ref mut v, ref mut b) = particle;

                for i in 0..self.chromosome_length {
                    let r1: f64 = rng.random(); 
                    let r2: f64 = rng.random(); 
                    v[i] = w * v[i]
                        + phi1 * r1 * (b[i] - x[i])
                        + phi2 * r2 * (self.best_individual[i] - x[i]);
                    x[i] = x[i] + v[i];
                    if x[i] < 0.0 {
                        x[i] = 0.0;
                    }
                    if x[i] > 1.0 {
                        x[i] = 1.0;
                    }
                }

                let fit = PSO::fitness(x);
                if fit > PSO::fitness(b) {
                    *b = x.clone();
                }
                if fit > self.best_fitness {
                    self.best_fitness = fit;
                    self.best_individual = x.clone();
                }
            }
            println!("Generation {} best fitness: {}", gen + 1, self.best_fitness);
        }
    }
}

fn main() {
    let mut pso = PSO::new(100, 5, 1000);
    pso.run();
    println!("Global best fitness: {}", pso.best_fitness);
    println!("Global best individual: {:?}", pso.best_individual);

}
