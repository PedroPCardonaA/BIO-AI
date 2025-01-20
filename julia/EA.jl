using Random
using Statistics
using LinearAlgebra
using DataFrames
using CSV
using VegaLite

abstract type EvolutionaryAlgorithm end

mutable struct EA <: EvolutionaryAlgorithm
    population_size::Int
    chromosome_length::Int
    mutation_rate::Float64
    crossover_rate::Float64
    max_generations::Int
    population::Matrix{Int}
    fitness::Vector{Float64}
    entropies::Vector{Float64}
    fitnesses::Vector{Vector{Float64}}

    function EA(population_size, chromosome_length, mutation_rate, crossover_rate, max_generations)
        population = rand(0:1, population_size, chromosome_length)
        fitness = zeros(Float64, population_size)
        new(population_size, chromosome_length, mutation_rate, crossover_rate, max_generations, population, fitness, Float64[], [])
    end
end

function tournament_selection(ea::EA, tournament_size)
    shuf = shuffle(1:ea.population_size)
    competitors = shuf[1:tournament_size]
    fitness_values = ea.fitness[competitors]
    return competitors[argmax(fitness_values)]
end

function roulette_wheel_selection(ea::EA)
    total_fitness = sum(ea.fitness)
    probabilities = ea.fitness ./ total_fitness
    return findfirst(rand() .< cumsum(probabilities))
end

function single_point_crossover(parent1, parent2, chromosome_length, crossover_rate)
    if rand() < crossover_rate
        point = rand(1:chromosome_length)
        return vcat(parent1[1:point], parent2[point+1:end]),
               vcat(parent2[1:point], parent1[point+1:end])
    else
        return parent1, parent2
    end
end

function uniform_crossover(parent1, parent2, chromosome_length, crossover_rate)
    if rand() < crossover_rate
        mask = rand(0:1, chromosome_length)
        child1 = mask .* parent1 + (1 .- mask) .* parent2
        child2 = mask .* parent2 + (1 .- mask) .* parent1
        return child1, child2
    else
        return parent1, parent2
    end
end

function bit_flip_mutation(chromosome, mutation_rate)
    mask = rand(Float64, length(chromosome)) .< mutation_rate
    mutated = copy(chromosome)
    for i in eachindex(mutated)
        if mask[i]
            mutated[i] = 1 - mutated[i]
        end
    end
    return mutated
end

function elitism_selection!(ea::EA, child_population, child_fitness)
    elite_index = argmax(ea.fitness)
    best_child_index = argmax(child_fitness)
    if child_fitness[best_child_index] > ea.fitness[elite_index]
        ea.population[elite_index, :] = child_population[best_child_index, :]
        ea.fitness[elite_index] = child_fitness[best_child_index]
    end
end

function calculate_entropy(ea::EA)
    num_ones = sum(ea.population, dims=1)
    probabilities = num_ones ./ ea.population_size
    probabilities = clamp.(probabilities, 1e-10, 1 .- 1e-10)
    entropies = -probabilities .* log2.(probabilities) .- (1 .- probabilities) .* log2.(1 .- probabilities)
    return sum(entropies)
end

function plot_entropy(entropies)
    data = DataFrame(x=1:length(entropies), y=entropies)
    data |> @vlplot(:line, x="x", y="y", title="Entropy Over Generations") |> save("entropy.png")
end

function plot_fitness(fitnesses)
    # Create a DataFrame from the fitness data
    data = DataFrame(
        generation = repeat(1:length(fitnesses), inner=3),
        fitness = vcat([f[2] for f in fitnesses], [f[3] for f in fitnesses], [f[4] for f in fitnesses]),
        category = repeat(["Max Fitness", "Avg Fitness", "Min Fitness"], outer=length(fitnesses))
    )

    # Plot using VegaLite
    data |>
    @vlplot(
        :line,
        x = "generation:o",
        y = "fitness:q",
        color = "category:n",
        title = "Fitness Over Generations"
    ) |>
    save("fitness.png")
end

mutable struct KnapsackProblem <: EvolutionaryAlgorithm
    ea::EA
    weights::Vector{Float64}
    values::Vector{Float64}
    capacity::Float64
end

function fitness_function(kp::KnapsackProblem, chromosome)
    total_weight = sum(kp.weights .* chromosome)
    total_value = sum(kp.values .* chromosome)
    return total_weight > kp.capacity ? 0 : total_value
end

function run_knapsack(kp::KnapsackProblem)
    ea = kp.ea
    for generation in 1:ea.max_generations
        child_population = zeros(Int, ea.population_size, ea.chromosome_length)
        child_fitness = zeros(Float64, ea.population_size)

        for i in 1:2:ea.population_size
            parent1_idx = tournament_selection(ea, 3)
            parent2_idx = tournament_selection(ea, 3)

            parent1 = ea.population[parent1_idx, :]
            parent2 = ea.population[parent2_idx, :]

            child1, child2 = single_point_crossover(parent1, parent2, ea.chromosome_length, ea.crossover_rate)

            child1 = bit_flip_mutation(child1, ea.mutation_rate)
            child2 = bit_flip_mutation(child2, ea.mutation_rate)

            child_population[i, :] = child1
            child_population[i + 1, :] = child2

            child_fitness[i] = fitness_function(kp, child1)
            child_fitness[i + 1] = fitness_function(kp, child2)
        end

        elitism_selection!(ea, child_population, child_fitness)
        ea.population = child_population
        ea.fitness = child_fitness

        push!(ea.entropies, calculate_entropy(ea))
        push!(ea.fitnesses, [generation, maximum(ea.fitness), mean(ea.fitness), minimum(ea.fitness)])

        if generation % 10 == 0
            println("Generation: $generation, Max Fitness: $(maximum(ea.fitness))")
        end
    end

    plot_entropy(ea.entropies)
    plot_fitness(ea.fitnesses)
    return ea.population[argmax(ea.fitness), :]
end

function knapsack_run()
    df = CSV.read("data/knapPI_12_500_1000_82.csv", DataFrame)
    weights = df.w
    values = df.p
    capacity = 280785

    ea = EA(100, length(weights), 1 / length(weights), 0.7, 100)
    kp = KnapsackProblem(ea, weights, values, capacity)
    run_knapsack(kp)
end

knapsack_run()
