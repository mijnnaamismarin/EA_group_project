from vangogh.evolution import *
from vangogh.population import *
from vangogh.fitness import *
from vangogh.variation import *
from vangogh.selection import *
from vangogh.util import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from multiprocess import Pool, cpu_count
from time import time
from tqdm import tqdm


def worker_2(args, tol = 5):
    opt = sp.optimize.minimize(individual_fitness, x0 = np.array(args[2]), tol = tol, args = (np.array(args[1])))
    print("MIN DONE")
    print(opt.nfev)
    return [opt.x, opt.nfev]


def find_local_max(initial_population):
    pixel_loc, pixel_colour = pixel_seperate(initial_population.genes)
    ziplet = list(zip(range(pixel_loc.shape[0]), pixel_loc.tolist(), pixel_colour.tolist()))
    with Pool(min(max(cpu_count() - 1, 1), 4)) as p:
        result = p.map(worker_2, ziplet)
    return result


# Seperate pixel locations from pixel colours
def pixel_seperate(genes):
    pixel_loc = np.zeros((len(genes), 2 * num_points))
    pixel_colour = np.zeros((len(genes), 3 * num_points))
    for i in range(len(genes)):
        a = genes[i].reshape((-1, 5))
        b = np.hsplit(a, [2])
        pixel_loc[i] = b[0].reshape(-1)
        pixel_colour[i] = b[1].reshape(-1)
    return pixel_loc, pixel_colour


# Join pixel locations with pixel colours
def pixel_join(pixel_loc, pixel_colour):
    genes = np.zeros((len(pixel_loc), len(pixel_loc[0]) + len(pixel_colour[0])))
    for i in range(len(pixel_loc)):
        a = pixel_loc[i].reshape((-1, 2))
        b = pixel_colour[i].reshape((-1, 3))
        genes[i] = np.hstack((a, b)).reshape(-1)
    return genes


# Function to estimate probabilities of the selected population
def estimate_probabilities(selected_population):
    population_size, genotype_length = selected_population.genes.shape
    probabilities = []
    gene_values_list = []

    for i in range(genotype_length):
        gene_values, counts = np.unique(selected_population.genes[:, i], return_counts = True)
        probabilities.append(counts / population_size)
        gene_values_list.append(gene_values)

    return probabilities, gene_values_list


# Function to sample new population based on probabilities
def sample_new_population(probabilities, gene_values_list, population_size, feature_intervals, generation,
                          total_generations):
    genotype_length = len(probabilities)
    new_population = Population(population_size, genotype_length, "RANDOM")
    new_population.initialize(feature_intervals = feature_intervals)

    decay_factor = max(0.0, np.exp(-generation / (total_generations - 50)))

    for i in range(genotype_length):
        sampled_genes = np.random.choice(gene_values_list[i], size = population_size, p = probabilities[i])
        new_population.genes[:, i] = sampled_genes
        new_population.genes[:, i] = np.round(new_population.genes[:, i]).astype(int)

    return new_population


if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode for plt

    # Problem parameters
    population_size = 500
    num_points = 100
    genotype_length = 5 * num_points
    reference_image = Image.open("img/reference_image_resized.jpg")
    width, height = reference_image.size
    feature_intervals = [(0, width), (0, height), (0, 255), (0, 255), (0, 255)] * num_points

    # Initialize population
    population = Population(population_size = population_size, genotype_length = genotype_length,
                            initialization = "RANDOM")
    population.initialize(feature_intervals = feature_intervals)
    pixel_loc, pixel_colour = pixel_seperate(population.genes)
    t1_start = time()
    result = find_local_max(population)
    refined_colours = np.zeros((len(result), 3 * num_points))
    evals = np.zeros((len(result), 1))
    for i in range(len(result)):
        refined_colours[i] = result[i][0]
        evals[i] = result[i][1]

    population.genes = np.array(pixel_join(pixel_loc, refined_colours))
    t1_stop = time()
    print("Initialization complete")
    print("Elapsed time during the initialization:",
          t1_stop - t1_start)
    print(f"Total number of evaluations: {np.sum(evals)}")
    # Selection parameters
    selection_size = 440
    tournament_size = 4
    num_generations = 300

    # Main loop for evolution process
    for generation in tqdm(range(num_generations)):
        # Fitness computation
        population.fitnesses = drawing_fitness_function(population.genes)

        # Selection process
        selected_population = select(population, selection_size, f'tournament_{tournament_size}')

        # Estimating probabilities
        probabilities, gene_values = estimate_probabilities(selected_population)

        # Sampling new population
        new_population = sample_new_population(probabilities, gene_values, population_size, feature_intervals,
                                               generation, num_generations)

        # Every 10 generations, print fitness and display best image
        if generation % 10 == 0:
            print(f"Generation {generation}: best fitness = {np.min(population.fitnesses)}")

            if generation > 0:
                plt.close()

            best_individual = population.genes[np.argmin(population.fitnesses)]
            best_image = draw_voronoi_image(best_individual, reference_image.width, reference_image.height)
            best_image.save(f"./output/best_image_{generation}.png")

            plt.imshow(best_image)
            plt.show(block = False)
            plt.pause(0.001)

        # Replacement of population
        population = new_population

    # Fitness computation for the last population and display final image
    population.fitnesses = drawing_fitness_function(population.genes, reference_image)
    best_individual = population.genes[np.argmin(population.fitnesses)]
    final_image = draw_voronoi_image(best_individual, reference_image.width, reference_image.height)
    final_image.save("./final_image.png")
    final_image.show()
