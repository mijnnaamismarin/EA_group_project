import os
import pickle

import numpy as np
import scipy as sp

from vangogh.fitness import individual_fitness
from multiprocessing import Pool, cpu_count


# Class to represent a population
class Population:
    def __init__(self, population_size, genotype_length, initialization, opt_fraction=0.2):
        # Initialize the population
        self.genes = np.empty(shape=(population_size, genotype_length), dtype=int)
        self.fitnesses = np.zeros(shape=(population_size,))
        self.initialization = initialization
        self.opt_fraction = opt_fraction

    # Method to initialize the population
    def initialize(self, feature_intervals):
        # Number of genes and length of genes
        n = self.genes.shape[0]
        l = self.genes.shape[1]

        num_points = 100
        pickle_file_name = f"result_{self.opt_fraction}.pickle"

        # Random initialization
        if self.initialization == "RANDOM":
            for i in range(l):
                init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                                high=feature_intervals[i][1], size=n)

                self.genes[:, i] = init_feat_i

        # Partial local optimum initialization
        elif self.initialization == "PARTIAL_LOCAL_OPT":
            # Calculate the size of the optimum population
            optimal_pop_size = round(n * self.opt_fraction)

            # Initialize genes randomly within the provided feature intervals
            for feature_index in range(l):
                random_feature_values = np.random.randint(low=feature_intervals[feature_index][0],
                                                          high=feature_intervals[feature_index][1],
                                                          size=n)
                self.genes[:, feature_index] = random_feature_values

            # Separate pixel locations and colors for the optimum part of the population
            pixel_locations, pixel_colors = pixel_seperate(self.genes[0:optimal_pop_size, :], num_points)

            # Find local maxima for the optimum part of the population
            local_maxima_results = find_local_max(self.genes[0:optimal_pop_size, :], num_points, optimal_pop_size)

            # Initialize arrays for storing refined colors and evaluation results
            refined_colors = np.zeros((len(local_maxima_results), int(3 * l / 5)))
            eval_results = np.zeros((len(local_maxima_results), 1))

            # Process the local maxima results and store the refined colors and evaluation results
            for index, result in enumerate(local_maxima_results):
                refined_colors[index] = result[0]
                eval_results[index] = result[1]
                print(f"Result refining progress: {index + 1}/{len(local_maxima_results)}")

            # Combine the pixel locations with the refined colors
            self.genes[0:optimal_pop_size, :] = np.array(pixel_join(pixel_locations, refined_colors))

            # Save the results using pickle
            with open(pickle_file_name, 'wb') as handle:
                pickle.dump(self.genes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Load from pickle file
        elif self.initialization == "PARTIAL_LOCAL_OPT_LOAD":
            if os.path.isfile(pickle_file_name):
                with open(pickle_file_name, 'rb') as handle:
                    self.genes = pickle.load(handle)
            elif os.path.isfile("result_0.2.pickle"):
                print("opt_fraction did not match, loading result_0.2.pickle")
                with open("result_0.2.pickle", 'rb') as handle:
                    self.genes = pickle.load(handle)
            else:
                raise Exception("No pickle file found")
        else:
            raise Exception("Unknown initialization method")

    # Other methods in the Population class
    def stack(self, other):
        self.genes = np.vstack((self.genes, other.genes))
        self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

    def shuffle(self):
        random_order = np.random.permutation(self.genes.shape[0])
        self.genes = self.genes[random_order, :]
        self.fitnesses = self.fitnesses[random_order]

    def is_converged(self):
        return len(np.unique(self.genes, axis=0)) < 2

    def delete(self, indices):
        self.genes = np.delete(self.genes, indices, axis=0)
        self.fitnesses = np.delete(self.fitnesses, indices)


# Function for worker processes in parallel computing
def worker_2(args, tol=5):
    # Using scipy's optimize function to minimize the individual fitness
    opt = sp.optimize.minimize(individual_fitness, x0=np.array(args[2]), tol=tol, args=(np.array(args[1])),
                               method='Nelder-Mead')

    # Print status messages
    print("MIN DONE")
    print(opt.nfev)

    # Return optimization result
    return [opt.x, opt.nfev]


# Function to separate pixel locations from pixel colours
def pixel_seperate(genes, num_points):
    pixel_loc = np.zeros((len(genes), 2 * num_points))
    pixel_colour = np.zeros((len(genes), 3 * num_points))

    # Separate the pixel locations and colors
    for i in range(len(genes)):
        reshaped_genes = genes[i].reshape((-1, 5))
        split_genes = np.hsplit(reshaped_genes, [2])

        pixel_loc[i] = split_genes[0].reshape(-1)
        pixel_colour[i] = split_genes[1].reshape(-1)

    return pixel_loc, pixel_colour


# Function to join pixel locations with pixel colours
def pixel_join(pixel_loc, pixel_colour):
    # Prepare an empty numpy array for the genes
    genes = np.zeros((len(pixel_loc), len(pixel_loc[0]) + len(pixel_colour[0])))

    # Combine the pixel locations and colors
    for i in range(len(pixel_loc)):
        reshaped_locations = pixel_loc[i].reshape((-1, 2))
        reshaped_colors = pixel_colour[i].reshape((-1, 3))

        genes[i] = np.hstack((reshaped_locations, reshaped_colors)).reshape(-1)

    return genes


# Function to find the local maximum
def find_local_max(genes, num_points, opt_pop):
    pixel_loc, pixel_colour = pixel_seperate(genes[0:opt_pop], num_points)
    ziplet = list(zip(range(pixel_loc.shape[0]), pixel_loc.tolist(), pixel_colour.tolist()))

    # Process in parallel using multiple cores
    with Pool(min(max(cpu_count() - 1, 1), 4)) as p:
        result = p.map(worker_2, ziplet)

    return result
