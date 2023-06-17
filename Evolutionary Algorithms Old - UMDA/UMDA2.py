# Initialization: As with a standard GA, start by initializing a population of individuals.
#  Each individual's genotype will be a string of five integers representing the (x, y)
#  position and (r, g, b) color values of a point in the Voronoi diagram.

from vangogh.evolution import *
from vangogh.population import *
from vangogh.fitness import *
from vangogh.variation import *
from vangogh.selection import *
from vangogh.util import *
import numpy as np
import matplotlib.pyplot as plt

# Turn on interactive mode
plt.ion()

# Define the problem parameters
population_size = 200
num_points = 50
genotype_length = 5 * num_points
reference_image = Image.open(
	"/home/marin/Documents/University/EA/project/evolutionary-van-gogh/evolutionary-van-gogh/img/reference_image_resized.jpg")
width = reference_image.size[0]
height = reference_image.size[1]
feature_intervals = [(0, width), (0, height), (0, 255), (0, 255), (0, 255)] * num_points

# Initialize population
population = Population(population_size = population_size, genotype_length = genotype_length, initialization = "RANDOM")
population.initialize(feature_intervals = feature_intervals)
# print(population.genes)

# Evaluation: Use the imgcompare function to evaluate the fitness of each individual.
# This function compares the candidate image with the reference image (Van Gogh’s
# “Wheat Field with Cypresses”), the lower the difference, the better.
population.fitnesses = drawing_fitness_function(population.genes, reference_image)
# print(population.fitnesses)

# # draw best initial individual
# best_individual_index = np.argmin(population.fitnesses)
# best_individual = population.genes[best_individual_index, :]
# best_img = draw_voronoi_image(best_individual, width, height)
# best_img.save("best_initial_individual.png")


# Selection: Use tournament selection (as already implemented) to select a subset of
# individuals to serve as the parents for the next generation.

# Estimation of Distribution: Instead of performing crossover and mutation on the selected
# individuals, estimate the distribution of these promising solutions. For each of the
# five genes (x, y, r, g, b), calculate the marginal distribution. This could be as
# simple as finding the mean and standard deviation of each gene's values in the
# selected individuals.

# Generation of New Individuals: Sample from the estimated distributions to generate
# new individuals. For each gene in each new individual, draw a value from the
# corresponding distribution. This serves as the equivalent of crossover and mutation
# in a standard GA.

# Replacement: Combine the new individuals with the existing population.
# You could entirely replace the old population with the new individuals,
# or you could keep the best individuals from the old population.

# Iteration: Repeat the Evaluation, Selection, Estimation of Distribution,
# Generation of New Individuals, and Replacement steps until a stopping condition
# is met (e.g., maximum number of generations, fitness threshold).
