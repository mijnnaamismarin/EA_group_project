from vangogh.evolution import *
from vangogh.population import *
from vangogh.fitness import *
from vangogh.variation import *
from vangogh.selection import *
from vangogh.util import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	plt.ion()  # Turn on interactive mode for plt

	# Problem parameters
	population_size = 2000
	num_points = 100
	genotype_length = 5 * num_points
	reference_image = Image.open("img/reference_image_resized.jpg")
	width, height = reference_image.size
	feature_intervals = [(0, width), (0, height), (0, 255), (0, 255), (0, 255)] * num_points

	# Initialize population
	population = Population(population_size = population_size, genotype_length = genotype_length,
							initialization = "RANDOM")
	population.initialize(feature_intervals = feature_intervals)

	# Selection parameters
	selection_size = 1750
	tournament_size = 8
	num_generations = 300


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


	# Main loop for evolution process
	for generation in range(num_generations):
		# Fitness computation
		population.fitnesses = drawing_fitness_function(population.genes, reference_image)

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
