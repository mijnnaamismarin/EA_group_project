import numpy as np
from multiprocess import Pool, cpu_count
import scipy as sp
from vangogh.fitness import individual_fitness


def worker_2(args, tol = 5):
	opt = sp.optimize.minimize(individual_fitness, x0 = np.array(args[2]), tol = tol, args = (np.array(args[1])),
							   method = 'Nelder-Mead')
	print("MIN DONE")
	print(opt.nfev)
	return [opt.x, opt.nfev]


# Seperate pixel locations from pixel colours
def pixel_seperate(genes, num_points):
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


def find_local_max(genes, num_points, opt_pop):
	pixel_loc, pixel_colour = pixel_seperate(genes[0:opt_pop], num_points)
	ziplet = list(zip(range(pixel_loc.shape[0]), pixel_loc.tolist(), pixel_colour.tolist()))
	with Pool(min(max(cpu_count() - 1, 1), 4)) as p:
		result = p.map(worker_2, ziplet)
	return result


class Population:
	def __init__(self, population_size, genotype_length, initialization):
		self.genes = np.empty(shape = (population_size, genotype_length), dtype = int)
		self.fitnesses = np.zeros(shape = (population_size,))
		self.initialization = initialization

	def initialize(self, feature_intervals):
		n = self.genes.shape[0]
		l = self.genes.shape[1]
		num_points = 100
		if self.initialization == "RANDOM":
			for i in range(l):
				init_feat_i = np.random.randint(low = feature_intervals[i][0],
												high = feature_intervals[i][1], size = n)
				self.genes[:, i] = init_feat_i
		elif self.initialization == "PARTIAL_LOCAL_OPT":
			opt_fraction = 1 / 5
			opt_pop = round(n * opt_fraction)
			for i in range(l):
				init_feat_i = np.random.randint(low = feature_intervals[i][0],
												high = feature_intervals[i][1], size = n)
				self.genes[:, i] = init_feat_i
			pixel_loc, pixel_colour = pixel_seperate(self.genes[0:opt_pop, :], num_points)
			result = find_local_max(self.genes[0:opt_pop, :], num_points, opt_pop)
			refined_colours = np.zeros((len(result), int(3 * l / 5)))
			evals = np.zeros((len(result), 1))
			for i in range(len(result)):
				refined_colours[i] = result[i][0]
				evals[i] = result[i][1]
			self.genes[0:opt_pop, :] = np.array(pixel_join(pixel_loc, refined_colours))
		else:
			raise Exception("Unknown initialization method")

	def stack(self, other):
		self.genes = np.vstack((self.genes, other.genes))
		self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

	def shuffle(self):
		random_order = np.random.permutation(self.genes.shape[0])
		self.genes = self.genes[random_order, :]
		self.fitnesses = self.fitnesses[random_order]

	def is_converged(self):
		return len(np.unique(self.genes, axis = 0)) < 2

	def delete(self, indices):
		self.genes = np.delete(self.genes, indices, axis = 0)
		self.fitnesses = np.delete(self.fitnesses, indices)
