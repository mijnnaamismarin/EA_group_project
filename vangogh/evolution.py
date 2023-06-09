import random
import time

import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

from vangogh import selection, variation
from vangogh.fitness import drawing_fitness_function, draw_voronoi_image
from vangogh.population import Population
from vangogh.util import NUM_VARIABLES_PER_POINT, IMAGE_SHRINK_SCALE, REFERENCE_IMAGE

from collections import defaultdict


class Evolution:
    def __init__(self,
                 num_points,
                 reference_image: Image,
                 evolution_type='p+o',
                 population_size=200,
                 generation_budget=-1,
                 evaluation_budget=-1,
                 crossover_method="ONE_POINT",
                 mutation_probability='inv_mutable_genotype_length',
                 num_features_mutation_strength=.25,
                 num_features_mutation_strength_decay=None,
                 num_features_mutation_strength_decay_generations=None,
                 selection_name='tournament_2',
                 initialization='RANDOM',
                 noisy_evaluations=False,
                 verbose=False,
                 generation_reporter=None,
                 learning_rate=0.1,
                 learning_rate_neg=0.075,
                 seed=0):

        self.probabilities = None
        self.reference_image: Image = reference_image.copy()
        self.reference_image.thumbnail((int(self.reference_image.width / IMAGE_SHRINK_SCALE),
                                        int(self.reference_image.height / IMAGE_SHRINK_SCALE)),
                                       Image.ANTIALIAS)
        self.reference_image_array = np.asarray(self.reference_image)

        num_variables = num_points * NUM_VARIABLES_PER_POINT
        feature_intervals = []
        for i in range(num_variables):
            if i % NUM_VARIABLES_PER_POINT == 0:  # X
                feature_intervals.append([0, self.reference_image.width])
            elif i % NUM_VARIABLES_PER_POINT == 1:  # Y
                feature_intervals.append([0, self.reference_image.height])
            else:  # color (RGBA)
                feature_intervals.append([0, 256])

        self.num_points = num_points
        self.feature_intervals = feature_intervals
        self.evolution_type = evolution_type
        self.population_size = population_size
        self.generation_budget = generation_budget
        self.evaluation_budget = evaluation_budget
        self.mutation_probability = mutation_probability
        self.num_features_mutation_strength = num_features_mutation_strength
        self.num_features_mutation_strength_decay = num_features_mutation_strength_decay
        self.num_features_mutation_strength_decay_generations = num_features_mutation_strength_decay_generations
        self.selection_name = selection_name
        self.noisy_evaluations = noisy_evaluations
        self.verbose = verbose
        self.generation_reporter = generation_reporter
        self.crossover_method = crossover_method
        self.num_evaluations = 0
        self.initialization = initialization
        self.learning_rate = learning_rate
        self.learning_rate_neg = learning_rate_neg

        np.random.seed(seed)
        self.seed = seed

        # set feature intervals to be a np.array
        if type(feature_intervals) != np.array:
            self.feature_intervals = np.array(feature_intervals, dtype=object)

        # check that tournament size is compatible
        if 'tournament' in selection_name:
            self.tournament_size = int(selection_name.split('_')[-1])
            if self.population_size % self.tournament_size != 0:
                raise ValueError('The population size must be a multiple of the tournament size')

        # set up population and elite
        self.genotype_length = len(feature_intervals)
        self.population = Population(self.population_size, self.genotype_length, self.initialization)
        self.elite = None
        self.elite_fitness = np.inf

        # set up mutation probability if set to default "inv_mutable_genotype_length"
        if mutation_probability == 'inv_genotype_length':
            self.mutation_probability = 1 / self.genotype_length
        elif mutation_probability == "inv_mutable_genotype_length":
            num_unmutable_features = 0
            self.mutation_probability = 1 / (self.genotype_length - num_unmutable_features)

            # incompatibilities
        if self.evolution_type == 'p+o' and self.noisy_evaluations:
            raise ValueError(
                "Using P+O is not compatible with noisy evaluations (you would need to re-evaluate the parents every generation, which is expensive)")
        elif 'age_reg' in self.evolution_type:
            print(
                "Warning: using noisy evaluations but age regularized evolution does not re-evaluate the entire population every generation")

    def __update_elite(self, population):
        best_fitness_idx = np.argmin(population.fitnesses)
        best_fitness = population.fitnesses[best_fitness_idx]
        if self.noisy_evaluations or best_fitness < self.elite_fitness:
            self.elite = population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

    def __classic_generation(self, merge_parent_offspring=False):
        # create offspring population
        # offspring = Population(self.population_size, self.genotype_length, self.initialization)###
        offspring = Population(self.population.genes.shape[0], self.genotype_length, self.initialization)  ###
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()
        # variation
        offspring.genes = variation.crossover(offspring.genes, self.crossover_method)
        offspring.genes = variation.mutate(offspring.genes, self.feature_intervals,
                                           mutation_probability=self.mutation_probability,
                                           num_features_mutation_strength=self.num_features_mutation_strength)
        # evaluate offspring
        offspring.fitnesses = drawing_fitness_function(offspring.genes,
                                                       self.reference_image)
        self.num_evaluations += len(offspring.genes)

        self.__update_elite(offspring)  ###jin

        # selection
        if merge_parent_offspring:
            # p+o mode
            self.population.stack(offspring)
        else:
            # just replace the entire thing
            self.population = offspring

        self.population = selection.select(self.population, self.population_size,
                                           selection_name=self.selection_name)

        # self.population.fitnesses = drawing_fitness_function(self.population.genes,
        #                                                      self.reference_image)  ###jin
        # self.__update_elite(self.population)  ###jin

    def __umda_generation(self):
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()

        for i in range(self.genotype_length):
            hist, bins = np.histogram(offspring.genes[:, i], bins=self.feature_intervals[i][1],
                                      range=(self.feature_intervals[i][0], self.feature_intervals[i][1]), density=True)
            distribution = (hist + 0.0001) / np.sum(hist + 0.0001)
            offspring.genes[:, i] = np.random.choice(np.arange(len(distribution)), size=self.population_size,
                                                     p=distribution).astype(int)

        offspring.fitnesses = drawing_fitness_function(offspring.genes, self.reference_image)

        self.num_evaluations += len(offspring.genes)

        self.__update_elite(offspring)

        self.population.stack(offspring)

        self.population = selection.select(self.population, self.population_size, selection_name=self.selection_name)

    def __pbil_generation(self):
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()

        offspring.fitnesses = drawing_fitness_function(offspring.genes, self.reference_image)

        self.__update_elite(offspring)

        parents = selection.select(offspring, self.population_size, selection_name=self.selection_name)

        for i in range(self.genotype_length):
            hist, bins = np.histogram(parents.genes[:, i], bins=self.feature_intervals[i][1],
                                      range=(self.feature_intervals[i][0], self.feature_intervals[i][1]), density=True)
            parent_probability = hist / np.sum(hist)

            self.probabilities[i] = (1.0 - self.learning_rate) * self.probabilities[i] \
                                    + self.learning_rate * parent_probability

            mut_shift = 0.05
            mut_prob = 0.02

            mask_mutate = np.random.choice([False, True], size=self.probabilities[i].shape,
                                           p=[1.0 - mut_prob, mut_prob])

            mutate_indices = np.arange(mask_mutate.size)[mask_mutate]

            mutation = np.random.randint(0, high=2, size=mutate_indices.size)

            self.probabilities[i][mutate_indices] = self.probabilities[i][mutate_indices] \
                                                    * (1.0 - mut_shift) + mutation * mut_shift

            self.probabilities[i] /= self.probabilities[i].sum()

            offspring.genes[:, i] = np.random.choice(np.arange(self.feature_intervals[i][1]), size=self.population_size,
                                                     p=self.probabilities[i]).astype(int)

        self.num_evaluations += len(offspring.genes)

        self.population = offspring

    def __pbil_paper_generation(self):
        offspring = Population(self.population_size, self.genotype_length, self.initialization)

        for i in range(self.population_size):
            for j in range(self.genotype_length):
                offspring.genes[i, j] = np.random.normal(self.probabilities[j], 30, 1)
                offspring.genes[i, j] = np.clip(offspring.genes[i, j],
                                                self.feature_intervals[j][0],
                                                self.feature_intervals[j][1])

        offspring.fitnesses = drawing_fitness_function(offspring.genes, self.reference_image)

        sort_order = np.argsort(offspring.fitnesses)
        offspring.genes = offspring.genes[sort_order, :]
        offspring.fitnesses = offspring.fitnesses[sort_order]

        for j in range(self.population_size):
            for i in range(self.genotype_length):
                self.probabilities[i] = self.probabilities[i] * (1.0 - self.learning_rate) + \
                                        offspring.genes[j][i] * self.learning_rate

        current_solution = Population(1, self.genotype_length, self.initialization)
        current_solution.genes[:] = self.probabilities.astype(int)
        current_solution.fitnesses = drawing_fitness_function(current_solution.genes, self.reference_image)
        self.__update_elite(current_solution)
        self.population = offspring

    def __pfda_generation(self):
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()

        rows = np.array([i for i in range(0, self.genotype_length, NUM_VARIABLES_PER_POINT)])
        x_row_indices = rows
        y_row_indices = rows + 1

        x_genes = offspring.genes[:, x_row_indices]
        y_genes = offspring.genes[:, y_row_indices]

        new_x_genes = np.zeros(x_genes.shape, dtype=np.int16)
        new_y_genes = np.zeros(y_genes.shape, dtype=np.int16)

        for i in range(x_genes.shape[1]):
            count_mat = np.zeros((self.reference_image.width, self.reference_image.height + 1))

            x_col, y_col = x_genes[:, i], y_genes[:, i]

            for x, y in zip(x_col, y_col):
                count_mat[x][y] += 1

            probs = (count_mat / x_genes.shape[0]).flatten()
            sampled_indices = np.random.choice(len(probs), size=x_genes.shape[0], p=probs)
            sampled_x = sampled_indices // self.reference_image.width
            sampled_y = sampled_indices % self.reference_image.height

            new_x_genes[:, i] = sampled_x
            new_y_genes[:, i] = sampled_y

        offspring.genes[:, x_row_indices] = new_y_genes
        offspring.genes[:, y_row_indices] = new_x_genes

        r_row_indices = rows + 2
        g_row_indices = rows + 3
        b_row_indices = rows + 4

        r_genes = offspring.genes[:, r_row_indices]
        g_genes = offspring.genes[:, g_row_indices]
        b_genes = offspring.genes[:, b_row_indices]

        new_r_genes = np.zeros(r_genes.shape, dtype=np.int16)
        new_g_genes = np.zeros(g_genes.shape, dtype=np.int16)
        new_b_genes = np.zeros(b_genes.shape, dtype=np.int16)

        for i in range(r_genes.shape[1]):
            r_col, g_col, b_col = r_genes[:, i], g_genes[:, i], b_genes[:, i]
            count_dict = {}
            for r, g, b in zip(r_col, g_col, b_col):
                key = str(r) + '|' + str(g) + '|' + str(b)
                count_dict[key] = count_dict.get(key, 0) + 1

            total_values = r_genes.shape[0]
            probs = {k: v / total_values for k, v in count_dict.items()}

            distribution = np.array(list(probs.values()))
            values = np.array(list(probs.keys()))
            sampled_indices = np.random.choice(np.arange(len(distribution)), size=r_genes.shape[0], p=distribution)
            sampled_values = values[sampled_indices]
            sampled_rgb = np.array(
                [[int(x.split('|')[0]), int(x.split('|')[1]), int(x.split('|')[2])] for x in sampled_values])

            sampled_r = sampled_rgb[:, 0]
            sampled_g = sampled_rgb[:, 1]
            sampled_b = sampled_rgb[:, 2]

            new_r_genes[:, i] = sampled_r
            new_g_genes[:, i] = sampled_g
            new_b_genes[:, i] = sampled_b

        offspring.genes[:, r_row_indices] = new_r_genes
        offspring.genes[:, g_row_indices] = new_g_genes
        offspring.genes[:, b_row_indices] = new_b_genes

        offspring.genes = variation.crossover(offspring.genes, self.crossover_method)
        offspring.genes = variation.mutate(offspring.genes, self.feature_intervals,
                                           mutation_probability=self.mutation_probability,
                                           num_features_mutation_strength=self.num_features_mutation_strength)

        offspring.fitnesses = drawing_fitness_function(offspring.genes, self.reference_image)

        self.num_evaluations += len(offspring.genes)

        self.__update_elite(offspring)

        self.population.stack(offspring)
        self.population = selection.select(self.population, self.population_size, selection_name=self.selection_name)

    def run(self, experiment_data):
        data = []
        self.population = Population(self.population_size, self.genotype_length, self.initialization)
        self.population.initialize(self.feature_intervals)

        self.population.fitnesses = drawing_fitness_function(self.population.genes,
                                                             self.reference_image)
        self.num_evaluations = len(self.population.genes)

        best_fitness_idx = np.argmin(self.population.fitnesses)
        best_fitness = self.population.fitnesses[best_fitness_idx]
        if best_fitness > self.elite_fitness:
            self.elite = self.population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

        start_time_seconds = time.time()

        # run generation_budget
        i_gen = 0
        while True:
            if self.num_features_mutation_strength_decay_generations is not None:
                if i_gen in self.num_features_mutation_strength_decay_generations:
                    self.num_features_mutation_strength *= self.num_features_mutation_strength_decay

            if self.evolution_type == 'classic':
                self.__classic_generation(merge_parent_offspring=False)
            elif self.evolution_type == 'PBIL':
                self.probabilities = np.empty(self.genotype_length, dtype=object)
                for i in range(self.genotype_length):
                    hist, bins = np.histogram(self.population.genes[:, i], bins=self.feature_intervals[i][1],
                                              range=(self.feature_intervals[i][0], self.feature_intervals[i][1]),
                                              density=True)
                    self.probabilities[i] = hist / np.sum(hist)

                self.__pbil_generation()
            elif self.evolution_type == 'UMDA':
                self.__umda_generation()
            elif self.evolution_type == 'cGA':
                self.probabilities = np.mean(self.population.genes, axis=0)
                self.__cga_generation()
            elif self.evolution_type == 'PFDA':
                self.__pfda_generation()
            elif self.evolution_type == 'p+o':
                self.__classic_generation(merge_parent_offspring=True)
            else:
                raise ValueError('unknown evolution type:', self.evolution_type)

            experiment_data.add_measurement(time.time() - start_time_seconds, self.elite_fitness)
            # generation terminated
            i_gen += 1
            if self.verbose:
                print('generation:', i_gen, 'best fitness:', self.elite_fitness, 'avg. fitness:',
                      np.mean(self.population.fitnesses))

            data.append({"num-generations": i_gen,
                         "num-evaluations": self.num_evaluations,
                         "time-elapsed": time.time() - start_time_seconds,
                         "best-fitness": self.elite_fitness,
                         "crossover-method": self.crossover_method,
                         "population-size": self.population_size, "num-points": self.num_points,
                         "initialization": self.initialization,
                         "seed": self.seed})
            if self.generation_reporter is not None:
                self.generation_reporter(
                    {"num-generations": i_gen, "num-evaluations": self.num_evaluations,
                     "time-elapsed": time.time() - start_time_seconds}, self)

            if 0 < self.generation_budget <= i_gen:
                break
            if 0 < self.evaluation_budget <= self.num_evaluations:
                break

            # check if evolution should terminate because optimum reached or population converged
            if self.population.is_converged():
                break

        draw_voronoi_image(self.elite, self.reference_image.width, self.reference_image.height,
                           scale=IMAGE_SHRINK_SCALE) \
            .save(
            f"./img/van_gogh_final_{self.seed}_{self.population_size}_{self.crossover_method}_{self.num_points}_{self.initialization}_{self.generation_budget}.png")
        return data

    def __cga_generation(self):
        pass
