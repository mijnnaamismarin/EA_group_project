import itertools
import numpy as np

from matplotlib import pyplot as plt
from time import time

from vangogh.experiment_module.experiment_data import ExperimentData


class Experiment:
    """
    Class representing an experiment.

    This class is responsible for running experiments, collecting data,
    and plotting convergence.

    Args:
        experiment_name (str): The name of the experiment.
        evo (object): The evolution object used for the experiment.

    Attributes:
        experiment_name (str): The name of the experiment.
        evo (object): The evolution object used for the experiment.
        experiment_data (list): A list to store experiment data.
        experiment_time (list): A list to store the measurement time for each experiment.
        total_experiment_time (list): A list to store the total runtime for each experiment.

    """

    def __init__(self, experiment_name, evo):
        self.experiment_name = experiment_name
        self.evo = evo
        self.experiments = []
        self.total_experiment_time = []

    def print_experiment_info(self):
        """Print a formatted overview of the experiment settings."""
        print("Experiment Settings:")
        print(f" - Number of Points: {self.evo.num_points}")
        print(f" - Reference Image: {self.evo.reference_image}")
        print(f" - Evolution Type: {self.evo.evolution_type}")
        print(f" - Population Size: {self.evo.population_size}")
        print(f" - Generation Budget: {self.evo.generation_budget}")
        print(f" - Evaluation Budget: {self.evo.evaluation_budget}")
        print(f" - Crossover Method: {self.evo.crossover_method}")
        print(f" - Mutation Probability: {self.evo.mutation_probability}")
        print(f" - Num Features Mutation Strength: {self.evo.num_features_mutation_strength}")
        print(f" - Num Features Mutation Strength Decay: {self.evo.num_features_mutation_strength_decay}")
        print(
            f" - Num Features Mutation Strength Decay Generations: {self.evo.num_features_mutation_strength_decay_generations}")
        print(f" - Selection Name: {self.evo.selection_name}")
        print(f" - Initialization: {self.evo.initialization}")
        print(f" - Noisy Evaluations: {self.evo.noisy_evaluations}")
        print(f" - Verbose: {self.evo.verbose}")
        print(f" - Generation Reporter: {self.evo.generation_reporter}")
        print(f" - Learning Rate: {self.evo.learning_rate}")
        print(f" - Negative Learning Rate: {self.evo.learning_rate_neg}")
        print(f" - Seed: {self.evo.seed}")
        print()

    def run_experiment(self, repeats=1, plot_converge=True, mode="generation"):
        """
        Run the experiment.

        Args:
            repeats (int): The number of times to repeat the experiment (default: 1).
            plot_converge (bool): Whether to plot the convergence (default: True).
            mode (str): The mode for plotting convergence, either "generation" or "time" (default: "generation").

        """
        self.clear_data()
        self.print_experiment_info()

        for run_idx in range(repeats):
            start = time()
            experiment_run = ExperimentData()

            self.evo.run(experiment_run)

            self.total_experiment_time.append(time() - start)
            self.experiments.append(experiment_run)

            fitness = self.experiments[run_idx].get_elite_fitness()
            total_time = self.total_experiment_time[run_idx]

            print(f"Run #{run_idx + 1}")
            print(f"Elite fitness {fitness}")
            print(f"Total Runtime {round(total_time, 2)} sec\n")

        all_elite_fitness = [experiment.get_elite_fitness() for experiment in self.experiments]
        average_elite_fitness = np.mean(all_elite_fitness, axis=0)
        print(f"Average Elite Fitness over {repeats} runs: {average_elite_fitness}")

        if plot_converge:
            self.__plot_convergence(repeats, mode)

        result = np.mean([experiment.get_fitness_data() for experiment in self.experiments], axis=0)

        return result

    def hyperparameter_search(self, hyperparameters, repeats=1, plot_converge=True, plot=True):
        """
        Run hyperparameter search.

        Args:
            hyperparameters (dict): A dictionary of hyperparameters to evaluate.
            repeats (int): The number of times to repeat the experiment (default: 1).
            plot_converge (bool): Whether to plot the convergence (default: True).
            mode (str): The mode for plotting convergence, either "generation" or "time" (default: "generation").
            plot (bool): Whether to plot the comparison.

        """
        param_combinations = [{key: value for key, value in zip(hyperparameters.keys(), combination)}
                              for combination in itertools.product(*hyperparameters.values())]

        best = None
        best_score = 0
        all_results = []
        original_name = self.experiment_name

        for experiment in param_combinations:
            param_name = ""
            for param in experiment:
                setattr(self.evo, param, experiment[param])
                param_name += f"{param}_{experiment[param]}_"

            self.experiment_name = f"{original_name}_{param_name}"

            print(f"Running new experiment: {self.experiment_name}\n")

            results = self.run_experiment(repeats, plot_converge=plot_converge, mode="generation")

            all_results.append(results)
            new_score = results[-1]

            if best is None:
                best = self.experiment_name
                best_score = new_score
            elif new_score < best_score:
                best_score = new_score
                best = self.experiment_name

        self.experiment_name = original_name

        print(f"Best hyperparameters found: {best} with score {best_score}")

        if plot:
            self.__plot_hs_convergence(all_results, param_combinations)

        return all_results

    def get_results(self):
        """
        Get the experiment results.

        Returns:
            tuple: A tuple containing the experiment data and total experiment time.

        """
        return self.experiments, self.total_experiment_time

    def __plot_convergence(self, repeats, mode):
        """
        Plot the convergence.

        Args:
            repeats (int): The number of experiment repeats.
            mode (str): The mode for plotting convergence, either "generation" or "time".

        """
        fitness_arrays = [experiment.get_fitness_data() for experiment in self.experiments]
        time = [experiment.get_measurement_time() for experiment in self.experiments]

        average_elite_fitness = np.mean(fitness_arrays, axis=0)
        std_deviations = np.std(fitness_arrays, axis=0)

        generations = list(range(len(average_elite_fitness)))
        time = np.mean(time, axis=0)

        if mode == "generation":
            plt.plot(generations, average_elite_fitness)
            plt.fill_between(generations, average_elite_fitness - std_deviations,
                             average_elite_fitness + std_deviations, alpha=0.3, label='Standard Deviation')
            plt.xlabel('Generations')
        elif mode == "time":
            plt.plot(time, average_elite_fitness)
            plt.fill_between(time, average_elite_fitness - std_deviations, average_elite_fitness + std_deviations,
                             alpha=0.3, label='Standard Deviation')
            plt.xlabel('Time(s)')

        plt.title(f'Convergence Plot for {self.experiment_name} - #repeats={repeats}')
        plt.ylabel('Fitness Score')
        plt.legend(['Elite Fitness'])
        plt.show()

    def __plot_hs_convergence(self, all_results, param_combinations):
        """
        Plot the convergence plots for a hyperparameter study.

        Args:
            all_results (list[ndarray]): A list of fitness scores for each generation, for each set of hyperparameters.
            param_combinations (list): A list of hyperparameter combinations.
        """
        generations = list(range(len(all_results[0])))

        for fitness, param in zip(all_results, param_combinations):
            plt.plot(generations, fitness, label=str(param))

        plt.xlabel('Generations')
        plt.title(f'Convergence Plots for Hyperparameter Study')
        plt.ylabel('Fitness Score')
        plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        plt.show()

    def clear_data(self):
        """
        Clear the data stored in the class variables.
        """
        self.experiments = []
        self.total_experiment_time = []
