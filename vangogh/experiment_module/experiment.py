import itertools
from time import time

from matplotlib import pyplot as plt

from vangogh.experiment_module.experiment_data import ExperimentData


class Experiment:

    def __init__(self, experiment_name, evo):
        self.experiment_name = experiment_name
        self.evo = evo
        self.experiment_data = []
        self.total_experiment_time = []

    def run_experiment(self, repeats=1, plot_converge=True, mode="generation"):
        for run_idx in range(repeats):
            start = time()
            data_obj = ExperimentData()
            self.evo.run(data_obj)
            self.total_experiment_time.append(time() - start)
            self.experiment_data.append(data_obj)

            if plot_converge:
                self.__plot_convergence(data_obj, run_idx, mode)

            cur_data = self.experiment_data[-1].get_data()
            total_time = self.total_experiment_time[-1]
            print(f"Average fitness {cur_data[1][-1]}")
            print(f"Elite fitness {cur_data[2][-1]}")
            print(f"Total Runtime {round(total_time, 2)} sec\n")

    def run_hyperparameter_eval(self, hyperparameters, plot_converge=True, mode="generation"):
        param_combinations = [
            {key: value for key, value in zip(hyperparameters.keys(), combination)}
            for combination in itertools.product(*hyperparameters.values())
        ]

        best = None
        best_score = 0

        original_name = self.experiment_name

        for experiment in param_combinations:
            param_name = ""
            for param in experiment:
                setattr(self.evo, param, experiment[param])
                param_name += f"{param}_{experiment[param]}_"

            self.experiment_name = f"{original_name}_{param_name}"
            self.run_experiment(plot_converge=plot_converge, mode=mode)

            new_score = self.experiment_data[-1].get_data()[2][-1]

            if best is None:
                best = self.experiment_name
                best_score = new_score
            elif new_score < best_score:
                best_score = new_score
                best = self.experiment_name

        self.experiment_name = original_name

        print(f"Best hyperparameters found: {best} with score {best_score}")

    def get_results(self):
        return self.experiment_data, self.total_experiment_time

    def __plot_convergence(self, data_obj, run_idx, mode):
        measurement_time, average_fitness, elite_fitness = data_obj.get_data()
        generations = list(range(len(average_fitness)))

        if mode == "generation":
            plt.plot(generations, average_fitness)
            plt.plot(generations, elite_fitness)
            plt.xlabel('Generations')
        elif mode == "time":
            plt.plot(measurement_time, average_fitness)
            plt.plot(measurement_time, elite_fitness)
            plt.xlabel('Time(s)')

        plt.title(f'Convergence Plot for {self.experiment_name} - run #{run_idx + 1}')
        plt.ylabel('Fitness Score')
        plt.legend(['Average Fitness', 'Elite Fitness'])
        plt.show()
