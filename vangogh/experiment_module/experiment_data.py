class ExperimentData:

    def __init__(self):
        self.measurement_time = []
        self.average_fitness = []
        self.elite_fitness = []

    def add_measurement(self, time, average, elite):
        self.measurement_time.append(time)
        self.average_fitness.append(average)
        self.elite_fitness.append(elite)

    def get_data(self):
        return self.measurement_time, \
            self.average_fitness, \
            self.elite_fitness
