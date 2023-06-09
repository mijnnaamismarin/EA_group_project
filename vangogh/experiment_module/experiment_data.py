class ExperimentData:

    def __init__(self):
        self.__measurement_time = []
        self.__fitness_data = []

    def add_measurement(self, time, elite):
        self.__measurement_time.append(time)
        self.__fitness_data.append(elite)

    def get_measurement_time(self):
        return self.__measurement_time

    def get_fitness_data(self):
        return self.__fitness_data

    def get_elite_fitness(self):
        return self.__fitness_data[-1]

