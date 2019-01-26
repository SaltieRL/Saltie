import math


class GeneticAlgorithm:
    def __init__(self):
        import torch
        self.torch = torch
        from examples.levi.torch_model import SymmetricModel
        self.Model = SymmetricModel

    def calc_fitness(self, list):
        """calculates fitness of each bot based on their minimum distance to the ball
        :param list of distances
        :returns minimum_distance_to_ball"""

        min_distance_to_ball = min(list)
        return min_distance_to_ball

    def avg_best_fitness(self, list):
        """calculate average fitness of five fittest (identical) genomes
        :param list is the list to find mean
        :returns the mean of the list"""

        mean = sum(list) / len(list)
        return mean

    def calc_fittest(self, list):
        """calculates the fittest bot of each generation
        :param list is input array to find minimum
        :returns tuple of the fittest score and index """

        index = 0
        temp = math.inf
        for count, i in enumerate(list):
            if i < temp:
                temp = i
                index = count
        return index

    def crossover(self, parent1, list):
        """ create new child with some weights from both parents
        :param parent1 is a parent
        :param list is the list to duplicate upon
        :returns new parameters from both parents"""

        state_dict = parent1.state_dict()
        for bot in list:
            bot.load_state_dict(state_dict)

        """param_new = []
        for param1, param2 in zip(parent1.parameters(), parent2.parameters()):
            param_new = self.torch.rand(param1.data.size())
            for index, value in enumerate(param_new):
                if index <= 4:
                    param_new.data[index] = param1.data[index]
                else:
                    param_new.data[index] = param2.data[index]

        return param_new.data"""

    def mutate(self, list, mut_rate):
        """Randomizes a certain amount of the first five models' parameters based on mutation rate
        :param list contains the parameters to be mutated
        :param mut_rate is the mutation rate"""

        for i, bot in enumerate(list):
            new_genes = self.Model()
            for param, param_new in zip(bot.parameters(), new_genes.parameters()):
                mask = self.torch.rand(param.data.size()) < mut_rate / (i + 1)
                param.data[mask] = param_new.data[mask]
