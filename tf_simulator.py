import numpy as np, os
import mod_hyper as mod, sys, math
from random import randint
import random
import tensorflow as tf

save_foldername = 'R_Simulator'

#TODO GENERALIZE THE NAMING AS INPUT
class Tracker(): #Tracker
    def __init__(self, parameters, foldername = save_foldername):
        self.foldername = foldername
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        self.file_save = 'Simulator.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/train_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/valid_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self):
        self.num_input = 21
        self.num_hnodes = 15
        self.num_output = 19

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 1000000000000
        self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):
            self.population_size = 100

            #SSNE stuff
            self.ssne_param = SSNE_param()
            self.total_gens = 25000
            #Determine the nerual archiecture
            self.arch_type = 3 #1 GRU
                               #2 GRU-MB
                               #3 TF_FEEDFORWARD

            if self.arch_type == 1: self.arch_type = 'GRU'
            elif self.arch_type ==2: self.arch_type = 'GRU-MB'
            elif self.arch_type == 3: self.arch_type = 'TF_Feedforward'
            else: sys.exit('Invalid choice of neural architecture')


class Simulator:
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        if parameters.arch_type == 'TF_Feedforward':
            self.agent = mod.TF_SSNE(self.parameters)
        else:
            self.agent = mod.SSNE(self.parameters)
        self.train_data, self.valid_data = self.data_preprocess()


    def run_bprop(self, num_gens):
        train_x = self.train_data[0:-1]
        train_y = self.train_data[1:,0:-2]

        #Run bprop in the individual 0 of the population
        self.agent.pop[0].run_bprop(num_gens, train_x, train_y)


    def data_preprocess(self, filename='ColdAir.csv', downsample_rate=25, split = 1000):
        # Import training data and clear away the two top lines
        data = np.loadtxt(filename, delimiter=',', skiprows=2)

        # Splice data (downsample)
        ignore = np.copy(data)
        data = data[0::downsample_rate]
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i != data.shape[0] - 1):
                    data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,
                                       j].sum() / downsample_rate
                else:
                    residue = ignore.shape[0] - i * downsample_rate
                    data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue, j].sum() / residue

        # Normalize between 0-0.99
        normalizer = np.zeros(data.shape[1], dtype=np.float64)
        min = np.zeros(len(data[0]), dtype=np.float64)
        max = np.zeros(len(data[0]), dtype=np.float64)
        for i in range(len(data[0])):
            min[i] = np.amin(data[:, i])
            max[i] = np.amax(data[:, i])
            normalizer[i] = max[i] - min[i] + 0.00001
            data[:, i] = (data[:, i] - min[i]) / normalizer[i]

        #Train/Valid split
        train_data = data[0:split]
        valid_data = data[split:len(data)]

        return train_data, valid_data

    def compute_fitness(self, agent_index, data):
        fitness = np.zeros(19)
        input = np.reshape(data[0], (21))  # First training example in its entirety
        for example in range(len(data) - 1):  # For all training examples

            model_out = self.agent.pop[agent_index].feedforward(input)  # Time domain simulation
            for index in range(19): # Calculate error (weakness)
                fitness[index] += math.fabs(model_out[index][0] - data[example + 1][index])  # Time variant simulation

            # Fill in new input data
            for k in range(len(model_out)):
                input[k] = model_out[k][0]
            # Fill in two control variables
            input[19] = data[example + 1][19]
            input[20] = data[example + 1][20]

        return -np.sum(np.square(fitness))/len(data)

    def evolve(self):
        best_epoch_reward = -1000000

        for agent_index in range(self.parameters.population_size): #Test all genomes/individuals
            fitness = self.compute_fitness(agent_index, self.train_data)
            self.agent.fitness_evals[agent_index] = fitness
            if fitness > best_epoch_reward: best_epoch_reward = fitness

        #HOF test net
        champion_index = self.agent.fitness_evals.index(max(self.agent.fitness_evals))
        valid_score = self.compute_fitness(champion_index, self.valid_data)

        #Save population and HOF
        if (gen + 1) % 1000 == 0:
            mod.pickle_object(self.agent.pop, save_foldername + '/simulator_pop')
            mod.pickle_object(self.agent.pop[champion_index], save_foldername + '/simulator_champion')
            np.savetxt(save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        self.agent.epoch()
        return best_epoch_reward, valid_score

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Running Simulator Training ', parameters.arch_type


    simulator = Simulator(parameters)
    simulator.run_bprop(10000)




    # for gen in range(parameters.total_gens):
    #
    #
    #
    #
    #     epoch_reward, valid_score = simulator.evolve()
    #     print 'Generation:', gen+1, ' Epoch_reward:', "%0.2f" % epoch_reward, '  Score:', "%0.2f" % valid_score, '  Cumul_Score:', "%0.2f" % tracker.hof_avg_fitness
    #     tracker.add_fitness(epoch_reward, gen)  # Add average global performance to tracker
    #     tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker














