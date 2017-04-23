import numpy as np, os
import mod_hyper as mod, sys, math
from random import randint
import random

#np.seterr(all='raise')
save_foldername = 'R_Controller'

class tracker(): #Tracker
    def __init__(self, parameters, foldername = save_foldername):
        self.foldername = foldername
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        if parameters.is_memoried:
            self.file_save = 'mem_seq_classifier.csv'
        else:
            self.file_save = 'norm_seq_classifier.csv'


    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/rough_' + self.file_save
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
    def __init__(self, is_memoried):
        self.num_input = 21
        self.num_hnodes = 10
        self.num_output = 2
        if is_memoried: self.type_id = 'memoried'
        else: self.type_id = 'normal'

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 1000000000000
        self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        if is_memoried:
            self.total_num_weights = 3 * (
                self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
        else:
            #Normalize network flexibility by changing hidden nodes
            naive_total_num_weights = self.num_hnodes*(self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #continue
            mem_weights = 3 * (
                 self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                 self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
            normalization_factor = int(mem_weights/naive_total_num_weights)

            #Set parameters for comparable flexibility with memoried net
            self.num_hnodes *= normalization_factor + 1
            self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
        print 'Num parameters: ', self.total_num_weights

class Parameters:
    def __init__(self):
        self.population_size = 100
        self.is_memoried = 1

        #DEAP/SSNE stuff
        self.use_ssne = 1
        self.ssne_param = SSNE_param( self.is_memoried)
        self.total_gens = 25000
        self.arch_type = 1 #0-Quasi GRU; #1-Quasi NTM #Determine the neural architecture
        if self.arch_type == 0: self.arch_type = 'quasi_gru'
        elif self.arch_type ==1: self.arch_type = 'quasi_ntm'
        else: sys.exit('Invalid choice of neural architecture')

        #The reading that we care about
        self.optimize_indices = [11] #Options: [11,15] permutations

        #Noise stuff
        self.sensor_noise = 0.0
        self.sensor_failure = None #Options: None, [11,15] permutations
        self.actuator_noise = 0.0
        self.actuator_failure = None #Options: None, [0,1] permutations
        self.run_time = 200 #Controller runtime

        #Reconfigurability parameters
        self.is_random_initial_state = False #Start state of controller
        self.num_profiles = 1

parameters = Parameters() #Create the Parameters class
tracker = tracker(parameters) #Initiate tracker

class Controller:
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.agent = mod.SSNE(self.parameters, self.ssne_param, parameters.arch_type)

        #Set up simulator
        self.simulator_data = self.read_all_simulator_data()
        self.simulator = mod.import_arch('R_Controller/sim.json')
        self.simulator.load_weights('R_Controller/sim.h5')
        self.simulator.compile(loss='mae', optimizer='adam')


    def read_all_simulator_data(self, filename='ColdAir.csv', downsample_rate=25, split = 1000):
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

        return data

    def compute_fitness(self, agent_index, initial_state, setpoints):
        weakness = np.zeros(len(self.parameters.optimize_indices))

        input = np.copy(initial_state)
        input[19] = setpoints[0][0]
        input[20] = setpoints[0][1]

        for example in range(len(setpoints) - 1):  # For all training examples
            # Add noise to the state input to the controller
            noise_input = np.copy(input)
            if self.parameters.sensor_noise != 0:  # Add sensor noise
                for i in range(19):
                    std = self.parameters.sensor_noise * abs(noise_input[0][i]) / 100.0
                    if std != 0:
                        noise_input[i] += np.random.normal(0, std / 2.0)

            if self.parameters.sensor_failure != None:  # Failed sensor outputs 0 regardless
                for i in self.parameters.sensor_failure:
                    noise_input[i] = 0

            # Get the controller output
            control_out = self.agent.pop[agent_index].feedforward(noise_input)

            # Add actuator noise (controls)
            if self.parameters.actuator_noise != 0:
                for i in range(len(control_out[0])):
                    std = self.parameters.actuator_noise * abs(control_out[0][i]) / 100.0
                    if std != 0:
                        control_out[0][i] += np.random.normal(0, std / 2.0)

            if self.parameters.actuator_failure != None:  # Failed actuator outputs 0 regardless
                for i in self.parameters.actuator_failure:
                    control_out[0][i] = 0

            # Fill in the controls
            input[19] = control_out[0][0]
            input[20] = control_out[1][0]

            # Use the simulator to get the next state
            model_out = self.simulator.predict(np.reshape(input, (1,21)))  # Time domain simulation
            # Calculate error (weakness)
            for i, index in enumerate(parameters.optimize_indices):
                weakness[i] += math.fabs(model_out[0][index] - setpoints[example][i])  # Time variant simulation

            # Fill in new input data
            for k in range(len(model_out[0])):
                input[k] = model_out[0][k]

            # Fill in next setpoints
            input[19] = setpoints[example + 1][0]
            input[20] = setpoints[example + 1][1]

        return np.sum(weakness)/(len(setpoints)*len(weakness))

    def get_setpoints(self):
        desired_setpoints = []
        for i in range(self.parameters.run_time):
            turbine = math.sin(i * 1.0 / 7.0)
            if turbine < 0: turbine = 0
            turbine *= 0.1
            turbine += 0.4
            fuel_cell = 0.75
            desired_setpoints.append([turbine, fuel_cell])
        return desired_setpoints

    def evolve(self):
        best_epoch_reward = -1000000

        #Get setpoints and initial_state for the generation
        setpoints = self.get_setpoints()
        if self.parameters.is_random_initial_state:
            initial_state = np.copy(self.simulator_data[randint(0,len(self.simulator_data))])
        else:
            initial_state = np.copy(self.simulator_data[0])

        for agent_index in range(self.parameters.population_size): #Test all genomes/individuals
            fitness = self.compute_fitness(agent_index, initial_state, setpoints)
            self.agent.fitness_evals[agent_index] = fitness
            if fitness > best_epoch_reward: best_epoch_reward = fitness

        #HOF test net
        champion_index = self.agent.fitness_evals.index(max(self.agent.fitness_evals))
        setpoints = self.get_setpoints()
        if self.parameters.is_random_initial_state:
            initial_state = np.copy(self.simulator_data[randint(0,len(self.simulator_data))])
        else:
            initial_state = np.copy(self.simulator_data[0])
        valid_score = self.compute_fitness(champion_index, initial_state, setpoints)

        #Save population and HOF
        if (gen + 1) % 1000 == 0:
            mod.pickle_object(self.agent.pop, save_foldername + '/simulator_pop')
            mod.pickle_object(self.agent.pop[champion_index], save_foldername + '/simulator_champion')
            np.savetxt(save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        self.agent.epoch()
        return best_epoch_reward, valid_score


if __name__ == "__main__":
    print 'Running  Training ', parameters.arch_type
    task = Controller(parameters)


    for gen in range(parameters.total_gens):
        epoch_reward, valid_score = task.evolve()
        print 'Generation:', gen+1, ' Epoch_reward:', "%0.2f" % epoch_reward, '  Score:', "%0.2f" % valid_score, '  Cumul_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(epoch_reward, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker














