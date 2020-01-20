import numpy as np
import pickle
import os
from tqdm import trange
from mpgm.mpgm.tpgm import Tpgm
from mpgm.mpgm.auxiliary_methods import generate_scale_free_graph

class Sampler():
    def __init__(self, graph_generator, graph_generator_params, generator_model, generator_model_params, sampling_method,
                 sampling_method_params):
        '''
        :param graph_generator: Graph generator used (e.g: scale-free graph).
        :param graph_generator_params: Dictionary containing the arguments that must be passed to the graph generator.
        :param generator_model: The model which generates the artificial data using the dependencies specified in the
            generated graph. Must accept the generated graph as the first parameter.
        :param generator_model_params: Dictionary containing the arguments that must be passed to the generator model.
        :param sampling_method: The sampler used to generate samples from the generator model. Must accept the model as its
            first parameter.
        :param sampling_method_params: Dictionary containing the arguments that must be passed to the sampler.
        '''

        self.graph_generator = graph_generator
        self.graph_generator_params = graph_generator_params
        self.generator_model = generator_model
        self.generator_model_params = generator_model_params
        self.sampling_method = sampling_method
        self.sampling_method_params = sampling_method_params

    def generate_samples_to_folder(self, folder_path):
        if os.path.isdir(folder_path):
            print('The sample parameters and samples in ' + folder_path + ' will be overwritten. Type Y to proceed, ' +
                                                                          + 'anything else to abort.')
            confirm = input()
            if confirm == "Y" or confirm == 'y':
                print('Overwrite confirmed.')
        else:
            os.mkdir(folder_path)

        folder_path = folder_path + '/'

        # Save the parameters corresponding to the sample we've generated:
        with open(folder_path + 'graph_sample_parameters.pkl', 'wb') as f:
            pickle.dump(self.graph_generator.__name__, f)
            pickle.dump(self.graph_generator_params, f)
            pickle.dump(self.generator_model.__name__, f)
            pickle.dump(self.generator_model_params, f)
            pickle.dump(self.sampling_method.__name__, f)
            pickle.dump(self.sampling_method_params, f)

        generator_model_theta = self.graph_generator(**self.graph_generator_params)
        np.save(folder_path + 'generator_model_theta.npy', generator_model_theta)
        model = self.generator_model(generator_model_theta, **self.generator_model_params)
        generated_samples = self.sampling_method(model, **self.sampling_method_params)
        np.save(folder_path + 'generated_samples.npy', generated_samples)

    @staticmethod
    def generate_samples_gibbs(model, init, nr_samples=50, burn_in=700, thinning_nr=100):
        """
        Gibbs sampler for this implementation of the TPGM distribution.

        :param model: model to draw samples from, must implement "generate_sample_cond_prob(node, data)"
        :param init: 1xN np array = starting value for the sampling process;
        :param nr_samples: number of samples we want to generate.
        :param burn_in: number of initial samples to be discarded.
        :param thinning_nr: we select every thinning_nr samples after the burn-in period;

        :return: (nr_samples X N) matrix containing one sample per row.
        """
        N = np.shape(init)[0]
        samples = np.zeros((nr_samples, N))
        nodes_values = np.array(init)

        for sample_nr in trange(burn_in + (nr_samples - 1) * thinning_nr + 1):
            # Generate one sample.
            for node in range(N):
                node_sample = model.generate_node_sample(node, nodes_values)
                nodes_values[node] = node_sample

            # Check if we should keep this sample.
            if sample_nr >= burn_in and (sample_nr - burn_in) % thinning_nr == 0:
                samples[(sample_nr - burn_in) // thinning_nr, :] = nodes_values

        return samples

random_seed = 200

nr_variables = 50
weak_pos_mean = 0.04
weak_neg_mean = -0.04
weak_std = 0.3
alpha = 0.1
beta = 0.8
gamma = 0.1
graph_generator_params = dict({'nr_variables': nr_variables, 'weak_pos_mean': weak_pos_mean,
                                'weak_neg_mean': weak_neg_mean, 'weak_std': weak_std, 'alpha': alpha, 'beta': beta,
                                'gamma': gamma})
graph_generator = generate_scale_free_graph

R = 50
generator_model = Tpgm
generator_model_params = dict({'R': R})

sampler_init = np.random.randint(1, 3, nr_variables)
nr_samples = 200
burn_in = 150
thinning_nr = 50
sampling_method_params = dict({'init': sampler_init, 'nr_samples': nr_samples, 'burn_in': burn_in,
                        'thinning_nr': thinning_nr})
sampling_method = Sampler.generate_samples_gibbs

def TPGM_scale_free_generate_samples1():
    samples_folder = 'Samples/TPGM_scale_free/Samples_1'
    sampler = Sampler(graph_generator, graph_generator_params, generator_model, generator_model_params, sampling_method,
                      sampling_method_params)
    sampler.generate_samples_to_folder(samples_folder)
