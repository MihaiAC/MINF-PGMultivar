import numpy as np
import pickle
import os
from tqdm import trange
from mpgm.mpgm.TPGM import TPGM
from mpgm.mpgm.QPGM import QPGM
from mpgm.mpgm.SPGM import SPGM
from mpgm.mpgm.auxiliary_methods import generate_scale_free_graph
from pathlib import Path

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
                  'anything else to abort.')
            confirm = input()
            if confirm == "Y" or confirm == 'y':
                print('Overwrite confirmed.')
            else:
                return
        else:
            Path(folder_path).mkdir(parents=True, exist_ok=True)

        folder_path = folder_path + '/'
        generator_model_theta = self.graph_generator(**self.graph_generator_params)
        model = self.generator_model(generator_model_theta, **self.generator_model_params)
        generated_samples = self.sampling_method(model, **self.sampling_method_params)

        # Save the parameters corresponding to the sample we've generated:
        with open(folder_path + 'graph_generator_params.pkl', 'wb') as f:
            pickle.dump(self.graph_generator.__name__, f)
            pickle.dump(self.graph_generator_params, f)

        with open(folder_path + 'generator_model_params.pkl', 'wb') as f:
            pickle.dump(self.generator_model.__name__, f)
            pickle.dump(self.generator_model_params, f)

        with open(folder_path + 'sampling_method_params.pkl', 'wb') as f:
            pickle.dump(self.sampling_method.__name__, f)
            pickle.dump(self.sampling_method_params, f)

        np.save(folder_path + 'generator_model_theta.npy', generator_model_theta)
        np.save(folder_path + 'samples.npy', generated_samples)

    @staticmethod
    def generate_samples_gibbs(model, init, nr_samples=50, burn_in=700, thinning_nr=100):
        """
        Gibbs sampler for this implementation of the TPGM distribution.

        :param model: model to draw samples from, must implement "generate_sample_cond_prob(node, data)"
        :param init: (N, ) np array = starting value for the sampling process;
        :param nr_samples: number of samples we want to generate.
        :param burn_in: number of initial samples to be discarded.
        :param thinning_nr: we select every thinning_nr samples after the burn-in period;

        :return: (nr_samples X N) matrix containing one sample per row.
        """
        N = len(init)
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

class SamplerWrapper():
    def __init__(self, graph_generator_name, generator_model_name, sampling_method_name, samples_name, random_seed=200,
                 nr_variables=10, weak_pos_mean=0.04, weak_neg_mean=-0.04, weak_std=0.03, scale_free_alpha=0.1,
                 scale_free_beta=0.8, scale_free_gamma=0.1, R=10, R0=5, nr_samples=50, burn_in=200, thinning_nr=50,
                 theta_quad=None, sampler_init_values=None):
        np.random.seed(random_seed)

        if(sampler_init_values is None):
            sampler_init_values = np.random.randint(0, 3, (nr_variables, ))

        if(graph_generator_name == 'scale_free'):
            self.graph_generator = generate_scale_free_graph
            self.graph_generator_params = dict({'nr_variables': nr_variables, 'weak_pos_mean': weak_pos_mean,
                                'weak_neg_mean': weak_neg_mean, 'weak_std': weak_std, 'alpha': scale_free_alpha,
                                'beta': scale_free_beta, 'gamma': scale_free_gamma})
        else:
            raise ValueError('Invalid graph generator name.')

        if(generator_model_name == 'TPGM'):
            self.generator_model = TPGM
            self.generator_model_params = dict({'R': R})
        elif(generator_model_name == 'QPGM'):
            self.generator_model = QPGM
            assert theta_quad is not None, 'Invalid theta_quad value'
            self.generator_model_params = dict({'theta_quad': theta_quad})
        elif(generator_model_name == 'SPGM'):
            self.generator_model = SPGM
            self.generator_model_params = dict({'R': R, 'R0': R0})
        else:
            raise ValueError('Invalid generator model name.')

        if(sampling_method_name == 'gibbs'):
            self.sampling_method = Sampler.generate_samples_gibbs
            self.sampling_method_params = dict({'init': sampler_init_values, 'nr_samples': nr_samples, 'burn_in': burn_in,
                                          'thinning_nr': thinning_nr})
        else:
            raise ValueError('Invalid sampling method name.')

        self.samples_folder = 'Samples/' + samples_name

    def generate_samples(self):
        sampler = Sampler(self.graph_generator, self.graph_generator_params, self.generator_model,
                          self.generator_model_params, self.sampling_method, self.sampling_method_params)
        sampler.generate_samples_to_folder(self.samples_folder)


def generate_TPGM_samples1():
    sampler_wrapper = SamplerWrapper('scale_free', 'TPGM', 'gibbs', 'test', random_seed=145, weak_std=0.03,
                                     nr_samples=60, burn_in=200, thinning_nr=300, nr_variables=10, R=10)
    sampler_wrapper.generate_samples()

def generate_TPGM_samples_higher_thinning_nr():
    sampler_wrapper = SamplerWrapper('scale_free', 'TPGM', 'gibbs', 'test', random_seed=145, weak_std=0.03,
                                     nr_samples=30, burn_in=200, thinning_nr=1000, nr_variables=10, R=10)
    sampler_wrapper.generate_samples()

generate_TPGM_samples1()