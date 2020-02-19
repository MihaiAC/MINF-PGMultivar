import numpy as np
import pickle
import os
from mpgm.mpgm.weight_assigners import assign_weights_bimodal_distr
from mpgm.mpgm.TPGM import TPGM
from mpgm.mpgm.QPGM import QPGM
from mpgm.mpgm.SPGM import SPGM
from mpgm.mpgm.graph_generators import generate_scale_free_graph
from mpgm.mpgm.graph_generators import generate_lattice_graph
from pathlib import Path

class Sampler():
    def __init__(self, samples_name, random_seed):
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
        np.random.seed(random_seed)

        self.folder_path = 'Samples/' + samples_name

        self.graph_generator = None
        self.graph_generator_params = None
        self.nr_variables = None

        self.weight_assigner = None
        self.weight_assigner_params = None

        self.generator_model = None
        self.generator_model_params = None

        self.sampling_method_name = None
        self.sampling_method_params = None

    def set_graph_generator(self, graph_generator, nr_variables, **graph_generator_params):
        self.graph_generator = graph_generator
        self.graph_generator_params = graph_generator_params
        self.nr_variables = nr_variables

    def set_weight_assigner(self, weight_assigner, **weight_assigner_params):
        self.weight_assigner = weight_assigner
        self.weight_assigner_params = weight_assigner_params

    def set_generator_model(self, generator_model, **generator_model_params):
        self.generator_model = generator_model
        self.generator_model_params = generator_model_params

    def set_sampling_method(self, sampling_method_name, **sampling_method_params):
        self.sampling_method_name = sampling_method_name
        self.sampling_method_params = sampling_method_params

    def generate_samples(self):
        if os.path.isdir(self.folder_path):
            print('The sample parameters and samples in ' + self.folder_path + ' will be overwritten. Type Y to proceed, ' +
                  'anything else to abort.')
            confirm = input()
            if confirm == "Y" or confirm == 'y':
                print('Overwrite confirmed.')
            else:
                return
        else:
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)

        folder_path = self.folder_path + '/'
        generator_model_theta = self.graph_generator(self.nr_variables, **self.graph_generator_params)
        self.weight_assigner(generator_model_theta, **self.weight_assigner_params)
        model = self.generator_model(theta=generator_model_theta, **self.generator_model_params)
        generated_samples = model.__getattribute__(self.sampling_method_name)(**self.sampling_method_params)

        # Save the parameters corresponding to the sample we've generated:
        with open(folder_path + 'graph_generator_params.pkl', 'wb') as f:
            pickle.dump(self.graph_generator.__name__, f)
            pickle.dump(self.graph_generator_params, f)

        with open(folder_path + 'weight_assigner_params.pkl', 'wb') as f:
            pickle.dump(self.weight_assigner.__name__, f)
            pickle.dump(self.weight_assigner_params, f)

        with open(folder_path + 'generator_model_params.pkl', 'wb') as f:
            pickle.dump(self.generator_model.__name__, f)
            pickle.dump(self.generator_model_params, f)

        with open(folder_path + 'sampling_method_params.pkl', 'wb') as f:
            pickle.dump(self.sampling_method_name, f)
            pickle.dump(self.sampling_method_params, f)

        np.save(folder_path + 'generator_model_theta.npy', generator_model_theta)
        np.save(folder_path + 'samples.npy', generated_samples)


def generate_TPGM_samples1():
    sampler = Sampler(samples_name='TPGM_test', random_seed=145)
    sampler.set_graph_generator(generate_scale_free_graph, nr_variables=10)
    sampler.set_weight_assigner(weight_assigner=assign_weights_bimodal_distr, neg_mean=-0.04, pos_mean=-0.04,
                                neg_threshold=0.05, std=0.03)
    sampler.set_generator_model(TPGM, R=10)
    sampler.set_sampling_method('generate_samples_gibbs', init=np.random.randint(0, 3, (sampler.nr_variables, )) ,
                                nr_samples=60, burn_in=5000, thinning_nr=1000)

    sampler.generate_samples()


if __name__ == '__main__':
    generate_TPGM_samples1()