import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from mpgm.mpgm.weight_assigners import assign_weights_bimodal_distr
from mpgm.mpgm.TPGM import TPGM
from mpgm.mpgm.QPGM import QPGM
from mpgm.mpgm.SPGM import SPGM
from mpgm.mpgm.graph_generators import generate_scale_free_graph
from mpgm.mpgm.graph_generators import generate_lattice_graph
from mpgm.mpgm.graph_generators import generate_hub_graph
from mpgm.mpgm.graph_generators import generate_random_nm_graph
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
        self.model = self.generator_model(theta=generator_model_theta, **self.generator_model_params)
        generated_samples = self.model.__getattribute__(self.sampling_method_name)(**self.sampling_method_params)

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

        return generated_samples


def generate_model_samples(**kwargs):
    sampler = Sampler(samples_name=kwargs["samples_name"], random_seed=kwargs['random_seed'])

    graph_generator_name = kwargs["graph_generator"]
    if graph_generator_name == "scale_free":
        sampler.set_graph_generator(generate_scale_free_graph, nr_variables=kwargs["nr_variables"],
                                    alpha=kwargs["alpha"], beta=kwargs["beta"], gamma=kwargs["gamma"])
    elif graph_generator_name == "lattice":
        sampler.set_graph_generator(generate_lattice_graph, nr_variables=kwargs["nr_variables"],
                                    sparsity_level=kwargs["sparsity_level"])
    elif graph_generator_name == "hub":
        sampler.set_graph_generator(generate_hub_graph, nr_variables=kwargs["nr_variables"], nr_hubs=kwargs["nr_hubs"])
    elif graph_generator_name == "random":
        sampler.set_graph_generator(generate_random_nm_graph, nr_variables=kwargs["nr_variables"],
                                    nr_edges=kwargs["nr_edges"])
    else:
        raise ValueError("No graph generator " + str(graph_generator_name) + " found.")

    weight_assigner_name = kwargs["weight_assigner"]
    if weight_assigner_name == 'bimodal':
        sampler.set_weight_assigner(weight_assigner=assign_weights_bimodal_distr, neg_mean=kwargs["neg_mean"],
                                    pos_mean=kwargs["pos_mean"], std=kwargs["std"], neg_threshold=kwargs["neg_threshold"])
    else:
        raise ValueError("No weight assigner " + str(weight_assigner_name) + " found.")

    model_name = kwargs["model"]
    if model_name == "TPGM":
        sampler.set_generator_model(TPGM, R=kwargs["R"])
    elif model_name == "QPGM":
        sampler.set_generator_model(QPGM, theta_quad=kwargs["theta_quad"])
    elif model_name == "SPGM":
        sampler.set_generator_model(SPGM, R=kwargs["R"], R0=kwargs["R0"])
    elif model_name == "LPGM":
        raise ValueError("Cannot sample from an LPGM model.")
    else:
        raise ValueError("Model to sample from " + str(model_name) + " not found.")

    sampler.set_sampling_method('generate_samples_gibbs', init=kwargs['init'], nr_samples=kwargs['nr_samples'],
                                burn_in=kwargs['burn_in'], thinning_nr=kwargs['thinning_nr'])
    sampler.generate_samples()


def generate_TPGM_samples_vary_thinning_nr():
    thinning_nrs = [1, 10, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    sampler = Sampler(samples_name='TPGM_burn_in_test', random_seed=145)
    sampler.set_graph_generator(generate_scale_free_graph, nr_variables=4)
    sampler.set_weight_assigner(weight_assigner=assign_weights_bimodal_distr, neg_mean=-0.2, pos_mean=-0.2,
                                neg_threshold=0.5, std=0.05)
    sampler.set_generator_model(TPGM, R=10)

    init_sample = np.random.randint(2, 5, (sampler.nr_variables, ))

    thinning_nr_nlls = []
    for thinning_nr in thinning_nrs:
        np.random.seed(99)
        sampler.set_sampling_method('generate_samples_gibbs', init=init_sample, nr_samples=60, burn_in=200,
                                    thinning_nr = thinning_nr)
        sampler.folder_path = 'Samples/' + 'TPGM_burn_in_200_thinning_nr_' + str(thinning_nr)

        samples = sampler.generate_samples()
        sample_nll = sampler.model.calculate_joint_nll(samples)
        thinning_nr_nlls.append(sample_nll)

    thinning_nr_nlls = np.array(thinning_nr_nlls)
    thinning_nrs = np.log(np.array(thinning_nrs))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(thinning_nrs, thinning_nr_nlls)
    ax.set_title('Sample NLL vs Log thinning number')
    ax.set_xlabel('log thinning nr')
    ax.set_ylabel('sample NLL')
    fig.savefig('Samples/sample_nll_vs_log_thinning_nr.png')
