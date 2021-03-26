from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter, Constrained_Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.samplers import *

from mpgm.mpgm.evaluation.preprocessing import ClampMax

import numpy as np
import matplotlib.pyplot as plt


samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
nr_variables = 10
nr_samples = 150
alpha = 0.5

if __name__ == '__main__':
    # SGW = SampleParamsWrapper(nr_variables=nr_variables,
    #                           sample_init=np.zeros((nr_variables, )),
    #                           nr_samples=nr_samples,
    #                           random_seed=1)
    #
    # SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)
    # # SGW.weight_assigner = Constant_Weight_Assigner(ct_weight=-0.1)
    # SGW.weight_assigner = Dummy_Weight_Assigner()
    # # SGW.model = TPGM(R=10)
    # SGW.model = Model(theta=np.zeros((nr_samples, nr_variables)))
    # # SGW.sampler = TPGMGibbsSampler(burn_in=200, thinning_nr=150)
    # SGW.sampler = SIPRVSampler(lambda_true=1, lambda_noise=0.5)
    # SGW.generate_samples_and_save("SPGM_debug", samples_file_name)

    samples_id = "SPGM_debug"
    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = SPGM(R=10, R0=5)

    FPW.fitter = Constrained_Prox_Grad_Fitter(alpha=9,
                                              constraint_solver='admm',
                                              save_regularization_paths=True)

    FPW.preprocessor = ClampMax(10)

    theta_init = np.random.normal(-0.05, 0.02, (nr_samples, nr_variables))
    theta_fit = FPW.fit_model_and_save(fit_id="SPGM_debug",
                                       fit_file_name=fit_file_name,
                                       parallelize=False,
                                       samples_file_name=samples_file_name,
                                       samples_id="SPGM_debug",
                                       theta_init=None)

    regularization_paths = FPW.FPS.regularization_paths
    print(regularization_paths)

    x = list(range(len(regularization_paths)))
    for var in range(nr_variables):
        plt.plot(x, regularization_paths[:, var])
    plt.show()

    print(theta_fit)
