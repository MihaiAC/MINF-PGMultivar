from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.gibbs_samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *

from mpgm.mpgm.evaluation.evaluation_metrics import EvalMetrics

from mpgm.mpgm.models.Model import Model
from mpgm.mpgm.models.TPGM import TPGM
from mpgm.mpgm.models.QPGM import QPGM
from mpgm.mpgm.models.SPGM import SPGM
from mpgm.mpgm.models.LPGM import LPGM

from typing import List, Any, Callable

import matplotlib.pyplot as plt
import os
from itertools import cycle

# Generates samples and fits models on those samples.
# Should remember the IDs of the samples it generated and model it fit.
class Experiment():
    def __init__(self, experiment_name: str, random_seeds: List[int], SPW: SampleParamsWrapper,
                 samples_file_name:str, FPW:FitParamsWrapper, fit_file_name:str):
        self.experiment_name = experiment_name
        self.random_seeds = random_seeds
        self.samples_per_batch = len(random_seeds)
        self.SPW = SPW
        self.samples_file_name = samples_file_name
        self.FPW = FPW
        self.fit_file_name = fit_file_name

    def generate_batch_samples_vary_seed(self, batch_nr:int):
        for sample_nr in range(self.samples_per_batch):
            self.SPW.random_seed = self.random_seeds[sample_nr]
            sample_name = self.get_sample_name(batch_nr, sample_nr)

            self.SPW.generate_samples_and_save(sample_name, self.samples_file_name)
            print("Sampling " + sample_name + " finished.")

    def get_sample_name(self, batch_nr: int, sample_nr: int) -> str:
        sample_name = self.experiment_name + "_batch_" + str(batch_nr) + "_sample_" + str(sample_nr)
        return sample_name

    def get_fit_name(self, batch_nr:int, sample_nr:int) -> str:
        return "fit_" + self.get_sample_name(batch_nr, sample_nr)

    def vary_x_generate_samples(self, xs:List[Any], f_vary:Callable[[SampleParamsWrapper, Any], None]):
        for batch_nr, x in enumerate(xs):
            f_vary(self.SPW, x)
            self.generate_batch_samples_vary_seed(batch_nr)

    def vary_x_fit_samples(self):
        pass

    def fit_all_samples_same_FPW(self, nr_batches:int):
        for batch_nr in range(nr_batches):
            for sample_nr in range(self.samples_per_batch):
                sample_name = self.get_sample_name(batch_nr, sample_nr)
                fit_name = self.get_fit_name(batch_nr, sample_nr)
                self.FPW.fit_model_and_save(fit_id=fit_name,
                                            fit_file_name=self.fit_file_name,
                                            parallelize=True,
                                            samples_file_name=self.samples_file_name,
                                            samples_id=sample_name)

# Object can be used for a single experiment; destroy after usage.
class StatsGenerator():
    def __init__(self, experiment:Experiment, var_name:str, var_values:List[Any]):
        self.experiment = experiment
        self.var_name = var_name
        self.var_values = var_values

        self.nr_batches = len(self.var_values)

        sample_name = experiment.get_sample_name(0, 0)
        SPS = SampleParamsWrapper.load_samples(sample_name, experiment.samples_file_name)
        self.nr_variables = SPS.nr_variables

    def get_theta_orig(self, sample_nr:int, batch_nr:int) -> np.ndarray:
        sample_name = self.experiment.get_sample_name(batch_nr, sample_nr)
        SPS = SampleParamsWrapper.load_samples(sample_name, self.experiment.samples_file_name)

        theta_orig = SPS.theta_orig
        return theta_orig

    def get_theta_fit(self, sample_nr, batch_nr) -> np.ndarray:
        fit_name = self.experiment.get_fit_name(batch_nr, sample_nr)
        FPS = FitParamsWrapper.load_fit(fit_name, self.experiment.fit_file_name)

        theta_fit = FPS.theta_fit
        return theta_fit

    def get_sample_model(self, sample_nr, batch_nr) -> Model:
        sample_name = self.experiment.get_sample_name(batch_nr, sample_nr)
        SPS = SampleParamsWrapper.load_samples(sample_name, self.experiment.samples_file_name)

        model = globals()[SPS.model_name](**SPS.model_params)
        return model

    def get_fit_model(self, sample_nr, batch_nr) -> Model:
        fit_name = self.experiment.get_fit_name(batch_nr, sample_nr)
        FPS = FitParamsWrapper.load_fit(fit_name, self.experiment.fit_file_name)

        model = globals()[FPS.model_name](**FPS.model_params)
        return model

    def get_TPRs_FPRs_ACCs_nonzero(self, symm_mode:EvalMetrics.SymmModes):
        TPRs = []
        FPRs = []
        ACCs = []

        for batch_nr in range(self.nr_batches):
            batch_TPR, batch_FPR, batch_ACC = 0, 0, 0
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_orig = self.get_theta_orig(sample_nr, batch_nr)
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                TPR_sample, FPR_sample, ACC_sample = EvalMetrics.calculate_tpr_fpr_acc_nonzero(theta_orig, theta_fit, symm_mode)
                batch_TPR += TPR_sample
                batch_FPR += FPR_sample
                batch_ACC += ACC_sample

            N = self.experiment.samples_per_batch
            TPRs.append(batch_TPR/N)
            FPRs.append(batch_FPR/N)
            ACCs.append(batch_ACC/N)

        return TPRs, FPRs, ACCs

    def get_signed_recalls(self, symm_mode:EvalMetrics.SymmModes):
        signed_recalls = []

        for batch_nr in range(self.nr_batches):
            batch_signed_recall = 0
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_orig = self.get_theta_orig(sample_nr, batch_nr)
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                batch_signed_recall += EvalMetrics.calculate_signed_recall(theta_orig, theta_fit, symm_mode)
            signed_recalls.append(batch_signed_recall/self.experiment.samples_per_batch)

        return signed_recalls

    def get_MSEs(self, symm_mode:EvalMetrics.SymmModes):
        MSEs = []
        diag_MSEs = []

        for batch_nr in range(self.nr_batches):
            batch_MSE, batch_diag_MSE = (0, 0)
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_orig = self.get_theta_orig(sample_nr, batch_nr)
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                sample_MSE, sample_diag_MSE = EvalMetrics.calculate_MSEs(theta_orig, theta_fit, symm_mode)
                batch_MSE += sample_MSE
                batch_diag_MSE += batch_diag_MSE

            N = self.experiment.samples_per_batch
            MSEs.append(batch_MSE/N)
            diag_MSEs.append(batch_diag_MSE/N)

        return MSEs, diag_MSEs

    def get_KL_divergences(self, sampler:GibbsSampler):
        node_KL_divergences = []

        for batch_nr in range(self.nr_batches):
            batch_node_KL_divergence = np.zeros((self.nr_variables, ))
            for sample_nr in range(self.experiment.samples_per_batch):
                model_init_P = self.get_sample_model(sample_nr, batch_nr)
                model_fit_Q = self.get_fit_model(sample_nr, batch_nr)

                batch_node_KL_divergence += EvalMetrics.node_cond_prob_KL_divergence(model_init_P, model_fit_Q, sampler)[0]

            N = self.experiment.samples_per_batch
            node_KL_divergences.append(batch_node_KL_divergence/N)

        return node_KL_divergences


    def get_fit_sparsities_and_symms(self):
        symm_nonzero = []
        symm_values = []
        symm_signs = []
        sparsities = []

        for batch_nr in range(self.nr_batches):
            batch_nonzero, batch_values, batch_signs, batch_sparsity = (0, 0, 0, 0)
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                batch_nonzero += EvalMetrics.calculate_percentage_symmetric_nonzero(theta_fit)
                batch_values += EvalMetrics.calculate_percentage_symmetric_values(theta_fit)
                batch_signs += EvalMetrics.calculate_percentage_symmetric_signs(theta_fit)
                batch_sparsity += EvalMetrics.calculate_percentage_sparsity(theta_fit)

            N = self.experiment.samples_per_batch
            symm_nonzero.append(batch_nonzero/N)
            symm_values.append(batch_values/N)
            symm_signs.append(batch_signs/N)
            sparsities.append(batch_sparsity/N)

        return sparsities, symm_nonzero, symm_signs, symm_values

    def create_experiment_plot_folder(self):
        if not os.path.exists(self.experiment.experiment_name):
            os.makedirs(self.experiment.experiment_name)

    def plot_ys_common_xs(self, plot_name:str, x_name:str, y_name:str, xs:List[Any], list_ys:List[List[Any]],
                          labels:List[str]):
        self.create_experiment_plot_folder()

        marker = cycle(('o', '^', 's', 'X', '*'))
        for ii, ys in enumerate(list_ys):
            plt.plot(xs, ys, linestyle='--', marker=next(marker), label=labels[ii])

        plt.title(plot_name)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend()
        plt.savefig(self.experiment.experiment_name + '/' + plot_name + '.png')

        print('Plotting ' + plot_name + ' finished!')

