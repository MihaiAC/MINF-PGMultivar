from mpgm.mpgm.evaluation.evaluation_metrics import *
from mpgm.mpgm.sample_generation.samplers import *
from mpgm.mpgm.sample_generation.weight_assigners import *

from mpgm.mpgm.evaluation.evaluation_metrics import EvalMetrics

from mpgm.mpgm.models.Model import Model

from typing import List, Any, Callable

import matplotlib.pyplot as plt
import os
from itertools import cycle

# Generates samples and fits models on those samples.
# Should remember the IDs of the samples it generated and model it fit.
class Experiment():
    def __init__(self, experiment_name: str, random_seeds: List[int], SPW: SampleParamsWrapper,
                 samples_file_name:str, FPW:FitParamsWrapper, fit_file_name:str, vary_fit:Optional[bool]=False):
        self.experiment_name = experiment_name
        self.random_seeds = random_seeds
        self.samples_per_batch = len(random_seeds)
        self.SPW = SPW
        self.samples_file_name = samples_file_name
        self.FPW = FPW
        self.fit_file_name = fit_file_name
        self.vary_fit = vary_fit

    def generate_batch_samples_vary_seed(self, batch_nr:int):
        for sample_nr in range(self.samples_per_batch):
            self.SPW.random_seed = self.random_seeds[sample_nr]
            sample_name = self.get_sample_name(batch_nr, sample_nr)

            self.SPW.generate_samples_and_save(sample_name, self.samples_file_name)
            print("Sampling " + sample_name + " finished.")

    def get_sample_name(self, batch_nr: int, sample_nr: int) -> str:
        if self.vary_fit:
            batch_nr = 0
        sample_name = self.experiment_name + "_batch_" + str(batch_nr) + "_sample_" + str(sample_nr)
        return sample_name

    def get_fit_name(self, batch_nr:int, sample_nr:int) -> str:
        fit_name = "fit_" + self.experiment_name + "_batch_" + str(batch_nr) + "_sample_" + str(sample_nr)
        return fit_name

    def vary_x_generate_samples(self, xs:List[Any], f_vary:Callable[[SampleParamsWrapper, Any], None]):
        for batch_nr, x in enumerate(xs):
            f_vary(self.SPW, x)
            self.generate_batch_samples_vary_seed(batch_nr)

    def generate_one_samples_batch(self):
        self.generate_batch_samples_vary_seed(0)

    def vary_x_fit_samples(self, xs:List[Any], f_vary:Callable[[FitParamsWrapper, Any], None]):
        for batch_nr, x in enumerate(xs):
            f_vary(self.FPW, x)
            self.fit_batch_samples_same_FPW(batch_nr)

    def fit_all_samples_same_FPW(self, nr_batches:int):
        for batch_nr in range(nr_batches):
            self.fit_batch_samples_same_FPW(batch_nr)

    def fit_batch_samples_same_FPW(self, batch_nr:int):
        sg = StatsGenerator(self, "", [])
        for sample_nr in range(self.samples_per_batch):
            sample_name = self.get_sample_name(batch_nr, sample_nr)
            fit_name = self.get_fit_name(batch_nr, sample_nr)
            self.FPW.fit_model_and_save(fit_id=fit_name,
                                        fit_file_name=self.fit_file_name,
                                        parallelize=True,
                                        samples_file_name=self.samples_file_name,
                                        samples_id=sample_name)
            print(self.get_fit_name(batch_nr, sample_nr) + " finished fitting. Average iterations: " +
                  str(sg.get_avg_nr_iterations_fit(sample_nr, batch_nr)))

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

    def get_avg_nr_iterations_fit(self, sample_nr, batch_nr) -> float:
        fit_name = self.experiment.get_fit_name(batch_nr, sample_nr)
        FPS = FitParamsWrapper.load_fit(fit_name, self.experiment.fit_file_name)

        nr_variables = len(FPS.likelihoods)
        nr_iterations = 0
        for likelihoods in FPS.likelihoods:
            nr_iterations += len(likelihoods)
        return nr_iterations/nr_variables

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

    def get_TPRs_FPRs_ACCs_nonzero(self, symm_mode:EvalMetrics.SymmModes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        TPRs = np.zeros((self.experiment.samples_per_batch, self.nr_batches))
        FPRs = np.zeros((self.experiment.samples_per_batch, self.nr_batches))
        ACCs = np.zeros((self.experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_orig = self.get_theta_orig(sample_nr, batch_nr)
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                TPR_sample, FPR_sample, ACC_sample = EvalMetrics.calculate_tpr_fpr_acc_nonzero(theta_orig, theta_fit, symm_mode)
                TPRs[sample_nr, batch_nr] = TPR_sample
                FPRs[sample_nr, batch_nr] = FPR_sample
                ACCs[sample_nr, batch_nr] = ACC_sample
        return np.array(TPRs), np.array(FPRs), np.array(ACCs)

    def get_edge_sign_recalls(self, symm_mode:EvalMetrics.SymmModes) -> np.ndarray:
        edge_sign_recalls = np.zeros((self.experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_orig = self.get_theta_orig(sample_nr, batch_nr)
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                edge_sign_recalls[sample_nr, batch_nr] = EvalMetrics.calculate_edge_sign_recall(theta_orig, theta_fit, symm_mode)
        return np.array(edge_sign_recalls)

    def get_MSEs(self, symm_mode:EvalMetrics.SymmModes) -> Tuple[np.ndarray, np.ndarray]:
        MSEs = np.zeros((self.experiment.samples_per_batch, self.nr_batches))
        diag_MSEs = np.zeros((self.experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_orig = self.get_theta_orig(sample_nr, batch_nr)
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                sample_MSE, sample_diag_MSE = EvalMetrics.calculate_MSEs(theta_orig, theta_fit, symm_mode)
                MSEs[sample_nr, batch_nr] = sample_MSE
                diag_MSEs[sample_nr, batch_nr] = sample_diag_MSE

        return MSEs, diag_MSEs

    def get_KL_divergences(self, sampler:GibbsSampler) -> np.ndarray:
        node_KL_divergences = np.zeros((self.experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(self.experiment.samples_per_batch):
                model_init_P = self.get_sample_model(sample_nr, batch_nr)
                model_fit_Q = self.get_fit_model(sample_nr, batch_nr)

                node_KL_divergences[sample_nr, batch_nr] = EvalMetrics.node_cond_prob_KL_divergence(model_init_P, model_fit_Q, sampler)[0]

        return node_KL_divergences

    def get_avg_node_fit_times(self) -> np.ndarray:
        avg_fit_times = np.zeros((self.experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(self.experiment.samples_per_batch):
                fit_name = self.experiment.get_fit_name(batch_nr, sample_nr)
                FPS = FitParamsWrapper.load_fit(fit_name, self.experiment.fit_file_name)

                if FPS.avg_node_fit_time is not None:
                    avg_fit_times[sample_nr, batch_nr] = FPS.avg_node_fit_time
        return avg_fit_times

    def get_fit_sparsities_and_symms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        symm_nonzero = np.zeros((self.experiment.samples_per_batch, self.nr_batches))
        symm_values = np.zeros((self.experiment.samples_per_batch, self.nr_batches))
        symm_signs = np.zeros((self.experiment.samples_per_batch, self.nr_batches))
        sparsities = np.zeros((self.experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(self.experiment.samples_per_batch):
                theta_fit = self.get_theta_fit(sample_nr, batch_nr)

                symm_nonzero[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_symmetric_nonzero(theta_fit)
                symm_values[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_symmetric_values(theta_fit)
                symm_signs[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_symmetric_signs(theta_fit)
                sparsities[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_sparsity(theta_fit)

        return sparsities, symm_nonzero, symm_signs, symm_values

    def get_nr_iterations(self) -> np.ndarray:
        nr_iterations = np.zeros((self.experiment.samples_per_batch, self.nr_batches))
        for batch_nr in range(self.nr_batches):
            for sample_nr in range(self.experiment.samples_per_batch):
                nr_iterations[sample_nr, batch_nr] = self.get_avg_nr_iterations_fit(sample_nr, batch_nr)
        return nr_iterations

    def create_experiment_plot_folder(self):
        if not os.path.exists('../' + self.experiment.experiment_name):
            os.makedirs('../' + self.experiment.experiment_name)

    def plot_ys_common_xs(self, xs:List[Any], list_ys:List[np.ndarray], ys_errors:List[np.ndarray], x_name:str,
                          y_name:str, plot_title:str, labels:List[str]):
        self.create_experiment_plot_folder()

        marker = cycle(('o', '^', 's', 'X', '*'))
        for ii, ys in enumerate(list_ys):
            if len(labels) == 0:
                if len(ys_errors) == 0:
                    plt.plot(xs, ys, linestyle='--', marker=next(marker))
                else:
                    plt.errorbar(xs, ys, yerr=ys_errors[ii], marker=next(marker))
            else:
                if len(ys_errors) == 0:
                    plt.plot(xs, ys, linestyle='--', marker=next(marker), label=labels[ii])
                else:
                    plt.errorbar(xs, ys, yerr=ys_errors[ii], marker=next(marker))

        plt.title(plot_title)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        if len(labels) != 0:
            plt.legend()

        new_title = y_name + ' vs ' + x_name + ": " + plot_title
        plt.savefig('../' + self.experiment.experiment_name + '/' + new_title + '.png')
        plt.close()

        print('Plotting ' + plot_title + ', ' + y_name + ' vs ' + x_name +  ' finished!')

    def plot_ys_common_xs_ALL(self, plot_title:str, x_name:str, xs:List[Any]):
        ww_min = EvalMetrics.SymmModes.WW_MIN
        ww_max = EvalMetrics.SymmModes.WW_MAX
        symm_none = EvalMetrics.SymmModes.NONE

        symm_modes = [ww_min, ww_max, symm_none]
        symm_modes_names = ['ww_min', 'ww_max', 'symm_none']

        for ii in range(len(symm_modes)):
            symm_mode = symm_modes[ii]
            symm_name = symm_modes_names[ii]
            TPRs, FPRs, ACCs = self.get_TPRs_FPRs_ACCs_nonzero(symm_mode)
            self.plot_ys_common_xs(xs=xs,
                                   list_ys=[np.mean(TPRs, axis=0)],
                                   ys_errors=[np.std(TPRs, axis=0)],
                                   x_name=x_name,
                                   y_name='TPR',
                                   plot_title=plot_title + ":symm_mode=" + symm_name,
                                   labels=[])

            self.plot_ys_common_xs(xs=xs,
                                   list_ys=[np.mean(FPRs, axis=0)],
                                   ys_errors=[np.std(FPRs, axis=0)],
                                   x_name=x_name,
                                   y_name='FPR',
                                   plot_title=plot_title + ":symm_mode=" + symm_name,
                                   labels=[])

            self.plot_ys_common_xs(xs=xs,
                                   list_ys=[np.mean(ACCs, axis=0)],
                                   ys_errors=[np.std(ACCs, axis=0)],
                                   x_name=x_name,
                                   y_name='ACC',
                                   plot_title=plot_title + ":symm_mode=" + symm_name,
                                   labels=[])

            edge_sign_recalls = self.get_edge_sign_recalls(symm_mode)
            self.plot_ys_common_xs(xs=xs,
                                   list_ys=[np.mean(edge_sign_recalls, axis=0)],
                                   ys_errors=[np.std(edge_sign_recalls, axis=0)],
                                   x_name=x_name,
                                   y_name='edge_sign_recall',
                                   plot_title=plot_title + ":symm_mode=" + symm_name,
                                   labels=[])

            MSEs, diag_MSEs = self.get_MSEs(symm_mode)
            self.plot_ys_common_xs(xs=xs,
                                   list_ys=[np.mean(MSEs, axis=0)],
                                   ys_errors=[np.std(MSEs, axis=0)],
                                   x_name=x_name,
                                   y_name='MSE',
                                   plot_title=plot_title + ":symm_mode=" + symm_name,
                                   labels=[])

            self.plot_ys_common_xs(xs=xs,
                                   list_ys=[np.mean(diag_MSEs, axis=0)],
                                   ys_errors=[np.std(diag_MSEs, axis=0)],
                                   x_name=x_name,
                                   y_name='diag_MSE',
                                   plot_title=plot_title + ":symm_mode=" + symm_name,
                                   labels=[])

        sparsities, symm_nonzero, symm_signs, symm_values = self.get_fit_sparsities_and_symms()
        self.plot_ys_common_xs(xs=xs,
                               list_ys=[np.mean(sparsities, axis=0)],
                               ys_errors=[np.std(sparsities, axis=0)],
                               x_name=x_name,
                               y_name='sparsity',
                               plot_title=plot_title,
                               labels=[])
        self.plot_ys_common_xs(xs=xs,
                               list_ys=[np.mean(symm_nonzero, axis=0)],
                               ys_errors=[np.std(symm_nonzero, axis=0)],
                               x_name=x_name,
                               y_name='symm_nonzero',
                               plot_title=plot_title,
                               labels=[])
        self.plot_ys_common_xs(xs=xs,
                               list_ys=[np.mean(symm_signs, axis=0)],
                               ys_errors=[np.std(symm_signs, axis=0)],
                               x_name=x_name,
                               y_name='symm_signs',
                               plot_title=plot_title,
                               labels=[])
        self.plot_ys_common_xs(xs=xs,
                               list_ys=[np.mean(symm_values, axis=0)],
                               ys_errors=[np.std(symm_values, axis=0)],
                               x_name=x_name,
                               y_name='symm_values',
                               plot_title=plot_title,
                               labels=[])

        nr_iterations = self.get_nr_iterations()
        self.plot_ys_common_xs(xs=xs,
                               list_ys=[np.mean(nr_iterations, axis=0)],
                               ys_errors=[np.std(nr_iterations, axis=0)],
                               x_name=x_name,
                               y_name='Average iterations',
                               plot_title=plot_title,
                               labels=[])

        avg_node_fit_times = self.get_avg_node_fit_times()
        self.plot_ys_common_xs(xs=xs,
                               list_ys=[np.mean(avg_node_fit_times, axis=0)],
                               ys_errors=[np.std(avg_node_fit_times, axis=0)],
                               x_name=x_name,
                               y_name='Average fit times',
                               plot_title=plot_title,
                               labels=[])
