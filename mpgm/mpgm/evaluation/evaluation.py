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
                 samples_file_name:str, FPW:FitParamsWrapper, fit_file_name:str, vary_fit:Optional[bool]=False,
                 fit_theta_init:Optional[np.ndarray]=None, fit_parallelize:Optional[bool]=True,
                 samples_name:Optional[str]=None):
        self.experiment_name = experiment_name
        self.random_seeds = random_seeds
        self.samples_per_batch = len(random_seeds)
        self.SPW = SPW
        self.samples_file_name = samples_file_name
        self.FPW = FPW
        self.fit_file_name = fit_file_name
        self.vary_fit = vary_fit
        self.fit_parallelize = fit_parallelize
        self.fit_theta_init = fit_theta_init
        self.samples_name = samples_name

    def generate_batch_samples_vary_seed(self, batch_nr:int):
        for sample_nr in range(self.samples_per_batch):
            self.SPW.random_seed = self.random_seeds[sample_nr]
            sample_name = self.get_sample_name(batch_nr, sample_nr)

            self.SPW.generate_samples_and_save(sample_name, self.samples_file_name)
            print("Sampling " + sample_name + " finished.")

    def get_sample_name(self, batch_nr: int, sample_nr: int) -> str:
        if self.vary_fit:
            batch_nr = 0
        if self.samples_name is None:
            sample_name = self.experiment_name
        else:
            sample_name = self.samples_name
        sample_name = sample_name + "_batch_" + str(batch_nr) + "_sample_" + str(sample_nr)
        return sample_name

    def get_fit_name(self, batch_nr:int, sample_nr:int) -> str:
        fit_name = "fit_" + self.experiment_name + "_batch_" + str(batch_nr) + "_sample_" + str(sample_nr)
        return fit_name

    def vary_x_generate_samples(self, xs:List[Any], f_vary:Callable[[SampleParamsWrapper, Any], None]):
        for batch_nr, x in enumerate(xs):
            f_vary(self.SPW, x)
            self.generate_batch_samples_vary_seed(batch_nr)

    def generate_single_batch_of_samples(self):
        self.generate_batch_samples_vary_seed(0)

    def vary_x_fit_samples(self, xs:List[Any], f_vary:Callable[[FitParamsWrapper, Any], None]):
        for batch_nr, x in enumerate(xs):
            f_vary(self.FPW, x)
            self.fit_batch_samples_same_FPW(batch_nr)
            print('Fitting for parameter ' + str(x) + ' finished!')

    def fit_all_samples_same_FPW(self, nr_batches:int):
        for batch_nr in range(nr_batches):
            self.fit_batch_samples_same_FPW(batch_nr)
            print('Batch ' + str(batch_nr) + ' finished.')

    # TODO: Need exception for LPGM.
    # nr_batches must be equal to nr_alphas.
    def fit_LPGM(self, nr_batches):
        for sample_nr in range(self.samples_per_batch):
            sample_name = self.get_sample_name(0, sample_nr)
            fit_names = []
            for batch_nr in range(nr_batches):
                fit_names.append(self.get_fit_name(batch_nr, sample_nr))
            self.FPW.fit_lpgm_and_save(fit_ids=fit_names,
                                       fit_file_name=self.fit_file_name,
                                       samples_file_name=self.samples_file_name,
                                       samples_id=sample_name,
                                       theta_init=self.fit_theta_init)



    def fit_batch_samples_same_FPW(self, batch_nr:int):
        # sg = StatsGenerator(self, "", [])
        for sample_nr in range(self.samples_per_batch):
            sample_name = self.get_sample_name(batch_nr, sample_nr)
            fit_name = self.get_fit_name(batch_nr, sample_nr)
            self.FPW.fit_model_and_save(fit_id=fit_name,
                                        fit_file_name=self.fit_file_name,
                                        parallelize=self.fit_parallelize,
                                        samples_file_name=self.samples_file_name,
                                        samples_id=sample_name,
                                        theta_init=self.fit_theta_init)
            # print(self.get_fit_name(batch_nr, sample_nr) + " finished fitting. Average iterations: " +
            #       str(sg.get_avg_nr_iterations_fit(sample_nr, batch_nr)))

    @staticmethod
    def generate_experiment_name(model:str,
                                 graph_type:str,
                                 sample_gen_method:str,
                                 experiment_shorthand:str,
                                 early_stop_method:str,
                                 *args) -> str:
        function_args = [model, graph_type, sample_gen_method, experiment_shorthand, early_stop_method] + list(args)
        return '_'.join(function_args)

# Object can be used for a single experiment; destroy after usage.
class StatsGenerator():
    def __init__(self,
                 experiments:List[Experiment],
                 experiment_labels:List[str],
                 var_name:str,
                 var_values:List[Any],
                 folder_name:str
                 ):
        self.experiments = experiments
        self.experiment_labels = experiment_labels
        self.var_name = var_name
        self.var_values = var_values
        self.folder_name = folder_name

        StatsGenerator.create_experiment_plot_folder(folder_name)

        self.nr_batches = len(self.var_values)

        sample_name = experiments[0].get_sample_name(0, 0)
        SPS = SampleParamsWrapper.load_samples(sample_name, experiments[0].samples_file_name)

        # Implicitly assumed that all the experiments for which we want plots have the same number of variables (
        # otherwise, comparing them is pointless).
        self.nr_variables = SPS.nr_variables

    def get_all_theta_origs(self, sample_nr:int, batch_nr:int) -> List[np.ndarray]:
        theta_origs = []

        for experiment in self.experiments:
            theta_origs.append(self.get_experiment_theta_orig(sample_nr, batch_nr, experiment))
        return theta_origs

    def get_all_theta_fits(self, sample_nr, batch_nr) -> List[np.ndarray]:
        theta_fits = []

        for experiment in self.experiments:
            theta_fits.append(self.get_experiment_theta_fit(sample_nr, batch_nr, experiment))
        return theta_fits

    def get_experiment_theta_orig(self, sample_nr:int, batch_nr:int, experiment:Experiment) -> np.ndarray:
        sample_name = experiment.get_sample_name(batch_nr, sample_nr)
        SPS = SampleParamsWrapper.load_samples(sample_name, experiment.samples_file_name)

        return SPS.theta_orig

    def get_experiment_theta_fit(self, sample_nr:int, batch_nr:int, experiment:Experiment) -> np.ndarray:
        fit_name = experiment.get_fit_name(batch_nr, sample_nr)
        FPS = FitParamsWrapper.load_fit(fit_name, experiment.fit_file_name)

        return FPS.theta_fit

    def get_all_avg_nr_iterations_fit(self, sample_nr, batch_nr) -> List[float]:
        avg_nr_iterations_fits = []
        for experiment in self.experiments:
            avg_nr_iterations_fits.append(self.get_experiment_avg_nr_iterations_fit(sample_nr, batch_nr, experiment))
        return avg_nr_iterations_fits

    def get_experiment_avg_nr_iterations_fit(self, sample_nr, batch_nr, experiment) -> float:
        fit_name = experiment.get_fit_name(batch_nr, sample_nr)
        FPS = FitParamsWrapper.load_fit(fit_name, experiment.fit_file_name)

        nr_variables = len(FPS.likelihoods)
        nr_iterations = 0
        for likelihoods in FPS.likelihoods:
            nr_iterations += len(likelihoods)
        return nr_iterations / nr_variables

    def get_all_sample_models(self, sample_nr, batch_nr) -> List[Model]:
        sample_models = []
        for experiment in self.experiments:
            sample_models.append(self.get_experiment_sample_model(sample_nr, batch_nr, experiment))
        return sample_models

    def get_experiment_sample_model(self, sample_nr, batch_nr, experiment) -> Model:
        sample_name = experiment.get_sample_name(batch_nr, sample_nr)
        SPS = SampleParamsWrapper.load_samples(sample_name, experiment.samples_file_name)

        model = globals()[SPS.model_name](**SPS.model_params)
        return model

    def get_all_fit_models(self, sample_nr, batch_nr) -> List[Model]:
        fit_models = []
        for experiment in self.experiments:
            fit_models.append(self.get_experiment_fit_model(sample_nr, batch_nr, experiment))
        return fit_models

    def get_experiment_fit_model(self, sample_nr, batch_nr, experiment) -> Model:
        fit_name = experiment.get_fit_name(batch_nr, sample_nr)
        FPS = FitParamsWrapper.load_fit(fit_name, experiment.fit_file_name)

        model = globals()[FPS.model_name](**FPS.model_params)
        return model

    def get_all_TPRs_FPRs_ACCs_nonzero(self, symm_mode:EvalMetrics.SymmModes) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        experiment_TPRs_FPRs_ACCs = []

        for experiment in self.experiments:
            experiment_TPRs_FPRs_ACCs.append(self.get_experiment_TPRs_FPRs_ACCs_nonzero(symm_mode, experiment))
        return experiment_TPRs_FPRs_ACCs

    def get_experiment_TPRs_FPRs_ACCs_nonzero(self, symm_mode:EvalMetrics.SymmModes, experiment:Experiment) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        TPRs = np.zeros((experiment.samples_per_batch, self.nr_batches))
        FPRs = np.zeros((experiment.samples_per_batch, self.nr_batches))
        ACCs = np.zeros((experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(experiment.samples_per_batch):
                theta_orig = self.get_experiment_theta_orig(sample_nr, batch_nr, experiment)
                theta_fit = self.get_experiment_theta_fit(sample_nr, batch_nr, experiment)

                TPR_sample, FPR_sample, ACC_sample = EvalMetrics.calculate_tpr_fpr_acc_nonzero(theta_orig, theta_fit, symm_mode)
                TPRs[sample_nr, batch_nr] = TPR_sample
                FPRs[sample_nr, batch_nr] = FPR_sample
                ACCs[sample_nr, batch_nr] = ACC_sample
        return np.array(TPRs), np.array(FPRs), np.array(ACCs)

    def get_all_edge_sign_recalls(self, symm_mode:EvalMetrics.SymmModes) -> List[np.ndarray]:
        edge_sign_recalls = []
        for experiment in self.experiments:
            edge_sign_recalls.append(self.get_experiment_edge_sign_recalls(symm_mode, experiment))
        return edge_sign_recalls

    def get_experiment_edge_sign_recalls(self, symm_mode:EvalMetrics.SymmModes, experiment:Experiment) -> np.ndarray:
        edge_sign_recalls = np.zeros((experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(experiment.samples_per_batch):
                theta_orig = self.get_experiment_theta_orig(sample_nr, batch_nr, experiment)
                theta_fit = self.get_experiment_theta_fit(sample_nr, batch_nr, experiment)

                edge_sign_recalls[sample_nr, batch_nr] = EvalMetrics.calculate_edge_sign_recall(theta_orig, theta_fit, symm_mode)
        return np.array(edge_sign_recalls)

    def get_all_MSEs(self, symm_mode:EvalMetrics.SymmModes) -> List[Tuple[np.ndarray, np.ndarray]]:
        experiment_MSEs = []
        for experiment in self.experiments:
            experiment_MSEs.append(self.get_experiment_MSEs(symm_mode, experiment))
        return experiment_MSEs

    def get_experiment_MSEs(self, symm_mode:EvalMetrics.SymmModes, experiment:Experiment) -> Tuple[np.ndarray, np.ndarray]:
        MSEs = np.zeros((experiment.samples_per_batch, self.nr_batches))
        diag_MSEs = np.zeros((experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(experiment.samples_per_batch):
                theta_orig = self.get_experiment_theta_orig(sample_nr, batch_nr, experiment)
                theta_fit = self.get_experiment_theta_fit(sample_nr, batch_nr, experiment)

                sample_MSE, sample_diag_MSE = EvalMetrics.calculate_MSEs(theta_orig, theta_fit, symm_mode)
                MSEs[sample_nr, batch_nr] = sample_MSE
                diag_MSEs[sample_nr, batch_nr] = sample_diag_MSE

        return MSEs, diag_MSEs

    def get_all_KL_divergences(self, sampler:GibbsSampler, experiment:Experiment) -> List[np.ndarray]:
        KL_divergences = []
        for experiment in self.experiments:
            KL_divergences.append(self.get_experiment_KL_divergences(sampler, experiment))
        return KL_divergences

    def get_experiment_KL_divergences(self, sampler:GibbsSampler, experiment:Experiment) -> np.ndarray:
        node_KL_divergences = np.zeros((experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(experiment.samples_per_batch):
                model_init_P = self.get_experiment_sample_model(sample_nr, batch_nr, experiment)
                model_fit_Q = self.get_experiment_fit_model(sample_nr, batch_nr, experiment)

                node_KL_divergences[sample_nr, batch_nr] = EvalMetrics.node_cond_prob_KL_divergence(model_init_P, model_fit_Q, sampler)[0]

        return node_KL_divergences

    def get_all_avg_node_fit_times(self) -> List[np.ndarray]:
        avg_node_fit_times = []
        for experiment in self.experiments:
            avg_fit_times = np.zeros((experiment.samples_per_batch, self.nr_batches))

            for batch_nr in range(self.nr_batches):
                for sample_nr in range(experiment.samples_per_batch):
                    fit_name = experiment.get_fit_name(batch_nr, sample_nr)
                    FPS = FitParamsWrapper.load_fit(fit_name, experiment.fit_file_name)

                    if FPS.avg_node_fit_time is not None:
                        avg_fit_times[sample_nr, batch_nr] = FPS.avg_node_fit_time
            avg_node_fit_times.append(avg_fit_times)
        return avg_node_fit_times

    def get_all_fit_sparsities_and_symms(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        all_fit_sas = []
        for experiment in self.experiments:
            all_fit_sas.append(self.get_fit_sparsities_and_symms(experiment))
        return all_fit_sas


    def get_fit_sparsities_and_symms(self, experiment:Experiment) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        symm_nonzero = np.zeros((experiment.samples_per_batch, self.nr_batches))
        symm_values = np.zeros((experiment.samples_per_batch, self.nr_batches))
        symm_signs = np.zeros((experiment.samples_per_batch, self.nr_batches))
        sparsities = np.zeros((experiment.samples_per_batch, self.nr_batches))

        for batch_nr in range(self.nr_batches):
            for sample_nr in range(experiment.samples_per_batch):
                theta_fit = self.get_experiment_theta_fit(sample_nr, batch_nr, experiment)

                symm_nonzero[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_symmetric_nonzero(theta_fit)
                symm_values[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_symmetric_values(theta_fit)
                symm_signs[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_symmetric_signs(theta_fit)
                sparsities[sample_nr, batch_nr] = EvalMetrics.calculate_percentage_sparsity(theta_fit)

        return sparsities, symm_nonzero, symm_signs, symm_values

    def get_all_experiment_nr_iterations(self) -> List[np.ndarray]:
        all_experiment_iter = []
        for experiment in self.experiments:
            all_experiment_iter.append(self.get_experiment_nr_iterations(experiment))
        return all_experiment_iter

    def get_experiment_nr_iterations(self, experiment:Experiment) -> np.ndarray:
        nr_iterations = np.zeros((experiment.samples_per_batch, self.nr_batches))
        for batch_nr in range(self.nr_batches):
            for sample_nr in range(experiment.samples_per_batch):
                nr_iterations[sample_nr, batch_nr] = self.get_experiment_avg_nr_iterations_fit(sample_nr, batch_nr, experiment)
        return nr_iterations


    # Save folder is to the latitude of the caller.
    @staticmethod
    def create_experiment_plot_folder(folder_name):
        if not os.path.exists('../' + folder_name):
            os.makedirs('../' + folder_name)

    def plot_ys_common_xs(self,
                          xs:List[Any],
                          list_ys:List[np.ndarray],
                          ys_errors:List[np.ndarray],
                          x_name:str,
                          y_name:str,
                          plot_title:str,
                          x_log_scale:bool):
        marker = cycle(('o', '^', 's', 'X', '*'))
        for ii, ys in enumerate(list_ys):
            if len(ys_errors) == 0:
                plt.plot(xs, ys, linestyle='--', marker=next(marker), label=self.experiment_labels[ii])
            else:
                plt.errorbar(xs, ys, yerr=ys_errors[ii], marker=next(marker), label=self.experiment_labels[ii])

        if x_log_scale:
            plt.xscale('log', basex=2)

        plt.title(plot_title)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend()

        new_title = y_name + ' vs ' + x_name + ": " + plot_title
        plt.savefig('../' + self.folder_name + '/' + new_title + '.png')
        plt.close()

        print('Plotting ' + plot_title + ', ' + y_name + ' vs ' + x_name +  ' finished!')

    def plot_xs_ys(self,
                   list_xs: List[np.ndarray],
                   list_ys: List[np.ndarray],
                   ys_errors: List[np.ndarray],
                   x_name: str,
                   y_name: str,
                   plot_title: str,
                   x_log_scale: bool
                   ):
        marker = cycle(('o', '^', 's', 'X', '*'))
        for ii in range(len(list_ys)):
            xs = list_xs[ii]
            ys = list_ys[ii]
            if len(ys_errors) == 0:
                plt.plot(xs, ys, linestyle='--', marker=next(marker), label=self.experiment_labels[ii])
            else:
                plt.errorbar(xs, ys, yerr=ys_errors[ii], marker=next(marker), label=self.experiment_labels[ii])

        if x_log_scale:
            plt.xscale('log', basex=2)

        plt.title(plot_title)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend()

        new_title = y_name + ' vs ' + x_name + ": " + plot_title
        plt.savefig('../' + self.folder_name + '/' + new_title + '.png')
        plt.close()

        print('Plotting ' + plot_title + ', ' + y_name + ' vs ' + x_name +  ' finished!')


    def plot_TPRs_FPRs_ACCs(self, plot_title, x_name, xs:List[Any], x_log_scale:Optional[bool]=False):
        ww_min = EvalMetrics.SymmModes.WW_MIN
        ww_max = EvalMetrics.SymmModes.WW_MAX
        symm_none = EvalMetrics.SymmModes.NONE

        symm_modes = [ww_min, ww_max, symm_none]
        symm_modes_names = ['ww_min', 'ww_max', 'symm_none']

        for ii in range(len(symm_modes)):
            symm_mode = symm_modes[ii]
            symm_name = symm_modes_names[ii]

            TPRs_FPRs_ACCs_list = self.get_all_TPRs_FPRs_ACCs_nonzero(symm_mode)
            TPRs_means = []
            TPRs_stds = []
            FPRs_means = []
            FPRs_stds = []
            ACCs_means = []
            ACCs_stds = []
            for TPRs, FPRs, ACCs in TPRs_FPRs_ACCs_list:
                TPRs_means.append(np.mean(TPRs, axis=0))
                TPRs_stds.append(np.std(TPRs, axis=0))
                FPRs_means.append(np.mean(FPRs, axis=0))
                FPRs_stds.append(np.std(FPRs, axis=0))
                ACCs_means.append(np.mean(ACCs, axis=0))
                ACCs_stds.append(np.std(ACCs, axis=0))

            # Plot TPRs.
            self.plot_ys_common_xs(xs=xs,
                                   list_ys=TPRs_means,
                                   ys_errors=TPRs_stds,
                                   x_name=x_name,
                                   y_name='TPR',
                                   plot_title=plot_title + ": symm_mode=" + symm_name,
                                   x_log_scale=x_log_scale)

            # Plot FPRs.
            self.plot_ys_common_xs(xs=xs,
                                   list_ys=FPRs_means,
                                   ys_errors=FPRs_stds,
                                   x_name=x_name,
                                   y_name='FPR',
                                   plot_title=plot_title + ": symm_mode=" + symm_name,
                                   x_log_scale=x_log_scale)

            #Plot ACCs.
            self.plot_ys_common_xs(xs=xs,
                                   list_ys=ACCs_means,
                                   ys_errors=ACCs_stds,
                                   x_name=x_name,
                                   y_name='ACC',
                                   plot_title=plot_title + ": symm_mode=" + symm_name,
                                   x_log_scale=x_log_scale)

            # Plot TPRs vs FPRs for each experiment.
            self.plot_xs_ys(list_xs=FPRs_means,
                            list_ys=TPRs_means,
                            ys_errors=[],
                            x_name='FPR',
                            y_name='TPR',
                            plot_title=plot_title + ' ROC curves: ' + symm_name,
                            x_log_scale=False)

            # TODO: remove (debugging).
            print(TPRs_means)
            print(FPRs_means)


    def plot_edge_sign_recalls(self, plot_title:str, x_name:str, xs:List[Any], x_log_scale:Optional[bool]=False):
        ww_min = EvalMetrics.SymmModes.WW_MIN
        ww_max = EvalMetrics.SymmModes.WW_MAX
        symm_none = EvalMetrics.SymmModes.NONE

        symm_modes = [ww_min, ww_max, symm_none]
        symm_modes_names = ['ww_min', 'ww_max', 'symm_none']

        for ii in range(len(symm_modes)):
            symm_mode = symm_modes[ii]
            symm_name = symm_modes_names[ii]
            edge_sign_recalls = self.get_all_edge_sign_recalls(symm_mode)
            edge_sign_means = []
            edge_sign_stds = []
            for x in edge_sign_recalls:
                edge_sign_means.append(np.mean(x, axis=0))
                edge_sign_stds.append(np.std(x, axis=0))

            self.plot_ys_common_xs(xs=xs,
                                   list_ys=edge_sign_means,
                                   ys_errors=edge_sign_stds,
                                   x_name=x_name,
                                   y_name='edge_sign_recall',
                                   plot_title=plot_title + ": symm_mode=" + symm_name,
                                   x_log_scale=x_log_scale)


    def plot_MSEs(self, plot_title:str, x_name:str, xs:List[Any], x_log_scale:Optional[bool]=False):
        ww_min = EvalMetrics.SymmModes.WW_MIN
        ww_max = EvalMetrics.SymmModes.WW_MAX
        symm_none = EvalMetrics.SymmModes.NONE

        symm_modes = [ww_min, ww_max, symm_none]
        symm_modes_names = ['ww_min', 'ww_max', 'symm_none']

        for ii in range(len(symm_modes)):
            symm_mode = symm_modes[ii]
            symm_name = symm_modes_names[ii]

            MSEs_means = []
            MSEs_stds = []
            diag_MSEs_means = []
            diag_MSEs_stds = []
            MSEs_diag_MSEs_list = self.get_all_MSEs(symm_mode)
            for MSEs, diag_MSEs in MSEs_diag_MSEs_list:
                MSEs_means.append(np.mean(MSEs, axis=0))
                MSEs_stds.append(np.std(MSEs, axis=0))
                diag_MSEs_means.append(np.mean(diag_MSEs, axis=0))
                diag_MSEs_stds.append(np.std(diag_MSEs, axis=0))

            self.plot_ys_common_xs(xs=xs,
                                   list_ys=MSEs_means,
                                   ys_errors=MSEs_stds,
                                   x_name=x_name,
                                   y_name='MSE',
                                   plot_title=plot_title + ": symm_mode=" + symm_name,
                                   x_log_scale=x_log_scale)

            self.plot_ys_common_xs(xs=xs,
                                   list_ys=diag_MSEs_means,
                                   ys_errors=diag_MSEs_stds,
                                   x_name=x_name,
                                   y_name='diag_MSE',
                                   plot_title=plot_title + ": symm_mode=" + symm_name,
                                   x_log_scale=x_log_scale)


    def plot_sparsities_and_symms(self, plot_title:str, x_name:str, xs:List[Any], x_log_scale:Optional[bool]=False):
        sparsities_means = []
        sparsities_stds = []
        symm_nonzero_means = []
        symm_nonzero_stds = []
        symm_signs_means = []
        symm_signs_stds = []
        symm_values_means = []
        symm_values_stds = []
        all_list = self.get_all_fit_sparsities_and_symms()
        for sparsities, symm_nonzero, symm_signs, symm_values in all_list:
            sparsities_means.append(np.mean(sparsities, axis=0))
            sparsities_stds.append(np.std(sparsities, axis=0))
            symm_nonzero_means.append(np.mean(symm_nonzero, axis=0))
            symm_nonzero_stds.append(np.std(symm_nonzero, axis=0))
            symm_signs_means.append(np.mean(symm_signs, axis=0))
            symm_signs_stds.append(np.std(symm_signs, axis=0))
            symm_values_means.append(np.mean(symm_values, axis=0))
            symm_values_stds.append(np.std(symm_values, axis=0))

        self.plot_ys_common_xs(xs=xs,
                               list_ys=sparsities_means,
                               ys_errors=sparsities_stds,
                               x_name=x_name,
                               y_name='sparsity',
                               plot_title=plot_title,
                               x_log_scale=x_log_scale)
        self.plot_ys_common_xs(xs=xs,
                               list_ys=symm_nonzero_means,
                               ys_errors=symm_nonzero_stds,
                               x_name=x_name,
                               y_name='symm_nonzero',
                               plot_title=plot_title,
                               x_log_scale=x_log_scale)
        self.plot_ys_common_xs(xs=xs,
                               list_ys=symm_signs_means,
                               ys_errors=symm_signs_stds,
                               x_name=x_name,
                               y_name='symm_signs',
                               plot_title=plot_title,
                               x_log_scale=x_log_scale)
        self.plot_ys_common_xs(xs=xs,
                               list_ys=symm_values_means,
                               ys_errors=symm_values_stds,
                               x_name=x_name,
                               y_name='symm_values',
                               plot_title=plot_title,
                               x_log_scale=x_log_scale)

    def plot_nr_iterations(self, plot_title:str, x_name:str, xs:List[Any], x_log_scale:Optional[bool]=False):
        nr_iterations_list = self.get_all_experiment_nr_iterations()
        nr_iterations_means = []
        nr_iterations_stds = []
        for nr_iterations in nr_iterations_list:
            nr_iterations_means.append(np.mean(nr_iterations, axis=0))
            nr_iterations_stds.append(np.std(nr_iterations, axis=0))

        self.plot_ys_common_xs(xs=xs,
                               list_ys=nr_iterations_means,
                               ys_errors=nr_iterations_stds,
                               x_name=x_name,
                               y_name='Average iterations',
                               plot_title=plot_title,
                               x_log_scale=x_log_scale)

    def plot_fit_times(self, plot_title:str, x_name:str, xs:List[Any], x_log_scale:Optional[bool]=False):
        avg_node_fit_times_list = self.get_all_avg_node_fit_times()
        means = []
        stds = []
        for avg_node_fit_times in avg_node_fit_times_list:
            means.append(np.mean(avg_node_fit_times, axis=0))
            stds.append(np.std(avg_node_fit_times, axis=0))
        self.plot_ys_common_xs(xs=xs,
                               list_ys=means,
                               ys_errors=stds,
                               x_name=x_name,
                               y_name='Average fit times',
                               plot_title=plot_title,
                               x_log_scale=x_log_scale)

    def plot_ALL(self, plot_title:str, x_name:str, xs:List[Any], x_log_scale:Optional[bool]=False,
                 SIPRV_active:Optional[bool]=False):
        kwargs = dict(**locals())
        del kwargs['self']
        del kwargs['SIPRV_active']
        self.plot_TPRs_FPRs_ACCs(**kwargs)
        self.plot_sparsities_and_symms(**kwargs)
        self.plot_nr_iterations(**kwargs)
        self.plot_fit_times(**kwargs)

        if not SIPRV_active:
            self.plot_edge_sign_recalls(**kwargs)
            self.plot_MSEs(**kwargs)
