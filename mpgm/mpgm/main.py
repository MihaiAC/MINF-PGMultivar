import numpy as np
import os
import argparse
from mpgm.mpgm.experiment_wrappers import Fit_Wrapper
from mpgm.mpgm.experiment_wrappers import QPGM_FitWrapper
from mpgm.mpgm.experiment_wrappers import TPGM_FitWrapper
from mpgm.mpgm.experiment_wrappers import SPGM_FitWrapper
from mpgm.mpgm.experiment_wrappers import LPGM_FitWrapper
from mpgm.mpgm.samplers import Sampler
from mpgm.mpgm.samplers import generate_model_samples


parser = argparse.ArgumentParser(description="Parsing script arguments.")
parser.add_argument('--mode', type=str, help='Mode in which the package will be used. Available options: sample|fit|experiment')
parser.add_argument('--model', type=str, help='TPGM|QPGM|SPGM|LPGM')
parser.add_argument('--random_seed', type=int, help='Random seed to be used.')

# Arguments for mode="sample" and mode="experiment".
parser.add_argument('--samples_name', type=str, help='Names of the samples to be generated | used.')
parser.add_argument('--graph_generator', type=str, default=None, help='scale_free|random|lattice|hub')
parser.add_argument('--nr_variables', type=int, default=None, help='Number of nodes in the graph.')
# - graph generators arguments -
parser.add_argument('--graph_generator', type=str, default=None, help="scale_free|lattice|hub|random")
parser.add_argument('--scale_free_alpha', type=float, default=0.1, help='Parameter for the scale free graph generator.')
parser.add_argument('--scale_free_beta', type=float, default=0.8, help='Parameter for the scale free graph generator.')
parser.add_argument('--scale_free_gamma', type=float, default=0.1, help='Parameter for the scale gree graph generator.')
parser.add_argument('--lattice_sparsity_level', type=int, default=None, help='Parameter for the lattice graph generator.')
parser.add_argument('--hub_nr_hubs', type=int, default=3, help='Parameter for the hub graph generator.')
parser.add_argument('--random_nr_edges', type=int, default=2, help='Parameter for the random graph generator.')
# - weight assigners arguments -
parser.add_argument('--weight_assigner', type=str, default=None, help='bimodal')
parser.add_argument('--wa_bimodal_neg_mean', type=float, default=-0.04, help='See assign_weights_bimodal_distr.')
parser.add_argument('--wa_bimodal_pos_mean', type=float, default=0.04, help='See assign_weights_bimodal_distr.')
parser.add_argument('--wa_bimodal_std', type=float, default=0.03, help='See assign_weights_bimodal_distr.')
parser.add_argument('--wa_bimodal_neg_threshold', type=float, default=0.5, help='See assign_weights_bimodal_distr.')
# -Gibbs sampler arguments -
parser.add_argument('--nr_samples', type=int, default=50, help="Number of samples to generate (per sampling process).")
parser.add_argument('--gibbs_burn_in', type=int, default=700, help="Number of samples to discard.")
parser.add_argument('--gibbs_thinning_nr', type=int, default=100, help="Thinning number for the Gibbs sampler.")

# - model args;
parser.add_argument('--R', type=int, default=10, help='Truncation value.')
parser.add_argument('--R0', type=int, default=5, help='R0 parameter used in the SPGM model')
# TODO: revise this.
parser.add_argument('--lpgm_m', type=int, default = 0, help='See LPGM model.')
parser.add_argument('--lpgm_B', type=float, default=0, help='See LPGM model.')
parser.add_argument('--lpgm_beta', type=float, default=0, help='See LPGM model.')
parser.add_argument('--nr_alphas', type=int, default=0, help='See LPGM model.')

# Arguments for mode='fit' and mode='experiment'.
parser.add_argument('--experiment_name', type=str, default=None, help="Name under which the parameters resulting from a "
                                                                      "fit procedure are saved.")
parser.add_argument('--alpha', type=float, default=1, help="Learning rate.")
parser.add_argument('--accelerated', type=str, default="True", help="Whether to use accelerated proximal gradient descent or normal proximal gradient descent.")
parser.add_argument('--max_line_search_iter', type=int, default=50, help="Prox grad argument.")
parser.add_argument('--max_iter', type=int, default=5000, help="Prox grad argument.")
parser.add_argument('--line_search_rel_tol', type=float, default=1e-4, help="Prox grad argument.")
parser.add_argument('--prox_grad_lambda_p', type=float, default=1.0)
parser.add_argument('--prox_grad_beta', type=float, default=0.5)
parser.add_argument('--prox_grad_rel_tol', type=float, default=1e-3)
parser.add_argument('--prox_grad_abs_tol', type=float, default=1e-6)
parser.add_argument('--early_stop_criterion', type=str, default="weight",
                    help="Early stop criterion. Select from: weight|likelihood convergence.")
parser.add_argument('--prox_method', type=str, default='SLSQP', help='See docstring of SLPGM.prox_grad. Choose from: '
                                                                     'SLSQP|COBYLA')


# TODO: also need to add an argument for the convergence criterion chosen + modify the prox grad function accordingly.
# TODO: I'm forgetting about a prox grad argument. If you remember, add it here.

parser.add_argument('--')

args = parser.parse_args()
print('Success!' + args.mode)

