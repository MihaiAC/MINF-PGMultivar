import numpy as np
import pickle as pkl
from mpgm.mpgm.TPGM import TPGM

def calculate_tpr_fpr(theta_orig, theta_fit):
    nr_variables = theta_orig.shape[0]

    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for ii in range(1, nr_variables):
        for kk in range(ii):
            real_edge = theta_orig[ii][kk] != 0
            inferred_edge = abs(theta_fit[ii][kk]) >= 1e-2 or abs(theta_fit[kk][ii]) >= 1e-2

            if (real_edge and inferred_edge):
                TP += 1
            elif (real_edge and not inferred_edge):
                FN += 1
            elif (not real_edge and not inferred_edge):
                TN += 1
            else:
                FP += 1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    print((TP + TN) / (TP + FP + TN + FN))
    return TPR, FPR

samples_loc = 'TPGM_test'
exp_name = 'TPGM_fit_test'
samples = np.load('Samples/' + samples_loc + '/samples.npy')
theta_orig = np.load('Samples/' + samples_loc + '/generator_model_theta.npy')
theta_fit = np.load('Samples/' + samples_loc + '/' + exp_name + '/fit_model_theta.npy')

model = TPGM(theta_orig[0, :], R=10)
print(model.calculate_nll(0, samples, theta_orig[0, :]))


print(theta_orig)
print(theta_fit)
print(calculate_tpr_fpr(theta_orig, theta_fit))