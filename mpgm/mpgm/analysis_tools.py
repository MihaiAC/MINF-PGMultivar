import numpy as np
import pickle as pkl

def calculate_tpr_fpr(theta_orig, theta_fit):
    nr_variables = theta_orig.shape[0]

    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for ii in range(1, nr_variables):
        for kk in range(ii):
            real_edge = theta_orig[ii][kk] != 0
            inferred_edge = abs(theta_fit[ii][kk]) >= 1e-4 or abs(theta_fit[kk][ii]) >= 1e-4

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

samples = np.load('Samples/test/samples.npy')
with open('Samples/test/TPGM_fit_test/likelihoods.pkl', 'rb') as f:
    likelihoods = pkl.load(f)
print(likelihoods[0])
print(likelihoods[1])
theta_orig = np.load('Samples/test/generator_model_theta.npy')
theta_fit = np.load('Samples/test/TPGM_fit_test/fit_model_theta.npy')
print(theta_orig)
print(theta_fit)
print(calculate_tpr_fpr(theta_orig, theta_fit))