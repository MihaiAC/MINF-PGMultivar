from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter

import numpy as np


def soft_thresholding_prox_operator(x, threshold):
    x_plus = x - threshold
    x_minus = x + threshold

    return x_plus * (x_plus > 0) + x_minus * (x_minus < 0)

def ackley_function(node:int, data_points:np.ndarray, theta:np.ndarray) -> float:
    x = theta[0]
    y = theta[1]
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - \
           np.exp(0.5 * (np.cos(2 * x * np.pi) + np.cos(2 * y * np.pi))) + np.exp(1) + 20

def grad_ackley_function(node:int, data_points:np.ndarray, theta:np.ndarray) -> np.ndarray:
    x = theta[0]
    y = theta[1]
    grad_x = 2 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) * x /np.sqrt(0.5 * (x**2 + y**2)) + \
        np.pi * np.sin(2 * np.pi * x) * np.exp(0.5 * (np.cos(2 * x * np.pi) + np.cos(2 * y * np.pi)))
    grad_y = 2 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) * y / np.sqrt(0.5 * (x ** 2 + y ** 2)) + \
             np.pi * np.sin(2 * np.pi * y) * np.exp(0.5 * (np.cos(2 * x * np.pi) + np.cos(2 * y * np.pi)))
    return np.array([grad_x, grad_y])

def spherical_function(node:int, data_points:np.ndarray, theta:np.ndarray) -> float:
    x = theta[0]
    y = theta[1]
    return x ** 2 + y ** 2

def grad_spherical_function(node:int, data_points:np.ndarray, theta:np.ndarray) -> np.ndarray:
    x = theta[0]
    y = theta[1]
    grad_x = 2 * x
    grad_y = 2 * y
    return np.array([grad_x, grad_y])




if __name__ == "__main__":
    # samples_file_name = "samples.sqlite"
    # samples_id = "PleaseWork"

    # fit_id = "debugProxGrad"
    # fit_file_name = "fit_models.sqlite"

    # SPS = SampleParamsWrapper.load_samples(samples_id, samples_file_name)
    # print(SPS.model_params)

    # a = np.array([2, 4, 5, 0.5, -0.95, -1.25])
    # print(soft_thresholding_prox_operator(a, 1))
    pgf = Prox_Grad_Fitter(0)
    # print(pgf.fit_node(0, ackley_function, grad_ackley_function, None, np.array([[4, 4], [1, 1]]))[0])
    print(pgf.fit_node(0, spherical_function, grad_spherical_function, None, np.array([[4, 4], [1, 1]]))[0])