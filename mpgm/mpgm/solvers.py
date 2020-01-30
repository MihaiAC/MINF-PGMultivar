import numpy as np


def prox_grad(node, model, data, alpha, max_iter=5000, max_line_search_iter=50, lambda_p=1.0, beta=0.5, rel_tol=1e-3,
              abs_tol=1e-6):
    """
    Proximal gradient descent for solving the l1-regularized node-wise regressions required to fit some models in this
        package.
    :param model: Model to be fit by prox_grad. Must implement "calculate_nll_and_grad_nll", "calculate_nll" and a
        condition function to be evaluated after each iteration (necessary for QPGM, SPGM) as per the Model "interface".
    :param data: Data used to fit the model.
    :param alpha: L1 regularization parameter.
    :param node: The node we're doing regression on.
    :param max_iter: Maximum iterations allowed for the algorithm.
    :param max_line_search_iter: Maximum iterations allowed for the line search performed at every iteration step.
    :param lambda_p: Initial step-size.
    :param beta: Line search parameter.
    :param rel_tol: Relative tolerance value for early stopping.
    :param abs_tol: Absolute tolerance value for early stopping. Should be 1e-3 * rel_tol.
    :return: (parameters, likelihood_values, converged) - tuple containing parameters and the NLL value for each
        iteration;
    """

    condition = model.condition
    f = model.calculate_nll
    f_and_grad_f = model.calculate_nll_and_grad_nll
    theta_init = model.theta

    likelihoods = np.zeros((max_iter, ))

    conditions = None
    if condition is not None:
        conditions = np.zeros((max_iter, ))

    theta_k_2 = np.array(theta_init)
    theta_k_1 = np.array(theta_init)
    lambda_k = lambda_p

    converged = False

    z = np.zeros(np.size(theta_init))
    f_z = 0

    for k in range(1, max_iter + 1):
        w_k = k / (k + 3)
        y_k = theta_k_1 + w_k * (theta_k_1 - theta_k_2)
        f_y_k, grad_y_k = f_and_grad_f(node, data, y_k)

        sw = False
        for _ in range(max_line_search_iter):
            z = soft_threshold(y_k - lambda_k * grad_y_k, threshold=lambda_k * alpha)
            f_tilde = f_y_k + np.dot(grad_y_k, z - y_k) + (1 / (2 * lambda_k)) * (np.linalg.norm(z - y_k) ** 2)
            f_z = f(node, data, z)  # NLL at current step.
            if f_z <= f_tilde:
                sw = True
                break
            else:
                lambda_k = lambda_k * beta
        if sw:
            theta_k_2 = theta_k_1
            theta_k_1 = z
            likelihoods[k-1] = f_z

            if condition is not None:
                conditions[k-1] = condition(node, z, data)

            if (k > 2 and (np.abs(f_z - likelihoods[k - 2]) < abs_tol or
                           (np.abs(f_z - likelihoods[k - 2]) / min(np.abs(f_z),
                                                                   np.abs(likelihoods[k - 2]))) < rel_tol)):
                converged = True
                break
        else:
            break
    return theta_k_1, likelihoods, conditions, converged


def soft_threshold(x, threshold):
    """
    Applies the soft thresholding operation to vector x with the specified threshold.

    :param x:
    :param threshold:
    :return:
    """
    x_plus = x - threshold
    x_minus = -x - threshold
    return x_plus * (x_plus > 0) - x_minus * (x_minus > 0)
