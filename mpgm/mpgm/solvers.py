import numpy as np
import tqdm


# TODO: Check what to do with theta st and theta ts not agreeing. Can you do anything?
# TODO: Check what to do with theta ss (node parameters). Include them too after testing?
def prox_grad(node, theta_init, alpha, data, f, f_and_grad_f, params_for_f_grad_f, condition=None, max_iter=5000,
              max_line_search_iter=50, lambda_p=1.0, beta=0.5, rel_tol=1e-3, abs_tol=1e-6):
    # TODO: Add references.
    """
    Proximal gradient descent for solving the l1-regularized node-wise regressions required to fit some models in this
        package.

    :param f_and_grad_f: Function which computes f and grad_f at the same time (lower cost). grad_f is the gradient
        of the NLL of the node-conditional distribution.
    :param f: Computes negative log-likelihood of the node-conditional distribution.
    :param params_for_f_grad_f: parameters to be unpacked as the last argument for every f, f_and_grad_f calls;
        (e.g: [R] for TPGM, [R, R0] for SPGM.
    :param condition: condition to be checked at every iteration; should accept: node, theta_current, data as arguments
        in this order.
    :param data:
    :param theta_init: Initial parameter values.
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
        # TODO: recheck this.
        f_y_k, grad_y_k = f_and_grad_f(node, data, y_k, *params_for_f_grad_f)

        sw = False
        for _ in range(max_line_search_iter):
            z = soft_threshold(y_k - lambda_k * grad_y_k, threshold=lambda_k * alpha)
            f_tilde = f_y_k + np.dot(grad_y_k, z - y_k) + (1 / (2 * lambda_k)) * (np.linalg.norm(z - y_k) ** 2)
            f_z = f(node, data, z, *params_for_f_grad_f)  # NLL at current step.
            # TODO: use decimal?
            if f_z <= f_tilde:
                sw = True
                break
            else:
                lambda_k = lambda_k * beta
        if sw:
            theta_k_2 = theta_k_1
            theta_k_1 = z
            likelihoods[k] = f_z

            if condition is not None:
                conditions[k] = condition(node, z, data)

            if (k > 2 and (np.abs(f_z - likelihoods[k - 1]) < abs_tol or
                           (np.abs(f_z - likelihoods[k - 1]) / min(np.abs(f_z),
                                                                   np.abs(likelihoods[k - 1]))) < rel_tol)):
                converged = True
                break
        else:
            print('Line search failed to converge for node ' + str(node))
            break
    print('prox_grad for node ' + str(node) + ' has finished. Converged?: ' + str(converged))
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
