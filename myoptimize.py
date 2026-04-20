"""
Optimization function for CSOmap 3D reconstruction.
Equivalent to MATLAB myoptimize.m
Inspired from t-SNE gradient descent.
"""
import numpy as np


def myoptimize(P, no_dims=3, condition='tight', verbose=True):
    """
    Optimize the target function using gradient descent.

    Parameters
    ----------
    P : np.ndarray
        Affinity matrix (n x n)
    no_dims : int or np.ndarray
        Number of dimensions (default 3) or initial solution
    condition : str
        'loose' or 'tight'. Use 'loose' for datasets with >10000 cells.
    verbose : bool
        Print progress

    Returns
    -------
    coord : np.ndarray
        Optimized coordinates
    process : np.ndarray
        Coordinates during iterations
    """
    # Check if initial solution is provided
    if isinstance(no_dims, np.ndarray) and no_dims.ndim > 0 and no_dims.shape[0] == P.shape[0]:
        coord = no_dims.astype(float)
        no_dims = coord.shape[1]
        initial_solution = True
    else:
        initial_solution = False
        no_dims = int(no_dims)

    n = P.shape[0]
    momentum = 0.5
    final_momentum = 0.8
    mom_switch_iter = 250
    max_iter = 1000
    epsilon = 1000.0
    min_gain = 0.01
    process = []

    # Prepare P
    np.fill_diagonal(P, 0)
    P = 0.5 * (P + P.T)
    P = np.maximum(P / P.sum(), np.finfo(float).tiny)
    const = np.sum(P * np.log(P))

    # Initialize solution
    if not initial_solution:
        coord = (np.random.rand(n, no_dims) - 0.5) * 50.0

    incs = np.zeros_like(coord)
    gains = np.ones_like(coord)

    for iter in range(1, max_iter + 1):
        # Compute joint probability that point i and j are neighbors
        sum_current = np.sum(coord ** 2, axis=1, keepdims=True)
        d = sum_current + sum_current.T - 2.0 * (coord @ coord.T)
        num = 1.0 / (1.0 + d)
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), np.finfo(float).tiny)

        # Compute gradients
        P_Q = P - Q
        P_Q[(P_Q > 0) & (d <= 0.01)] = -0.01
        L = P_Q * num
        grads = 4.0 * (np.diag(L.sum(axis=1)) - L) @ coord

        # Update solution
        gains = (gains + 0.2) * (np.sign(grads) != np.sign(incs)) + \
                (gains * 0.8) * (np.sign(grads) == np.sign(incs))
        gains[gains < min_gain] = min_gain
        incs = momentum * incs - epsilon * (gains * grads)
        coord = coord + incs
        coord = coord - coord.mean(axis=0, keepdims=True)

        # Update momentum
        if iter == mom_switch_iter:
            momentum = final_momentum

        # Print progress
        if verbose and iter % 10 == 0:
            cost = const - np.sum(P * np.log(Q))
            print(f'Iteration {iter}: error is {cost:.4f}')

        # Rescale to 50
        range_val = np.max(np.abs(coord))
        if condition == 'tight':
            if range_val > 50 and iter % 10 == 0:
                coord = coord * 50.0 / range_val
        else:
            if range_val > 50 and iter % 1000 == 0:
                coord = coord * 50.0 / range_val

        process.append(coord.copy())

    process = np.hstack(process)  # shape: (n, no_dims * max_iter)
    return coord, process


if __name__ == '__main__':
    # Simple test
    n = 100
    P = np.random.rand(n, n)
    P = (P + P.T) / 2.0
    np.fill_diagonal(P, 0)
    coord, process = myoptimize(P, 3, 'tight')
    print(coord.shape, process.shape)
