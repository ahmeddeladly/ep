# clutter.py
#
# Implementation of the Clutter Problem for Expectation Propagation (Minka 2001)
# as described in Expectation Propagation for approximate Bayesian inference,
# Minka, 2001.
#
# The goal is to infer the mean, theta, of a multivariate Gaussian distribution
# over a variable, x, given a set of observations drawn from that distribution.
# To make the problem interesting, the observations are embedded in background
# clutter, which is itself Gaussian distributed. The distribution of the
# observed values, x, is therefore a mixture of Gaussians.
#
# Author: Alan Aberdeen

import numpy as np
from scipy.stats import bernoulli, norm
from tabulate import tabulate
import argparse
import json
import os

import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt


# Save EP iteration data
def save_EP_data(p_mean, p_var):
    asdfa = 1


def make_1d_clutter_data(
    n_observations: int,
    clutter_var: float,
    clutter_mean: float,
    clutter_ratio: float,
    target_mean: float = 3,
    target_var: float = 1,
) -> list[float]:
    """Generate 1d data for the clutter problem.

    The rate at which the clutter distribution is sampled from is determined
    by the clutter ratio.

    Args:
        n_observations: Number of observations.
        clutter_var: The clutter variance.
        clutter_mean: The clutter mean.
        clutter_ratio: The clutter ratio.
        target_mean: The true mean of the target distribution (i.e, the data).
            Defaults to 3.
        target_var: The true variance of the target distribution (i.e, the data).
            Defaults to 1.

    Returns:
        The sampled data.
    """
    N = n_observations
    w = clutter_ratio

    clutter = norm(clutter_mean, np.sqrt(clutter_var))
    target_dist = norm(target_mean, np.sqrt(target_var))

    sample_clutter = bernoulli(w).rvs

    samples = [
        (clutter.rvs() if sample_clutter() else target_dist.rvs()) for _ in range(N)
    ]

    return samples


def plot_data(theta, X, clutter_var, clutter_mean):
    """Plot the clutter problem data.

    Args:
        theta: The true target distribution mean.
        X: The data.
        clutter_var: The variance of the clutter distribution.
        clutter_mean: The mean of the clutter distribution.
    """

    fig, ax = plt.subplots(1, 1)

    # X axes scaling
    data_range = max(X) - min(X)
    left = min(X) - 0.2 * data_range
    right = max(X) + 0.2 * data_range
    x = np.linspace(left, right, 200)

    clutter = norm(clutter_mean, np.sqrt(clutter_var)).pdf(x)
    target = norm(theta, 1).pdf(x)
    sample_points = np.zeros(len(X))

    ax.plot(x, clutter, "r-", label="Clutter Distribution")
    ax.plot(x, target, "g-", label="Target Distribution")
    ax.plot(X, sample_points, "bx", label="Samples")
    ax.legend(loc="best", frameon=False)
    plt.show()


def plot_factor(X, n, fn_mean, fn_var, fn_s, cav_mean, cav_var, c_ratio, c_var):
    """Plot the gaussian factors.

    Args:
        X: sampled observations.
        n: current factor.
        fn_mean: Approximated factor mean.
        fn_var: Approximated factor variance.
        fn_s: Approximated factor scaler.
        cav_mean: Cavity mean.
        cav_var: Cavity variance.
        c_ratio: the known clutter ratio.
        c_var: the known clutter variance.

    """
    w = c_ratio
    a = c_var

    fig, ax = plt.subplots(1, 1)

    x = np.linspace(X[n] - 5, X[n] + 5, 200)

    f_true = list(map(true_factor(X[n], w, a), x))
    f_approx = list(map(approx_factor(fn_s, fn_mean, fn_var), x))
    cavity = norm(cav_mean, np.sqrt(cav_var)).pdf(x)

    ax.plot(x, f_true, "b-", label="True Factor", linewidth=4, alpha=0.5)
    ax.plot(x, f_approx, "r-", label="Approximate Factor")
    ax.plot(x, cavity, "g-", label="Cavity Distribution")
    ax.legend(loc="best", frameon=False)

    plt.ylim(ymax=(1.2 * max(f_true)))
    plt.show()


def true_factor(Xn: float, w: float, a: float) -> float:
    """Compute true factor distribution.

    Args:
        Xn: The data point.
        w: The problem's clutter ratio.
        a: The clutter ration variance.

    Returns:
        The true factor value.
    """
    return lambda x: ((1 - w) * gaussian(Xn, x, 1)) + (w * gaussian(Xn, 0, a))


def approx_factor(Sn, mn, vn) -> float:
    """Calculate approximate factor distribution.

    Args:
        Sn: Scale factor.
        mn: The mean.
        vn: The variance.

    Returns:
        The approximate factor value.
    """
    return lambda x: Sn * np.exp(-(1 / (2 * vn)) * ((x - mn) ** 2))


def Zi(Xn: float, cav_mean: float, cav_var: float, w: float, a: float) -> float:
    """Calculate the normalizing factor.

    Args:
        Xn: The data point.
        cav_mean: The mean of the cavity distribution computed for factor fn.
        cav_var: The variance of the cavity distribution computed for factor fn.
        w: The clutter ratio. Note this is a known quantity as per the original
            clutter factor problem.
        a: The clutter variance. Note this is a known quantity as per the original
            clutter factor problem.

    Returns:
        The normalizing factor.
    """
    return ((1 - w) * gaussian(Xn, cav_mean, (cav_var + 1))) + (w * gaussian(Xn, 0, a))


def gaussian(x, m, v) -> float:
    """Evaluate gaussian at specific point given mean and variance.

    Args:
        x: The data point.
        m: The mean.
        v: The variance.

    Returns:
        The probability under the Gaussian natural statistics.
    """
    return np.exp(-0.5 * ((x - m) ** 2) * (1 / v)) / ((abs(2 * np.pi * v)) ** 0.5)


# Save the generated data to a file
def save_data(theta, X):
    # Current working directory
    cwd = os.getcwd()

    # Configuration Variables
    c_var = approx_config["clutter_var"]
    c_mean = approx_config["clutter_mean"]

    data = {"target_dist": (theta, 1), "clutter_dist": (c_mean, c_var), "samples": X}

    # Write to file
    with open((cwd + "data.json"), "w") as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)


# Run the EP example procedure
def run_ep(
    observations: list[float],
    n_dimensions: int,
    clutter_ratio: float,
    clutter_mean: float,
    clutter_var: float,
    prior_mean: float,
    prior_var: float,
    max_iter: int,
    tolerance: float,
    interactive: bool,
):
    """Run EP approximation.

    Args:
        observations: The data points.
        n_dimensions: The dimension of the data points.
        clutter_ratio: The clutter ratio.
        clutter_mean: The mean of the clutter distribution.
        clutter_var: The variance of the clutter distribution.
        prior_mean: The approximation prior mean.
        prior_var: The approximation prior variance.
        max_iter: The number of iterations
        tolerance: The parameter change tolerance used to stop the approximation.
        interactive: Whether to plot the approximation at each iteration.

    Returns:
        The approximation of the target distribution mean.
    """
    N = len(observations)
    D = n_dimensions
    X = observations
    w = clutter_ratio

    # Initialise prior, this is the first factor, f0, and the EP update
    # algorithm does not change it's value.
    p_s = (2 * np.pi * prior_var) ** (-D / 2)

    # Initialise the factors 1...N to unity.
    # Will store all calculated values in order to visualize the factor
    # development. We just need to use a distribution from the exponential
    # family for this problem, a Gaussian is appropriate. This corresponds to:
    f_means = np.zeros((max_iter, N))
    f_vars = np.full((max_iter, N), np.inf)  # infinity variance.
    f_ss = np.ones((max_iter, N))

    # Initialise our estimate for the approximation to the true underlying
    # distribution, q(theta), to simply be the prior.
    est_means = np.full((max_iter, N), prior_mean)
    est_vars = np.full((max_iter, N), prior_var)
    est_mean = prior_mean
    est_var = prior_var

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialise storage for the cavity mean and variance.
    # Not necessary but will be useful for visualization of algorithm
    # progression.
    cavity_vars = np.full((max_iter, N), prior_var)
    cavity_means = np.full((max_iter, N), prior_mean)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    converged = False

    for iteration in range(max_iter):
        if converged:
            break

        for n in range(N):
            # moments for the current approximate factor
            fn_mean = f_means[iteration, n]
            fn_var = f_vars[iteration, n]

            # create the cavity distribution by removing the estimate for the
            # current factor from the current posterior. This is achieved
            # with the division operation for Gaussians.
            cav_var = 1 / ((1 / est_var) - (1 / fn_var))
            cav_mean = est_mean + (cav_var * (1 / fn_var) * (est_mean - fn_mean))

            # compute the new estimate for the posterior by multiply the
            # cavity distribution by the approximate factor distribution.
            # When you are multiplying to distributions of the exponential
            # family you can follow the known formula to find the defining
            # moments of the new distribution.

            # evaluate normalization constant
            Zn = Zi(X[n], cav_mean, cav_var, w=clutter_ratio, a=clutter_var)

            # compute rho_n, which is the probability of the sampled
            # point, X[n], not being clutter.
            rho_n = 1 - ((w / Zn) * gaussian(X[n], clutter_mean, clutter_var))

            # find the mean and variance of the new posterior.
            est_mean = cav_mean + (rho_n * cav_var * (X[n] - cav_mean) / (cav_var + 1))
            est_var = (
                cav_var
                - (rho_n * (cav_var**2) / (cav_var + 1))
                + (rho_n * (1 - rho_n) * (cav_var**2) * (abs(X[n] - cav_mean)) ** 2)
                / (D * ((cav_var + 1) ** 2))
            )

            # calculate the parameters of the refined approximate factor
            # Careful with divide by zero errors.
            # If there are undefined results no need to update factor, can just
            # skip this factor and return on next iteration.
            if est_var != cav_var:
                fn_var_new = 1 / ((1 / est_var) - (1 / cav_var))
                fn_mean_new = cav_mean + (
                    (fn_var_new + cav_var) * (1 / cav_var) * (est_mean - cav_mean)
                )
                fn_ss_new = Zn / (
                    ((2 * np.pi * abs(fn_var_new)) ** (D / 2))
                    * gaussian(fn_mean_new, cav_mean, (fn_var_new + cav_var))
                )

                # check for convergence
                if (
                    abs(f_means[iteration, n] - fn_mean_new) > tolerance
                    and abs(f_vars[iteration, n] - fn_var_new) > tolerance
                    and abs(f_ss[iteration, n] - fn_ss_new) > tolerance
                ):
                    converged = False
                else:
                    converged = True

                # saved refined parameters
                f_means[iteration, n] = fn_mean_new
                f_vars[iteration, n] = fn_var_new
                f_ss[iteration, n] = fn_ss_new

                cavity_vars[iteration, n] = cav_var
                cavity_means[iteration, n] = cav_mean

                est_means[iteration, n] = est_mean
                est_vars[iteration, n] = est_var

            # plot the factor
            if f_vars[iteration, n] != np.inf and interactive:
                plot_factor(
                    X,
                    n,
                    fn_mean_new,
                    fn_var_new,
                    fn_ss_new,
                    cav_mean,
                    cav_var,
                    c_ratio=clutter_ratio,
                    c_var=clutter_var,
                )

    return est_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to show a plot of the ongoing approximation",
    )
    args = parser.parse_args()

    approx_config = {
        "prior_mean": 0,
        "prior_var": 100,
        "n_dimensions": 1,
        "max_iter": 20,
        "tolerance": 10**-4,
        "interactive": True,
    }

    data_config = {
        "n_observations": 41,
        "clutter_var": 10,
        "clutter_mean": 0,
        "clutter_ratio": 0.5,
        "target_mean": 3,
        "target_var": 1,
    }

    data = make_1d_clutter_data(**data_config)

    plot_data(
        data_config["target_mean"],
        data,
        clutter_mean=data_config["clutter_mean"],
        clutter_var=data_config["clutter_var"],
    )

    est_mean = run_ep(
        observations=data,
        **approx_config,
        clutter_mean=data_config["clutter_mean"],
        clutter_var=data_config["clutter_var"],
        clutter_ratio=data_config["clutter_ratio"],
    )

    # Print summary of results
    print("EP results summary:")
    print(
        tabulate(
            [
                ["Iterations", len(est_mean)],
                ["Theta", data_config["target_mean"]],
                ["Approx Theta", est_mean],
            ]
        )
    )
