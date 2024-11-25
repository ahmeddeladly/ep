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

from math import sqrt, pi
import numpy as np
from scipy.stats import bernoulli, norm
from tabulate import tabulate
import json
import os

import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt


# Object holding the configuration params
config = {
    "n_observations": 41,
    "n_dimensions": 1,
    "clutter_ratio": 0.5,
    "clutter_mean": 0,
    "clutter_var": 10,
    "prior_mean": 0,
    "prior_var": 100,
    "max_n_iterations": 20,
    "tolerance": 10**-4,
    "interactive": True,
}


# Run the EP example procedure
def main():
    # Configuration variables
    N = config["n_observations"]
    D = config["n_dimensions"]
    max_i = config["max_n_iterations"]
    w = config["clutter_ratio"]
    clut_var = config["clutter_var"]
    clut_mean = config["clutter_mean"]
    tol = config["tolerance"]

    # Generate data for the problem. Theta is the true mean and X is an array
    # containing the sampled values.
    theta, X = gen_data()

    # Save the generated data
    save_data(theta, X)

    # Visually inspect data via a plot
    if config["interactive"]:
        plot_data(theta, X)

    # Initialise prior, this is the first factor, f0, and the EP update
    # algorithm does not change it's value.
    p_mean = config["prior_mean"]
    p_var = config["prior_var"]
    p_s = (2 * pi * p_var) ** (-D / 2)

    # Initialise the factors 1...N to unity.
    # Will store all calculated values in order to visualise the factor
    # development. We just need to use a distribution from the exponential
    # family for this problem, a Gaussian is appropriate. This corresponds to:
    f_means = np.zeros((max_i, N))
    f_vars = np.full((max_i, N), np.inf)
    f_ss = np.ones((max_i, N))

    # Initialise our estimate for the approximation to the true underlying
    # distribution, q(theta), to be simply be the prior.
    est_means = np.full((max_i, N), p_mean)
    est_vars = np.full((max_i, N), p_var)
    est_mean = p_mean
    est_var = p_var

    # Initialise storage for the cavity mean and variance.
    # Not necessary but will be useful for visualisation of algorithm
    # progression.
    cavity_vars = np.full((max_i, N), p_mean)
    cavity_means = np.full((max_i, N), p_var)

    # Initialise convergence flag
    converged = False

    # Iterate EP algorithm
    for iteration in range(max_i):
        if converged:
            break

        # For each data point in our sampled observations
        for n in range(N):
            # Moments for the current approximate factor
            fn_mean = f_means[iteration, n]
            fn_var = f_vars[iteration, n]

            # Create the cavity distribution by removing the estimate for the
            # current factor from the current posterior.
            cav_var = 1 / ((1 / est_var) - (1 / fn_var))
            cav_mean = est_mean + (cav_var * (1 / fn_var) * (est_mean - fn_mean))

            # Compute the new estimate for the posterior by multiply the
            # cavity distribution by the approximate factor distribution.
            # When you are multiplying to distributions of the exponential
            # family you can follow the known formula to find the defining
            # moments of the new distribution.

            # Evaluate normalisation constant
            Zn = Zi(X[n], cav_mean, cav_var)

            # Compute rho_n, which is the probability of the sampled
            # point, X[n], not being clutter.
            rho_n = 1 - ((w / Zn) * gaus(X[n], clut_mean, clut_var))

            # Find the mean and variance of the new posterior.
            est_mean = cav_mean + (rho_n * cav_var * (X[n] - cav_mean) / (cav_var + 1))
            est_var = (
                cav_var
                - (rho_n * (cav_var**2) / (cav_var + 1))
                + (rho_n * (1 - rho_n) * (cav_var**2) * (abs(X[n] - cav_mean)) ** 2)
                / (D * ((cav_var + 1) ** 2))
            )

            # Calculate the parameters of the refined approximate factor
            # Careful with divide by zero errors.
            # If there are undefined results no need to update factor, can just
            # skip this factor and return on next iteration.
            if est_var != cav_var:
                fn_var_new = 1 / ((1 / est_var) - (1 / cav_var))
                fn_mean_new = cav_mean + (
                    (fn_var_new + cav_var) * (1 / cav_var) * (est_mean - cav_mean)
                )
                fn_ss_new = Zn / (
                    ((2 * pi * abs(fn_var_new)) ** (D / 2))
                    * gaus(fn_mean_new, cav_mean, (fn_var_new + cav_var))
                )

                # Check for convergence
                if (
                    abs(f_means[iteration, n] - fn_mean_new) > tol
                    and abs(f_vars[iteration, n] - fn_var_new) > tol
                    and abs(f_ss[iteration, n] - fn_ss_new) > tol
                ):
                    converged = False
                else:
                    converged = True

                # Saved refined parameters
                f_means[iteration, n] = fn_mean_new
                f_vars[iteration, n] = fn_var_new
                f_ss[iteration, n] = fn_ss_new

                cavity_vars[iteration, n] = cav_var
                cavity_means[iteration, n] = cav_mean

                est_means[iteration, n] = est_mean
                est_vars[iteration, n] = est_var

                print(est_mean)

            # Plot the factors
            if f_vars[iteration, n] != np.inf and config["interactive"]:
                plot_factors(
                    X, n, fn_mean_new, fn_var_new, fn_ss_new, cav_mean, cav_var
                )

    # Print summary of results
    print("EP results summary:")
    print(
        tabulate(
            [
                ["Iterations", iteration],
                ["Theta", theta],
                ["Approx Theta", est_mean],
            ]
        )
    )


# Save EP iteration data
def save_EP_data(p_mean, p_var):
    asdfa = 1


# Generate the data for the clutter problem.
# At the moment, looking at the simple 1 dimensional case.
def gen_data():
    # Default configuration Variables
    N = config["n_observations"]
    c_var = config["clutter_var"]
    c_mean = config["clutter_mean"]
    w = config["clutter_ratio"]
    # p_var = config["prior_var"]
    # p_mean = config["prior_mean"]

    # Define distributions
    clutter = norm(c_mean, sqrt(c_var))
    # prior = norm(p_mean, p_var)

    # Could generate the mean of the target distribution by sampling a point
    # from the prior, but for testing purposes prefer to hard code.
    theta = 3
    # theta = prior.rvs()
    target_dist = norm(theta, 1)

    # The rate at which the clutter distribution is sampled from is determined
    # by the clutter ratio.
    sample_clutter = bernoulli(w).rvs

    # Sample from the distributions
    samples = [
        (clutter.rvs() if sample_clutter() else target_dist.rvs()) for _ in range(N)
    ]

    return theta, samples


# Plot the generated data
def plot_data(theta, X):
    # Configuration Variables
    c_var = config["clutter_var"]
    c_mean = config["clutter_mean"]

    # Create figure
    fig, ax = plt.subplots(1, 1)

    # X axes scaling
    data_range = max(X) - min(X)
    left = min(X) - 0.2 * data_range
    right = max(X) + 0.2 * data_range
    x = np.linspace(left, right, 200)

    # Create distributions
    clutter = norm(c_mean, sqrt(c_var)).pdf(x)
    target = norm(theta, 1).pdf(x)
    sample_points = np.zeros(len(X))

    # Plot
    ax.plot(x, clutter, "r-", label="Clutter Distribution")
    ax.plot(x, target, "g-", label="Target Distribution")
    ax.plot(X, sample_points, "bx", label="Samples")
    ax.legend(loc="best", frameon=False)
    plt.show()


# Function for plotting and visually inspecting the gaussian factors
def plot_factors(X, n, fn_mean, fn_var, fn_s, cav_mean, cav_var):
    # Inputs
    # X:        sampled observations
    # n:        current factor
    # fn_mean:  Approximated factor mean
    # fn_var:   Approximated factor variance
    # fn_var:   Approximated factor scaler
    # cav_mean: Cavity mean
    # cav_var:  Cavity variance

    # Create figure
    fig, ax = plt.subplots(1, 1)

    # X axes scaling
    x = np.linspace(X[n] - 5, X[n] + 5, 200)

    # Create distributions over space that will result in a good plot
    f_true = list(map(true_factor(X[n]), x))
    f_approx = list(map(approx_factor(fn_s, fn_mean, fn_var), x))
    cavity = norm(cav_mean, sqrt(cav_var)).pdf(x)

    # Scale y axis
    plt.ylim(ymax=(1.2 * max(f_true)))

    # Plot
    ax.plot(x, f_true, "b-", label="True Factor", linewidth=4, alpha=0.5)
    ax.plot(x, f_approx, "r-", label="Approximate Factor")
    ax.plot(x, cavity, "g-", label="Cavity Distribution")
    ax.legend(loc="best", frameon=False)
    plt.show()


# Function outputs given values of the true factor distribution
def true_factor(Xn):
    # Configuration variables
    w = config["clutter_ratio"]
    a = config["clutter_var"]

    # Using explicit calculation rather than scipy.norm
    return lambda x: ((1 - w) * gaus(Xn, x, 1)) + (w * gaus(Xn, 0, a))


# Approximate factor distribution
def approx_factor(Sn, mn, vn):
    # Inputs:
    # Sn:   Scale factor
    # mn:   mean
    # vn:   variance

    return lambda x: Sn * np.exp(-(1 / (2 * vn)) * ((x - mn) ** 2))


# Calculate normalising factor
def Zi(Xn, cav_mean, cav_var):
    # Inputs
    # Xn:       Sample
    # cav_mean: Cavity mean
    # cav_var:  Cavity Variance

    # Configuration variables
    w = config["clutter_ratio"]
    a = config["clutter_var"]

    return ((1 - w) * gaus(Xn, cav_mean, (cav_var + 1))) + (w * gaus(Xn, 0, a))


# Evaluate gaussian at specific point given mean and variance
def gaus(x, m, v):
    # Inputs:
    # x:    point
    # m:    mean
    # v:    variance

    return np.exp(-0.5 * ((x - m) ** 2) * (1 / v)) / ((abs(2 * pi * v)) ** 0.5)


# Save the generated data to a file
def save_data(theta, X):
    # Current working directory
    cwd = os.getcwd()

    # Configuration Variables
    c_var = config["clutter_var"]
    c_mean = config["clutter_mean"]

    data = {"target_dist": (theta, 1), "clutter_dist": (c_mean, c_var), "samples": X}

    # Write to file
    with open((cwd + "data.json"), "w") as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)


main()
