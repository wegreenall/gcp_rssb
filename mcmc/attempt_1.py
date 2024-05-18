import warnings
from itertools import product

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from matplotlib import MatplotlibDeprecationWarning
from numpy.random import default_rng

warnings.filterwarnings(action="ignore", category=MatplotlibDeprecationWarning)


if __name__ == "__main__":
    run_new_samples = True

    # redwood

    data = pd.read_csv("../datasets/spatial-2D/redwood_full.csv")
    n = data.shape[0]
    print("How Many Points!??", n)

    plt.scatter(data["x"], data["y"])
    plt.show()
    xy = data[["x", "y"]].values

    resolution = 0.1
    area_per_cell = resolution**2
    grid_size = 1
    cells_x = int(grid_size / resolution)
    cells_y = int(grid_size / resolution)
    quadrat_x = np.linspace(0, grid_size, cells_x + 1)
    quadrat_y = np.linspace(0, grid_size, cells_y + 1)
    centroids = np.array(
        list(
            product(
                quadrat_x[:-1] + resolution / 2,
                quadrat_y[:-1] + resolution / 2,
            )
        )
    )

    cell_counts, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=[quadrat_x, quadrat_y]
    )
    cell_counts = cell_counts.ravel().astype(int)

    line_kwargs = {"color": "k", "linewidth": 1, "alpha": 0.5}

    plt.figure(figsize=(6, 4.5))
    [plt.axhline(y, **line_kwargs) for y in quadrat_y]
    [plt.axvline(x, **line_kwargs) for x in quadrat_x]
    plt.scatter(data["x"], data["y"], s=6)

    for i, row in enumerate(centroids):
        shifted_row = row - 2
        plt.annotate(cell_counts[i], shifted_row, alpha=0.75)

    print("Cell counts:", cell_counts)
    plt.title("TREE counts per grid cell")
    plt.show()

    # "Inference"
    with pm.Model() as lgcp_model:
        mu = pm.Normal("mu", sigma=3)
        rho = pm.Uniform("rho", lower=0.1, upper=2)
        variance = pm.InverseGamma("variance", alpha=1, beta=1)

        # second layer of the hierarchical
        cov_func = variance * pm.gp.cov.Matern52(2, ls=rho)
        mean_func = pm.gp.mean.Constant(mu)

    with lgcp_model:
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        f = gp.prior("f", X=centroids)

        # Likelihood
        y = pm.Poisson("y", mu=pm.math.exp(f), observed=cell_counts)
        log_intensity = gp.prior("log_intensity", X=centroids)
        intensity = pm.math.exp(log_intensity)

        rates = intensity * area_per_cell
        counts = pm.Poisson("counts", mu=rates, observed=cell_counts)
        print("We got down to here...")

    y_new = np.linspace(0, 1, 40)
    x_new = np.linspace(0, 1, 40)
    xs, ys = np.asarray(np.meshgrid(x_new, y_new))
    xy_new = np.asarray([xs.ravel(), ys.ravel()]).T
    if run_new_samples:
        with lgcp_model:
            trace = pm.sample(1000, tune=2000, target_accept=0.95)

        az.summary(trace, var_names=["mu", "rho", "variance"])

        with lgcp_model:
            intensity_new = gp.conditional("log_intensity_new", Xnew=xy_new)
            app_trace = pm.sample_posterior_predictive(
                trace, var_names=["log_intensity_new"]
            )

        trace.extend(app_trace)
        intensity_samples = np.exp(
            trace.posterior_predictive["log_intensity_new"]
        )
        np.save("intensity_samples_mcmc.npy", intensity_samples)
    else:
        intensity_samples = np.load("intensity_samples_mcmc.npy")

    fig = plt.figure(figsize=(5, 4))

    mean = intensity_samples.mean(axis=0).mean(axis=0)
    plt.scatter(
        xy_new[:, 0],
        xy_new[:, 1],
        c=mean,
        marker="o",
        alpha=0.75,
        s=100,
        edgecolor=None,
    )

    plt.title("$E[\\lambda(s) \\vert Y]$")
    plt.colorbar(label="Posterior mean")
    plt.savefig("lgcp_posterior_mean.png")
