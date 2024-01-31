"""
In this script, we will present evidence for the Gaussianity of the basis function
observation noise.
"""
import torch
from ortho.basis_functions import Basis
from mercergp.MGP import HilbertSpaceElement
from mercergp.eigenvalue_gen import (
    EigenvalueGenerator,
    SmoothExponentialFasshauer,
)
from dataclasses import dataclass
from typing import Callable

from gcp_rssb.data import PoissonProcess
from ortho.basis_functions import (
    Basis,
    standard_chebyshev_basis,
    smooth_exponential_basis,
    smooth_exponential_basis_fasshauer,
)

import matplotlib.pyplot as plt
import tikzplotlib

from gcp_rssb.methods.gcp_ose import (
    OrthogonalSeriesCoxProcess,
    GCPOSEHyperparameters,
)

from scipy.stats import kstest


if __name__ == "__main__":
    max_time = 50.0

    def intensity_1(t: float) -> float:
        return 2 * torch.exp(-t / 15) + torch.exp(-(((t - 25) / 10) ** 2))

    def intensity_2(t: float) -> float:
        return 5 * torch.sin(t**2) + 6

    def intensity_3(X: torch.Tensor) -> float:
        idx_less_25 = [i for i in range(len(X)) if X[i] < 25]
        idx_less_50 = [i for i in range(len(X)) if 25 <= X[i] < 50]
        idx_less_75 = [i for i in range(len(X)) if 50 <= X[i] < 75]
        other_idx = [i for i in range(len(X)) if X[i] >= 75]
        return torch.cat(
            [
                0.04 * X[idx_less_25] + 2,
                -0.08 * X[idx_less_50] + 5,
                0.06 * X[idx_less_75] - 2,
                0.02 * X[other_idx] + 1,
            ]
        )

    basis = standard_chebyshev_basis
    parameters: dict = {
        "lower_bound": 0.0,
        "upper_bound": max_time + 0.1,
        "chebyshev": "second",
    }
    order = 20
    basis = Basis(standard_chebyshev_basis, 1, order, parameters)

    experiment_count = 5000
    for j, intensity in enumerate([intensity_1, intensity_2, intensity_3]):
        basis_evals_buffer = torch.zeros(experiment_count, order)
        x_data = torch.linspace(0, max_time, 1000)
        pp = PoissonProcess(intensity, max_time, 1.0)
        for i in range(experiment_count):
            pp.simulate()
            data = pp.get_data()

            basis_evals = basis(data)
            basis_evals_buffer[i, :] = basis_evals.sum(dim=0)

        table_string = ""
        for i in range(order):
            basis_evals = (
                basis_evals_buffer[:, i] - torch.mean(basis_evals_buffer[:, i])
            ) / torch.std(basis_evals_buffer[:, i])

            # perform a ks-test
            ks_stat, p_value = kstest(basis_evals, "norm")
            string = "{} & {:.4f} & {:.4f}\\\\ \n".format(i, ks_stat, p_value)
            table_string += string
            print(
                "KS-Test for basis function {}: ".format(i), ks_stat, p_value
            )
            # if p_value < 0.1:
            # print("KS-Test failed for basis function {}".format(i))
            # print("because of p-value {}".format(p_value))
            # plt.hist(basis_evals.numpy(), bins=100)
            # plt.show()

        # save the string to a file
        with open("ks_test_results_{}.txt".format(j), "w") as f:
            f.write(table_string)
