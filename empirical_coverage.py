import torch
import torch.distributions as D
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from ortho.basis_functions import Basis, standard_chebyshev_basis
from gcp_rssb.methods.gcp_ose import GCPOSEHyperparameters
from gcp_rssb.methods.gcp_ose_bayesian import (
    BayesianOrthogonalSeriesCoxProcess,
    BayesianOrthogonalSeriesCoxProcessObservationNoise,
    PriorParameters,
    DataInformedPriorParameters,
)
import matplotlib.pyplot as plt
import math
from termcolor import colored

"""
To do: re do the sample iterations!!!
"""

"""
In this file we will construct the method for empirical coverage as outlined in
"Bayesian Inference and Model Assessment for Spatial Point Patterns Using
Posterior Predictive Samples" by THomas J Leininger and Alan E Gelfand.
"""


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


class Method(ABCMeta):
    def _get_posterior_predictive_sample(self) -> torch.Tensor:
        """
        Returns a tensor of size (n_points, n_dim) where n_points is a Poisson distributed random variable.

        The three steps are:
            -
        """
        pass

    def _get_domain(self):
        pass


class EmpiricalCoverageRunner:
    def __init__(
        self,
        domain: torch.Tensor,
        dimension: int,
    ):
        """
        The idea behind the Empirical coverage is to generate posterior
        predictive samples for random intervals. Then we compare the difference
        between the number of points in the intervals with the true number of
        points in that interval in the data.
        """
        self.domain = domain  # the domain of the data
        self.dimension = dimension

        assert (
            self.dimension == 1 or self.dimension == 2
        ), "Dimension must be 1 or 2"

    def get_random_sets(self, set_count: int):
        """
        Returns a list of Area objects, which are random subsets of the domain.
        """
        coordinate_dists = []
        for i in range(self.dimension):
            coordinate_dist = D.Uniform(
                low=self.domain[i][0], high=self.domain[i][1]
            )
            coordinate_dists.append(coordinate_dist)

        # get the coordinates
        full_coord_pairs = torch.Tensor(set_count, 2, self.dimension)
        for d in range(self.dimension):
            coord_pairs = coordinate_dists[d].sample((set_count, 2))
            full_coord_pairs[:, :, d] = coord_pairs

        random_sets = []
        for i in range(set_count):
            random_sets.append(
                Area(full_coord_pairs[i, :, :], self.domain, self.dimension)
            )
        return random_sets


class Area:
    def __init__(self, coords: torch.Tensor, domain, dimension):
        self.coords = coords.sort(dim=0).values
        self.domain = domain
        self.dimension = dimension

    def get_area(self):
        pass

    def count_points_in_area(self, sample: torch.Tensor):
        """
        Counts the number of points in the sample that are in the area.

        input sample: torch.Tensor of size (n_points, n_dim)
        """
        counter = 0
        for point in sample:
            if self._is_point_in_area(point):
                counter += 1

        return counter

    def _is_point_in_area(self, point):
        """
        Returns true if the point is in the area, and false otherwise.
        """
        if self.dimension == 1:
            try:
                if point < self.coords[0][0] or point > self.coords[1][0]:
                    return False
            except IndexError:
                print("yuo messied it up!!")
                breakpoint()
        else:
            for d in range(self.dimension):
                if (
                    point[d] < self.coords[0][d]
                    or point[d] > self.coords[1][d]
                ):
                    return False
        return True


if __name__ == "__main__":
    generate_results = True
    dimension = 1
    domains = [
        torch.Tensor([[0, 50]]),
        torch.Tensor([[0, 5]]),
        torch.Tensor([[0, 100]]),
    ]
    synth_data_sets = []
    data_loc = "/home/william/phd/programming_projects/gcp_rssb/datasets/comparison_experiments/"
    for c in range(3):
        df = pd.read_csv(data_loc + "synth{}.csv".format(c + 1))
        data_set = torch.tensor(df.values).squeeze()
        synth_data_sets.append(data_set)
    # print(torch.max(data_set))
    # set up the basis
    orders = [8, 16, 8]
    sample_count = 100
    mean_results = torch.zeros(sample_count, 4, 3)
    std_results = torch.zeros(sample_count, 4, 3)
    mean_data: dict = {}
    std_data: dict = {}
    set_count = 5000
    if generate_results:
        for j, intensity, data_set, domain, order, intensity_function in zip(
            range(3),
            ["lambda_1", "lambda_2", "lambda_3"],
            synth_data_sets,
            domains,
            orders,
            [intensity_1, intensity_2, intensity_3],
        ):
            # now that I have added the data, generate a sample
            params = [
                {
                    "lower_bound": domain[0][0] - 0.15,
                    "upper_bound": domain[0][1] + 0.15,
                    "chebyshev": "second",
                }
            ]
            ortho_basis = Basis(standard_chebyshev_basis, 1, order, params)

            # set up the model
            gcp_ose_hyperparams = GCPOSEHyperparameters(
                basis=ortho_basis, dimension=dimension
            )
            prior_parameters = DataInformedPriorParameters(nu=0.02)
            osegcp = BayesianOrthogonalSeriesCoxProcess(
                gcp_ose_hyperparams, prior_parameters
            )
            # add the data
            osegcp.add_data(data_set, domain)
            empirical_coverage_runner = EmpiricalCoverageRunner(
                domain, dimension
            )
            method_names = ["osegcp", "vbpp", "lbpp", "rkhs"]
            method_names = ["rkhs"]
            sample_names = ["synth1", "synth2", "synth3"]
            mean_std_data = {"synth1": [], "synth2": [], "synth3": []}
            random_areas: List[
                Area
            ] = empirical_coverage_runner.get_random_sets(set_count)
            for i in range(sample_count):
                osegcp_sample = osegcp.get_posterior_predictive_sample()
                vbpp_sample = torch.load(
                    data_loc + "vbpp_synth{}_{}.pt".format(j + 1, i + 1)
                )
                lbpp_sample = torch.load(
                    data_loc + "lbpp_synth{}_{}.pt".format(j + 1, i + 1)
                )
                rkhs_sample = torch.load(
                    data_loc + "rkhs_synth{}_{}.pt".format(j + 1, i + 1)
                )

                predictive_samples = [
                    # osegcp_sample,
                    # vbpp_sample,
                    # lbpp_sample,
                    rkhs_sample,
                ]

                # First we calculate the "data" side of the residuals. for each of
                # the random sets, count the points generated in our posterior
                # predictive sample, and compare it to the number of points in the
                # data
                data_counts = torch.zeros(set_count)
                method_counts_tensor = torch.zeros(
                    set_count, len(predictive_samples)
                )
                for area_index, area in enumerate(random_areas):
                    data_observed_points = area.count_points_in_area(data_set)
                    data_counts[area_index] = data_observed_points
                    counts_list = []

                    for sample_index, predictive_sample in enumerate(
                        predictive_samples
                    ):  # For each method,
                        sample_observed_points = area.count_points_in_area(
                            predictive_sample
                        )
                        method_counts_tensor[
                            area_index, sample_index
                        ] = sample_observed_points

                data_counts_extended = data_counts.repeat(
                    len(predictive_samples), 1
                ).T
                predictive_residuals = (
                    data_counts_extended - method_counts_tensor
                )

                # method_comparison_values = []
                # for method_counts in method_counts_list:
                # now we have a list of counts for each method
                # we can calculate the empirical coverage
                # empirical_coverage = 0
                # predictive_residuals = []
                # for data_count, method_count in zip(
                # data_counts, method_counts
                # ):
                # predictive_residual = method_count - data_count
                # predictive_residuals.append(predictive_residual)
                mean_results[i, :, j] = torch.mean(predictive_residuals, dim=0)
                std_results[i, :, j] = torch.std(predictive_residuals, dim=0)
        torch.save(mean_results, "mean_results.pt")
        torch.save(std_results, "std_results.pt")
        breakpoint()
    else:
        mean_results = torch.load("mean_results.pt")
        std_results = torch.load("std_results.pt")

    plt.hist(mean_results[:, 1, 0].numpy(), bins=40, color="red")
    plt.hist(mean_results[:, 2, 0].numpy(), bins=40, color="green")
    plt.hist(mean_results[:, 0, 0].numpy(), bins=40, color="blue")
    osgcp_mean = torch.mean(mean_results[:, 0, 0] ** 2)
    vbpp_mean = torch.mean(mean_results[:, 1, 0] ** 2)
    lbpp_mean = torch.mean(mean_results[:, 2, 0] ** 2)
    rkhs_mean = torch.mean(mean_results[:, 3, 0] ** 2)
    print("Synth 1 Empirical Coverage: sum of squared residuals")
    print("osegcp mean: {}".format(osgcp_mean))
    print("vbpp mean: {}".format(vbpp_mean))
    print("lbpp mean: {}".format(lbpp_mean))
    print("rkhs mean: {}".format(rkhs_mean))
    osgcp_mean = torch.mean(mean_results[:, 0, 1] ** 2)
    vbpp_mean = torch.mean(mean_results[:, 1, 1] ** 2)
    lbpp_mean = torch.mean(mean_results[:, 2, 1] ** 2)
    rkhs_mean = torch.mean(mean_results[:, 3, 1] ** 2)
    print("Synth 2 Empirical Coverage: sum of squared residuals")
    print("osegcp mean: {}".format(osgcp_mean))
    print("vbpp mean: {}".format(vbpp_mean))
    print("lbpp mean: {}".format(lbpp_mean))
    print("rkhs mean: {}".format(rkhs_mean))
    osgcp_mean = torch.mean(mean_results[:, 0, 2] ** 2)
    vbpp_mean = torch.mean(mean_results[:, 1, 2] ** 2)
    lbpp_mean = torch.mean(mean_results[:, 2, 2] ** 2)
    rkhs_mean = torch.mean(mean_results[:, 3, 2] ** 2)
    print("Synth 3 Empirical Coverage: sum of squared residuals")
    print("osegcp mean: {}".format(osgcp_mean))
    print("vbpp mean: {}".format(vbpp_mean))
    print("lbpp mean: {}".format(lbpp_mean))
    print("rkhs mean: {}".format(rkhs_mean))

    breakpoint()
    # plt.show()
    # plt.hist(std_results[:, 0, 0], bins=20)
    # plt.show()
    # plt.show()
    # plt.hist(std_results[:, 1, 0], bins=20)
    # plt.show()
    plt.show()
    # plt.hist(std_results[:, 2, 0], bins=20)
    # plt.show()

    # print("Mean data:")
    # print(colored(mean_data, "red"))
    # print("    ")
    # print("Std data:")
    # print(colored(std_data, "blue"))
    # for j in range(3):
    # print("synth {}".format(j + 1))
    # print("osegcp:")
    # print(torch.mean(mean_results[:, 0, j]))
    # # print(torch.std(mean_results[:, 0]))
    # print(torch.mean(std_results[:, 0, j]))
    # # print(torch.std(std_results[:, 0]))
    # print("vbpp")
    # print(torch.mean(mean_results[:, 1, j]))
    # # print(torch.std(mean_results[:, 1]))
    # print(torch.mean(std_results[:, 1, j]))
    # # print(torch.std(std_results[:, 1]))
    # print("lbpp")
    # print(torch.mean(mean_results[:, 2, j]))
    # # print(torch.std(mean_results[:, 2]))
    # print(torch.mean(std_results[:, 2, j]))
    # # print(torch.std(std_results[:, 2]))
