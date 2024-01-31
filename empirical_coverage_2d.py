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

from empirical_coverage import EmpiricalCoverageRunner, Area

if __name__ == "__main__":
    generate_results = True
    sample_count = 100
    dimension = 2
    order = 20
    set_count = 5000
    data_loc = "/home/william/phd/programming_projects/gcp_rssb/datasets/comparison_experiments/data_and_samples/samples_update/samples2D/"
    domains = [
        torch.Tensor([[0, 1], [0, 1]]),  # redwood
        torch.Tensor([[0, 1], [0, 1]]),  # white oak
    ]
    synth_data_sets = []
    redwood_df = pd.read_csv(
        "/home/william/phd/programming_projects/gcp_rssb/datasets/spatial-2D/redwood_full.csv"
    )
    white_oak_df = pd.read_csv(
        "/home/william/phd/programming_projects/gcp_rssb/datasets/spatial-2D/white_oak.csv"
    )
    # data_set = torch.tensor(df.values).squeeze()
    data_sets = [
        torch.Tensor(redwood_df.values).squeeze(),
        torch.Tensor(white_oak_df.values).squeeze(),
    ]

    # set up the basis
    mean_results = torch.zeros(sample_count, 4, 2)
    std_results = torch.zeros(sample_count, 4, 2)
    if generate_results:
        for j, intensity, data_set, domain in zip(
            range(2), ["redwood", "whiteoak"], data_sets, domains
        ):
            # now that I have added the data, generate a sample
            params = [
                {
                    "lower_bound": domain[0][0] - 0.2,
                    "upper_bound": domain[0][1] + 0.2,
                }
            ] * dimension
            basis_functions = [standard_chebyshev_basis] * 2
            ortho_basis = Basis(basis_functions, dimension, order, params)

            # set up the model
            gcp_ose_hyperparams = GCPOSEHyperparameters(
                basis=ortho_basis, dimension=dimension
            )
            prior_parameters = DataInformedPriorParameters(nu=0.02)
            osegcp = BayesianOrthogonalSeriesCoxProcess(
                gcp_ose_hyperparams, prior_parameters
            )

            # add the data
            print("About to add the data:", intensity)
            osegcp.add_data(data_set, domain)
            print("Just added the data - should have printed the times...")
            # breakpoint()
            empirical_coverage_runner = EmpiricalCoverageRunner(
                domain, dimension
            )
            random_areas: List[
                Area
            ] = empirical_coverage_runner.get_random_sets(set_count)
            for i in range(sample_count):
                print("iteration:", i)
                print("about to get osegcp sample")
                osegcp_sample = osegcp.get_posterior_predictive_sample()
                print("osegcp_sample acquired")
                vbpp_sample = torch.load(
                    data_loc + "vbpp_{}_{}.pt".format(intensity, i + 1)
                )
                lbpp_sample = torch.load(
                    data_loc + "lbpp_{}_{}.pt".format(intensity, i + 1)
                )
                rkhs_sample = torch.load(
                    data_loc + "rkhs_{}_{}.pt".format(intensity, i + 1)
                )

                predictive_samples = [
                    osegcp_sample,
                    # vbpp_sample,
                    # lbpp_sample,
                    # rkhs_sample,
                ]

                # First we calculate the "data" side of the residuals. for each of
                # the random sets, count the points generated in our posterior
                # predictive sample, and compare it to the number of points in the
                # data
                print("about to check the data counts")
                data_counts = torch.zeros(set_count)
                method_counts_tensor = torch.zeros(
                    set_count, len(predictive_samples)
                )
                for area_index, area in enumerate(random_areas):
                    data_observed_points = area.count_points_in_area(data_set)
                    data_counts[area_index] = data_observed_points

                    # method_counts_list = []
                    for (
                        sample_index,
                        predictive_sample,
                    ) in enumerate(predictive_samples):
                        # For each method,
                        sample_observed_points = area.count_points_in_area(
                            predictive_sample
                        )
                        method_counts_tensor[
                            area_index, sample_index
                        ] = sample_observed_points

                        # Then save them.
                        # method_counts_list.append(counts_list)
                data_counts_extended = data_counts.repeat(
                    len(predictive_samples), 1
                ).T
                predictive_residuals = (
                    data_counts_extended - method_counts_tensor
                )
                method_comparison_values = []
                mean_results[i, :, j] = torch.mean(predictive_residuals, dim=0)
                std_results[i, :, j] = torch.std(predictive_residuals, dim=0)
        torch.save(mean_results, "mean_results_2d.pt")
        torch.save(std_results, "std_results_2d.pt")
        print("About to print the method comparison values!")
        print(
            method_comparison_values
        )  # now prints, for each method, a mean and std
    else:
        mean_results = torch.load("mean_results_2d.pt")
        std_results = torch.load("std_results_2d.pt")
    osgcp_mean = torch.mean(mean_results[:, 0, 0] ** 2)
    vbpp_mean = torch.mean(mean_results[:, 1, 0] ** 2)
    lbpp_mean = torch.mean(mean_results[:, 2, 0] ** 2)
    rkhs_mean = torch.mean(mean_results[:, 3, 0] ** 2)
    print("Redwood : sum of squared residuals")
    print("osegcp mean: {}".format(osgcp_mean))
    print("vbpp mean: {}".format(vbpp_mean))
    print("lbpp mean: {}".format(lbpp_mean))
    print("rkhs mean: {}".format(rkhs_mean))
    osgcp_mean = torch.mean(mean_results[:, 0, 1] ** 2)
    vbpp_mean = torch.mean(mean_results[:, 1, 1] ** 2)
    lbpp_mean = torch.mean(mean_results[:, 2, 1] ** 2)
    rkhs_mean = torch.mean(mean_results[:, 3, 1] ** 2)
    print("White Oak : sum of squared residuals")
    print("osegcp mean: {}".format(osgcp_mean))
    print("vbpp mean: {}".format(vbpp_mean))
    print("lbpp mean: {}".format(lbpp_mean))
    print("rkhs mean: {}".format(rkhs_mean))
