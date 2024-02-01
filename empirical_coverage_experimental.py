import torch
import torch.distributions as D
import pandas as pd
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
import time


# def print_results_table(residuals, intensities):
# final_means = torch.mean(residuals, dim=0)
# final_squared_means = torch.mean(final_means**2, dim=0)
# results_table = "\\toprule"
# results_table += "\n"
# for data_set in data_set_names:
# results_table += "& \multicolumn\{2\}\{c\}\{{}\}".format(data_set)
# results_table +="\\\\"
# results_table += "\n"
# results_table += "& \\hskip -0.14in  EC &\\hskip -0.14in time(s)& \\hskip -0.14in EC &\\hskip -0.14in time(s)\\\\"
# results_table += "\\midrule\n"
# for i in range(len(final_squared_means)):

# results_table += "& Mean & Std. Dev. \\\\"
# for i in range(len(final_squared_means)):


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


class EmpiricalCoverageRunner:
    def __init__(
        self,
        data_set: torch.Tensor,
        domains: torch.Tensor,
        dimension: torch.Tensor,
        set_count=5000,
        gpu=True,
    ):
        assert domains.shape
        self.domains = domains
        self.dimension = dimension
        self.uniform = D.Uniform(
            low=self.domains[:, 0], high=self.domains[:, 1]
        )

        # check that the data set has enough dimensions
        if len(data_set.shape) == 1:
            data_set = data_set.unsqueeze(1)
        self.data_set = data_set
        self.set_count = set_count

        # flag to check if we should use the GPU
        self.gpu = gpu

    def _get_random_sets(self):
        sets = self.uniform.sample((self.set_count, 2))
        return torch.sort(sets, dim=1).values

    def check_sample(self, sample):
        """
        Accepts a sample from an inhomogeneous Poisson process, and
        returns a tensor of shape (set_count, 1) where each element is
        the number of points in the sample that fall within the
        corresponding random set.

        Returns: the predictive residuals for a given sample
        """
        random_sets = self._get_random_sets()
        sample_counts = self._get_sample_counts(sample, random_sets)
        data_counts = self._get_sample_counts(self.data_set, random_sets)
        residuals = sample_counts - data_counts
        return residuals

    def _get_sample_counts(self, sample, random_sets):
        # sample shape: (sample_size, dimension)
        # random_sets shape: (set_count, 2, dimension)

        # push to the GPU
        if self.gpu:
            sample = sample.cuda()
            random_sets = random_sets.cuda()

        # expand the sample to match the random sets
        extended_sample = sample.unsqueeze(1).repeat(1, self.set_count, 1)
        extended_random_sets = random_sets.repeat(sample.shape[0], 1, 1, 1)

        # get lower bounds and upper bounds
        lower_bounds = extended_random_sets[:, :, 0, :]
        upper_bounds = extended_random_sets[:, :, 1, :]
        upper_diffs = (
            upper_bounds - extended_sample
        )  # samples is below the UPPER bound
        lower_diffs = (
            extended_sample - lower_bounds
        )  # samples is above the LOWER bound
        upper_mask = (
            upper_diffs > 0
        ) * 1  # 1 if the sample is below the upper bound
        lower_mask = (
            lower_diffs > 0
        ) * 1  # 1 if the sample is above the lower bound

        in_set_mask = torch.prod(upper_mask * lower_mask, dim=2)
        set_counts = torch.sum(in_set_mask, dim=0)

        return set_counts


if __name__ == "__main__":
    run_one_dimensional = True
    run_two_dimensional = True
    run_cross_validation = False
    generate_results = True
    if run_one_dimensional:
        dimension = 1
        torch.manual_seed(3)
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
        set_count = 8000
        if generate_results:
            for (
                j,
                intensity,
                data_set,
                domain,
                order,
                intensity_function,
            ) in zip(
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
                    data_set,
                    domains[j],
                    dimension,
                    set_count=set_count,
                    gpu=True,
                )

                method_names = ["osegcp", "vbpp", "lbpp", "rkhs"]
                sample_names = ["synth1", "synth2", "synth3"]
                mean_std_data = {"synth1": [], "synth2": [], "synth3": []}
                residuals = torch.zeros(
                    set_count, sample_count, len(method_names)
                )
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
                        osegcp_sample,
                        vbpp_sample,
                        lbpp_sample,
                        rkhs_sample,
                    ]

                    # get the residuals for this sample
                    residual_means = torch.zeros(
                        sample_count, len(predictive_samples)
                    )
                    residual_std = torch.zeros(
                        sample_count, len(predictive_samples)
                    )
                    for k, sample, method_name in zip(
                        range(len(predictive_samples)),
                        predictive_samples,
                        method_names,
                    ):
                        sample = sample.unsqueeze(1)
                        residuals[
                            :, i, k
                        ] = empirical_coverage_runner.check_sample(sample)
                final_means = torch.mean(residuals, dim=0)
                final_squared_means = torch.mean(final_means**2, dim=0)
                print("\n")
                # print("final means:", final_means)
                print("intensity:", intensity)
                print("final squared means:", final_squared_means)

    if run_two_dimensional:
        # print(torch.seed())
        torch.manual_seed(3)
        sample_count = 100
        dimension = 2
        order = 15
        set_count = 8000
        data_loc = "/home/william/phd/programming_projects/gcp_rssb/datasets/comparison_experiments/data_and_samples/samples_update/samples2D/"
        domains = [
            torch.Tensor([[0, 1], [0, 1]]),  # redwood
            torch.Tensor([[0, 1], [0, 1]]),  # white oak
        ]
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
        mean_results = torch.zeros(sample_count, 4, len(data_sets))
        std_results = torch.zeros(sample_count, 4, len(data_sets))
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
                osegcp.add_data(data_set, domain)
                empirical_coverage_runner = EmpiricalCoverageRunner(
                    data_set,
                    domains[j],
                    dimension,
                    set_count=set_count,
                    gpu=True,
                )

                method_names = ["osegcp", "vbpp", "lbpp", "rkhs"]
                residuals = torch.zeros(
                    set_count, sample_count, len(method_names)
                )
                for i in range(sample_count):
                    osegcp_sample = osegcp.get_posterior_predictive_sample()
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
                        vbpp_sample,
                        lbpp_sample,
                        rkhs_sample,
                    ]

                    # get the residuals for this sample
                    residual_means = torch.zeros(
                        sample_count, len(predictive_samples)
                    )
                    residual_std = torch.zeros(
                        sample_count, len(predictive_samples)
                    )
                    for k, sample, method_name in zip(
                        range(len(predictive_samples)),
                        predictive_samples,
                        method_names,
                    ):
                        residuals[
                            :, i, k
                        ] = empirical_coverage_runner.check_sample(sample)
                final_means = torch.mean(residuals, dim=0)
                final_squared_means = torch.mean(final_means**2, dim=0)
                print("\n")
                # print("final means:", final_means)
                print("intensity:", intensity)
                print("final squared means:", final_squared_means)
                results_table = print_results_table(residuals)

    if run_cross_validation:
        # check each order and run the EC for it
        sample_count = 100
        dimension = 2
        min_range = 10
        orders = range(min_range, 20)
        set_count = 9000
        data_loc = "/home/william/phd/programming_projects/gcp_rssb/datasets/comparison_experiments/data_and_samples/samples_update/samples2D/"
        synth_data_sets = []
        redwood_df = pd.read_csv(
            "/home/william/phd/programming_projects/gcp_rssb/datasets/spatial-2D/redwood_full.csv"
        )
        white_oak_df = pd.read_csv(
            "/home/william/phd/programming_projects/gcp_rssb/datasets/spatial-2D/white_oak.csv"
        )
        # data_set = torch.tensor(df.values).squeeze()
        domains = [
            torch.Tensor([[0, 1], [0, 1]]),  # redwood
            # torch.Tensor([[0, 1], [0, 1]]),  # white oak
        ]
        data_sets = [
            torch.Tensor(redwood_df.values).squeeze(),
            # torch.Tensor(white_oak_df.values).squeeze(),
        ]
        start_time = time.perf_counter()
        # set up the basis
        mean_results = torch.zeros(sample_count, 1, len(data_sets))
        std_results = torch.zeros(sample_count, 1, len(data_sets))
        ec_criterion = torch.zeros(len(orders), len(data_sets))
        for order in orders:
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
                empirical_coverage_runner = EmpiricalCoverageRunner(
                    data_set,
                    domains[j],
                    dimension,
                    set_count=set_count,
                    gpu=True,
                )

                # add the data
                osegcp.add_data(data_set, domain)

                # method_names = ["osegcp", "vbpp", "lbpp", "rkhs"]
                residuals = torch.zeros(set_count, sample_count)
                for i in range(sample_count):
                    osegcp_sample = osegcp.get_posterior_predictive_sample()

                    # get the residuals for this sample
                    residuals[:, i] = empirical_coverage_runner.check_sample(
                        osegcp_sample
                    )
                final_means = torch.mean(residuals, dim=0)
                final_squared_means = torch.mean(final_means**2, dim=0)
                ec_criterion[order - min_range, j] = final_squared_means

        end_time = time.perf_counter()
        print("time taken:", end_time - start_time)
        print("EC criterion:", ec_criterion)
        print("recommended order:", orders[torch.argmin(ec_criterion)])
        plt.plot(orders, ec_criterion[:, 0], label="redwood")
        # plt.plot(orders, ec_criterion[:, 1], label="white oak")
        plt.legend()
        plt.show()
