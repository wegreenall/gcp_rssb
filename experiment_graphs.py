import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import pandas as pd
from phd_diagrams.plot import Plot, Figure, Section

from gcp_rssb.methods.gcp_ose import GCPOSEHyperparameters
from gcp_rssb.methods.gcp_ose_classifier import loop_hafnian_estimate
from gcp_rssb.methods.gcp_ose_bayesian import (
    PriorParameters,
    DataInformedPriorParameters,
    BayesianOrthogonalSeriesCoxProcess,
    BayesianOrthogonalSeriesCoxProcessObservationNoise,
)

from ortho.basis_functions import Basis, standard_chebyshev_basis
from typing import List

torch.manual_seed(3)
from termcolor import colored
import time


class SynthPlot(Plot):
    def _plot_code(self):
        # set up the model
        dim = 1
        basis_functions = [standard_chebyshev_basis]
        fineness = 500
        # min_time = -0.1
        # max_time = 50.1
        parameters: List = [
            {
                "lower_bound": self.min_time - 0.1,
                "upper_bound": self.max_time + 0.1,
                "variance_parameter": 1.0,
                # "chebyshev": "first",
            },
        ]
        chebyshev_basis = Basis(basis_functions, dim, self.order, parameters)
        gcp_ose_hyperparameters = GCPOSEHyperparameters(
            basis=chebyshev_basis,
            dimension=dim,
        )
        prior_parameters = PriorParameters(mean=0.0, alpha=1.5, beta=2.0, nu=0.12)
        gcp_ose_model = BayesianOrthogonalSeriesCoxProcessObservationNoise(
            gcp_ose_hyperparameters, prior_parameters
        )

        """
        Now we load each of the data sets and run the model.
        """
        # load the data
        synth_data_sets = []
        for i in range(10):
            df = pd.read_csv(
                "/home/william/phd/programming_projects/gcp_rssb/datasets/{}/observation{}.csv".format(
                    self.folder_name, i
                )
            )
            data_set = torch.tensor(df.values).squeeze()
            synth_data_sets.append(data_set)
            print(torch.max(data_set))

        posterior_means_list = []
        posterior_mean_coeffics_list = []
        eigenvalues_list = []
        data_set = synth_data_sets[0]

        # timing
        start_value = time.perf_counter_ns()
        gcp_ose_model.add_data(data_set)
        end_value = time.perf_counter_ns()
        print("Time taken to add data: {}".format(end_value - start_value))

        # get the various stuff
        eigenvalues = gcp_ose_model._get_posterior_eigenvalue_estimates().cpu().numpy()
        eigenvalues_list.append(eigenvalues)
        posterior_mean_coeffics = (
            gcp_ose_model._get_posterior_mean_coefficients().cpu().numpy()
        )
        posterior_mean_coeffics_list.append(posterior_mean_coeffics)
        posterior_mean = gcp_ose_model._get_posterior_mean()
        posterior_means_list.append(posterior_mean)

        x_axis = torch.linspace(0, self.max_time, fineness)

        for posterior_mean in posterior_means_list:
            plt.scatter(
                data_set.cpu().numpy(),
                torch.zeros_like(data_set).cpu().numpy(),
                marker="x",
            )
            plt.plot(x_axis.cpu().numpy(), posterior_mean(x_axis).cpu().numpy())
            plt.plot(x_axis.cpu().numpy(), self.ground_truth(x_axis).cpu().numpy())
        plt.legend(
            (
                "Posterior mean",
                "Ground truth",
            )
        )

    @staticmethod
    def ground_truth(x_axis):
        raise NotImplementedError


class Synth1Figure(Figure):
    def _get_short_caption(self):
        return "Gaussian Cox Process method: synthetic example 1"

    def _get_long_caption(self):
        return "An example of the proposed Cox process method applied to a \
                synthetic data set."


class Synth1Plot(SynthPlot):
    def __init__(self, name: str):
        self.folder_name = "synth1"
        self.min_time = -0.1
        self.max_time = 50.1
        self.order = 8
        super().__init__(name)

    @staticmethod
    def ground_truth(x_axis):
        return 2 * torch.exp(-x_axis / 15) + torch.exp(-(((x_axis - 25) / 10) ** 2))


class Synth2Figure(Figure):
    def _get_short_caption(self):
        return "Gaussian Cox Process method: synthetic example 2"

    def _get_long_caption(self):
        return "An example of the proposed Cox process method applied to a synth\
    etic data set."


class Synth2Plot(SynthPlot):
    def __init__(self, name: str):
        self.folder_name = "synth2"
        self.min_time = -0.1
        self.max_time = 5.1
        self.order = 22
        super().__init__(name)

    @staticmethod
    def ground_truth(x_axis):
        return 5 * torch.sin(x_axis**2) + 6


class Synth3Figure(Figure):
    def _get_short_caption(self):
        return "Gaussian Cox Process method: synthetic example 3"

    def _get_long_caption(self):
        return "An example of the proposed Cox process method applied to a\
    synthetic data set."


class Synth3Plot(SynthPlot):
    def __init__(self, name: str):
        self.folder_name = "synth3"
        self.min_time = -2
        self.max_time = 102
        self.order = 8
        super().__init__(name)

    @staticmethod
    def ground_truth(x_axis):
        idx_less_25 = [i for i in range(len(x_axis)) if x_axis[i] < 25]
        idx_less_50 = [i for i in range(len(x_axis)) if 25 <= x_axis[i] < 50]
        idx_less_75 = [i for i in range(len(x_axis)) if 50 <= x_axis[i] < 75]
        other_idx = [i for i in range(len(x_axis)) if x_axis[i] >= 75]
        return torch.cat(
            [
                0.04 * x_axis[idx_less_25] + 2,
                -0.08 * x_axis[idx_less_50] + 5,
                0.06 * x_axis[idx_less_75] - 2,
                0.02 * x_axis[other_idx] + 1,
            ]
        )


class Spatial2dFigure(Figure):
    def _get_short_caption(self):
        return ""

    def _get_long_caption(self):
        return "Synthetic 1"


class Spatial2dPlot(Plot):
    def _plot_code(self):
        dim = 2
        basis_functions = [standard_chebyshev_basis] * 2
        fineness = 100
        # breakpoint()
        df = pd.read_csv(
            "/home/william/phd/programming_projects/gcp_rssb/datasets/spatial-2D/{}.csv".format(
                self.name
            )
        )
        data_set = torch.tensor(df.values)

        min_time_x = data_set[:, 0].min()
        max_time_x = data_set[:, 0].max()
        max_time_y = data_set[:, 1].max()
        min_time_y = data_set[:, 1].min()

        # 2d plotting stuff
        x_axis = torch.linspace(0, 1, fineness)
        y_axis = torch.linspace(0, 1, fineness)
        X, Y = torch.meshgrid(x_axis, y_axis, indexing="ij")
        Z = torch.stack((X, Y), dim=2)
        Z = Z.reshape(-1, 2)

        parameters: List = [
            {
                "lower_bound": min_time_x - 0.1,
                "upper_bound": max_time_x + 0.1,
                #                "variance_parameter": 1.0,
            },
            {
                "lower_bound": min_time_y - 0.1,
                "upper_bound": max_time_y + 0.1,
                #                "variance_parameter": 1.0,
            },
        ]
        order = 20
        # print("data set shape: {}".format(data_set.shape))
        # print("data set name: {}".format(data_set_name))

        chebyshev_basis = Basis(basis_functions, dim, order, parameters)
        gcp_ose_hyperparameters = GCPOSEHyperparameters(
            basis=chebyshev_basis,
            dimension=dim,
        )
        prior_parameters = PriorParameters(mean=0.0, alpha=1.5, beta=2.0, nu=0.12)
        gcp_ose_model = BayesianOrthogonalSeriesCoxProcessObservationNoise(
            gcp_ose_hyperparameters, prior_parameters
        )
        gcp_ose_model.add_data(data_set)

        posterior_mean = gcp_ose_model._get_posterior_mean()

        # plot posterior mean
        output = posterior_mean(Z).cpu().numpy().reshape(fineness, fineness)
        plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), output, cmap="YlOrRd")
        plt.scatter(
            data_set[:, 0].cpu().numpy(),
            data_set[:, 1].cpu().numpy(),
            marker="x",
            color="black",
        )


class RedWoodsFullPlot(Spatial2dPlot):
    def __init__(self):
        name = "redwood_full"
        super().__init__(name)


class RedWoodsCalifornia(Spatial2dPlot):
    def __init__(self):
        name = "redwoods_california"
        super().__init__(name)


class WhiteOak(Spatial2dPlot):
    def __init__(self):
        name = "white_oak"
        super().__init__(name)


def run_synthetic_plots(save):
    section = Section.GAUSSIAN_COX_PROCESSES
    name = "synthone"
    synth_1_plot = Synth1Plot(name)
    plots = [synth_1_plot]
    synth_1_figure = Synth1Figure(plots, section, name)
    synth_1_figure.run_figure(save=save)

    name = "synthtwo"
    synth_2_plot = Synth2Plot(name)
    plots = [synth_2_plot]
    synth_2_figure = Synth2Figure(plots, section, name)
    synth_2_figure.run_figure(save=save)

    name = "synththree"
    synth_3_plot = Synth3Plot(name)
    plots = [synth_3_plot]
    synth_3_figure = Synth3Figure(plots, section, name)
    synth_3_figure.run_figure(save=save)

    dataset_names = [
        # "new_zealand",
        "redwood_full",
        "redwoods_california",
        # "swedish_pines",
        "white_oak",
    ]

    # Redwoods Full Plot
    plot = [RedWoodsFullPlot()]
    redwoods_full_figure = Spatial2dFigure(plot, section, "redwoodfull")
    redwoods_full_figure.run_figure(save=save)

    # Redwoods California Plot
    plot = [RedWoodsCalifornia()]
    redwoods_california_figure = Spatial2dFigure(plot, section, "redwoodscalifornia")
    redwoods_california_figure.run_figure(save=save)

    # White Oak Plot
    plot = [WhiteOak()]
    white_oak_figure = Spatial2dFigure(plot, section, "whiteoak")
    white_oak_figure.run_figure(save=save)


class SynthPlotICML(Plot):
    def __init__(self, name: str):
        # self.folder_name = "synth3"
        # self.min_time = -2
        # self.max_time = 102
        self.order = 8
        super().__init__(name)

    def _plot_code(self):
        # set up the model
        dim = 1
        basis_functions = [standard_chebyshev_basis]
        fineness = 500
        # min_time = -0.1
        # max_time = 50.1

        """
        Now we load each of the data sets and run the model.
        """
        # load the data
        synth_data_sets = []
        synth_data_sets_boundaries = []
        for i in range(3):
            df = pd.read_csv(
                "/home/william/phd/programming_projects/gcp_rssb/apostolis/seed_three/synth{}.csv".format(
                    i + 1
                )
            )
            data_set = torch.tensor(df.values).squeeze()
            synth_data_sets.append(data_set)
            synth_data_sets_boundaries.append(
                torch.tensor(
                    [
                        [torch.min(data_set)],
                        [torch.max(data_set)],
                    ]
                )
            )
            print(torch.max(data_set))
            print(torch.min(data_set))

        posterior_means_list = []
        posterior_mean_coeffics_list = []
        eigenvalues_list = []
        obs_boundaries = [50, 5, 100]
        print(colored("WHAT THE HELL?!", "red"))
        breakpoint()
        orders = [8, 16, 8]
        plot_types = [Synth1Plot, Synth2Plot, Synth3Plot]
        for i, data_set, boundaries, obs_boundary, order, plot_type in zip(
            range(0, 3),
            synth_data_sets,
            synth_data_sets_boundaries,
            obs_boundaries,
            orders,
            plot_types,
        ):
            parameters: List = [
                {
                    "lower_bound": -0.1,
                    "upper_bound": obs_boundary + 0.1,
                    "variance_parameter": 1.0,
                    "chebyshev": "second",
                },
            ]
            chebyshev_basis = Basis(basis_functions, dim, order, parameters)
            gcp_ose_hyperparameters = GCPOSEHyperparameters(
                basis=chebyshev_basis,
                dimension=dim,
            )
            prior_parameters = PriorParameters(mean=0.0, alpha=1.5, beta=2.0, nu=0.12)
            gcp_ose_model = BayesianOrthogonalSeriesCoxProcessObservationNoise(
                gcp_ose_hyperparameters, prior_parameters
            )
            gcp_ose_model.add_data(data_set)

            # get the various stuff
            eigenvalues = (
                gcp_ose_model._get_posterior_eigenvalue_estimates().cpu().numpy()
            )
            eigenvalues_list.append(eigenvalues)
            posterior_mean_coeffics = (
                gcp_ose_model._get_posterior_mean_coefficients().cpu().numpy()
            )
            posterior_mean_coeffics_list.append(posterior_mean_coeffics)
            posterior_mean = gcp_ose_model._get_posterior_mean()
            posterior_means_list.append(posterior_mean)
            fineness = 100
            x_axis = torch.linspace(0, obs_boundary, fineness)
            plt.plot(x_axis.cpu().numpy(), posterior_mean(x_axis).cpu().numpy())
            plt.plot(x_axis.cpu().numpy(), plot_type.ground_truth(x_axis).cpu().numpy())
            plt.scatter(
                data_set.cpu().numpy(),
                torch.zeros_like(data_set).cpu().numpy(),
                marker="x",
            )
            plt.show()
            torch.save(
                posterior_mean(x_axis).cpu(),
                "synth{}_posterior_mean.pt".format(i + 1),
            )

    # def ground_truth(self, x_axis):
    # raise NotImplementedError


class ComparisonDiagramRedwood(RedWoodsFullPlot):
    def __init__(self):
        super().__init__()
        self.use_tikz = False
        self.name = "redwood_full"

    def _plot_code(self):
        dim = 2
        basis_functions = [standard_chebyshev_basis] * 2
        fineness = 100
        # breakpoint()
        df = pd.read_csv(
            "/home/william/phd/programming_projects/gcp_rssb/datasets/spatial-2D/{}.csv".format(
                self.name
            )
        )
        data_set = torch.tensor(df.values)

        min_time_x = data_set[:, 0].min()
        max_time_x = data_set[:, 0].max()
        max_time_y = data_set[:, 1].max()
        min_time_y = data_set[:, 1].min()

        # 2d plotting stuff
        x_axis = torch.linspace(0, 1, fineness)
        y_axis = torch.linspace(0, 1, fineness)
        X, Y = torch.meshgrid(x_axis, y_axis, indexing="ij")
        Z = torch.stack((X, Y), dim=2)
        Z = Z.reshape(-1, 2)

        parameters: List = [
            {
                "lower_bound": min_time_x - 0.1,
                "upper_bound": max_time_x + 0.1,
                #                "variance_parameter": 1.0,
            },
            {
                "lower_bound": min_time_y - 0.1,
                "upper_bound": max_time_y + 0.1,
                #                "variance_parameter": 1.0,
            },
        ]
        order = 10
        # print("data set shape: {}".format(data_set.shape))
        # print("data set name: {}".format(data_set_name))

        chebyshev_basis = Basis(basis_functions, dim, order, parameters)
        gcp_ose_hyperparameters = GCPOSEHyperparameters(
            basis=chebyshev_basis,
            dimension=dim,
        )
        prior_parameters = PriorParameters(mean=0.0, alpha=1.5, beta=2.0, nu=0.12)
        gcp_ose_model = BayesianOrthogonalSeriesCoxProcessObservationNoise(
            gcp_ose_hyperparameters, prior_parameters
        )
        gcp_ose_model.add_data(data_set)

        posterior_mean = gcp_ose_model._get_posterior_mean()
        intensity = posterior_mean(Z).cpu().numpy().reshape(fineness, fineness)
        contourf = plt.contourf(
            X.cpu().numpy(),
            Y.cpu().numpy(),
            posterior_mean(Z).cpu().numpy().reshape(fineness, fineness),
            cmap="YlOrRd",
            levels=500,
        )
        plt.contour(X.cpu().numpy(), Y.cpu().numpy(), intensity, cmap="YlOrRd")
        plt.scatter(
            data_set[:, 0].cpu().numpy(),
            data_set[:, 1].cpu().numpy(),
            s=10.0,
            color="black",
            edgecolors="white",
        )


class ComparisonDiagramWhiteOak(ComparisonDiagramRedwood):
    def __init__(self):
        super().__init__()
        self.use_tikz = False
        self.name = "white_oak"


if __name__ == "__main__":
    save = False
    section = Section.GAUSSIAN_COX_PROCESSES
    run_comparison_diagrams = False
    run_phd_plots = False
    run_synth_plot_ICML = True
    if run_phd_plots:
        run_synthetic_plots(save)

    if run_synth_plot_ICML:
        plot = SynthPlotICML("synth")
        plot._plot_code()

    if run_comparison_diagrams:
        spatial_2d_plot = ComparisonDiagramRedwood()
        plots = [spatial_2d_plot]
        spatial_2d_figure = Spatial2dFigure(plots, section, "redwoodfull")
        spatial_2d_figure.run_figure(save=True)

        spatial_2d_plot = ComparisonDiagramWhiteOak()
        plots = [spatial_2d_plot]
        spatial_2d_figure = Spatial2dFigure(plots, section, "whiteoak")
        spatial_2d_figure.run_figure(save=True)
