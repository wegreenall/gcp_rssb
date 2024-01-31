import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from phd_diagrams.plot import Plot, Figure, Section
import pandas as pd

from gcp_rssb.data import PoissonProcess
from gcp_rssb.methods.gcp_ose import GCPOSEHyperparameters
from gcp_rssb.methods.gcp_ose_bayesian import (
    PriorParameters,
    BayesianOrthogonalSeriesCoxProcessObservationNoise,
    DataInformedPriorParameters,
    BayesianOrthogonalSeriesCoxProcess,
)

from ortho.basis_functions import Basis, standard_chebyshev_basis
from typing import List, Tuple
from gcp_rssb.methods.gcp_ose_classifier import GCPClassifier

torch.manual_seed(1)


class ClassificationPlot2d(Plot):
    def _plot_code(self):
        max_time = 1.0
        class_1_data = torch.load(
            "/home/william/phd/programming_projects/phd_diagrams/gaussian_cox_processes/classification_diagrams_2d_data/class_1_data.pt"
        )
        class_2_data = torch.load(
            "/home/william/phd/programming_projects/phd_diagrams/gaussian_cox_processes/classification_diagrams_2d_data/class_2_data.pt"
        )
        fineness = 50
        test_axis = torch.linspace(0.0, max_time, fineness)
        test_axes_x, test_axes_y = torch.meshgrid(test_axis, test_axis)
        estimators = torch.load(
            "/home/william/phd/programming_projects/phd_diagrams/gaussian_cox_processes/classification_diagrams_2d_data/estimators.pt"
        )
        estimators_2 = torch.load(
            "/home/william/phd/programming_projects/phd_diagrams/gaussian_cox_processes/classification_diagrams_2d_data/estimators_2.pt"
        )

        denominator = estimators + estimators_2

        tensor_estimators = (
            torch.Tensor(estimators).cpu() / torch.Tensor(denominator).cpu()
        )
        reshaped_tensor_estimators = tensor_estimators.reshape(
            fineness, fineness
        )
        plt.contourf(reshaped_tensor_estimators, levels=10)

        scaler = 50
        plt.scatter(
            scaler - scaler * class_1_data[:, 0].cpu(),
            scaler - scaler * class_1_data[:, 1].cpu(),
            marker="x",
            color="red",
        )

        plt.scatter(
            scaler - scaler * class_2_data[:, 0].cpu(),
            scaler - scaler * class_2_data[:, 1].cpu(),
            marker="o",
            color="blue",
        )
        plt.colorbar()


class ClassificationPlotBananaDiagram(Plot):
    def _plot_code(self):
        max_time = 1.0
        data = pd.read_csv(
            "/home/william/phd/programming_projects/phd_diagrams/gaussian_cox_processes/bananadataset.csv"
        )
        # data.head()
        # breakpoint()
        class_1_data_full = torch.Tensor(data[data["class"] == 1].values)
        class_1_data_x = class_1_data_full[:, 0]
        class_1_data_y = class_1_data_full[:, 1]
        class_2_data_full = torch.Tensor(data[data["class"] == 2].values)
        class_2_data_x = class_2_data_full[:, 0]
        class_2_data_y = class_2_data_full[:, 1]
        class_1_data = torch.vstack((class_1_data_x, class_1_data_y)).t()
        class_2_data = torch.vstack((class_2_data_x, class_2_data_y)).t()

        fineness = 50

        # get the axes for the plot
        test_axis = torch.linspace(-1.0, 1.0, fineness)
        test_axes_x, test_axes_y = torch.meshgrid(test_axis, test_axis)
        test_points = torch.vstack(
            (test_axes_x.ravel(), test_axes_y.ravel())
        ).t()

        # denominator = estimators + estimators_2
        plt.scatter(
            class_1_data_x.cpu(), class_1_data_y.cpu(), marker="x", color="red"
        )
        plt.scatter(
            class_2_data_x.cpu(),
            class_2_data_y.cpu(),
            marker="o",
            color="blue",
        )

        order = 10
        dimension = 2
        nu = torch.tensor(0.01)
        classifier_parameters = DataInformedPriorParameters(0.01)
        extra_window = 0.05
        min_time = -3.5
        max_time = 3.5
        parameters: dict = [
            {
                "lower_bound": min_time - extra_window,
                "upper_bound": max_time + extra_window,
            },
        ] * 2
        basis_functions = [standard_chebyshev_basis] * 2
        ortho_basis = Basis(basis_functions, dimension, order, parameters)
        hyperparameters = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=dimension
        )

        classifier = GCPClassifier(
            2, classifier_parameters, hyperparameters, data_informed=True
        )
        print("119")
        breakpoint()
        classifier.add_data(class_1_data, 0)
        print("121")
        breakpoint()
        classifier.add_data(class_2_data, 1)
        print("122")
        breakpoint()
        class_probs = classifier.predict_point(test_points)
        print("124")
        breakpoint()
        torch.save(class_probs, "./banana_class_probs.pt")
        plt.contourf(
            test_axes_x.cpu().numpy(),
            test_axes_y.cpu().numpy(),
            class_probs[:, 0].reshape(test_axes_x.shape).cpu().numpy(),
        )
        plt.show()


class ClassificationFigure2d(Figure):
    def _get_short_caption(self):
        return "Classification example: two-dimensional"

    def _get_long_caption(self):
        return "Application of the proposed point process classification method\
    to a two-dimensional example. The presented function, presented as a contour\
    plot, is the normalised probability ascribed to the class here whose\
    observations are denoted with red crosses."


class ClassificationFigureBanana(Figure):
    def _get_short_caption(self):
        return "Classification example: banana"

    def _get_long_caption(self):
        return ""


def run_classification_2d_diagrams(save):
    section = Section.GAUSSIAN_COX_PROCESSES
    name = "classification_two_dimensional"
    figure_name = "classification_two_dimensional"
    plot = ClassificationPlot2d(name)
    figure_2d = ClassificationFigure2d([plot], section, figure_name)
    figure_2d.run_figure(save=save)

    name = "classification_banana"
    figure_name = "classification_banana"
    banana_plot = ClassificationPlotBananaDiagram(name)
    figure = ClassificationFigureBanana([banana_plot], section, figure_name)
    figure.run_figure(save=save)


if __name__ == "__main__":
    name = "classification_two_dimensional"
    figure_name = "classification_two_dimensional"
    save = False
    # run_classification_2d_diagrams(save=save)

    section = Section.GAUSSIAN_COX_PROCESSES
    name = "classification_two_dimensional"
    figure_name = "classification_two_dimensional"
    banana_plot = ClassificationPlotBananaDiagram(name)
    figure = ClassificationFigureBanana([banana_plot], section, figure_name)
    figure.run_figure(save=save)
