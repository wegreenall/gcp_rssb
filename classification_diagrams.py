import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from phd_diagrams.plot import Plot, Figure, Section

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

torch.manual_seed(3)


class ClassificationPlot1d(Plot):
    def _get_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Class 1 point process
        max_time = 10.0
        alpha_1 = 8.0
        beta_1 = 1.0
        intensity_1 = lambda x: 100 * torch.exp(D.Gamma(alpha_1, beta_1).log_prob(x))

        alpha_2 = 3.0
        beta_2 = 1.0
        intensity_2 = lambda x: 100 * torch.exp(D.Gamma(alpha_2, beta_2).log_prob(x))
        # x = torch.linspace(0.1, max_time, 1000)

        poisson_process_1 = PoissonProcess(intensity_1, max_time)
        poisson_process_2 = PoissonProcess(intensity_2, max_time)
        poisson_process_1.simulate()
        poisson_process_2.simulate()

        class_1_data = poisson_process_1.get_data()
        class_2_data = poisson_process_2.get_data()
        return class_1_data, class_2_data


class ClassificationPlot1dBasic(ClassificationPlot1d):
    def _plot_code(self):
        order = 6
        max_time = 10.0
        fineness = 300
        basis_functions = standard_chebyshev_basis
        dimension = 1
        parameters: dict = {"lower_bound": 0.0, "upper_bound": max_time + 0.1}
        ortho_basis = Basis(basis_functions, dimension, order, parameters)
        test_axis = torch.linspace(0.0, max_time, fineness)

        # prior parameters
        prior_mean = torch.tensor(0.0)
        alpha = torch.tensor(1.5)
        beta = torch.tensor(4.0)
        nu = torch.tensor(0.01)

        prior_parameters_1 = PriorParameters(prior_mean, alpha, beta, nu)
        prior_parameters_2 = PriorParameters(prior_mean, alpha, beta, nu)

        hyperparameters_1 = GCPOSEHyperparameters(basis=ortho_basis, dimension=1)
        hyperparameters_2 = GCPOSEHyperparameters(basis=ortho_basis, dimension=1)

        classifier = GCPClassifier(
            2,
            [prior_parameters_1, prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
        )
        class_1_data, class_2_data = self._get_data()
        # add the data
        classifier.add_data(class_1_data, 0)
        classifier.add_data(class_2_data, 1)

        # now predict
        class_probs = classifier.predict_point(test_axis).cpu().numpy()
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 0])
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 1])


class ClassificationPlot1dDataInformed(ClassificationPlot1d):
    def _plot_code(self):
        order = 6
        max_time = 10.0
        fineness = 300
        basis_functions = standard_chebyshev_basis
        dimension = 1
        parameters: dict = {"lower_bound": 0.0, "upper_bound": max_time + 0.1}
        ortho_basis = Basis(basis_functions, dimension, order, parameters)
        test_axis = torch.linspace(0.0, max_time, fineness)

        # prior parameters
        nu = torch.tensor(0.01)

        prior_parameters_1 = DataInformedPriorParameters(nu)
        prior_parameters_2 = DataInformedPriorParameters(nu)

        hyperparameters_1 = GCPOSEHyperparameters(basis=ortho_basis, dimension=1)
        hyperparameters_2 = GCPOSEHyperparameters(basis=ortho_basis, dimension=1)

        classifier = GCPClassifier(
            2,
            [prior_parameters_1, prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
            data_informed=True,
        )
        class_1_data, class_2_data = self._get_data()

        # add the data
        classifier.add_data(class_1_data, 0)
        classifier.add_data(class_2_data, 1)

        # now predict
        class_probs = classifier.predict_point(test_axis).cpu().numpy()
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 0])
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 1])


class ClassificationPlot1dCombination(ClassificationPlot1d):
    def _plot_code(self):
        order = 10
        max_time = 10.0
        fineness = 200
        basis_functions = standard_chebyshev_basis
        dimension = 1
        parameters: dict = {"lower_bound": 0.0, "upper_bound": max_time + 0.1}
        ortho_basis = Basis(basis_functions, dimension, order, parameters)
        test_axis = torch.linspace(0.0, max_time, fineness)

        # prior parameters
        prior_mean = torch.tensor(0.0)
        alpha = torch.tensor(1.5)
        beta = torch.tensor(0.0)
        nu = torch.tensor(0.01)

        # data_informed_nu
        data_informed_nu = torch.tensor(0.02)

        prior_parameters_1 = PriorParameters(prior_mean, alpha, beta, nu)
        prior_parameters_2 = PriorParameters(prior_mean, alpha, beta, nu)

        # data informed prior parameters
        data_informed_prior_parameters_1 = DataInformedPriorParameters(data_informed_nu)
        data_informed_prior_parameters_2 = DataInformedPriorParameters(data_informed_nu)

        hyperparameters_1 = GCPOSEHyperparameters(basis=ortho_basis, dimension=1)
        hyperparameters_2 = GCPOSEHyperparameters(basis=ortho_basis, dimension=1)

        basic_classifier = GCPClassifier(
            2,
            [prior_parameters_1, prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
            data_informed=False,
        )
        data_informed_classifier = GCPClassifier(
            2,
            [data_informed_prior_parameters_1, data_informed_prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
            data_informed=True,
        )
        class_1_data, class_2_data = self._get_data()

        # add the data
        basic_classifier.add_data(class_1_data, 0)
        basic_classifier.add_data(class_2_data, 1)
        data_informed_classifier.add_data(class_1_data, 0)
        data_informed_classifier.add_data(class_2_data, 1)

        # now predict: basic
        basic_class_probs = basic_classifier.predict_point(test_axis).cpu().numpy()
        plt.plot(test_axis.cpu().numpy(), basic_class_probs[:, 0], color="red")
        plt.plot(test_axis.cpu().numpy(), basic_class_probs[:, 1], color="green")

        # now predict: data_informed
        data_informed_class_probs = (
            data_informed_classifier.predict_point(test_axis).cpu().numpy()
        )
        plt.plot(
            test_axis.cpu().numpy(),
            data_informed_class_probs[:, 0],
            color="red",
            linestyle="--",
        )
        plt.plot(
            test_axis.cpu().numpy(),
            data_informed_class_probs[:, 1],
            color="green",
            linestyle="--",
        )

        plt.scatter(
            class_1_data.cpu().numpy(),
            torch.zeros_like(class_1_data).cpu().numpy(),
            color="red",
        )
        plt.scatter(
            class_2_data.cpu().numpy(),
            torch.zeros_like(class_2_data).cpu().numpy(),
            color="green",
        )


class ClassificationFigure1d(Figure):
    def _get_short_caption(self):
        return "One-dimensional classification model example"

    def _get_long_caption(self):
        caption = "One-dimensional example of a classification problem.\
        Curves shown are probability estimates as constructed using the Barvinok estimator outlined in this chapter.\
        Intensities are given by scaled Gamma distributions with different parameters;\
        class 1 has intensity $100 \\times Gamma(8, 1)$; class 2 has intensity $100 \\times Gamma(3, 1)$.\
        Solid line: Prior hyperparameters are $\\mu_i=0.0$, $\\alpha=1.5$, $\\beta=0.0$, $\\weightingparameter$=0.01.\
        Dashed line: Prior hyperparameters are $\\mu_i=0.0$, $\\alpha=1.5$, $\\beta=0.0$, $\\weightingparameter$=0.02.\
        Note the effect of increasing the weighting parameter $\\weightingparameter$ leads to less certain estimates."
        return caption


def run_classification_diagrams(save):
    section = Section.GAUSSIAN_COX_PROCESSES
    run_basic_plot = False
    # run_data_informed_plot = False
    run_combined_plot = True

    # basic plot
    figure_name = "classification_one_dimensional"
    if run_basic_plot:
        plot_1_name = "classification_one_dimensional_basic"
        # plot_2_name = "classification_one_dimensional_data"
        basic_plot = ClassificationPlot1dBasic(plot_1_name)
        # data_informed_plot = ClassificationPlot1dDataInformed(plot_2_name)
        basic_figure = ClassificationFigure1d([basic_plot], section, figure_name)
        basic_figure.run_figure(save=save)

    if run_combined_plot:
        figure_name = "classification_one_dimensional"
        combined_plot_name = "classification_one_dimensional_basic"
        combined_plot = ClassificationPlot1dCombination(combined_plot_name)
        combined_figure = ClassificationFigure1d([combined_plot], section, figure_name)
        combined_figure.run_figure(save=save)


if __name__ == "__main__":
    save = True
    run_classification_diagrams(save)
