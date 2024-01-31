import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import pandas as pd
from phd_diagrams.plot import Plot, Figure, Section

from gcp_rssb.methods.gcp_ose import GCPOSEHyperparameters

from ortho.basis_functions import Basis, standard_chebyshev_basis
from typing import List


def gaussian_kernel(x, y):
    """
    The Gaussian kernel, for the MMD calculation.
    """
    # x = torch.expand_dims(x, axis=1).repeat(len(y), axis=1)
    # y = torch.expand_dims(y, axis=1).repeat(len(x), axis=1).transpose()
    b = 2
    x = torch.unsqueeze(x, 1)
    y = torch.unsqueeze(y, 1).transpose(1, 0)
    result = torch.exp(-((x - y) ** 2) / b)
    return result


class ExpositionPlot(Plot):
    def _plot_code(self):
        fineness = 600
        x_axis = torch.linspace(-6, 6, fineness)
        z = D.Normal(0, 1).sample((fineness,))

        # get kernel matrix
        K = gaussian_kernel(x_axis, x_axis) + torch.eye(fineness) * 1e-5
        # plt.imshow(K)
        # plt.show()
        chol = torch.linalg.cholesky(torch.tensor(K))
        f_sample = chol @ z

        plt.plot(x_axis.cpu(), f_sample.cpu())
        sub_samples = torch.where(f_sample > 0, f_sample, torch.nan)
        plt.plot(x_axis.cpu(), sub_samples.cpu(), color="red")
        plt.xlabel("$\mathcal{X}$")
        plt.ylabel("$f(x)$")
        plt.legend(["$f(x)$", "$f^+(x)$"])


class ExpositionFigure(Figure):
    def _get_short_caption(self):
        caption = "Examples of semipositive Cox processes"
        return caption

    def _get_long_caption(self):
        caption = "Examples of the Cox process model outlined in this chapter. \
        Note the difference between this\
        and using $\max(0, \cdot)$ as a link function.\
        We denote the Gaussian process sample as $f(x)$ and the semipositive sample as $f^+(x)$."
        return caption


def run_exposition_graphs(save):
    plot_name = "semipositive_gaussian_cox_process"
    figure_name = "semipositive_gaussian_cox_process"
    plots = [ExpositionPlot(plot_name)]
    figure = ExpositionFigure(
        plots, Section.GAUSSIAN_COX_PROCESSES, figure_name
    )
    figure.run_figure(save=save)


if __name__ == "__main__":
    torch.manual_seed(16399665084767555165)
    save = False
    section = Section.GAUSSIAN_COX_PROCESSES
    run_exposition_graphs(save)
