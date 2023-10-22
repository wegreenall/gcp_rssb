import torch
from gcp_rssb.method import Method
from data import Data

from ortho.basis_functions import Basis
from mercergp.MGP import HilbertSpaceElement
from dataclasses import dataclass


@dataclass
class GCPOSEHyperparameters:
    basis: Basis


class OrthogonalSeriesCoxProcess(Method):
    """
    Represents our Gaussian Cox process method.
    """

    def __init__(self, GCPOSEHyperparameters):
        self.hyperparameters = GCPOSEHyperparameters
        self.data_points = None
        self.ose_coeffics = None

    def add_data(self, data_points):
        """
        Adds the data to the model.

        Then, captures the orthogonal series estimate coefficients of the
        intensity function.
        """
        self.data_points = data_points
        self.ose_coeffics = self._get_ose_coeffics()

    def get_kernel(self, left_points, right_points):
        pass

    def train(self):
        pass

    def predict(self, test_points):
        pass

    def evaluate(self, test_points, method):
        pass

    def _get_ose_coeffics(self):
        """
        Calculates the orthogonal series estimate given the data points and
        the basis functions.

        Return shape: [order]
        """
        basis = self.hyperparameters.basis
        design_matrix = basis(self.data_points)
        coeffics = torch.sum(design_matrix, dim=0)
        return coeffics

    def _get_posterior_mean(self):
        """
        Calculates the posterior mean of the intensity function.

        Return shape: [order]
        """
        if self.ose_coeffics is None:
            raise ValueError(
                "The orthogonal series estimate coefficients are not set.\
                Please add data to the model first."
            )
        return HilbertSpaceElement(
            self.hyperparameters.basis, self.ose_coeffics
        )


if __name__ == "__main__":
    pass
