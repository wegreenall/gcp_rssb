import unittest
import torch
import torch.distributions as D
from gcp_rssb.methods.gcp_ose import (
    OrthogonalSeriesCoxProcess,
    GCPOSEHyperparameters,
)
from ortho.basis_functions import Basis, standard_chebyshev_basis


class TestGcpOse(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.sample_data = D.Uniform(0, 1).sample((10, 1))

        # set up the class
        self.basis_functions = standard_chebyshev_basis
        self.dimension = 1
        self.order = 10
        self.parameters = {}
        self.ortho_basis = Basis(
            self.basis_functions, self.dimension, self.order, self.parameters
        )
        self.hyperparameters = GCPOSEHyperparameters(basis=self.ortho_basis)
        self.gcp_ose = OrthogonalSeriesCoxProcess(self.hyperparameters)

    def test_get_ose_coeffics(self):
        self.gcp_ose.add_data(self.sample_data)
        ose_coeffics = self.gcp_ose._get_ose_coeffics()
        self.assertEqual(ose_coeffics.shape, (self.order,))

    def test_add_data(self):
        self.gcp_ose.add_data(self.sample_data)
        self.assertTrue((self.gcp_ose.data_points == self.sample_data).all())


if __name__ == "__main__":
    unittest.main()
