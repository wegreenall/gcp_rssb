import unittest
import torch
import torch.distributions as D
from gcp_rssb.methods.gcp_ose import (
    OrthogonalSeriesCoxProcess,
    GCPOSEHyperparameters,
    Mapping,
)
from ortho.basis_functions import Basis, standard_chebyshev_basis


# @unittest.skip("just testing Mapping tests")
class TestGcpOse(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.sample_size = 50
        self.sample_data = D.Uniform(0, 1).sample((50, 1))
        self.sigma = 1.0

        # set up the class
        self.basis_functions = standard_chebyshev_basis
        self.dimension = 1
        self.order = 10
        self.parameters = {}
        self.ortho_basis = Basis(
            self.basis_functions, self.dimension, self.order, self.parameters
        )
        self.hyperparameters = GCPOSEHyperparameters(basis=self.ortho_basis)
        self.gcp_ose = OrthogonalSeriesCoxProcess(
            self.hyperparameters, self.sigma
        )

    def test_get_ose_coeffics(self):
        self.gcp_ose.add_data(self.sample_data)
        ose_coeffics = self.gcp_ose._get_ose_coeffics()
        self.assertEqual(ose_coeffics.shape, (self.order,))

    def test_add_data(self):
        self.gcp_ose.add_data(self.sample_data)
        self.assertTrue((self.gcp_ose.data_points == self.sample_data).all())

    def test_get_posterior_mean_order(self):
        self.gcp_ose.add_data(self.sample_data)
        posterior_mean = self.gcp_ose._get_posterior_mean()
        self.assertEqual(posterior_mean.get_order(), self.order)

    def test_get_posterior_mean_values(self):
        self.gcp_ose.add_data(self.sample_data)
        posterior_mean = self.gcp_ose._get_posterior_mean()

        # check the coefficients
        torch.manual_seed(1)
        self.sample_data = D.Uniform(0, 1).sample((self.sample_size, 1))
        test_coefficients = torch.sum(
            self.ortho_basis(self.sample_data), dim=0
        )
        self.assertTrue(
            torch.allclose(
                posterior_mean.get_coefficients(), test_coefficients
            )
        )

    @unittest.skip("Not implemented")
    def test_predict(self):
        pass

    @unittest.skip("Not implemented")
    def test_evaluate(self):
        pass

    def test_train(self):
        self.gcp_ose.add_data(self.sample_data)
        self.gcp_ose.train()
        self.assertTrue(self.gcp_ose.trained)
        self.assertTrue(self.gcp_ose.eigenvalues is not None)


class TestMapping(unittest.TestCase):
    def setUp(self):
        # set up the class
        self.sample_size = 50
        self.sample_data = D.Uniform(0, 1).sample((self.sample_size, 1))

        # set up the basis
        self.basis_functions = standard_chebyshev_basis
        self.dimension = 1
        self.order = 10
        self.parameters = {}
        self.sigma = torch.tensor(1.0)
        self.ortho_basis = Basis(
            self.basis_functions, self.dimension, self.order, self.parameters
        )
        self.hyperparameters = GCPOSEHyperparameters(basis=self.ortho_basis)
        self.gcp_ose = OrthogonalSeriesCoxProcess(
            self.hyperparameters, self.sigma
        )
        self.gcp_ose.add_data(self.sample_data)
        self.posterior_mean = self.gcp_ose._get_posterior_mean()

        # mapping class
        self.mapping = Mapping(
            self.ortho_basis, self.sample_data, self.posterior_mean, self.sigma
        )

        # initial eigenvalues
        self.initial_eigenvalues = torch.ones((self.sample_size, 1))

    def test_call(self):
        result = self.mapping(self.initial_eigenvalues)
        self.assertEqual(result.shape, torch.Size([self.order]))

    def test_get_basis_matrix(self):
        basis_matrix = self.mapping._get_basis_matrix()
        self.assertEqual(
            basis_matrix.shape, torch.Size([self.sample_size, self.order])
        )

    def test_get_weight_matrix(self):
        weight_matrix = self.mapping._get_weight_matrix()
        self.assertEqual(
            weight_matrix.shape, torch.Size([self.sample_size, self.order])
        )

    def test_get_design_matrix(self):
        design_matrix = self.mapping._get_design_matrix()
        self.assertEqual(
            design_matrix.shape, torch.Size([self.order, self.order])
        )

    def test_get_sigma(self):
        """
        Check that it's a scalar
        """
        sigma = self.mapping._get_sigma()
        self.assertEqual(sigma.shape, torch.Size([]))

    def test_get_pseudodata(self):
        y = self.mapping._get_pseudodata()
        self.assertEqual(y.shape, torch.Size([self.sample_size]))

    def test_get_Wy(self):
        Wy = self.mapping._get_Wy()
        self.assertEqual(Wy.shape, torch.Size([self.order]))

    def test_get_WPhi(self):
        WPhi = self.mapping._get_WPhi()
        self.assertEqual(WPhi.shape, torch.Size([self.order, self.order]))

    def test_get_PhiY(self):
        PhiY = self.mapping._get_PhiY()
        self.assertEqual(PhiY.shape, torch.Size([self.order]))


if __name__ == "__main__":
    unittest.main()
