import unittest
import torch
import torch.distributions as D

from gcp_rssb.empirical_coverage_experimental import EmpiricalCoverageRunner


class TestEmpiricalCoverage(unittest.TestCase):
    def setUp(self):
        self.data_set = torch.Tensor([[0.5, 0.5]])
        domains = torch.Tensor([[0, 1]])
        self.dimension = 1
        self.set_count = 200
        self.runner = EmpiricalCoverageRunner(
            self.data_set, domains, self.dimension, set_count=self.set_count
        )

    def test_random_areas_shape(self):
        runner = EmpiricalCoverageRunner(
            self.data_set,
            torch.Tensor([[0, 1]]),
            1,
            self.set_count,
        )
        random_areas = runner._get_random_sets()
        self.assertEqual(random_areas.shape, (self.set_count, 2, 1))

    def test_random_areas_sorted(self):
        random_areas = self.runner._get_random_sets()
        self.assertTrue(
            torch.all(random_areas[:, 0, :] <= random_areas[:, 1, :])
        )

    def test_residuals_shape(self):
        residuals = self.runner.check_sample(self.data_set)
        self.assertEqual(residuals.shape, torch.Size([self.set_count]))


class TestEmpiricalCoverage2d(unittest.TestCase):
    def setUp(self):
        self.data_set = torch.Tensor([[0.5, 0.5], [0.5, 0.5]])
        domains = torch.Tensor([[0, 1], [0, 1]])
        self.dimension = 2
        self.set_count = 200
        self.runner = EmpiricalCoverageRunner(
            self.data_set, domains, self.dimension, set_count=self.set_count
        )

    def test_random_areas_shape_2d(self):
        random_areas = self.runner._get_random_sets()
        self.assertEqual(
            random_areas.shape, (self.set_count, 2, self.dimension)
        )

    def test_random_areas_sorted(self):
        random_areas = self.runner._get_random_sets()
        self.assertTrue(
            torch.all(random_areas[:, 0, :] <= random_areas[:, 1, :])
        )

    def test_residuals_shape(self):
        residuals = self.runner.check_sample(self.data_set)
        self.assertEqual(residuals.shape, torch.Size([self.set_count]))


if __name__ == "__main__":
    unittest.main()
