import unittest
import numpy as np
from deepred.ce import Operator, analysis, synthesis, solve_red


class OperatorsAndRED(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)

    def complex(self, shape):
        return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)

    def test_adjoint(self):
        op = Operator(self.rng.integers(0, 2, (256, 32)))
        x, y = self.complex((32, 32)), self.complex((256, 32))
        np.testing.assert_allclose(np.vdot(op.forward(x), y), np.vdot(x, op.adjoint(y)), rtol=1e-12, atol=1e-12)

    def test_roundtrip_and_norm(self):
        x = self.complex((32, 32))
        np.testing.assert_allclose(analysis(synthesis(x)), x, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(synthesis(x))**2/np.linalg.norm(x)**2, 8)

    def test_data_gradient(self):
        op = Operator(self.rng.integers(0, 2, (256, 32)))
        x, v = self.complex((32, 32)), self.complex((32, 32))
        b = self.complex((256, 32)); eps = 1e-5
        f = lambda z: np.linalg.norm(op.forward(z)-b)**2/2
        numerical = (f(x+eps*v)-f(x-eps*v))/(2*eps)
        expected = np.vdot(v, op.adjoint(op.forward(x)-b)).real
        self.assertAlmostEqual(numerical, expected, places=5)

    def test_linear_denoiser_known_solution(self):
        op = Operator(np.ones((256, 32)))
        truth_x = self.complex((32, 32)); truth = synthesis(truth_x)
        result, x = solve_red(op, truth, np.zeros_like(truth_x), lambda x,s: .5*x,
                              truth, lam=.2, step=.5, iterations=100, tolerance=1e-9)
        np.testing.assert_allclose(x, truth_x/1.1, atol=1e-8)
        self.assertEqual(result['status'], 'converged')
        self.assertEqual(result['denoiser_calls'], len(result['history']))

    def test_instability_and_parameters(self):
        op = Operator(np.ones((256, 32))); x = np.ones((32, 32), complex)
        result, _ = solve_red(op, synthesis(x), x*0, lambda z,s: 0*z,
                             synthesis(x), step=10, iterations=20, growth_limit=100)
        self.assertEqual(result['status'], 'unstable')
        with self.assertRaises(ValueError):
            solve_red(op, synthesis(x), x, lambda z,s:z, synthesis(x), step=-1)

    def test_nonfinite_denoiser_is_reportable(self):
        import json
        op = Operator(np.ones((256, 32))); x = np.ones((32,32),complex)
        result, _ = solve_red(op, synthesis(x), x,
                             lambda z,s: np.full_like(z, np.nan), synthesis(x))
        self.assertEqual(result['status'], 'unstable')
        json.dumps(result, allow_nan=False)

    def test_zero_mask_identity_equilibrium(self):
        op = Operator(np.zeros((256,32))); x = self.complex((32,32))
        result, actual = solve_red(op, np.zeros((256,32)), x, lambda z,s:z, synthesis(x))
        self.assertEqual(result['status'], 'converged')
        self.assertEqual(result['denoiser_calls'], 1)
        np.testing.assert_array_equal(actual, x)


if __name__ == '__main__':
    unittest.main()
