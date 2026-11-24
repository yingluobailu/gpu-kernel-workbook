"""Performance experiment contracts that run in CPU-only CI."""

import unittest

import numpy as np

from workbook.bench import EXPERIMENTS, make_case, statistics
from workbook.cuda import GpuError
from workbook.problems import PROBLEMS


class ExperimentTests(unittest.TestCase):
    def test_dimensions_hold_output_size_constant(self):
        for name, (_, dims) in EXPERIMENTS.items():
            with self.subTest(name=name):
                sizes = {shape[0] if name == "stride" else shape[0] * shape[1] for shape in dims}
                self.assertEqual(len(sizes), 1)
                self.assertEqual(len(set(dims)), 3)

    def test_softmax_uses_full_rows_and_repeats_inputs(self):
        p = PROBLEMS["08"]
        a, b = make_case(p, (3, 8)), make_case(p, (3, 8))
        np.testing.assert_array_equal(a.inputs[0], b.inputs[0])
        np.testing.assert_array_equal(a.inputs[1], [8, 8, 8])
        self.assertTrue(np.isfinite(a.inputs[0]).all())
        np.testing.assert_allclose(a.expected[0].sum(axis=1), 1, rtol=1e-6)

    def test_time_units_and_quantiles(self):
        result = statistics([60, 20, 40, 30, 50], 2_000_000)
        self.assertEqual(result["median_us"], 40)
        self.assertAlmostEqual(result["p10_us"], 24)
        self.assertAlmostEqual(result["p90_us"], 56)
        self.assertEqual(result["g_elements_s"], 50)

    def test_single_invalid_sample_rejects_report(self):
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(GpuError):
                statistics([10] * 20 + [invalid], 100)
