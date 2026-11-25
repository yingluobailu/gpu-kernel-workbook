"""Independent checks for dense, quantized and streaming inference exercises."""

import unittest

import numpy as np

from workbook.judge import compare
from workbook.problems import PROBLEMS, Case, biased_relu_gemm, decode_attention, int4_case


class InferenceTests(unittest.TestCase):
    def test_bias_is_per_column_and_relu_follows_dot(self):
        a = np.array([[1, -2], [-1, 2]], dtype=np.float32)
        b = np.array([[2, 1, 0], [1, -1, 2]], dtype=np.float32)
        bias = np.array([-1, -2, 3], dtype=np.float32)
        np.testing.assert_array_equal(biased_relu_gemm(a, b, bias), [[0, 1, 0], [0, 0, 7]])

    def test_zero_inner_dimension_preserves_positive_bias(self):
        a = np.empty((2, 0), np.float32)
        b = np.empty((0, 2), np.float32)
        np.testing.assert_array_equal(biased_relu_gemm(a, b, [-1, 3]), [[0, 3], [0, 3]])

    def test_wire_format_matches_dense_int4_oracle(self):
        width, group_size = 19, 3
        (packed, scales, zeros, x), (expected,) = int4_case(np.random.default_rng(21), 5, width, group_size)
        for row in range(5):
            decoded = []
            for col in range(width):
                word = int(packed[row, col // 8]) & 0xffffffff
                nibble = (word >> (4 * (col % 8))) & 15
                group = col // group_size
                decoded.append((nibble - int(zeros[row, group])) * float(scales[row, group]))
            self.assertEqual(int(packed[row, -1]) >> 12, -1)  # All unused nibbles contain ones.
            self.assertEqual(sum(w * float(v) for w, v in zip(decoded, x)), float(expected[row]))

    def test_decode_averages_only_live_tokens(self):
        q = np.zeros((2, 32), dtype=np.float32)
        keys = np.full((2, 3, 32), np.nan, dtype=np.float32)
        values = np.full_like(keys, np.nan)
        keys[1, :2] = 0
        values[1, 0], values[1, 1] = -1, 5
        lengths = np.array([0, 2], dtype=np.int32)
        expected = np.stack([np.zeros(32), np.full(32, 2)]).astype(np.float32)
        np.testing.assert_array_equal(decode_attention(q, keys, values, lengths), expected)
        case = Case((2, 3), (q, keys, values, lengths), (expected,))
        wrong = expected.copy()
        wrong[0, 0] = 1e-8
        with self.assertRaises(AssertionError):
            compare(PROBLEMS["16"], case, (wrong,))
