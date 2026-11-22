"""Independent hand-calculated examples. Run on a machine without a GPU."""

import unittest

import numpy as np

from workbook.judge import compare
from workbook.problems import (
    PROBLEMS, Case, halo_blur, hierarchical_sum, masked_softmax, pitched_transpose,
    row_argmax, strided_affine, warp_prefix, warp_vote,
)


class ReferenceTests(unittest.TestCase):
    def test_strided_affine_skips_unselected_values(self):
        values = np.array([99, 2, 99, 5, 99, -1], dtype=np.float32)
        np.testing.assert_array_equal(strided_affine(values, 3, 2, 1), [1, 7, -5])

    def test_pitched_transpose_ignores_padding(self):
        values = np.array([[1, 2, 3, np.nan], [4, 5, 6, np.nan]], dtype=np.float32)
        np.testing.assert_array_equal(pitched_transpose(values, 3), [[1, 4], [2, 5], [3, 6]])

    def test_votes_use_strictly_positive_and_keep_last_group(self):
        values = np.array([1] * 31 + [0, -1, 2, 4], dtype=np.int32)
        np.testing.assert_array_equal(warp_vote(values), [31, 2])

    def test_exclusive_scan_resets_on_second_group(self):
        values = np.array([1] * 32 + [3, 1, 4, 2], dtype=np.int32)
        np.testing.assert_array_equal(warp_prefix(values), [*range(32), 0, 3, 4, 8])

    def test_blur_zero_pads_a_thin_matrix(self):
        values = np.array([[4, 8]], dtype=np.float32)
        np.testing.assert_array_equal(halo_blur(values), [[2, 2.5]])

    def test_argmax_is_stable_across_ties(self):
        values = np.array([[-8, -2, -2], [-5, -5, -5]], dtype=np.float32)
        maximums, first = row_argmax(values)
        np.testing.assert_array_equal(maximums, [-2, -5])
        np.testing.assert_array_equal(first, [1, 0])

    def test_full_sum_handles_empty_and_two_initial_tiles(self):
        np.testing.assert_array_equal(hierarchical_sum(np.ones(513, dtype=np.float32)), [513])
        np.testing.assert_array_equal(hierarchical_sum(np.array([], dtype=np.float32)), [0])

    def test_softmax_stable_positive_logits_and_empty_prefix(self):
        values = np.array([[1000, 1000, np.nan], [np.nan] * 3], dtype=np.float32)
        lengths = np.array([2, 0], dtype=np.int32)
        np.testing.assert_array_equal(masked_softmax(values, lengths), [[.5, .5, 0], [0, 0, 0]])

    def test_judge_rejects_nonzero_mask_even_inside_numeric_tolerance(self):
        source, lengths = np.array([[0, np.nan]], dtype=np.float32), np.array([1], dtype=np.int32)
        expected = np.array([[1, 0]], dtype=np.float32)
        almost_correct = np.array([[1, 1e-10]], dtype=np.float32)
        with self.assertRaises(AssertionError):
            compare(PROBLEMS["08"], Case((1, 2), (source, lengths), (expected,)), (almost_correct,))


if __name__ == "__main__":
    unittest.main()
