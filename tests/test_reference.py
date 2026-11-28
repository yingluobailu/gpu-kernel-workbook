"""Independent hand-calculated examples. Run on a machine without a GPU."""

import unittest
from unittest.mock import Mock

import numpy as np

from workbook.cuda import Device
from workbook.judge import compare
from workbook.problems import (
    PROBLEMS, Case, compaction_counts, csr_spmv, halo_blur, hierarchical_sum, histogram_counts,
    masked_softmax, masked_top2, partial_histogram, pitched_transpose, ragged_rmsnorm,
    row_argmax, stable_compact, strided_affine, warp_prefix, warp_vote,
)


class ReferenceTests(unittest.TestCase):
    def test_unwritten_int_output_differs_from_missing_index(self):
        device = Device(0)
        device.alloc = Mock(return_value=0)
        device.clear = Mock(return_value=0)
        device.free = Mock(return_value=0)
        with device.buffer(np.empty(4, dtype=np.int32)):
            fill = device.clear.call_args.args[1]
            initial = np.frombuffer(bytes([fill]) * 4, dtype=np.int32)[0]
            self.assertNotEqual(initial, -1)

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

    def test_histogram_ignores_invalid_labels_and_preserves_extreme_bins(self):
        labels = np.array([0, 0, -1, 64, 63, -2_147_483_648], dtype=np.int32)
        expected = np.zeros(64, dtype=np.int32)
        expected[0], expected[63] = 2, 1
        np.testing.assert_array_equal(histogram_counts(labels), expected)

    def test_tiled_histogram_layout_keeps_the_last_partial(self):
        labels = np.array([7] * 1024 + [11, 7, 11], dtype=np.int32)
        expected = np.zeros((64, 2), dtype=np.int32)
        expected[7] = [1024, 1]
        expected[11] = [0, 2]
        np.testing.assert_array_equal(partial_histogram(labels), expected)

    def test_rmsnorm_ignores_nan_padding_and_empty_rows(self):
        x = np.array([[1, 2, np.nan], [np.nan] * 3], dtype=np.float32)
        residual = np.array([[2, 2, np.nan], [np.nan] * 3], dtype=np.float32)
        weights = np.array([1, 2, 9], dtype=np.float32)
        result = ragged_rmsnorm(x, residual, weights, np.array([2, 0], dtype=np.int32))
        np.testing.assert_allclose(result[0], np.array([3, 8, 0]) / np.sqrt(12.5 + 1e-5), atol=1e-7)
        np.testing.assert_array_equal(result[1], [0, 0, 0])

    def test_masked_top2_returns_distinct_stable_columns_and_missing_values(self):
        scores = np.array([[5, 9, 5, 5], [-2, 1, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        mask = np.array([[1, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        values, indices = masked_top2(scores, mask)
        np.testing.assert_array_equal(values, [[5, 5], [-2, -np.inf], [-np.inf, -np.inf]])
        np.testing.assert_array_equal(indices, [[0, 2], [0, -1], [-1, -1]])

    def test_csr_counts_duplicate_columns_and_empty_rows(self):
        inputs = (np.array([0, 3, 3, 4]), np.array([2, 0, 2, 1]),
                  np.array([2, 1, -1, 4], dtype=np.float32), np.array([3, 5, 7], dtype=np.float32))
        np.testing.assert_array_equal(csr_spmv(*inputs), [10, 0, 20])

    def test_compaction_preserves_original_indices_and_fills_tail(self):
        source = np.array([-2, 7, 0, 7, -1, 3], dtype=np.int32)
        values, indices, total = stable_compact(source)
        np.testing.assert_array_equal(values, [7, 7, 3, 0, 0, 0])
        np.testing.assert_array_equal(indices, [1, 3, 5, -1, -1, -1])
        np.testing.assert_array_equal(total, [3])
        np.testing.assert_array_equal(compaction_counts(np.array([0] * 255 + [1, 1, 1], dtype=np.int32)), [1, 2])

    def test_judge_validates_output_count_and_dtype(self):
        problem = PROBLEMS["06"]
        case = next(case for case in problem.cases() if case.dims == (1, 1))
        values, indices = case.expected
        with self.assertRaisesRegex(AssertionError, "outputs"):
            compare(problem, case, (values,))
        with self.assertRaisesRegex(AssertionError, "dtype|int32"):
            compare(problem, case, (values, indices.astype(np.int64)))

    def test_judge_rejects_nonzero_mask_even_inside_numeric_tolerance(self):
        source, lengths = np.array([[0, np.nan]], dtype=np.float32), np.array([1], dtype=np.int32)
        expected = np.array([[1, 0]], dtype=np.float32)
        almost_correct = np.array([[1, 1e-10]], dtype=np.float32)
        with self.assertRaises(AssertionError):
            compare(PROBLEMS["08"], Case((1, 2), (source, lengths), (expected,)), (almost_correct,))


if __name__ == "__main__":
    unittest.main()
