import unittest

import numpy as np

from workbook.ordering import merge_order, segmented_sum, sort_case


class OrderingTests(unittest.TestCase):
    def test_permutation_uses_signed_key_then_original_column(self):
        (keys,), (order,) = sort_case(np.random.default_rng(15), 3, 129)
        for row in range(len(keys)):
            expected = sorted(range(129), key=lambda i: (int(keys[row, i]), i))
            np.testing.assert_array_equal(order[row], expected)

    def test_equal_merge_keys_retain_source_priority(self):
        a = np.array([-2, 1, 1, 9], dtype=np.int32)
        b = np.array([-2, 0, 1], dtype=np.int32)
        values, origin = merge_order(a, b)
        np.testing.assert_array_equal(values, [-2, -2, 0, 1, 1, 1, 9])
        np.testing.assert_array_equal(origin, [0, 4, 5, 1, 2, 6, 3])

    def test_carry_crosses_tile_then_resets(self):
        values = np.ones((2, 131), dtype=np.int32)
        heads = np.zeros_like(values)
        heads[0, 129] = 1
        heads[1, 127:129] = 1
        sums = segmented_sum(values, heads)
        np.testing.assert_array_equal(sums[:, 127:], [[128, 129, 1, 2], [1, 1, 2, 3]])
