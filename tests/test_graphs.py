import unittest

import numpy as np

from workbook.graphs import csr, locate_case, pull_parents, triangle_counts


class GraphTests(unittest.TestCase):
    def test_positions_reconstruct_coordinates_across_empty_rows(self):
        (offsets, positions), (owners, local) = locate_case(np.random.default_rng(42), 33, 129)
        self.assertTrue(np.any(np.diff(offsets) == 0))
        for pos, row, within in zip(positions, owners, local):
            if 0 <= pos < offsets[-1]:
                self.assertTrue(offsets[row] <= pos < offsets[row + 1])
                self.assertEqual(offsets[row] + within, pos)
            else:
                self.assertEqual((row, within), (-1, -1))

    def test_triangle_assignments_on_clique_and_open_wedge(self):
        graph = [[1, 2, 3], [2, 3], [3, 4], [], []]
        offsets, edges = csr(graph)
        np.testing.assert_array_equal(offsets, [0, 3, 5, 7, 7, 7])
        np.testing.assert_array_equal(edges, [1, 2, 3, 2, 3, 3, 4])
        np.testing.assert_array_equal(triangle_counts(graph), [3, 1, 0, 0, 0])

    def test_frontier_chooses_smallest_and_advances_once(self):
        incoming = [[1], [], [1, 1, 0], [2], [0], [5, 4]]
        state = np.array([1, 1, 0, 0, 2, 0], dtype=np.int32)
        np.testing.assert_array_equal(pull_parents(incoming, state), [-1, -1, 0, -1, -1, -1])
