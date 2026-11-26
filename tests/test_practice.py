import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workbook.practice import CUDA_ENTRY, start, starter_text
from workbook.problems import PROBLEMS


class PracticeTests(unittest.TestCase):
    def test_starters_cover_all_public_entries_and_have_empty_bodies(self):
        for key, problem in PROBLEMS.items():
            with self.subTest(problem=key):
                names = ["solve"]
                if key == "09":
                    names += ["merge"]
                if key == "13":
                    names += ["scan", "scatter"]
                cuda = starter_text(problem, "cuda")
                self.assertEqual([m[1] for m in CUDA_ENTRY.finditer(cuda)], names)
                self.assertEqual(cuda.count("// TODO:"), len(names))
                tree = ast.parse(starter_text(problem, "triton"))
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                self.assertEqual([f.name for f in functions], names)
                self.assertTrue(all(len(f.body) == 1 and isinstance(f.body[0], ast.Pass) for f in functions))

    def test_existing_submission_survives_repeated_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "my answers"
            path = start(PROBLEMS["13"], "cuda", directory)
            path.write_text("my unfinished work", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                start(PROBLEMS["13"], "cuda", directory)
            self.assertEqual(path.read_text(encoding="utf-8"), "my unfinished work")

    def test_direct_selection_preserves_original_samples(self):
        p = PROBLEMS["01"]
        for number, dims in enumerate(p.shapes, 1):
            inputs, expected = p.case_factory(np.random.default_rng(343 + number - 1), *dims)
            selected = p.case(number)
            for actual, old in zip(selected.inputs + selected.expected, inputs + expected):
                np.testing.assert_array_equal(actual, old)
        alternate = p.case(4, 99)
        self.assertFalse(np.array_equal(p.case(4).inputs[0], alternate.inputs[0]))
        np.testing.assert_array_equal(alternate.inputs[0], p.case(4, 99).inputs[0])
        for number in (0, len(p.shapes) + 1):
            with self.assertRaises(ValueError):
                p.case(number)
