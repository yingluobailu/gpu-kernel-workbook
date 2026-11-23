"""Compile and run reference kernels or learner submissions against the same cases."""

import argparse
from contextlib import ExitStack
import importlib.util
from math import prod
import os
from pathlib import Path
import sys

import numpy as np

from workbook.cuda import Device, GpuError, Int, Nvrtc, Ptr
from workbook.problems import HISTOGRAM_BINS, PROBLEMS, ROOT, partial_histogram


TRITON_TYPES = {np.dtype("float32"): "*fp32", np.dtype("int32"): "*i32"}


class CudaKernel:
    def __init__(self, device, problem, path):
        self.device = device
        self.block = problem.cuda_block
        self.module = device.load(Nvrtc(device.capability).compile(path))
        self.functions = {}

    def launch(self, grid, pointers, dims, constants, arrays, *, entry="solve"):
        if entry not in self.functions:
            self.functions[entry] = self.device.function(self.module, entry)
        self.device.launch(self.functions[entry], grid, self.block, 0, [*pointers, *map(Int, dims)])


class TritonKernel:
    def __init__(self, device, problem, path):
        try:
            import triton
            from triton.backends.compiler import GPUTarget
            from triton.compiler import ASTSource
        except ImportError as error:
            raise GpuError("Install the Triton dependencies from requirements-triton.txt.") from error
        if triton.__version__.split(".")[:2] != ["3", "3"]:
            raise GpuError("The offline launcher uses the Triton 3.3 compiler ABI. Install requirements-triton.txt.")
        if device.capability < 80:
            raise GpuError("The Triton track requires an NVIDIA GPU with compute capability 8.0 or newer.")
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton-cache"))
        spec = importlib.util.spec_from_file_location(f"submission_{problem.key}", path)
        if spec is None or spec.loader is None:
            raise GpuError("The Triton submission must be a Python source file.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.submission = module
        self.compile = triton.compile
        self.source = ASTSource
        self.target = GPUTarget("cuda", device.capability, 32)
        self.device = device
        self.warps = problem.triton_warps
        self.binaries = {}

    def launch(self, grid, pointers, dims, constants, arrays, *, entry="solve"):
        types = tuple(TRITON_TYPES[array.dtype] for array in arrays) + ("i32",) * len(dims)
        key = entry, types, tuple(constants.items())
        if key not in self.binaries:
            kernel = getattr(self.submission, entry)
            parameters = (*types, *("constexpr",) * len(constants))
            if len(parameters) != len(kernel.arg_names):
                raise GpuError(f"Triton {entry}() does not match the problem signature.")
            signature = dict(zip(kernel.arg_names, parameters))
            source = self.source(kernel, signature, constexprs=constants)
            try:
                binary = self.compile(source, target=self.target, options={"num_warps": self.warps})
            except Exception as error:
                raise GpuError(f"Triton compilation failed: {error}") from error
            module = self.device.load(binary.asm["cubin"])
            function = self.device.function(module, binary.name)
            self.binaries[key] = function, binary.metadata
        function, metadata = self.binaries[key]
        with ExitStack() as scratch:
            memory = Ptr(0)
            if metadata.global_scratch_size:
                count = metadata.global_scratch_size * prod(grid)
                memory = scratch.enter_context(self.device.buffer(np.empty(count, dtype=np.uint8)))
            # The NVIDIA Triton 3.3 launcher appends an optional global-scratch pointer.
            self.device.launch(function, grid, (metadata.num_warps * 32, 1, 1), metadata.shared,
                               [*pointers, *map(Int, dims), memory])
            if metadata.global_scratch_size:
                self.device.synchronize()


def run_case(device, kernel, problem, case):
    if problem.key == "07":
        return (run_sum(device, kernel, problem, case.inputs[0]),)
    if problem.key == "09":
        return (run_histogram(device, kernel, problem, case.inputs[0]),)
    outputs = tuple(np.empty_like(array) for array in case.expected)
    grid, constants = problem.launch_config(case.dims)
    if not prod(grid):
        return outputs
    all_arrays = case.inputs + outputs
    with ExitStack() as resources:
        pointers = [
            resources.enter_context(device.buffer(array, upload=index < len(case.inputs)))
            for index, array in enumerate(all_arrays)
        ]
        kernel.launch(grid, pointers, case.dims, constants, all_arrays)
        device.synchronize()
        for output, pointer in zip(outputs, pointers[len(case.inputs):]):
            device.read(pointer, output)
    return outputs


def run_sum(device, kernel, problem, source):
    if not source.size:
        return np.zeros(1, dtype=np.float32)
    with ExitStack() as resources:
        previous = resources.enter_context(device.buffer(source, upload=True))
        reference = source
        while True:
            dims = (source.size,)
            grid, constants = problem.launch_config(dims)
            output = np.empty(grid[0], dtype=np.float32)
            current = resources.enter_context(device.buffer(output))
            kernel.launch(grid, [previous, current], dims, constants, (source, output))
            device.synchronize()
            device.read(current, output)
            # Check the per-block contract. A correct final sum can hide misplaced partials.
            reference = np.add.reduceat(reference.astype(np.float64), np.arange(0, source.size, 512)).astype(np.float32)
            np.testing.assert_allclose(output, reference, atol=1e-4, rtol=1e-6)
            if output.size == 1:
                return output
            previous, source = current, output


def run_histogram(device, kernel, problem, labels):
    if not labels.size:
        return np.zeros(HISTOGRAM_BINS, dtype=np.int32)
    grid, constants = problem.launch_config((labels.size,))
    tiles = grid[0]
    partials = np.empty((HISTOGRAM_BINS, tiles), dtype=np.int32)
    histogram = np.empty(HISTOGRAM_BINS, dtype=np.int32)
    with (
        device.buffer(labels, upload=True) as source,
        device.buffer(partials) as intermediate,
        device.buffer(histogram) as output,
    ):
        kernel.launch(grid, [source, intermediate], (labels.size, tiles), constants, (labels, partials))
        kernel.launch((HISTOGRAM_BINS, 1, 1), [intermediate, output], (tiles,),
                      {"TILE_COUNT": 1 << (tiles - 1).bit_length()}, (partials, histogram), entry="merge")
        device.synchronize()
        device.read(intermediate, partials)
        device.read(output, histogram)
    np.testing.assert_array_equal(partials, partial_histogram(labels), err_msg="Incorrect per-tile histogram")
    return histogram


def compare(problem, case, actual):
    for result, expected in zip(actual, case.expected):
        if problem.key == "08":
            # Masked outputs are part of the exact contract, not a tolerance test.
            lengths = case.inputs[1]
            inactive = np.arange(result.shape[1])[None, :] >= lengths[:, None]
            np.testing.assert_array_equal(result[inactive], 0)
            np.testing.assert_allclose(result, expected, atol=1e-6, rtol=4e-5)
        elif problem.key == "07":
            np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-6)
        else:
            np.testing.assert_array_equal(result, expected)


def check(device, problems, backends, submission=None):
    total = failures = 0
    print(f"GPU: {device.name} (sm_{device.capability})", flush=True)
    for problem in problems:
        for backend in backends:
            filename, compiler = (
                ("cuda.cu", CudaKernel) if backend == "cuda" else ("triton_answer.py", TritonKernel)
            )
            kernel = compiler(device, problem, submission or problem.path / filename)
            count = errors = 0
            for case in problem.cases():
                count += 1
                try:
                    compare(problem, case, run_case(device, kernel, problem, case))
                except AssertionError as error:
                    errors += 1
                    print(f"FAIL {problem.key} {backend} [{case.label}]\n{error}", file=sys.stderr)
            total += count
            failures += errors
            print(f"{problem.key} {backend:<6} {count - errors:2}/{count:<2}  {problem.title}", flush=True)
    print(f"{total - failures}/{total} GPU checks passed.")
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show all exercises")
    test = commands.add_parser("check", help="run cases on a real GPU")
    test.add_argument("problem", choices=("all", *PROBLEMS))
    test.add_argument("--backend", choices=("cuda", "triton", "both"), default="both")
    test.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    test.add_argument("--submission", type=Path, help="a trusted CUDA or Triton source file")
    args = parser.parse_args()
    if args.command == "list":
        for problem in PROBLEMS.values():
            print(f"{problem.key}  {problem.title} ({len(problem.shapes)} cases)")
        return 0
    if args.submission and (args.problem == "all" or args.backend == "both"):
        parser.error("--submission needs one problem and either --backend cuda or --backend triton.")
    problems = list(PROBLEMS.values()) if args.problem == "all" else [PROBLEMS[args.problem]]
    backends = ("cuda", "triton") if args.backend == "both" else (args.backend,)
    try:
        with Device(args.device) as device:
            return check(device, problems, backends, args.submission)
    except (GpuError, OSError, AttributeError) as error:
        print(f"GPU judge: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
