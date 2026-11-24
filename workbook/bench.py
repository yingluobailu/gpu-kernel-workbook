"""Compare kernel implementations on controlled, reproducible GPU workloads."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import sys

import numpy as np

from workbook.cuda import Device, GpuError
from workbook.judge import CudaKernel, TritonKernel, compare, run_case
from workbook.problems import Case, PROBLEMS, fixed, masked_softmax
from workbook.timing import timed_graph


# Each family keeps the logical output size fixed while varying one aspect of layout.
EXPERIMENTS = {
    "stride": ("01", ((1048576, 1, 0), (1048576, 4, 0), (1048576, 8, 0))),
    "pitch": ("02", ((1024, 2048, 2048), (1024, 2048, 2049), (1024, 2048, 2080))),
    "stencil": ("05", ((256, 8192), (1024, 2048), (8192, 256))),
    "softmax": ("08", ((8192, 256), (2048, 1024), (512, 4096))),
}
SEED = 343


def make_case(problem, dims):
    rng = np.random.default_rng(SEED)
    if problem.key == "08":
        rows, cols = dims
        values = fixed(rng, dims)
        lengths = np.full(rows, cols, dtype=np.int32)
        return Case(dims, (values, lengths), (masked_softmax(values, lengths),))
    inputs, expected = problem.case_factory(rng, *dims)
    return Case(dims, inputs, expected)


def measure(device, kernel, problem, case, warmup, samples):
    outputs = tuple(np.empty_like(array) for array in case.expected)
    arrays = case.inputs + outputs
    grid, constants = problem.launch_config(case.dims)
    with ExitStack() as resources:
        pointers = [resources.enter_context(device.buffer(a, upload=i < len(case.inputs)))
                    for i, a in enumerate(arrays)]
        launch = kernel.prepare(resources, grid, pointers, case.dims, constants, arrays)
        device.synchronize()  # The benchmark stream is independent from the upload stream.
        sample_us = resources.enter_context(timed_graph(device, launch))

        def verify():
            for pointer, array in zip(pointers[len(case.inputs):], outputs):
                device.read(pointer, array)
            compare(problem, case, outputs)

        sample_us()
        verify()
        for _ in range(warmup):
            sample_us()
        times = [sample_us() for _ in range(samples)]
        verify()
    return times


def statistics(times, elements):
    if not times or min(times) <= 0 or not np.all(np.isfinite(times)):
        raise GpuError("Invalid event timestamps. Try a larger workload or inspect the GPU driver.")
    p10, median, p90 = np.percentile(times, [10, 50, 90])
    return {"p10_us": float(p10), "median_us": float(median), "p90_us": float(p90),
            "g_elements_s": elements / (median * 1000)}


def run(device, args):
    results = []
    names = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    backends = ("cuda", "triton") if args.backend == "both" else (args.backend,)
    for name in names:
        key, shapes = EXPERIMENTS[name]
        problem = PROBLEMS[key]
        kernels = {}
        for backend in backends:
            compiler, source = ((CudaKernel, "cuda.cu") if backend == "cuda"
                                else (TritonKernel, "triton_answer.py"))
            kernel = compiler(device, problem, args.submission or problem.path / source)
            if args.submission:
                for case in problem.cases():
                    compare(problem, case, run_case(device, kernel, problem, case))
            kernels[backend] = kernel
        for dims in shapes:
            case = make_case(problem, dims)
            elements = case.expected[0].size
            for backend, kernel in kernels.items():
                times = measure(device, kernel, problem, case, args.warmup, args.samples)
                results.append({"experiment": name, "problem": key, "backend": backend,
                                "dims": dims, "elements": elements, "samples_us": times,
                                **statistics(times, elements)})
    return {"schema_version": 1, "gpu": device.name, "sm": device.capability,
            "platform": platform.system(), "python": platform.python_version(),
            "warmup": args.warmup, "samples": args.samples, "seed": SEED,
            "implementation": args.submission.name if args.submission else "reference",
            "timing": "cuda_graph_external_events", "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("all", *EXPERIMENTS))
    parser.add_argument("--backend", choices=("cuda", "triton", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--submission", type=Path, help="trusted source for one experiment and backend")
    parser.add_argument("--json", action="store_true", help="write the full report to stdout")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be nonnegative and --samples must be positive.")
    if args.submission and (args.experiment == "all" or args.backend == "both"):
        parser.error("--submission requires one experiment and one backend.")
    try:
        with Device(args.device) as device:
            report = run(device, args)
    except AssertionError as error:
        print(f"Benchmark correctness check failed: {error}", file=sys.stderr)
        return 1
    except (GpuError, OSError, AttributeError) as error:
        print(f"GPU benchmark: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, allow_nan=False))
        return 0
    print(f"{report['gpu']} (sm_{report['sm']}) | {args.samples} samples per shape")
    print("experiment backend dimensions               p10 us  median us   p90 us   Gelem/s")
    for row in report["results"]:
        dims = "x".join(map(str, row["dims"]))
        print(f"{row['experiment']:<10} {row['backend']:<7} {dims:<23} {row['p10_us']:7.2f}"
              f" {row['median_us']:10.2f} {row['p90_us']:8.2f} {row['g_elements_s']:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
