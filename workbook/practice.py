"""Create learner-owned starter files from the workbook's public kernel signatures."""

import ast
import os
from pathlib import Path
import re
import shlex


CUDA_ENTRY = re.compile(r'^extern "C" __global__ void (\w+)\s*\([^()]*\)(?=\s*\{)', re.MULTILINE)


def command_line(arguments):
    """Quote paths for the host shell (PowerShell on Windows)."""
    if os.name == "nt":
        return " ".join(arg if re.fullmatch(r"[\w./:\\-]+", arg) else "'" + arg.replace("'", "''") + "'"
                        for arg in arguments)
    return shlex.join(arguments)


def starter_text(problem, backend):
    source = (problem.path / "cuda.cu").read_text(encoding="utf-8")
    entries = list(CUDA_ENTRY.finditer(source))
    if not entries or entries[0][1] != "solve":
        raise ValueError(f"No solve entry point in {problem.path.name}")
    if backend == "cuda":
        return "\n\n".join(entry[0] + " {\n    // TODO: implement this stage.\n}" for entry in entries) + "\n"
    module = ast.parse((problem.path / "triton_answer.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    kernels = []
    for entry in entries:
        name = entry[1]
        args = ast.unparse(functions[name].args)
        kernels.append(f"@triton.jit\ndef {name}({args}):\n    # TODO: implement this stage.\n    pass")
    return "import triton\nimport triton.language as tl\n\n\n" + "\n\n\n".join(kernels) + "\n"


def start(problem, backend, directory):
    extension = "cu" if backend == "cuda" else "py"
    destination = Path(directory) / f"{problem.key}-{problem.slug}.{extension}"
    template = starter_text(problem, backend)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        file.write(template)
    return destination
