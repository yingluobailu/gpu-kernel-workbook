"""Minimal driver and NVRTC adapter shared by the two GPU backends."""

from contextlib import ExitStack, contextmanager
import ctypes as c
from ctypes.util import find_library
import os
from pathlib import Path

Handle = c.c_void_p
Ptr = c.c_uint64
Int = c.c_int
UInt = c.c_uint
Size = c.c_size_t
Str = c.c_char_p
Ref = c.POINTER


class GpuError(RuntimeError):
    pass


def bind(library, name, *args, result=Int):
    function = getattr(library, name)
    function.argtypes = list(args)
    function.restype = result
    return function


class Device:
    """Own a reference to one CUDA primary context and its loaded modules."""

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.stack = ExitStack()

    def __enter__(self):
        try:
            if os.name == "nt":
                self.lib = c.WinDLL("nvcuda.dll")
            else:
                self.lib = c.CDLL("libcuda.so.1")
            library = self.lib
            self.error_name = bind(library, "cuGetErrorName", Int, Ref(Str))
            self.error_text = bind(library, "cuGetErrorString", Int, Ref(Str))
            self.check(bind(library, "cuInit", UInt)(0), "Initialize CUDA")
            device = Int()
            self.check(bind(library, "cuDeviceGet", Ref(Int), Int)(c.byref(device), self.ordinal), "Select CUDA device")
            self.handle = device.value
            name = c.create_string_buffer(128)
            self.check(bind(library, "cuDeviceGetName", Handle, Int, Int)(name, len(name), device), "Read device name")
            self.name = name.value.decode(errors="replace")
            attr = bind(library, "cuDeviceGetAttribute", Ref(Int), Int, Int)
            major, minor = Int(), Int()
            self.check(attr(c.byref(major), 75, device), "Read compute capability major")
            self.check(attr(c.byref(minor), 76, device), "Read compute capability minor")
            self.capability = major.value * 10 + minor.value
            context = Handle()
            self.check(bind(library, "cuDevicePrimaryCtxRetain", Ref(Handle), Int)(c.byref(context), device),
                       "Retain CUDA context")
            release = bind(library, "cuDevicePrimaryCtxRelease_v2", Int)
            self.stack.callback(release, device)
            set_context = bind(library, "cuCtxSetCurrent", Handle)
            previous = Handle()
            self.check(bind(library, "cuCtxGetCurrent", Ref(Handle))(c.byref(previous)), "Read previous context")
            self.check(set_context(context), "Activate CUDA context")
            self.stack.callback(set_context, previous)

            self.module_load = bind(library, "cuModuleLoadData", Ref(Handle), Handle)
            self.module_unload = bind(library, "cuModuleUnload", Handle)
            self.get_function = bind(library, "cuModuleGetFunction", Ref(Handle), Handle, Str)
            self.alloc = bind(library, "cuMemAlloc_v2", Ref(Ptr), Size)
            self.free = bind(library, "cuMemFree_v2", Ptr)
            self.upload = bind(library, "cuMemcpyHtoD_v2", Ptr, Handle, Size)
            self.download = bind(library, "cuMemcpyDtoH_v2", Handle, Ptr, Size)
            self.clear = bind(library, "cuMemsetD8_v2", Ptr, c.c_ubyte, Size)
            self.run = bind(library, "cuLaunchKernel", Handle, UInt, UInt, UInt, UInt, UInt, UInt,
                            UInt, Handle, Ref(Handle), Ref(Handle))
            self.wait = bind(library, "cuCtxSynchronize")
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self.stack.close()

    def check(self, code, operation):
        if code:
            name, description = Str(), Str()
            self.error_name(code, c.byref(name))
            self.error_text(code, c.byref(description))
            error = name.value or b"unknown CUDA error"
            detail = description.value or b"no description"
            raise GpuError(f"{operation}: {error.decode()} ({code}): {detail.decode()}")

    def function(self, cubin, name):
        module, function = Handle(), Handle()
        image = c.create_string_buffer(cubin)
        self.check(self.module_load(c.byref(module), image), "Load GPU binary")
        self.stack.callback(self.module_unload, module)
        self.check(self.get_function(c.byref(function), module, name.encode()), f"Find kernel {name}")
        return function

    @contextmanager
    def buffer(self, array, *, upload=False):
        pointer = Ptr()
        self.check(self.alloc(c.byref(pointer), max(1, array.nbytes)), "Allocate device buffer")
        try:
            if array.nbytes:
                if upload:
                    self.check(self.upload(pointer, array.ctypes.data, array.nbytes), "Upload input")
                else:
                    self.check(self.clear(pointer, 0xFF, array.nbytes), "Poison output")
            yield pointer
        finally:
            self.free(pointer)

    def read(self, pointer, output):
        if output.nbytes:
            self.check(self.download(output.ctypes.data, pointer, output.nbytes), "Read kernel output")

    def launch(self, function, grid, block, shared, arguments):
        pointers = (Handle * len(arguments))(*(c.cast(c.byref(value), Handle) for value in arguments))
        self.check(self.run(function, *grid, *block, shared, None, pointers, None), "Launch GPU kernel")

    def synchronize(self):
        self.check(self.wait(), "Synchronize GPU kernel")


class Nvrtc:
    def __init__(self, capability):
        self.capability = capability
        toolkit = Path(os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda")
        if os.name == "nt":
            location = toolkit / "bin"
            dll = next(iter(sorted(location.glob("nvrtc64_*_0.dll"), reverse=True)), None)
            if not dll:
                raise GpuError("NVRTC not found. Install the CUDA Toolkit and set CUDA_PATH.")
            with os.add_dll_directory(str(location)):
                library = c.CDLL(str(dll))
        else:
            installed = toolkit / "lib64" / "libnvrtc.so"
            library = c.CDLL(str(installed) if installed.is_file() else find_library("nvrtc") or "libnvrtc.so")
        self.create = bind(library, "nvrtcCreateProgram", Ref(Handle), Str, Str, Int, Ref(Str), Ref(Str))
        self.compile_source = bind(library, "nvrtcCompileProgram", Handle, Int, Ref(Str))
        self.log_size = bind(library, "nvrtcGetProgramLogSize", Handle, Ref(Size))
        self.read_log = bind(library, "nvrtcGetProgramLog", Handle, Handle)
        self.cubin_size = bind(library, "nvrtcGetCUBINSize", Handle, Ref(Size))
        self.read_cubin = bind(library, "nvrtcGetCUBIN", Handle, Handle)
        self.destroy = bind(library, "nvrtcDestroyProgram", Ref(Handle))
        self.error = bind(library, "nvrtcGetErrorString", Int, result=Str)

    def check(self, code):
        if code:
            raise GpuError(f"NVRTC: {self.error(code).decode()}")

    def compile(self, path):
        program = Handle()
        self.check(self.create(c.byref(program), path.read_bytes(), path.name.encode(), 0, None, None))
        try:
            flags = (Str * 2)(f"--gpu-architecture=sm_{self.capability}".encode(), b"--std=c++14")
            status = self.compile_source(program, len(flags), flags)
            if status:
                size = Size()
                self.log_size(program, c.byref(size))
                log = c.create_string_buffer(max(1, size.value))
                self.read_log(program, log)
                raise GpuError(log.value.decode(errors="replace").strip() or self.error(status).decode())
            size = Size()
            self.check(self.cubin_size(program, c.byref(size)))
            cubin = c.create_string_buffer(size.value)
            self.check(self.read_cubin(program, cubin))
            return cubin.raw
        finally:
            self.destroy(c.byref(program))
