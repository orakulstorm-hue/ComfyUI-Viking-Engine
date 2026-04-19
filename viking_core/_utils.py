# mypy: allow-untyped-defs
import ctypes
import sys
from typing import Any, Optional, Union

import torch
from torch._utils import _get_device_index as _torch_get_device_index

def _get_hip_runtime_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        lib = ctypes.CDLL(f"amdhip64_{torch.version.hip[0]}.dll")
    else:
        lib = ctypes.CDLL("libamdhip64.so")
    lib.cuGetErrorString = lib.hipGetErrorString
    lib.cuModuleLoadData = lib.hipModuleLoadData
    lib.cuModuleGetFunction = lib.hipModuleGetFunction
    lib.cuLaunchKernel = lib.hipModuleLaunchKernel
    lib.cuFuncSetAttribute = lib.hipFuncSetAttribute
    return lib

def _get_cuda_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        return ctypes.CDLL("nvcuda.dll")
    else:
        return ctypes.CDLL("libcuda.so.1")

def _get_gpu_runtime_library() -> ctypes.CDLL:
    if torch.version.hip:
        return _get_hip_runtime_library()
    else:
        return _get_cuda_library()

def _check_cuda(result: int) -> None:
    if result == 0:
        return
    
    # VIKING AUTO-RECOVERY: Если память забита, пробуем очистить и выжить
    if result in [2, 999, 700]: 
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
    err_str = ctypes.c_char_p()
    libcuda = _get_gpu_runtime_library()
    libcuda.cuGetErrorString(result, ctypes.byref(err_str))
    error_message = err_str.value.decode() if err_str.value is not None else "Unknown CUDA error"
    raise RuntimeError(f"CUDA error: {error_message}")

def _get_hiprtc_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        version_str = "".join(["0", torch.version.hip[0], "0", torch.version.hip[2]])
        lib = ctypes.CDLL(f"hiprtc{version_str}.dll")
    else:
        lib = ctypes.CDLL("libhiprtc.so")
    lib.nvrtcGetErrorString = lib.hiprtcGetErrorString
    lib.nvrtcCreateProgram = lib.hiprtcCreateProgram
    lib.nvrtcDestroyProgram = lib.hiprtcDestroyProgram
    lib.nvrtcCompileProgram = lib.hiprtcCompileProgram
    lib.nvrtcGetPTXSize = lib.hiprtcGetCodeSize
    lib.nvrtcGetPTX = lib.hiprtcGetCode
    lib.nvrtcGetProgramLogSize = lib.hiprtcGetProgramLogSize
    lib.nvrtcGetProgramLog = lib.hiprtcGetProgramLog
    lib.nvrtcAddNameExpression = lib.hiprtcAddNameExpression
    lib.nvrtcGetLoweredName = lib.hiprtcGetLoweredName
    return lib

def _get_nvrtc_library() -> ctypes.CDLL:
    major_version = int(torch.version.cuda.split(".")[0])
    if sys.platform == "win32":
        nvrtc_libs = [f"nvrtc64_{major_version}0_0.dll"]
    else:
        nvrtc_libs = [f"libnvrtc.so.{major_version}", "libnvrtc.so"]
    for lib_name in nvrtc_libs:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue
    raise OSError("Could not find any NVRTC library")

def _get_gpu_rtc_library() -> ctypes.CDLL:
    if torch.version.hip:
        return _get_hiprtc_library()
    else:
        return _get_nvrtc_library()

def _get_gpu_rtc_compatible_flags() -> list[str]:
    from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS, COMMON_NVCC_FLAGS
    nvrtc_unsupported_flags = {"--expt-relaxed-constexpr"}
    compatible_flags = [f for f in COMMON_NVCC_FLAGS if f not in nvrtc_unsupported_flags]
    if torch.version.hip:
        compatible_flags.extend(COMMON_HIPCC_FLAGS)
    return compatible_flags

def _nvrtc_compile(kernel_source, kernel_name, compute_capability=None, cuda_include_dirs=None, nvcc_options=None, auto_pch=False):
    import torch.cuda
    libnvrtc = _get_gpu_rtc_library()
    source_bytes = kernel_source.encode("utf-8")
    if compute_capability is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        compute_capability = props.gcnArchName if torch.version.hip else f"{props.major}{props.minor}"
    
    options = []
    options.append(f"--offload-arch={compute_capability}".encode() if torch.version.hip else f"--gpu-architecture=sm_{compute_capability}".encode())
    
    from torch.utils.cpp_extension import include_paths
    for path in include_paths("cuda"):
        options.append(f"-I{path}".encode())

    if auto_pch and str(torch.version.cuda) >= "12.8":
        if nvcc_options is None: nvcc_options = []
        nvcc_options.append("--pch")

    if nvcc_options:
        for opt in nvcc_options: options.append(opt.encode("utf-8"))

    options.extend([f.encode("utf-8") for f in _get_gpu_rtc_compatible_flags()])
    prog = ctypes.c_void_p()
    libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), source_bytes, f"{kernel_name}.cu".encode(), 0, None, None)
    libnvrtc.nvrtcAddNameExpression(prog, kernel_name.encode("utf-8"))
    libnvrtc.nvrtcCompileProgram(prog, len(options), (ctypes.c_char_p * len(options))(*options))
    
    ptx_size = ctypes.c_size_t()
    libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    libnvrtc.nvrtcGetPTX(prog, ptx)
    
    res_ptx = ptx.raw if torch.version.hip else ptx.value
    libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return res_ptx, kernel_name

class _CudaModule:
    def __init__(self, module):
        self._module = module
        self._kernels = {}
    def __getattr__(self, name):
        if name in self._kernels: return self._kernels[name]
        libcuda = _get_gpu_runtime_library()
        func = ctypes.c_void_p()
        _check_cuda(libcuda.cuModuleGetFunction(ctypes.byref(func), self._module, name.encode("utf-8")))
        kernel = _CudaKernel(func, self._module)
        self._kernels[name] = kernel
        return kernel

class _CudaKernel:
    def __init__(self, func, module):
        self.func = func
        self.module = module
    def __call__(self, grid=(1,1,1), block=(1,1,1), args=None, shared_mem=0, stream=None):
        import torch.cuda
        
        # --- VIKING ENGINE START ---
        # Принудительная чистка перед каждым шагом для Flux
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # --- VIKING ENGINE END ---

        libcuda = _get_gpu_runtime_library()
        if args is None: args = []
        c_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                ptr = ctypes.c_void_p(arg.data_ptr())
                c_args.append(ctypes.byref(ptr))
            elif isinstance(arg, int):
                c_args.append(ctypes.byref(ctypes.c_int(arg)))
            elif isinstance(arg, float):
                c_args.append(ctypes.byref(ctypes.c_double(arg)))

        c_args_arr = (ctypes.c_void_p * len(c_args))(*[ctypes.cast(a, ctypes.c_void_p) for a in c_args])
        curr_stream = stream._as_parameter_ if stream else torch.cuda.current_stream()._as_parameter_
        
        _check_cuda(libcuda.cuLaunchKernel(self.func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem, curr_stream, c_args_arr, None))

def _cuda_load_module(ptx, kernel_names=None):
    import torch.cuda
    libcuda = _get_gpu_runtime_library()
    if isinstance(ptx, str): ptx = ptx.encode("utf-8")
    module = ctypes.c_void_p()
    _check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))
    if not kernel_names: return _CudaModule(module)
    return {n: _CudaModule(module).__getattr__(n) for n in kernel_names}

def _get_device_index(device: Any, optional: bool = False, allow_cpu: bool = False) -> int:
    if isinstance(device, int): return device
    if isinstance(device, str): device = torch.device(device)
    if not torch.jit.is_scripting():
        if isinstance(device, torch.cuda.device): return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)