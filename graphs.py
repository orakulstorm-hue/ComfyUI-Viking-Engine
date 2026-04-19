# pylint: disable=useless-parent-delegation
from __future__ import annotations
import gc
import typing
from collections.abc import Callable
from typing import Optional, overload, TYPE_CHECKING, Union
from typing_extensions import ParamSpec, Self, TypeVar

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE

from .._utils import _dummy_type

__all__ = [
    "is_current_stream_capturing",
    "graph_pool_handle",
    "CUDAGraph",
    "graph",
    "make_graphed_callables",
]

_R = TypeVar("_R")
_P = ParamSpec("_P")

if not hasattr(torch._C, "_CudaStreamBase"):
    torch._C.__dict__["_CUDAGraph"] = _dummy_type("_CUDAGraph")
    torch._C.__dict__["_graph_pool_handle"] = _dummy_type("_graph_pool_handle")
    torch._C.__dict__["_cuda_isCurrentStreamCapturing"] = _dummy_type("_cuda_isCurrentStreamCapturing")

from torch._C import (  
    _cuda_isCurrentStreamCapturing,
    _CUDAGraph,
    _graph_pool_handle,
)

def is_current_stream_capturing() -> bool:
    return _cuda_isCurrentStreamCapturing()

def graph_pool_handle() -> _POOL_HANDLE:
    return torch.cuda._POOL_HANDLE(_graph_pool_handle())

class CUDAGraph(torch._C._CUDAGraph):
    def __new__(cls, keep_graph: bool = False) -> Self:
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool: Optional[_POOL_HANDLE] = None, capture_error_mode: str = "thread_local") -> None:
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)

class graph:
    default_capture_stream: Optional[torch.cuda.Stream] = None

    def __init__(self, cuda_graph: CUDAGraph, pool: Optional[_POOL_HANDLE] = None, stream: Optional[torch.cuda.Stream] = None, capture_error_mode: str = "thread_local"):
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()
        self.pool = () if pool is None else (pool,)
        self.capture_stream = stream if stream is not None else self.__class__.default_capture_stream
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode

    def __enter__(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.stream_ctx.__enter__()
        self.cuda_graph.capture_begin(*self.pool, capture_error_mode=self.capture_error_mode)

    def __exit__(self, *args: object) -> None:
        self.cuda_graph.capture_end()
        self.stream_ctx.__exit__(*args)

# Повертаємо оригінальну логіку make_graphed_callables (без само-імпорту), 
# щоб уникнути циклічної помилки.
def make_graphed_callables(callables, sample_args, num_warmup_iters=3, allow_unused_input=False, pool=None):
    # Ця функція тепер буде використовувати наші виправлені класи CUDAGraph автоматично
    from torch.cuda._graphs import make_graphed_callables as _real_make
    return _real_make(callables, sample_args, num_warmup_iters, allow_unused_input, pool)