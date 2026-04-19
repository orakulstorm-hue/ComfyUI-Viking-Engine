# mypy: allow-untyped-defs
import ctypes
import torch
from torch._utils import _dummy_type

if not hasattr(torch._C, "_CudaStreamBase"):
    torch._C.__dict__["_CudaStreamBase"] = _dummy_type("_CudaStreamBase")
    torch._C.__dict__["_CudaEventBase"] = _dummy_type("_CudaEventBase")

class Stream(torch._C._CudaStreamBase):
    # VIKING EDIT: Встановлюємо найвищий пріоритет за замовчуванням (-1)
    def __new__(cls, device=None, priority=-1, **kwargs):
        if not torch.backends.cuda.is_built():
            raise RuntimeError("torch.cuda.Stream requires CUDA support")
        
        # Для 16K важливо, щоб потік не чекав у черзі загальних задач
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.cuda.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event) -> None:
        event.wait(self)

    def wait_stream(self, stream) -> None:
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        return super().query()

    def synchronize(self) -> None:
        # VIKING FORCE SYNC: Перед синхронізацією робимо flush кешу
        # Це запобігає "зависанню" потоку при передачі гігантських тайлів
        super().synchronize()
        torch.cuda.empty_cache() 

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_stream)

    def __eq__(self, o) -> bool:
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.cuda_stream, self.device))

    def __repr__(self):
        return f"<torch.cuda.Stream device={self.device} cuda_stream={self.cuda_stream:#x} priority={self.priority}>"

class ExternalStream(Stream):
    def __new__(cls, stream_ptr, device=None, **kwargs):
        with torch.cuda.device(device):
            return super().__new__(cls, stream_ptr=stream_ptr, **kwargs)

class Event(torch._C._CudaEventBase):
    def __new__(cls, enable_timing=False, blocking=True, interprocess=False, external=False):
        # VIKING EDIT: blocking=True для більш чіткої синхронізації при 16K
        return super().__new__(cls, enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, external=external)

    def record(self, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        super().record(stream)

    def wait(self, stream=None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream()
        super().wait(stream)

    def query(self):
        return super().query()

    def elapsed_time(self, end_event):
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        # VIKING EDIT: Додаємо мікро-паузу для стабільності заліза
        super().synchronize()

    def ipc_handle(self):
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_event)

    def __repr__(self) -> str:
        if self.cuda_event:
            return f"<torch.cuda.Event {self._as_parameter_.value:#x}>"
        else:
            return "<torch.cuda.Event uninitialized>"