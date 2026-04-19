import torch
from torch._C import dtype

__all__ = ["GPULimits"]

class GPULimits:
    def __init__(self, target_device: torch.device):
        self.device_properties = torch.cuda.get_device_properties(target_device)
        self.compute_capability = int(
            f"{self.device_properties.major}{self.device_properties.minor}"
        )

    def get_fma_per_cycle_per_sm_cuda_cores(self, data_type: dtype) -> int:
        hardcoded_device_values = {
            "fp16_80": 256, "fp32_80": 64, "fp64_80": 32,
            "fp16_89": 256, "fp32_89": 128, "fp64_89": 64, # VIKING: Додано для RTX 4090
            "fp16_90": 64,  "fp32_90": 128, "fp64_90": 64,
            "fp16_100": 256, "fp32_100": 128, "fp64_100": 64,
        }
        dict_key = ""
        if data_type is torch.float16: dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.float32: dict_key = f"fp32_{self.compute_capability}"
        elif data_type is torch.float64: dict_key = f"fp64_{self.compute_capability}"
        else: dict_key = "unknown"

        if dict_key not in hardcoded_device_values:
            # Fallback to nearest architecture if exact not found
            return hardcoded_device_values.get(f"fp16_80", 256)

        return hardcoded_device_values[dict_key]

    def get_fma_per_cycle_per_sm_tensor_cores(self, data_type: dtype) -> int:
        hardcoded_device_values = {
            "int8_80": 2048, "fp16_80": 1024, "fp32_80": 512, "fp64_80": 64,
            "int8_89": 4096, "fp16_89": 2048, "fp32_89": 1024, "fp64_89": 128, # VIKING: Розгін тензорних ядер 4090
            "int8_90": 4096, "fp8_90": 4096, "fp16_90": 2048, "fp32_90": 1024, "fp64_90": 128,
            "int8_100": 8192, "fp8_100": 8192, "fp16_100": 4096, "fp32_100": 2048,
        }
        dict_key = ""
        if data_type is torch.float16 or data_type is torch.bfloat16:
            dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.float32: dict_key = f"fp32_{self.compute_capability}"
        elif data_type is torch.int8: dict_key = f"int8_{self.compute_capability}"
        elif data_type is torch.float64: dict_key = f"fp64_{self.compute_capability}"
        else: dict_key = "unknown"

        if dict_key not in hardcoded_device_values:
            return hardcoded_device_values.get(f"fp16_80", 1024)

        return hardcoded_device_values[dict_key]

    def get_tflops_per_second(self, data_type: dtype, use_tensor_cores: bool = True) -> float:
        num_sms = self.device_properties.multi_processor_count
        clock_rate = self.device_properties.clock_rate
        fma_per_cycle = self.get_fma_per_cycle_per_sm_tensor_cores(data_type) if use_tensor_cores else self.get_fma_per_cycle_per_sm_cuda_cores(data_type)
        return num_sms * fma_per_cycle * 2 * clock_rate / 1e9

    def get_memory_bandwidth_Bps(self) -> int:
        bus_bytes_per_cycle = int(2 * self.device_properties.memory_bus_width / 8)
        mem_clock_rate_Hz = self.device_properties.memory_clock_rate * 1000
        return bus_bytes_per_cycle * mem_clock_rate_Hz * 2

    def get_shared_memory_bandwidth_Bps(self) -> int:
        num_sms = self.device_properties.multi_processor_count
        return num_sms * 128 * self.device_properties.clock_rate * 1000