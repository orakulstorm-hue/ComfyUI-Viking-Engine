

# Viking Memory Bridge: Direct RAM Mapping for PyTorch/CUDA

High-performance memory management overhaul for Stability Matrix and ComfyUI. Designed to eliminate System Pagefile (SSD Swap) bottlenecks when working with high-parameter models and extreme resolutions (8K/16K).

## Technical Overview
Standard CUDA pinning in PyTorch often defaults to slow system swap when VRAM is exhausted. This bridge implements a direct VRAM-to-RAM mapping protocol using cudaHostRegisterMapped flags, allowing the GPU to treat system RAM as local address space.

## Bottleneck
When working with models with large numbers of parameters (e.g., 32 bytes) or generating at extreme resolutions (8K/16K) on consumer hardware like the RTX 4090 (24 GB of VRAM), memory exhaustion is inevitable. The default behavior of PyTorch/CUDA is to move overflow tensors to the system swap file (SSD Swap). This results in significant I/O latency, causing output samplers to hang for several minutes waiting for disk reads.

## Solution: Direct Host Mapping
This backend modification for ComfyUI/Stability Matrix completely bypasses the SSD swap file. By forcing the GPU to map system RAM directly into its address space, we utilize high-speed system memory (DDR5) as an expanded video memory pool, eliminating the disk I/O bottleneck.

---

## Main Modifications and Detailed Analysis of Engineering Solutions

### 1. Non-Copy Memory Access (`_pin_memory_utils.py`)
Instead of standard memory pinning, this modification forces `flags = 3` (`cudaHostRegisterMapped` + `cudaHostRegisterPortable`).

* **Mechanism:** Instructs the CUDA driver to map allocated host memory directly into the device address space.

* **Impact:** The GPU reads excess data directly from system RAM via the PCIe bus, completely bypassing the OS I/O manager and the SSD pagefile. This reduces data transfer latency from minutes to seconds.

### 2. Unlocking Hardware Architecture (`_device_limits.py`)
Common backend profiles often fall back to legacy compute limits, underutilizing modern flagship GPUs.

* **Mechanism:** Hard-coded hardware limits specifically for compute capability 8.9 (Ada Lovelace / RTX 4090).

* **Impact:** Accurate scheduling of Tensor Core int8 and fp16 matrix multiplications, leveraging the maximum native TFLOPS and memory bandwidth of the GPU.

### 3. Strict Video Memory Allocation (`graphs.py` and `_utils.py`)
Capturing CUDA graphs at 8K/16K resolution requires a perfectly clean memory state to avoid out-of-memory (OOM) crashes.

* **Mechanism:** Implement `torch.cuda.empty_cache()` and force synchronization events to be executed immediately before CUDA graph capture.

* **Impact:** Ensures the maximum number of available contiguous memory blocks before allocating complex mathematical graphs, significantly improving stability under extreme scaling.

### 4. Stream Execution Priority (`streams.py`)
* **Mechanism:** Set `priority = -1` for stream execution.

* **Impact:** Gives generation tasks the highest possible execution priority. This prevents CUDA streams from being interrupted by OS background tasks, eliminating micro-stutters and driver timeouts during resource-intensive tensor operations.

---
### Key Optimizations:
- Zero-Swap Latency: Bypasses Windows I/O manager to prevent sampler "hangs".
- Hardware-Specific Profiles: Hardcoded limits for Ada Lovelace (CC 8.9 / RTX 4090).
- Aggressive Cache Control: Forced synchronization and cache clearing before CUDA Graph capture.

## Modified Files Reference

| File | Change Log | Purpose |
| :--- | :--- | :--- |
| _pin_memory_utils.py | Forced flags = 3 | Enables cudaHostRegisterMapped for direct RAM access. |
| _device_limits.py | Added CC 8.9 profile | Unlocks native RTX 4090 TFLOPS/Bandwidth limits. |
| graphs.py | Injected empty_cache() | Ensures clean VRAM before CUDA Graph capture. |
| streams.py | Set priority = -1 | Assigns highest execution priority to generation tasks. |
| _utils.py | Kernel Sync Overhaul | Prevents driver-level crashes on kernel failure. |

## Installation
Warning: This is a backend modification. Backup your environment before proceeding.
1. Locate your Python environment's CUDA directory: Data\Packages\ComfyUI\venv\Lib\site-packages\torch\cuda\
2. Replace the original files with the modified versions from this repository.
3. Restart your Stability Matrix/ComfyUI backend.

## Benchmarks (RTX 4090 + 128GB RAM)
- Inference Speed: Up to 3x faster on heavy models when VRAM is full.
- Stability: 0% Pagefile usage; 100% stable 16K tiled encoding.

## Example images
![Example images](images/image1.jpg)
![Example images](images/image2.jpg)

## Video: 120 steps, resolution 32768x18432, 6 min generation time
<video src="video/video1.mp4" controls="controls" style="max-width: 100%;"></video>

## Special thanks:

- Jaret Burkett (Ostris) - AI-Toolkit author, [PR request](https://github.com/orakulstorm-hue/ai-toolkit)
- ComfyUI developers - Official (https://github.com/comfy-org/ComfyUI)


## 📜 License
- Viking Engine modifications released under MIT License.
- Free to use, modify, distribute commercially or non-commercially.
- Attribution appreciated but not required.
- Open source spirit: Share improvements back to community! 💚

  From basement to history 🌍
  Location: Chernihiv, UA
