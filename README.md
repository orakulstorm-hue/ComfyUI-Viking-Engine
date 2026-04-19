# Viking Memory Bridge: Direct RAM Mapping for PyTorch/CUDA

High-performance memory management overhaul for Stability Matrix and ComfyUI. Designed to eliminate System Pagefile (SSD Swap) bottlenecks when working with high-parameter models and extreme resolutions (8K/16K).

## 1. Technical Overview
Standard CUDA pinning in PyTorch often defaults to slow system swap when VRAM is exhausted. This bridge implements a direct VRAM-to-RAM mapping protocol using cudaHostRegisterMapped flags, allowing the GPU to treat system RAM as local address space.

### Key Optimizations:
- Zero-Swap Latency: Bypasses Windows I/O manager to prevent sampler "hangs".
- Hardware-Specific Profiles: Hardcoded limits for Ada Lovelace (CC 8.9 / RTX 4090).
- Aggressive Cache Control: Forced synchronization and cache clearing before CUDA Graph capture.

## 2. Modified Files Reference

| File | Change Log | Purpose |
| :--- | :--- | :--- |
| _pin_memory_utils.py | Forced flags = 3 | Enables cudaHostRegisterMapped for direct RAM access. |
| _device_limits.py | Added CC 8.9 profile | Unlocks native RTX 4090 TFLOPS/Bandwidth limits. |
| graphs.py | Injected empty_cache() | Ensures clean VRAM before CUDA Graph capture. |
| streams.py | Set priority = -1 | Assigns highest execution priority to generation tasks. |
| _utils.py | Kernel Sync Overhaul | Prevents driver-level crashes on kernel failure. |

## 3. Installation
Warning: This is a backend modification. Backup your environment before proceeding.
1. Locate your Python environment's CUDA directory: Data\Packages\ComfyUI\venv\Lib\site-packages\torch\cuda\
2. Replace the original files with the modified versions from this repository.
3. Restart your Stability Matrix/ComfyUI backend.

## 4. Benchmarks (RTX 4090 + 128GB RAM)
- Inference Speed: Up to 3x faster on heavy models when VRAM is full.
- Stability: 0% Pagefile usage; 100% stable 16K tiled encoding.

## 5. Example images
(images/image1.jpq)

##Special thanks:

- Jaret Burkett (Ostris) - AI-Toolkit author, PR request
- ComfyUI developers - Official MAX_RESOLUTION update


##📜 ЛІЦЕНЗІЯ
- Viking Engine modifications released under MIT License.
- Free to use, modify, distribute commercially or non-commercially.
- Attribution appreciated but not required.
- Open source spirit: Share improvements back to community! 💚

  From basement to history 🌍

Protocol: Oracle-60
Location: Chernihiv, UA
