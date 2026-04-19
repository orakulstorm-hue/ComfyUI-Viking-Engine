import torch

def pin_memory(data_ptr: int, size: int) -> None:
    cudart = torch.cuda.cudart()
    
    # VIKING OPTIMIZATION: 
    # Використовуємо 0x02 (cudaHostRegisterMapped) + 0x01 (Portable) = 3
    # Це дозволяє GPU мапити RAM прямо в свій адресний простір. 
    # Ідеально для 16K, коли дані не влазять у VRAM.
    flags = 3 
    
    succ = int(
        cudart.cudaHostRegister(
            data_ptr,
            size,
            flags,
        )
    )

    if succ != 0:
        # Якщо прапорець 3 не підтримується драйвером, пробуємо стандартний 1
        succ = int(cudart.cudaHostRegister(data_ptr, size, 1))
        
    if succ != 0:
        # Примусова синхронізація при помилці, щоб не вилітало
        torch.cuda.synchronize()
        raise RuntimeError(
            f"Viking Memory Bridge: Registering {size} bytes failed. CUDA Error: {succ}."
        )

def unpin_memory(data_ptr: int) -> None:
    # Перед розреєстрацією чекаємо, поки GPU закінчить роботу з цим блоком
    torch.cuda.synchronize()
    succ = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    # Не даємо системі падати, якщо блок уже чистий
    if succ != 0 and succ != 3: # 3 - це 'not registered'
        pass