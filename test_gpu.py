"""
quick check that torch can see the GPU before we build anything on top of it.
run this with: uv run python test_gpu.py
"""

import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda version (torch built with):", torch.version.cuda) #type: ignore
    print("gpu name:", torch.cuda.get_device_name(0))
    print("gpu count:", torch.cuda.device_count())

    # actually run something on the gpu, not just check the flag
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print("matmul on gpu worked, result shape:", z.shape)

    # check how much vram is free, since we only have 6gb
    free, total = torch.cuda.mem_get_info()
    print(f"vram free: {free / 1e9:.2f} GB / total: {total / 1e9:.2f} GB")
else:
    print("CUDA NOT AVAILABLE - torch will run on cpu only")
    print("check: did you install torch with cu128 wheels? run 'uv pip list | grep torch'")