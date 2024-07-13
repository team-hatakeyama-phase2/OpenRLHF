import torch

from flash_attn import flash_attn_func

q, k, v = torch.randn(1, 128, 3, 16, 64, dtype=torch.float16, device='cuda').unbind(2)

out = flash_attn_func(q, k, v)
