import torch
from torch import Tensor
def ref_quantizeblockwise_fp4(A: Tensor,absmax: Tensor,out: Tensor,blocksize=64,compress_statics=False,quant_type="fp4"):
    n = A.numel()
    input_shape = A.shape

    if absmax is None:
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)


    if out is None:
        out = torch.zeros(((n+1)//2, 1), dtype=torch.uint8, device=A.device)

    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]

    
