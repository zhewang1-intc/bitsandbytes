import torch
from torch import Tensor
# TODO: support double-quant for absmax
torch.ops.load_library("/home/zhe/bitsandbytes/tests/custom_op/build/libcustom_allreduce_op.so")
def ref_quantizeblockwise_fp4(A: Tensor,absmax: Tensor=None,out: Tensor=None,blocksize=64,compress_statics=False,quant_type="fp4"):
    n = A.numel()
    input_shape = A.shape

    if absmax is None:
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)


    if out is None:
        out = torch.zeros(((n+1)//2, 1), dtype=torch.uint8, device=A.device)

    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]

    torch.ops.my_ops.ref_fp4_quantize(A,absmax,out,64,n)

    return absmax,out

# TODO: support absmax-dequant 
def ref_dequantizeblockwise_fp4(A: Tensor,out: Tensor,absmax: Tensor=None,blocksize=64,compress_statics=False,quant_type="fp4"):
    assert(out != None)
    shape = out.shape
    dtype = out.dtype
    n = out.numel()
    torch.ops.my_ops.ref_fp4_dequantize(A,absmax,out,64,n)
    return out
    