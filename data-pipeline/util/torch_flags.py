from torch.backends import cuda, cudnn, mha


def set_torch_flags(cfg=None):
    # cuda
    cuda.matmul.allow_tf32 = True
    cuda.enable_mem_efficient_sdp(True)
    cuda.enable_flash_sdp(True)
    cuda.enable_math_sdp(True)
    cuda.enable_cudnn_sdp(True)

    # cudnn
    cudnn.enabled = True
    cudnn.deterministic = False
    cudnn.benchmark = True
    cudnn.allow_tf32 = True

    # multi-head attention
    mha.set_fastpath_enabled(True)
