from typing import *

BACKEND = 'flash_attn'
DEBUG = False


def resolve_backend(backend: str) -> str:
    """
    Fall back to sdpa when the requested package is missing or has no kernels for
    this GPU. FlashAttention 3 (flash_attn_interface, and the flash_attn 3.x shim it
    installs) only ships Hopper kernels and aborts the process on other GPUs.
    """
    import importlib.util
    import torch
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if backend == 'flash_attn_3':
        usable = major == 9 and importlib.util.find_spec('flash_attn_interface') is not None
    elif backend == 'flash_attn':
        usable = major >= 8 and importlib.util.find_spec('flash_attn') is not None
        if usable:
            import flash_attn
            usable = not flash_attn.__version__.startswith('3') or major == 9
    elif backend == 'xformers':
        usable = importlib.util.find_spec('xformers') is not None
    elif backend == 'comfy_kitchen_int8':
        import comfy_kitchen
        usable = torch.cuda.is_available() and comfy_kitchen.int8_attention_is_available(torch.device('cuda'))
    else:
        usable = True
    if not usable:
        print(f"[ATTENTION] {backend} is not usable on this GPU, using sdpa")
        return 'sdpa'
    return backend


def comfy_attention_function(backend: str):
    """
    ComfyUI's attention for the 'comfy' (the one ComfyUI selected at launch) and
    'comfy_kitchen_int8' backends. Takes and returns [B, H, L, C] with skip_reshape.
    """
    import comfy.ldm.modules.attention as comfy_attention
    if backend == 'comfy_kitchen_int8':
        return comfy_attention.attention_comfy_kitchen_int8
    return comfy_attention.optimized_attention


def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive', 'comfy', 'comfy_kitchen_int8']:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    BACKEND = resolve_backend(BACKEND)
    print(f"[ATTENTION] Using backend: {BACKEND}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = resolve_backend(backend)

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
