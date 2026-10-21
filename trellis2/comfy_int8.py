"""
Loading of the Comfy-Org repackaged TRELLIS.2 / Pixal3D checkpoints.

Comfy-Org ships every model of the pipeline in a few single files instead of the
HF `ckpts/` folder this node was written for:

    diffusion_models/trellis_2_int8_convrot.safetensors      (or pixal3d_*, *_bf16)
        model.structure_model.*   -> sparse_structure_flow_model
        model.img2shape_512.*     -> shape_slat_flow_model_512
        model.img2shape.*         -> shape_slat_flow_model_1024
        model.shape2txt.*         -> tex_slat_flow_model_1024
    vae/trellis_2_shape_vae_bf16.safetensors
        struct_dec.*              -> sparse_structure_decoder
        shape_dec.*               -> shape_slat_decoder
    vae/trellis_2_texture_vae_bf16.safetensors
        txt_dec.*                 -> tex_slat_decoder
    clip_vision/dino_v3_vit_l.safetensors (HF transformers DINOv3ViTModel keys)

Below the prefix the key names are the original ones, so the models are built
from the original configs and loaded as-is. Linear layers stored as
`int8_tensorwise` (with ConvRot) keep their int8 weights and run through
comfy_kitchen, which ships with ComfyUI.

A model path in the pipeline args can be a "comfy source" string:

    comfy::<folder>/<file>::<key prefix>::<config name>

which models.from_pretrained() hands over to load_model() below.
"""
import os
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_TAG = 'comfy::'
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'comfy')

# Prefixes inside the Comfy-Org diffusion model file
PREFIX_SS_FLOW = 'model.structure_model.'
PREFIX_SHAPE_512 = 'model.img2shape_512.'
PREFIX_SHAPE_1024 = 'model.img2shape.'
PREFIX_TEX = 'model.shape2txt.'
PREFIX_SS_DEC = 'struct_dec.'
PREFIX_SHAPE_DEC = 'shape_dec.'
PREFIX_TEX_DEC = 'txt_dec.'


def make_source(folder: str, filename: str, prefix: str, config_name: str) -> str:
    return f"{SOURCE_TAG}{folder}/{filename}::{prefix}::{config_name}"


def is_source(path: str) -> bool:
    return SOURCE_TAG in path


def parse_source(path: str):
    # The pipeline may have prepended its own folder ("{self.path}/comfy::...")
    spec = path[path.index(SOURCE_TAG) + len(SOURCE_TAG):]
    file_spec, prefix, config_name = spec.split('::')
    folder, filename = file_spec.split('/', 1)
    return folder, filename, prefix, config_name


def resolve_file(folder: str, filename: str) -> str:
    import folder_paths
    full_path = folder_paths.get_full_path(folder, filename)
    if full_path is None:
        raise FileNotFoundError(f"'{filename}' not found in ComfyUI models/{folder}")
    return full_path


def read_header(file_path: str) -> dict:
    import struct
    with open(file_path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    return header


def load_prefixed_state_dict(file_path: str, prefix: str) -> dict:
    from safetensors import safe_open
    sd = {}
    with safe_open(file_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            if key.startswith(prefix):
                sd[key[len(prefix):]] = f.get_tensor(key)
    if not sd:
        raise KeyError(f"No tensors with prefix '{prefix}' in {os.path.basename(file_path)}")
    return sd


def detect_diffusion_file(file_path: str) -> dict:
    """
    Tell what a Comfy-Org diffusion model file holds, from its header only.
    """
    keys = read_header(file_path).keys()
    quant_keys = [k for k in keys if k.endswith('.comfy_quant')]
    return {
        'is_pixal3d': any('.cross_attn.proj_linear.' in k for k in keys),
        'is_int8': len(quant_keys) > 0,
        'has_shape_512': any(k.startswith(PREFIX_SHAPE_512) for k in keys),
        'has_tex': any(k.startswith(PREFIX_TEX) for k in keys),
    }


SHAPE_SLAT_NORMALIZATION = {
    "mean": [
        0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
        -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
        0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
        -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665
    ],
    "std": [
        5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
        5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
        4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
        5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691
    ],
}

TEX_SLAT_NORMALIZATION = {
    "mean": [
        3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075,
        0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149,
        -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
        1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862
    ],
    "std": [
        2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822,
        2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588,
        2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
        2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190
    ],
}


def _sampler(guidance_strength, guidance_rescale, guidance_interval, rescale_t):
    return {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 12,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": guidance_interval,
            "rescale_t": rescale_t,
        },
    }


def build_pipeline_args(diffusion_file: str, shape_vae_file: str, texture_vae_file: str, is_pixal3d: bool) -> dict:
    """
    The equivalent of TRELLIS.2-4B / Pixal3D pipeline.json, pointing every model
    at its part of the Comfy-Org files. Both repos share samplers and normalizations.
    """
    kind = 'pixal3d' if is_pixal3d else 'trellis2'
    dm = lambda prefix, config: make_source('diffusion_models', diffusion_file, prefix, config)
    models = {
        "sparse_structure_flow_model": dm(PREFIX_SS_FLOW, f'{kind}_ss_flow'),
        "shape_slat_flow_model_512": dm(PREFIX_SHAPE_512, f'{kind}_shape_flow_512'),
        "shape_slat_flow_model_1024": dm(PREFIX_SHAPE_1024, f'{kind}_shape_flow_1024'),
        "tex_slat_flow_model_1024": dm(PREFIX_TEX, f'{kind}_tex_flow_1024'),
        "sparse_structure_decoder": make_source('vae', shape_vae_file, PREFIX_SS_DEC, 'ss_dec'),
        "shape_slat_decoder": make_source('vae', shape_vae_file, PREFIX_SHAPE_DEC, 'shape_dec'),
        "tex_slat_decoder": make_source('vae', texture_vae_file, PREFIX_TEX_DEC, 'tex_dec'),
    }
    if not is_pixal3d:
        # Comfy-Org ships a single texture denoiser. It is rope based, so it also
        # runs the 512 stage, the same way the Pixal3D path does.
        models["tex_slat_flow_model_512"] = models["tex_slat_flow_model_1024"]

    return {
        "models": models,
        "sparse_structure_sampler": _sampler(7.5, 0.7, [0.6, 1.0], 5.0),
        "shape_slat_sampler": _sampler(7.5, 0.5, [0.6, 1.0], 3.0),
        "tex_slat_sampler": _sampler(1.0, 0.0, [0.6, 0.9], 3.0),
        "shape_slat_normalization": SHAPE_SLAT_NORMALIZATION,
        "tex_slat_normalization": TEX_SLAT_NORMALIZATION,
        "image_cond_model": {"name": "DinoV3FeatureExtractor", "args": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"}},
        "rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}},
        "default_pipeline_type": "1536_cascade" if is_pixal3d else "1024_cascade",
    }


# =============================================================================
# INT8 ConvRot linear
# =============================================================================

_ck_int8_linear = None
_ck_dequant = None


def _kitchen():
    global _ck_int8_linear, _ck_dequant
    if _ck_int8_linear is None:
        try:
            import comfy_kitchen as ck
            _ck_int8_linear = ck.int8_linear
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                'The int8 models need comfy_kitchen, which ships with ComfyUI. '
                'Update ComfyUI (python -m pip install -r ComfyUI/requirements.txt).') from e
        try:
            from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight, dequantize_int8_simple
            _ck_dequant = (dequantize_int8_convrot_weight, dequantize_int8_simple)
        except ImportError:
            _ck_dequant = None
    return _ck_int8_linear, _ck_dequant


class Int8ConvRotLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear / SparseLinear holding an int8_tensorwise
    weight (optionally ConvRot rotated) and its per-row scale.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, convrot: bool, convrot_groupsize: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        self.register_buffer('weight', torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.empty((out_features, 1), dtype=torch.float32))
        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None
        # Same knobs as SparseLinear
        self.low_vram = False
        self.chunk_size = 65536
        self._fast_failed = False

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, convrot={self.convrot}'

    def dequantized_weight(self, dtype):
        _, dequant = _kitchen()
        scale = self.weight_scale.float()
        if dequant is None:
            raise RuntimeError('comfy_kitchen has no eager int8 dequantize')
        if self.convrot:
            w = dequant[0](self.weight, scale, self.convrot_groupsize)
        else:
            w = dequant[1](self.weight, scale)
        return w.to(dtype)

    def _linear_2d(self, x: torch.Tensor) -> torch.Tensor:
        int8_linear, _ = _kitchen()
        out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
        bias = self.bias.to(out_dtype) if self.bias is not None else None
        if x.is_cuda and not self._fast_failed:
            try:
                return int8_linear(
                    x.contiguous(), self.weight, self.weight_scale.float(), bias, out_dtype,
                    convrot=self.convrot, convrot_groupsize=self.convrot_groupsize,
                )
            except Exception as e:
                # GPU without int8 tensor cores or an unsupported shape: dequantize instead
                print(f'[Trellis2 int8] fast int8 matmul unavailable ({e}), using dequantized weights')
                self._fast_failed = True
        return F.linear(x.to(out_dtype), self.dequantized_weight(out_dtype), bias)

    def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.low_vram and x2.shape[0] > self.chunk_size:
            out = torch.cat([self._linear_2d(c) for c in x2.split(self.chunk_size)], dim=0)
        else:
            out = self._linear_2d(x2)
        return out.reshape(*shape[:-1], self.out_features)

    def forward(self, x):
        if hasattr(x, 'feats') and hasattr(x, 'replace'):
            return x.replace(self._forward_tensor(x.feats))
        return self._forward_tensor(x)


def _quant_conf(sd: dict, name: str) -> Optional[dict]:
    t = sd.get(f'{name}.comfy_quant')
    if t is None:
        return None
    return json.loads(bytes(t.tolist()).decode('utf-8'))


def _swap_quantized_linears(model: nn.Module, sd: dict) -> int:
    """
    Replace every linear layer that is stored quantized in `sd` by an
    Int8ConvRotLinear. Returns how many layers were swapped.
    """
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        conf = _quant_conf(sd, name)
        if conf is None:
            continue
        if conf.get('format') != 'int8_tensorwise':
            raise NotImplementedError(f"Quantization '{conf.get('format')}' of layer {name} is not supported")
        new = Int8ConvRotLinear(
            module.in_features, module.out_features, module.bias is not None,
            convrot=bool(conf.get('convrot', False)),
            convrot_groupsize=int(conf.get('convrot_groupsize', 256)),
        )
        new.low_vram = getattr(module, 'low_vram', False)
        new.chunk_size = getattr(module, 'chunk_size', 65536)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, new)
        swapped += 1
    return swapped


def load_model(path: str, **kwargs) -> nn.Module:
    from . import models

    folder, filename, prefix, config_name = parse_source(path)
    file_path = resolve_file(folder, filename)
    with open(os.path.join(CONFIG_DIR, f'{config_name}.json'), 'r') as f:
        config = json.load(f)

    sd = load_prefixed_state_dict(file_path, prefix)

    with torch.device('meta'):
        model = getattr(models, config['name'])(**config['args'], **kwargs)
    swapped = _swap_quantized_linears(model, sd)
    for key in [k for k in sd if k.endswith('.comfy_quant')]:
        del sd[key]

    # Buffers that are not in the checkpoint (e.g. rope_phases) are rebuilt on CPU
    model = model.to_empty(device='cpu')
    _rebuild_non_persistent(model, config, kwargs)

    # Cast float tensors to the dtype the model was built for, keep int8 and scales
    target = {n: t for n, t in list(model.named_parameters()) + list(model.named_buffers())}
    for key, value in sd.items():
        if key in target and value.is_floating_point() and not key.endswith('weight_scale'):
            sd[key] = value.to(target[key].dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith('rope_phases')]
    if missing or unexpected:
        raise RuntimeError(f'{filename} [{prefix}] does not match {config_name}: missing={missing[:5]} unexpected={unexpected[:5]}')

    print(f'[Trellis2] Loaded {config_name} from {filename} ({prefix.rstrip(".")}), {swapped} int8 layers')
    return model


def _rebuild_non_persistent(model: nn.Module, config: dict, kwargs: dict):
    """
    to_empty() leaves buffers uninitialised. Recompute the ones the Comfy-Org
    files do not carry, the same way the model constructors do.
    """
    rope_phases = getattr(model, 'rope_phases', None)
    if isinstance(rope_phases, torch.Tensor):
        from .modules.attention import RotaryPositionEmbedder
        resolution = config['args']['resolution']
        pos_embedder = RotaryPositionEmbedder(model.model_channels // model.num_heads, 3)
        coords = torch.meshgrid(*[torch.arange(resolution) for _ in range(3)], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        model.rope_phases = pos_embedder(coords).to(rope_phases.dtype)

    # The position embedders keep `freqs` as a plain attribute, so the meta
    # device construction leaves it without data
    for module in model.modules():
        freqs = module.__dict__.get('freqs')
        if isinstance(freqs, torch.Tensor) and freqs.is_meta:
            f = torch.arange(module.freq_dim, dtype=torch.float32) / module.freq_dim
            if hasattr(module, 'rope_freq'):
                module.freqs = module.rope_freq[0] / (module.rope_freq[1] ** f)
            else:
                module.freqs = 1.0 / (10000 ** f)


# =============================================================================
# DINOv3 from the Comfy-Org clip_vision file
# =============================================================================

DINOV3_VITL16_CONFIG = dict(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    hidden_act='gelu',
    use_gated_mlp=False,
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_register_tokens=4,
    rope_theta=100.0,
    pos_embed_rescale=2.0,
    layer_norm_eps=1e-5,
    layerscale_value=1.0,
    drop_path_rate=0.0,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    proj_bias=True,
    mlp_bias=True,
)


def load_dinov3(model_name: str):
    """
    Same as DINOv3ViTModel.from_pretrained(model_name), but model_name can also be
    a .safetensors file in the HF transformers key layout (Comfy-Org clip_vision).
    """
    from transformers import DINOv3ViTModel, DINOv3ViTConfig
    if not model_name.endswith('.safetensors'):
        return DINOv3ViTModel.from_pretrained(model_name)

    from safetensors.torch import load_file
    sd = load_file(model_name)
    # dino_v3_L_naf_fp32 also carries the NAF upsampler under naf.*
    sd = {k: v for k, v in sd.items() if not k.startswith('naf.')}
    model = DINOv3ViTModel(DINOv3ViTConfig(**DINOV3_VITL16_CONFIG))
    # transformers >= 5 nests the encoder blocks under model.layer.*, the file keeps them at layer.*
    if hasattr(model, 'model'):
        sd = {('model.' + k if k.startswith('layer.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'{os.path.basename(model_name)} is not a DINOv3 ViT-L/16: missing={missing[:5]} unexpected={unexpected[:5]}')
    model.eval()
    return model
