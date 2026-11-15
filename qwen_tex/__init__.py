"""Mesh texturing with Qwen Image 2.1 (D3D QwenTex nodes).

Mesh prep, camera rig and back-projection bake come from UltraTex (MIT, vendored in ultratex_core/).
The diffusion step uses ComfyUI's stock Qwen Image 2.1 nodes: the 6 views are generated together as one
2x3 atlas, guided by a control atlas rendered from the mesh (Fun ControlNet: depth / grayscale) and by the
reference image (Text Encode Qwen Image 2.1).
"""
