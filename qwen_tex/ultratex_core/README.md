Vendored from https://github.com/visualbruno/ComfyUI-UltraTex (commit 1f2392308e7d08374cb1c3844afe7e3ee2fadbe0),
`ultratex_core/render.py`, `prep.py` and `bake.py`, MIT license (see LICENSE).

Local change: `render._context()` falls back to the EGL backend when no X display is available
(headless Linux / Kubernetes pods).
