import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

# three.js and the quad mesh viewer live outside WEB_DIRECTORY, because ComfyUI loads
# every .js file there as an extension. web/quad_mesh_preview.js imports them from here.
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_lib")

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/d3d-3d/lib/{name}")
    async def _d3d_lib(request):
        name = request.match_info["name"]
        path = os.path.join(_LIB_DIR, name)
        if os.path.basename(name) != name or not name.endswith(".js") or not os.path.isfile(path):
            return web.Response(status=404)
        return web.FileResponse(path, headers={"Content-Type": "text/javascript; charset=utf-8",
                                               "Cache-Control": "no-cache"})
except Exception as e:
    print(f"comfyui-d-3D: quad mesh preview route not registered: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
