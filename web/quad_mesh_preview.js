// Frontend for "Trellis2 - Preview Quad Mesh": a three.js viewport that draws the
// quads of the saved OBJ (quad edges only, no triangle diagonals), like Blender.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "Trellis2PreviewQuadMesh";
// The viewer and three.js are served by the pack's own route (see __init__.py) so that
// ComfyUI does not load them as extensions. Relative to /extensions/<pack>/ this file.
const LIB = new URL("../../d3d-3d/lib/", import.meta.url).href;
let viewerModule = null;
const getViewerModule = () => (viewerModule ??= import(LIB + "quad_mesh_viewer.js"));

const STYLE = `
.d3d-qm { display:flex; flex-direction:column; width:100%; height:100%; background:#303030;
  border-radius:6px; overflow:hidden; font:12px sans-serif; color:#ddd; box-sizing:border-box; }
.d3d-qm-bar { display:flex; flex-wrap:wrap; gap:4px 10px; align-items:center; padding:4px 6px;
  background:#262626; border-bottom:1px solid #1a1a1a; }
.d3d-qm-bar label { display:flex; align-items:center; gap:3px; cursor:pointer; user-select:none; }
.d3d-qm-bar select, .d3d-qm-bar button { background:#3a3a3a; color:#ddd; border:1px solid #1f1f1f;
  border-radius:4px; padding:1px 6px; font:12px sans-serif; cursor:pointer; }
.d3d-qm-view { position:relative; flex:1; min-height:0; }
.d3d-qm-stats { position:absolute; left:6px; top:4px; pointer-events:none; color:#e6e6e6;
  text-shadow:0 1px 2px #000; line-height:1.35; white-space:pre; }
.d3d-qm-msg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  color:#999; pointer-events:none; text-align:center; padding:8px; }
`;

function injectStyle() {
    if (document.getElementById("d3d-qm-style")) return;
    const s = document.createElement("style");
    s.id = "d3d-qm-style";
    s.textContent = STYLE;
    document.head.appendChild(s);
}

function el(tag, attrs = {}, children = []) {
    const e = document.createElement(tag);
    Object.assign(e, attrs);
    for (const c of children) e.append(c);
    return e;
}

function viewURL(file, name) {
    const p = new URLSearchParams({ filename: name, subfolder: file.subfolder || "", type: file.type || "output" });
    return api.apiURL(`/view?${p.toString()}`);
}

function setupNode(node) {
    injectStyle();
    const root = el("div", { className: "d3d-qm" });
    const bar = el("div", { className: "d3d-qm-bar" });
    const view = el("div", { className: "d3d-qm-view" });
    const stats = el("div", { className: "d3d-qm-stats" });
    const msg = el("div", { className: "d3d-qm-msg", textContent: "Run the workflow to preview the quad mesh" });
    view.append(stats, msg);
    root.append(bar, view);

    const mode = el("select", { title: "Shading" }, [
        el("option", { value: "solid", textContent: "Solid" }),
        el("option", { value: "texture", textContent: "Texture" }),
        el("option", { value: "color", textContent: "Vertex color" }),
    ]);
    const check = (label, opt, value, title) => {
        const box = el("input", { type: "checkbox", checked: value });
        box.addEventListener("change", () => node._d3dViewer?.set(opt, box.checked));
        return [el("label", { title }, [box, label]), box];
    };
    const [wireLabel, wireBox] = check("Wireframe", "wireframe", true, "Show quad edges (W)");
    const [smoothLabel] = check("Smooth", "smooth", false, "Smooth shading");
    const [triLabel] = check("Tris", "highlight", true, "Tint triangles orange and n-gons purple");
    const frameBtn = el("button", { textContent: "Frame", title: "Frame the mesh (F / numpad .)" });
    const viewSel = el("select", { title: "View (numpad 1 / 3 / 7, Ctrl for the opposite side)" }, [
        el("option", { value: "", textContent: "View" }),
        ...["front", "back", "right", "left", "top", "bottom"].map((v) =>
            el("option", { value: v, textContent: v[0].toUpperCase() + v.slice(1) })),
    ]);
    bar.append(mode, wireLabel, smoothLabel, triLabel, viewSel, frameBtn);

    mode.addEventListener("change", () => node._d3dViewer?.set("mode", mode.value));
    frameBtn.addEventListener("click", () => node._d3dViewer?.frame());
    viewSel.addEventListener("change", () => {
        if (viewSel.value) node._d3dViewer?.view(viewSel.value);
        viewSel.value = "";
    });
    view.addEventListener("keydown", (e) => {
        if (e.code === "KeyW" && node._d3dViewer) {
            wireBox.checked = !wireBox.checked;
            node._d3dViewer.set("wireframe", wireBox.checked);
            e.preventDefault();
            e.stopPropagation();
        }
    });
    // Keep the graph from zooming or dragging the node while the mouse is in the viewport.
    for (const type of ["wheel", "pointerdown", "mousedown"]) {
        bar.addEventListener(type, (e) => e.stopPropagation());
    }

    const widget = node.addDOMWidget("quad_mesh_preview", "D3D_QUAD_MESH", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 260,
    });
    widget.computeSize = (width) => [width, 360];

    const ensureViewer = async () => {
        if (node._d3dViewer) return node._d3dViewer;
        const { QuadMeshViewer } = await getViewerModule();
        node._d3dViewer = new QuadMeshViewer(view);
        view.insertBefore(node._d3dViewer.canvas, stats);
        return node._d3dViewer;
    };

    node._d3dShow = async (file) => {
        if (!file?.filename) return;
        node.properties = node.properties || {};
        node.properties.d3d_quad_mesh = file;
        msg.textContent = "Loading…";
        msg.style.display = "flex";
        try {
            const viewer = await ensureViewer();
            const { loadQuadOBJ } = await getViewerModule();
            const s = await loadQuadOBJ(viewer, file.filename, (name) => viewURL(file, name));
            const hasTexture = !!viewer.texture;
            mode.querySelector('option[value="texture"]').disabled = !hasTexture;
            mode.querySelector('option[value="color"]').disabled = !viewer.hasColors;
            mode.value = viewer.opts.mode;
            const lines = [
                file.filename,
                `Verts ${s.vertices.toLocaleString()}   Faces ${s.faces.toLocaleString()}   Edges ${s.edges.toLocaleString()}`,
                `Quads ${s.quads.toLocaleString()}   Tris ${s.tris.toLocaleString()}` + (s.ngons ? `   N-gons ${s.ngons.toLocaleString()}` : ""),
            ];
            stats.textContent = lines.join("\n");
            msg.style.display = "none";
        } catch (err) {
            console.error("Preview Quad Mesh:", err);
            node._d3dViewer?.clear();
            stats.textContent = "";
            msg.textContent = `Could not load ${file.filename}: ${err.message || err}`;
            msg.style.display = "flex";
        }
    };

    if (!node.size || node.size[1] < 480) node.setSize([Math.max(node.size?.[0] || 0, 480), 520]);
}

app.registerExtension({
    name: "d3d.QuadMeshPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const file = message?.quad_mesh?.[0];
            if (file) this._d3dShow?.(file);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            // Reopen the last preview when the workflow is loaded again.
            const file = this.properties?.d3d_quad_mesh;
            if (file) setTimeout(() => this._d3dShow?.(file), 0);
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._d3dViewer?.dispose();
            this._d3dViewer = null;
            return onRemoved?.apply(this, arguments);
        };
    },
});
