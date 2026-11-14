// Quad mesh viewer for "Trellis2 - Preview Quad Mesh".
//
// Reads the OBJ written by the node itself (instead of three.js' OBJLoader, which
// triangulates on load and forgets the polygons) so the wireframe draws the edges of
// each quad and never the diagonal used to render it, like Blender's edit mode overlay.
import * as THREE from "./three.module.min.js";

const BG = 0x303030;
const SOLID_COLOR = 0xb8b8b8;
const WIRE_COLOR = 0x0a0a0a;
const TRI_TINT = [1.0, 0.55, 0.15];   // triangles
const NGON_TINT = [0.85, 0.3, 0.9];   // faces with more than 4 corners

export function parseOBJ(text) {
    const positions = [];
    const colors = [];
    const uvs = [];
    const faces = [];      // each: { v: [..], t: [..] | null }
    let mtllib = null;
    let hasColors = false;

    let start = 0;
    const len = text.length;
    while (start < len) {
        let end = text.indexOf("\n", start);
        if (end < 0) end = len;
        const line = text.slice(start, end).trim();
        start = end + 1;
        if (!line || line[0] === "#") continue;
        const parts = line.split(/\s+/);
        const tag = parts[0];
        if (tag === "v") {
            positions.push(+parts[1], +parts[2], +parts[3]);
            if (parts.length >= 7) {
                colors.push(+parts[4], +parts[5], +parts[6]);
                hasColors = true;
            } else {
                colors.push(1, 1, 1);
            }
        } else if (tag === "vt") {
            uvs.push(+parts[1], +parts[2]);
        } else if (tag === "f") {
            const nv = positions.length / 3;
            const nt = uvs.length / 2;
            const v = [];
            const t = [];
            let hasT = true;
            for (let i = 1; i < parts.length; i++) {
                const ref = parts[i].split("/");
                let vi = parseInt(ref[0], 10);
                v.push(vi < 0 ? nv + vi : vi - 1);
                if (ref.length > 1 && ref[1] !== "") {
                    const ti = parseInt(ref[1], 10);
                    t.push(ti < 0 ? nt + ti : ti - 1);
                } else {
                    hasT = false;
                }
            }
            if (v.length >= 3) faces.push({ v, t: hasT ? t : null });
        } else if (tag === "mtllib") {
            mtllib = line.slice(7).trim();
        }
    }
    return { positions, colors: hasColors ? colors : null, uvs, faces, mtllib };
}

export function parseMTL(text) {
    // Only the base color texture is used by the preview.
    for (const raw of text.split("\n")) {
        const line = raw.trim();
        if (line.startsWith("map_Kd")) {
            const parts = line.split(/\s+/);
            return parts[parts.length - 1];
        }
    }
    return null;
}

function buildGeometry(obj) {
    const P = obj.positions;
    const nPos = P.length / 3;
    let nTri = 0;
    let quads = 0, tris = 0, ngons = 0;
    for (const f of obj.faces) {
        nTri += f.v.length - 2;
        if (f.v.length === 4) quads++;
        else if (f.v.length === 3) tris++;
        else ngons++;
    }
    const nCorner = nTri * 3;
    const pos = new Float32Array(nCorner * 3);
    const uv = new Float32Array(nCorner * 2);
    const flatN = new Float32Array(nCorner * 3);
    const smoothN = new Float32Array(nCorner * 3);
    const baseCol = new Float32Array(nCorner * 3);
    const tint = new Float32Array(nCorner * 3);
    const cornerPos = new Uint32Array(nCorner);    // OBJ vertex of each rendered corner
    const accum = new Float32Array(nPos * 3);
    const hasUV = obj.uvs.length > 0 && obj.faces.every((f) => f.t);

    // Unique polygon edges (quad sides), keyed by the two OBJ vertex indices.
    const edgeSet = new Set();
    const edges = [];
    const key = (a, b) => (a < b ? a * nPos + b : b * nPos + a);

    let c = 0;
    for (const f of obj.faces) {
        const v = f.v;
        const n = v.length;
        // Newell normal: area weighted and robust for non-planar quads.
        let nx = 0, ny = 0, nz = 0;
        for (let i = 0; i < n; i++) {
            const a = v[i] * 3, b = v[(i + 1) % n] * 3;
            nx += (P[a + 1] - P[b + 1]) * (P[a + 2] + P[b + 2]);
            ny += (P[a + 2] - P[b + 2]) * (P[a] + P[b]);
            nz += (P[a] - P[b]) * (P[a + 1] + P[b + 1]);
        }
        for (let i = 0; i < n; i++) {
            const a = v[i] * 3;
            accum[a] += nx; accum[a + 1] += ny; accum[a + 2] += nz;
            const b = v[(i + 1) % n];
            const k = key(v[i], b);
            if (!edgeSet.has(k)) { edgeSet.add(k); edges.push(v[i], b); }
        }
        const l = Math.hypot(nx, ny, nz) || 1;
        nx /= l; ny /= l; nz /= l;
        const t = n === 4 ? null : n === 3 ? TRI_TINT : NGON_TINT;
        for (let i = 1; i < n - 1; i++) {
            for (const j of [0, i, i + 1]) {
                const vi = v[j];
                cornerPos[c] = vi;
                pos[c * 3] = P[vi * 3]; pos[c * 3 + 1] = P[vi * 3 + 1]; pos[c * 3 + 2] = P[vi * 3 + 2];
                flatN[c * 3] = nx; flatN[c * 3 + 1] = ny; flatN[c * 3 + 2] = nz;
                if (hasUV) {
                    const ti = f.t[j];
                    uv[c * 2] = obj.uvs[ti * 2]; uv[c * 2 + 1] = obj.uvs[ti * 2 + 1];
                }
                if (obj.colors) {
                    baseCol[c * 3] = obj.colors[vi * 3]; baseCol[c * 3 + 1] = obj.colors[vi * 3 + 1]; baseCol[c * 3 + 2] = obj.colors[vi * 3 + 2];
                } else {
                    baseCol[c * 3] = baseCol[c * 3 + 1] = baseCol[c * 3 + 2] = 1;
                }
                if (t) { tint[c * 3] = t[0]; tint[c * 3 + 1] = t[1]; tint[c * 3 + 2] = t[2]; }
                else { tint[c * 3] = tint[c * 3 + 1] = tint[c * 3 + 2] = 1; }
                c++;
            }
        }
    }
    for (let i = 0; i < nPos; i++) {
        const l = Math.hypot(accum[i * 3], accum[i * 3 + 1], accum[i * 3 + 2]) || 1;
        accum[i * 3] /= l; accum[i * 3 + 1] /= l; accum[i * 3 + 2] /= l;
    }
    for (let k = 0; k < nCorner; k++) {
        const vi = cornerPos[k] * 3;
        smoothN[k * 3] = accum[vi]; smoothN[k * 3 + 1] = accum[vi + 1]; smoothN[k * 3 + 2] = accum[vi + 2];
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(flatN, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nCorner * 3), 3));
    if (hasUV) geom.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geom.computeBoundingBox();
    geom.computeBoundingSphere();

    const edgePos = new Float32Array(edges.length * 3);
    for (let i = 0; i < edges.length; i++) {
        const vi = edges[i] * 3;
        edgePos[i * 3] = P[vi]; edgePos[i * 3 + 1] = P[vi + 1]; edgePos[i * 3 + 2] = P[vi + 2];
    }
    const edgeGeom = new THREE.BufferGeometry();
    edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));

    return {
        geom, edgeGeom, flatN, smoothN, baseCol, tint, hasUV,
        stats: { vertices: nPos, faces: obj.faces.length, quads, tris, ngons, edges: edges.length / 2 },
    };
}

export class QuadMeshViewer {
    constructor(container) {
        this.container = container;
        this.opts = { mode: "solid", wireframe: true, smooth: false, highlight: true };
        this.data = null;
        this.texture = null;
        this.hasColors = false;

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio || 1);
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.canvas = this.renderer.domElement;
        this.canvas.style.display = "block";
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.outline = "none";
        this.canvas.tabIndex = 0;
        container.appendChild(this.canvas);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(BG);
        this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
        this.scene.add(this.camera);
        // Blender-like studio light: soft ambient from above plus a key light that follows the view.
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
        const key = new THREE.DirectionalLight(0xffffff, 1.8);
        key.position.set(-1, 1.5, 2);
        this.camera.add(key);

        this.material = new THREE.MeshStandardMaterial({
            color: SOLID_COLOR, roughness: 0.75, metalness: 0.0, vertexColors: true,
            side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
        });
        this.wireMaterial = new THREE.LineBasicMaterial({ color: WIRE_COLOR, transparent: true, opacity: 0.75 });

        this.grid = new THREE.Group();
        this.scene.add(this.grid);

        this.target = new THREE.Vector3();
        this.spherical = new THREE.Spherical(3, Math.PI / 2.6, Math.PI / 5);
        this.radius = 1;
        this._bindControls();

        this._resizeObserver = new ResizeObserver(() => this.resize());
        this._resizeObserver.observe(container);
        this.resize();
    }

    load(objText, texture) {
        this.clear();
        const obj = parseOBJ(objText);
        this.hasColors = !!obj.colors;
        const data = buildGeometry(obj);
        this.data = data;
        this.texture = texture && data.hasUV ? texture : null;
        if (this.texture) this.texture.colorSpace = THREE.SRGBColorSpace;

        this.mesh = new THREE.Mesh(data.geom, this.material);
        this.wire = new THREE.LineSegments(data.edgeGeom, this.wireMaterial);
        this.scene.add(this.mesh);
        this.scene.add(this.wire);
        if (this.texture) this.opts.mode = "texture";
        else if (this.hasColors) this.opts.mode = "color";
        else this.opts.mode = "solid";
        this._buildGrid();
        this.apply();
        this.frame();
        return data.stats;
    }

    clear() {
        if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh = null; }
        if (this.wire) { this.scene.remove(this.wire); this.wire.geometry.dispose(); this.wire = null; }
        if (this.texture) { this.texture.dispose(); this.texture = null; }
        this.data = null;
        this.requestRender();
    }

    set(opt, value) {
        this.opts[opt] = value;
        this.apply();
    }

    apply() {
        const d = this.data;
        if (!d) return;
        const o = this.opts;
        const geom = d.geom;
        geom.setAttribute("normal", new THREE.BufferAttribute(o.smooth ? d.smoothN : d.flatN, 3));
        const col = geom.getAttribute("color");
        const arr = col.array;
        const useBase = o.mode === "color";
        for (let i = 0; i < arr.length; i++) {
            arr[i] = (useBase ? d.baseCol[i] : 1) * (o.highlight ? d.tint[i] : 1);
        }
        col.needsUpdate = true;
        this.material.map = o.mode === "texture" ? this.texture : null;
        this.material.color.set(o.mode === "solid" ? SOLID_COLOR : 0xffffff);
        this.material.needsUpdate = true;
        this.wire.visible = o.wireframe;
        // Thinner looking wire when the mesh is dense: fade it with the edge count.
        this.wireMaterial.opacity = d.stats.edges > 150000 ? 0.45 : d.stats.edges > 40000 ? 0.6 : 0.8;
        this.wireMaterial.color.set(o.mode === "solid" ? WIRE_COLOR : 0x000000);
        this.requestRender();
    }

    _buildGrid() {
        this.grid.clear();
        const box = this.data.geom.boundingBox;
        const size = new THREE.Vector3();
        box.getSize(size);
        const extent = Math.max(size.x, size.z, size.y) || 1;
        const step = Math.pow(10, Math.floor(Math.log10(extent / 2)));
        const half = Math.ceil(extent * 1.5 / step) * step;
        const divisions = Math.round(half * 2 / step);
        const g = new THREE.GridHelper(half * 2, divisions, 0x4a4a4a, 0x3c3c3c);
        g.position.y = box.min.y;
        this.grid.add(g);
        // Blender axis colors: X red, Y (three.js -Z) green.
        const axis = (a, b, color) => {
            const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
            const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
            line.position.y = box.min.y + extent * 1e-4;
            this.grid.add(line);
        };
        axis(new THREE.Vector3(-half, 0, 0), new THREE.Vector3(half, 0, 0), 0x9b3a3f);
        axis(new THREE.Vector3(0, 0, -half), new THREE.Vector3(0, 0, half), 0x6d8a2b);
    }

    frame() {
        if (!this.data) return;
        const s = this.data.geom.boundingSphere;
        this.target.copy(s.center);
        this.radius = s.radius || 1;
        const fov = THREE.MathUtils.degToRad(this.camera.fov);
        this.spherical.radius = this.radius / Math.sin(fov / 2) * 1.1;
        this._updateCamera();
    }

    view(name) {
        // Numpad-like views: front (1), right (3), top (7).
        const views = { front: [Math.PI / 2, 0], back: [Math.PI / 2, Math.PI], right: [Math.PI / 2, Math.PI / 2],
            left: [Math.PI / 2, -Math.PI / 2], top: [1e-4, 0], bottom: [Math.PI - 1e-4, 0] };
        const v = views[name];
        if (!v) return;
        this.spherical.phi = v[0];
        this.spherical.theta = v[1];
        this._updateCamera();
    }

    _updateCamera() {
        this.spherical.makeSafe();
        const offset = new THREE.Vector3().setFromSpherical(this.spherical);
        this.camera.position.copy(this.target).add(offset);
        this.camera.near = Math.max(this.spherical.radius / 1000, 1e-4);
        this.camera.far = this.spherical.radius * 100 + this.radius * 10;
        this.camera.updateProjectionMatrix();
        this.camera.lookAt(this.target);
        this.requestRender();
    }

    _bindControls() {
        const el = this.canvas;
        let drag = null;
        const stop = (e) => { e.stopPropagation(); };
        el.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
        el.addEventListener("pointerdown", (e) => {
            e.preventDefault();
            e.stopPropagation();
            el.focus();
            el.setPointerCapture(e.pointerId);
            // Blender style: middle drag orbits, Shift+middle pans, Ctrl+middle zooms.
            // Left drag also orbits and right drag pans, for mice without a usable middle button.
            let mode = "orbit";
            if (e.button === 2 || e.shiftKey) mode = "pan";
            else if (e.ctrlKey) mode = "zoom";
            drag = { x: e.clientX, y: e.clientY, mode };
        });
        el.addEventListener("pointermove", (e) => {
            if (!drag) return;
            e.stopPropagation();
            const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
            drag.x = e.clientX; drag.y = e.clientY;
            const h = el.clientHeight || 1;
            if (drag.mode === "orbit") {
                this.spherical.theta -= dx / h * Math.PI * 1.5;
                this.spherical.phi -= dy / h * Math.PI * 1.5;
            } else if (drag.mode === "pan") {
                const dist = this.spherical.radius * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 2;
                const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
                const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
                this.target.addScaledVector(right, -dx / h * dist).addScaledVector(up, dy / h * dist);
            } else {
                this.spherical.radius *= Math.pow(1.01, dy);
            }
            this._updateCamera();
        });
        const end = (e) => {
            if (!drag) return;
            e.stopPropagation();
            drag = null;
            try { el.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
        };
        el.addEventListener("pointerup", end);
        el.addEventListener("pointercancel", end);
        el.addEventListener("wheel", (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.spherical.radius *= Math.pow(1.0015, e.deltaY);
            this.spherical.radius = Math.min(Math.max(this.spherical.radius, this.radius * 0.02), this.radius * 200);
            this._updateCamera();
        }, { passive: false });
        el.addEventListener("keydown", (e) => {
            const k = e.code;
            const map = { Numpad1: e.ctrlKey ? "back" : "front", Numpad3: e.ctrlKey ? "left" : "right",
                Numpad7: e.ctrlKey ? "bottom" : "top" };
            if (map[k]) { this.view(map[k]); }
            else if (k === "NumpadDecimal" || k === "Home" || k === "KeyF") { this.frame(); }
            else return;
            e.preventDefault();
            e.stopPropagation();
        });
        el.addEventListener("pointerenter", stop);
    }

    resize() {
        const w = Math.max(this.container.clientWidth, 1);
        const h = Math.max(this.container.clientHeight, 1);
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.requestRender();
    }

    requestRender() {
        if (this._pending) return;
        this._pending = true;
        requestAnimationFrame(() => {
            this._pending = false;
            this.renderer.render(this.scene, this.camera);
        });
    }

    dispose() {
        this._resizeObserver.disconnect();
        this.clear();
        this.material.dispose();
        this.wireMaterial.dispose();
        this.renderer.dispose();
        this.canvas.remove();
    }
}

// Loads an OBJ (and the base color texture its MTL points to) from ComfyUI's /view.
// urlFor(filename) returns the /view URL for a file next to the OBJ.
export async function loadQuadOBJ(viewer, objName, urlFor) {
    const resp = await fetch(urlFor(objName));
    if (!resp.ok) throw new Error(`${objName}: HTTP ${resp.status}`);
    const text = await resp.text();
    let texture = null;
    const m = text.match(/^mtllib\s+(.+)$/m);
    if (m) {
        try {
            const mtl = await fetch(urlFor(m[1].trim()));
            if (mtl.ok) {
                const map = parseMTL(await mtl.text());
                if (map) texture = await new THREE.TextureLoader().loadAsync(urlFor(map));
                if (texture) texture.flipY = true;
            }
        } catch (err) {
            console.warn("Preview Quad Mesh: texture not loaded", err);
        }
    }
    return viewer.load(text, texture);
}
