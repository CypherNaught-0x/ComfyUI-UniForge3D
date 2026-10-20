"""Regenerate the example workflows: python workflows/build_example.py"""

import json
from pathlib import Path

HERE = Path(__file__).parent
VIEWS = ["uniforge3d/001.png", "uniforge3d/002.png", "uniforge3d/003.png", "uniforge3d/004.png"]

# (id, type, pos, widgets, inputs [(name, type)], outputs [(name, type)])
nodes = [
    *[(i + 1, "LoadImage", [40, 40 + 330 * i], [path, "image"], [], [("IMAGE", "IMAGE"), ("MASK", "MASK")])
      for i, path in enumerate(VIEWS)],
    (5, "UniForge3DViews", [420, 300], ["inverted (LoadImage alpha)"],
     [("previous", "UNIFORGE3D_VIEWS")] + [p for i in range(1, 7) for p in ((f"image_{i}", "IMAGE"), (f"mask_{i}", "MASK"))],
     [("UNIFORGE3D_VIEWS", "UNIFORGE3D_VIEWS")]),
    (6, "CLIPVisionLoader", [420, 40], ["dino_v3_L_naf_fp32.safetensors"], [], [("CLIP_VISION", "CLIP_VISION")]),
    (7, "UniForge3DLoader", [760, 40], ["UniForge3D", False], [("clip_vision", "CLIP_VISION")], [("UNIFORGE3D_PIPELINE", "UNIFORGE3D_PIPELINE")]),
    (8, "UniForge3DGenerate", [1100, 200], ["textured", 42, "fixed", 2048, 1000000, 0, 0, 0, False],
     [("pipeline", "UNIFORGE3D_PIPELINE"), ("views", "UNIFORGE3D_VIEWS")], [("MESH", "MESH")]),
    (9, "SaveGLB", [1460, 200], ["3d/uniforge3d"], [("mesh", "MESH")], []),
]
# (from_node, from_slot, to_node, to_input_name)
edges = [(i, 0, 5, f"image_{i}") for i in range(1, 5)] + [(i, 1, 5, f"mask_{i}") for i in range(1, 5)] + [
    (6, 0, 7, "clip_vision"), (7, 0, 8, "pipeline"), (5, 0, 8, "views"), (8, 0, 9, "mesh"),
]

links, ui_nodes = [], {}
for nid, ntype, pos, widgets, inputs, outputs in nodes:
    ui_nodes[nid] = {
        "id": nid, "type": ntype, "pos": pos, "size": [300, 120], "flags": {}, "order": nid - 1, "mode": 0,
        "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
        "outputs": [{"name": n, "type": t, "links": [], "slot_index": s} for s, (n, t) in enumerate(outputs)],
        "properties": {"Node name for S&R": ntype}, "widgets_values": widgets,
    }
for link_id, (src, slot, dst, name) in enumerate(edges, start=1):
    target = ui_nodes[dst]
    index = next(i for i, x in enumerate(target["inputs"]) if x["name"] == name)
    target["inputs"][index]["link"] = link_id
    ui_nodes[src]["outputs"][slot]["links"].append(link_id)
    links.append([link_id, src, slot, dst, index, target["inputs"][index]["type"]])

ui = {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": list(ui_nodes.values()), "links": links,
      "groups": [], "config": {}, "extra": {}, "version": 0.4}
(HERE / "uniforge3d_multiview.json").write_text(json.dumps(ui, indent=1))

widget_names = {
    "LoadImage": ["image"], "UniForge3DViews": ["mask_mode"], "CLIPVisionLoader": ["clip_name"],
    "UniForge3DLoader": ["model", "keep_models_in_ram"], "SaveGLB": ["filename_prefix"],
    "UniForge3DGenerate": ["output", "seed", None, "texture_size", "decimation_target",
                           "structure_steps", "shape_steps", "texture_steps", "drop_inner_shell"],
}
api = {}
for nid, ntype, _, widgets, _, _ in nodes:
    inputs = {name: value for name, value in zip(widget_names[ntype], widgets) if name}
    api[str(nid)] = {"class_type": ntype, "inputs": inputs}
for src, slot, dst, name in edges:
    api[str(dst)]["inputs"][name] = [str(src), slot]
(HERE / "uniforge3d_multiview.api.json").write_text(json.dumps(api, indent=1))
print("wrote", len(nodes), "nodes,", len(links), "links")
