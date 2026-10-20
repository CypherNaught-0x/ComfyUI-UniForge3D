"""Checks of the ROCm replacements against independent references.

Run inside a ComfyUI Python environment on a GPU:
    PYTHONPATH=/path/to/ComfyUI:/path/to/ComfyUI-UniForge3D python -m unittest -v tests.test_compat
"""

import math
import unittest

import torch
import torch.nn.functional as F

from uniforge3d.compat import cumesh
from uniforge3d.compat.grid_sample import grid_sample_3d
from uniforge3d.compat.lora import merge_peft_state_dict
from uniforge3d.compat.rasterize import interpolate, rasterize_uv
from uniforge3d.models.lora import LoRALinear
from uniforge3d.modules import sparse as sp
from uniforge3d.modules.sparse.attention.full_attn import varlen_sdpa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def icosphere(subdivisions=3):
    t = (1 + 5 ** 0.5) / 2
    v = torch.tensor([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                      [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=torch.float32)
    f = torch.tensor([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                      [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                      [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(subdivisions):
        edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).sort(1).values
        unique, inverse = torch.unique(edges, dim=0, return_inverse=True)
        mid = v.shape[0] + inverse.view(3, -1).T
        v = torch.cat([v, (v[unique[:, 0]] + v[unique[:, 1]]) / 2])
        a, b, c = f.T
        ab, bc, ca = mid.T
        f = torch.stack([torch.stack([a, ab, ca], 1), torch.stack([b, bc, ab], 1),
                         torch.stack([c, ca, bc], 1), torch.stack([ab, bc, ca], 1)], 1).reshape(-1, 3)
    return F.normalize(v, dim=1) * 0.4, f.int()


class SparseConvTest(unittest.TestCase):
    def test_matches_dense_conv3d_at_active_sites(self):
        torch.manual_seed(0)
        R, Ci, Co = 12, 8, 5
        occupied = torch.rand(R, R, R) < 0.3
        coords = torch.argwhere(occupied).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1).to(DEVICE)
        feats = torch.randn(coords.shape[0], Ci, device=DEVICE)
        conv = sp.SparseConv3d(Ci, Co, 3).to(DEVICE)
        out = conv(sp.SparseTensor(feats=feats, coords=coords)).feats

        dense = torch.zeros(1, Ci, R, R, R, device=DEVICE)
        dense[0, :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()] = feats.T
        weight = conv.weight.permute(0, 4, 1, 2, 3)        # back to (Co, Ci, Kd, Kh, Kw)
        reference = F.conv3d(dense, weight, conv.bias, padding=1)
        reference = reference[0, :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()].T
        torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)


class GridSampleTest(unittest.TestCase):
    @staticmethod
    def kernel_reference(feats, coords, grid):
        """Literal transcription of FlexGEMM's CUDA trilinear kernel."""
        table = {tuple(c.tolist()): i for i, c in enumerate(coords[:, 1:].cpu())}
        out = []
        for q in grid[0].cpu().tolist():
            base = [math.floor(x - 0.5) for x in q]
            acc, total = torch.zeros(feats.shape[1]), 0.0
            for i in range(8):
                c = (base[0] + (i & 1), base[1] + ((i >> 1) & 1), base[2] + ((i >> 2) & 1))
                if c in table:
                    w = math.prod(1 - abs(q[k] - c[k] - 0.5) for k in range(3))
                    acc += w * feats[table[c]].cpu()
                    total += w
            out.append(acc / max(total, 1e-12))
        return torch.stack(out)

    def test_trilinear_matches_cuda_kernel_semantics(self):
        torch.manual_seed(1)
        R = 10
        coords = torch.argwhere(torch.rand(R, R, R) < 0.5).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1).to(DEVICE)
        feats = torch.randn(coords.shape[0], 6, device=DEVICE)
        grid = (torch.rand(1, 500, 3, device=DEVICE) * (R + 1)) - 0.5   # includes out-of-range queries
        out = grid_sample_3d(feats, coords, torch.Size([1, 6, R, R, R]), grid)[0]
        torch.testing.assert_close(out.cpu(), self.kernel_reference(feats, coords, grid), atol=1e-5, rtol=1e-5)

    def test_dense_grid_equals_torch_grid_sample(self):
        torch.manual_seed(2)
        R = 8
        volume = torch.randn(1, 3, R, R, R, device=DEVICE)
        coords = torch.argwhere(torch.ones(R, R, R)).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1).to(DEVICE)
        feats = volume[0, :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()].T
        grid = 0.5 + torch.rand(1, 300, 3, device=DEVICE) * (R - 1)   # interior: all 8 neighbours exist
        out = grid_sample_3d(feats, coords, torch.Size([1, 3, R, R, R]), grid)[0]
        normalized = (grid / R) * 2 - 1                                 # voxel centres at i + 0.5
        reference = F.grid_sample(volume, normalized.flip(-1).view(1, 1, 1, -1, 3), align_corners=False)
        torch.testing.assert_close(out, reference.view(3, -1).T, atol=1e-5, rtol=1e-5)


class LoraTest(unittest.TestCase):
    def test_merged_equals_unmerged(self):
        torch.manual_seed(3)
        base = torch.nn.Linear(16, 12)
        lora = LoRALinear(base, rank=4, alpha=8)
        torch.nn.init.normal_(lora.lora_A)
        torch.nn.init.normal_(lora.lora_B)
        state = {
            "base_model.model.layer.base_layer.weight": base.weight.detach(),
            "base_model.model.layer.base_layer.bias": base.bias.detach(),
            "base_model.model.layer.lora_A.default.weight": lora.lora_A.detach(),
            "base_model.model.layer.lora_B.default.weight": lora.lora_B.detach(),
        }
        merged = merge_peft_state_dict(state, r=4, lora_alpha=8)
        plain = torch.nn.Linear(16, 12)
        plain.load_state_dict({k.removeprefix("layer."): v for k, v in merged.items()})
        x = torch.randn(5, 16)
        torch.testing.assert_close(plain(x), lora(x), atol=1e-4, rtol=1e-4)


class AttentionTest(unittest.TestCase):
    def test_varlen_sdpa_matches_per_sequence_softmax(self):
        torch.manual_seed(4)
        q_lens, kv_lens = [7, 30, 1], [11, 5, 40]
        q = torch.randn(sum(q_lens), 4, 16, device=DEVICE)
        k = torch.randn(sum(kv_lens), 4, 16, device=DEVICE)
        v = torch.randn(sum(kv_lens), 4, 16, device=DEVICE)
        out = varlen_sdpa(q, k, v, q_lens, kv_lens)
        qs, ks = 0, 0
        for ql, kl in zip(q_lens, kv_lens):
            att = torch.einsum("qhc,khc->hqk", q[qs:qs + ql], k[ks:ks + kl]) / 4.0
            ref = torch.einsum("hqk,khc->qhc", att.softmax(-1), v[ks:ks + kl])
            torch.testing.assert_close(out[qs:qs + ql], ref, atol=1e-4, rtol=1e-4)
            qs, ks = qs + ql, ks + kl


class MeshOpsTest(unittest.TestCase):
    def setUp(self):
        v, f = icosphere(4)
        self.vertices, self.faces = v.to(DEVICE), f.to(DEVICE)

    def test_simplify_reaches_target(self):
        mesh = cumesh.CuMesh()
        mesh.init(self.vertices, self.faces)
        mesh.simplify(1000)
        self.assertLessEqual(mesh.num_faces, 1100)
        self.assertGreater(mesh.num_faces, 500)

    def test_fill_holes_closes_small_hole(self):
        fan = (self.faces == 0).any(dim=1)          # faces around vertex 0 -> one simple loop
        mesh = cumesh.CuMesh()
        mesh.init(self.vertices, self.faces)
        mesh.remove_faces(fan)
        holed = mesh.num_faces
        mesh.fill_holes(max_hole_perimeter=0.5)
        self.assertGreater(mesh.num_faces, holed)
        # Closed again: every edge is shared by exactly two faces.
        f = mesh.faces.long()
        edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).sort(1).values
        _, counts = torch.unique(edges, dim=0, return_counts=True)
        self.assertTrue(bool((counts == 2).all()))

    def test_bvh_distance_and_barycentrics(self):
        bvh = cumesh.cuBVH(self.vertices, self.faces)
        points = torch.randn(200, 3, device=DEVICE) * 0.3
        distance, face, uvw = bvh.unsigned_distance(points, return_uvw=True)
        tri = self.vertices[self.faces.long()]
        brute = torch.stack([_point_triangle_distance(p, tri) for p in points])
        torch.testing.assert_close(distance, brute, atol=1e-4, rtol=1e-4)
        rebuilt = (tri[face] * uvw[..., None]).sum(1)
        torch.testing.assert_close((rebuilt - points).norm(dim=-1), distance, atol=1e-4, rtol=1e-4)

    def test_remesh_returns_closed_surface_near_input(self):
        center = torch.zeros(3, device=DEVICE)
        v, f = cumesh.remeshing.remesh_narrow_band_dc(self.vertices, self.faces, center, 1.0, 128, band=1)
        self.assertGreater(f.shape[0], 1000)
        radius = v.norm(dim=1)
        # UDF band at 1 voxel: shells sit within ~1 voxel of the 0.4 sphere.
        self.assertLess((radius - 0.4).abs().max().item(), 2.5 / 128)

    def test_uv_unwrap_and_rasterize_reproduce_positions(self):
        mesh = cumesh.CuMesh()
        mesh.init(self.vertices, self.faces)
        v, f, uv, vmap = mesh.uv_unwrap(return_vmaps=True, uv_resolution=512)
        self.assertTrue(torch.equal(v, self.vertices[vmap]))
        self.assertTrue(((uv >= 0) & (uv <= 1)).all())
        face_index, bary, mask = rasterize_uv(uv, f, 512)
        self.assertGreater(mask.float().mean().item(), 0.2)
        pos = interpolate(v, f, face_index, bary, mask)[mask]
        # Interpolated texel positions lie on the (flat-faceted) sphere.
        self.assertLess((pos.norm(dim=-1) - 0.4).abs().max().item(), 0.01)


def _point_triangle_distance(p, tri):
    from comfy_extras.mesh3d.postprocess.remesh import _point_tri_closest
    closest, _ = _point_tri_closest(p.expand(tri.shape[0], 3), tri)
    return (closest - p).norm(dim=-1).min()


if __name__ == "__main__":
    unittest.main()
