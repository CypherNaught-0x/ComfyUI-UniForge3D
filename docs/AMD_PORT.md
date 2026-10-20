# Porting UniForge3D to AMD ROCm

The target was a Radeon 8060S (Strix Halo APU, gfx1151) with PyTorch 2.13.0 + ROCm 10.1 (TheRock pip SDK), running the
`comfyui-gfx1151:rocm101` image. All numbers below were measured there.

## What upstream needs and what replaces it

| Upstream dependency | Used by UniForge3D for | Replacement here (default) | Natively built for ROCm |
|---|---|---|---|
| **FlexGEMM** | submanifold sparse conv in every sparse VAE; trilinear sampling of the PBR voxel field while baking | `conv_torch` backend (ComfyUI's gather + GEMM conv); a port of the FlexGEMM CUDA trilinear kernel | **Builds unmodified** (upstream has a HIP target). 2–17× faster, results match |
| **o-voxel** | `flexible_dual_grid_to_mesh` in the shape decoder | ComfyUI's port of the same function | Builds unmodified. The mesh is identical and the speed is the same, because that function is plain PyTorch upstream too, so there is nothing to gain |
| **CuMesh** | simplify, hole filling, narrow-band DC remesh, BVH closest point, UV unwrap, vertex normals | a facade over ComfyUI's pure-torch mesh library | **Builds with a 15-line patch** (`patches/cumesh-hip.patch`). 4–5× faster |
| **nvdiffrast** | UV-space rasterization + `interpolate` for texture baking | ComfyUI's tiled point-in-triangle UV rasterizer | Does not build. This would be a real port (see below), and baking does not need it |
| flash-attn | sparse / routed varlen attention | per-sequence PyTorch SDPA | The installed Triton-AMD flash-attn works, but it is **1.7× slower** than SDPA |
| peft | LoRA wrapping of the stage-2/3 denoisers | merge `W += (α/r)·B·A` at load time | n/a |
| easydict | config access | 20-line `AttrDict` | n/a |
| DINOv3 (gated HF repo) | image features | ComfyUI's DINOv3 ViT-L (`dino_v3_L_naf_fp32`), the one TRELLIS.2/Pixal3D already use | n/a |
| BRIA RMBG-2.0 | background removal | bring your own masks (RGBA, or any ComfyUI background-removal node) | n/a |

The native `flex_gemm` / `cumesh` builds are **opt-in**:

- `UNIFORGE3D_NATIVE_EXT=auto` uses whichever of them can be imported.
- `UNIFORGE3D_NATIVE_EXT=1` requires both.
- The default is the pure-PyTorch path.

## Correctness checks (`tests/test_compat.py`, all on GPU)

- **Sparse conv** equals dense `F.conv3d` evaluated at the active sites, to 1e-4. Weight layout and neighbour order match
  the released checkpoints.
- **Grid sample** equals a literal transcription of FlexGEMM's CUDA kernel (corner `floor(q-0.5)`, renormalised weights,
  zeros when nothing is occupied). On a fully occupied grid it equals `F.grid_sample`.
- **LoRA merge** equals the unmerged LoRA forward.
- **Varlen SDPA** equals an explicit softmax for each sequence.
- **BVH** distances equal brute force, and the reconstructed barycentric points equal the distances.
- **Hole filling** closes the mesh again: every edge ends up shared by two faces.
- **Remesh** stays within one voxel of the surface.
- **UV unwrap + rasterization** reproduces surface positions.

End to end, upstream `case01` (4 photos) gives a coherent textured asset. See `docs/case01.png`.

## Native builds on ROCm

PyTorch's `CUDAExtension` hipifies the sources automatically, and FlexGEMM, CuMesh and o-voxel already have HIP code
paths in their `setup.py` (`IS_HIP_EXTENSION`). Almost all of the friction was toolchain setup:

1. TheRock pip wheels split the SDK. `hipcc`, the HIP headers and the device bitcode are in `_rocm_sdk_core`. The library
   headers (rocThrust, hipCUB, hipSPARSE) are in `_rocm_sdk_devel`, which is only unpacked after `rocm-sdk path --root`
   runs. PyTorch's `ROCM_HOME` points at core only. `scripts/build_native_rocm.sh` merges the two.
2. `GPU_ARCHS` defaults to `native`, which fails in a build container that has no GPU. Pass `gfx1151` (or your arch).

After that, the code-level results were:

- **FlexGEMM** (Triton GEMM kernels + CUDA hashmap kernels) and **o-voxel** compiled with no source changes.
- **CuMesh** had 3 errors, all in one file plus cubvh's build flags:
  - libcu++ `::cuda::std::tuple` in a radix-sort decomposer. The fix is `::rocprim::tuple`.
  - `Vec3f` constructors were `__device__`-only, but hipCUB's `DeviceSegmentedReduce` builds the initial value on the
    host. The fix is `__host__ __device__`.
  - The nvcc-only flag `--extended-lambda` was passed unconditionally to the bundled cubvh. HIP needs no flag.
- **nvdiffrast** (v0.4.0) needs real porting work. The CudaRaster core is warp-synchronous code written for 32-wide
  warps:
  - about 135 `__ballot_sync` / `__any_sync` / `__syncwarp` calls with 32-bit masks (HIP requires 64-bit mask types)
  - 26 `__popc`
  - 54 inline-PTX `asm` blocks, all SIMD-video arithmetic (`vadd/vsub/vmad/vmin/vmax`) in one file, `Util.inl`
  - `__frcp_rz` and friends
  - a few non-hipified CUDA header includes

  gfx1151 compiles HIP as wave32 by default, which matches the warp model, so this is a moderate port: rewrite
  `Util.inl` in portable C++, widen the masks, and replace a few intrinsics. I estimate 1–3 days including validation.
  It is not on UniForge3D's critical path, since only UV-space rasterization is needed. ModernGL/EGL (which already runs
  UltraTex in this container) is the hardware-raster alternative if baking speed matters.

### Native vs pure-PyTorch on gfx1151

| Operation | Pure PyTorch | Native HIP | Agreement |
|---|---|---|---|
| sparse conv 64³, 6k voxels, C=1024 (fp16) | 31.9 ms | 15.3 ms | rel. max err 6.5e-4 |
| sparse conv 256³, 100k voxels, C=128 | 19.9 ms | 2.7 ms | 6.1e-4 |
| sparse conv 512³, 405k voxels, C=64 | 55.0 ms | 6.7 ms | 3.9e-4 |
| grid sample, 400k queries | 13.7 ms | 0.8 ms | 7e-7 abs |
| `flexible_dual_grid_to_mesh` (o-voxel) | 1.2 ms | 0.9 ms | identical mesh |
| CuMesh simplify 327k → 50k faces | 941 ms | 180 ms | 50.0k vs 46.7k faces |
| CuMesh remesh, resolution 512 | 1380 ms | 314 ms | identical face count |

The per-op wins do not carry over to the whole pipeline. These are textured runs on `case01`:

| Path | Total | Structure | Shape 512 | Shape 1024 | PBR texture |
|---|---|---|---|---|---|
| Pure PyTorch (default) | **450 s** | 51 s | 66 s | 140 s | 191 s |
| Native, cold (Triton autotune) | 544 s | 47 s | 76 s | 153 s | 264 s |
| Native, warm (autotune cached) | 459 s | 48 s | 52 s | 153 s | 202 s |

Time goes to the DiT samplers (dense GEMMs and attention), which no extension touches. On the native path, the texture
stage also runs CuMesh's xatlas UV packing on the CPU. The point of the native builds is therefore exact upstream
mesh processing, not speed.

## Coexistence with TRELLIS.2 / Pixal3D

The node pack patches nothing in ComfyUI or in other packs. It adds 3 nodes and imports only as a nested package, so no
top-level `uniforge3d` module ends up in `sys.modules`.

Before the pack existed, both native pipelines already had a property that matters here. Their post-processed meshes
(remesh/decimate) differ from run to run even with identical inputs, because the GPU work is non-deterministic. Their
raw meshes are reproducible.

That reproducibility exposed one interaction. UniForge3D seeds the global torch RNG, and Pixal3D's raw mesh depends on
global RNG state. So running UniForge3D before Pixal3D in the same process changed Pixal3D's output. The node now runs
inside `torch.random.fork_rng`, which leaves the caller's RNG state untouched.

All checks ran on throwaway ComfyUI servers using the production image, models and custom-node set, each with private
input and output directories.

- **Startup:** the pack adds exactly its 3 nodes (1546 → 1549). All 10 TRELLIS.2/Pixal3D nodes are present, and the
  import-failure list is unchanged.
- **Raw meshes:** these are the reproducible outputs. Sequence: UniForge3D, then TRELLIS.2 512, then Pixal3D 512, in one
  process. Both raw GLBs are **byte-identical** to the runs without the pack. Before the `fork_rng` fix, Pixal3D's
  differed.
- **Run time:** TRELLIS.2 512 took 50–55 s and Pixal3D 512 took 55 s, with or without the pack.
