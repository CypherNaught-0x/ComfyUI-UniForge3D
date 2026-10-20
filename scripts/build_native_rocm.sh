#!/usr/bin/env bash
# Optional: build FlexGEMM and CuMesh for ROCm into a wheel directory. UniForge3D picks them up
# automatically when importable (UNIFORGE3D_NATIVE_EXT=0 disables). Not needed for correctness.
#
#   scripts/build_native_rocm.sh <out_dir> [gfx_arch]      e.g. scripts/build_native_rocm.sh ./wheels gfx1151
#
# Run inside the Python environment ComfyUI uses (same torch). Tested with torch 2.13 + ROCm 10.1
# "TheRock" pip SDK, which splits the toolkit across _rocm_sdk_core and _rocm_sdk_devel; the
# script merges them into one ROCM_HOME. With a classic /opt/rocm install, set ROCM_HOME yourself.
set -euo pipefail
OUT=$(realpath -m "${1:?usage: $0 <out_dir> [gfx_arch]}")
ARCH=${2:-gfx1151}
HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

if [ -z "${ROCM_HOME:-}" ] && command -v rocm-sdk >/dev/null; then
  rocm-sdk path --root >/dev/null   # unpacks the devel SDK on first use
  SITE=$(python3 -c "import _rocm_sdk_core, os; print(os.path.dirname(os.path.dirname(_rocm_sdk_core.__file__)))")
  mkdir -p "$WORK/rocm"
  cp -rs "$SITE/_rocm_sdk_core/." "$WORK/rocm/"
  cp -rsn "$SITE/_rocm_sdk_devel/." "$WORK/rocm/" 2>/dev/null || true
  export ROCM_HOME="$WORK/rocm" CPATH="$SITE/_rocm_sdk_devel/include${CPATH:+:$CPATH}"
fi
export ROCM_PATH="$ROCM_HOME" HIP_PATH="$ROCM_HOME" PYTORCH_ROCM_ARCH="$ARCH" GPU_ARCHS="$ARCH"
export HIP_DEVICE_LIB_PATH="$ROCM_HOME/lib/llvm/amdgcn/bitcode"
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=$ARCH" MAX_JOBS=${MAX_JOBS:-8}

git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$WORK/FlexGEMM"
git -C "$WORK/FlexGEMM" checkout -q 6dd94a859c26ee8246888502eada3dd8ad85532e
git -C "$WORK/FlexGEMM" submodule update -q --init --recursive

git clone -q --recursive https://github.com/JeffreyXiang/CuMesh.git "$WORK/CuMesh"
git -C "$WORK/CuMesh" checkout -q 12289e1062f0603f2f0d0771b02e1395d247f26f
git -C "$WORK/CuMesh" submodule update -q --init --recursive
git -C "$WORK/CuMesh" apply "$HERE/patches/cumesh-hip.patch"

for pkg in FlexGEMM CuMesh; do
  echo "Building $pkg for $ARCH ..."
  python3 -m pip wheel --no-build-isolation --no-deps -w "$OUT" "$WORK/$pkg"
done
echo "Wheels in $OUT. Install with: python3 -m pip install --no-deps $OUT/*.whl"
