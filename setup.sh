#!/usr/bin/env bash
# setup.sh — one-shot bootstrap for the gsplat_tt repository.
#
# Idempotent. Safe to re-run; each step skips work that's already done.
#
# Steps:
#   1. Project venv at ./venv with requirements.txt + `pip install -e .`
#   2. Vendor tt-metal into backends/tt/tt-metal/ (~5 GB clone)
#   3. Register our kernel subdir in tt-metal's programming_examples CMake
#   4. Build tt-metal C++ libs + our kernel host binary
#      (requires sudo for SFPI / .cpmcache; ~10-20 min on first run)
#
# Usage:
#   ./setup.sh
#   TT_METAL_REF=v1.2.3 ./setup.sh    # pin a specific tt-metal tag
#
# After this finishes the kernel viewer runs as:
#   source venv/bin/activate
#   export TT_METAL_HOME=$PWD/backends/tt/tt-metal
#   export TT_METAL_RUNTIME_ROOT=$PWD/backends/tt/tt-metal
#   gsplat scenes/luigi.ply --backend tt
#
# (CPU viewer needs only `source venv/bin/activate && gsplat scenes/luigi.ply`.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TT_METAL_DIR="backends/tt/tt-metal"
TT_METAL_REPO="https://github.com/tenstorrent/tt-metal.git"
TT_METAL_REF="${TT_METAL_REF:-main}"

say()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
warn() { printf "\033[1;33m!! %s\033[0m\n" "$*" >&2; }
fail() { printf "\033[1;31m!! %s\033[0m\n" "$*" >&2; exit 1; }

# ----- Pre-flight ------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || fail "python3 not found on PATH"
command -v git     >/dev/null 2>&1 || fail "git not found on PATH"
command -v sudo    >/dev/null 2>&1 || fail "sudo not found (build_metal.sh needs root for SFPI cache)"

# ----- 1. Project venv -------------------------------------------------------
if [ ! -d "venv" ]; then
    say "Creating project venv at ./venv"
    python3 -m venv venv
fi

say "Installing project requirements + editable package"
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt
pip install -e .
deactivate

# ----- 2. Vendor tt-metal ----------------------------------------------------
# Our repo tracks the kernel subdir at
#   $TT_METAL_DIR/tt_metal/programming_examples/gaussian_splatting/
# so $TT_METAL_DIR is *partially populated* on a fresh clone (just the kernel
# subdir, no upstream tt-metal). Detect "needs vendoring" by looking for an
# upstream-only marker file (build_metal.sh) rather than the directory.
# `git clone` can't target a non-empty dir, so we clone to a temp location and
# merge upstream files in with `cp -rn` (no-clobber preserves our tracked
# kernel subdir).
mkdir -p "$TT_METAL_DIR"
if [ ! -f "$TT_METAL_DIR/build_metal.sh" ]; then
    say "Vendoring tenstorrent/tt-metal into $TT_METAL_DIR (~5 GB, may take several minutes)"
    tmp_clone=$(mktemp -d -t gsplat_tt_setup_XXXXXX)
    trap 'rm -rf "$tmp_clone"' EXIT
    git clone --recurse-submodules --branch "$TT_METAL_REF" \
        "$TT_METAL_REPO" "$tmp_clone/tt-metal"
    rm -rf "$tmp_clone/tt-metal/.git"
    cp -rn "$tmp_clone/tt-metal/." "$TT_METAL_DIR/"
    rm -rf "$tmp_clone"
    trap - EXIT
else
    say "$TT_METAL_DIR already vendored, skipping clone"
fi

# Drop embedded .git so the parent repo can track files inside the vendored
# tree (the .gitignore allow-lists our kernel subdir; rest stays untracked).
if [ -d "$TT_METAL_DIR/.git" ]; then
    say "Removing embedded $TT_METAL_DIR/.git"
    rm -rf "$TT_METAL_DIR/.git"
fi

# ----- 3. Register the in-process ttnn op (alpha_blend) in the ttnn build ----
# (The old standalone daemon programming-example is gone; the kernels now live
# in the ttnn op below. Nothing to register under programming_examples.)

# ----- 3b. Register the in-process ttnn op (alpha_blend) in the ttnn build ---
# Our op subtree under ttnn/cpp/.../experimental/gaussian_splatting/ is tracked
# by git, but the upstream files that REFERENCE it (ttnn/CMakeLists.txt and
# experimental_nanobind.cpp) are not — so re-inject those edits idempotently on
# every (re)vendor, mirroring the programming-examples registration above.
TTNN_CMAKE="$TT_METAL_DIR/ttnn/CMakeLists.txt"
EXP_NB="$TT_METAL_DIR/ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp"
OP_SUBDIR="cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend"
OP_LINK="TTNN::Ops::Experimental::GaussianSplatting::AlphaBlend"
if [ -f "$TTNN_CMAKE" ]; then
    if ! grep -qF "add_subdirectory($OP_SUBDIR)" "$TTNN_CMAKE"; then
        say "Registering alpha_blend op add_subdirectory in $TTNN_CMAKE"
        # Insert right after the post_combine_reduce add_subdirectory line.
        python3 - "$TTNN_CMAKE" "$OP_SUBDIR" <<'PY'
import sys
path, subdir = sys.argv[1], sys.argv[2]
s = open(path).read()
anchor = "add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce)\n"
ins = f"add_subdirectory({subdir})\n"
s = s.replace(anchor, anchor + ins, 1) if anchor in s else s + "\n" + ins
open(path, "w").write(s)
PY
    fi
    if ! grep -qF "$OP_LINK" "$TTNN_CMAKE"; then
        say "Linking $OP_LINK into the ttnn target"
        python3 - "$TTNN_CMAKE" "$OP_LINK" <<'PY'
import sys
path, lib = sys.argv[1], sys.argv[2]
s = open(path).read()
anchor = "        TTNN::Ops::Experimental::DeepSeekPrefill::PostCombineReduce\n"
ins = f"        {lib}\n"
s = s.replace(anchor, anchor + ins, 1) if anchor in s else s
open(path, "w").write(s)
PY
    fi
fi
# The op's nanobind .cpp is compiled into the ttnn python module via the
# aggregated nanobind source list in ttnn/sources.cmake.
TTNN_SRCS="$TT_METAL_DIR/ttnn/sources.cmake"
OP_NB_SRC="cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend/alpha_blend_nanobind.cpp"
if [ -f "$TTNN_SRCS" ] && ! grep -qF "$OP_NB_SRC" "$TTNN_SRCS"; then
    say "Registering alpha_blend nanobind source in $TTNN_SRCS"
    python3 - "$TTNN_SRCS" "$OP_NB_SRC" <<'PY'
import sys
path, src = sys.argv[1], sys.argv[2]
s = open(path).read()
anchor = "    cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/post_combine_reduce_nanobind.cpp\n"
ins = f"    {src}\n"
s = s.replace(anchor, anchor + ins, 1) if anchor in s else s
open(path, "w").write(s)
PY
fi
if [ -f "$EXP_NB" ] && ! grep -qF "gaussian_splatting/alpha_blend/alpha_blend_nanobind.hpp" "$EXP_NB"; then
    say "Wiring alpha_blend nanobind into $EXP_NB"
    python3 - "$EXP_NB" <<'PY'
import sys
path = sys.argv[1]
s = open(path).read()
inc_anchor = '#include "ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/post_combine_reduce_nanobind.hpp"\n'
inc = '#include "ttnn/operations/experimental/gaussian_splatting/alpha_blend/alpha_blend_nanobind.hpp"\n'
if inc_anchor in s:
    s = s.replace(inc_anchor, inc_anchor + inc, 1)
call_anchor = '    deepseek_prefill::detail::bind_post_combine_reduce(mod);\n'
call = '    gaussian_splatting::alpha_blend::detail::bind_gaussian_alpha_blend(mod);\n'
if call_anchor in s:
    s = s.replace(call_anchor, call_anchor + call, 1)
open(path, "w").write(s)
PY
fi

# ----- 4. Build tt-metal + ttnn (incl. our op) ------------------------------
# We no longer pass --without-python-bindings, so build_metal.sh builds the
# ttnn Python extensions (_ttnn.so) including our gaussian_splatting/alpha_blend
# op. The in-process TT backend imports ttnn directly (no daemon subprocess), so
# the ttnn wheel is installed into ./venv afterward via `pip install -e`.
# tt-metal's create_venv.sh / python_env is intentionally NOT used — one
# interpreter (./venv) runs everything.
#   --build-programming-examples is kept only because build_metal.sh's other
#   paths expect it; we build no programming-example of our own anymore.
say "Building tt-metal + ttnn (Python bindings ON; sudo required for SFPI / CPM caches)"
warn "First build can take 10-20 minutes."
pushd "$TT_METAL_DIR" >/dev/null
sudo ./build_metal.sh --build-programming-examples
popd >/dev/null

# Install the freshly built ttnn editable wheel into the PROJECT venv (./venv),
# not tt-metal's python_env. ttnn's editable wheel writes a .pth so `import ttnn`
# resolves to the vendored tree + built .so files. --no-build-isolation lets the
# install see the build artifacts and our already-installed setuptools/wheel.
say "Installing ttnn into ./venv"
# shellcheck disable=SC1091
source venv/bin/activate
pip install --no-build-isolation -e "$TT_METAL_DIR"
deactivate

# ----- Done ------------------------------------------------------------------
say "Setup complete."
cat <<EOF

  CPU viewer:
    source venv/bin/activate
    gsplat scenes/luigi.ply

  TT viewer (Tenstorrent Wormhole):
    source venv/bin/activate
    export TT_METAL_HOME=\$PWD/${TT_METAL_DIR}
    export TT_METAL_RUNTIME_ROOT=\$PWD/${TT_METAL_DIR}
    gsplat scenes/luigi.ply --backend tt

EOF
