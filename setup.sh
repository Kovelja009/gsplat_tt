#!/usr/bin/env bash
# setup.sh — one-shot bootstrap for the gsplat_tt repository.
#
# Idempotent. Safe to re-run; each step skips work that's already done.
#
# What this does:
#   1. Creates the project venv at ./venv (if missing) and installs
#      requirements.txt — this is the venv used for main.py / viewer.
#   2. Clones tenstorrent/tt-metal into ./tt-metal (if missing) — large,
#      ~5 GB. The .gitignore narrowly allow-lists our kernel subdir
#      (tt_metal/programming_examples/gaussian_splatting/) so the parent
#      repo can track it without picking up the rest of the SDK.
#   3. Removes the embedded tt-metal/.git so the parent repo can see
#      files inside the vendored clone.
#   4. Registers our kernel subdir in tt-metal's
#      programming_examples/CMakeLists.txt so the tt-metal build picks
#      it up automatically.
#   5. Sets up tt-metal's python_env (a separate venv with ttnn/tt-metal
#      Python bindings) via tt-metal/create_venv.sh.
#   6. Builds tt-metal (requires sudo — tt-metal's runtime/sfpi/ and
#      .cpmcache/ are root-owned from initial dependency install).
#
# Usage:
#   ./setup.sh
#
# After this completes:
#
#   # CPU viewer (no kernel)
#   source venv/bin/activate
#   python main.py scene/luigi.ply
#
#   # Tenstorrent kernel-accelerated viewer
#   source venv/bin/activate
#   export TT_METAL_HOME=$PWD/tt-metal
#   export TT_METAL_RUNTIME_ROOT=$PWD/tt-metal
#   python main.py scene/luigi.ply --backend kernel

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# tt-metal upstream source. Pinning to a specific tag/commit is safer for
# reproducibility (set TT_METAL_REF=v1.2.3 in the environment to override);
# main is the default for users who just want the latest.
TT_METAL_REPO="https://github.com/tenstorrent/tt-metal.git"
TT_METAL_REF="${TT_METAL_REF:-main}"

say()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
warn() { printf "\033[1;33m!! %s\033[0m\n" "$*" >&2; }
fail() { printf "\033[1;31m!! %s\033[0m\n" "$*" >&2; exit 1; }

# ----- Pre-flight checks -----------------------------------------------------
command -v python3 >/dev/null 2>&1 || fail "python3 not found on PATH"
command -v git     >/dev/null 2>&1 || fail "git not found on PATH"
command -v sudo    >/dev/null 2>&1 || fail "sudo not found (build_metal.sh needs root for SFPI cache)"

# ----- 1. Project venv -------------------------------------------------------
if [ ! -d "venv" ]; then
    say "Creating project venv at ./venv"
    python3 -m venv venv
fi

say "Installing project requirements into ./venv"
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt
deactivate

# ----- 2. Vendor tt-metal ----------------------------------------------------
if [ ! -d "tt-metal" ]; then
    say "Cloning tenstorrent/tt-metal (~5 GB, may take several minutes)"
    git clone --recurse-submodules --branch "$TT_METAL_REF" \
        "$TT_METAL_REPO" tt-metal
elif [ ! -d "tt-metal/tt_metal" ]; then
    fail "tt-metal/ exists but doesn't look like a tt-metal checkout. Remove or restore it before running setup."
else
    say "tt-metal/ already present, skipping clone"
fi

# Remove embedded .git so the parent repo can track files inside tt-metal/.
# (The .gitignore allow-lists our kernel subdir specifically; the rest of
# tt-metal stays untracked.)
if [ -d "tt-metal/.git" ]; then
    say "Removing embedded tt-metal/.git so parent repo can track our kernel subdir"
    rm -rf tt-metal/.git
fi

# ----- 3. Register kernel subdir in tt-metal's CMake ------------------------
PE_CMAKE="tt-metal/tt_metal/programming_examples/CMakeLists.txt"
if [ ! -f "$PE_CMAKE" ]; then
    fail "$PE_CMAKE not found — tt-metal layout has changed; update setup.sh"
fi
if grep -qF "add_subdirectory(gaussian_splatting)" "$PE_CMAKE"; then
    say "gaussian_splatting subdir already registered in $PE_CMAKE"
else
    say "Adding 'add_subdirectory(gaussian_splatting)' to $PE_CMAKE"
    printf "\nadd_subdirectory(gaussian_splatting)\n" >> "$PE_CMAKE"
fi

# ----- 4. tt-metal python_env (separate venv used by ttnn / build) ----------
if [ -d "tt-metal/python_env" ]; then
    say "tt-metal/python_env already present, skipping"
else
    say "Setting up tt-metal/python_env (ttnn bindings; takes a few minutes)"
    pushd tt-metal >/dev/null
    if [ ! -x "./create_venv.sh" ]; then
        fail "tt-metal/create_venv.sh not found or not executable"
    fi
    ./create_venv.sh
    popd >/dev/null
fi

# ----- 5. Build tt-metal -----------------------------------------------------
say "Building tt-metal (sudo required for SFPI / CPM caches)"
warn "This step can take 10-20 minutes on a first build."
pushd tt-metal >/dev/null
# shellcheck disable=SC1091
source python_env/bin/activate
sudo ./build_metal.sh
deactivate
popd >/dev/null

# ----- Done ------------------------------------------------------------------
say "Setup complete."
cat <<'EOF'

  Try the CPU viewer:
    source venv/bin/activate
    python main.py scene/luigi.ply

  Try the kernel-accelerated viewer:
    source venv/bin/activate
    export TT_METAL_HOME=$PWD/tt-metal
    export TT_METAL_RUNTIME_ROOT=$PWD/tt-metal
    python main.py scene/luigi.ply --backend kernel

EOF
