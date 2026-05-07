#!/usr/bin/env bash
# setup.sh — one-shot bootstrap for the gsplat_tt repository.
#
# Idempotent. Safe to re-run; each step skips work that's already done.
#
# Steps:
#   1. Project venv at ./venv with requirements.txt + `pip install -e .`
#   2. Vendor tt-metal into backends/tt/tt-metal/ (~5 GB clone)
#   3. Drop the embedded .git so the parent repo can track our kernel subdir
#   4. Register our kernel subdir in tt-metal's programming_examples CMake
#   5. Set up tt-metal's python_env (separate venv with ttnn bindings)
#   6. Build tt-metal (requires sudo for SFPI / .cpmcache)
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

# ----- 3. Register our kernel subdir in tt-metal's CMake --------------------
PE_CMAKE="$TT_METAL_DIR/tt_metal/programming_examples/CMakeLists.txt"
if [ ! -f "$PE_CMAKE" ]; then
    fail "$PE_CMAKE not found — tt-metal layout has changed; update setup.sh"
fi
if ! grep -qF "add_subdirectory(gaussian_splatting)" "$PE_CMAKE"; then
    say "Registering gaussian_splatting subdir in $PE_CMAKE"
    printf "\nadd_subdirectory(gaussian_splatting)\n" >> "$PE_CMAKE"
else
    say "gaussian_splatting subdir already registered"
fi

# ----- 4. tt-metal python_env (ttnn bindings) -------------------------------
if [ -d "$TT_METAL_DIR/python_env" ]; then
    say "$TT_METAL_DIR/python_env already present, skipping"
else
    say "Setting up $TT_METAL_DIR/python_env (ttnn bindings; takes a few minutes)"
    pushd "$TT_METAL_DIR" >/dev/null
    [ -x "./create_venv.sh" ] || fail "create_venv.sh missing/not executable in $TT_METAL_DIR"
    ./create_venv.sh
    popd >/dev/null
fi

# ----- 5. Build tt-metal + our kernel ---------------------------------------
# `--build-programming-examples` is required so tt-metal's CMake includes
# `programming_examples/`, which contains our gaussian_splatting subproject.
# Without the flag, BUILD_PROGRAMMING_EXAMPLES=OFF and the
# `metal_example_gaussian_splatting` target is never created.
say "Building tt-metal + gaussian_splatting kernel (sudo required for SFPI / CPM caches)"
warn "First build can take 10-20 minutes."
pushd "$TT_METAL_DIR" >/dev/null
# shellcheck disable=SC1091
source python_env/bin/activate
sudo ./build_metal.sh --build-programming-examples
deactivate
popd >/dev/null

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
