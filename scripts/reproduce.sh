#!/usr/bin/env bash
# Reproduce the GraviTune results end-to-end on a fresh Arm box.
#
# Tested from a clean Ubuntu 24.04 arm64 EC2 instance (c8g/c7g/c6g.4xlarge).
# Safe to re-run: each step is skipped if its output already exists.
set -euo pipefail

WORK="${GRAVITUNE_WORK:-$HOME/gravitune-repro}"
MODEL_BASE="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main"
MODELS=(
  qwen2.5-1.5b-instruct-q4_0.gguf
  qwen2.5-1.5b-instruct-q4_k_m.gguf
  qwen2.5-1.5b-instruct-q8_0.gguf
)

say() { printf '\n=== %s ===\n' "$*"; }

# --- 0. sanity: are we actually on Arm? -----------------------------------
# Fail loudly here rather than producing meaningless numbers under emulation.
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
  echo "ERROR: this is $ARCH, not Arm. GraviTune measures Arm silicon;" >&2
  echo "running it under x86 or QEMU emulation produces numbers that mean nothing." >&2
  exit 1
fi

say "target: $ARCH, $(nproc) cores"

# --- 1. dependencies -------------------------------------------------------
if ! command -v cmake >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  say "installing build dependencies"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git curl libcurl4-openssl-dev python3
fi

mkdir -p "$WORK"
cd "$WORK"

# --- 2. llama.cpp ----------------------------------------------------------
if [ ! -x llama.cpp/build/bin/llama-bench ]; then
  say "building llama.cpp (this takes a few minutes)"
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=ON
  cmake --build build --config Release -j"$(nproc)"
  cd "$WORK"
else
  say "llama.cpp already built, skipping"
fi

# --- 3. GraviTune ----------------------------------------------------------
if [ ! -d gravitune ]; then
  say "fetching GraviTune"
  git clone --depth 1 https://github.com/aswin-giridhar/gravitune
else
  say "GraviTune already present, skipping"
fi

# --- 4. models -------------------------------------------------------------
mkdir -p models
for m in "${MODELS[@]}"; do
  if [ ! -s "models/$m" ]; then
    say "downloading $m"
    curl -fsSL -o "models/$m" "$MODEL_BASE/$m"
  fi
done

# --- 5. detect + tune ------------------------------------------------------
cd gravitune

say "detecting Arm target"
python3 -m gravitune detect

say "running the sweep"
python3 -m gravitune tune \
  --bench "$WORK/llama.cpp/build/bin/llama-bench" \
  --models "$WORK"/models/*.gguf \
  --objective interactive \
  --outdir "$WORK/results"

say "done"
echo "Artifacts in $WORK/results:"
ls -1 "$WORK/results"
