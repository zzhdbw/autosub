#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$ROOT/llama.cpp_stq"
BUILD_DIR="$LLAMA_DIR/build"

echo "Building llama.cpp (STQ kernel) …"
echo "  source: $LLAMA_DIR"
echo "  build:  $BUILD_DIR"

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo ""
echo "✓ Build complete!"
echo "  llama-server: $BUILD_DIR/bin/llama-server"
echo "  run 'autosub' to start using it."
