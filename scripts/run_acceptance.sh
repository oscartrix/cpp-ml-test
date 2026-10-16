#!/usr/bin/env bash
# Ejecuta todos los criterios de aceptación automáticos (ver README):
#   1. test_suite en Release
#   2. test_suite bajo AddressSanitizer + UndefinedBehaviorSanitizer
#   3. benchmark_cli en Release (sale con código != 0 si no cumple los umbrales)
# Uso (desde la raíz de candidate_starter): ./scripts/run_acceptance.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== [1/3] Build Release + test_suite ==="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j
./build/test_suite

echo "=== [2/3] Build ASan/UBSan + test_suite ==="
cmake -S . -B build_asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON >/dev/null
cmake --build build_asan -j
UBSAN_OPTIONS=halt_on_error=1 ./build_asan/test_suite

echo "=== [3/3] benchmark_cli (Release) ==="
./build/benchmark_cli data/train.csv data/test.csv

echo "=== Todos los criterios automáticos se cumplen ==="
