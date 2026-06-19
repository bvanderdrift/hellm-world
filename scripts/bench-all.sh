#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mapfile -t benches < <(find . -name '*.bench.ts' -not -path './node_modules/*' | sort)

if [ "${#benches[@]}" -eq 0 ]; then
  echo "No *.bench.ts files found."
  exit 0
fi

echo "Running ${#benches[@]} benchmark(s) in series..."

for bench in "${benches[@]}"; do
  results="${bench%.ts}-results.txt"
  args=()
  if [[ "${bench}" == *"/llm.bench.ts" ]]; then
    args+=(--gpu-only)
  fi
  echo ""
  echo "==> ${bench} ${args[*]} -> ${results}"
  bun "${bench}" ${args[@]+"${args[@]}"} | tee "${results}"
done

echo ""
echo "Done. Swept ${#benches[@]} benchmark(s)."
