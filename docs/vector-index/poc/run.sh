#!/usr/bin/env bash
# Reproduce the vector-core proof-of-concept (needs only system g++ + header-only usearch).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
USEARCH_INC="${USEARCH_INC:-$here/../../../build/_deps/usearch-src/include}"
if [ ! -f "$USEARCH_INC/usearch/index_dense.hpp" ]; then
  tmp="$(mktemp -d)"; git clone --depth 1 https://github.com/unum-cloud/usearch "$tmp/usearch"
  USEARCH_INC="$tmp/usearch/include"
fi
g++ -std=c++17 -O2 -I "$USEARCH_INC" "$here/vec_poc.cpp" -o "$here/vec_poc"
cd "$here" && ./vec_poc
