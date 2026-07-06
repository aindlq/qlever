#!/usr/bin/env bash
# End-to-end: vLLM(CPU, SigLIP) + QLever cross-modal vector search.
#
#   up vLLM -> wait for socket+embed -> precompute image embeddings ->
#   build QLever vector index -> start QLever server -> run cross-modal queries.
#
# Re-runnable. Keeps the HF cache volume between runs (model downloads once).
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DC="docker compose"
DATA="$PWD/data"
IMG_DIR="$DATA/images"
NAME="img"                      # vector index name -> IRI vidx:img
SVC_INDEX='{"vectorSearch":[{"name":"'"$NAME"'","npy":"/data/img.npy","iris":"/data/img.iris","metric":"cosine","scalar":"bf16","hnsw":false,"embeddingUrl":"unix:/sockets/vllm.sock","embeddingModel":"siglip"}]}'

log() { echo -e "\n\033[1;34m== $* ==\033[0m"; }
bail() { echo -e "\033[1;31mFAILED: $*\033[0m" >&2; exit 1; }

# --- 0. corpus (download once via Wikimedia Special:FilePath) -----------------
declare -A CORPUS=(
  [cat]=Cat_November_2010-1a.jpg
  [dog]=YellowLabradorLooking_new.jpg
  [pizza]=Eq_it-na_pizza-margherita_sep2005_sml.jpg
  [mountain]=Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg
  [banana]=Banana-Single.jpg
  [sunflower]=A_sunflower.jpg
)
fetch_corpus() {
  log "0. corpus"
  mkdir -p "$IMG_DIR"
  local ua="qlever-e2e-test/1.0 (artem@rem.sh)"
  for s in "${!CORPUS[@]}"; do
    local f="$IMG_DIR/$s.jpg"
    if [[ -s "$f" ]] && file -b --mime-type "$f" | grep -q image; then continue; fi
    curl -sL -A "$ua" --max-time 60 -o "$f" \
      "https://commons.wikimedia.org/wiki/Special:FilePath/${CORPUS[$s]}"
    file -b --mime-type "$f" | grep -q image || bail "corpus fetch failed for $s"
    echo "  fetched $s.jpg"
  done
}

# --- 1. vLLM up + wait for a working embed -----------------------------------
start_vllm() {
  log "1. start vLLM (model: ${VLLM_MODEL:-from .env})"
  $DC up -d vllm
  echo "waiting for the UNIX socket + a successful text/image embed (up to ~14 min)..."
  local i
  for ((i=0; i<84; i++)); do
    if $DC exec -T vllm test -S /sockets/vllm.sock 2>/dev/null \
       && $DC exec -T vllm python3 /work/probe.py >/tmp/qvec_probe.log 2>&1; then
      cat /tmp/qvec_probe.log
      return 0
    fi
    sleep 10
    [[ $((i % 6)) -eq 5 ]] && echo "  ...still waiting ($(( (i+1)*10 ))s)"
  done
  echo "--- last vLLM logs ---"; $DC logs --tail 40 vllm
  bail "vLLM did not become ready in time"
}

# --- 2. precompute image embeddings ------------------------------------------
precompute() {
  log "2. precompute SigLIP image embeddings (inside vLLM container)"
  $DC exec -T vllm python3 /work/precompute.py
  [[ -s "$DATA/img.npy" && -s "$DATA/img.iris" ]] || bail "precompute produced no npy/iris"
}

# --- 3. knowledge graph + vector index ---------------------------------------
build_index() {
  log "3. build knowledge graph + QLever vector index"
  # kg.ttl: one :Image entity per embedded image, aligned with img.iris.
  jq -r '.[] | "<\(.iri)> a <http://ex/Image> ; <http://ex/label> \"\(.subject)\" ."' \
    "$DATA/manifest.json" > "$DATA/kg.ttl"
  echo "--- kg.ttl ---"; cat "$DATA/kg.ttl"
  rm -rf "$DATA/index"; mkdir -p "$DATA/index"
  $DC run --rm --no-deps --entrypoint /qlever/qlever-index qlever \
    -i /data/index/vec -f /data/kg.ttl -F ttl --service-index "$SVC_INDEX" \
    2>&1 | tee "$DATA/index-build.log"
  grep -Eiq 'indexed .* vector|vectors' "$DATA/index-build.log" \
    || echo "(note: could not find an 'indexed N vectors' line; see index-build.log)"
}

# --- 4. QLever server up + wait ----------------------------------------------
start_server() {
  log "4. start QLever server"
  $DC up -d --force-recreate qlever
  local i
  for ((i=0; i<60; i++)); do
    if curl -s -m 5 -H 'Content-Type: application/sparql-query' \
         -H 'Accept: application/sparql-results+json' \
         --data 'SELECT (1 AS ?x) {}' http://127.0.0.1:7001 2>/dev/null \
         | grep -q '"x"'; then
      echo "  server ready"; return 0
    fi
    sleep 2
  done
  $DC logs --tail 40 qlever; bail "QLever server not ready"
}

# --- 5. cross-modal queries --------------------------------------------------
run_queries() {
  log "5. cross-modal SPARQL queries"
  python3 "$PWD/query.py"
}

main() {
  fetch_corpus
  start_vllm
  precompute
  build_index
  start_server
  run_queries
  log "DONE"
}
main "$@"
