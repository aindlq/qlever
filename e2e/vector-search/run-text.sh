#!/usr/bin/env bash
# Text-only fallback e2e: TEXT query -> nearest document, through the SAME
# vLLM(CPU)-over-unix-socket + QLever vec:embed wiring, using a real text
# embedding model (BGE). This gives CORRECT text-query rankings (unlike SigLIP
# text->image, which vLLM CPU does not serve correctly -- see diagnose_siglip.py).
#
#   up vllm-text (BGE) -> wait -> build a text index (QLever embeds each doc via
#   the socket at BUILD time from the `texts` input) -> serve -> text queries.
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DC="docker compose --profile text"
DATA="$PWD/data"
SVC='{"vectorSearch":[{"name":"docs","texts":"/data/text/docs.txt","iris":"/data/text/docs.iris","metric":"cosine","scalar":"f32","hnsw":false,"embeddingUrl":"unix:/sockets/vllm-text.sock","embeddingModel":"bge"}]}'

log() { echo -e "\n\033[1;34m== $* ==\033[0m"; }
bail() { echo -e "\033[1;31mFAILED: $*\033[0m" >&2; exit 1; }

log "1. start vLLM-text (BAAI/bge-small-en-v1.5)"
$DC up -d vllm-text
echo "waiting for the BGE socket + a text embed (up to ~10 min)..."
for ((i=0; i<60; i++)); do
  if docker compose exec -T vllm-text test -S /sockets/vllm-text.sock 2>/dev/null \
     && docker compose exec -T vllm-text python3 - <<'PY' 2>/dev/null
import http.client,json,socket
class U(http.client.HTTPConnection):
    def __init__(s): super().__init__("localhost")
    def connect(s):
        so=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); so.settimeout(60)
        so.connect("/sockets/vllm-text.sock"); s.sock=so
c=U(); c.request("POST","/v1/embeddings",body=json.dumps({"model":"bge","input":["hello world"]}),headers={"Content-Type":"application/json"})
r=c.getresponse(); d=json.loads(r.read()); print("bge dim", len(d["data"][0]["embedding"]))
PY
  then echo "  BGE ready"; break; fi
  sleep 10; [[ $((i%6)) -eq 5 ]] && echo "  ...still waiting ($(( (i+1)*10 ))s)"
done

log "2. build text KG + vector index (QLever embeds docs via the socket at build)"
python3 - "$DATA" <<'PY'
import sys,os
d=sys.argv[1]; base=os.path.join(d,"text")
iris=[l.strip() for l in open(os.path.join(base,"docs.iris")) if l.strip()]
txts=[l.rstrip("\n") for l in open(os.path.join(base,"docs.txt")) if l.strip()]
assert len(iris)==len(txts), (len(iris),len(txts))
with open(os.path.join(base,"kg-text.ttl"),"w") as f:
    for iri,t in zip(iris,txts):
        t=t.replace("\\","\\\\").replace('"','\\"')
        f.write(f'{iri} a <http://ex/Doc> ; <http://ex/text> "{t}" .\n')
print("wrote kg-text.ttl with", len(iris), "documents")
PY
rm -rf "$DATA/index-text"; mkdir -p "$DATA/index-text"
$DC run --rm --no-deps --entrypoint /qlever/qlever-index qlever-text \
  -i /data/index-text/docs -f /data/text/kg-text.ttl -F ttl --service-index "$SVC" \
  2>&1 | tee "$DATA/index-text-build.log" | grep -iE 'indexed .* vectors|error|fail' || true

log "3. start qlever-text server (port 7002)"
$DC up -d qlever-text
for ((i=0; i<60; i++)); do
  curl -s -m 5 -H 'Content-Type: application/sparql-query' \
    -H 'Accept: application/sparql-results+json' \
    --data 'SELECT (1 AS ?x) {}' http://127.0.0.1:7002 2>/dev/null | grep -q '"x"' \
    && { echo "  server ready"; break; }
  sleep 2
done

log "4. TEXT queries"
python3 "$PWD/query_text.py"
log "DONE (text fallback)"
