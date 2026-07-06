# vLLM (CPU) + QLever vector search — live end-to-end

Two docker-compose demos that prove QLever's `vec:embed` / `vec:distance`
wiring against a **vLLM CPU** embedding server reached over a **UNIX socket**.
Every query embedding is computed LIVE by vLLM through the socket; QLever ranks
it against a stored vector index.

| demo | model | run | proves |
|------|-------|-----|--------|
| **1. multimodal (SigLIP)** | `google/siglip2-base-patch16-224` | `./run.sh` | image index + **correct cross-modal retrieval by IMAGE query** |
| **2. text fallback (BGE)** | `BAAI/bge-small-en-v1.5` | `./run-text.sh` | text-document index + **correct semantic ranking by TEXT query** |

## TL;DR results (actual output)

Demo 1 — SigLIP image index, queried through `vec:embed` over the socket:
```
self[cat]  -> cat      d0=0.000000   OK   (exact image -> itself, distance 0)
cat2       -> cat      OK   1.cat(0.177) 2.dog(0.355) ...   (a DIFFERENT cat photo)
dog2       -> dog      OK   1.dog(0.230) 2.cat(0.314) ...
pizza2     -> pizza    OK   1.pizza(0.245) 2.banana(0.386) ...
```
Demo 2 — BGE text index, queried by paraphrase (little lexical overlap): 8/8 correct
```
"how do green plants make food using light"     -> photosynthesis  OK
"share prices plunged as rates were expected..." -> stockmarket     OK
"a huge arena in ancient Rome for combat"        -> colosseum       OK   ... (8/8)
```

## Known limitation: SigLIP **text → image** does not work on vLLM CPU

The headline goal was a *text* query ranking the matching *image* first. That
does **not** work here, and the cause is external to QLever:

vLLM 0.24.0 serves SigLIP's text encoder without SigLIP's canonical
**fixed-64-token padding**; it pools the last *real* token instead of the fixed
64th position. That lands text embeddings in a different subspace from image
embeddings, so text↔image cosine is near-zero/noise. `diagnose_siglip.py`
reproduces this deterministically with plain HuggingFace transformers:

```
padding="max_length"=64 (SigLIP canonical)  -> 6/6 text↔image argmax correct
padding="longest"       (vLLM-equivalent)   -> 1/6   (matches the vLLM output)
```

Both the SigLIP **image** encoder and the **text** encoder work in isolation
(image↔image and text↔text retrieval are correct); only their cross-modal
*text→image* alignment is broken at the serving layer. This affects SigLIP v1
and v2 identically. QLever computes distances faithfully over whatever vLLM
returns — the wiring is correct. Demo 1 therefore proves cross-modal retrieval
with **image queries** (which are correctly served), and Demo 2 provides the
task's sanctioned **text-query** fallback with a text-embedding model.

Run `./run.sh` then `docker compose exec -T vllm python3 /work/diagnose_siglip.py`
to see the root-cause table.

## How it works

- **vLLM** serves the model with `--runner pooling --uds /sockets/<name>.sock`
  in a shared named volume; CPU only. `--gpu-memory-utilization 0.3` caps the
  *CPU* RAM vLLM reserves (its flag controls CPU memory on the CPU backend);
  `VLLM_CPU_OMP_THREADS_BIND` pins it to a small core set so it shares the host.
  SigLIP needs an **empty text prompt** for image-only embedding, so
  `siglip_image_template.jinja` renders image messages to an empty string.
- **QLever** builds the index (`--service-index '{"vectorSearch":[...]}'`,
  `embeddingUrl: unix:/sockets/<name>.sock`) and answers SPARQL, calling the
  socket via `vec:embed`. It runs as **root** so it can connect to the
  root-owned socket (mode `srwxr-xr-x`; a non-root user gets EACCES).

The cross-modal query shape:
```sparql
PREFIX vec:  <https://qlever.cs.uni-freiburg.de/vectorSearch/>
PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>
SELECT ?img ?d WHERE {
  ?img a <http://ex/Image> .
  BIND(vec:distance(vidx:img, ?img, vec:embed(vidx:img, <IMG-OR-TEXT>)) AS ?d)
  FILTER(BOUND(?d))
} ORDER BY ?d LIMIT 3
```
`vec:embed(vidx:img, "text")` embeds a **literal as text**; an **IRI**
(`<data:image/jpeg;base64,...>` or an image URL) is embedded as an **image**.

## Files

| file | purpose |
|------|---------|
| `docker-compose.yml` | both stacks; the `text` compose profile adds demo 2 |
| `.env` | SigLIP model + resource knobs |
| `run.sh` | demo 1: up → wait → precompute → index → serve → query |
| `run-text.sh` | demo 2: up BGE → build text index (embeds docs via socket) → serve → query |
| `query.py` / `query_text.py` | host-side queries + assertions (ports 7001 / 7002) |
| `data/precompute.py` | embed corpus images over the socket → `img.npy` + `img.iris` |
| `data/probe.py` | text+image embed smoke test over the socket |
| `data/diagnose_siglip.py` | root-cause reproduction of the SigLIP text-pad issue |
| `data/siglip_image_template.jinja` | empty-prompt chat template for image-only embedding |
| `data/images/`, `data/queries/` | indexed corpus + held-out query images |
| `data/text/docs.txt`, `docs.iris` | text corpus for demo 2 |

## Teardown

```bash
docker compose --profile text down          # stop + remove all 4 containers + network
docker compose --profile text down -v       # also drop the sockets + HF-cache volumes
./clean.sh                                   # down -v AND restore ./data ownership
```

QLever runs as root, so some generated files under `./data` (e.g. `img.npy`,
`images_small/`, `index*/`) are root-owned on the host; `clean.sh` chowns
`./data` back via a throwaway container (no sudo needed).
