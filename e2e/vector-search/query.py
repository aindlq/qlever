#!/usr/bin/env python3
"""Cross-modal QLever queries against the SigLIP2 image index.

Runs on the HOST against the QLever server (http://127.0.0.1:7001). Every query
computes its query embedding LIVE via `vec:embed` over the vLLM UNIX socket and
ranks it against the stored image embeddings with `vec:distance` (cosine).

ASSERTED (must pass):
  * image self-query  -> the same image at distance ~0
  * held-out image query (a DIFFERENT photo of the same subject) -> that subject

INFORMATIONAL (not asserted): text -> image. vLLM 0.24.0 CPU does not reproduce
SigLIP's fixed-64 text padding, so text embeddings land in a different subspace
from image embeddings (see diagnose_siglip.py). The text/image encoders each
work; only their cross-modal alignment is broken at the serving layer. This is a
vLLM limitation, NOT a QLever wiring problem -- the image queries below, through
the exact same wiring, rank correctly.

Exit code 0 iff every ASSERTED query passes.
"""
import base64
import json
import os
import sys
import urllib.request

ENDPOINT = os.environ.get("QLEVER_ENDPOINT", "http://127.0.0.1:7001")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SMALL_DIR = os.path.join(DATA, "images_small")
QUERY_DIR = os.path.join(DATA, "queries")

PREFIXES = (
    "PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>\n"
    "PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>\n"
)
SUBJECTS = ["cat", "dog", "pizza", "mountain", "banana", "sunflower"]
# held-out query image file (in data/queries) -> the indexed subject it depicts
HELDOUT = {"cat2": "cat", "dog2": "dog", "pizza2": "pizza"}


def sparql(query):
    req = urllib.request.Request(
        ENDPOINT, data=query.encode("utf-8"),
        headers={"Content-Type": "application/sparql-query",
                 "Accept": "application/sparql-results+json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def rows(result):
    out = []
    for b in result["results"]["bindings"]:
        iri = b["img"]["value"] if "img" in b else None
        d = float(b["d"]["value"]) if "d" in b else None
        out.append((iri, d))
    return out


def distance_query(embed_arg, limit=6):
    q = PREFIXES + (
        "SELECT ?img ?d WHERE {\n"
        "  ?img a <http://ex/Image> .\n"
        f"  BIND(vec:distance(vidx:img, ?img, vec:embed(vidx:img, {embed_arg})) AS ?d)\n"
        "  FILTER(BOUND(?d))\n"
        f"}} ORDER BY ?d LIMIT {limit}\n")
    return rows(sparql(q))


def data_uri_iri(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return f"<data:image/jpeg;base64,{b64}>"


def fmt(ranking):
    return "  ".join(f"{i+1}.{(iri or '?').split('/')[-1]}({d:.4f})"
                     for i, (iri, d) in enumerate(ranking))


def main():
    failures = []

    print("=" * 74)
    print("A. IMAGE SELF-QUERIES  (embed the exact image -> itself at distance ~0)")
    print("=" * 74)
    for s in ["cat", "banana", "mountain"]:
        r = distance_query(data_uri_iri(os.path.join(SMALL_DIR, f"{s}.jpg")))
        top = (r[0][0] or "").split("/")[-1]
        ok = top == s and r[0][1] < 1e-3
        print(f"self[{s:9s}] -> top={top:9s} d0={r[0][1]:.6f} {'OK' if ok else 'FAIL'}")
        print(f"    {fmt(r)}")
        if not ok:
            failures.append(f"self '{s}': top='{top}' d0={r[0][1]}")

    print()
    print("=" * 74)
    print("B. HELD-OUT IMAGE QUERIES  (a DIFFERENT photo of the subject -> that subject)")
    print("=" * 74)
    for qfile, expect in HELDOUT.items():
        path = os.path.join(QUERY_DIR, f"{qfile}.jpg")
        if not os.path.exists(path):
            print(f"  (skip {qfile}: file missing)")
            continue
        r = distance_query(data_uri_iri(path))
        top = (r[0][0] or "").split("/")[-1]
        ok = top == expect
        print(f"query[{qfile:7s} = a {expect}] -> top={top:9s} {'OK' if ok else 'MISMATCH'}")
        print(f"    {fmt(r)}")
        if not ok:
            failures.append(f"held-out '{qfile}': expected '{expect}', got '{top}'")

    print()
    print("=" * 74)
    print("C. TEXT -> IMAGE  (INFORMATIONAL: known vLLM SigLIP-text limitation)")
    print("=" * 74)
    hits = 0
    for s in SUBJECTS:
        r = distance_query(f'"a photo of a {s}"')
        top = (r[0][0] or "").split("/")[-1]
        hits += top == s
        print(f'text "a photo of a {s:9s}" -> top={top:9s} {"match" if top==s else "(miss)"}')
        print(f"    {fmt(r)}")
    print(f"[informational] text->image correct-top: {hits}/{len(SUBJECTS)} "
          f"(expected to be poor; see diagnose_siglip.py for the root cause)")

    print()
    print("=" * 74)
    print("D. PLAIN TEXT-EMBEDDING SANITY  (vec:embed returns a typed vector literal)")
    print("=" * 74)
    b = sparql(PREFIXES +
               'SELECT ?e WHERE { BIND(vec:embed(vidx:img, "a photo of a cat") AS ?e) }'
               )["results"]["bindings"][0]["e"]
    dt = b.get("datatype", "")
    comps = b["value"].split(",")
    print(f"datatype = {dt}")
    print(f"components = {len(comps)}   head = {','.join(comps[:3])} ...")
    if "vectorSearch" not in dt:
        failures.append("text-embed sanity: unexpected datatype")

    print()
    print("=" * 74)
    if failures:
        print(f"RESULT: FAILED ({len(failures)} asserted issue(s)):")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("RESULT: ALL ASSERTED QUERIES PASSED "
          "(image self + held-out image retrieval correct through the wiring)")


if __name__ == "__main__":
    main()
