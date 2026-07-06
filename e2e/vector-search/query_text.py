#!/usr/bin/env python3
"""Text-only fallback: TEXT query -> nearest text document, via the wiring.

Runs on the HOST against qlever-text (http://127.0.0.1:7002). Each query is a
PARAPHRASE with little lexical overlap with its target document, so a correct
top-1 demonstrates genuine SEMANTIC retrieval -- the query is embedded live via
`vec:embed` over the BGE vLLM unix socket and ranked with `vec:distance`.

Exit code 0 iff every query returns the expected document first.
"""
import json
import os
import sys
import urllib.request

ENDPOINT = os.environ.get("QLEVER_TEXT_ENDPOINT", "http://127.0.0.1:7002")
PREFIXES = (
    "PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>\n"
    "PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>\n"
)
# BGE retrieval convention: queries carry an instruction prefix; passages don't.
QPREFIX = "Represent this sentence for searching relevant passages: "

# (paraphrased query, expected doc IRI suffix)
QUERIES = [
    ("how do green plants make food using light", "photosynthesis"),
    ("a famous iron landmark in the French capital", "eiffel"),
    ("training neural networks on big data to make predictions", "ml"),
    ("a robotic probe arriving at the red planet", "mars"),
    ("staying fit and keeping your heart healthy", "exercise"),
    ("insects that help flowers reproduce", "bees"),
    ("share prices plunged as rates were expected to climb", "stockmarket"),
    ("a huge arena in ancient Rome for spectator combat", "colosseum"),
]


def sparql(query):
    req = urllib.request.Request(
        ENDPOINT, data=query.encode("utf-8"),
        headers={"Content-Type": "application/sparql-query",
                 "Accept": "application/sparql-results+json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run(text, limit=3):
    esc = text.replace("\\", "\\\\").replace('"', '\\"')
    q = PREFIXES + (
        "SELECT ?doc ?d WHERE {\n"
        "  ?doc a <http://ex/Doc> .\n"
        f'  BIND(vec:distance(vidx:docs, ?doc, vec:embed(vidx:docs, "{esc}")) AS ?d)\n'
        "  FILTER(BOUND(?d))\n"
        f"}} ORDER BY ?d LIMIT {limit}\n")
    out = []
    for b in sparql(q)["results"]["bindings"]:
        out.append((b["doc"]["value"].split("/")[-1], float(b["d"]["value"])))
    return out


def main():
    print("=" * 74)
    print("TEXT -> DOCUMENT semantic retrieval (BGE via vec:embed over unix socket)")
    print("=" * 74)
    failures = []
    for qtext, expect in QUERIES:
        r = run(QPREFIX + qtext)
        top = r[0][0] if r else None
        ok = top == expect
        rank = "  ".join(f"{i+1}.{d}({v:.3f})" for i, (d, v) in enumerate(r))
        print(f'query "{qtext}"')
        print(f'   -> top={top}  expected={expect}  {"OK" if ok else "MISMATCH"}')
        print(f"      {rank}")
        if not ok:
            failures.append(f"'{qtext}': expected {expect}, got {top}")
    print()
    print("=" * 74)
    if failures:
        print(f"RESULT: FAILED ({len(failures)}):")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print(f"RESULT: ALL {len(QUERIES)} TEXT QUERIES RANKED THE CORRECT DOCUMENT FIRST")


if __name__ == "__main__":
    main()
