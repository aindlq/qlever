#!/usr/bin/env python3
"""Precompute SigLIP image embeddings via the local vLLM UNIX socket.

Runs INSIDE the vLLM container (has numpy + Pillow + the socket locally).
For each corpus image it:
  1. optionally shrinks it (Pillow) to keep the base64 `data:` URI small,
  2. POSTs the vLLM multimodal image-embedding request over the unix socket,
  3. collects `data[0].embedding`.
Then writes an aligned `img.npy` (float32, N x D, '<f4') + `img.iris`
(`<http://ex/img/SUBJECT>` per line) and a small resized copy of each image
(used later for the image self-query).

Output (in /work, the bind-mounted ./data dir):
  img.npy, img.iris, manifest.json, images_small/<subject>.jpg
"""
import base64
import http.client
import io
import json
import os
import socket
import sys

import numpy as np

SOCKET_PATH = "/sockets/vllm.sock"
MODEL = "siglip"
WORK = "/work"
IMG_DIR = os.path.join(WORK, "images")
SMALL_DIR = os.path.join(WORK, "images_small")
MAX_SIDE = 384  # shrink longest side to this before embedding

# Fixed order -> npy rows and .iris lines stay aligned.
CORPUS = [
    ("cat", "cat.jpg"),
    ("dog", "dog.jpg"),
    ("pizza", "pizza.jpg"),
    ("mountain", "mountain.jpg"),
    ("banana", "banana.jpg"),
    ("sunflower", "sunflower.jpg"),
]


class UnixHTTPConnection(http.client.HTTPConnection):
    """http.client over a UNIX domain socket."""

    def __init__(self, path):
        super().__init__("localhost")
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect(self._path)
        self.sock = s


def post_json(path, payload):
    conn = UnixHTTPConnection(SOCKET_PATH)
    body = json.dumps(payload)
    conn.request("POST", path, body=body,
                 headers={"Content-Type": "application/json",
                          "Accept": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status} from vLLM: {data[:300]!r}")
    return json.loads(data)


def shrink_to_data_uri(path):
    """Return (data_uri, saved_small_path). Uses Pillow if available."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        scale = min(1.0, MAX_SIDE / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        raw = buf.getvalue()
    except Exception as e:  # noqa: BLE001 - Pillow missing/failed: use original
        sys.stderr.write(f"[precompute] Pillow unavailable/failed ({e}); "
                         f"using original bytes for {path}\n")
        with open(path, "rb") as f:
            raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", raw


def embed_image(data_uri):
    payload = {
        "model": MODEL,
        "encoding_format": "float",
        "messages": [
            {"role": "user",
             "content": [{"type": "image_url",
                          "image_url": {"url": data_uri}}]},
        ],
    }
    parsed = post_json("/v1/embeddings", payload)
    return parsed["data"][0]["embedding"]


def main():
    os.makedirs(SMALL_DIR, exist_ok=True)
    vecs = []
    iris = []
    manifest = []
    for subject, fname in CORPUS:
        path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            sys.stderr.write(f"[precompute] missing {path}, skipping\n")
            continue
        data_uri, raw = shrink_to_data_uri(path)
        small_path = os.path.join(SMALL_DIR, f"{subject}.jpg")
        with open(small_path, "wb") as f:
            f.write(raw)
        emb = embed_image(data_uri)
        vecs.append(emb)
        iris.append(f"<http://ex/img/{subject}>")
        manifest.append({"subject": subject, "file": fname,
                         "iri": f"http://ex/img/{subject}",
                         "dim": len(emb),
                         "small_bytes": len(raw)})
        print(f"[precompute] {subject:10s} dim={len(emb)} "
              f"|v|={float(np.linalg.norm(emb)):.4f} small={len(raw)}B")

    if not vecs:
        raise SystemExit("[precompute] no embeddings produced")

    dims = {len(v) for v in vecs}
    if len(dims) != 1:
        raise SystemExit(f"[precompute] inconsistent embedding dims: {dims}")

    arr = np.asarray(vecs, dtype="<f4")
    np.save(os.path.join(WORK, "img.npy"), arr, allow_pickle=False)
    with open(os.path.join(WORK, "img.iris"), "w") as f:
        f.write("\n".join(iris) + "\n")
    with open(os.path.join(WORK, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[precompute] wrote img.npy shape={arr.shape} dtype={arr.dtype}")
    print(f"[precompute] wrote img.iris ({len(iris)} entities)")

    # Shrink held-out query images in place so their base64 `data:` URIs stay
    # small when embedded as IRIs in the SPARQL query body.
    qdir = os.path.join(WORK, "queries")
    if os.path.isdir(qdir):
        for fn in sorted(os.listdir(qdir)):
            if not fn.lower().endswith(".jpg"):
                continue
            p = os.path.join(qdir, fn)
            _, raw = shrink_to_data_uri(p)
            with open(p, "wb") as f:
                f.write(raw)
        print(f"[precompute] shrank held-out query images in {qdir}")


if __name__ == "__main__":
    main()
