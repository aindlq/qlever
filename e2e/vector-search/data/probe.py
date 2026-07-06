#!/usr/bin/env python3
"""Prove the vLLM UNIX-socket embedding endpoint works for BOTH shapes.

Runs inside the vLLM container. Prints, for a text query and an image query,
the HTTP status and the returned embedding dimension. Exit non-zero on failure.
"""
import base64
import http.client
import json
import os
import socket
import sys

SOCKET_PATH = "/sockets/vllm.sock"
MODEL = "siglip"


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost")
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect(self._path)
        self.sock = s


def post(payload):
    conn = UnixHTTPConnection(SOCKET_PATH)
    conn.request("POST", "/v1/embeddings", body=json.dumps(payload),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, data


def main():
    ok = True

    # ---- TEXT shape ----
    st, data = post({"model": MODEL, "input": ["a photo of a cat"]})
    if st == 200:
        emb = json.loads(data)["data"][0]["embedding"]
        print(f"[probe] TEXT  status=200 dim={len(emb)} head={emb[:3]}")
    else:
        ok = False
        print(f"[probe] TEXT  status={st} body={data[:300]!r}")

    # ---- IMAGE shape ----
    img_path = "/work/images_small/cat.jpg"
    if not os.path.exists(img_path):
        img_path = "/work/images/cat.jpg"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    uri = f"data:image/jpeg;base64,{b64}"
    st, data = post({
        "model": MODEL, "encoding_format": "float",
        "messages": [{"role": "user",
                      "content": [{"type": "image_url",
                                   "image_url": {"url": uri}}]}],
    })
    if st == 200:
        emb = json.loads(data)["data"][0]["embedding"]
        print(f"[probe] IMAGE status=200 dim={len(emb)} head={emb[:3]}")
    else:
        ok = False
        print(f"[probe] IMAGE status={st} body={data[:300]!r}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
