#!/usr/bin/env bash
# Tear down the stack and restore host ownership of ./data (the qlever container
# runs as root, so files it writes are root-owned on the host).
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"

docker compose down -v --remove-orphans || true

# Restore ownership of ./data to the invoking user via a throwaway container
# (avoids needing sudo on the host).
docker run --rm -v "$PWD/data:/data" alpine \
  chown -R "$(id -u):$(id -g)" /data 2>/dev/null || \
  echo "note: could not chown ./data (run: sudo chown -R \$(id -u):\$(id -g) data)"

echo "cleaned up."
