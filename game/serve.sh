#!/usr/bin/env bash
# Serve ASCENT locally. ES modules require http:// — file:// will not work.
cd "$(dirname "$0")"
PORT="${1:-8765}"
echo "ASCENT local: http://127.0.0.1:${PORT}/index.html"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
