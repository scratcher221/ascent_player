#!/usr/bin/env bash
# Serve ASCENT locally. ES modules require http:// — file:// will not work.
cd "$(dirname "$0")"
PORT="${1:-8765}"
echo "ASCENT offline: http://127.0.0.1:${PORT}/ASCENT%20%E2%80%94%20Ride%20the%20pump.html"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
