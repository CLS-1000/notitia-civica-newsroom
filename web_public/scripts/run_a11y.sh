#!/usr/bin/env bash
# Serve the built site and run pa11y-ci against every route in pa11y.json.
# Set PUPPETEER_EXECUTABLE_PATH to use an installed Chromium.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

npx astro preview --host 127.0.0.1 --port 4321 >/dev/null 2>&1 &
server=$!
trap 'kill "${server}" 2>/dev/null || true' EXIT

for _ in $(seq 1 50); do
  if curl --silent --fail --output /dev/null http://127.0.0.1:4321/; then
    break
  fi
  sleep 0.2
done

npx pa11y-ci --config pa11y.json
