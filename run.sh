#!/bin/bash

set -x

kill_processes() {
  echo "Killing processes... $api_pid $npm_pid"
  trap 'kill $(jobs -p)' EXIT
}

if [ "$1" == "--debug" ]; then
  .venv/bin/uvicorn api.app:app --env-file=.env.debug --reload --log-level=info --use-colors --timeout-keep-alive=30  --timeout-graceful-shutdown=1 --port=3000&
else
  .venv/bin/uvicorn api.app:app --log-level=info --timeout-keep-alive=30 --port=3000 --host=0.0.0.0&
fi
api_pid=$!

cd client
if [ "$1" == "--debug" ]; then
  echo "Running in debug mode"
  npm run dev -- -p 3001 &
else
  HOSTNAME="127.0.0.1" PORT=3001 node .next/standalone/server.js &
fi
npm_pid=$!
cd ..

trap kill_processes EXIT
wait
