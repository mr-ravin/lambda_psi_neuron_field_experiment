#!/bin/sh
set -eu

nohup python3 run.py "$@" > run.log 2>&1 &
echo "Started in background. PID: $!"
echo "Log: run.log"