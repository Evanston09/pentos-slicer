#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
exec uv run watchfiles --filter python "python main.py" .
