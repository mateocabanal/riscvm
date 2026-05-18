#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

make -C tests/rv64gc/synthetic
cargo bench -p riscvm-core --bench synthetic "$@"
