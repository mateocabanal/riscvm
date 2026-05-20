# Repository Guidelines

## Project Structure & Module Organization

This repository is a Rust workspace for an RV64GC userspace emulator. `riscvm-core/` contains CPU state, instruction execution, memory, syscalls, tracing, and the AArch64 JIT (`src/jit/`). `riscvm-runner/` builds the `riscvm` CLI and runner library. `riscvm-debugger/` contains the terminal debugger. Guest programs and fixtures live under `tests/`, grouped by ISA/language such as `tests/rv64gc/c/`, `tests/rv64gc/cpp/`, and `tests/rv64gc/synthetic/`. Generated Rust build output stays in `target/`.

## Build, Test, and Development Commands

- `cargo build --workspace`: build all crates, matching CI coverage.
- `cargo test --workspace`: run unit and integration tests for the workspace.
- `cargo test -p riscvm-core`: run core emulator tests while iterating on CPU, memory, syscall, or JIT behavior.
- `cargo run -p riscvm-runner -- tests/rv64gc/c/bin/hello`: run a fixture ELF through the CLI.
- `make -C tests/rv64gc/c`: rebuild static RV64GC C fixture binaries; requires `riscv64-linux-gnu-gcc`.
- `./scripts/bench-synthetic.sh`: rebuild synthetic fixtures and run the Criterion benchmark harness.
- `RISCVM_BENCH_NESTED=1 cargo bench --bench synthetic -- outer-jit`: run nested JIT benchmarks only.

## Coding Style & Naming Conventions

Use Rust 2021 defaults and keep code formatted with `cargo fmt --all`. Use `snake_case` for functions, modules, and local variables; `PascalCase` for types; `SCREAMING_SNAKE_CASE` for constants. Keep emulator behavior explicit: prefer named helpers for architectural edge cases, CSR behavior, syscall translation, and JIT lowering decisions. Split growing code into focused modules rather than expanding large files.

## Testing Guidelines

Instruction-level tests live in `riscvm-core/src/instruction_tests/`, while fixture-mode tests live in `riscvm-core/tests/`. Name tests after the instruction, mode, or regression being protected. For guest ELF regressions, add or rebuild the relevant fixture under `tests/rv64gc/...` and verify both interpreter and JIT-family behavior when practical. Use timed probes for infinite-output fixtures such as `tests/rv64gc/c/bin/fib`.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative summaries such as `Add compiler glossary` or `Optimize decoder and JIT passes`. Keep the subject focused on the observable change. PRs should describe the emulator behavior affected, list commands run, mention required toolchains, and include benchmark deltas when touching JIT, dispatch, memory, or syscall paths.

## Architecture & Configuration Notes

RISCVM runs Linux ELF userspace programs, not bare-metal kernels. The AArch64 JIT, hybrid, and AOT paths reject unsupported native lowering rather than silently falling back. Keep `Cargo.lock` committed for reproducible CLI builds. Do not commit `target/` output or ad hoc local benchmark artifacts.
