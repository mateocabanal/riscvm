# RISCVM Compiler Modes

This document describes how the current RISCVM interpreter, JIT, hybrid, and AOT
execution modes work at the implementation level. It is written against the
current code in:

- `riscvm-runner/src/lib.rs`
- `riscvm-core/src/cpu.rs`
- `riscvm-core/src/jit/mod.rs`
- `riscvm-core/src/jit/aarch64.rs`
- `riscvm-core/src/jit/optimizer.rs`
- `riscvm-core/src/jit/trace.rs`
- `riscvm-core/src/tracer.rs`

## Executive Summary

RISCVM has one interpreter and one native compiler pipeline.

- The interpreter decodes and executes guest RV64GC instructions one at a time.
- The JIT pipeline compiles guest basic blocks or regions into host AArch64 code.
- The hybrid mode currently uses the same native-only compilation semantics as
  JIT mode. It is the default engine label on supported hosts, but it is not an
  interpreter fallback mode in the current implementation.
- The AOT mode precompiles native code before first execution. By default it can
  still lazily compile native runtime misses; strict AOT is enabled with
  `--aot-no-compile-misses`.

The JIT-family modes intentionally do not execute unsupported blocks through the
interpreter. If a block cannot be lowered to native code, execution returns a
JIT error instead of silently falling back.

## Platform Support

The native backend is currently enabled only on Unix AArch64 hosts:

```rust
#[cfg(all(target_arch = "aarch64", unix))]
type NativeBackend = aarch64::AArch64Backend;
```

On other hosts, `JitEngine::new()` and `JitEngine::with_options(...)` return
`JitError::UnsupportedHost`. The runner defaults to:

- `--hybrid` on Unix AArch64 hosts
- `--interp` elsewhere

## Runner Entry Points

The runner accepts four engine selections:

```text
--interp
--jit
--hybrid
--aot
```

The runner maps those flags to `RunnerEngineMode`, then to an optional
`JitExecutionMode`:

```text
Interpreter -> None
Jit         -> Some(JitExecutionMode::Jit)
Hybrid      -> Some(JitExecutionMode::Hybrid)
Aot         -> Some(JitExecutionMode::Aot)
```

If no JIT execution mode is selected, the runner calls:

```rust
riscvm.start();
```

If a JIT execution mode is selected, the runner builds `JitOptions` and calls:

```rust
riscvm.start_jit_with_options(jit_options);
```

## Mode Matrix

| Mode | Precompile before run | Runtime compile misses | Dynamic recompilation | Interpreter fallback |
| --- | --- | --- | --- | --- |
| Interpreter | No | No | No | N/A |
| JIT | No | Yes | Configurable, default yes | No |
| Hybrid | No | Yes | Configurable, default yes | No |
| AOT | Yes | Configurable, default yes | Forced off | No |

Important current detail: `Jit` and `Hybrid` differ mainly by tracing/profile
engine label and CLI intent. Both modes allow runtime native compilation, both
can dynamically recompile hot blocks, and both reject blocks that would require
interpreter fallback.

## Interpreter Mode

The interpreter loop lives in `RV64GC::start()`:

```rust
while !self.should_quit {
    self.step();
}
```

Each `step()`:

1. Reads `pc`.
2. Decodes the current instruction through `decode_current_instruction()`.
3. Executes the decoded `RV64GCInstruction`.
4. Resets `x0` to zero.
5. Advances `pc` by the decoded instruction length unless the instruction
   modified control flow.

The decode cache is keyed by `pc` and RAM `code_version`, so self-modifying code
invalidates cached decodes when executable memory changes.

Interpreter mode is the reference behavior used by the fixture matrix.

## JIT Engine State

`JitEngine` owns the native compilation pipeline:

```rust
pub struct JitEngine {
    backend: NativeBackend,
    cache: FastU64Map<CompiledBlock>,
    pending_optimized_compiles: FastU64Set,
    background_compiler: Option<BackgroundCompiler>,
    options: JitOptions,
    precompiled_entry: bool,
    host_libc_symbols: FastU64Map<HostLibcFunction>,
    host_libc_symbol_range: Option<(u64, u64)>,
    host_libc_initialized: bool,
}
```

The central cache maps guest block start PC to `CompiledBlock`. A compiled block
contains:

- the executable host code object
- a fingerprint of guest `(pc, opcode)` pairs
- the RAM `code_version` at compile time
- guest instruction count and native code length
- optional native listing for `--jit-dump`
- tier metadata and execution counters
- loop/back-edge flags used by the promotion policy

The fingerprint and code version are used to detect stale native blocks after
guest executable memory changes.

## JIT Execution Loop

All JIT-family modes enter the same high-level loop:

```rust
cpu.trace_start(mode);
ensure_precompiled(cpu)?;
while !cpu.should_quit {
    step_precompiled(cpu)?;
    assert_eq!(cpu.registers[Zero], 0);
}
cpu.trace_finish();
```

`step_precompiled()` is the heart of the runtime:

1. Drain completed background compiler jobs.
2. Read current guest `pc`.
3. Handle the host-libc exit trampoline.
4. Try host-libc direct handling for recognized symbols.
5. Look up a compiled block for `pc`.
6. Decide whether to compile, recompile, or report an AOT miss.
7. Execute the native block.
8. Normalize `x0` back to zero.
9. Record trace/profile information if enabled.

The native block returns the number of guest instructions it executed. The block
also writes the next guest PC into the CPU register file before returning.

## Block Planning

The compiler does not lower arbitrary guest memory directly. It first creates a
`BlockPlan`.

### Baseline Plan

`BlockPlan::from_cpu(cpu, start_pc)`:

1. Starts at `start_pc`.
2. Reads and decodes up to `MAX_BLOCK_INSTRUCTIONS` guest instructions.
3. Lowers each decoded instruction to `NativeInstruction`.
4. Stops on:
   - control flow
   - fetch fault
   - unsupported lowering
   - max instruction count
5. Records a fingerprint of `(pc, opcode)`.
6. Records the RAM `code_version`.

Forward direct jumps may be inlined as `NativeInstruction::InlinedJump`, so a
baseline block can cover a small direct-jump chain when it is statically obvious.

### Optimized Plan

`BlockPlan::optimized_from_cpu(cpu, start_pc)` first tries specialized region
recognizers:

- counted diamond loop
- arithmetic xor toggle loop
- Fibonacci recurrence loop
- store-load-forward loop

If no specialized region matches, it falls back to the normal baseline planner.

These region recognizers are hand-written compiler analyses. When one matches,
the AArch64 backend can emit a custom native loop rather than a generic sequence
of lowered instructions.

### Trace Plan

`BlockPlan::trace_from_cpu(cpu, start_pc)` observes the current guest register
state and follows the currently taken direct-branch path. It emits:

- normal native operations along the observed path
- `TraceGuard` operations for conditional branches
- `TraceLoopGuard` when the observed path loops back to the trace start
- side-exit PCs for paths that were not taken during tracing

Trace plans require at least one side-exit guard. If the planner cannot form a
guarded path, it returns no trace plan and the compiler falls back to optimized
planning.

## Native Lowering Contract

The JIT-family modes only run plans that can be executed without interpreter
fallback:

```rust
if !plan.can_run_without_interpreter_fallback() {
    return Err(JitError::InterpreterFallbackDisabled { pc });
}
```

This is deliberate. A missing lowering is a correctness signal, not something
hidden by executing one instruction in the interpreter.

## Tiers

The current compiler has three tiers:

```text
Baseline
Optimized
Trace
```

### Baseline

Baseline compiles a planned native block with minimal optimization. It is the
first tier for runtime JIT and hybrid misses.

Baseline code:

- emits an AArch64 prologue
- lowers each `NativeInstruction`
- stores the next PC if the block did not end in a native control-flow op
- returns the guest instruction count

### Optimized

Optimized tier uses `optimize_plan()` and backend-specific fast paths.

Current generic optimizer passes are deliberately aggressive and run to a fixed
point before backend emission. They include:

- canonical peephole simplification for identity, zero, copy, and algebraic
  integer operations
- sparse integer constant propagation across the native block
- copy propagation with alias invalidation when a source register is overwritten
- constant folding for integer ALU, shift, compare, multiply, divide, remainder,
  runtime-binary, direct/indirect jump, and branch operations
- branch folding from known register values, including trace guards proven to
  stay on the compiled path
- masked-zero branch fusion and constant masked-branch lowering while preserving
  the architectural destination-register write
- conservative store/load forwarding for identical adjacent 64-bit accesses
- backward liveness removal of overwritten pure integer writes
- liveness barriers around memory operations, syscalls, traps, runtime calls,
  trace side exits, and control-flow exits so fault or side-exit state remains
  guest-visible

The AArch64 backend can also emit:

- register-allocated self loops
- register-allocated optimized blocks
- custom loop regions recognized by `BlockPlan::optimized_from_cpu`
- aggressively unrolled native recurrence loops for recognized integer kernels
- native RV64M division, remainder, and `mulhsu` lowering without the
  `jit_runtime_binary` helper
- fused adjacent `DIV[U]`/`REM[U]` and `DIV[U]W`/`REM[U]W` pairs inside
  register-allocated loops, including pairs separated by one pure integer
  consumer, so quotient and remainder share one hardware divide
- deferred quotient and remainder consumers for the common
  `div; use-quotient; rem; use-remainder` shape, so the consumer instructions
  read the fused divide results directly from scratch registers instead of
  forcing guest-register materialization in the hot loop
- scheduled division-recurrence lowering for loops that reduce several
  independent RV64M quotient/remainder results into a checksum; the hot path
  issues the independent divides first, then consumes remainders and checksum
  updates, with a runtime zero-divisor guard to an out-of-line masked loop
- guarded reciprocal-multiply lowering for recognized division recurrence loops
  whose full-width and word-width divisors are the same positive invariant
  values; the generated block checks the live divisor registers before using
  hoisted magic constants, then falls back to the scheduled hardware-divide or
  masked zero-divisor loop when the cached block is reused with different
  divisor values
- loop-invariant divisor zero masks for register-allocated loops, including a
  guarded nonzero fast path that removes hot quotient divide-by-zero fixups and
  a native masked fallback path for zero divisors; pure loops keep those masks
  in caller-saved temporaries outside the guest register bank, so RV64M loops
  can still use no-prologue guest allocation
- direct fused-remainder consumers for simple binary integer operations, with
  deferred guest-register restoration skipped when the remainder register is
  overwritten before the next read
- leaf prologues for pure register-allocated self loops, so loops that do not
  call Rust runtime helpers avoid saving the frame pointer, link register, CPU
  pointer, and register-file pointer; these loops use the incoming register-file
  pointer argument directly and save only the callee-saved host registers that
  hold guest state
- caller-saved guest-register allocation for pure self loops, eliminating the
  loop prologue entirely when neither guest state nor the dynamic instruction
  count needs callee-saved hosts
- single-instruction return-count scaling for counted loops whose native block
  instruction count is a power of two
- flag-setting decrement backedges for register-allocated loops and selected
  handwritten regions, replacing `sub; cmp; branch` with `subs; branch` when
  the final decrement feeds the loop condition
- single-instruction `csinv` quotient fixups for RISC-V divide-by-zero results
- mask-based `mulhsu` sign correction, avoiding a compare/select pair, and
  direct destination writes when the destination does not alias either source

If an optimized plan is identical to the previous baseline plan and does not
benefit from special register allocation or region handling, the engine can
promote the block metadata without emitting a second native block.

### Trace

Trace tier compiles an observed branch path. It can use:

- a general trace plan with guards and side exits
- a register-allocated trace loop when the trace shape is suitable
- a direct-read trace loop for counted unsigned load/guard/address-increment
  loops when guest RAM exposes one contiguous host-readable range

Side exits flush dirty guest registers, store the side-exit PC, and return to
the Rust dispatcher. The dispatcher then continues with a cached or newly
compiled block for that PC.

## Dynamic Recompilation

Dynamic recompilation is controlled by:

```rust
dynamic_recompilation: bool
trace_compilation: bool
hot_threshold: u64
tier_budgeting: bool
non_loop_hot_threshold_multiplier: u64
min_optimized_block_instructions: usize
background_compilation: bool
```

Defaults:

```text
dynamic_recompilation = true
trace_compilation = true
hot_threshold = 1
tier_budgeting = true
non_loop_hot_threshold_multiplier = 4096
min_optimized_block_instructions = 2
background_compilation = false
```

When a baseline block reaches `hot_threshold`, the engine asks whether it should
promote. The policy is:

- self-loop blocks promote immediately at the hot threshold
- blocks with loop back edges promote immediately at the hot threshold
- blocks with at least `min_optimized_block_instructions` promote at the hot
  threshold
- very small non-loop blocks wait for
  `hot_threshold * non_loop_hot_threshold_multiplier`
- `--jit-no-tier-budget` disables that extra budgeting and promotes as soon as
  the block is hot

If `trace_compilation` is true, hot blocks try `JitTier::Trace`; otherwise they
try `JitTier::Optimized`.

If `--jit-bg-compile` is enabled, optimized or trace compilation can be queued
on background compiler threads while the current baseline block continues to run.
Completed jobs are drained at the start of later JIT steps. Stale or redundant
background results are discarded.

### Aggressive Trace Regions

The trace tier can replace narrow hot traces with handwritten AArch64 regions
when the planner proves the loop shape exactly. These regions still obey normal
guest side-exit semantics, but they avoid generic per-instruction lowering.

Current specialized regions include:

- counted arithmetic loops
- counted diamond loops with closed-form exits when the branch mask is a
  power-of-two-minus-one and the counter decrements by one
- Fibonacci recurrence loops
- store/load forwarding loops
- direct contiguous guest-RAM load traces for unsigned byte, halfword, word, and
  doubleword loads

The Fibonacci recurrence region emits a native 256-iteration unrolled recurrence
rather than calling back into the runtime. Within each unrolled block it splits
the checksum reduction across even and odd temporary accumulators, then merges
them into the guest checksum before the side exit. This keeps the exact guest
final state but shortens the hot checksum dependency chain.

Closed-form arithmetic, counted-diamond fast paths, and Fibonacci recurrence
regions are emitted as no-frame leaf functions. They use caller-saved
temporaries and read/write the guest register file through the incoming
register-file pointer argument, so these regions avoid the normal callee-save
prologue when they do not call back into Rust.

The direct load trace is the most backend-like example. The generated ARM hot
loop loads directly from a proven contiguous host RAM pointer and unrolls four
guarded loads per native iteration. The backend selects `ldrb`, `ldrh`, `ldr w`,
or `ldr x` from the guest load width, then reconstructs guest counter, guest
address, and guest instruction count from the completed byte count at normal
exits and lane-specific side exits. This removes loop-control work from the hot
path while preserving exact guest-visible state for side exits.

### Runtime Versus Compilation Benchmarks

The synthetic Criterion harness has two kinds of JIT-family rows:

- cold rows construct a fresh CPU and engine for each sample, so they include
  compilation, symbol discovery, startup, and native execution
- cached-engine rows reuse the same `JitEngine` cache across samples while
  constructing a fresh CPU, so they isolate native runtime quality much more
  closely
- host Rust rows provide LLVM-compiled references for the direct synthetic
  kernels where a close source-level equivalent exists
- the `synthetic/direct-trace-load` group includes doubleword and byte-load
  rows, so the same direct-read backend is measured both on word-sized integer
  kernels and byte-scanning kernels that are closer to libc/string behavior
- the `synthetic/fibonacci-region` group bypasses ELF and libc startup by
  running the recognized recurrence loop from a raw guest block with a cached
  engine, giving the clearest current signal for native recurrence code quality
- the `synthetic/counted-diamond-region` group measures branch-diamond loops
  that can be collapsed into zero-arm and nonzero-arm trip counts when the mask
  is a power-of-two-minus-one
- the `synthetic/division-region` group measures RV64M `div`, `divu`, `divw`,
  `divuw`, `rem`, `remu`, `remw`, `remuw`, and `mulhsu` lowering in one hot
  loop, including fused div/rem lowering and guarded reciprocal lowering for
  invariant positive divisors

For compiler-backend comparisons against host LLVM code, the cached rows are the
more direct signal. The cold rows remain useful for startup and compilation-cost
work.

## AOT Mode

AOT mode uses the same native backend and `CompiledBlock` cache, but it changes
when compilation happens.

When `JitExecutionMode::Aot` is selected:

```rust
options.dynamic_recompilation = false;
options.background_compilation = false;
```

Before the first execution step, `ensure_precompiled()` calls
`precompile_reachable_blocks(cpu, entry_pc)`.

The default AOT behavior:

1. Compile the entry PC before execution.
2. Use optimized planning for compiler-recognized regions.
3. Use trace planning for guarded trace loops when lazy AOT misses are enabled.
4. Start execution.
5. Lazily compile reached misses as native code if `aot_compile_misses` is true.

By default:

```text
aot_compile_misses = true
aot_symbol_entries = false
aot_linear_sweep = false
```

This means default AOT is not a closed-world full-text compiler. It is
entry-precompiled native execution with native lazy misses.

### Strict AOT

`--aot-no-compile-misses` disables lazy runtime compilation. If execution reaches
a PC that was not already compiled, the engine returns:

```text
JitError::AotMiss { pc }
```

Strict AOT also seeds discovered ELF function and PLT entry points into the AOT
worklist, because there will be no runtime miss compiler to rescue unvisited
targets.

### AOT Symbol Entries

`--aot-symbol-entries` adds ELF function symbols and PLT entries to the
precompile worklist. This is useful when indirect calls would otherwise hide
targets from simple control-flow discovery.

### AOT Linear Sweep

`--aot-linear-sweep` walks executable ELF section ranges and seeds possible block
starts around decoded control-flow boundaries. This is conservative and broader
than entry reachability, but it still uses the same native lowering contract:
unsupported blocks are skipped unless they are required entry blocks.

### AOT Runtime Misses

When runtime misses are allowed, AOT chooses a native tier for the missed block:

- compiler-recognized regions compile as optimized
- if tracing is enabled and a loop trace can be formed, compile as trace
- otherwise loops compile as optimized
- otherwise compile as baseline

This keeps AOT native-only without requiring the initial precompile pass to know
every possible dynamic target.

## JIT Mode

JIT mode does not precompile. The first time execution reaches a PC:

1. No cache entry exists.
2. The mode allows runtime compilation.
3. The engine builds a baseline plan.
4. The backend emits AArch64 code.
5. The block is inserted into the cache.
6. The native block executes immediately.

Subsequent visits hit the cache until the block is stale or hot enough to
promote.

Stale blocks are detected by checking RAM `code_version` and the block
fingerprint. If guest executable memory changed but the fingerprinted opcodes
are unchanged, the block updates its code version and stays valid. If any
fingerprinted opcode changed, the block recompiles.

## Hybrid Mode

Hybrid mode currently follows the same compilation path as JIT mode:

- no precompile
- runtime compilation on cold misses
- dynamic recompilation by default
- trace tier by default
- no interpreter fallback

The differences today are operational:

- the runner defaults to `--hybrid` on supported hosts
- tracing/profile reports identify the engine as `Hybrid`
- the name leaves room for future policy differences

If future work reintroduces interpreter fallback, this section should be updated
carefully. The current code explicitly rejects fallback-required blocks.

## Host Libc Shortcuts

JIT-family modes can map recognized guest libc symbols to host helper paths.

Recognized names include:

- `exit`
- `__libc_start_main`
- `memcmp`
- `memcpy`
- `memmove`
- `memset`
- `printf`
- `puts`
- `strcmp`
- `strlen`
- `strncmp`

The engine discovers symbols from:

- normal ELF symbols
- dynamic ELF symbols
- PLT relocation entries

Host libc handling is controlled by:

```text
--jit-host-libc
--jit-no-host-libc
--jit-libc-start-main-shortcut
--jit-no-libc-start-main-shortcut
```

These shortcuts are still emulator-mediated. They copy data through the guest RAM
and guest filesystem APIs rather than forwarding arbitrary guest syscalls to the
host.

`__libc_start_main` shortcutting is opt-in. When enabled, static guest
`__libc_start_main`, `exit`, `printf`, and `puts` symbols can use
emulator-mediated helpers. The start helper sets up `argc`, `argv`, and `envp`,
then enters guest `main` with a guest return trampoline for `exit`. This is
useful for hot-kernel benchmarking, but the default still exercises guest libc
startup because some programs inspect the initial stack, environment, or
auxiliary vector.

## AArch64 Backend

The AArch64 backend emits raw ARMv8 machine words into an executable memory
mapping.

The emitted function type is:

```rust
unsafe extern "C" fn(*mut RV64GC, *mut u64) -> u64
```

Arguments:

- `x0`: pointer to the `RV64GC` CPU
- `x1`: pointer to the integer register file

Return:

- number of guest instructions executed by the native block

The backend allocates writable anonymous memory, copies the generated code,
flushes the instruction cache, then marks the mapping read-execute with
`mprotect`. After that, the mapping is not mutated.

Generic emitted blocks use runtime helpers for operations that need emulator
services:

- memory loads and stores
- atomics
- floating-point operations
- CSR operations
- `ecall`
- traps and runtime faults

Optimized blocks may use direct guest RAM pointers when the RAM API can prove the
target range is contiguous.

## Native Block Exit Protocol

A native block is responsible for leaving CPU state coherent before it returns:

- guest registers are written back to the CPU register array
- `pc` is set to the next guest PC or side-exit PC
- runtime helper faults set `jit_runtime_fault`
- return value reports executed guest instruction count

After return, Rust-side dispatcher code:

1. Forces `x0 = 0`.
2. Converts pending runtime faults into `JitError::RuntimeFault`.
3. Increments the block execution count.
4. Records trace/profile data.

## Debugging and Observability

Useful runner flags:

```text
--jit-log
--jit-dump
--trace
--trace-limit N
--profile
--profile-top N
```

`--jit-log` prints compiler and execution decisions, including cache activity,
promotion reasons, host-libc calls, and AOT precompile summaries.

`--jit-dump` prints:

- block metadata
- lowered RV64 operations
- emitted AArch64 words
- optimization report

`--trace` and `--profile` use `ExecutionTracer` to report instruction and block
execution behavior across interpreter and JIT-family modes.

## Important Correctness Invariants

1. `x0` must be zero after every interpreter or native step.
2. Native blocks must update guest `pc` before returning.
3. Native blocks must not require interpreter fallback.
4. Compiled block fingerprints must detect self-modifying code.
5. AOT strict mode must reject unknown runtime PCs.
6. Runtime helper faults must return as `JitError::RuntimeFault`, not panic.
7. Host-libc shortcuts must preserve guest-visible filesystem, stdout, stderr,
   memory, and exit behavior.
8. Trace side exits must flush dirty guest registers before returning to the
   dispatcher.

## Current Caveats

- Native compilation is AArch64 Unix only.
- JIT and hybrid are not semantically distinct today beyond mode identity and
  defaults.
- Default AOT allows lazy native compile misses. Use `--aot-no-compile-misses`
  for strict closed-world behavior.
- The optimizer is aggressive inside a planned block or trace, but it still does
  not perform global inter-block SSA, alias analysis, or object-file style link
  optimization.
- Trace compilation follows the current observed branch path and emits guards,
  rather than building a full control-flow graph.
- There is no persistent object-file AOT output yet; AOT currently means
  precompilation into the in-memory native block cache before and during
  emulator execution.
