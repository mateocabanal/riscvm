<div align="center" id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1 align="center"> 
        RISCVM 
        <a href="https://github.com/mateocabanal/riscvm/actions/workflows/rust.yml">
          <img align="center" src="https://github.com/mateocabanal/riscvm/actions/workflows/rust.yml/badge.svg" />
        </a>
      </h1>
    </summary>
  </ul>
</div>

<p align="center"> A RV64GC userspace emulator, written in Rust 🦀. </p>

<hr/>

<h2> Description </h2>

<h4> What is RISCVM? </h4>

<p>
  RISCVM is a userspace emulator. It emulates the RVGC64 unprivileged spec, so this is not meant to run any baremetal software (e.g kernels). 
  It only runs Linux ELF files.
</p>

<h4> How is RISCVM emulating RVGC64? </h4>

<p> RISCVM has an interpreter and an AArch64 JIT backend with dynamic recompilation. Hot guest blocks are promoted into an optimized tier, an experimental tracing tier can record hot direct-branch paths with native side exits, and JIT-family modes reject blocks that would require interpreter fallback. </p>

<h2> Installation </h2>

```bash
cargo install --git https://github.com/mateocabanal/riscvm riscvm-runner # Installs the 'riscvm' binary

# OPTIONAL
cargo install --git https://github.com/mateocabanal/riscvm riscvm-debugger # Installs the 'riscvm-debugger' binary
```
<h2> Usage </h2>

`riscvm <ELF_FILE>`

The default execution engine is the AArch64 hybrid engine on supported hosts and the interpreter elsewhere. Hybrid lazily JITs discovered blocks, dynamically recompiles hot code, and fails clearly if native lowering coverage is missing.

Select an explicit engine:

```bash
riscvm --hybrid <ELF_FILE>
riscvm --jit <ELF_FILE>
riscvm --aot <ELF_FILE>
riscvm --interp <ELF_FILE>
```

AOT precompiles the entry block as optimized native code, then lazily compiles reached misses as optimized native blocks without allowing interpreter fallback. Use `--aot-symbol-entries` to seed discovered ELF function and PLT entry points before execution, `--aot-linear-sweep` to also seed every executable-section block start up front for conservative whole-text discovery, or `--aot-no-compile-misses` to make AOT fail if runtime reaches a block that was not precompiled. JIT-family modes can forward supported dynamic PLT libc calls such as `__libc_start_main`, `printf`, and `puts` through the host; statically linked libc runtime and stdio entry points stay guest-owned so initialization, buffering, and exit semantics match the interpreter. Pure static memory/string helpers such as `memcpy`, `memset`, and `strlen` can still use host helpers. Pass `--jit-no-host-libc` to force guest libc execution for all supported calls. Dynamic recompilation is enabled by default for JIT and hybrid modes. Use `--jit-hot-threshold N` to tune hot-block promotion and `--jit-no-dynarec` to force baseline-only JIT execution. Pass `--jit-trace` to let hot direct-branch paths promote into the experimental trace tier, which compiles the observed path across basic-block boundaries and emits guarded side exits for untaken branches. JIT, hybrid, and AOT all reject blocks that would require interpreter fallback. Optimized tier budgeting is also enabled by default: self-loop blocks promote at the normal hot threshold, while non-loop blocks need a stronger hotness signal before promotion. Use `--jit-no-tier-budget`, `--jit-non-loop-hot-multiplier N`, and `--jit-min-optimized-block-instructions N` to tune that policy. Use `--jit-bg-compile` to compile optimized hot blocks on background host threads while baseline blocks keep running; tune that path with `--jit-compiler-threads N` and `--jit-compile-queue-limit N`.

Run the debugger:

```bash
riscvm-debugger <ELF_FILE> [guest-args...]
```

The debugger starts in a TUI with register, stop, status, and disassembly panes.
Use `n` or `s` to step, `c` to continue, Enter to run to the selected instruction,
and `:` to enter debugger commands.

Debugger commands include:

```text
step|s [n]                 step one or n guest instructions
continue|c [addr]          run until exit, breakpoint, watchpoint, or addr
continue limit <n>         run at most n instructions
break|b <addr>             set breakpoint
b list | b delete <addr>   list or delete breakpoints
watch reg <reg>            stop when a register changes
watch mem <addr> [width]   stop when memory changes
regs | reg <reg>           inspect registers
set reg <reg> <value>      edit register
x <addr> [count] [width]   read memory
mem write <addr> <value> [width]
disasm|u [addr] [count]    disassemble guest instructions
status | reset | quit
```

For scripts and regression checks, run debugger commands without the TUI:

```bash
riscvm-debugger --batch -ex "break pc" -ex "continue limit 10" <ELF_FILE>
```

<h2> Features </h2>

- [X] ELF execution
- [X] Support statically linked binaries
- [X] AArch64 JIT with dynamic recompilation
- [X] Start libc (gets to `int main()` when using libc)
- [X] Start libstdc++ (gets to `int main()` when using libstdc++ (C++))
- [ ] Start Rust (gets to `fn main()` when using Rust) [see issue](https://github.com/mateocabanal/riscvm/issues/2)
- [ ] Support dynamically linked binaries
- [ ] Multi-threading support
