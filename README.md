<div align="center" id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1 align="center"> RISCVM </h1>
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

<p> RISCVM is an interpreted emulator, but I have plans to implement a JIT for x86_64 and ARM later down the road. </p>

<h2> Installation </h2>

```bash
cargo install --git https://github.com/mateocabanal/riscvm riscvm-runner // Installs the 'riscvm' binary

// OPTIONAL
cargo install --git https://github.com/mateocabanal/riscvm riscvm-debugger // Installs the 'riscvm-debugger' binary
```

<h2> To-Do List </h2>
- [X] ELF execution
- [X] Support statically linked binaries
- [X] Start libc (gets to `int main()` when using libc)
- [X] Start libstdc++ (gets to `int main()` when using libstdc++ (C++))
- [] Start Rust (gets to `fn main()` when using Rust) [see issue](https://github.com/mateocabanal/riscvm/issues/2)
- [] Support dynamically linked binaries 
