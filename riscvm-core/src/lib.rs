pub mod cpu;
pub mod csr;
pub mod debug;
pub mod exception;
pub mod fcsr;
pub mod filesystem;
pub mod jit;
pub mod mmu;
pub mod opcodes;
pub mod ram;
pub mod syscalls;
pub mod tracer;

#[cfg(test)]
mod instruction_tests;

pub fn sign_extend12(n: u32) -> i64 {
    sign_extend(n.into(), 12)
}

pub fn sign_extend(n: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((n << shift) as i64) >> shift
}
