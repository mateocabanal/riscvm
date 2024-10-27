pub mod cpu;
pub mod csr;
pub mod exception;
pub mod mmu;
pub mod opcodes;
pub mod ram;

pub fn sign_extend12(n: u32) -> i64 {
    sign_extend(n.into(), 12)
}

pub fn sign_extend(n: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((n << shift) as i64) >> shift
}
