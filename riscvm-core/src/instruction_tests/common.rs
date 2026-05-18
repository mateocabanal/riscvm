use crate::cpu::{RV64GCRegAbiName::*, RV64GC};
use crate::ram::MemoryRegion;

pub const BASE: u64 = 0x1000;
pub const RD: u8 = A0 as u8;
pub const RS1: u8 = A1 as u8;
pub const RS2: u8 = A2 as u8;
pub const RS3: u8 = A3 as u8;

pub fn cpu() -> RV64GC {
    RV64GC::new()
}

pub fn cpu_with_memory() -> RV64GC {
    let mut cpu = RV64GC::new();
    cpu.ram
        .add_region(MemoryRegion::new(BASE, 0x1000, vec![0; 0x1000]))
        .unwrap();
    cpu
}

pub fn sx32(value: u32) -> u64 {
    i64::from(value as i32) as u64
}

pub fn f32_box(value: f32) -> u64 {
    f32_box_bits(value.to_bits())
}

pub fn f32_box_bits(bits: u32) -> u64 {
    0xffff_ffff_0000_0000 | u64::from(bits)
}
