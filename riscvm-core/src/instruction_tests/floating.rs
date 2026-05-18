use super::common::*;
use crate::cpu::{RV64GCInstruction::*, RV64GC};

fn set_f32(cpu: &mut RV64GC, reg: u8, value: f32) {
    cpu.float_registers[reg as usize] = f32_box(value);
}

fn f32_bits(cpu: &RV64GC, reg: u8) -> u32 {
    cpu.float_registers[reg as usize] as u32
}

fn set_f64(cpu: &mut RV64GC, reg: u8, value: f64) {
    cpu.float_registers[reg as usize] = value.to_bits();
}

#[test]
fn rv64f_basic_arithmetic_and_square_root_write_nan_boxed_single_results() {
    let mut cpu = cpu();
    set_f32(&mut cpu, RS1, 6.0);
    set_f32(&mut cpu, RS2, 2.0);

    Fadds(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(8.0));

    Fsubs(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(4.0));

    Fmuls(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(12.0));

    Fdivs(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(3.0));

    Fsqrts(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(6.0f32.sqrt()));
}

#[test]
fn rv64f_fused_multiply_add_variants_match_riscv_sign_definitions() {
    let mut cpu = cpu();
    set_f32(&mut cpu, RS1, 2.0);
    set_f32(&mut cpu, RS2, 3.0);
    set_f32(&mut cpu, RS3, 4.0);

    Fmadds(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(10.0));

    Fmsubs(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(2.0));

    Fnmsubs(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(-2.0));

    Fnmadds(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(-10.0));
}

#[test]
fn rv64f_sign_injection_uses_magnitude_from_rs1_and_sign_from_rs2() {
    let mut cpu = cpu();
    set_f32(&mut cpu, RS1, 1.5);
    set_f32(&mut cpu, RS2, -2.0);

    Fsgnjs(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(f32_bits(&cpu, RD), (-1.5f32).to_bits());

    Fsgnjns(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(f32_bits(&cpu, RD), 1.5f32.to_bits());

    Fsgnjxs(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(f32_bits(&cpu, RD), (-1.5f32).to_bits());

    assert_eq!(cpu.float_registers[RD as usize] >> 32, 0xffff_ffff);
}

#[test]
fn rv64f_min_max_compare_class_and_moves_follow_single_precision_rules() {
    let mut cpu = cpu();
    set_f32(&mut cpu, RS1, 1.5);
    set_f32(&mut cpu, RS2, -2.0);

    Fmins(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(f32_bits(&cpu, RD), (-2.0f32).to_bits());

    Fmaxs(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(f32_bits(&cpu, RD), 1.5f32.to_bits());

    Feqs(RD, RS1, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Flts(RD, RS2, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Fles(RD, RS2, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Fclasss(RD, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1 << 6);

    Fmvxw(RD, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32((-2.0f32).to_bits()));

    cpu.registers[RS1 as usize] = 0xbf80_0000;
    Fmvwx(RD, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(-1.0));
}

#[test]
fn rv64f_conversions_loads_and_stores_preserve_specified_widths() {
    let mut cpu = cpu_with_memory();
    set_f32(&mut cpu, RS1, -42.75);

    Fcvtws(RD, 1, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], (-42i64) as u64);

    set_f32(&mut cpu, RS1, 0x8000_0000u32 as f32);
    Fcvtwus(RD, 1, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32(0x8000_0000));

    set_f32(&mut cpu, RS1, -42.75);
    Fcvtls(RD, 1, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], (-42i64) as u64);

    set_f32(&mut cpu, RS1, 42.75);
    Fcvtlus(RD, 1, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 42);

    cpu.registers[RS1 as usize] = (-7i64) as u64;
    Fcvtsw(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(-7.0));

    cpu.registers[RS1 as usize] = 7;
    Fcvtswu(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(7.0));

    cpu.registers[RS1 as usize] = (-7_000_000_000i64) as u64;
    Fcvtsl(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(
        cpu.float_registers[RD as usize],
        f32_box(-7_000_000_000i64 as f32)
    );

    cpu.registers[RS1 as usize] = 1u64 << 40;
    Fcvtslu(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(
        cpu.float_registers[RD as usize],
        f32_box((1u64 << 40) as f32)
    );

    cpu.registers[RS1 as usize] = BASE;
    cpu.ram.write_word(BASE + 4, 1.25f32.to_bits()).unwrap();
    Flw(RD, RS1, 4).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(1.25));

    Fsw(RS1, RD, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.ram.read_word(BASE + 8).unwrap(), 1.25f32.to_bits());
}

#[test]
fn rv64d_arithmetic_fused_compare_class_and_conversions_follow_current_model() {
    let mut cpu = cpu();
    set_f64(&mut cpu, RS1, 6.0);
    set_f64(&mut cpu, RS2, 2.0);
    set_f64(&mut cpu, RS3, 1.5);

    Faddd(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 8.0f64.to_bits());

    Fsubd(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 4.0f64.to_bits());

    Fmuld(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 12.0f64.to_bits());

    Fdivd(RD, 0, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 3.0f64.to_bits());

    Fsqrtd(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 6.0f64.sqrt().to_bits());

    Fmaddd(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 13.5f64.to_bits());

    Fmsubd(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 10.5f64.to_bits());

    Fnmaddd(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], (-13.5f64).to_bits());

    Fnmsubd(RD, 0, RS1, RS2, RS3).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], (-10.5f64).to_bits());

    set_f64(&mut cpu, RS2, -2.0);
    Fmind(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], (-2.0f64).to_bits());

    Fmaxd(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 6.0f64.to_bits());

    Feqd(RD, RS1, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Fltd(RD, RS2, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Fled(RD, RS2, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Fclassd(RD, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1 << 6);

    Fcvtsd(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], f32_box(6.0));

    Fcvtwd(RD, 1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], (-2i64) as u64);

    Fcvtwud(RD, 1, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 6);

    cpu.registers[RS1 as usize] = (-7i64) as u64;
    Fcvtdw(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], (-7.0f64).to_bits());

    cpu.registers[RS1 as usize] = 7;
    Fcvtdwu(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 7.0f64.to_bits());
}

#[test]
fn rv64d_implemented_double_transfers_and_sign_injection_follow_spec() {
    let mut cpu = cpu_with_memory();

    cpu.registers[RS1 as usize] = BASE;
    cpu.float_registers[RS2 as usize] = (-3.5f64).to_bits();
    Fsd(RS1, RS2, 8).execute_instruction(&mut cpu);
    assert_eq!(
        cpu.ram.read_doubleword(BASE + 8).unwrap(),
        (-3.5f64).to_bits()
    );

    cpu.ram
        .write_doubleword(BASE + 16, 2.5f64.to_bits())
        .unwrap();
    Fld(RD, RS1, 16).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 2.5f64.to_bits());

    Fmvxd(RD, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], (-3.5f64).to_bits());

    cpu.float_registers[RS1 as usize] = 1.5f64.to_bits();
    cpu.float_registers[RS2 as usize] = (-2.0f64).to_bits();

    Fsgnjd(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], (-1.5f64).to_bits());

    Fsgnjnd(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 1.5f64.to_bits());

    Fsgnjxd(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], (-1.5f64).to_bits());

    cpu.float_registers[RS1 as usize] = f32_box(1.25);
    Fcvtds(RD, 0, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], (1.25f64).to_bits());
}

#[test]
fn decoder_keeps_adjacent_float_variants_distinct() {
    let cpu = cpu();

    assert!(matches!(cpu.find_instruction(0x205302d3), Fsgnjs(..)));
    assert!(matches!(cpu.find_instruction(0x205312d3), Fsgnjns(..)));
    assert!(matches!(cpu.find_instruction(0x205322d3), Fsgnjxs(..)));
    assert!(matches!(cpu.find_instruction(0xc00372d3), Fcvtws(..)));
    assert!(matches!(cpu.find_instruction(0xc01372d3), Fcvtwus(..)));
    assert!(matches!(cpu.find_instruction(0xc02372d3), Fcvtls(..)));
    assert!(matches!(cpu.find_instruction(0xc03372d3), Fcvtlus(..)));
    assert!(matches!(cpu.find_instruction(0xd022f1d3), Fcvtsl(..)));
    assert!(matches!(cpu.find_instruction(0xd032f1d3), Fcvtslu(..)));
    assert!(matches!(cpu.find_instruction(0x2252f343), Fmaddd(..)));
    assert!(matches!(cpu.find_instruction(0x2252f347), Fmsubd(..)));
    assert!(matches!(cpu.find_instruction(0x2252f34b), Fnmsubd(..)));
    assert!(matches!(cpu.find_instruction(0x2252f34f), Fnmaddd(..)));
    assert!(matches!(cpu.find_instruction(0x0252f353), Faddd(..)));
    assert!(matches!(cpu.find_instruction(0x0a52f353), Fsubd(..)));
    assert!(matches!(cpu.find_instruction(0x1252f353), Fmuld(..)));
    assert!(matches!(cpu.find_instruction(0x1a52f353), Fdivd(..)));
    assert!(matches!(cpu.find_instruction(0x5a02f353), Fsqrtd(..)));
    assert!(matches!(cpu.find_instruction(0x00833287), Fld(..)));
    assert!(matches!(cpu.find_instruction(0x22628353), Fsgnjd(..)));
    assert!(matches!(cpu.find_instruction(0x22629353), Fsgnjnd(..)));
    assert!(matches!(cpu.find_instruction(0x2262a353), Fsgnjxd(..)));
    assert!(matches!(cpu.find_instruction(0x2a628353), Fmind(..)));
    assert!(matches!(cpu.find_instruction(0x2a629353), Fmaxd(..)));
    assert!(matches!(cpu.find_instruction(0x4012f353), Fcvtsd(..)));
    assert!(matches!(cpu.find_instruction(0xa2522353), Feqd(..)));
    assert!(matches!(cpu.find_instruction(0xa2520353), Fled(..)));
    assert!(matches!(cpu.find_instruction(0xa2521353), Fltd(..)));
    assert!(matches!(cpu.find_instruction(0xe2021353), Fclassd(..)));
    assert!(matches!(cpu.find_instruction(0xc20272d3), Fcvtwd(..)));
    assert!(matches!(cpu.find_instruction(0xc21272d3), Fcvtwud(..)));
    assert!(matches!(cpu.find_instruction(0xd20272d3), Fcvtdw(..)));
    assert!(matches!(cpu.find_instruction(0xd21272d3), Fcvtdwu(..)));
}
