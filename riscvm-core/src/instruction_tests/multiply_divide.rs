use super::common::*;
use crate::cpu::RV64GCInstruction::*;

#[test]
fn rv64m_multiply_variants_return_architectural_halves() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = 0xffff_ffff_ffff_fffe;
    cpu.registers[RS2 as usize] = 3;

    Mul(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_fffa);

    Mulh(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_ffff);

    Mulhsu(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_ffff);

    cpu.registers[RS1 as usize] = u64::MAX;
    cpu.registers[RS2 as usize] = u64::MAX;
    Mulhu(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_fffe);
}

#[test]
fn rv64m_word_multiply_uses_low_word_and_sign_extends() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = 0xffff_ffff;
    cpu.registers[RS2 as usize] = 2;

    Mulw(RD, RS1, RS2).execute_instruction(&mut cpu);

    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_fffe);
}

#[test]
fn rv64m_division_rounds_toward_zero_and_handles_overflow() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = (-22i64) as u64;
    cpu.registers[RS2 as usize] = 7;

    Div(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], (-3i64) as u64);

    Rem(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], (-1i64) as u64);

    cpu.registers[RS1 as usize] = i64::MIN as u64;
    cpu.registers[RS2 as usize] = (-1i64) as u64;
    Div(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], i64::MIN as u64);

    Rem(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);
}

#[test]
fn rv64m_unsigned_division_and_remainder_use_full_xlen() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = u64::MAX;
    cpu.registers[RS2 as usize] = 10;

    Divu(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX / 10);

    Remu(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX % 10);
}

#[test]
fn rv64m_division_by_zero_returns_specified_values() {
    let mut cpu = cpu();

    cpu.registers[RS1 as usize] = 0x1234_5678_9abc_def0;
    cpu.registers[RS2 as usize] = 0;
    Div(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX);

    Rem(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x1234_5678_9abc_def0);

    Divu(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX);

    Remu(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x1234_5678_9abc_def0);
}

#[test]
fn rv64m_word_division_sign_extends_even_for_zero_divisors() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = 0xffff_ffff_8000_0000;
    cpu.registers[RS2 as usize] = 2;

    Divw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_c000_0000);

    Remw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);

    Divuw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32(0x4000_0000));

    Remuw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);

    cpu.registers[RS2 as usize] = 0;
    Divw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX);

    Remw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32(0x8000_0000));

    Divuw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX);

    Remuw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32(0x8000_0000));
}
