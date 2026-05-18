use super::common::*;
use crate::cpu::{RV64GCInstruction::*, RV64GCRegAbiName::*};

#[test]
fn rv64c_integer_register_operations_expand_like_base_ops() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = 10;
    cpu.registers[RS2 as usize] = 3;

    Cadd(RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 13);

    Csub(RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 10);

    Cxor(RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 9);

    Cor(RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 11);

    Cand(RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 3);

    Cmv(RD, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 3);
}

#[test]
fn rv64c_immediate_and_shift_operations_match_expanded_forms() {
    let mut cpu = cpu();

    cpu.registers[RS1 as usize] = 10;
    Caddi(RS1, -11).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], u64::MAX);

    Cli(RD, 0b11_1111).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX);

    Clui(RD, 0x2_000).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x2000);

    cpu.registers[Sp] = 0x8000;
    Caddi16sp(-16).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Sp], 0x7ff0);

    Caddi4spn(RD, 64).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x8030);

    cpu.registers[RS1 as usize] = 0xf0;
    Candi(RS1, 0x0f).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 0);

    cpu.registers[RS1 as usize] = 1;
    Cslli(RS1, 6).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 64);

    Csrli(RS1, 2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 16);

    cpu.registers[RS1 as usize] = 0xffff_ffff_ffff_ff00;
    Csrai(RS1, 4).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 0xffff_ffff_ffff_fff0);
}

#[test]
fn rv64c_word_operations_sign_extend_low_word_results() {
    let mut cpu = cpu();

    cpu.registers[RS1 as usize] = 0xffff_ffff;
    Caddiw(RS1, 1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 0);

    cpu.registers[RS1 as usize] = 0x7fff_ffff;
    cpu.registers[RS2 as usize] = 1;
    Caddw(RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 0xffff_ffff_8000_0000);

    Csubw(RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RS1 as usize], 0x7fff_ffff);
}

#[test]
fn rv64c_loads_and_stores_use_stack_or_compact_register_offsets() {
    let mut cpu = cpu_with_memory();
    cpu.registers[Sp] = BASE + 0x100;
    cpu.registers[RS1 as usize] = BASE + 0x200;
    cpu.registers[RS2 as usize] = 0x1122_3344_5566_7788;
    cpu.float_registers[RS2 as usize] = 0x8877_6655_4433_2211;

    Csdsp(RS2, 8).execute_instruction(&mut cpu);
    Cldsp(RD, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x1122_3344_5566_7788);

    Cfldsp(RD, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 0x1122_3344_5566_7788);

    Cswsp(RS2, 16).execute_instruction(&mut cpu);
    Clwsp(RD, 16).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32(0x5566_7788));

    Csd(RS1, RS2, 24).execute_instruction(&mut cpu);
    Cld(RD, RS1, 24).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x1122_3344_5566_7788);

    Cfld(RD, RS1, 24).execute_instruction(&mut cpu);
    assert_eq!(cpu.float_registers[RD as usize], 0x1122_3344_5566_7788);

    Csw(RS1, RS2, 32).execute_instruction(&mut cpu);
    Clw(RD, RS1, 32).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32(0x5566_7788));

    Cfsd(RS1, RS2, 40).execute_instruction(&mut cpu);
    assert_eq!(
        cpu.ram.read_doubleword(BASE + 0x228).unwrap(),
        0x8877_6655_4433_2211
    );

    Cfsdsp(RS2, 48).execute_instruction(&mut cpu);
    assert_eq!(
        cpu.ram.read_doubleword(BASE + 0x130).unwrap(),
        0x8877_6655_4433_2211
    );
}

#[test]
fn rv64c_decoder_recognizes_double_float_stack_loads_from_real_rust_codegen() {
    let cpu = cpu();

    assert!(matches!(cpu.find_instruction(0x0000_247a), Cfldsp(8, 408)));
}

#[test]
fn rv64c_jumps_and_branches_use_halfword_pc_adjustment() {
    let mut cpu = cpu();

    cpu.registers[Pc] = 0x1000;
    cpu.registers[RS1 as usize] = 0x2222;
    Cjr(RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x2220);

    cpu.registers[Pc] = 0x1000;
    Cjalr(RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Ra], 0x1002);
    assert_eq!(cpu.registers[Pc], 0x2220);

    cpu.registers[Pc] = 0x5000;
    Cj(0x100).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x50fe);

    cpu.registers[Pc] = 0x5000;
    cpu.registers[RS1 as usize] = 0;
    Cbeqz(RS1, 0x100).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x4efe);

    cpu.registers[Pc] = 0x5000;
    Cbnez(RS1, 0x100).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x5000);

    cpu.registers[RS1 as usize] = 1;
    Cbnez(RS1, 0x100).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x4efe);
}

#[test]
fn rv64c_nop_has_no_architectural_effect() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = 0x1234;

    Cnop.execute_instruction(&mut cpu);

    assert_eq!(cpu.registers[RS1 as usize], 0x1234);
}

#[test]
fn rv64c_ebreak_enters_the_debug_trap_path_without_advancing_pc() {
    let mut cpu = cpu();
    cpu.registers[Pc] = 0x1234;

    Cebreak.execute_instruction(&mut cpu);

    assert!(cpu.should_quit);
    assert_eq!(cpu.registers[Pc], 0x1234);
}
