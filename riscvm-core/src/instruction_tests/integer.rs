use super::common::*;
use crate::cpu::{RV64GCInstruction::*, RV64GCRegAbiName::*};
use crate::fcsr::RoundingMode;

#[test]
fn rv64i_register_immediate_arithmetic_matches_spec() {
    let mut cpu = cpu();

    cpu.registers[RS1 as usize] = u64::MAX;
    Addi(RD, RS1, 1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);

    cpu.registers[RS1 as usize] = 0xffff_ffff_ffff_fffe;
    Slti(RD, RS1, 1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    cpu.registers[RS1 as usize] = 0x1000;
    Sltiu(RD, RS1, 0xfff).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    cpu.registers[RS1 as usize] = 0x55aa;
    Xori(RD, RS1, -1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], !0x55aa);

    Ori(RD, RS1, 0x0f0f).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x5faf);

    Andi(RD, RS1, 0x0f0f).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x050a);
}

#[test]
fn rv64i_shift_immediates_and_register_shifts_use_rv64_widths() {
    let mut cpu = cpu();

    cpu.registers[RS1 as usize] = 1;
    Slli(RD, RS1, 63).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x8000_0000_0000_0000);

    cpu.registers[RS1 as usize] = 0x8000_0000_0000_0000;
    Srli(RD, RS1, 63).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Srai(RD, RS1, 63).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], u64::MAX);

    cpu.registers[RS1 as usize] = 1;
    cpu.registers[RS2 as usize] = 70;
    Sll(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 64);

    cpu.registers[RS1 as usize] = 0x8000_0000_0000_0000;
    Srl(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x0200_0000_0000_0000);

    Sra(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xfe00_0000_0000_0000);
}

#[test]
fn rv64i_register_register_arithmetic_and_logic_match_spec() {
    let mut cpu = cpu();

    cpu.registers[RS1 as usize] = u64::MAX;
    cpu.registers[RS2 as usize] = 2;
    Add(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Sub(RD, RS2, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 3);

    cpu.registers[RS1 as usize] = 0xffff_ffff_ffff_ffff;
    cpu.registers[RS2 as usize] = 1;
    Slt(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Sltu(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);

    cpu.registers[RS1 as usize] = 0b1010;
    cpu.registers[RS2 as usize] = 0b1100;
    Xor(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0b0110);

    Or(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0b1110);

    And(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0b1000);
}

#[test]
fn rv64i_lui_and_auipc_sign_extend_u_immediates() {
    let mut cpu = cpu();

    Lui(RD, -4096).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_f000);

    cpu.registers[Pc] = 0x2000;
    Auipc(RD, -4096).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x1000);
}

#[test]
fn rv64i_loads_sign_or_zero_extend_and_stores_write_low_bits() {
    let mut cpu = cpu_with_memory();
    cpu.registers[RS1 as usize] = BASE + 0x80;

    cpu.ram.write_byte(BASE + 0x7f, 0x80).unwrap();
    Lb(RD, RS1, -1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_ff80);

    Lbu(RD, RS1, -1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x80);

    cpu.ram.write_halfword(BASE + 0x82, 0x8001).unwrap();
    Lh(RD, RS1, 2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_ffff_8001);

    Lhu(RD, RS1, 2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x8001);

    cpu.ram.write_word(BASE + 0x84, 0x8000_0001).unwrap();
    Lw(RD, RS1, 4).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_8000_0001);

    Lwu(RD, RS1, 4).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x8000_0001);

    cpu.ram
        .write_doubleword(BASE + 0x88, 0xfeed_face_cafe_beef)
        .unwrap();
    Ld(RD, RS1, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xfeed_face_cafe_beef);

    cpu.registers[RS2 as usize] = 0x1122_3344_5566_7788;
    Sb(RS1, RS2, 0x10).execute_instruction(&mut cpu);
    Sh(RS1, RS2, 0x12).execute_instruction(&mut cpu);
    Sw(RS1, RS2, 0x14).execute_instruction(&mut cpu);
    Sd(RS1, RS2, 0x18).execute_instruction(&mut cpu);

    assert_eq!(cpu.ram.read_byte(BASE + 0x90).unwrap(), 0x88);
    assert_eq!(cpu.ram.read_halfword(BASE + 0x92).unwrap(), 0x7788);
    assert_eq!(cpu.ram.read_word(BASE + 0x94).unwrap(), 0x5566_7788);
    assert_eq!(
        cpu.ram.read_doubleword(BASE + 0x98).unwrap(),
        0x1122_3344_5566_7788
    );
}

#[test]
fn rv64i_decoder_recognizes_halfword_loads_from_real_rust_codegen() {
    let cpu = cpu();

    assert!(matches!(cpu.find_instruction(0x20c4_1503), Lh(10, 8, 524)));
}

#[test]
fn rv64i_decoder_recognizes_register_arithmetic_shift_right_from_real_rust_codegen() {
    let cpu = cpu();

    assert!(matches!(cpu.find_instruction(0x40c5_5533), Sra(10, 10, 12)));
}

#[test]
fn rv64i_jumps_and_branches_update_pc_with_encoded_offsets() {
    let mut cpu = cpu();

    cpu.registers[Pc] = 0x1000;
    Jal(RD, 0x40).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x1004);
    assert_eq!(cpu.registers[Pc], 0x103c);

    cpu.registers[Pc] = 0x2000;
    cpu.registers[RS1 as usize] = 0x1235;
    Jalr(RD, RS1, 0).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x2004);
    assert_eq!(cpu.registers[Pc], 0x1230);

    cpu.registers[Pc] = 0x5000;
    cpu.registers[RS1 as usize] = 7;
    cpu.registers[RS2 as usize] = 7;
    Beq(RS1, RS2, 0x1000).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x3ffc);

    cpu.registers[Pc] = 0x5000;
    Bne(RS1, RS2, 0x1000).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x5000);

    cpu.registers[RS1 as usize] = u64::MAX;
    cpu.registers[RS2 as usize] = 1;
    Blt(RS1, RS2, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x5004);

    cpu.registers[Pc] = 0x5000;
    Bge(RS2, RS1, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x5004);

    cpu.registers[Pc] = 0x5000;
    Bltu(RS2, RS1, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x5004);

    cpu.registers[Pc] = 0x5000;
    Bgeu(RS1, RS2, 8).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Pc], 0x5004);
}

#[test]
fn rv64i_word_instructions_ignore_upper_inputs_and_sign_extend_results() {
    let mut cpu = cpu();

    cpu.registers[RS1 as usize] = 0xffff_ffff;
    Addiw(RD, RS1, 1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);

    cpu.registers[RS1 as usize] = 0x0000_0000_0800_0001;
    Slliw(RD, RS1, 4).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_8000_0010);

    cpu.registers[RS1 as usize] = 0x0000_0000_8000_0001;
    Srliw(RD, RS1, 1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x4000_0000);

    Sraiw(RD, RS1, 1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_c000_0000);

    cpu.registers[RS1 as usize] = 0xffff_ffff;
    cpu.registers[RS2 as usize] = 2;
    Addw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 1);

    Subw(RD, RS2, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 3);

    cpu.registers[RS1 as usize] = 0x8000_0001;
    cpu.registers[RS2 as usize] = 36;
    Sllw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x10);

    Srlw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x0800_0000);

    Sraw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0xffff_ffff_f800_0000);
}

#[test]
fn rv64i_fence_is_a_noop_in_the_single_hart_interpreter_model() {
    let mut cpu = cpu();
    cpu.registers[RS1 as usize] = 0x1234;
    cpu.registers[RD as usize] = 0x5678;

    Fence(0b1111, 0b1111).execute_instruction(&mut cpu);
    FenceI.execute_instruction(&mut cpu);

    assert_eq!(cpu.registers[RS1 as usize], 0x1234);
    assert_eq!(cpu.registers[RD as usize], 0x5678);
}

#[test]
fn zicsr_fcsr_instructions_read_modify_write_without_touching_x0() {
    let mut cpu = cpu();

    Csrrwi(RD, 0b1_0101, 0x001).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);
    assert_eq!(cpu.fcsr.flags(), 0b1_0101);

    Csrrsi(RD, 0b0_1010, 0x001).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0b1_0101);
    assert_eq!(cpu.fcsr.flags(), 0b1_1111);

    Csrrci(RD, 0b1_0000, 0x001).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0b1_1111);
    assert_eq!(cpu.fcsr.flags(), 0b0_1111);

    cpu.registers[RS1 as usize] = 0;
    Csrrs(RD, RS1, 0x001).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0b0_1111);
    assert_eq!(cpu.fcsr.flags(), 0b0_1111);

    cpu.registers[RS1 as usize] = (u64::from(RoundingMode::Rup.bits()) << 5) | 0b0_0011;
    Csrrw(Zero as u8, RS1, 0x003).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[Zero as usize], 0);
    assert_eq!(cpu.fcsr.frm, RoundingMode::Rup);
    assert_eq!(cpu.fcsr.flags(), 0b0_0011);
}

#[test]
fn zicsr_decoder_recognizes_fence_i_and_csr_forms() {
    let cpu = cpu();

    assert!(matches!(cpu.find_instruction(0x0000_100f), FenceI));
    assert!(matches!(
        cpu.find_instruction(0x0012_9073),
        Csrrw(0, 5, 0x001)
    ));
    assert!(matches!(
        cpu.find_instruction(0x0022_a0f3),
        Csrrs(1, 5, 0x002)
    ));
    assert!(matches!(
        cpu.find_instruction(0x0032_b173),
        Csrrc(2, 5, 0x003)
    ));
    assert!(matches!(
        cpu.find_instruction(0x0012_d1f3),
        Csrrwi(3, 5, 0x001)
    ));
    assert!(matches!(
        cpu.find_instruction(0x0012_e273),
        Csrrsi(4, 5, 0x001)
    ));
    assert!(matches!(
        cpu.find_instruction(0x0012_f2f3),
        Csrrci(5, 5, 0x001)
    ));
}
