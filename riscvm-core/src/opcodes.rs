// RV64I Base Integer Instructions

use bit::BitIndex;

pub const fn is_rv64i_add_instruction(ins: u32) -> bool {
    let add_format: u32 = 0b0000_0000_0000_0000_0000_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == add_format
}

pub const fn is_rv64i_addi_instruction(ins: u32) -> bool {
    let addi_format: u32 = 0b0000_0000_0000_0000_0000_0000_0001_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == addi_format
}

pub const fn is_rv64i_auipc_instruction(ins: u32) -> bool {
    let auipc_format: u32 = 0b0000_0000_0000_0000_0000_0000_0001_0111;
    let mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == auipc_format
}

pub const fn is_rv64i_lui_instruction(ins: u32) -> bool {
    let lui_format: u32 = 0b0000_0000_0000_0000_0000_0000_0011_0111;
    let mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == lui_format
}

pub const fn is_rv64i_slti_instruction(ins: u32) -> bool {
    let slti_format: u32 = 0b0000_0000_0000_0000_0010_0000_0001_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == slti_format
}

pub const fn is_rv64i_sltiu_instruction(ins: u32) -> bool {
    let sltiu_format: u32 = 0b0000_0000_0000_0000_0011_0000_0001_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == sltiu_format
}

pub const fn is_rv64i_xori_instruction(ins: u32) -> bool {
    let xori_format: u32 = 0b0000_0000_0000_0000_0100_0000_0001_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == xori_format
}

pub const fn is_rv64i_ori_instruction(ins: u32) -> bool {
    let ori_format: u32 = 0b0000_0000_0000_0000_0110_0000_0001_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == ori_format
}

pub const fn is_rv64i_andi_instruction(ins: u32) -> bool {
    let andi_format: u32 = 0b0000_0000_0000_0000_0111_0000_0001_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == andi_format
}

pub const fn is_rv64i_slli_instruction(ins: u32) -> bool {
    let slli_format: u32 = 0b0000_0000_0000_0000_0001_0000_0001_0011;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == slli_format
}

pub const fn is_rv64i_srli_instruction(ins: u32) -> bool {
    let srli_format: u32 = 0b0000_0000_0000_0000_0101_0000_0001_0011;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == srli_format
}

pub const fn is_rv64i_srai_instruction(ins: u32) -> bool {
    let srai_format: u32 = 0b0100_0000_0000_0000_0101_0000_0001_0011;
    let mask: u32 = 0b1111_1100_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == srai_format
}

pub const fn is_rv64i_sub_instruction(ins: u32) -> bool {
    let sub_format: u32 = 0b0100_0000_0000_0000_0000_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == sub_format
}

pub const fn is_rv64i_sll_instruction(ins: u32) -> bool {
    let sll_format: u32 = 0b0000_0000_0000_0000_0001_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == sll_format
}

pub const fn is_rv64i_slt_instruction(ins: u32) -> bool {
    let slt_format: u32 = 0b0000_0000_0000_0000_0010_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

    let extracted = ins & mask;
    extracted == slt_format
}

pub const fn is_rv64i_sltu_instruction(ins: u32) -> bool {
    let sltu_format: u32 = 0b0000_0000_0000_0000_0011_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

    let extracted = ins & mask;
    extracted == sltu_format
}

pub const fn is_rv64i_xor_instruction(ins: u32) -> bool {
    let xor_format: u32 = 0b0000_0000_0000_0000_0100_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

    let extracted = ins & mask;
    extracted == xor_format
}

pub const fn is_rv64i_srl_instruction(ins: u32) -> bool {
    let srl_format: u32 = 0b0000_0000_0000_0000_0101_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

    let extracted = ins & mask;
    extracted == srl_format
}

pub const fn is_rv64i_sra_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0000_0000_0000_0101_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_or_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0110_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_and_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0111_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_ecall_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0111_0011;
    let mask: u32 = 0xFFFFFFFF;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_ebreak_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0001_0000_0000_0000_0111_0011;
    let mask: u32 = 0xFFFFFFFF;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_fence_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0000_1111;
    let mask: u32 = 0b1111_0000_0000_1111_1111_1111_1111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_fencei_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0001_0000_0011_0011;
    let mask: u32 = 0b1111_1111_1111_1111_1111_1111_1111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_csrrw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0001_0000_0111_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_csrrs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0010_0000_0111_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_csrrc_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0011_0000_0111_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_csrrwi_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0101_0000_0111_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_csrrsi_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0110_0000_0111_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_csrrci_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0111_0000_0111_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_uret_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0010_0000_0000_0000_0111_0011;
    let mask: u32 = 0xFFFFFFFF;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_lb_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0000_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_lh_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0001_0000_0000_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_lw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0010_0000_0000_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_lbu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0100_0000_0000_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_lhu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0101_0000_0000_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_sb_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0010_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_sh_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0001_0000_0010_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_sw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0010_0000_0010_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_jal_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0110_1111;
    let mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_jalr_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0110_0111;
    let mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_beq_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0110_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_bne_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0001_0000_0110_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_blt_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0100_0000_0110_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_bge_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0101_0000_0110_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_bltu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0110_0000_0110_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_bgeu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0111_0000_0110_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_ld_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0011_0000_0000_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_sd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0011_0000_0010_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_addiw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0001_1011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_sraiw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0000_0000_0000_0101_0000_0001_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_addw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_subw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0000_0000_0000_0000_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_slliw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0001_0000_0001_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_srliw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0101_0000_0001_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_sllw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0001_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_srlw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0101_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_sraw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0000_0000_0000_0101_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64i_lwu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0110_0000_0000_0011;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_mul_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0000_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_mulh_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0001_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_mulhsu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0010_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_mulhu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0011_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_div_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0100_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_divu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0101_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_rem_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0110_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_remu_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0111_0000_0011_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_mulw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0000_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_divw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0100_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_divuw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0101_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_remw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0110_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64m_remuw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0111_0000_0011_1011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

// RV64A

pub const fn is_rv64a_lrw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1001_1111_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_scw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_1000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoswapw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_1000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoaddw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoxorw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoandw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0110_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoorw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amominw_instruction(ins: u32) -> bool {
    let format: u32 = 0b1000_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amomaxw_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amominuw_instruction(ins: u32) -> bool {
    let format: u32 = 0b1100_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amomaxuw_instruction(ins: u32) -> bool {
    let format: u32 = 0b1110_0000_0000_0000_0010_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_lrd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_scd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_1000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoswapd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_1000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoaddd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoxord_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoandd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0110_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amoord_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amomind_instruction(ins: u32) -> bool {
    let format: u32 = 0b1000_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amomaxd_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amominud_instruction(ins: u32) -> bool {
    let format: u32 = 0b1100_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64a_amomaxud_instruction(ins: u32) -> bool {
    let format: u32 = 0b1110_0000_0000_0000_0011_0000_0010_1111;
    let mask: u32 = 0b1111_1000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

// RV64FD

pub const fn is_rv64f_fmadds_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0100_0011;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmsubs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0100_0111;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fnmsubs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0100_1011;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fnmadds_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0100_1111;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fadds_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsubs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_1000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmuls_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fdivs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_1000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsqrts_instruction(ins: u32) -> bool {
    let format: u32 = 0b0101_1000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsgnjs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsgnjns_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0000_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsgnjxs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0000_0000_0000_0010_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmins_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_1000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmaxs_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_1000_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtws_instruction(ins: u32) -> bool {
    let format: u32 = 0b1100_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtwus_instruction(ins: u32) -> bool {
    let format: u32 = 0b1100_0000_0001_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmvxw_instruction(ins: u32) -> bool {
    let format: u32 = 0b1110_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_feqs_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0000_0000_0000_0010_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_flts_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0000_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fles_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fclasss_instruction(ins: u32) -> bool {
    let format: u32 = 0b1110_0000_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtsw_instruction(ins: u32) -> bool {
    let format: u32 = 0b1101_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtswu_instruction(ins: u32) -> bool {
    let format: u32 = 0b1101_0000_0001_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmvwx_instruction(ins: u32) -> bool {
    let format: u32 = 0b1111_0000_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

// RV64D

pub const fn is_rv64f_fmaddd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0000_0000_0100_0011;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmsubd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0000_0000_0100_0111;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fnmsubd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0000_0000_0100_1011;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fnmaddd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0000_0000_0100_1111;
    let mask: u32 = 0b0000_0110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_faddd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsubd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_1010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmuld_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fdivd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0001_1010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsqrtd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0101_1010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsgnjd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsgnjnd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0010_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsgnjxd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_0010_0000_0000_0010_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmind_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_1010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmaxd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0010_1010_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtwd_instruction(ins: u32) -> bool {
    let format: u32 = 0b1100_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtwud_instruction(ins: u32) -> bool {
    let format: u32 = 0b1100_0010_0001_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtdw_instruction(ins: u32) -> bool {
    let format: u32 = 0b1101_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtdwu_instruction(ins: u32) -> bool {
    let format: u32 = 0b1101_0010_0001_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fmvxd_instruction(ins: u32) -> bool {
    let format: u32 = 0b1110_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_feqd_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0010_0000_0000_0010_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fltd_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0010_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fled_instruction(ins: u32) -> bool {
    let format: u32 = 0b1010_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fclassd_instruction(ins: u32) -> bool {
    let format: u32 = 0b1110_0010_0000_0000_0001_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtsd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0000_0001_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fcvtds_instruction(ins: u32) -> bool {
    let format: u32 = 0b0100_0010_0000_0000_0000_0000_0101_0011;
    let mask: u32 = 0b1111_1111_1111_0000_0000_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_flw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0010_0000_0000_0111;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsw_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0010_0000_0010_0111;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fld_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0011_0000_0000_0111;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64f_fsd_instruction(ins: u32) -> bool {
    let format: u32 = 0b0000_0000_0000_0000_0011_0000_0010_0111;
    let mask: u32 = 0b0000_0000_0000_0000_0111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

// NOTE: RV64C
pub const fn is_rv64c_addi4spn_instruction(ins: u16) -> bool {
    let format: u16 = 0b0000_0000_0000_0000;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_fld_instruction(ins: u16) -> bool {
    let format: u16 = 0b0010_0000_0000_0000;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_lw_instruction(ins: u16) -> bool {
    let format: u16 = 0b0100_0000_0000_0000;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_ld_instruction(ins: u16) -> bool {
    let format: u16 = 0b0110_0000_0000_0000;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_fsd_instruction(ins: u16) -> bool {
    let format: u16 = 0b1010_0000_0000_0000;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_sw_instruction(ins: u16) -> bool {
    let format: u16 = 0b1100_0000_0000_0000;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_sd_instruction(ins: u16) -> bool {
    let format: u16 = 0b1110_0000_0000_0000;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_nop_instruction(ins: u16) -> bool {
    let format: u16 = 0b0000_0000_0000_0001;
    let mask: u16 = u16::MAX;

    let extracted = ins & mask;
    extracted == format
}

pub fn is_rv64c_addi_instruction(ins: u16) -> bool {
    let format: u16 = 0b0000_0000_0000_0001;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_addiw_instruction(ins: u16) -> bool {
    let format: u16 = 0b0010_0000_0000_0001;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_li_instruction(ins: u16) -> bool {
    let format: u16 = 0b0100_0000_0000_0001;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_addi16sp_instruction(ins: u16) -> bool {
    let format: u16 = 0b0110_0001_0000_0001;
    let mask: u16 = 0b1110_1111_1000_0011;

    let extracted = ins & mask;
    extracted == format
}

// WARNING: This must come before matching c.addi16sp
pub const fn is_rv64c_lui_instruction(ins: u16) -> bool {
    let format: u16 = 0b0110_0000_0000_0001;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_srli_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_0000_0000_0001;
    let mask: u16 = 0b1110_1100_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_srai_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_0100_0000_0001;
    let mask: u16 = 0b1110_1100_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_andi_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_1000_0000_0001;
    let mask: u16 = 0b1110_1100_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_sub_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_1100_0000_0001;
    let mask: u16 = 0b1111_1100_0110_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_xor_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_1100_0010_0001;
    let mask: u16 = 0b1111_1100_0110_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_or_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_1100_0100_0001;
    let mask: u16 = 0b1111_1100_0110_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_and_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_1100_0110_0001;
    let mask: u16 = 0b1111_1100_0110_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_subw_instruction(ins: u16) -> bool {
    let format: u16 = 0b1001_1100_0000_0001;
    let mask: u16 = 0b1111_1100_0110_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_addw_instruction(ins: u16) -> bool {
    let format: u16 = 0b1001_1100_0010_0001;
    let mask: u16 = 0b1111_1100_0110_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_j_instruction(ins: u16) -> bool {
    let format: u16 = 0b1010_0000_0000_0001;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_beqz_instruction(ins: u16) -> bool {
    let format: u16 = 0b1100_0000_0000_0001;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_bnez_instruction(ins: u16) -> bool {
    let format: u16 = 0b1110_0000_0000_0001;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_slli_instruction(ins: u16) -> bool {
    let format: u16 = 0b0000_0000_0000_0010;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_fldsp_instruction(ins: u16) -> bool {
    let format: u16 = 0b0010_0000_0000_0010;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_lwsp_instruction(ins: u16) -> bool {
    let format: u16 = 0b0100_0000_0000_0010;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_ldsp_instruction(ins: u16) -> bool {
    let format: u16 = 0b0110_0000_0000_0010;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

// WARNING: Must be checked before c.mv
pub const fn is_rv64c_jr_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_0000_0000_0010;
    let mask: u16 = 0b1111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}

pub const fn is_rv64c_mv_instruction(ins: u16) -> bool {
    let format: u16 = 0b1000_0000_0000_0010;
    let mask: u16 = 0b1111_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_ebreak_instruction(ins: u16) -> bool {
    let format: u16 = 0b1001_0000_0000_0010;
    let mask: u16 = 0b1111_1111_1111_1111;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_jalr_instruction(ins: u16) -> bool {
    let format: u16 = 0b1001_0000_0000_0010;
    let mask: u16 = 0b1111_0000_0111_1111;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_add_instruction(ins: u16) -> bool {
    let format: u16 = 0b1001_0000_0000_0010;
    let mask: u16 = 0b1111_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_fsdsp_instruction(ins: u16) -> bool {
    let format: u16 = 0b1010_0000_0000_0010;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_swsp_instruction(ins: u16) -> bool {
    let format: u16 = 0b1100_0000_0000_0010;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}
pub const fn is_rv64c_sdsp_instruction(ins: u16) -> bool {
    let format: u16 = 0b1110_0000_0000_0010;
    let mask: u16 = 0b1110_0000_0000_0011;

    let extracted = ins & mask;
    extracted == format
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    #[allow(clippy::vec_init_then_push)]
    fn test_all_instructions() -> Result<(), Box<dyn std::error::Error>> {
        let mut map: Vec<(&str, u32)> = Vec::new();
        map.push(("add", 0x00a282b3));
        map.push(("addi", 0x00500013));
        map.push(("auipc", 0x00123297));
        map.push(("slti", 0x0052a293));
        map.push(("sltiu", 0x0052b293));
        map.push(("xori", 0x0052c293));
        map.push(("ori", 0x2142e293));
        map.push(("andi", 0x0051fb93));
        map.push(("slli", 0x02d61393));
        map.push(("srli", 0x02d65393));
        map.push(("srai", 0x42d65393));
        map.push(("sub", 0x405302b3));
        map.push(("sll", 0x019192b3));
        map.push(("slt", 0x0191a2b3));
        map.push(("sltu", 0x0191b2b3));
        map.push(("xor", 0x001d4fb3));
        map.push(("srl", 0x0191d2b3));
        map.push(("sra", 0x4191d2b3));
        map.push(("or", 0x0191e2b3));
        map.push(("and", 0x0191f2b3));
        map.push(("ecall", 0x00000073));
        map.push(("ebreak", 0x00100073));
        map.push(("lb", 0x000e0283));
        map.push(("lh", 0x000e1283));
        map.push(("lw", 0x000e2283));
        map.push(("lbu", 0x000e4283));
        map.push(("lhu", 0x000e5283));
        map.push(("sb", 0x005e0023));
        map.push(("sh", 0x005e1023));
        map.push(("sw", 0x005e2023));
        map.push(("jal", 0x00400c6f));
        map.push(("jalr", 0x00428c67));
        map.push(("beq", 0x005c0263));
        map.push(("bne", 0x005c1263));
        map.push(("bge", 0x005c5263));
        map.push(("bgeu", 0x005c7263));
        map.push(("blt", 0x005c4263));
        map.push(("bltu", 0x005c6263));
        map.push(("ld", 0x00043283));
        map.push(("sd", 0x00543023));
        map.push(("addiw", 0x0007879b));
        map.push(("slliw", 0x0052929b));
        map.push(("srliw", 0x0052d29b));
        map.push(("sraiw", 0x41f7d79b));
        map.push(("subw", 0x40f007bb));
        map.push(("addw", 0x00f707bb));
        map.push(("sllw", 0x005292bb));
        map.push(("srlw", 0x0052d2bb));
        map.push(("sraw", 0x4052d2bb));
        map.push(("lwu", 0x0002e283));
        map.push(("mul", 0x025282b3));
        map.push(("mulh", 0x025292b3));
        map.push(("mulhu", 0x0252b2b3));
        map.push(("mulhsu", 0x0252a2b3));
        map.push(("div", 0x0252c2b3));
        map.push(("divu", 0x0252d2b3));
        map.push(("rem", 0x0252e2b3));
        map.push(("remu", 0x0252f2b3));
        map.push(("mulw", 0x035282bb));
        map.push(("divw", 0x0352c2bb));
        map.push(("divuw", 0x0352d2bb));
        map.push(("remw", 0x0352e2bb));
        map.push(("remuw", 0x0352f2bb));
        map.push(("lrw", 0x1002a2af));
        map.push(("scw", 0x1852a2af));
        map.push(("amoswapw", 0x0852a2af));
        map.push(("amoaddw", 0x0052a2af));
        map.push(("amoxorw", 0x2052a2af));
        map.push(("amoorw", 0x4052a2af));
        map.push(("amoandw", 0x6052a2af));
        map.push(("amominw", 0x8052a2af));
        map.push(("amominuw", 0xc052a2af));
        map.push(("amomaxw", 0xa052a2af));
        map.push(("amomaxuw", 0xe052a2af));
        map.push(("lrd", 0x1002b2af));
        map.push(("scd", 0x185032af));
        map.push(("amoswapd", 0x085332af));
        map.push(("amoandd", 0x605332af));
        map.push(("amoaddd", 0x005332af));
        map.push(("amoxord", 0x205332af));
        map.push(("amoord", 0x405332af));
        map.push(("amomind", 0x805332af));
        map.push(("amominud", 0xc05332af));
        map.push(("amomaxd", 0xa05332af));
        map.push(("amomaxud", 0xe05332af));
        map.push(("fmadds", 0x1852f343));
        map.push(("fmsubs", 0x1852f347));
        map.push(("fnmsubs", 0x1852f34b));
        map.push(("fnmadds", 0x1852f34f));
        map.push(("fsubs", 0x0852f353));
        map.push(("fadds", 0x0010f053));
        map.push(("fmuls", 0x1052f353));
        map.push(("fdivs", 0x1852f353));
        map.push(("fsqrts", 0x5802f353));
        map.push(("fsgnjs", 0x205302d3));
        map.push(("fsgnjns", 0x205312d3));
        map.push(("fsgnjxs", 0x205322d3));
        map.push(("fmins", 0x285302d3));
        map.push(("fmaxs", 0x285312d3));
        map.push(("fcvtws", 0xc00372d3));
        map.push(("fcvtwus", 0xc01372d3));
        map.push(("fmvwx", 0xf00082d3));
        map.push(("feqs", 0xa07322d3));
        map.push(("flts", 0xa07312d3));
        map.push(("fles", 0xa07302d3));
        map.push(("fclasss", 0xe00312d3));
        map.push(("fcvtsw", 0xd002f1d3));
        map.push(("fcvtswu", 0xd012f1d3));
        map.push(("fmvxw", 0xe00280d3));

        // NOTE: RV64D
        map.push(("fmaddd", 0x2252f343));
        map.push(("fmsubd", 0x2252f347));
        map.push(("fnmsubd", 0x2252f34b));
        map.push(("fnmaddd", 0x2252f34f));
        map.push(("faddd", 0x0252f353));
        map.push(("fsubd", 0x0a52f353));
        map.push(("fmuld", 0x1252f353));
        map.push(("fdivd", 0x1a52f353));
        map.push(("fsqrtd", 0x5a02f353));
        map.push(("fsgnjd", 0x22628353));
        map.push(("fsgnjnd", 0x22629353));
        map.push(("fsgnjxd", 0x2262a353));
        map.push(("fmind", 0x2a628353));
        map.push(("fmaxd", 0x2a629353));
        map.push(("fcvtsd", 0x4012f353));
        map.push(("fcvtds", 0x4202f353));
        map.push(("feqd", 0xa2522353));
        map.push(("fled", 0xa2520353));
        map.push(("fltd", 0xa2521353));
        map.push(("fclassd", 0xe2021353));
        map.push(("fcvtwd", 0xc20272d3));
        map.push(("fcvtwud", 0xc21272d3));
        map.push(("fcvtdwu", 0xd21272d3));
        map.push(("fcvtdw", 0xd20272d3));
        map.push(("flw", 0x00002287));
        map.push(("fsw", 0x00502027));
        map.push(("fld", 0x00003287));
        map.push(("fsd", 0x0052b227));

        // NOTE: RV64C
        map.push(("caddi4spn", 0x0020));
        map.push(("cfld", 0x2008));
        map.push(("clw", 0x4008));
        map.push(("cld", 0x6008));
        map.push(("cfsd", 0xa008));
        map.push(("csw", 0xc008));
        map.push(("csd", 0xe008));
        map.push(("cnop", 0x0001));
        map.push(("caddi", 0x0515));
        map.push(("caddiw", 0x2515));
        map.push(("cli", 0x4515));
        map.push(("caddi16sp", 0x6141));
        map.push(("clui", 0x6521));
        map.push(("csrli", 0x8121));
        map.push(("csrai", 0x8521));
        map.push(("candi", 0x8921));
        map.push(("csub", 0x8d01));
        map.push(("cxor", 0x8d21));
        map.push(("cor", 0x8d41));
        map.push(("cand", 0x8d61));
        map.push(("csubw", 0x9d01));
        map.push(("caddw", 0x9d21));
        map.push(("cbeqz", 0xc801));
        map.push(("cbnez", 0xe801));
        map.push(("cslli", 0x0442));
        map.push(("cfldsp", 0x2442));
        map.push(("clwsp", 0x4442));
        map.push(("cldsp", 0x6442));
        map.push(("cjr", 0x8402));
        map.push(("cmv", 0x843a));
        map.push(("cebreak", 0x9002));
        map.push(("cjalr", 0x9402));
        map.push(("cadd", 0x942a));
        map.push(("cfsdsp", 0xa42a));
        map.push(("cswsp", 0xc42a));
        map.push(("csdsp", 0xe416));

        for (name, ins) in map {
            if (is_rv64i_add_instruction(ins) && name != "add")
                || (!is_rv64i_add_instruction(ins) && name == "add")
            {
                return Err(format!("{name}: {ins:#08x}, is not an add instruction!").into());
            }

            if is_rv64i_addi_instruction(ins) && name != "addi"
                || (!is_rv64i_addi_instruction(ins) && name == "addi")
            {
                return Err(format!("{name}: {ins:#08x}, is not an addi instruction!").into());
            }

            if is_rv64i_auipc_instruction(ins) && name != "auipc"
                || (!is_rv64i_auipc_instruction(ins) && name == "auipc")
            {
                return Err(format!("{name}: {ins:#08x}, is not an auipc instruction!").into());
            }

            if is_rv64i_slti_instruction(ins) && name != "slti"
                || (!is_rv64i_slti_instruction(ins) && name == "slti")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sltui instruction!").into());
            }

            if is_rv64i_sltiu_instruction(ins) && name != "sltiu"
                || (!is_rv64i_sltiu_instruction(ins) && name == "sltiu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sltui instruction!").into());
            }

            if is_rv64i_xori_instruction(ins) && name != "xori"
                || (!is_rv64i_xori_instruction(ins) && name == "xori")
            {
                return Err(format!("{name}: {ins:#08x}, is not an xori instruction!").into());
            }

            if is_rv64i_ori_instruction(ins) && name != "ori"
                || (!is_rv64i_ori_instruction(ins) && name == "ori")
            {
                return Err(format!("{name}: {ins:#08x}, is not an ori instruction!").into());
            }

            if is_rv64i_andi_instruction(ins) && name != "andi"
                || (!is_rv64i_andi_instruction(ins) && name == "andi")
            {
                return Err(format!("{name}: {ins:#08x}, is not an andi instruction!").into());
            }

            if is_rv64i_slli_instruction(ins) && name != "slli"
                || (!is_rv64i_slli_instruction(ins) && name == "slli")
            {
                return Err(format!("{name}: {ins:#08x}, is not an slli instruction!").into());
            }

            if is_rv64i_srli_instruction(ins) && name != "srli"
                || (!is_rv64i_srli_instruction(ins) && name == "srli")
            {
                return Err(format!("{name}: {ins:#08x}, is not an srli instruction!").into());
            }

            if is_rv64i_srai_instruction(ins) && name != "srai"
                || (!is_rv64i_srai_instruction(ins) && name == "srai")
            {
                return Err(format!("{name}: {ins:#08x}, is not an srai instruction!").into());
            }

            if is_rv64i_sub_instruction(ins) && name != "sub"
                || (!is_rv64i_sub_instruction(ins) && name == "sub")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sub instruction!").into());
            }

            if is_rv64i_sll_instruction(ins) && name != "sll"
                || (!is_rv64i_sll_instruction(ins) && name == "sll")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sll instruction!").into());
            }

            if is_rv64i_slt_instruction(ins) && name != "slt"
                || (!is_rv64i_slt_instruction(ins) && name == "slt")
            {
                return Err(format!("{name}: {ins:#08x}, is not an slt instruction!").into());
            }

            if is_rv64i_sltu_instruction(ins) && name != "sltu"
                || (!is_rv64i_sltu_instruction(ins) && name == "sltu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sltu instruction!").into());
            }

            if is_rv64i_xor_instruction(ins) && name != "xor"
                || (!is_rv64i_xor_instruction(ins) && name == "xor")
            {
                return Err(format!("{name}: {ins:#08x}, is not an xor instruction!").into());
            }

            if is_rv64i_and_instruction(ins) && name != "and"
                || (!is_rv64i_and_instruction(ins) && name == "and")
            {
                return Err(format!("{name}: {ins:#08x}, is not an and instruction!").into());
            }

            if is_rv64i_ecall_instruction(ins) && name != "ecall"
                || (!is_rv64i_ecall_instruction(ins) && name == "ecall")
            {
                return Err(format!("{name}: {ins:#08x}, is not an ecall instruction!").into());
            }

            if is_rv64i_ebreak_instruction(ins) && name != "ebreak"
                || (!is_rv64i_ebreak_instruction(ins) && name == "ebreak")
            {
                return Err(format!("{name}: {ins:#08x}, is not an ebreak instruction!").into());
            }

            if is_rv64i_lb_instruction(ins) && name != "lb"
                || (!is_rv64i_lb_instruction(ins) && name == "lb")
            {
                return Err(format!("{name}: {ins:#08x}, is not an lb instruction!").into());
            }

            if is_rv64i_lh_instruction(ins) && name != "lh"
                || (!is_rv64i_lh_instruction(ins) && name == "lh")
            {
                return Err(format!("{name}: {ins:#08x}, is not an lh instruction!").into());
            }

            if is_rv64i_lw_instruction(ins) && name != "lw"
                || (!is_rv64i_lw_instruction(ins) && name == "lw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an lw instruction!").into());
            }

            if is_rv64i_lbu_instruction(ins) && name != "lbu"
                || (!is_rv64i_lbu_instruction(ins) && name == "lbu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an lbu instruction!").into());
            }

            if is_rv64i_lhu_instruction(ins) && name != "lhu"
                || (!is_rv64i_lhu_instruction(ins) && name == "lhu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an ebreak instruction!").into());
            }

            if is_rv64i_sb_instruction(ins) && name != "sb"
                || (!is_rv64i_sb_instruction(ins) && name == "sb")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sb instruction!").into());
            }

            if is_rv64i_sh_instruction(ins) && name != "sh"
                || (!is_rv64i_sh_instruction(ins) && name == "sh")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sh instruction!").into());
            }

            if is_rv64i_sw_instruction(ins) && name != "sw"
                || (!is_rv64i_sw_instruction(ins) && name == "sw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sw instruction!").into());
            }

            if is_rv64i_jal_instruction(ins) && name != "jal"
                || (!is_rv64i_jal_instruction(ins) && name == "jal")
            {
                return Err(format!("{name}: {ins:#08x}, is not an jal instruction!").into());
            }

            if is_rv64i_jalr_instruction(ins) && name != "jalr"
                || (!is_rv64i_jalr_instruction(ins) && name == "jalr")
            {
                return Err(format!("{name}: {ins:#08x}, is not an jalr instruction!").into());
            }

            if is_rv64i_beq_instruction(ins) && name != "beq"
                || (!is_rv64i_beq_instruction(ins) && name == "beq")
            {
                return Err(format!("{name}: {ins:#08x}, is not an beq instruction!").into());
            }

            if is_rv64i_bne_instruction(ins) && name != "bne"
                || (!is_rv64i_bne_instruction(ins) && name == "bne")
            {
                return Err(format!("{name}: {ins:#08x}, is not an bne instruction!").into());
            }

            if is_rv64i_blt_instruction(ins) && name != "blt"
                || (!is_rv64i_blt_instruction(ins) && name == "blt")
            {
                return Err(format!("{name}: {ins:#08x}, is not an blt instruction!").into());
            }

            if is_rv64i_bge_instruction(ins) && name != "bge"
                || (!is_rv64i_bge_instruction(ins) && name == "bge")
            {
                return Err(format!("{name}: {ins:#08x}, is not an bge instruction!").into());
            }

            if is_rv64i_bltu_instruction(ins) && name != "bltu"
                || (!is_rv64i_bltu_instruction(ins) && name == "bltu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an bltu instruction!").into());
            }

            if is_rv64i_bgeu_instruction(ins) && name != "bgeu"
                || (!is_rv64i_bgeu_instruction(ins) && name == "bgeu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an bgeu instruction!").into());
            }

            if is_rv64i_ld_instruction(ins) && name != "ld"
                || (!is_rv64i_ld_instruction(ins) && name == "ld")
            {
                return Err(format!("{name}: {ins:#08x}, is not an ld instruction!").into());
            }

            if is_rv64i_sd_instruction(ins) && name != "sd"
                || (!is_rv64i_sd_instruction(ins) && name == "sd")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sd instruction!").into());
            }

            if is_rv64i_addiw_instruction(ins) && name != "addiw"
                || (!is_rv64i_addiw_instruction(ins) && name == "addiw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an addiw instruction!").into());
            }

            if is_rv64i_slliw_instruction(ins) && name != "slliw"
                || (!is_rv64i_slliw_instruction(ins) && name == "slliw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an slliw instruction!").into());
            }

            if is_rv64i_srliw_instruction(ins) && name != "srliw"
                || (!is_rv64i_srliw_instruction(ins) && name == "srliw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an srliw instruction!").into());
            }

            if is_rv64i_sraiw_instruction(ins) && name != "sraiw"
                || (!is_rv64i_sraiw_instruction(ins) && name == "sraiw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sraiw instruction!").into());
            }

            if is_rv64i_addw_instruction(ins) && name != "addw"
                || (!is_rv64i_addw_instruction(ins) && name == "addw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an addw instruction!").into());
            }

            if is_rv64i_subw_instruction(ins) && name != "subw"
                || (!is_rv64i_subw_instruction(ins) && name == "subw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an subw instruction!").into());
            }

            if is_rv64i_sllw_instruction(ins) && name != "sllw"
                || (!is_rv64i_sllw_instruction(ins) && name == "sllw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sllw instruction!").into());
            }

            if is_rv64i_srlw_instruction(ins) && name != "srlw"
                || (!is_rv64i_srlw_instruction(ins) && name == "srlw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an srlw instruction!").into());
            }

            if is_rv64i_sraw_instruction(ins) && name != "sraw"
                || (!is_rv64i_sraw_instruction(ins) && name == "sraw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sraw instruction!").into());
            }

            if is_rv64i_lwu_instruction(ins) && name != "lwu"
                || (!is_rv64i_lwu_instruction(ins) && name == "lwu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an srlw instruction!").into());
            }

            if is_rv64m_mul_instruction(ins) && name != "mul"
                || (!is_rv64m_mul_instruction(ins) && name == "mul")
            {
                return Err(format!("{name}: {ins:#08x}, is not an mul instruction!").into());
            }

            if is_rv64m_mulh_instruction(ins) && name != "mulh"
                || (!is_rv64m_mulh_instruction(ins) && name == "mulh")
            {
                return Err(format!("{name}: {ins:#08x}, is not an mulh instruction!").into());
            }

            if is_rv64m_mulhu_instruction(ins) && name != "mulhu"
                || (!is_rv64m_mulhu_instruction(ins) && name == "mulhu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an mulhu instruction!").into());
            }

            if is_rv64m_mulhsu_instruction(ins) && name != "mulhsu"
                || (!is_rv64m_mulhsu_instruction(ins) && name == "mulhsu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an mulhsu instruction!").into());
            }

            if is_rv64m_div_instruction(ins) && name != "div"
                || (!is_rv64m_div_instruction(ins) && name == "div")
            {
                return Err(format!("{name}: {ins:#08x}, is not an div instruction!").into());
            }

            if is_rv64m_divu_instruction(ins) && name != "divu"
                || (!is_rv64m_divu_instruction(ins) && name == "divu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an divu instruction!").into());
            }

            if is_rv64m_rem_instruction(ins) && name != "rem"
                || (!is_rv64m_rem_instruction(ins) && name == "rem")
            {
                return Err(format!("{name}: {ins:#08x}, is not an rem instruction!").into());
            }

            if is_rv64m_remu_instruction(ins) && name != "remu"
                || (!is_rv64m_remu_instruction(ins) && name == "remu")
            {
                return Err(format!("{name}: {ins:#08x}, is not an remu instruction!").into());
            }

            {
                let value = is_rv64m_mulw_instruction(ins);
                let ins_name = "mulw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64m_divw_instruction(ins);
                let ins_name = "divw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64m_divuw_instruction(ins);
                let ins_name = "divuw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64m_remw_instruction(ins);
                let ins_name = "remw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64m_remuw_instruction(ins);
                let ins_name = "remuw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            // RV64A

            {
                let value = is_rv64a_lrw_instruction(ins);
                let ins_name = "lrw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_scw_instruction(ins);
                let ins_name = "scw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_amoaddw_instruction(ins);
                let ins_name = "amoaddw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_amoandw_instruction(ins);
                let ins_name = "amoandw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_amoxorw_instruction(ins);
                let ins_name = "amoxorw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_amoorw_instruction(ins);
                let ins_name = "amoorw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_amoswapw_instruction(ins);
                let ins_name = "amoswapw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_amominw_instruction(ins);
                let ins_name = "amominw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64a_amominuw_instruction(ins);
                let ins_name = "amominuw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amomaxw_instruction(ins);
                let ins_name = "amomaxw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amomaxuw_instruction(ins);
                let ins_name = "amomaxuw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_lrd_instruction(ins);
                let ins_name = "lrd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_scd_instruction(ins);
                let ins_name = "scd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amoandd_instruction(ins);
                let ins_name = "amoandd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amoaddd_instruction(ins);
                let ins_name = "amoaddd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amoswapd_instruction(ins);
                let ins_name = "amoswapd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amoxord_instruction(ins);
                let ins_name = "amoxord";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amoord_instruction(ins);
                let ins_name = "amoord";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amomaxd_instruction(ins);
                let ins_name = "amomaxd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amomaxud_instruction(ins);
                let ins_name = "amomaxud";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amomind_instruction(ins);
                let ins_name = "amomind";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64a_amominud_instruction(ins);
                let ins_name = "amominud";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            // RV64F

            {
                let value = is_rv64f_fmadds_instruction(ins);
                let ins_name = "fmadds";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_fmsubs_instruction(ins);
                let ins_name = "fmsubs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_fnmadds_instruction(ins);
                let ins_name = "fnmadds";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fnmsubs_instruction(ins);
                let ins_name = "fnmsubs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_fadds_instruction(ins);
                let ins_name = "fadds";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_fsubs_instruction(ins);
                let ins_name = "fsubs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmuls_instruction(ins);
                let ins_name = "fmuls";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fdivs_instruction(ins);
                let ins_name = "fdivs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsqrts_instruction(ins);
                let ins_name = "fsqrts";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsgnjs_instruction(ins);
                let ins_name = "fsgnjs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsgnjns_instruction(ins);
                let ins_name = "fsgnjns";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsgnjxs_instruction(ins);
                let ins_name = "fsgnjxs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmins_instruction(ins);
                let ins_name = "fmins";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmaxs_instruction(ins);
                let ins_name = "fmaxs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtws_instruction(ins);
                let ins_name = "fcvtws";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtwus_instruction(ins);
                let ins_name = "fcvtwus";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmvxw_instruction(ins);
                let ins_name = "fmvxw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_feqs_instruction(ins);
                let ins_name = "feqs";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_flts_instruction(ins);
                let ins_name = "flts";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fles_instruction(ins);
                let ins_name = "fles";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fclasss_instruction(ins);
                let ins_name = "fclasss";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtsw_instruction(ins);
                let ins_name = "fcvtsw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtswu_instruction(ins);
                let ins_name = "fcvtswu";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmvwx_instruction(ins);
                let ins_name = "fmvwx";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            // NOTE: RV64D

            {
                let value = is_rv64f_fmaddd_instruction(ins);
                let ins_name = "fmaddd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_fmsubd_instruction(ins);
                let ins_name = "fmsubd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_fnmaddd_instruction(ins);
                let ins_name = "fnmaddd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fnmsubd_instruction(ins);
                let ins_name = "fnmsubd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_faddd_instruction(ins);
                let ins_name = "faddd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64f_fsubd_instruction(ins);
                let ins_name = "fsubd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmuld_instruction(ins);
                let ins_name = "fmuld";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fdivd_instruction(ins);
                let ins_name = "fdivd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsqrtd_instruction(ins);
                let ins_name = "fsqrtd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsgnjd_instruction(ins);
                let ins_name = "fsgnjd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsgnjnd_instruction(ins);
                let ins_name = "fsgnjnd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsgnjxd_instruction(ins);
                let ins_name = "fsgnjxd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmind_instruction(ins);
                let ins_name = "fmind";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fmaxd_instruction(ins);
                let ins_name = "fmaxd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtwd_instruction(ins);
                let ins_name = "fcvtwd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtwud_instruction(ins);
                let ins_name = "fcvtwud";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_feqd_instruction(ins);
                let ins_name = "feqd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fltd_instruction(ins);
                let ins_name = "fltd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fled_instruction(ins);
                let ins_name = "fled";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fclassd_instruction(ins);
                let ins_name = "fclassd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtsd_instruction(ins);
                let ins_name = "fcvtsd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtds_instruction(ins);
                let ins_name = "fcvtds";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtdw_instruction(ins);
                let ins_name = "fcvtdw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fcvtwd_instruction(ins);
                let ins_name = "fcvtwd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_flw_instruction(ins);
                let ins_name = "flw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsw_instruction(ins);
                let ins_name = "fsw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fld_instruction(ins);
                let ins_name = "fld";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64f_fsd_instruction(ins);
                let ins_name = "fsd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            let c_ins = ins as u16;

            // NOTE: RV64C
            {
                let value = is_rv64c_addi4spn_instruction(c_ins);
                let ins_name = "caddi4spn";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_fld_instruction(c_ins);
                let ins_name = "cfld";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_lw_instruction(c_ins);
                let ins_name = "clw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_ld_instruction(c_ins);
                let ins_name = "cld";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_fsd_instruction(c_ins);
                let ins_name = "cfsd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_sw_instruction(c_ins);
                let ins_name = "csw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_sd_instruction(c_ins);
                let ins_name = "csd";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_nop_instruction(c_ins);
                let ins_name = "cnop";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_addi_instruction(c_ins);
                let ins_name = "caddi";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_addiw_instruction(c_ins);
                let ins_name = "caddiw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_li_instruction(c_ins);
                let ins_name = "cli";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_addi16sp_instruction(c_ins);
                let ins_name = "caddi16sp";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            // These two ops overlap, caddi16sp will always match clui.
            // Only way to differentiate is to match addi16sp first.
            if name != "caddi16sp" {
                {
                    let value = is_rv64c_lui_instruction(c_ins);
                    let ins_name = "clui";

                    if (value && name != ins_name) || (!value && name == ins_name) {
                        return Err(format!(
                            "{name}: {c_ins:#08x}, is not an {ins_name} instruction!"
                        )
                        .into());
                    }
                }
            }

            {
                let value = is_rv64c_srli_instruction(c_ins);
                let ins_name = "csrli";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_srai_instruction(c_ins);
                let ins_name = "csrai";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_andi_instruction(c_ins);
                let ins_name = "candi";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_sub_instruction(c_ins);
                let ins_name = "csub";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_xor_instruction(c_ins);
                let ins_name = "cxor";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_or_instruction(c_ins);
                let ins_name = "cor";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_subw_instruction(c_ins);
                let ins_name = "csubw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_addw_instruction(c_ins);
                let ins_name = "caddw";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_beqz_instruction(c_ins);
                let ins_name = "cbeqz";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_bnez_instruction(c_ins);
                let ins_name = "cbnez";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_slli_instruction(c_ins);
                let ins_name = "cslli";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_fldsp_instruction(c_ins);
                let ins_name = "cfldsp";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_lwsp_instruction(c_ins);
                let ins_name = "clwsp";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            {
                let value = is_rv64c_ldsp_instruction(c_ins);
                let ins_name = "cldsp";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_jr_instruction(c_ins);
                let ins_name = "cjr";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            if name != "cjr" {
                {
                    let value = is_rv64c_mv_instruction(c_ins);
                    let ins_name = "cmv";

                    if (value && name != ins_name) || (!value && name == ins_name) {
                        return Err(format!(
                            "{name}: {c_ins:#08x}, is not an {ins_name} instruction!"
                        )
                        .into());
                    }
                }
            }
            {
                let value = is_rv64c_ebreak_instruction(c_ins);
                let ins_name = "cebreak";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }

            if name != "cebreak" {
                {
                    let value = is_rv64c_jalr_instruction(c_ins);
                    let ins_name = "cjalr";

                    if (value && name != ins_name) || (!value && name == ins_name) {
                        return Err(format!(
                            "{name}: {c_ins:#08x}, is not an {ins_name} instruction!"
                        )
                        .into());
                    }
                }
            }
            if name != "cjalr" && name != "cebreak" {
                {
                    let value = is_rv64c_add_instruction(c_ins);
                    let ins_name = "cadd";

                    if (value && name != ins_name) || (!value && name == ins_name) {
                        return Err(format!(
                            "{name}: {c_ins:#08x}, is not an {ins_name} instruction!"
                        )
                        .into());
                    }
                }
            }
            {
                let value = is_rv64c_fsdsp_instruction(c_ins);
                let ins_name = "cfsdsp";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_swsp_instruction(c_ins);
                let ins_name = "cswsp";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
            {
                let value = is_rv64c_sdsp_instruction(c_ins);
                let ins_name = "csdsp";

                if (value && name != ins_name) || (!value && name == ins_name) {
                    return Err(
                        format!("{name}: {c_ins:#08x}, is not an {ins_name} instruction!").into(),
                    );
                }
            }
        }

        Ok(())
    }
}
