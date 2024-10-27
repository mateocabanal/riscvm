// RV64I Base Integer Instructions

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
    let mask: u32 = 0b1111_1110_0000_0000_0111_0000_0011_0011;

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
    let format: u32 = 0b0000_0000_0010_0000_0000_0000_0011_0011;
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

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_all_instructions() -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert("add", 0x00a282b3);
        map.insert("addi", 0x00500013);
        map.insert("auipc", 0x00123297);
        map.insert("slti", 0x0052a293);
        map.insert("sltiu", 0x0052b293);
        map.insert("xori", 0x0052c293);
        map.insert("ori", 0x2142e293);
        map.insert("andi", 0x0051fb93);
        map.insert("slli", 0x02d61393);
        map.insert("srli", 0x02d65393);
        map.insert("srai", 0x42d65393);
        map.insert("sub", 0x405302b3);
        map.insert("sll", 0x019192b3);
        map.insert("slt", 0x0191a2b3);
        map.insert("sltu", 0x0191b2b3);
        map.insert("xor", 0x001d4fb3);
        map.insert("srl", 0x0191d2b3);
        map.insert("sra", 0x4191d2b3);
        map.insert("or", 0x0191e2b3);
        map.insert("and", 0x0191f2b3);
        map.insert("ecall", 0x00000073);
        map.insert("ebreak", 0x00100073);
        map.insert("lb", 0x000e0283);
        map.insert("lh", 0x000e1283);
        map.insert("lw", 0x000e2283);
        map.insert("lbu", 0x000e4283);
        map.insert("lhu", 0x000e5283);
        map.insert("sb", 0x005e0023);
        map.insert("sh", 0x005e1023);
        map.insert("sw", 0x005e2023);
        map.insert("jal", 0x00400c6f);
        map.insert("jalr", 0x00428c67);
        map.insert("beq", 0x005c0263);
        map.insert("bne", 0x005c1263);
        map.insert("bge", 0x005c5263);
        map.insert("bgeu", 0x005c7263);
        map.insert("blt", 0x005c4263);
        map.insert("bltu", 0x005c6263);
        map.insert("ld", 0x00043283);
        map.insert("sd", 0x00543023);
        map.insert("addiw", 0x0007879b);
        map.insert("sraiw", 0x41f7d79b);
        map.insert("subw", 0x40f007bb);
        map.insert("addw", 0x00f707bb);

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

            if is_rv64i_sraiw_instruction(ins) && name != "sraiw"
                || (!is_rv64i_sraiw_instruction(ins) && name == "sraiw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an sraiw instruction!").into());
            }

            if is_rv64i_addw_instruction(ins) && name != "addw"
                || (!is_rv64i_addw_instruction(ins) && name == "addw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an subw instruction!").into());
            }

            if is_rv64i_subw_instruction(ins) && name != "subw"
                || (!is_rv64i_subw_instruction(ins) && name == "subw")
            {
                return Err(format!("{name}: {ins:#08x}, is not an subw instruction!").into());
            }
        }

        Ok(())
    }
}
