use std::collections::BTreeSet;

use riscvm_core::cpu::{RV64GCInstruction, RV64GCRegAbiName::Pc, RV64GC};
use riscvm_core::{sign_extend, sign_extend12};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReverseCompiledInstruction {
    pub address: u64,
    pub opcode: u32,
    pub len: u64,
    pub pseudo: String,
    pub assembly: String,
    pub target: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverseOutputStyle {
    Annotated,
    C,
}

pub fn reverse_compile_from(
    cpu: &RV64GC,
    mut address: u64,
    count: usize,
) -> Result<Vec<ReverseCompiledInstruction>, String> {
    let count = count.max(1);
    let mut instructions = Vec::with_capacity(count);

    for _ in 0..count {
        let instruction = reverse_compile_instruction(cpu, address)?;
        address = address.wrapping_add(instruction.len);
        instructions.push(instruction);
    }

    Ok(instructions)
}

pub fn reverse_compile_executable(
    cpu: &RV64GC,
    limit: usize,
) -> Result<(Vec<ReverseCompiledInstruction>, bool), String> {
    let limit = limit.max(1);
    let mut ranges = cpu.elf_executable_ranges();
    if ranges.is_empty() {
        ranges = cpu.ram.executable_ranges();
    }

    if ranges.is_empty() {
        return reverse_compile_from(cpu, cpu.registers[Pc], limit).map(|instructions| {
            let truncated = instructions.len() == limit;
            (instructions, truncated)
        });
    }

    let mut instructions = Vec::new();
    for (start, end) in ranges {
        let mut address = start;
        while address < end {
            if instructions.len() >= limit {
                return Ok((instructions, true));
            }

            let instruction = reverse_compile_instruction(cpu, address)?;
            address = address.wrapping_add(instruction.len);
            instructions.push(instruction);
        }
    }

    Ok((instructions, false))
}

pub fn format_reverse_compiled(
    instructions: &[ReverseCompiledInstruction],
    truncated: bool,
) -> String {
    format_reverse_compiled_with_style(instructions, truncated, ReverseOutputStyle::Annotated)
}

pub fn format_reverse_compiled_with_style(
    instructions: &[ReverseCompiledInstruction],
    truncated: bool,
    style: ReverseOutputStyle,
) -> String {
    let Some(first) = instructions.first() else {
        return "no instructions reverse compiled".to_string();
    };

    let instruction_addresses = instructions
        .iter()
        .map(|instruction| instruction.address)
        .collect::<BTreeSet<_>>();
    let labels = instructions
        .iter()
        .filter_map(|instruction| instruction.target)
        .filter(|target| instruction_addresses.contains(target))
        .collect::<BTreeSet<_>>();

    let mut lines = Vec::new();
    if style == ReverseOutputStyle::C {
        lines.extend(c_prelude());
    }
    lines.push(format!("void guest_code_{:016x}(void) {{", first.address));
    if style == ReverseOutputStyle::C {
        lines.extend(c_register_declarations());
    }
    let mut expected_address = None;

    for instruction in instructions {
        if expected_address != Some(instruction.address) {
            lines.push(format!("    // region 0x{:016x}", instruction.address));
        }
        if labels.contains(&instruction.address) {
            lines.push(format!("{}:", label_name(instruction.address)));
        }

        let opcode = if instruction.len == 2 {
            format!("0x{:04x}", instruction.opcode & 0xffff)
        } else {
            format!("0x{:08x}", instruction.opcode)
        };
        let pseudo = render_pseudo(instruction, &instruction_addresses, style);
        lines.push(format!(
            "    {:<56} // 0x{:016x}: {:>10} {}",
            pseudo, instruction.address, opcode, instruction.assembly
        ));
        expected_address = Some(instruction.address.wrapping_add(instruction.len));
    }

    lines.push("}".to_string());
    if truncated {
        lines.push(
            "// reverse compilation truncated; pass a larger count to include more code"
                .to_string(),
        );
    }

    lines.join("\n")
}

fn render_pseudo(
    instruction: &ReverseCompiledInstruction,
    instruction_addresses: &BTreeSet<u64>,
    style: ReverseOutputStyle,
) -> String {
    let Some(target) = instruction.target else {
        return instruction.pseudo.clone();
    };
    if style != ReverseOutputStyle::C || instruction_addresses.contains(&target) {
        return instruction.pseudo.clone();
    }

    let label = label_name(target);
    let external_target = format!("goto_address(UINT64_C(0x{target:016x}))");
    instruction
        .pseudo
        .replace(&format!("goto {label}"), &external_target)
}

fn reverse_compile_instruction(
    cpu: &RV64GC,
    address: u64,
) -> Result<ReverseCompiledInstruction, String> {
    let opcode = cpu
        .ram
        .read_word(address)
        .map_err(|error| error.to_string())?;
    let instruction = cpu.find_instruction(opcode);
    let len = instruction_len(opcode);
    Ok(ReverseCompiledInstruction {
        address,
        opcode,
        len,
        pseudo: pseudocode(address, len, &instruction),
        assembly: assembly_text(&instruction),
        target: static_target(address, &instruction),
    })
}

fn pseudocode(address: u64, len: u64, instruction: &RV64GCInstruction) -> String {
    use RV64GCInstruction::*;

    match *instruction {
        Add(rd, rs1, rs2) => assign(rd, format!("{} + {}", reg(rs1), reg(rs2))),
        Addi(rd, rs1, simm) => assign(rd, add_immediate_expr(rs1, simm)),
        Auipc(rd, simm) => assign(rd, add_address_expr(address, simm)),
        Lui(rd, simm) => assign(rd, format_signed_value(simm)),
        Slti(rd, rs1, simm) => assign(rd, format!("({} < {simm}) ? 1 : 0", reg(rs1))),
        Sltiu(rd, rs1, imm) => assign(
            rd,
            format!("({} < {}) ? 1 : 0", reg(rs1), sign_extend12(imm) as u64),
        ),
        Xori(rd, rs1, simm) => assign(rd, format!("{} ^ {simm}", reg(rs1))),
        Ori(rd, rs1, simm) => assign(rd, format!("{} | {simm}", reg(rs1))),
        Andi(rd, rs1, simm) => assign(rd, format!("{} & {simm}", reg(rs1))),
        Slli(rd, rs1, shamt) => assign(rd, format!("{} << {shamt}", reg(rs1))),
        Srli(rd, rs1, shamt) => assign(rd, format!("((uint64_t){} >> {shamt})", reg(rs1))),
        Srai(rd, rs1, shamt) => assign(rd, format!("((int64_t){} >> {shamt})", reg(rs1))),
        Sub(rd, rs1, rs2) => assign(rd, format!("{} - {}", reg(rs1), reg(rs2))),
        Sll(rd, rs1, rs2) => assign(rd, format!("{} << {}", reg(rs1), reg(rs2))),
        Slt(rd, rs1, rs2) => assign(
            rd,
            format!("((int64_t){} < (int64_t){}) ? 1 : 0", reg(rs1), reg(rs2)),
        ),
        Sltu(rd, rs1, rs2) => assign(
            rd,
            format!("((uint64_t){} < (uint64_t){}) ? 1 : 0", reg(rs1), reg(rs2)),
        ),
        Xor(rd, rs1, rs2) => assign(rd, format!("{} ^ {}", reg(rs1), reg(rs2))),
        Srl(rd, rs1, rs2) => assign(
            rd,
            format!("((uint64_t){} >> ({} & 63))", reg(rs1), reg(rs2)),
        ),
        Sra(rd, rs1, rs2) => assign(
            rd,
            format!("((int64_t){} >> ({} & 63))", reg(rs1), reg(rs2)),
        ),
        Or(rd, rs1, rs2) => assign(rd, format!("{} | {}", reg(rs1), reg(rs2))),
        And(rd, rs1, rs2) => assign(rd, format!("{} & {}", reg(rs1), reg(rs2))),

        Fence(_, _) => "fence();".to_string(),
        FenceI => "fence_i();".to_string(),
        Csrrw(rd, rs1, csr) => assign(rd, format!("csr_swap(0x{csr:03x}, {})", reg(rs1))),
        Csrrs(rd, rs1, csr) => assign(rd, format!("csr_set(0x{csr:03x}, {})", reg(rs1))),
        Csrrc(rd, rs1, csr) => assign(rd, format!("csr_clear(0x{csr:03x}, {})", reg(rs1))),
        Csrrwi(rd, imm, csr) => assign(rd, format!("csr_swap(0x{csr:03x}, {imm})")),
        Csrrsi(rd, imm, csr) => assign(rd, format!("csr_set(0x{csr:03x}, {imm})")),
        Csrrci(rd, imm, csr) => assign(rd, format!("csr_clear(0x{csr:03x}, {imm})")),
        Ecall => "syscall(a7);".to_string(),
        Ebreak | Cebreak => "debug_break();".to_string(),
        Uret => "return_from_trap(\"user\");".to_string(),
        Sret => "return_from_trap(\"supervisor\");".to_string(),
        Mret => "return_from_trap(\"machine\");".to_string(),
        Wfi => "wait_for_interrupt();".to_string(),
        SfenceVma(rs1, rs2, _) => format!("sfence_vma({}, {});", reg(rs1), reg(rs2)),

        Lb(rd, rs1, simm) => assign(rd, load_expr("i8", rs1, simm)),
        Lh(rd, rs1, simm) => assign(rd, load_expr("i16", rs1, simm)),
        Lw(rd, rs1, simm) => assign(rd, load_expr("i32", rs1, simm)),
        Lbu(rd, rs1, simm) => assign(rd, load_expr("u8", rs1, simm)),
        Lhu(rd, rs1, simm) => assign(rd, load_expr("u16", rs1, simm)),
        Lwu(rd, rs1, imm) => assign(rd, load_expr("u32", rs1, sign_extend12(imm))),
        Ld(rd, rs1, simm) => assign(rd, load_expr("u64", rs1, simm)),
        Sb(rs1, rs2, simm) => store_expr("u8", rs1, simm, reg(rs2)),
        Sh(rs1, rs2, simm) => store_expr("u16", rs1, simm, reg(rs2)),
        Sw(rs1, rs2, simm) => store_expr("u32", rs1, simm, reg(rs2)),
        Sd(rs1, rs2, simm) => store_expr("u64", rs1, simm, reg(rs2)),

        Jal(rd, simm) => jump_and_link(address, len, rd, address.wrapping_add_signed(simm)),
        Jalr(rd, rs1, simm) => jump_register(address, len, rd, rs1, simm),
        Beq(rs1, rs2, imm) => branch(address, imm, 13, format!("{} == {}", reg(rs1), reg(rs2))),
        Bne(rs1, rs2, imm) => branch(address, imm, 13, format!("{} != {}", reg(rs1), reg(rs2))),
        Blt(rs1, rs2, imm) => branch(
            address,
            imm,
            13,
            format!("(int64_t){} < (int64_t){}", reg(rs1), reg(rs2)),
        ),
        Bge(rs1, rs2, imm) => branch(
            address,
            imm,
            13,
            format!("(int64_t){} >= (int64_t){}", reg(rs1), reg(rs2)),
        ),
        Bltu(rs1, rs2, imm) => branch(
            address,
            imm,
            13,
            format!("(uint64_t){} < (uint64_t){}", reg(rs1), reg(rs2)),
        ),
        Bgeu(rs1, rs2, imm) => branch(
            address,
            imm,
            13,
            format!("(uint64_t){} >= (uint64_t){}", reg(rs1), reg(rs2)),
        ),

        Addiw(rd, rs1, imm) => assign(
            rd,
            format!("sext32({})", add_immediate_expr(rs1, sign_extend12(imm))),
        ),
        Slliw(rd, rs1, shamt) => assign(rd, format!("sext32({} << {shamt})", reg(rs1))),
        Srliw(rd, rs1, shamt) => assign(rd, format!("sext32((uint32_t){} >> {shamt})", reg(rs1))),
        Sraiw(rd, rs1, shamt) => assign(rd, format!("sext32((int32_t){} >> {shamt})", reg(rs1))),
        Addw(rd, rs1, rs2) => assign(rd, format!("sext32({} + {})", reg(rs1), reg(rs2))),
        Subw(rd, rs1, rs2) => assign(rd, format!("sext32({} - {})", reg(rs1), reg(rs2))),
        Sllw(rd, rs1, rs2) => assign(rd, format!("sext32({} << {})", reg(rs1), reg(rs2))),
        Srlw(rd, rs1, rs2) => assign(
            rd,
            format!("sext32((uint32_t){} >> ({} & 31))", reg(rs1), reg(rs2)),
        ),
        Sraw(rd, rs1, rs2) => assign(
            rd,
            format!("sext32((int32_t){} >> ({} & 31))", reg(rs1), reg(rs2)),
        ),
        Mul(rd, rs1, rs2) => assign(rd, format!("{} * {}", reg(rs1), reg(rs2))),
        Mulh(rd, rs1, rs2) => assign(rd, format!("mulh_s64({}, {})", reg(rs1), reg(rs2))),
        Mulhsu(rd, rs1, rs2) => assign(rd, format!("mulh_su64({}, {})", reg(rs1), reg(rs2))),
        Mulhu(rd, rs1, rs2) => assign(rd, format!("mulh_u64({}, {})", reg(rs1), reg(rs2))),
        Div(rd, rs1, rs2) => assign(rd, format!("div_s64({}, {})", reg(rs1), reg(rs2))),
        Divu(rd, rs1, rs2) => assign(rd, format!("div_u64({}, {})", reg(rs1), reg(rs2))),
        Rem(rd, rs1, rs2) => assign(rd, format!("rem_s64({}, {})", reg(rs1), reg(rs2))),
        Remu(rd, rs1, rs2) => assign(rd, format!("rem_u64({}, {})", reg(rs1), reg(rs2))),
        Mulw(rd, rs1, rs2) => assign(rd, format!("sext32({} * {})", reg(rs1), reg(rs2))),
        Divw(rd, rs1, rs2) => assign(rd, format!("sext32(div_s32({}, {}))", reg(rs1), reg(rs2))),
        Divuw(rd, rs1, rs2) => assign(rd, format!("sext32(div_u32({}, {}))", reg(rs1), reg(rs2))),
        Remw(rd, rs1, rs2) => assign(rd, format!("sext32(rem_s32({}, {}))", reg(rs1), reg(rs2))),
        Remuw(rd, rs1, rs2) => assign(rd, format!("sext32(rem_u32({}, {}))", reg(rs1), reg(rs2))),

        Lrw(rd, rs1) => assign(rd, format!("atomic_load_i32({})", address_expr(rs1, 0))),
        Lrd(rd, rs1) => assign(rd, format!("atomic_load_i64({})", address_expr(rs1, 0))),
        Scw(rd, rs1, rs2) => assign(
            rd,
            format!(
                "atomic_store_conditional_i32({}, {})",
                address_expr(rs1, 0),
                reg(rs2)
            ),
        ),
        Scd(rd, rs1, rs2) => assign(
            rd,
            format!(
                "atomic_store_conditional_i64({}, {})",
                address_expr(rs1, 0),
                reg(rs2)
            ),
        ),
        Amoswapw(rd, rs1, rs2) => amo("swap_i32", rd, rs1, rs2),
        Amoaddw(rd, rs1, rs2) => amo("add_i32", rd, rs1, rs2),
        Amoxorw(rd, rs1, rs2) => amo("xor_i32", rd, rs1, rs2),
        Amoandw(rd, rs1, rs2) => amo("and_i32", rd, rs1, rs2),
        Amoorw(rd, rs1, rs2) => amo("or_i32", rd, rs1, rs2),
        Amominw(rd, rs1, rs2) => amo("min_i32", rd, rs1, rs2),
        Amomaxw(rd, rs1, rs2) => amo("max_i32", rd, rs1, rs2),
        Amominuw(rd, rs1, rs2) => amo("min_u32", rd, rs1, rs2),
        Amomaxuw(rd, rs1, rs2) => amo("max_u32", rd, rs1, rs2),
        Amoswapd(rd, rs1, rs2) => amo("swap_i64", rd, rs1, rs2),
        Amoaddd(rd, rs1, rs2) => amo("add_i64", rd, rs1, rs2),
        Amoxord(rd, rs1, rs2) => amo("xor_i64", rd, rs1, rs2),
        Amoandd(rd, rs1, rs2) => amo("and_i64", rd, rs1, rs2),
        Amoord(rd, rs1, rs2) => amo("or_i64", rd, rs1, rs2),
        Amomind(rd, rs1, rs2) => amo("min_i64", rd, rs1, rs2),
        Amomaxd(rd, rs1, rs2) => amo("max_i64", rd, rs1, rs2),
        Amominud(rd, rs1, rs2) => amo("min_u64", rd, rs1, rs2),
        Amomaxud(rd, rs1, rs2) => amo("max_u64", rd, rs1, rs2),

        Flw(rd, rs1, imm) => fassign(rd, load_expr("f32", rs1, sign_extend12(imm))),
        Fld(rd, rs1, imm) => fassign(rd, load_expr("f64", rs1, sign_extend12(imm))),
        Fsw(rs1, rs2, imm) => store_expr("f32", rs1, sign_extend12(imm), freg(rs2)),
        Fsd(rs1, rs2, simm) => store_expr("f64", rs1, simm, freg(rs2)),
        Fadds(rd, rs1, rs2, _) | Faddd(rd, rs1, rs2, _) => {
            fassign(rd, format!("{} + {}", freg(rs1), freg(rs2)))
        }
        Fsubs(rd, rs1, rs2, _) | Fsubd(rd, rs1, rs2, _) => {
            fassign(rd, format!("{} - {}", freg(rs1), freg(rs2)))
        }
        Fmuls(rd, rs1, rs2, _) | Fmuld(rd, rs1, rs2, _) => {
            fassign(rd, format!("{} * {}", freg(rs1), freg(rs2)))
        }
        Fdivs(rd, rs1, rs2, _) | Fdivd(rd, rs1, rs2, _) => {
            fassign(rd, format!("{} / {}", freg(rs1), freg(rs2)))
        }
        Fsqrts(rd, rs1, _) | Fsqrtd(rd, rs1, _) => fassign(rd, format!("sqrt({})", freg(rs1))),
        Feqs(rd, rs1, rs2) | Feqd(rd, rs1, rs2) => {
            assign(rd, format!("({} == {}) ? 1 : 0", freg(rs1), freg(rs2)))
        }
        Flts(rd, rs1, rs2) | Fltd(rd, rs1, rs2) => {
            assign(rd, format!("({} < {}) ? 1 : 0", freg(rs1), freg(rs2)))
        }
        Fles(rd, rs1, rs2) | Fled(rd, rs1, rs2) => {
            assign(rd, format!("({} <= {}) ? 1 : 0", freg(rs1), freg(rs2)))
        }
        Fmvxw(rd, rs1) | Fmvxd(rd, rs1) => assign(rd, format!("bits({})", freg(rs1))),
        Fmvwx(rd, rs1) => fassign(rd, format!("bits({})", reg(rs1))),

        Cnop => "/* nop */".to_string(),
        Caddi(rd, simm) => assign(rd, add_immediate_expr(rd, simm)),
        Caddiw(rd, simm) => assign(rd, format!("sext32({})", add_immediate_expr(rd, simm))),
        Caddi16sp(simm) => assign(2, add_immediate_expr(2, simm)),
        Caddi4spn(rd, imm) => assign(rd, add_immediate_expr(2, imm as i64)),
        Cli(rd, imm) => assign(rd, format!("{}", sign_extend(imm as u64, 6))),
        Clui(rd, imm) => assign(rd, format_signed_value(sign_extend(imm as u64, 18))),
        Cmv(rd, rs2) => assign(rd, reg(rs2)),
        Cadd(rd, rs2) => assign(rd, format!("{} + {}", reg(rd), reg(rs2))),
        Cjr(rs1) => {
            if rs1 == 1 {
                "return;".to_string()
            } else {
                format!("goto_indirect({});", reg(rs1))
            }
        }
        Cjalr(rs1) => format!(
            "ra = 0x{:016x}; goto_indirect({});",
            address + len,
            reg(rs1)
        ),
        Cbeqz(rs1, imm) => branch(address, imm, 9, format!("{} == 0", reg(rs1))),
        Cbnez(rs1, imm) => branch(address, imm, 9, format!("{} != 0", reg(rs1))),
        Cj(imm) => format!(
            "goto {};",
            label_name(address.wrapping_add_signed(sign_extend12(imm)))
        ),
        Cslli(rd, imm) => assign(rd, format!("{} << {imm}", reg(rd))),
        Csrli(rd, imm) => assign(rd, format!("((uint64_t){} >> {imm})", reg(rd))),
        Csrai(rd, imm) => assign(rd, format!("((int64_t){} >> {imm})", reg(rd))),
        Candi(rd, simm) => assign(rd, format!("{} & {simm}", reg(rd))),
        Csub(rd, rs2) => assign(rd, format!("{} - {}", reg(rd), reg(rs2))),
        Cxor(rd, rs2) => assign(rd, format!("{} ^ {}", reg(rd), reg(rs2))),
        Cor(rd, rs2) => assign(rd, format!("{} | {}", reg(rd), reg(rs2))),
        Cand(rd, rs2) => assign(rd, format!("{} & {}", reg(rd), reg(rs2))),
        Csubw(rd, rs2) => assign(rd, format!("sext32({} - {})", reg(rd), reg(rs2))),
        Caddw(rd, rs2) => assign(rd, format!("sext32({} + {})", reg(rd), reg(rs2))),
        Clw(rd, rs1, imm) => assign(rd, load_expr("i32", rs1, imm as i64)),
        Cld(rd, rs1, imm) => assign(rd, load_expr("u64", rs1, imm as i64)),
        Cfld(rd, rs1, imm) => fassign(rd, load_expr("f64", rs1, imm as i64)),
        Csw(rs1, rs2, imm) => store_expr("u32", rs1, imm as i64, reg(rs2)),
        Csd(rs1, rs2, imm) => store_expr("u64", rs1, imm as i64, reg(rs2)),
        Cfsw(rs1, rs2, imm) => store_expr("f32", rs1, imm as i64, freg(rs2)),
        Cfsd(rs1, rs2, imm) => store_expr("f64", rs1, imm as i64, freg(rs2)),
        Clwsp(rd, imm) => assign(rd, load_expr("i32", 2, imm as i64)),
        Cldsp(rd, imm) => assign(rd, load_expr("u64", 2, imm as i64)),
        Cflwsp(rd, imm) => fassign(rd, load_expr("f32", 2, imm as i64)),
        Cfldsp(rd, imm) => fassign(rd, load_expr("f64", 2, imm as i64)),
        Cswsp(rs2, imm) => store_expr("u32", 2, imm as i64, reg(rs2)),
        Csdsp(rs2, imm) => store_expr("u64", 2, imm as i64, reg(rs2)),
        Cfsdsp(rs2, imm) => store_expr("f64", 2, imm as i64, freg(rs2)),

        IllegalInstruction(raw) => format!("trap_illegal(0x{raw:08x});"),
        _ => format!(
            "execute_unmodeled({});",
            c_string_literal(&format!("{instruction:?}"))
        ),
    }
}

fn static_target(address: u64, instruction: &RV64GCInstruction) -> Option<u64> {
    use RV64GCInstruction::*;

    match *instruction {
        Jal(_, simm) => Some(address.wrapping_add_signed(simm)),
        Beq(_, _, imm)
        | Bne(_, _, imm)
        | Blt(_, _, imm)
        | Bge(_, _, imm)
        | Bltu(_, _, imm)
        | Bgeu(_, _, imm) => Some(relative_target(address, imm, 13)),
        Cbeqz(_, imm) | Cbnez(_, imm) => Some(relative_target(address, imm, 9)),
        Cj(imm) => Some(address.wrapping_add_signed(sign_extend12(imm))),
        _ => None,
    }
}

fn assembly_text(instruction: &RV64GCInstruction) -> String {
    instruction.to_string()
}

fn instruction_len(opcode: u32) -> u64 {
    if opcode & 0b11 == 0b11 {
        4
    } else {
        2
    }
}

fn relative_target(address: u64, encoded_offset: u32, bits: u8) -> u64 {
    address.wrapping_add_signed(sign_extend(u64::from(encoded_offset), bits))
}

fn branch(address: u64, encoded_offset: u32, bits: u8, condition: String) -> String {
    let target = relative_target(address, encoded_offset, bits);
    format!("if ({condition}) goto {};", label_name(target))
}

fn jump_and_link(address: u64, len: u64, rd: u8, target: u64) -> String {
    let target = label_name(target);
    let return_address = address.wrapping_add(len);

    match rd {
        0 => format!("goto {target};"),
        1 => format!("ra = 0x{return_address:016x}; goto {target};"),
        _ => format!("{} = 0x{return_address:016x}; goto {target};", reg(rd)),
    }
}

fn jump_register(address: u64, len: u64, rd: u8, rs1: u8, simm: i64) -> String {
    if rd == 0 && rs1 == 1 && simm == 0 {
        return "return;".to_string();
    }

    let target = format!("({} & ~UINT64_C(1))", add_immediate_expr(rs1, simm));
    if rd == 0 {
        format!("goto_indirect({target});")
    } else {
        format!(
            "{} = 0x{:016x}; goto_indirect({target});",
            reg(rd),
            address + len
        )
    }
}

fn assign(rd: u8, expr: impl Into<String>) -> String {
    let expr = expr.into();
    if rd == 0 {
        format!("discard({expr});")
    } else {
        format!("{} = {expr};", reg(rd))
    }
}

fn fassign(rd: u8, expr: impl Into<String>) -> String {
    format!("{} = {};", freg(rd), expr.into())
}

fn amo(operation: &str, rd: u8, rs1: u8, rs2: u8) -> String {
    assign(
        rd,
        format!("{operation}({}, {})", address_expr(rs1, 0), reg(rs2)),
    )
}

fn load_expr(width: &str, rs1: u8, simm: i64) -> String {
    format!("{}({})", memory_accessor(width), address_expr(rs1, simm))
}

fn store_expr(width: &str, rs1: u8, simm: i64, value: String) -> String {
    format!(
        "{}({}) = {value};",
        memory_accessor(width),
        address_expr(rs1, simm)
    )
}

fn memory_accessor(width: &str) -> String {
    format!("MEM_{}", width.to_ascii_uppercase())
}

fn address_expr(rs1: u8, simm: i64) -> String {
    add_expr(reg(rs1), simm)
}

fn add_immediate_expr(rs1: u8, simm: i64) -> String {
    if rs1 == 0 {
        return format_signed_value(simm);
    }
    add_expr(reg(rs1), simm)
}

fn add_address_expr(address: u64, simm: i64) -> String {
    add_expr(format!("0x{address:016x}"), simm)
}

fn add_expr(base: String, simm: i64) -> String {
    match simm.cmp(&0) {
        std::cmp::Ordering::Equal => base,
        std::cmp::Ordering::Greater => format!("{base} + {simm}"),
        std::cmp::Ordering::Less => format!("{base} - {}", simm.wrapping_neg()),
    }
}

fn format_signed_value(value: i64) -> String {
    if value < 0 {
        format!("{value}")
    } else {
        format!("0x{value:x}")
    }
}

fn c_string_literal(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\x{:02x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn c_prelude() -> Vec<String> {
    vec![
        "#include <stdint.h>".to_string(),
        "#include <math.h>".to_string(),
        "".to_string(),
        "typedef uint8_t u8;".to_string(),
        "typedef int8_t i8;".to_string(),
        "typedef uint16_t u16;".to_string(),
        "typedef int16_t i16;".to_string(),
        "typedef uint32_t u32;".to_string(),
        "typedef int32_t i32;".to_string(),
        "typedef uint64_t u64;".to_string(),
        "typedef int64_t i64;".to_string(),
        "typedef float f32;".to_string(),
        "typedef double f64;".to_string(),
        "".to_string(),
        "extern uint8_t guest_memory[];".to_string(),
        "#define MEM_U8(addr)  (*(uint8_t *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_I8(addr)  (*(int8_t *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_U16(addr) (*(uint16_t *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_I16(addr) (*(int16_t *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_U32(addr) (*(uint32_t *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_I32(addr) (*(int32_t *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_U64(addr) (*(uint64_t *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_F32(addr) (*(float *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "#define MEM_F64(addr) (*(double *)&guest_memory[(uint64_t)(addr)])".to_string(),
        "".to_string(),
        "extern void fence(void);".to_string(),
        "extern void fence_i(void);".to_string(),
        "extern uint64_t csr_swap(uint16_t csr, uint64_t value);".to_string(),
        "extern uint64_t csr_set(uint16_t csr, uint64_t value);".to_string(),
        "extern uint64_t csr_clear(uint16_t csr, uint64_t value);".to_string(),
        "extern void syscall(uint64_t number);".to_string(),
        "extern void debug_break(void);".to_string(),
        "extern void return_from_trap(const char *mode);".to_string(),
        "extern void wait_for_interrupt(void);".to_string(),
        "extern void sfence_vma(uint64_t address, uint64_t asid);".to_string(),
        "extern void goto_address(uint64_t target);".to_string(),
        "extern void goto_indirect(uint64_t target);".to_string(),
        "extern void execute_unmodeled(const char *instruction);".to_string(),
        "extern void trap_illegal(uint32_t instruction);".to_string(),
        "extern uint64_t sext32(uint64_t value);".to_string(),
        "extern uint64_t bits(double value);".to_string(),
        "extern double bits_to_f64(uint64_t value);".to_string(),
        "extern uint64_t mulh_s64(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint64_t mulh_su64(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint64_t mulh_u64(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint64_t div_s64(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint64_t div_u64(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint64_t rem_s64(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint64_t rem_u64(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint32_t div_s32(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint32_t div_u32(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint32_t rem_s32(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint32_t rem_u32(uint64_t lhs, uint64_t rhs);".to_string(),
        "extern uint64_t atomic_load_i32(uint64_t address);".to_string(),
        "extern uint64_t atomic_load_i64(uint64_t address);".to_string(),
        "extern uint64_t atomic_store_conditional_i32(uint64_t address, uint64_t value);"
            .to_string(),
        "extern uint64_t atomic_store_conditional_i64(uint64_t address, uint64_t value);"
            .to_string(),
        "extern uint64_t swap_i32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t add_i32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t xor_i32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t and_i32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t or_i32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t min_i32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t max_i32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t min_u32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t max_u32(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t swap_i64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t add_i64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t xor_i64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t and_i64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t or_i64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t min_i64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t max_i64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t min_u64(uint64_t address, uint64_t value);".to_string(),
        "extern uint64_t max_u64(uint64_t address, uint64_t value);".to_string(),
        "static inline void discard(uint64_t value) { (void)value; }".to_string(),
        "".to_string(),
    ]
}

fn c_register_declarations() -> Vec<String> {
    vec![
        "    uint64_t zero = 0, ra = 0, sp = 0, gp = 0, tp = 0;".to_string(),
        "    uint64_t t0 = 0, t1 = 0, t2 = 0, fp = 0, s1 = 0;".to_string(),
        "    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;"
            .to_string(),
        "    uint64_t s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0, s9 = 0, s10 = 0, s11 = 0;"
            .to_string(),
        "    uint64_t t3 = 0, t4 = 0, t5 = 0, t6 = 0;".to_string(),
        "    double f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0, f7 = 0;".to_string(),
        "    double f8 = 0, f9 = 0, f10 = 0, f11 = 0, f12 = 0, f13 = 0, f14 = 0, f15 = 0;"
            .to_string(),
        "    double f16 = 0, f17 = 0, f18 = 0, f19 = 0, f20 = 0, f21 = 0, f22 = 0, f23 = 0;"
            .to_string(),
        "    double f24 = 0, f25 = 0, f26 = 0, f27 = 0, f28 = 0, f29 = 0, f30 = 0, f31 = 0;"
            .to_string(),
        "    (void)zero; (void)ra; (void)sp; (void)gp; (void)tp; (void)fp;".to_string(),
        "".to_string(),
    ]
}

fn label_name(address: u64) -> String {
    format!("L_{address:016x}")
}

fn freg(register: u8) -> String {
    format!("f{register}")
}

fn reg(register: u8) -> String {
    match register {
        0 => "zero",
        1 => "ra",
        2 => "sp",
        3 => "gp",
        4 => "tp",
        5 => "t0",
        6 => "t1",
        7 => "t2",
        8 => "fp",
        9 => "s1",
        10 => "a0",
        11 => "a1",
        12 => "a2",
        13 => "a3",
        14 => "a4",
        15 => "a5",
        16 => "a6",
        17 => "a7",
        18 => "s2",
        19 => "s3",
        20 => "s4",
        21 => "s5",
        22 => "s6",
        23 => "s7",
        24 => "s8",
        25 => "s9",
        26 => "s10",
        27 => "s11",
        28 => "t3",
        29 => "t4",
        30 => "t5",
        31 => "t6",
        _ => return format!("x{register}"),
    }
    .to_string()
}
