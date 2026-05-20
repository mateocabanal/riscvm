use bit::BitIndex;
use goblin::elf::{reloc, sym, Elf};
use rand::RngCore;
use tracing::span;
use tracing::trace;
use tracing::Level;
use tracing::{debug, info};

use crate::fcsr::classify_f32;
use crate::fcsr::classify_f64;
use crate::fcsr::round_f32;
use crate::fcsr::round_f64;
use crate::fcsr::RoundingMode;
use crate::fcsr::FCSR;
use crate::filesystem::{FileSystemError, GuestFileSystem};
use crate::ram::{align_up, MemoryRegion, Ram, PAGE_SIZE};
use crate::sign_extend;
use crate::sign_extend12;
use crate::syscalls::*;
use crate::tracer::{ExecutionEngine, ExecutionTracer, TraceOptions};
use std::collections::BTreeSet;
use std::fmt::Display;
use std::fs;
use std::ops::{Index, IndexMut};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::cpu::RV64GCRegAbiName::*;

type Reg = u8;
type Imm = u32;
type Simm = i64;
type Csr = u16;

const DYNAMIC_LINKER_BASE: u64 = 0x1000_0000_0000;
const RISCV_HWCAP_IMAFDC: usize = (1usize << ('I' as u8 - b'A'))
    | (1usize << ('M' as u8 - b'A'))
    | (1usize << ('A' as u8 - b'A'))
    | (1usize << ('F' as u8 - b'A'))
    | (1usize << ('D' as u8 - b'A'))
    | (1usize << ('C' as u8 - b'A'));
const SYNTHETIC_THREAD_DEFER_TICKS: u32 = 8;

const CSR_FFLAGS: Csr = 0x001;
const CSR_FRM: Csr = 0x002;
const CSR_FCSR: Csr = 0x003;

fn nan_box_f32(bits: u32) -> u64 {
    0xffff_ffff_0000_0000 | u64::from(bits)
}

const DECODE_CACHE_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy)]
struct DecodedInstruction {
    pc: u64,
    code_version: u64,
    opcode: u32,
    instruction: RV64GCInstruction,
    len: u64,
}

fn dump_ops_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("DUMP_OPS").is_ok_and(|value| value == "1"))
}

fn instruction_len(opcode: u32) -> u64 {
    if opcode & 0b11 == 0b11 {
        4
    } else {
        2
    }
}

fn decode_cache_index(pc: u64) -> usize {
    ((pc >> 1) as usize) & (DECODE_CACHE_SIZE - 1)
}

#[inline]
fn decode_instruction(current_ins: u32) -> RV64GCInstruction {
    if current_ins & 0b11 == 0b11 {
        decode_standard_instruction(current_ins)
    } else {
        decode_compressed_instruction(current_ins as u16)
    }
}

#[inline]
fn decode_standard_instruction(current_ins: u32) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    let rd = current_ins.bit_range(7..12) as Reg;
    let funct3 = current_ins.bit_range(12..15);
    let rs1 = current_ins.bit_range(15..20) as Reg;
    let rs2 = current_ins.bit_range(20..25) as Reg;
    let rs3 = current_ins.bit_range(27..32) as Reg;
    let funct7 = current_ins.bit_range(25..32);
    let imm = current_ins.bit_range(20..32) as Imm;
    let rm = funct3 as Reg;

    match current_ins & 0x7f {
        0x03 => match funct3 {
            0 => Lb(rd, rs1, sign_extend12(imm)),
            1 => Lh(rd, rs1, sign_extend12(imm)),
            2 => Lw(rd, rs1, sign_extend12(imm)),
            3 => Ld(rd, rs1, sign_extend12(imm)),
            4 => Lbu(rd, rs1, sign_extend12(imm)),
            5 => Lhu(rd, rs1, sign_extend12(imm)),
            6 => Lwu(rd, rs1, imm),
            _ => IllegalInstruction(current_ins),
        },
        0x07 => match funct3 {
            2 => {
                trace!("flw: {current_ins:08x}");
                Flw(rd, rs1, imm)
            }
            3 => Fld(rd, rs1, imm),
            _ => IllegalInstruction(current_ins),
        },
        0x0f => match current_ins {
            0x0000_100f => FenceI,
            i if i & 0x000f_ffff == 0x0000_000f => {
                Fence(i.bit_range(20..24) as u8, i.bit_range(24..28) as u8)
            }
            _ => IllegalInstruction(current_ins),
        },
        0x13 => match funct3 {
            0 => Addi(rd, rs1, sign_extend12(imm)),
            1 if current_ins >> 27 == 0 => Slli(rd, rs1, current_ins.bit_range(20..26)),
            2 => Slti(rd, rs1, sign_extend12(imm)),
            3 => Sltiu(rd, rs1, imm),
            4 => Xori(rd, rs1, sign_extend12(imm)),
            5 if current_ins >> 27 == 0 => Srli(rd, rs1, current_ins.bit_range(20..26)),
            5 if current_ins >> 26 == 0x10 => Srai(rd, rs1, current_ins.bit_range(20..26)),
            6 => Ori(rd, rs1, sign_extend12(imm)),
            7 => Andi(rd, rs1, sign_extend12(imm)),
            _ => IllegalInstruction(current_ins),
        },
        0x17 => {
            let ov_imm = current_ins.bit_range(12..32) << 12;
            Auipc(rd, sign_extend(ov_imm.into(), 32))
        }
        0x1b => match funct3 {
            0 => Addiw(rd, rs1, imm),
            1 if funct7 == 0 => Slliw(rd, rs1, current_ins.bit_range(20..26)),
            5 => match funct7 {
                0x00 => Srliw(rd, rs1, current_ins.bit_range(20..26)),
                0x20 => Sraiw(rd, rs1, current_ins.bit_range(20..25)),
                _ => IllegalInstruction(current_ins),
            },
            _ => IllegalInstruction(current_ins),
        },
        0x23 => {
            let offset = store_offset(current_ins);
            match funct3 {
                0 => Sb(rs1, rs2, sign_extend12(offset)),
                1 => Sh(rs1, rs2, sign_extend12(offset)),
                2 => Sw(rs1, rs2, sign_extend12(offset)),
                3 => Sd(rs1, rs2, sign_extend12(offset)),
                _ => IllegalInstruction(current_ins),
            }
        }
        0x27 => {
            let offset = store_offset(current_ins);
            match funct3 {
                2 => {
                    trace!("fsw: {current_ins:08x}");
                    trace!("imm: {}", sign_extend12(offset));
                    Fsw(rs1, rs2, offset)
                }
                3 => Fsd(rs1, rs2, sign_extend12(offset)),
                _ => IllegalInstruction(current_ins),
            }
        }
        0x2f => decode_atomic_instruction(current_ins, rd, rs1, rs2, funct3),
        0x33 => decode_op_instruction(current_ins, rd, rs1, rs2, funct3, funct7),
        0x37 => {
            let ov_imm = current_ins.bit_range(12..32) << 12;
            Lui(rd, sign_extend(ov_imm.into(), 32))
        }
        0x3b => decode_op32_instruction(current_ins, rd, rs1, rs2, funct3, funct7),
        0x43 => match current_ins.bit_range(25..27) {
            0 => Fmadds(rd, rm, rs1, rs2, rs3),
            1 => Fmaddd(rd, rm, rs1, rs2, rs3),
            _ => IllegalInstruction(current_ins),
        },
        0x47 => match current_ins.bit_range(25..27) {
            0 => Fmsubs(rd, rm, rs1, rs2, rs3),
            1 => Fmsubd(rd, rm, rs1, rs2, rs3),
            _ => IllegalInstruction(current_ins),
        },
        0x4b => match current_ins.bit_range(25..27) {
            0 => Fnmsubs(rd, rm, rs1, rs2, rs3),
            1 => Fnmsubd(rd, rm, rs1, rs2, rs3),
            _ => IllegalInstruction(current_ins),
        },
        0x4f => match current_ins.bit_range(25..27) {
            0 => Fnmadds(rd, rm, rs1, rs2, rs3),
            1 => Fnmaddd(rd, rm, rs1, rs2, rs3),
            _ => IllegalInstruction(current_ins),
        },
        0x53 => decode_float_op_instruction(current_ins, rd, rm, rs1, rs2, funct3, funct7),
        0x63 => {
            let offset = branch_offset(current_ins);
            match funct3 {
                0 => Beq(rs1, rs2, offset),
                1 => Bne(rs1, rs2, offset),
                4 => Blt(rs1, rs2, offset),
                5 => Bge(rs1, rs2, offset),
                6 => Bltu(rs1, rs2, offset),
                7 => Bgeu(rs1, rs2, offset),
                _ => IllegalInstruction(current_ins),
            }
        }
        0x67 => Jalr(rd, rs1, sign_extend12(imm)),
        0x6f => {
            let offset = (current_ins.bit(31) as u32) << 20
                | current_ins.bit_range(12..20) << 12
                | (current_ins.bit(20) as u32) << 11
                | current_ins.bit_range(21..31) << 1;
            let s_offset = sign_extend(offset.into(), 21);
            trace!("offset: {offset:#020b}");
            Jal(rd, s_offset)
        }
        0x73 => decode_system_instruction(current_ins, rd, rs1, imm, funct3),
        _ => IllegalInstruction(current_ins),
    }
}

#[inline]
fn decode_op_instruction(
    current_ins: u32,
    rd: Reg,
    rs1: Reg,
    rs2: Reg,
    funct3: u32,
    funct7: u32,
) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    match (funct7, funct3) {
        (0x00, 0) => Add(rd, rs1, rs2),
        (0x00, 1) => Sll(rd, rs1, rs2),
        (0x00, 2) => Slt(rd, rs1, rs2),
        (0x00, 3) => Sltu(rd, rs1, rs2),
        (0x00, 4) => Xor(rd, rs1, rs2),
        (0x00, 5) => Srl(rd, rs1, rs2),
        (0x00, 6) => Or(rd, rs1, rs2),
        (0x00, 7) => And(rd, rs1, rs2),
        (0x20, 0) => Sub(rd, rs1, rs2),
        (0x20, 5) => Sra(rd, rs1, rs2),
        (0x01, 0) => Mul(rd, rs1, rs2),
        (0x01, 1) => Mulh(rd, rs1, rs2),
        (0x01, 2) => Mulhsu(rd, rs1, rs2),
        (0x01, 3) => Mulhu(rd, rs1, rs2),
        (0x01, 4) => Div(rd, rs1, rs2),
        (0x01, 5) => Divu(rd, rs1, rs2),
        (0x01, 6) => Rem(rd, rs1, rs2),
        (0x01, 7) => Remu(rd, rs1, rs2),
        _ => IllegalInstruction(current_ins),
    }
}

#[inline]
fn decode_op32_instruction(
    current_ins: u32,
    rd: Reg,
    rs1: Reg,
    rs2: Reg,
    funct3: u32,
    funct7: u32,
) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    match (funct7, funct3) {
        (0x00, 0) => Addw(rd, rs1, rs2),
        (0x00, 1) => Sllw(rd, rs1, rs2),
        (0x00, 5) => Srlw(rd, rs1, rs2),
        (0x20, 0) => Subw(rd, rs1, rs2),
        (0x20, 5) => Sraw(rd, rs1, rs2),
        (0x01, 0) => Mulw(rd, rs1, rs2),
        (0x01, 4) => Divw(rd, rs1, rs2),
        (0x01, 5) => Divuw(rd, rs1, rs2),
        (0x01, 6) => Remw(rd, rs1, rs2),
        (0x01, 7) => Remuw(rd, rs1, rs2),
        _ => IllegalInstruction(current_ins),
    }
}

#[inline]
fn decode_atomic_instruction(
    current_ins: u32,
    rd: Reg,
    rs1: Reg,
    rs2: Reg,
    funct3: u32,
) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    let funct5 = current_ins.bit_range(27..32);
    if funct5 == 0b00010 && rs2 != 0 {
        return IllegalInstruction(current_ins);
    }

    match (funct3, funct5) {
        (2, 0b00010) => Lrw(rd, rs1),
        (2, 0b00011) => Scw(rd, rs1, rs2),
        (2, 0b00001) => Amoswapw(rd, rs1, rs2),
        (2, 0b00000) => Amoaddw(rd, rs1, rs2),
        (2, 0b00100) => Amoxorw(rd, rs1, rs2),
        (2, 0b01100) => Amoandw(rd, rs1, rs2),
        (2, 0b01000) => Amoorw(rd, rs1, rs2),
        (2, 0b10000) => Amominw(rd, rs1, rs2),
        (2, 0b10100) => Amomaxw(rd, rs1, rs2),
        (2, 0b11000) => Amominuw(rd, rs1, rs2),
        (2, 0b11100) => Amomaxuw(rd, rs1, rs2),
        (3, 0b00010) => Lrd(rd, rs1),
        (3, 0b00011) => Scd(rd, rs1, rs2),
        (3, 0b00001) => Amoswapd(rd, rs1, rs2),
        (3, 0b00000) => Amoaddd(rd, rs1, rs2),
        (3, 0b00100) => Amoxord(rd, rs1, rs2),
        (3, 0b01100) => Amoandd(rd, rs1, rs2),
        (3, 0b01000) => Amoord(rd, rs1, rs2),
        (3, 0b10000) => Amomind(rd, rs1, rs2),
        (3, 0b10100) => Amomaxd(rd, rs1, rs2),
        (3, 0b11000) => Amominud(rd, rs1, rs2),
        (3, 0b11100) => Amomaxud(rd, rs1, rs2),
        _ => IllegalInstruction(current_ins),
    }
}

#[inline]
fn decode_float_op_instruction(
    current_ins: u32,
    rd: Reg,
    rm: Reg,
    rs1: Reg,
    rs2: Reg,
    funct3: u32,
    funct7: u32,
) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    match funct7 {
        0x00 => Fadds(rd, rm, rs1, rs2),
        0x01 => Faddd(rd, rm, rs1, rs2),
        0x04 => Fsubs(rd, rm, rs1, rs2),
        0x05 => Fsubd(rd, rm, rs1, rs2),
        0x08 => Fmuls(rd, rm, rs1, rs2),
        0x09 => Fmuld(rd, rm, rs1, rs2),
        0x0c => Fdivs(rd, rm, rs1, rs2),
        0x0d => Fdivd(rd, rm, rs1, rs2),
        0x10 => match funct3 {
            0 => Fsgnjs(rd, rs1, rs2),
            1 => Fsgnjns(rd, rs1, rs2),
            2 => Fsgnjxs(rd, rs1, rs2),
            _ => IllegalInstruction(current_ins),
        },
        0x11 => match funct3 {
            0 => Fsgnjd(rd, rs1, rs2),
            1 => Fsgnjnd(rd, rs1, rs2),
            2 => Fsgnjxd(rd, rs1, rs2),
            _ => IllegalInstruction(current_ins),
        },
        0x14 => match funct3 {
            0 => Fmins(rd, rs1, rs2),
            1 => Fmaxs(rd, rs1, rs2),
            _ => IllegalInstruction(current_ins),
        },
        0x15 => match funct3 {
            0 => Fmind(rd, rs1, rs2),
            1 => Fmaxd(rd, rs1, rs2),
            _ => IllegalInstruction(current_ins),
        },
        0x20 => match rs2 {
            1 => Fcvtsd(rd, rm, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x21 => match rs2 {
            0 => Fcvtds(rd, rm, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x2c => Fsqrts(rd, rm, rs1),
        0x2d => Fsqrtd(rd, rm, rs1),
        0x50 => match funct3 {
            0 => Fles(rd, rs1, rs2),
            1 => Flts(rd, rs1, rs2),
            2 => Feqs(rd, rs1, rs2),
            _ => IllegalInstruction(current_ins),
        },
        0x51 => match funct3 {
            0 => Fled(rd, rs1, rs2),
            1 => Fltd(rd, rs1, rs2),
            2 => Feqd(rd, rs1, rs2),
            _ => IllegalInstruction(current_ins),
        },
        0x60 => match rs2 {
            0 => Fcvtws(rd, rm, rs1),
            1 => Fcvtwus(rd, rm, rs1),
            2 => Fcvtls(rd, rm, rs1),
            3 => Fcvtlus(rd, rm, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x61 => match rs2 {
            0 => Fcvtwd(rd, rm, rs1),
            1 => Fcvtwud(rd, rm, rs1),
            2 => Fcvtld(rd, rm, rs1),
            3 => Fcvtlud(rd, rm, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x68 => match rs2 {
            0 => Fcvtsw(rd, rm, rs1),
            1 => Fcvtswu(rd, rm, rs1),
            2 => Fcvtsl(rd, rm, rs1),
            3 => Fcvtslu(rd, rm, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x69 => match rs2 {
            0 => Fcvtdw(rd, rm, rs1),
            1 => Fcvtdwu(rd, rm, rs1),
            2 => Fcvtdl(rd, rm, rs1),
            3 => Fcvtdlu(rd, rm, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x70 => match (funct3, rs2) {
            (0, 0) => Fmvxw(rd, rs1),
            (1, 0) => Fclasss(rd, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x71 => match (funct3, rs2) {
            (0, 0) => Fmvxd(rd, rs1),
            (1, 0) => Fclassd(rd, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x78 => match (funct3, rs2) {
            (0, 0) => Fmvwx(rd, rs1),
            _ => IllegalInstruction(current_ins),
        },
        0x79 => match (funct3, rs2) {
            (0, 0) => Fmvdx(rd, rs1),
            _ => IllegalInstruction(current_ins),
        },
        _ => IllegalInstruction(current_ins),
    }
}

#[inline]
fn decode_system_instruction(
    current_ins: u32,
    rd: Reg,
    rs1: Reg,
    imm: Imm,
    funct3: u32,
) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    match current_ins {
        0x0000_0073 => Ecall,
        0x0010_0073 => Ebreak,
        _ => match funct3 {
            1 => Csrrw(rd, rs1, imm as Csr),
            2 => Csrrs(rd, rs1, imm as Csr),
            3 => Csrrc(rd, rs1, imm as Csr),
            5 => Csrrwi(rd, rs1 as Imm, imm as Csr),
            6 => Csrrsi(rd, rs1 as Imm, imm as Csr),
            7 => Csrrci(rd, rs1 as Imm, imm as Csr),
            _ => IllegalInstruction(current_ins),
        },
    }
}

#[inline]
fn decode_compressed_instruction(c_ins: u16) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    let c_rs1 = c_ins.bit_range(7..12) as Reg;
    let x_rs1 = c_ins.bit_range(7..10) as Reg;
    let x2_rs1 = c_ins.bit_range(2..5) as Reg;
    let c_rs2 = c_ins.bit_range(2..7) as Reg;
    let quadrant = c_ins & 0b11;
    let funct3 = c_ins.bit_range(13..16);

    match (quadrant, funct3) {
        (0, 0) => {
            let imm = u32::from(c_ins.bit_range(7..11)) << 6
                | u32::from(c_ins.bit_range(11..13)) << 4
                | (c_ins.bit(5) as u32) << 3
                | (c_ins.bit(6) as u32) << 2;
            trace!("c.addi4spn opcode: {c_ins:04x}");
            Caddi4spn(x2_rs1 + 8, imm)
        }
        (0, 1) => {
            let imm = (c_ins.bit_range(5..7) as u32) << 6 | (c_ins.bit_range(10..13) as u32) << 3;
            Cfld(x2_rs1 + 8, x_rs1 + 8, imm)
        }
        (0, 2) => {
            let imm = (c_ins.bit(5) as u32) << 6
                | (c_ins.bit_range(10..13) as u32) << 3
                | (c_ins.bit(6) as u32) << 2;
            Clw(x2_rs1 + 8, x_rs1 + 8, imm)
        }
        (0, 3) => {
            let imm = (c_ins.bit_range(5..7) as u32) << 6 | (c_ins.bit_range(10..13) as u32) << 3;
            Cld(x2_rs1 + 8, x_rs1 + 8, imm)
        }
        (0, 5) => {
            let imm = c_ins.bit_range(5..7) << 6 | c_ins.bit_range(10..13) << 3;
            Cfsd(x_rs1 + 8, x2_rs1 + 8, imm.into())
        }
        (0, 6) => {
            let imm = (c_ins.bit(5) as u32) << 6
                | (c_ins.bit_range(10..13) as u32) << 3
                | (c_ins.bit(6) as u32) << 2;
            Csw(x_rs1 + 8, x2_rs1 + 8, imm)
        }
        (0, 7) => {
            let imm = (c_ins.bit_range(5..7) as u32) << 6 | (c_ins.bit_range(10..13) as u32) << 3;
            Csd(x_rs1 + 8, x2_rs1 + 8, imm)
        }
        (1, 0) if c_ins == 0x0001 => Cnop,
        (1, 0) => {
            let simm = sign_extend(compressed_ci_imm(c_ins).into(), 6);
            Caddi(c_rs1, simm)
        }
        (1, 1) => {
            let simm = sign_extend(compressed_ci_imm(c_ins).into(), 6);
            Caddiw(c_rs1, simm)
        }
        (1, 2) => {
            trace!("c.li instruction: {c_ins:04x}");
            Cli(c_rs1, compressed_ci_imm(c_ins))
        }
        (1, 3) if c_rs1 == Sp as Reg => {
            let imm = (c_ins.bit(12) as u32) << 9
                | (c_ins.bit_range(3..5) as u32) << 7
                | (c_ins.bit(5) as u32) << 6
                | (c_ins.bit(2) as u32) << 5
                | (c_ins.bit(6) as u32) << 4;
            Caddi16sp(sign_extend(imm as u64, 10))
        }
        (1, 3) => {
            let imm = (c_ins.bit(12) as u32) << 17 | u32::from(c_ins.bit_range(2..7)) << 12;
            trace!("c.lui imm: {}", sign_extend(imm as u64, 18));
            Clui(c_rs1, imm)
        }
        (1, 4) => decode_compressed_alu_instruction(c_ins, x_rs1, x2_rs1),
        (1, 5) => Cj(compressed_jump_offset(c_ins)),
        (1, 6) => {
            let imm = compressed_branch_offset(c_ins);
            trace!("beqz imm: {imm}");
            Cbeqz(x_rs1 + 8, imm)
        }
        (1, 7) => Cbnez(x_rs1 + 8, compressed_branch_offset(c_ins)),
        (2, 0) => Cslli(c_rs1, compressed_ci_imm(c_ins)),
        (2, 1) => {
            let imm = (c_ins.bit_range(2..5) as u32) << 6
                | (c_ins.bit(12) as u32) << 5
                | (c_ins.bit_range(5..7) as u32) << 3;
            Cfldsp(c_rs1, imm)
        }
        (2, 2) => {
            let imm = (c_ins.bit_range(2..4) as u32) << 6
                | (c_ins.bit(12) as u32) << 5
                | (c_ins.bit_range(4..7) as u32) << 2;
            trace!("c.lwsp opcode: {c_ins:04x}");
            Clwsp(c_rs1, imm)
        }
        (2, 3) => {
            let imm = (c_ins.bit_range(2..5) as u32) << 6
                | (c_ins.bit(12) as u32) << 5
                | (c_ins.bit_range(5..7) as u32) << 3;
            trace!("c.ldsp opcode: {c_ins:04x}");
            Cldsp(c_rs1, imm)
        }
        (2, 4) => decode_compressed_jump_register_instruction(c_ins, c_rs1, c_rs2),
        (2, 5) => {
            let imm = c_ins.bit_range(7..10) << 6 | c_ins.bit_range(10..13) << 3;
            Cfsdsp(c_rs2, imm.into())
        }
        (2, 6) => {
            let imm = c_ins.bit_range(7..9) << 6 | c_ins.bit_range(9..13) << 2;
            Cswsp(c_rs2, imm.into())
        }
        (2, 7) => {
            let imm = (c_ins.bit_range(7..10) as u32) << 6 | (c_ins.bit_range(10..13) as u32) << 3;
            Csdsp(c_rs2, imm)
        }
        _ => IllegalInstruction(c_ins.into()),
    }
}

#[inline]
fn decode_compressed_alu_instruction(c_ins: u16, x_rs1: Reg, x2_rs1: Reg) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    let rd = x_rs1 + 8;
    let rs2 = x2_rs1 + 8;
    match c_ins.bit_range(10..12) {
        0 => Csrli(rd, compressed_ci_imm(c_ins)),
        1 => Csrai(rd, compressed_ci_imm(c_ins)),
        2 => Candi(rd, sign_extend(compressed_ci_imm(c_ins).into(), 6)),
        3 => match (c_ins.bit(12), c_ins.bit_range(5..7)) {
            (false, 0) => Csub(rd, rs2),
            (false, 1) => Cxor(rd, rs2),
            (false, 2) => Cor(rd, rs2),
            (false, 3) => Cand(rd, rs2),
            (true, 0) => Csubw(rd, rs2),
            (true, 1) => Caddw(rd, rs2),
            _ => IllegalInstruction(c_ins.into()),
        },
        _ => IllegalInstruction(c_ins.into()),
    }
}

#[inline]
fn decode_compressed_jump_register_instruction(
    c_ins: u16,
    c_rs1: Reg,
    c_rs2: Reg,
) -> RV64GCInstruction {
    use RV64GCInstruction::*;

    if !c_ins.bit(12) {
        if c_rs2 == 0 {
            Cjr(c_rs1)
        } else {
            trace!("c.mv opcode: {c_ins:04x}");
            Cmv(c_rs1, c_rs2)
        }
    } else if c_rs1 == 0 && c_rs2 == 0 {
        Cebreak
    } else if c_rs2 == 0 {
        Cjalr(c_rs1)
    } else {
        Cadd(c_rs1, c_rs2)
    }
}

#[inline]
fn store_offset(ins: u32) -> Imm {
    (ins.bit_range(25..32) << 5) | ins.bit_range(7..12)
}

#[inline]
fn branch_offset(ins: u32) -> Imm {
    (ins.bit(31) as u32) << 12
        | (ins.bit(7) as u32) << 11
        | ins.bit_range(25..31) << 5
        | ins.bit_range(8..12) << 1
}

#[inline]
fn compressed_ci_imm(c_ins: u16) -> Imm {
    (c_ins.bit(12) as u32) << 5 | u32::from(c_ins.bit_range(2..7))
}

#[inline]
fn compressed_branch_offset(c_ins: u16) -> Imm {
    (c_ins.bit(12) as u32) << 8
        | (c_ins.bit_range(5..7) as u32) << 6
        | (c_ins.bit(2) as u32) << 5
        | (c_ins.bit_range(10..12) as u32) << 3
        | (c_ins.bit_range(3..5) as u32) << 1
}

#[inline]
fn compressed_jump_offset(c_ins: u16) -> Imm {
    (c_ins.bit(12) as u32) << 11
        | (c_ins.bit(8) as u32) << 10
        | (c_ins.bit_range(9..11) as u32) << 8
        | (c_ins.bit(6) as u32) << 7
        | (c_ins.bit(7) as u32) << 6
        | (c_ins.bit(2) as u32) << 5
        | (c_ins.bit(11) as u32) << 4
        | (c_ins.bit_range(3..6) as u32) << 1
}

#[derive(Debug)]
pub struct RV64GC {
    pub registers: RV64GCRegisters,
    pub float_registers: RV64GCFloatRegisters,
    pub fcsr: FCSR,
    pub ram: Ram,
    pub filesystem: GuestFileSystem,
    tracer: Option<ExecutionTracer>,
    pub should_quit: bool,
    thread_id: u64,
    synthetic_thread: bool,
    synthetic_thread_yielded: bool,
    clear_child_tid: Option<u64>,
    synthetic_threads: Vec<RV64GC>,
    synthetic_thread_defer: u32,
    synthetic_thread_jit_options: Option<crate::jit::JitOptions>,
    executable_path: Option<PathBuf>,
    linux_sysroot: Option<PathBuf>,
    argv: Vec<String>,
    elf_bin: Vec<u8>,
    decode_cache: Vec<Option<DecodedInstruction>>,
    jit_runtime_fault: Option<String>,
}

pub(crate) const RV64GC_RAM_OFFSET: usize = std::mem::offset_of!(RV64GC, ram);

impl Clone for RV64GC {
    fn clone(&self) -> Self {
        Self {
            registers: self.registers.clone(),
            float_registers: self.float_registers.clone(),
            fcsr: self.fcsr,
            ram: self.ram.clone(),
            filesystem: self.filesystem.clone(),
            tracer: self
                .tracer
                .as_ref()
                .map(|tracer| ExecutionTracer::new(tracer.options())),
            should_quit: self.should_quit,
            thread_id: self.thread_id,
            synthetic_thread: self.synthetic_thread,
            synthetic_thread_yielded: false,
            clear_child_tid: self.clear_child_tid,
            synthetic_threads: Vec::new(),
            synthetic_thread_defer: 0,
            synthetic_thread_jit_options: self.synthetic_thread_jit_options,
            executable_path: self.executable_path.clone(),
            linux_sysroot: self.linux_sysroot.clone(),
            argv: self.argv.clone(),
            elf_bin: self.elf_bin.clone(),
            decode_cache: vec![None; DECODE_CACHE_SIZE],
            jit_runtime_fault: None,
        }
    }
}

impl Default for RV64GC {
    fn default() -> Self {
        Self::new()
    }
}

impl RV64GC {
    fn initialize_stack_with_ext_lib(
        &mut self,
        elf: Elf,
        phdr_addr: Option<u64>,
        interpreter_base: Option<u64>,
    ) {
        use linux_libc_auxv::{AuxVar, AuxVarFlags, InitialLinuxLibcStackLayoutBuilder};

        let stack_top = 0x7FFF_FFFF_FFFF_FFF0;
        let stack_size: u64 = 8 * 1024 * 1024; // 8 MB
        let stack_start = stack_top - stack_size;
        let ram = &mut self.ram;
        let stack_region = MemoryRegion::new(stack_start, stack_size, vec![0; stack_size as usize]);
        ram.add_region(stack_region).unwrap();

        let mut builder = InitialLinuxLibcStackLayoutBuilder::new();
        let args_vec = self.guest_argv();
        let prog_name = args_vec.first().expect("guest argv is never empty");

        for arg in &args_vec {
            builder.arg_v.push(arg);
        }

        let envp_vec = [
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "TBB_NUM_THREADS=1",
        ];
        for env in envp_vec {
            builder.env_v.push(env);
        }

        let mut rand_bytes = [0u8; 16];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut rand_bytes);
        let mut auxv = vec![
            AuxVar::Phdr(phdr_addr.unwrap() as *const u8),
            AuxVar::Phent(elf.header.e_phentsize.into()),
            AuxVar::Phnum(elf.header.e_phnum.into()),
            AuxVar::Pagesz(4096),
            AuxVar::Entry(elf.header.e_entry as *const u8),
            AuxVar::HwCap(RISCV_HWCAP_IMAFDC),
            AuxVar::Platform("riscv64"),
            AuxVar::Uid(1000),
            AuxVar::Gid(1000),
            AuxVar::EUid(1000),
            AuxVar::EGid(1000),
            AuxVar::Secure(false),
            AuxVar::Random(rand_bytes),
            AuxVar::Clktck(100),
            AuxVar::Flags(AuxVarFlags::empty()),
            AuxVar::ExecFn(prog_name),
        ];

        if let Some(base) = interpreter_base {
            auxv.push(AuxVar::Base(base as *const u8));
        }

        auxv.into_iter().for_each(|e| {
            builder.aux_v.insert(e);
        });

        let stack_layout_size = builder.total_size();
        let mut stack_bytes = vec![0u8; stack_layout_size];
        let low_addr = (stack_top - stack_layout_size as u64) & !0xf;
        unsafe {
            builder.serialize_into_buf(stack_bytes.as_mut_slice(), low_addr);
        }

        for (idx, byte) in stack_bytes.into_iter().enumerate() {
            self.ram.write_byte(low_addr + idx as u64, byte).unwrap();
        }
        self.registers[Sp] = low_addr;
    }

    pub fn new() -> RV64GC {
        let mut registers = RV64GCRegisters::new();
        registers[Sp] = 0x7FFF_FFFF_FFFF_FFF0;

        let ram = Ram::new();

        let float_registers = RV64GCFloatRegisters::new();

        RV64GC {
            registers,
            float_registers,
            ram,
            filesystem: GuestFileSystem::new(),
            tracer: None,
            fcsr: FCSR::new(),
            should_quit: false,
            thread_id: std::process::id().into(),
            synthetic_thread: false,
            synthetic_thread_yielded: false,
            clear_child_tid: None,
            synthetic_threads: Vec::new(),
            synthetic_thread_defer: 0,
            synthetic_thread_jit_options: None,
            executable_path: None,
            linux_sysroot: None,
            argv: Vec::new(),
            elf_bin: vec![],
            decode_cache: vec![None; DECODE_CACHE_SIZE],
            jit_runtime_fault: None,
        }
    }

    pub fn set_stdin(&mut self, bytes: impl Into<Vec<u8>>) {
        self.filesystem.set_stdin(bytes);
    }

    pub fn mount_host_directory(
        &mut self,
        root: impl Into<std::path::PathBuf>,
    ) -> Result<(), FileSystemError> {
        self.filesystem.mount_host_directory(root)
    }

    pub fn mount_host_directory_at(
        &mut self,
        guest_prefix: &str,
        root: impl Into<std::path::PathBuf>,
    ) -> Result<(), FileSystemError> {
        self.filesystem.mount_host_directory_at(guest_prefix, root)
    }

    pub fn stdout(&self) -> &[u8] {
        self.filesystem.stdout()
    }

    pub fn stderr(&self) -> &[u8] {
        self.filesystem.stderr()
    }

    pub fn set_trace_options(&mut self, options: TraceOptions) {
        self.tracer = options.enabled().then(|| ExecutionTracer::new(options));
    }

    pub fn trace_report(&self) -> Option<String> {
        self.tracer.as_ref().map(ExecutionTracer::report)
    }

    pub fn hot_jit_blocks(&self, limit: usize) -> Option<Vec<(u64, u64)>> {
        self.tracer
            .as_ref()
            .map(|tracer| tracer.hot_jit_blocks(limit))
    }

    pub(crate) fn trace_start(&mut self, engine: ExecutionEngine) {
        if let Some(tracer) = self.tracer.as_mut() {
            tracer.start(engine);
        }
    }

    pub(crate) fn trace_finish(&mut self) {
        if let Some(tracer) = self.tracer.as_mut() {
            tracer.finish();
        }
    }

    pub(crate) fn tracer_mut(&mut self) -> Option<&mut ExecutionTracer> {
        self.tracer.as_mut()
    }

    pub(crate) fn tracer_enabled(&self) -> bool {
        self.tracer.is_some()
    }

    pub(crate) fn thread_id(&self) -> u64 {
        self.thread_id
    }

    pub(crate) fn set_thread_id(&mut self, thread_id: u64) {
        self.thread_id = thread_id;
    }

    pub(crate) fn is_synthetic_thread(&self) -> bool {
        self.synthetic_thread
    }

    pub(crate) fn set_synthetic_thread(&mut self, synthetic_thread: bool) {
        self.synthetic_thread = synthetic_thread;
    }

    pub(crate) fn prepare_synthetic_thread_run(&mut self) {
        self.should_quit = false;
        self.synthetic_thread_yielded = false;
    }

    pub(crate) fn yield_synthetic_thread(&mut self) {
        self.synthetic_thread_yielded = true;
        self.should_quit = true;
    }

    pub(crate) fn synthetic_thread_yielded(&self) -> bool {
        self.synthetic_thread_yielded
    }

    pub(crate) fn clear_child_tid(&self) -> Option<u64> {
        self.clear_child_tid
    }

    pub(crate) fn set_clear_child_tid(&mut self, clear_child_tid: Option<u64>) {
        self.clear_child_tid = clear_child_tid;
    }

    pub(crate) fn enqueue_synthetic_thread(&mut self, thread: RV64GC) {
        self.synthetic_threads.push(thread);
        self.synthetic_thread_defer = self
            .synthetic_thread_defer
            .max(SYNTHETIC_THREAD_DEFER_TICKS);
    }

    pub(crate) fn take_synthetic_threads(&mut self) -> Vec<RV64GC> {
        self.synthetic_thread_defer = 0;
        std::mem::take(&mut self.synthetic_threads)
    }

    pub(crate) fn make_synthetic_threads_ready(&mut self) {
        self.synthetic_thread_defer = 0;
    }

    pub(crate) fn tick_synthetic_threads(&mut self) {
        if !self.synthetic_threads.is_empty() {
            self.synthetic_thread_defer = self.synthetic_thread_defer.saturating_sub(1);
        }
    }

    pub(crate) fn synthetic_threads_ready(&self) -> bool {
        !self.synthetic_threads.is_empty() && self.synthetic_thread_defer == 0
    }

    pub(crate) fn merge_synthetic_thread_trace(&mut self, thread: &mut RV64GC) {
        if let (Some(parent), Some(child)) = (self.tracer.as_mut(), thread.tracer.as_mut()) {
            parent.merge_child(child);
            *child = ExecutionTracer::new(child.options());
        }
    }

    pub(crate) fn merge_synthetic_thread(&mut self, mut thread: RV64GC) {
        self.merge_synthetic_thread_trace(&mut thread);
        self.ram.copy_data_from(&thread.ram);
        self.filesystem = thread.filesystem;
    }

    pub(crate) fn synthetic_thread_jit_options(&self) -> Option<crate::jit::JitOptions> {
        self.synthetic_thread_jit_options
    }

    pub(crate) fn set_jit_runtime_fault(&mut self, reason: impl Into<String>) {
        if self.jit_runtime_fault.is_none() {
            self.jit_runtime_fault = Some(reason.into());
        }
        self.should_quit = true;
    }

    pub(crate) fn take_jit_runtime_fault(&mut self) -> Option<String> {
        self.jit_runtime_fault.take()
    }

    pub fn set_executable_path(&mut self, path: impl Into<PathBuf>) {
        self.executable_path = Some(path.into());
    }

    pub fn set_linux_sysroot(&mut self, path: impl Into<PathBuf>) {
        self.linux_sysroot = Some(path.into());
    }

    pub fn set_argv<I, S>(&mut self, argv: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.argv = argv.into_iter().map(Into::into).collect();
    }

    pub(crate) fn executable_path(&self) -> Option<&Path> {
        self.executable_path.as_deref()
    }

    pub(crate) fn elf_aot_entry_points(&self) -> Vec<u64> {
        let Ok(elf) = Elf::parse(&self.elf_bin) else {
            return Vec::new();
        };

        let mut entry_points = BTreeSet::new();
        for symbol in elf.syms.iter().chain(elf.dynsyms.iter()) {
            if symbol.st_value == 0 {
                continue;
            }

            match symbol.st_type() {
                sym::STT_FUNC | sym::STT_GNU_IFUNC => {
                    entry_points.insert(symbol.st_value);
                }
                _ => {}
            }
        }

        for section in &elf.section_headers {
            if !section.is_executable() || section.sh_addr == 0 || section.sh_size == 0 {
                continue;
            }

            let name = elf.shdr_strtab.get_at(section.sh_name).unwrap_or_default();
            if !name.contains("plt") {
                continue;
            }

            let mut offset = 0;
            while offset < section.sh_size {
                entry_points.insert(section.sh_addr + offset);
                offset += 16;
            }
        }

        entry_points.into_iter().collect()
    }

    pub(crate) fn elf_symbol_names(&self) -> Vec<(u64, String)> {
        let Ok(elf) = Elf::parse(&self.elf_bin) else {
            return Vec::new();
        };

        let mut symbols = Vec::new();
        for symbol in elf.syms.iter() {
            if symbol.st_value == 0 || symbol.st_type() != sym::STT_FUNC {
                continue;
            }
            if let Some(name) = elf.strtab.get_at(symbol.st_name) {
                symbols.push((symbol.st_value, name.to_string()));
            }
        }
        for symbol in elf.dynsyms.iter() {
            if symbol.st_value == 0 || symbol.st_type() != sym::STT_FUNC {
                continue;
            }
            if let Some(name) = elf.dynstrtab.get_at(symbol.st_name) {
                symbols.push((symbol.st_value, name.to_string()));
            }
        }

        symbols
    }

    pub(crate) fn elf_plt_symbol_names(&self) -> Vec<(u64, String)> {
        let Ok(elf) = Elf::parse(&self.elf_bin) else {
            return Vec::new();
        };
        let Some(plt_start) = elf.section_headers.iter().find_map(|section| {
            let name = elf.shdr_strtab.get_at(section.sh_name).unwrap_or_default();
            (name == ".plt" && section.sh_addr != 0).then_some(section.sh_addr)
        }) else {
            return Vec::new();
        };

        const RISCV_PLT_HEADER_SIZE: u64 = 32;
        const RISCV_PLT_ENTRY_SIZE: u64 = 16;
        elf.pltrelocs
            .iter()
            .enumerate()
            .filter_map(|(index, relocation)| {
                if relocation.r_type != reloc::R_RISCV_JUMP_SLOT {
                    return None;
                }
                let symbol = elf.dynsyms.get(relocation.r_sym)?;
                let name = elf.dynstrtab.get_at(symbol.st_name)?;
                let entry = plt_start
                    + RISCV_PLT_HEADER_SIZE
                    + (index as u64).saturating_mul(RISCV_PLT_ENTRY_SIZE);
                Some((entry, name.to_string()))
            })
            .collect()
    }

    pub fn elf_executable_ranges(&self) -> Vec<(u64, u64)> {
        let Ok(elf) = Elf::parse(&self.elf_bin) else {
            return Vec::new();
        };

        elf.section_headers
            .iter()
            .filter(|section| {
                section.is_executable() && section.sh_addr != 0 && section.sh_size != 0
            })
            .map(|section| (section.sh_addr, section.sh_addr + section.sh_size))
            .collect()
    }

    fn guest_argv(&self) -> Vec<String> {
        if !self.argv.is_empty() {
            return self.argv.clone();
        }

        let argv0 = self
            .executable_path
            .as_deref()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|| "riscvm".to_string());
        vec![argv0]
    }

    pub fn load_bin(&mut self, bin: Vec<u8>) {
        self.clear_decode_cache();
        let bin_load = MemoryRegion::new_with_flags(0, bin.len() as u64, bin, 1);
        self.ram.add_region(bin_load).unwrap();
        self.registers[Pc] = 0;
    }

    pub fn load_elf(&mut self, bin: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        self.clear_decode_cache();
        let span = span!(Level::TRACE, "load_elf");
        let _guard = span.enter();

        let elf = goblin::elf::Elf::parse(&bin)?;

        if elf.header.e_machine != goblin::elf::header::EM_RISCV {
            return Err("Not a RISC-V ELF".into());
        }

        let (ehdr, program_break_base) = self.load_elf_segments(&elf, &bin, 0)?;
        let interpreter_base = self.load_program_interpreter(&elf)?;

        if interpreter_base.is_none() {
            self.apply_dynamic_relocations(&elf)?;
        }

        self.ram.set_program_break_base(program_break_base);
        self.initialize_stack_with_ext_lib(elf, ehdr, interpreter_base);
        self.elf_bin = bin;

        trace!("mem regions: {}", self.ram);

        Ok(())
    }

    fn load_elf_segments(
        &mut self,
        elf: &Elf<'_>,
        bin: &[u8],
        load_bias: u64,
    ) -> Result<(Option<u64>, u64), Box<dyn std::error::Error>> {
        let mut ehdr = None;
        let mut program_break_base = 0;

        for ph in &elf.program_headers {
            trace!("Reading ph of type: {:#08x}", ph.p_type);
            match ph.p_type {
                goblin::elf::program_header::PT_LOAD => {
                    let v_addr = load_bias.wrapping_add(ph.p_vaddr);
                    let map_start = v_addr & !(PAGE_SIZE - 1);
                    let page_offset = v_addr - map_start;
                    if ph.p_offset == 0 {
                        ehdr = Some(v_addr + elf.header.e_phoff);
                    }
                    let mem_size = ph.p_memsz;
                    let mapped_size = align_up(page_offset + mem_size, PAGE_SIZE);
                    program_break_base = program_break_base.max(v_addr + mem_size);

                    let mut data = vec![0u8; mapped_size as usize];
                    let file_map_start = ph
                        .p_offset
                        .checked_sub(page_offset)
                        .ok_or("invalid ELF load segment alignment")?
                        as usize;
                    let file_map_len = page_offset
                        .checked_add(ph.p_filesz)
                        .ok_or("ELF load segment is too large")?
                        as usize;

                    for (i, byte) in bin[file_map_start..file_map_start + file_map_len]
                        .iter()
                        .enumerate()
                    {
                        data[i] = *byte;
                    }

                    let memory_region = MemoryRegion::new_with_flags(
                        map_start,
                        mapped_size,
                        data,
                        ph.p_flags.into(),
                    );

                    trace!(
                        "adding region, start: {}\t len: {}\toffset: {}",
                        map_start,
                        mapped_size,
                        ph.p_offset
                    );
                    self.ram.add_region(memory_region)?;
                }

                _ => trace!("skipping over ph type: {:08x}", ph.p_type),
            }
        }

        Ok((ehdr, program_break_base))
    }

    fn load_program_interpreter(
        &mut self,
        elf: &Elf<'_>,
    ) -> Result<Option<u64>, Box<dyn std::error::Error>> {
        let Some(interpreter) = elf.interpreter else {
            self.registers[Pc] = elf.entry;
            return Ok(None);
        };
        let Some(sysroot) = self.linux_sysroot.clone() else {
            self.registers[Pc] = elf.entry;
            return Ok(None);
        };

        let interpreter_path = interpreter.trim_start_matches('/');
        let host_interpreter_path = sysroot.join(interpreter_path);
        let interpreter_bin = fs::read(&host_interpreter_path)?;
        let interpreter_elf = goblin::elf::Elf::parse(&interpreter_bin)?;
        if interpreter_elf.header.e_machine != goblin::elf::header::EM_RISCV {
            return Err("ELF interpreter is not a RISC-V ELF".into());
        }

        self.load_elf_segments(&interpreter_elf, &interpreter_bin, DYNAMIC_LINKER_BASE)?;
        self.registers[Pc] = DYNAMIC_LINKER_BASE + interpreter_elf.entry;
        Ok(Some(DYNAMIC_LINKER_BASE))
    }

    fn apply_dynamic_relocations(
        &mut self,
        elf: &Elf<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for relocation in elf.dynrelas.iter().chain(elf.dynrels.iter()) {
            let addend = relocation.r_addend.unwrap_or(0) as u64;
            match relocation.r_type {
                reloc::R_RISCV_RELATIVE => {
                    self.ram.write_doubleword(relocation.r_offset, addend)?;
                }
                reloc::R_RISCV_64 => {
                    let symbol_value = elf
                        .dynsyms
                        .get(relocation.r_sym)
                        .map(|symbol| symbol.st_value)
                        .unwrap_or(0);
                    self.ram
                        .write_doubleword(relocation.r_offset, symbol_value.wrapping_add(addend))?;
                }
                reloc::R_RISCV_NONE => {}
                _ => {}
            }
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.registers = RV64GCRegisters::new();
        self.registers[Sp] = 0x7FFF_FFFF_FFFF_FFF0;
        self.float_registers = RV64GCFloatRegisters::new();
        self.ram = Ram::new();
        self.filesystem = GuestFileSystem::new();
        self.should_quit = false;
        self.thread_id = std::process::id().into();
        self.synthetic_thread = false;
        self.synthetic_thread_yielded = false;
        self.clear_child_tid = None;
        self.synthetic_threads.clear();
        self.synthetic_thread_defer = 0;
        self.synthetic_thread_jit_options = None;
        self.jit_runtime_fault = None;
        self.clear_decode_cache();

        self.load_elf(self.elf_bin.clone()).unwrap();
    }

    // NOTE: Takes mutable reference, to pass down the call stack
    pub fn start(&mut self) {
        let span = span!(Level::TRACE, "cpu loop");
        let _guard = span.enter();
        self.trace_start(ExecutionEngine::Interpreter);
        // while self.registers[Pc] <= (self.program.len() - 4) as u64 {
        //     self.step();
        // }

        while !self.should_quit {
            if crate::debug::termination_requested() {
                self.should_quit = true;
                break;
            }
            self.step();
        }
        self.trace_finish();
    }

    pub fn start_jit(&mut self) -> Result<(), crate::jit::JitError> {
        self.start_jit_with_options(crate::jit::JitOptions::default())
    }

    pub fn start_jit_with_options(
        &mut self,
        options: crate::jit::JitOptions,
    ) -> Result<(), crate::jit::JitError> {
        let mut jit = crate::jit::JitEngine::with_options(options)?;
        let previous_thread_jit_options = self.synthetic_thread_jit_options.replace(options);
        let result = jit.run(self);
        self.synthetic_thread_jit_options = previous_thread_jit_options;
        result
    }

    pub fn start_jit_with_options_and_profile<I>(
        &mut self,
        options: crate::jit::JitOptions,
        startup_profile: I,
    ) -> Result<(), crate::jit::JitError>
    where
        I: IntoIterator<Item = crate::jit::JitStartupProfileEntry>,
    {
        let mut jit = crate::jit::JitEngine::with_startup_profile(options, startup_profile)?;
        let previous_thread_jit_options = self.synthetic_thread_jit_options.replace(options);
        let result = jit.run(self);
        self.synthetic_thread_jit_options = previous_thread_jit_options;
        result
    }

    pub fn step(&mut self) {
        trace!("pc: {:08x}", self.registers[Pc]);

        // if self.points_to_break.contains(&self.registers[Pc]) {
        //     println!("{}", &self.registers);
        // }

        self.execute();
        self.registers[Zero] = 0;
        crate::syscalls::run_ready_synthetic_threads(self);
        assert_eq!(self.registers[Zero], 0);
    }

    pub fn execute(&mut self) {
        let decoded = self.decode_current_instruction();
        let current_ins = decoded.opcode;

        if dump_ops_enabled() {
            trace!("opcode: {current_ins:08x}");
        }

        let ins = decoded.instruction;
        self.trace_interpreter_instruction(decoded.pc, current_ins, &ins);
        ins.execute_instruction(self);

        if self.jit_runtime_fault.is_some() {
            return;
        }

        self.registers[Pc] = self.registers[Pc].wrapping_add(decoded.len);
    }

    fn clear_decode_cache(&mut self) {
        self.decode_cache.fill(None);
    }

    fn decode_current_instruction(&mut self) -> DecodedInstruction {
        let pc = self.registers[Pc];
        let code_version = self.ram.code_version();
        let index = decode_cache_index(pc);

        if let Some(decoded) = self.decode_cache[index] {
            if decoded.pc == pc && decoded.code_version == code_version {
                return decoded;
            }
        }

        let opcode = self.ram.read_word(pc).unwrap();
        let instruction = self.find_instruction(opcode);
        let decoded = DecodedInstruction {
            pc,
            code_version,
            opcode,
            instruction,
            len: instruction_len(opcode),
        };
        self.decode_cache[index] = Some(decoded);
        decoded
    }

    pub fn find_instruction(&self, current_ins: u32) -> RV64GCInstruction {
        decode_instruction(current_ins)
    }

    pub fn syscall_handler(&mut self) {
        let span = span!(Level::TRACE, "syscall_handler");
        let _guard = span.enter();

        let syscall_id = self.registers[A7];
        debug!("system call: {syscall_id}");
        self.trace_syscall(syscall_id);

        match syscall_id {
            17 => getcwd(self),

            20 => epoll_create1(self),

            21 => epoll_ctl(self),

            22 => epoll_pwait(self),

            23 => dup(self),

            24 => dup3(self),

            25 => fcntl(self),

            29 => ioctl(self),

            48 => faccessat(self),

            56 => openat(self),

            57 => close(self),

            61 => getdents64(self),

            62 => lseek(self),

            63 => read(self),

            64 => write(self),

            65 => readv(self),

            66 => writev(self),

            67 => pread64(self),

            68 => pwrite64(self),

            73 => ppoll(self),

            78 => readlink(self),

            79 => newfstatat(self),

            80 => fstat(self),

            93 => {
                let error_code = self.registers[A0];
                info!("Program exited with code: {error_code}");
                self.should_quit = true;
            }

            94 => {
                let error_code = self.registers[A0];
                info!("Program exited with code: {error_code}");
                self.should_quit = true;
            }

            96 => set_tid_address(self),

            98 => futex(self),

            // NOTE: set_robust_list
            99 => {
                self.registers[A0] = 0;
            }

            113 => clock_gettime(self),

            115 => clock_getres(self),

            122 => sched_setaffinity(self),

            123 => sched_getaffinity(self),

            131 => tgkill(self),

            132 => sigaltstack(self),

            134 => sig_action(self),

            135 => rt_sigprocmask(self),

            160 => uname(self),

            172 => getpid(self),
            173 => getppid(self),
            174 => getuid(self),
            175 => geteuid(self),
            176 => getgid(self),
            177 => getegid(self),
            178 => gettid(self),
            179 => sysinfo(self),

            214 => brk(self),

            215 => munmap(self),

            220 => sys_clone(self),

            222 => mmap(self),

            226 => mprotect(self),

            233 => madvise(self),

            258 => riscv_hwprobe(self),

            261 => prlimit64(self),

            278 => getrandom(self),

            293 => rseq(self),

            435 => clone3(self),

            // NOTE: Print i64
            1000 => {
                let ptr = self.registers[A0] as i64;
                info!("i64: {}", ptr);
            }

            // NOTE: Dump registers
            1001 => {
                info!("{}", self.registers);
            }

            // NOTE: Print i64 from ptr
            1100 => {
                let ptr = self.registers[A0];
                let val = self.ram.read_doubleword(ptr).unwrap();

                info!("i64: {}", val as i64);
            }
            //
            // NOTE: Print i32 from ptr
            1101 => {
                let ptr = self.registers[A0];
                let val = self.ram.read_word(ptr).unwrap();

                info!("i64: {}", val as i32);
            }

            // NOTE: Print float from ptr
            1110 => {
                let ptr = self.registers[A0];
                let value = f32::from_bits(self.ram.read_word(ptr).unwrap());

                info!("float: {}", value);
            }

            id => unimplemented_syscall(self, id),
        }
    }

    fn trace_interpreter_instruction(
        &mut self,
        pc: u64,
        opcode: u32,
        instruction: &RV64GCInstruction,
    ) {
        if let Some(tracer) = self.tracer.as_mut() {
            tracer.record_interpreter_instruction_lazy(pc, opcode, || instruction.to_string());
        }
    }

    fn trace_syscall(&mut self, syscall_id: u64) {
        if let Some(tracer) = self.tracer.as_mut() {
            tracer.record_syscall(syscall_id);
        }
    }

    pub(crate) fn read_csr(&self, csr: Csr) -> Result<u64, String> {
        match csr {
            CSR_FFLAGS => Ok(u64::from(self.fcsr.flags())),
            CSR_FRM => Ok(u64::from(self.fcsr.frm.bits())),
            CSR_FCSR => Ok(self.fcsr.bits()),
            _ => Err(format!("unsupported CSR 0x{csr:03x}")),
        }
    }

    pub(crate) fn write_csr(&mut self, csr: Csr, value: u64) -> Result<(), String> {
        match csr {
            CSR_FFLAGS => {
                self.fcsr.set_flags(value as u8);
                Ok(())
            }
            CSR_FRM => self
                .fcsr
                .set_rounding_mode_bits(value as u8)
                .map_err(|()| format!("invalid frm value {}", value & 0b111)),
            CSR_FCSR => self
                .fcsr
                .write_bits(value)
                .map_err(|()| format!("invalid fcsr.frm value {}", (value >> 5) & 0b111)),
            _ => Err(format!("unsupported CSR 0x{csr:03x}")),
        }
    }

    fn write_csr_result(&mut self, rd: Reg, value: u64) {
        if rd != Zero as u8 {
            self.registers[rd as usize] = value;
        }
    }

    pub(crate) fn csrrw(&mut self, rd: Reg, rs1: Reg, csr: Csr) -> Result<(), String> {
        let old = (rd != Zero as u8).then(|| self.read_csr(csr)).transpose()?;
        self.write_csr(csr, self.registers[rs1 as usize])?;
        if let Some(old) = old {
            self.write_csr_result(rd, old);
        }
        Ok(())
    }

    pub(crate) fn csrrs(&mut self, rd: Reg, rs1: Reg, csr: Csr) -> Result<(), String> {
        let old = self.read_csr(csr)?;
        if rs1 != Zero as u8 {
            self.write_csr(csr, old | self.registers[rs1 as usize])?;
        }
        self.write_csr_result(rd, old);
        Ok(())
    }

    pub(crate) fn csrrc(&mut self, rd: Reg, rs1: Reg, csr: Csr) -> Result<(), String> {
        let old = self.read_csr(csr)?;
        if rs1 != Zero as u8 {
            self.write_csr(csr, old & !self.registers[rs1 as usize])?;
        }
        self.write_csr_result(rd, old);
        Ok(())
    }

    pub(crate) fn csrrwi(&mut self, rd: Reg, uimm: Imm, csr: Csr) -> Result<(), String> {
        let old = (rd != Zero as u8).then(|| self.read_csr(csr)).transpose()?;
        self.write_csr(csr, u64::from(uimm & 0x1f))?;
        if let Some(old) = old {
            self.write_csr_result(rd, old);
        }
        Ok(())
    }

    pub(crate) fn csrrsi(&mut self, rd: Reg, uimm: Imm, csr: Csr) -> Result<(), String> {
        let old = self.read_csr(csr)?;
        if uimm != 0 {
            self.write_csr(csr, old | u64::from(uimm & 0x1f))?;
        }
        self.write_csr_result(rd, old);
        Ok(())
    }

    pub(crate) fn csrrci(&mut self, rd: Reg, uimm: Imm, csr: Csr) -> Result<(), String> {
        let old = self.read_csr(csr)?;
        if uimm != 0 {
            self.write_csr(csr, old & !u64::from(uimm & 0x1f))?;
        }
        self.write_csr_result(rd, old);
        Ok(())
    }

    fn record_instruction_fault(&mut self, reason: impl Into<String>) {
        self.set_jit_runtime_fault(reason);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RV64GCInstruction {
    Add(Reg, Reg, Reg),
    Addi(Reg, Reg, Simm),
    Auipc(Reg, Simm),
    Lui(Reg, Simm),
    Slti(Reg, Reg, Simm),
    Sltiu(Reg, Reg, Imm),
    Xori(Reg, Reg, Simm),
    Ori(Reg, Reg, Simm),
    Andi(Reg, Reg, Simm),
    Slli(Reg, Reg, Imm),
    Srli(Reg, Reg, Imm),
    Srai(Reg, Reg, Imm),
    Sub(Reg, Reg, Reg),
    Sll(Reg, Reg, Reg),
    Slt(Reg, Reg, Reg),
    Sltu(Reg, Reg, Reg),
    Xor(Reg, Reg, Reg),
    Srl(Reg, Reg, Reg),
    Sra(Reg, Reg, Reg),
    Or(Reg, Reg, Reg),
    And(Reg, Reg, Reg),
    Fence(Reg, Reg),
    FenceI,
    Csrrw(Reg, Reg, Csr),
    Csrrs(Reg, Reg, Csr),
    Csrrc(Reg, Reg, Csr),
    Csrrwi(Reg, Imm, Csr),
    Csrrsi(Reg, Imm, Csr),
    Csrrci(Reg, Imm, Csr),
    Ecall,
    Ebreak,
    Uret,
    Sret,
    Mret,
    Wfi,
    SfenceVma(Reg, Reg, Reg),
    Lb(Reg, Reg, Simm),
    Lh(Reg, Reg, Simm),
    Lw(Reg, Reg, Simm),
    Lbu(Reg, Reg, Simm),
    Lhu(Reg, Reg, Simm),
    Sb(Reg, Reg, Simm),
    Sh(Reg, Reg, Simm),
    Sw(Reg, Reg, Simm),
    Jal(Reg, Simm),
    Jalr(Reg, Reg, Simm),
    Beq(Reg, Reg, Imm),
    Bne(Reg, Reg, Imm),
    Blt(Reg, Reg, Imm),
    Bge(Reg, Reg, Imm),
    Bltu(Reg, Reg, Imm),
    Bgeu(Reg, Reg, Imm),
    IllegalInstruction(u32),
    Ld(Reg, Reg, Simm),
    Sd(Reg, Reg, Simm),
    Addiw(Reg, Reg, Imm),
    Slliw(Reg, Reg, Imm),
    Srliw(Reg, Reg, Imm),
    Sraiw(Reg, Reg, Imm),
    Addw(Reg, Reg, Reg),
    Subw(Reg, Reg, Reg),
    Sllw(Reg, Reg, Reg),
    Srlw(Reg, Reg, Reg),
    Sraw(Reg, Reg, Reg),
    Lwu(Reg, Reg, Imm),
    Mul(Reg, Reg, Reg),
    Mulh(Reg, Reg, Reg),
    Mulhsu(Reg, Reg, Reg),
    Mulhu(Reg, Reg, Reg),
    Div(Reg, Reg, Reg),
    Divu(Reg, Reg, Reg),
    Rem(Reg, Reg, Reg),
    Remu(Reg, Reg, Reg),
    Mulw(Reg, Reg, Reg),
    Divw(Reg, Reg, Reg),
    Divuw(Reg, Reg, Reg),
    Remw(Reg, Reg, Reg),
    Remuw(Reg, Reg, Reg),

    // NOTE: RV64A
    Lrw(Reg, Reg),
    Scw(Reg, Reg, Reg),
    Amoswapw(Reg, Reg, Reg),
    Amoaddw(Reg, Reg, Reg),
    Amoxorw(Reg, Reg, Reg),
    Amoandw(Reg, Reg, Reg),
    Amoorw(Reg, Reg, Reg),
    Amominw(Reg, Reg, Reg),
    Amomaxw(Reg, Reg, Reg),
    Amominuw(Reg, Reg, Reg),
    Amomaxuw(Reg, Reg, Reg),
    Lrd(Reg, Reg),
    Scd(Reg, Reg, Reg),
    Amoswapd(Reg, Reg, Reg),
    Amoaddd(Reg, Reg, Reg),
    Amoxord(Reg, Reg, Reg),
    Amoandd(Reg, Reg, Reg),
    Amoord(Reg, Reg, Reg),
    Amomind(Reg, Reg, Reg),
    Amomaxd(Reg, Reg, Reg),
    Amominud(Reg, Reg, Reg),
    Amomaxud(Reg, Reg, Reg),

    // NOTE: RV64F
    Fmadds(Reg, Reg, Reg, Reg, Reg),
    Fmsubs(Reg, Reg, Reg, Reg, Reg),
    Fnmsubs(Reg, Reg, Reg, Reg, Reg),
    Fnmadds(Reg, Reg, Reg, Reg, Reg),
    Fadds(Reg, Reg, Reg, Reg),
    Fsubs(Reg, Reg, Reg, Reg),
    Fmuls(Reg, Reg, Reg, Reg),
    Fdivs(Reg, Reg, Reg, Reg),
    Fsqrts(Reg, Reg, Reg),
    Fsgnjs(Reg, Reg, Reg),
    Fsgnjns(Reg, Reg, Reg),
    Fsgnjxs(Reg, Reg, Reg),
    Fmins(Reg, Reg, Reg),
    Fmaxs(Reg, Reg, Reg),
    Fcvtws(Reg, Reg, Reg),
    Fcvtwus(Reg, Reg, Reg),
    Fcvtls(Reg, Reg, Reg),
    Fcvtlus(Reg, Reg, Reg),
    Fmvxw(Reg, Reg),
    Feqs(Reg, Reg, Reg),
    Flts(Reg, Reg, Reg),
    Fles(Reg, Reg, Reg),
    Fclasss(Reg, Reg),
    Fcvtsw(Reg, Reg, Reg),
    Fcvtswu(Reg, Reg, Reg),
    Fcvtsl(Reg, Reg, Reg),
    Fcvtslu(Reg, Reg, Reg),
    Fmvwx(Reg, Reg),

    // NOTE: RV64D
    Fmaddd(Reg, Reg, Reg, Reg, Reg),
    Fmsubd(Reg, Reg, Reg, Reg, Reg),
    Fnmaddd(Reg, Reg, Reg, Reg, Reg),
    Fnmsubd(Reg, Reg, Reg, Reg, Reg),
    Faddd(Reg, Reg, Reg, Reg),
    Fsubd(Reg, Reg, Reg, Reg),
    Fmuld(Reg, Reg, Reg, Reg),
    Fdivd(Reg, Reg, Reg, Reg),
    Fsqrtd(Reg, Reg, Reg),
    Fsgnjd(Reg, Reg, Reg),
    Fsgnjnd(Reg, Reg, Reg),
    Fsgnjxd(Reg, Reg, Reg),
    Fmind(Reg, Reg, Reg),
    Fmaxd(Reg, Reg, Reg),
    Feqd(Reg, Reg, Reg),
    Fltd(Reg, Reg, Reg),
    Fled(Reg, Reg, Reg),
    Fclassd(Reg, Reg),
    Fcvtsd(Reg, Reg, Reg),
    Fcvtds(Reg, Reg, Reg),
    Fcvtwd(Reg, Reg, Reg),
    Fcvtwud(Reg, Reg, Reg),
    Fcvtld(Reg, Reg, Reg),
    Fcvtlud(Reg, Reg, Reg),
    Fcvtdwu(Reg, Reg, Reg),
    Fcvtdw(Reg, Reg, Reg),
    Fcvtdl(Reg, Reg, Reg),
    Fcvtdlu(Reg, Reg, Reg),
    Fmvdx(Reg, Reg),
    Flw(Reg, Reg, Imm),
    Fsw(Reg, Reg, Imm),
    Fld(Reg, Reg, Imm),
    Fsd(Reg, Reg, Simm),
    Fmvxd(Reg, Reg),

    // NOTE: RV64C
    Cebreak,
    Cjalr(Reg),
    Cadd(Reg, Reg),
    Cjr(Reg),
    Cmv(Reg, Reg),
    Caddi16sp(Simm),
    Clui(Reg, Imm),
    Caddi4spn(Reg, Imm),
    Cbeqz(Reg, Imm),
    Cbnez(Reg, Imm),
    Cli(Reg, Imm),
    Csw(Reg, Reg, Imm),
    Cfld(Reg, Reg, Imm),
    Clw(Reg, Reg, Imm),
    Cld(Reg, Reg, Imm),
    Cfsd(Reg, Reg, Imm),
    Cfsw(Reg, Reg, Imm),
    Csd(Reg, Reg, Imm),
    Cnop,
    Caddi(Reg, Simm),
    Caddiw(Reg, Simm),
    Csrli(Reg, Imm),
    Csrai(Reg, Imm),
    Candi(Reg, Simm),
    Csub(Reg, Reg),
    Cxor(Reg, Reg),
    Cor(Reg, Reg),
    Cand(Reg, Reg),
    Csubw(Reg, Reg),
    Caddw(Reg, Reg),
    Cj(Imm),
    Cslli(Reg, Imm),
    Cfldsp(Reg, Imm),
    Clwsp(Reg, Imm),
    Cflwsp(Reg, Imm),
    Cldsp(Reg, Imm),
    Cfsdsp(Reg, Imm),
    Cswsp(Reg, Imm),
    Csdsp(Reg, Imm),
}

impl RV64GCInstruction {
    pub fn execute_instruction(&self, cpu: &mut RV64GC) {
        use RV64GCInstruction::*;

        trace!("{}", self);

        match self {
            IllegalInstruction(i) => {
                cpu.record_instruction_fault(format!("illegal instruction 0x{i:08x}"));
            }

            Add(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_add(cpu.registers[rs2]);
            }

            Addi(rd, rs1, simm) => {
                trace!("addi rs1: {}", cpu.registers[rs1]);
                cpu.registers[rd] = cpu.registers[rs1].wrapping_add_signed(*simm);
            }

            Auipc(rd, simm) => {
                cpu.registers[rd] = cpu.registers[Pc].wrapping_add_signed(*simm);
            }

            Lui(rd, simm) => {
                cpu.registers[rd] = *simm as u64;
            }

            Slti(rd, rs1, simm) => {
                let rs1 = cpu.registers[rs1] as i64;

                if rs1 < *simm {
                    cpu.registers[rd] = 1;
                } else {
                    cpu.registers[rd] = 0;
                }
            }

            Sltiu(rd, rs1, imm) => {
                if cpu.registers[rs1] < sign_extend12(*imm) as u64 {
                    cpu.registers[rd] = 1;
                } else {
                    cpu.registers[rd] = 0;
                }
            }

            Xori(rd, rs1, simm) => {
                cpu.registers[rd] = cpu.registers[rs1] ^ *simm as u64;
            }

            Ori(rd, rs1, simm) => {
                cpu.registers[rd] = cpu.registers[rs1] | *simm as u64;
            }

            Andi(rd, rs1, simm) => {
                trace!("andi {} & {simm}", cpu.registers[rs1]);

                cpu.registers[rd] = cpu.registers[rs1] & *simm as u64;
            }

            Slli(rd, rs1, imm) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_shl(*imm);
            }

            Srli(rd, rs1, imm) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_shr(*imm);
            }

            Srai(rd, rs1, imm) => {
                cpu.registers[rd] = (cpu.registers[rs1] as i64).wrapping_shr(*imm) as u64;
            }

            Sub(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_sub(cpu.registers[rs2]);
            }

            Sll(rd, rs1, rs2) => {
                cpu.registers[rd] =
                    cpu.registers[rs1].wrapping_shl(cpu.registers[rs2] as u32 & 0b11_1111);
            }

            Slt(rd, rs1, rs2) => {
                if (cpu.registers[rs1] as i64) < (cpu.registers[rs2] as i64) {
                    cpu.registers[rd] = 1
                } else {
                    cpu.registers[rd] = 0
                }
            }

            Sltu(rd, rs1, rs2) => {
                if cpu.registers[rs1] < cpu.registers[rs2] {
                    cpu.registers[rd] = 1
                } else {
                    cpu.registers[rd] = 0
                }
            }

            Xor(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] ^ cpu.registers[rs2];
            }

            Srl(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] >> (cpu.registers[rs2] & 0b11_1111);
            }

            Sra(rd, rs1, rs2) => {
                cpu.registers[rd] =
                    ((cpu.registers[rs1] as i64) >> (cpu.registers[rs2] & 0b11_1111)) as u64;
            }

            Or(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] | cpu.registers[rs2];
            }

            And(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] & cpu.registers[rs2];
            }

            Fence(_, _) => {}

            FenceI => {}

            Uret => cpu.record_instruction_fault("uret is not supported in user-mode emulation"),

            Sret => cpu.record_instruction_fault("sret is not supported in user-mode emulation"),

            Mret => cpu.record_instruction_fault("mret is not supported in user-mode emulation"),

            Wfi => cpu.record_instruction_fault("wfi is not supported in user-mode emulation"),

            SfenceVma(_, _, _) => {
                cpu.record_instruction_fault("sfence.vma is not supported in user-mode emulation");
            }

            Csrrw(rd, rs1, csr) => {
                if let Err(reason) = cpu.csrrw(*rd, *rs1, *csr) {
                    cpu.record_instruction_fault(reason);
                }
            }
            Csrrs(rd, rs1, csr) => {
                if let Err(reason) = cpu.csrrs(*rd, *rs1, *csr) {
                    cpu.record_instruction_fault(reason);
                }
            }
            Csrrc(rd, rs1, csr) => {
                if let Err(reason) = cpu.csrrc(*rd, *rs1, *csr) {
                    cpu.record_instruction_fault(reason);
                }
            }
            Csrrwi(rd, uimm, csr) => {
                if let Err(reason) = cpu.csrrwi(*rd, *uimm, *csr) {
                    cpu.record_instruction_fault(reason);
                }
            }
            Csrrsi(rd, uimm, csr) => {
                if let Err(reason) = cpu.csrrsi(*rd, *uimm, *csr) {
                    cpu.record_instruction_fault(reason);
                }
            }
            Csrrci(rd, uimm, csr) => {
                if let Err(reason) = cpu.csrrci(*rd, *uimm, *csr) {
                    cpu.record_instruction_fault(reason);
                }
            }

            Ecall => {
                cpu.syscall_handler();
            }

            Ebreak => {
                cpu.record_instruction_fault(format!("ebreak at pc 0x{:08x}", cpu.registers[Pc]));
            }

            // This was previously checking the sign bit at the 4th bit,
            // absolutely stupid...
            Lb(rd, rs1, simm) => {
                let res = cpu
                    .ram
                    .read_byte(cpu.registers[rs1].wrapping_add_signed(*simm));

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                cpu.registers[rd] = sign_extend(res.into(), 8) as u64;
            }

            Lbu(rd, rs1, simm) => {
                let res = cpu
                    .ram
                    .read_byte(cpu.registers[rs1].wrapping_add_signed(*simm));

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();
                cpu.registers[rd] = res.into();
            }

            Lhu(rd, rs1, simm) => {
                let res = cpu
                    .ram
                    .read_halfword(cpu.registers[rs1].wrapping_add_signed(*simm));

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                cpu.registers[rd] = res;
            }

            Sb(rs1, rs2, simm) => {
                let addr = (cpu.registers[rs1] as i64).wrapping_add(*simm);

                cpu.ram
                    .write_byte(addr as u64, cpu.registers[rs2] as u8)
                    .unwrap();
            }

            Sh(rs1, rs2, simm) => {
                let addr = (cpu.registers[rs1] as i64).wrapping_add(*simm);
                let value = cpu.registers[rs2] as u16;

                cpu.ram.write_halfword(addr as u64, value as u64).unwrap();
            }

            Lh(rd, rs1, simm) => {
                let res = cpu
                    .ram
                    .read_halfword(cpu.registers[rs1].wrapping_add_signed(*simm));

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                cpu.registers[rd] = sign_extend(res, 16) as u64;
            }

            Lw(rd, rs1, simm) => {
                let res = cpu
                    .ram
                    .read_word(cpu.registers[rs1].wrapping_add_signed(*simm));

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                cpu.registers[rd] = sign_extend(res.into(), 32) as u64;
            }

            Sw(rs1, rs2, simm) => {
                let addr = cpu.registers[rs1] as i64 + simm;

                cpu.ram
                    .write_word(addr as u64, cpu.registers[rs2] as u32)
                    .inspect_err(|e| panic!("{e}"))
                    .unwrap();
            }

            Ld(rd, rs1, simm) => {
                let addr = cpu.registers[rs1].wrapping_add_signed(*simm);

                trace!("ld addr: {addr:08x}");

                cpu.registers[rd] = cpu
                    .ram
                    .read_doubleword(addr)
                    .inspect_err(|e| panic!("{e}"))
                    .unwrap();
            }

            Sd(rs1, rs2, simm) => {
                let addr = cpu.registers[rs1].wrapping_add_signed(*simm);

                trace!("sd addr: {addr:08x}");

                cpu.ram
                    .write_doubleword(addr, cpu.registers[rs2])
                    .inspect_err(|e| panic!("{e}\nAddress: {:08x}", cpu.registers[rs1]))
                    .unwrap();
            }

            Jal(rd, simm) => {
                let span = span!(Level::TRACE, "jal");
                let _guard = span.enter();

                if *rd > 0 {
                    cpu.registers[rd] = cpu.registers[Pc] + 4;
                }
                cpu.registers[Pc] = (cpu.registers[Pc] as i64 + simm) as u64;

                // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                // behaviour
                cpu.registers[Pc] -= 4;
            }

            Jalr(rd, rs1, simm) => {
                let jump_addr = (cpu.registers[rs1] as i64).wrapping_add(*simm);

                if *rd > 0 {
                    cpu.registers[rd] = cpu.registers[Pc] + 4;
                }
                cpu.registers[Pc] = (jump_addr as u64) & !1;

                // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                // behaviour
                cpu.registers[Pc] -= 4;
            }

            Beq(rs1, rs2, imm) => {
                if cpu.registers[rs1] == cpu.registers[rs2] {
                    let simm = sign_extend(u64::from(*imm), 13);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bne(rs1, rs2, imm) => {
                if cpu.registers[rs1] != cpu.registers[rs2] {
                    let simm = sign_extend(u64::from(*imm), 13);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Blt(rs1, rs2, imm) => {
                let rs1 = cpu.registers[rs1] as i64;
                let rs2 = cpu.registers[rs2] as i64;
                if rs1 < rs2 {
                    trace!("blt: {rs1} < {rs2}");
                    let simm = sign_extend(u64::from(*imm), 13);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bge(rs1, rs2, imm) => {
                if (cpu.registers[rs1] as i64) >= (cpu.registers[rs2] as i64) {
                    let simm = sign_extend(u64::from(*imm), 13);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bltu(rs1, rs2, imm) => {
                if cpu.registers[rs1] < cpu.registers[rs2] {
                    let simm = sign_extend(u64::from(*imm), 13);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bgeu(rs1, rs2, imm) => {
                if cpu.registers[rs1] >= cpu.registers[rs2] {
                    let simm = sign_extend(u64::from(*imm), 13);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Addiw(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let value = (cpu.registers[rs1] as i32).wrapping_add(simm as i32);

                cpu.registers[rd] = sign_extend(value as u64, 32) as u64;
            }

            Slliw(rd, rs1, shamt) => {
                let val = (cpu.registers[rs1] as u32).wrapping_shl(*shamt);

                cpu.registers[rd] = sign_extend(val.into(), 32) as u64;
            }

            Srliw(rd, rs1, shamt) => {
                let val = (cpu.registers[rs1] as u32).wrapping_shr(*shamt);

                cpu.registers[rd] = sign_extend(val.into(), 32) as u64;
            }

            Sraiw(rd, rs1, shamt) => {
                trace!("sraiw");
                let bit_reg = cpu.registers[rs1].bit_range(0..32) as i32;
                let shifted_reg = sign_extend(u64::from((bit_reg.wrapping_shr(*shamt)) as u32), 32);

                cpu.registers[rd] = shifted_reg as u64;
            }

            Addw(rd, rs1, rs2) => {
                let rs1_low = (cpu.registers[rs1] & (u32::MAX as u64)) as i32;
                let rs2_low = (cpu.registers[rs2] & (u32::MAX as u64)) as i32;

                cpu.registers[rd] = sign_extend(rs1_low.wrapping_add(rs2_low) as u64, 32) as u64;
            }

            Subw(rd, rs1, rs2) => {
                let rs1_low = (cpu.registers[rs1] & (u32::MAX as u64)) as i32;
                let rs2_low = (cpu.registers[rs2] & (u32::MAX as u64)) as i32;

                cpu.registers[rd] = sign_extend(rs1_low.wrapping_sub(rs2_low) as u64, 32) as u64;
            }

            Sllw(rd, rs1, rs2) => {
                let shifted_val = cpu.registers[rs1] << (cpu.registers[rs2] & 0b11111);
                cpu.registers[rd] = sign_extend(shifted_val.bit_range(0..32), 32) as u64;
            }

            Srlw(rd, rs1, rs2) => {
                let shifted_val =
                    (cpu.registers[rs1].bit_range(0..32)) >> (cpu.registers[rs2] & 0b11111);
                cpu.registers[rd] = sign_extend(shifted_val, 32) as u64;
            }

            Sraw(rd, rs1, rs2) => {
                let shifted_val =
                    (cpu.registers[rs1].bit_range(0..32) as i32) >> (cpu.registers[rs2] & 0b11111);
                cpu.registers[rd] = sign_extend(u64::from(shifted_val as u32), 32) as u64;
            }

            Lwu(rd, rs1, offset) => {
                let addr = (cpu.registers[rs1] as i64).wrapping_add(sign_extend12(*offset));
                let mem = cpu.ram.read_word(addr as u64).unwrap();

                cpu.registers[rd] = u64::from(mem);
            }

            Mul(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_mul(cpu.registers[rs2]);
            }

            Mulh(rd, rs1, rs2) => {
                let multiplicand = cpu.registers[rs1] as i64 as i128;
                let multiplier = cpu.registers[rs2] as i64 as i128;

                cpu.registers[rd] = ((multiplicand * multiplier) >> 64) as u64;
            }

            Mulhsu(rd, rs1, rs2) => {
                let multiplicand = cpu.registers[rs1] as i64 as i128;
                let multiplier = cpu.registers[rs2] as u128 as i128;

                cpu.registers[rd] = ((multiplicand * multiplier) >> 64) as u64;
            }

            Mulhu(rd, rs1, rs2) => {
                let multiplicand = cpu.registers[rs1] as u128;
                let multiplier = cpu.registers[rs2] as u128;

                cpu.registers[rd] = ((multiplicand * multiplier) >> 64) as u64;
            }

            Div(rd, rs1, rs2) => {
                let dividend = cpu.registers[rs1] as i64;
                let divisor = cpu.registers[rs2] as i64;
                if divisor == 0 {
                    cpu.registers[rd] = u64::MAX;
                    return;
                }

                if dividend == i64::MIN && divisor == -1 {
                    cpu.registers[rd] = dividend as u64;
                    return;
                }

                let value = dividend.wrapping_div(divisor);
                cpu.registers[rd] = value as u64;
            }

            Divu(rd, rs1, rs2) => {
                let dividend = cpu.registers[rs1];
                let divisor = cpu.registers[rs2];
                if divisor == 0 {
                    cpu.registers[rd] = u64::MAX;
                    return;
                }

                let value = dividend.wrapping_div(divisor);
                cpu.registers[rd] = value;
            }

            Rem(rd, rs1, rs2) => {
                let dividend = cpu.registers[rs1] as i64;
                let divisor = cpu.registers[rs2] as i64;
                if divisor == 0 {
                    cpu.registers[rd] = dividend as u64;
                    return;
                }

                if dividend == i64::MIN && divisor == -1 {
                    cpu.registers[rd] = 0;
                    return;
                }

                let value = dividend.wrapping_rem(divisor);
                cpu.registers[rd] = value as u64;
            }

            Remu(rd, rs1, rs2) => {
                let dividend = cpu.registers[rs1];
                let divisor = cpu.registers[rs2];
                if divisor == 0 {
                    cpu.registers[rd] = dividend;
                    return;
                }

                let value = dividend.wrapping_rem(divisor);
                cpu.registers[rd] = value;
            }

            Mulw(rd, rs1, rs2) => {
                let result = (cpu.registers[rs1] as i64).wrapping_mul(cpu.registers[rs2] as i64);

                cpu.registers[rd] = sign_extend((result as u64) & u32::MAX as u64, 32) as u64;
            }

            Divw(rd, rs1, rs2) => {
                let signed_rs1 = (cpu.registers[rs1] & (u32::MAX as u64)) as i32;
                let signed_rs2 = (cpu.registers[rs2] & (u32::MAX as u64)) as i32;

                if signed_rs2 == 0 {
                    cpu.registers[rd] = u64::MAX;
                    return;
                }

                cpu.registers[rd] =
                    sign_extend(signed_rs1.wrapping_div(signed_rs2) as u64, 32) as u64;
            }

            Divuw(rd, rs1, rs2) => {
                let unsigned_rs1 = (cpu.registers[rs1] & (u32::MAX as u64)) as u32;
                let unsigned_rs2 = (cpu.registers[rs2] & (u32::MAX as u64)) as u32;

                if unsigned_rs2 == 0 {
                    cpu.registers[rd] = u64::MAX;
                    return;
                }

                cpu.registers[rd] =
                    sign_extend(unsigned_rs1.wrapping_div(unsigned_rs2) as u64, 32) as u64;
            }

            Remw(rd, rs1, rs2) => {
                let signed_rs1 = (cpu.registers[rs1] & (u32::MAX as u64)) as i32;
                let signed_rs2 = (cpu.registers[rs2] & (u32::MAX as u64)) as i32;

                if signed_rs2 == 0 {
                    cpu.registers[rd] = sign_extend(signed_rs1 as u64, 32) as u64;
                    return;
                }

                cpu.registers[rd] =
                    sign_extend(signed_rs1.wrapping_rem(signed_rs2) as u64, 32) as u64;
            }

            Remuw(rd, rs1, rs2) => {
                let unsigned_rs1 = (cpu.registers[rs1] & (u32::MAX as u64)) as u32;
                let unsigned_rs2 = (cpu.registers[rs2] & (u32::MAX as u64)) as u32;

                if unsigned_rs2 == 0 {
                    cpu.registers[rd] = sign_extend(unsigned_rs1 as u64, 32) as u64;
                    return;
                }

                cpu.registers[rd] =
                    sign_extend(unsigned_rs1.wrapping_rem(unsigned_rs2) as u64, 32) as u64;
            }

            // WARNING: RV64A
            // TODO: Properly implement RV64A for multithreading
            Lrw(rd, rs1) => {
                cpu.registers[rd] = sign_extend(
                    u64::from(cpu.ram.read_word(cpu.registers[rs1]).unwrap()),
                    32,
                ) as u64;
            }

            // WARNING: Does not check that the previous value was changed!
            Scw(rd, rs1, rs2) => {
                cpu.ram
                    .write_word(cpu.registers[rs1], cpu.registers[rs2] as u32)
                    .unwrap();
                cpu.registers[rd] = 0;
            }

            Amoswapw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as i32;
                cpu.ram
                    .write_word(cpu.registers[rs1], cpu.registers[rs2] as u32)
                    .unwrap();
                if *rd != 0 {
                    cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
                }
            }

            Amoaddw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as i32;
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        rs1_value.wrapping_add(cpu.registers[rs2] as i32) as u32,
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }

            Amoxorw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as i32;
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        (rs1_value ^ (cpu.registers[rs2] as i32)) as u32,
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }

            Amoorw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as i32;
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        (rs1_value | (cpu.registers[rs2] as i32)) as u32,
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }
            Amoandw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as i32;
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        (rs1_value & (cpu.registers[rs2] as i32)) as u32,
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }

            Amominw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as i32;
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        (rs1_value.min(cpu.registers[rs2] as i32)) as u32,
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }

            Amomaxw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as i32;
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        (rs1_value.max(cpu.registers[rs2] as i32)) as u32,
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }

            Amominuw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap();
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        rs1_value.min((cpu.registers[rs2] & u32::MAX as u64) as u32),
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }

            Amomaxuw(rd, rs1, rs2) => {
                let rs1_value = cpu.ram.read_word(cpu.registers[rs1]).unwrap();
                cpu.ram
                    .write_word(
                        cpu.registers[rs1],
                        rs1_value.max((cpu.registers[rs2] & u32::MAX as u64) as u32),
                    )
                    .unwrap();
                cpu.registers[rd] = sign_extend(rs1_value as u64, 32) as u64;
            }

            Lrd(rd, rs1) => {
                cpu.registers[rd] = cpu.ram.read_doubleword(cpu.registers[rs1]).unwrap();
            }

            Scd(rd, rs1, rs2) => {
                cpu.ram
                    .write_doubleword(cpu.registers[rs1], cpu.registers[rs2])
                    .unwrap();

                cpu.registers[rd] = 0;
            }

            Amoswapd(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap();

                cpu.ram
                    .write_doubleword(rs1_ptr, cpu.registers[rs2])
                    .unwrap();

                if *rd != 0 {
                    cpu.registers[rd] = rs1_value;
                }
            }

            Amoaddd(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap() as i64;

                cpu.ram
                    .write_doubleword(
                        rs1_ptr,
                        rs1_value.wrapping_add(cpu.registers[rs2] as i64) as u64,
                    )
                    .unwrap();
                cpu.registers[rd] = rs1_value as u64;
            }

            Amoandd(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap() as i64;

                cpu.ram
                    .write_doubleword(rs1_ptr, (rs1_value & cpu.registers[rs2] as i64) as u64)
                    .unwrap();
                cpu.registers[rd] = rs1_value as u64;
            }

            Amoxord(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap() as i64;

                cpu.ram
                    .write_doubleword(rs1_ptr, (rs1_value ^ cpu.registers[rs2] as i64) as u64)
                    .unwrap();
                cpu.registers[rd] = rs1_value as u64;
            }

            Amoord(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap() as i64;

                cpu.ram
                    .write_doubleword(rs1_ptr, (rs1_value | cpu.registers[rs2] as i64) as u64)
                    .unwrap();
                cpu.registers[rd] = rs1_value as u64;
            }

            Amomind(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap() as i64;

                cpu.ram
                    .write_doubleword(rs1_ptr, rs1_value.min(cpu.registers[rs2] as i64) as u64)
                    .unwrap();
                cpu.registers[rd] = rs1_value as u64;
            }

            Amominud(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap();

                cpu.ram
                    .write_doubleword(rs1_ptr, rs1_value.min(cpu.registers[rs2]))
                    .unwrap();
                cpu.registers[rd] = rs1_value;
            }

            Amomaxd(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap() as i64;

                cpu.ram
                    .write_doubleword(rs1_ptr, rs1_value.max(cpu.registers[rs2] as i64) as u64)
                    .unwrap();
                cpu.registers[rd] = rs1_value as u64;
            }

            Amomaxud(rd, rs1, rs2) => {
                let rs1_ptr = cpu.registers[rs1];
                let rs1_value = cpu.ram.read_doubleword(rs1_ptr).unwrap();

                cpu.ram
                    .write_doubleword(rs1_ptr, rs1_value.max(cpu.registers[rs2]))
                    .unwrap();
                cpu.registers[rd] = rs1_value;
            }

            // NOTE: RV64F
            Fmadds(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };

                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);
                let rs3 = f32::from_bits(cpu.float_registers[rs3] as u32);

                cpu.float_registers[rd] = nan_box_f32(round_f32((rs1 * rs2) + rs3, rm).to_bits());
            }

            Fmsubs(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };

                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);
                let rs3 = f32::from_bits(cpu.float_registers[rs3] as u32);

                cpu.float_registers[rd] = nan_box_f32(round_f32((rs1 * rs2) - rs3, rm).to_bits());
            }

            Fnmadds(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };

                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);
                let rs3 = f32::from_bits(cpu.float_registers[rs3] as u32);

                cpu.float_registers[rd] = nan_box_f32(round_f32(-(rs1 * rs2) - rs3, rm).to_bits());
            }

            Fnmsubs(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };

                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);
                let rs3 = f32::from_bits(cpu.float_registers[rs3] as u32);

                cpu.float_registers[rd] = nan_box_f32(round_f32(-(rs1 * rs2) + rs3, rm).to_bits());
            }

            Fadds(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                let res = round_f32(rs1 + rs2, rm);

                trace!("fadds rs1: {rs1}");

                cpu.float_registers[rd] = nan_box_f32(res.to_bits());
            }

            Fsubs(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                let res = round_f32(rs1 - rs2, rm);

                cpu.float_registers[rd] = nan_box_f32(res.to_bits());
            }

            Fmuls(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                let res = round_f32(rs1 * rs2, rm);

                cpu.float_registers[rd] = nan_box_f32(res.to_bits());
            }

            Fdivs(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                let res = round_f32(rs1 / rs2, rm);

                cpu.float_registers[rd] = nan_box_f32(res.to_bits());
            }

            Fsqrts(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);

                let res = round_f32(rs1.sqrt(), rm);

                cpu.float_registers[rd] = nan_box_f32(res.to_bits());
            }

            Fsgnjs(rd, rs1, rs2) => {
                let rs1 = cpu.float_registers[rs1] as u32;
                let rs2 = cpu.float_registers[rs2] as u32;

                let sign_bit = rs2.bit(31);
                let res = *rs1.bit_range(0..31).set_bit(31, sign_bit);

                cpu.float_registers[rd] = nan_box_f32(res);
            }

            Fsgnjns(rd, rs1, rs2) => {
                let rs1 = cpu.float_registers[rs1] as u32;
                let rs2 = cpu.float_registers[rs2] as u32;

                let sign_bit = rs2.bit(31);
                let res = *rs1.bit_range(0..31).set_bit(31, !sign_bit);

                cpu.float_registers[rd] = nan_box_f32(res);
            }

            Fsgnjxs(rd, rs1, rs2) => {
                let rs1 = cpu.float_registers[rs1] as u32;
                let rs2 = cpu.float_registers[rs2] as u32;

                let rs1_sign_bit = rs1.bit(31);
                let rs2_sign_bit = rs2.bit(31);
                let res = *rs1
                    .bit_range(0..31)
                    .set_bit(31, rs1_sign_bit ^ rs2_sign_bit);

                cpu.float_registers[rd] = nan_box_f32(res);
            }

            Fmins(rd, rs1, rs2) => {
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                cpu.float_registers[rd] = nan_box_f32(rs1.min(rs2).to_bits())
            }

            Fmaxs(rd, rs1, rs2) => {
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                cpu.float_registers[rd] = nan_box_f32(rs1.max(rs2).to_bits())
            }

            Fcvtws(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let val = round_f32(rs1, rm);

                cpu.registers[rd] = val as i64 as u64
            }

            Fcvtwus(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let val = round_f32(rs1, rm);

                cpu.registers[rd] = sign_extend(u64::from(val as u32), 32) as u64
            }

            Fcvtls(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let val = round_f32(rs1, rm);

                cpu.registers[rd] = val as i64 as u64
            }

            Fcvtlus(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let val = round_f32(rs1, rm);

                cpu.registers[rd] = val as u64
            }

            Fmvxw(rd, rs1) => {
                cpu.registers[rd] =
                    sign_extend(u64::from(cpu.float_registers[rs1] as u32), 32) as u64
            }

            Feqs(rd, rs1, rs2) => {
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                if rs1.is_nan() || rs2.is_nan() {
                    cpu.fcsr.set_flag(FCSR::NV);
                    cpu.registers[rd] = 0;
                    return;
                }

                let res = if rs1 == rs2 { 1 } else { 0 };

                cpu.registers[rd] = res;
            }

            Flts(rd, rs1, rs2) => {
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                if rs1.is_nan() || rs2.is_nan() {
                    cpu.fcsr.set_flag(FCSR::NV);
                    cpu.registers[rd] = 0;
                    return;
                }

                let res = if rs1 < rs2 { 1 } else { 0 };

                cpu.registers[rd] = res;
            }

            Fles(rd, rs1, rs2) => {
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                if rs1.is_nan() || rs2.is_nan() {
                    cpu.fcsr.set_flag(FCSR::NV);
                    cpu.registers[rd] = 0;
                    return;
                }

                let res = if rs1 <= rs2 { 1 } else { 0 };

                cpu.registers[rd] = res;
            }

            Fclasss(rd, rs1) => {
                let res = classify_f32(f32::from_bits(cpu.float_registers[rs1] as u32));
                cpu.registers[rd] = res as u64;
            }

            Fcvtsw(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let int = cpu.registers[rs1] as i32;
                cpu.float_registers[rd] = nan_box_f32(round_f32(int as f32, rm).to_bits());
            }

            Fcvtswu(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let int = cpu.registers[rs1] as u32;
                cpu.float_registers[rd] = nan_box_f32(round_f32(int as f32, rm).to_bits());
            }

            Fcvtsl(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let int = cpu.registers[rs1] as i64;
                cpu.float_registers[rd] = nan_box_f32(round_f32(int as f32, rm).to_bits());
            }

            Fcvtslu(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let int = cpu.registers[rs1];
                cpu.float_registers[rd] = nan_box_f32(round_f32(int as f32, rm).to_bits());
            }

            Fmvwx(rd, rs1) => {
                cpu.float_registers[rd] = nan_box_f32(cpu.registers[rs1] as u32);
            }

            Flw(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let addr = (cpu.registers[rs1] as i64).wrapping_add(simm) as u64;
                trace!("simm: {simm}");
                let value = cpu.ram.read_word(addr).unwrap();

                trace!("flw addr: {addr:08x}");

                cpu.float_registers[rd] = nan_box_f32(value);
            }

            Fsw(rs1, rs2, imm) => {
                let value = cpu.float_registers[rs2] as u32;
                trace!("fsw: {value}");
                cpu.ram
                    .write_word(
                        (cpu.registers[rs1] as i64).wrapping_add(sign_extend12(*imm)) as u64,
                        value,
                    )
                    .unwrap();
            }

            Fmaddd(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                let rs3 = f64::from_bits(cpu.float_registers[rs3]);
                cpu.float_registers[rd] = round_f64((rs1 * rs2) + rs3, rm).to_bits();
            }

            Fmsubd(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                let rs3 = f64::from_bits(cpu.float_registers[rs3]);
                cpu.float_registers[rd] = round_f64((rs1 * rs2) - rs3, rm).to_bits();
            }

            Fnmaddd(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                let rs3 = f64::from_bits(cpu.float_registers[rs3]);
                cpu.float_registers[rd] = round_f64(-(rs1 * rs2) - rs3, rm).to_bits();
            }

            Fnmsubd(rd, rm, rs1, rs2, rs3) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                let rs3 = f64::from_bits(cpu.float_registers[rs3]);
                cpu.float_registers[rd] = round_f64(-(rs1 * rs2) + rs3, rm).to_bits();
            }

            Faddd(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                cpu.float_registers[rd] = round_f64(rs1 + rs2, rm).to_bits();
            }

            Fsubd(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                cpu.float_registers[rd] = round_f64(rs1 - rs2, rm).to_bits();
            }

            Fmuld(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                cpu.float_registers[rd] = round_f64(rs1 * rs2, rm).to_bits();
            }

            Fdivd(rd, rm, rs1, rs2) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                cpu.float_registers[rd] = round_f64(rs1 / rs2, rm).to_bits();
            }

            Fsqrtd(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                cpu.float_registers[rd] = round_f64(rs1.sqrt(), rm).to_bits();
            }

            Fsd(rs1, rs2, simm) => {
                let value = cpu.float_registers[rs2];

                cpu.ram
                    .write_doubleword(cpu.registers[rs1].wrapping_add_signed(*simm), value)
                    .unwrap();
            }

            Fld(rd, rs1, imm) => {
                let addr = (cpu.registers[rs1] as i64).wrapping_add(sign_extend12(*imm)) as u64;
                cpu.float_registers[rd] = cpu.ram.read_doubleword(addr).unwrap();
            }

            Fmind(rd, rs1, rs2) => {
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                cpu.float_registers[rd] = rs1.min(rs2).to_bits();
            }

            Fmaxd(rd, rs1, rs2) => {
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                cpu.float_registers[rd] = rs1.max(rs2).to_bits();
            }

            Feqd(rd, rs1, rs2) => {
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                if rs1.is_nan() || rs2.is_nan() {
                    cpu.fcsr.set_flag(FCSR::NV);
                    cpu.registers[rd] = 0;
                    return;
                }
                cpu.registers[rd] = u64::from(rs1 == rs2);
            }

            Fltd(rd, rs1, rs2) => {
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                if rs1.is_nan() || rs2.is_nan() {
                    cpu.fcsr.set_flag(FCSR::NV);
                    cpu.registers[rd] = 0;
                    return;
                }
                cpu.registers[rd] = u64::from(rs1 < rs2);
            }

            Fled(rd, rs1, rs2) => {
                let rs1 = f64::from_bits(cpu.float_registers[rs1]);
                let rs2 = f64::from_bits(cpu.float_registers[rs2]);
                if rs1.is_nan() || rs2.is_nan() {
                    cpu.fcsr.set_flag(FCSR::NV);
                    cpu.registers[rd] = 0;
                    return;
                }
                cpu.registers[rd] = u64::from(rs1 <= rs2);
            }

            Fclassd(rd, rs1) => {
                let res = classify_f64(f64::from_bits(cpu.float_registers[rs1]));
                cpu.registers[rd] = res as u64;
            }

            Fcvtsd(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let double_precision = f64::from_bits(cpu.float_registers[rs1]);
                cpu.float_registers[rd] =
                    nan_box_f32(round_f32(double_precision as f32, rm).to_bits());
            }

            Fcvtds(rd, _, rs1) => {
                let single_precision = f32::from_bits(cpu.float_registers[rs1] as u32);
                cpu.float_registers[rd] = (single_precision as f64).to_bits();
            }

            Fcvtwd(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let value = f64::from_bits(cpu.float_registers[rs1]);
                cpu.registers[rd] = round_f64(value, rm) as i64 as u64;
            }

            Fcvtwud(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let value = f64::from_bits(cpu.float_registers[rs1]);
                cpu.registers[rd] = sign_extend(u64::from(round_f64(value, rm) as u32), 32) as u64;
            }

            Fcvtld(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let value = f64::from_bits(cpu.float_registers[rs1]);
                cpu.registers[rd] = round_f64(value, rm) as i64 as u64;
            }

            Fcvtlud(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let value = f64::from_bits(cpu.float_registers[rs1]);
                cpu.registers[rd] = round_f64(value, rm) as u64;
            }

            Fcvtdw(rd, _, rs1) => {
                cpu.float_registers[rd] = (cpu.registers[rs1] as i32 as f64).to_bits();
            }

            Fcvtdwu(rd, _, rs1) => {
                cpu.float_registers[rd] = (cpu.registers[rs1] as u32 as f64).to_bits();
            }

            Fcvtdl(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let value = cpu.registers[rs1] as i64 as f64;
                cpu.float_registers[rd] = round_f64(value, rm).to_bits();
            }

            Fcvtdlu(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let value = cpu.registers[rs1] as f64;
                cpu.float_registers[rd] = round_f64(value, rm).to_bits();
            }

            Fmvxd(rd, rs1) => cpu.registers[rd] = cpu.float_registers[rs1],

            Fmvdx(rd, rs1) => cpu.float_registers[rd] = cpu.registers[rs1],

            Fsgnjd(rd, rs1, rs2) => {
                let sign_bit = cpu.float_registers[rs2] & 0x8000000000000000;
                let magnitude = cpu.float_registers[rs1] & 0x7fff_ffff_ffff_ffff;
                cpu.float_registers[rd] = magnitude | sign_bit;
            }

            Fsgnjnd(rd, rs1, rs2) => {
                let sign_bit = (!cpu.float_registers[rs2]) & 0x8000000000000000;
                let magnitude = cpu.float_registers[rs1] & 0x7fff_ffff_ffff_ffff;
                cpu.float_registers[rd] = magnitude | sign_bit;
            }

            Fsgnjxd(rd, rs1, rs2) => {
                let sign_bit =
                    (cpu.float_registers[rs1] ^ cpu.float_registers[rs2]) & 0x8000000000000000;
                let magnitude = cpu.float_registers[rs1] & 0x7fff_ffff_ffff_ffff;
                cpu.float_registers[rd] = magnitude | sign_bit;
            }

            // NOTE: RV64C
            Cebreak => {
                cpu.record_instruction_fault(format!("c.ebreak at pc 0x{:08x}", cpu.registers[Pc]));
            }

            Cjalr(rs1) => {
                cpu.registers[Ra] = cpu.registers[Pc] + 2;
                // Subtract 2, since we add 2 after this instruction
                cpu.registers[Pc] = cpu.registers[rs1].wrapping_sub(2);
            }

            Cadd(rd, rs1) => cpu.registers[rd] = cpu.registers[rd].wrapping_add(cpu.registers[rs1]),

            Cor(rd, rs1) => cpu.registers[rd] |= cpu.registers[rs1],

            Cand(rd, rs1) => cpu.registers[rd] &= cpu.registers[rs1],

            Cxor(rd, rs1) => cpu.registers[rd] ^= cpu.registers[rs1],

            Cjr(rs1) => {
                // Subtract 2, since we add 2 after this instruction
                cpu.registers[Pc] = cpu.registers[rs1].wrapping_sub(2);
            }

            Cmv(rd, rs1) => cpu.registers[rd] = cpu.registers[rs1],

            Cldsp(rd, imm) => {
                let addr = cpu.registers[Sp] + *imm as u64;
                let res = cpu.ram.read_doubleword(addr);

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                cpu.registers[rd] = res;
            }

            Cfldsp(rd, imm) => {
                let addr = cpu.registers[Sp].wrapping_add(*imm as u64);
                cpu.float_registers[rd] = cpu.ram.read_doubleword(addr).unwrap();
            }

            Caddi4spn(rd, imm) => cpu.registers[rd] = cpu.registers[Sp] + u64::from(*imm),

            Caddi16sp(simm) => {
                cpu.registers[Sp] = cpu.registers[Sp].wrapping_add_signed(*simm);
            }

            Cli(rd, imm) => {
                let simm = sign_extend(u64::from(*imm), 6);
                trace!("c.li x{rd}, {simm}");
                cpu.registers[rd] = simm as u64;
            }

            Cslli(rd, imm) => cpu.registers[rd] = cpu.registers[rd].wrapping_shl(*imm),

            Csdsp(rs1, imm) => {
                let offset = cpu.registers[Sp].wrapping_add(*imm as u64);
                let res = cpu.ram.write_doubleword(offset, cpu.registers[rs1]);

                if let Err(res) = res {
                    panic!("{}", res);
                }
            }

            Cld(rd, rs1, imm) => {
                let res = cpu
                    .ram
                    .read_doubleword(cpu.registers[rs1].wrapping_add(*imm as u64));

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                cpu.registers[rd] = res;
            }

            Cfld(rd, rs1, imm) => {
                let addr = cpu.registers[rs1].wrapping_add(*imm as u64);
                cpu.float_registers[rd] = cpu.ram.read_doubleword(addr).unwrap();
            }

            Caddi(rd, simm) => {
                cpu.registers[rd] = (cpu.registers[rd] as i64).wrapping_add(*simm) as u64;
            }

            Cbeqz(rs1, imm) => {
                if cpu.registers[rs1] == 0 {
                    let simm = sign_extend(*imm as u64, 9);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;
                    trace!("c.beqz addr: {:08x}", cpu.registers[Pc]);
                    trace!("c.beqz simm: {simm}");

                    // Subtract 2, since we add 2 after this instruction
                    cpu.registers[Pc] = cpu.registers[Pc].wrapping_sub(2);
                }
            }

            Cbnez(rs1, imm) => {
                if cpu.registers[rs1] != 0 {
                    let simm = sign_extend(*imm as u64, 9);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Subtract 2, since we add 2 after this instruction
                    cpu.registers[Pc] = cpu.registers[Pc].wrapping_sub(2);
                }
            }

            Csd(rs1, rs2, imm) => {
                let offset = cpu.registers[rs1].wrapping_add(*imm as u64);
                let res = cpu.ram.write_doubleword(offset, cpu.registers[rs2]);

                if let Err(res) = res {
                    panic!("{}", res);
                }
            }

            Clui(rd, imm) => {
                let simm = sign_extend(*imm as u64, 18);
                cpu.registers[rd] = simm as u64;
            }

            Candi(rd, simm) => {
                cpu.registers[rd] = ((cpu.registers[rd] as i64) & simm) as u64;
            }

            Cj(imm) => {
                let simm = sign_extend12(*imm);
                cpu.registers[Pc] = (cpu.registers[Pc] as i64)
                    .wrapping_add(simm)
                    .wrapping_sub(2) as u64;
            }

            Csw(rs1, rs2, imm) => {
                let offset = cpu.registers[rs1].wrapping_add(*imm as u64);
                let res = cpu.ram.write_word(offset, cpu.registers[rs2] as u32);

                if let Err(res) = res {
                    panic!("{}", res);
                }
            }

            Csrli(rd, imm) => {
                cpu.registers[rd] = cpu.registers[rd].wrapping_shr(*imm);
            }

            Csrai(rd, imm) => {
                cpu.registers[rd] = (cpu.registers[rd] as i64).wrapping_shr(*imm) as u64;
            }

            Caddiw(rd, simm) => {
                let rd_val = cpu.registers[rd] as i64 as i32;

                cpu.registers[rd] =
                    sign_extend(rd_val.wrapping_add(*simm as i32) as u64, 32) as u64;
            }

            Clwsp(rd, imm) => {
                let res = cpu
                    .ram
                    .read_word(cpu.registers[Sp].wrapping_add(*imm as u64));

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                let val = sign_extend(res.into(), 32);
                cpu.registers[rd] = val as u64;
            }

            Cnop => {}

            Csub(rd, rs1) => {
                cpu.registers[rd] = cpu.registers[rd].wrapping_sub(cpu.registers[rs1]);
            }

            Clw(rd, rs1, imm) => {
                let offset = cpu.registers[rs1].wrapping_add(*imm as u64);
                let res = cpu.ram.read_word(offset);

                if let Err(res) = res {
                    panic!("{}", res);
                }

                let res = res.unwrap();

                cpu.registers[rd] = sign_extend(res.into(), 32) as u64;
            }

            Caddw(rd, rs1) => {
                let rd_val = cpu.registers[rd] as i32;
                let rs1_val = cpu.registers[rs1] as i32;

                cpu.registers[rd] = rd_val.wrapping_add(rs1_val) as i64 as u64;
            }

            Csubw(rd, rs1) => {
                let rd_val = cpu.registers[rd] as i32;
                let rs1_val = cpu.registers[rs1] as i32;

                cpu.registers[rd] = rd_val.wrapping_sub(rs1_val) as i64 as u64;
            }

            Cfsd(rs1, rs2, imm) => cpu
                .ram
                .write_doubleword(
                    cpu.registers[rs1] + u64::from(*imm),
                    cpu.float_registers[rs2],
                )
                .unwrap(),

            Cfsdsp(rs1, imm) => {
                cpu.ram
                    .write_doubleword(
                        cpu.registers[Sp] + u64::from(*imm),
                        cpu.float_registers[rs1],
                    )
                    .unwrap();
            }

            Cswsp(rs1, offset) => cpu
                .ram
                .write_word(
                    cpu.registers[Sp] + *offset as u64,
                    cpu.registers[rs1] as u32,
                )
                .unwrap(),

            _ => cpu.record_instruction_fault(format!("unimplemented instruction: {self:?}")),
        }
    }
}

impl Display for RV64GCInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use RV64GCInstruction::*;
        match self {
            Addi(rd, rs1, simm) => {
                write!(f, "addi x{rd}, x{rs1}, {}", simm)
            }

            Auipc(rd, imm) => {
                let simm = sign_extend(*imm as u64, 32);
                write!(f, "auipc x{rd}, {simm}")
            }

            Xori(rd, rs1, imm) => {
                write!(f, "xori x{rd}, x{rs1}, {imm}")
            }

            Lui(rd, simm) => {
                write!(f, "lui x{rd}, {simm}")
            }

            Srai(rd, rs1, imm) => {
                write!(f, "srai x{rd}, x{rs1}, {imm}")
            }

            Add(rd, rs1, rs2) => {
                write!(f, "add x{rd}, x{rs1}, x{rs2}")
            }

            Sub(rd, rs1, rs2) => {
                write!(f, "sub x{rd}, x{rs1}, x{rs2}")
            }

            Xor(rd, rs1, rs2) => {
                write!(f, "xor x{rd}, x{rs1}, x{rs2}")
            }

            Ecall => {
                write!(f, "ecall")
            }

            Sd(rs1, rs2, simm) => {
                write!(f, "sd x{rs2}, {simm}(x{rs1})")
            }

            Ld(rd, rs1, simm) => {
                write!(f, "ld x{rd}, {simm}(x{rs1})")
            }

            Jal(rd, imm) => {
                let simm = crate::sign_extend(*imm as u64, 20);
                write!(f, "jal x{rd}, {simm}")
            }

            Bne(rs1, rs2, imm) => {
                let simm = crate::sign_extend(*imm as u64, 13);
                write!(f, "bne x{rs1}, x{rs2}, {simm}")
            }

            Bge(rs1, rs2, imm) => {
                let simm = crate::sign_extend(*imm as u64, 13);
                write!(f, "bge x{rs1}, x{rs2}, {simm}")
            }

            Lw(rd, rs1, simm) => {
                write!(f, "lw x{rd}, x{rs1}, {simm}")
            }

            e => write!(f, "{e:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RV64GCRegisters {
    registers: [u64; 33],
}

impl Display for RV64GCRegisters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buf = String::new();
        for (i, c) in self.registers.iter().take(32).enumerate() {
            buf.push_str(&format!("x{i}: 0x{c:016x}\n"));
        }

        write!(f, "{buf}")
    }
}

impl Index<&u8> for RV64GCRegisters {
    type Output = u64;

    fn index(&self, index: &u8) -> &Self::Output {
        self.registers.get(*index as usize).unwrap()
    }
}

impl IndexMut<&u8> for RV64GCRegisters {
    fn index_mut(&mut self, index: &u8) -> &mut Self::Output {
        self.registers.get_mut(*index as usize).unwrap()
    }
}

impl Index<usize> for RV64GCRegisters {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        self.registers.get(index).unwrap()
    }
}

impl IndexMut<usize> for RV64GCRegisters {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.registers.get_mut(index).unwrap()
    }
}

impl Index<RV64GCRegAbiName> for RV64GCRegisters {
    type Output = u64;

    fn index(&self, index: RV64GCRegAbiName) -> &Self::Output {
        self.registers.get(index as usize).unwrap()
    }
}

impl IndexMut<RV64GCRegAbiName> for RV64GCRegisters {
    fn index_mut(&mut self, index: RV64GCRegAbiName) -> &mut Self::Output {
        self.registers.get_mut(index as usize).unwrap()
    }
}

impl Default for RV64GCRegisters {
    fn default() -> Self {
        Self::new()
    }
}

impl RV64GCRegisters {
    pub fn new() -> RV64GCRegisters {
        RV64GCRegisters {
            registers: [0u64; 33],
        }
    }

    #[cfg_attr(not(all(target_arch = "aarch64", unix)), allow(dead_code))]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u64 {
        self.registers.as_mut_ptr()
    }

    pub const fn float_reg(value: u8) -> u8 {
        value + 33
    }
}

#[derive(Debug, Clone)]
pub struct RV64GCFloatRegisters {
    registers: [u64; 32],
}

impl Index<&u8> for RV64GCFloatRegisters {
    type Output = u64;

    fn index(&self, index: &u8) -> &Self::Output {
        self.registers.get(*index as usize).unwrap()
    }
}

impl IndexMut<&u8> for RV64GCFloatRegisters {
    fn index_mut(&mut self, index: &u8) -> &mut Self::Output {
        self.registers.get_mut(*index as usize).unwrap()
    }
}

impl Index<usize> for RV64GCFloatRegisters {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        self.registers.get(index).unwrap()
    }
}

impl IndexMut<usize> for RV64GCFloatRegisters {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.registers.get_mut(index).unwrap()
    }
}

impl Index<RV64GCRegAbiName> for RV64GCFloatRegisters {
    type Output = u64;

    fn index(&self, index: RV64GCRegAbiName) -> &Self::Output {
        self.registers.get(index as usize).unwrap()
    }
}

impl IndexMut<RV64GCRegAbiName> for RV64GCFloatRegisters {
    fn index_mut(&mut self, index: RV64GCRegAbiName) -> &mut Self::Output {
        self.registers.get_mut(index as usize).unwrap()
    }
}

impl Default for RV64GCFloatRegisters {
    fn default() -> Self {
        Self::new()
    }
}

impl RV64GCFloatRegisters {
    pub fn new() -> RV64GCFloatRegisters {
        RV64GCFloatRegisters {
            registers: [0u64; 32],
        }
    }
}

#[derive(Debug)]
pub enum RV64GCRegAbiName {
    Zero = 0,
    Ra = 1,
    Sp = 2,
    Gp = 3,
    Tp = 4,
    T0 = 5,
    T1 = 6,
    T2 = 7,
    Fp = 8,
    S1 = 9,
    A0 = 10,
    A1 = 11,
    A2 = 12,
    A3 = 13,
    A4 = 14,
    A5 = 15,
    A6 = 16,
    A7 = 17,
    S2 = 18,
    S3 = 19,
    S4 = 20,
    S5 = 21,
    S6 = 22,
    S7 = 23,
    S8 = 24,
    S9 = 25,
    S10 = 26,
    S11 = 27,
    T3 = 28,
    T4 = 29,
    T5 = 30,
    T6 = 31,
    Pc = 32,
    F0 = 33,
    F1 = 34,
    F2 = 35,
    F3 = 36,
    F4 = 37,
    F5 = 38,
    F6 = 39,
    F7 = 40,
    F8 = 41,
    F9 = 42,
    F10 = 43,
    F11 = 44,
    F12 = 45,
    F13 = 46,
    F14 = 47,
    F15 = 48,
    F16 = 49,
    F17 = 50,
    F18 = 51,
    F19 = 52,
    F20 = 53,
    F21 = 54,
    F22 = 55,
    F23 = 56,
    F24 = 57,
    F25 = 58,
    F26 = 59,
    F27 = 60,
    F28 = 61,
    F29 = 62,
    F30 = 63,
    F31 = 64,
}

impl Display for RV64GCRegAbiName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reg = match self {
            Self::Zero => "zero",
            Self::Ra => "ra",
            Self::Sp => "sp",
            Self::Gp => "gp",
            Self::Tp => "tp",
            Self::T0 => "t0",
            Self::T1 => "t1",
            Self::T2 => "t2",
            Self::Fp => "fp",
            Self::S1 => "s1",
            Self::A0 => "a0",
            Self::A1 => "a1",
            Self::A2 => "a2",
            Self::A3 => "a3",
            Self::A4 => "a4",
            Self::A5 => "a5",
            Self::A6 => "a6",
            Self::A7 => "a7",
            Self::S2 => "s2",
            Self::S3 => "s3",
            Self::S4 => "s4",
            Self::S5 => "s5",
            Self::S6 => "s6",
            Self::S7 => "s7",
            Self::S8 => "s8",
            Self::S9 => "s9",
            Self::S10 => "s10",
            Self::S11 => "s11",
            Self::T3 => "t3",
            Self::T4 => "t4",
            Self::T5 => "t5",
            Self::T6 => "t6",
            Self::Pc => "pc",
            Self::F0 => "f0",
            Self::F1 => "f1",
            Self::F2 => "f2",
            Self::F3 => "f3",
            Self::F4 => "f4",
            Self::F5 => "f5",
            Self::F6 => "f6",
            Self::F7 => "f7",
            Self::F8 => "f8",
            Self::F9 => "f9",
            Self::F10 => "f10",
            Self::F11 => "f11",
            Self::F12 => "f12",
            Self::F13 => "f13",
            Self::F14 => "f14",
            Self::F15 => "f15",
            Self::F16 => "f16",
            Self::F17 => "f17",
            Self::F18 => "f18",
            Self::F19 => "f19",
            Self::F20 => "f20",
            Self::F21 => "f21",
            Self::F22 => "f22",
            Self::F23 => "f23",
            Self::F24 => "f24",
            Self::F25 => "f25",
            Self::F26 => "f26",
            Self::F27 => "f27",
            Self::F28 => "f28",
            Self::F29 => "f29",
            Self::F30 => "f30",
            Self::F31 => "f31",
        };

        write!(f, "{reg}")
    }
}

#[cfg(test)]
mod tests {
    use super::{RV64GCInstruction, RV64GC};
    use crate::cpu::RV64GCRegAbiName::*;
    use crate::ram::MemoryRegion;

    #[test]
    fn rv64_register_shifts_use_six_bit_shift_amounts() {
        let mut cpu = RV64GC::new();
        cpu.registers[A1] = 1;
        cpu.registers[A2] = 35;

        RV64GCInstruction::Sll(A0 as u8, A1 as u8, A2 as u8).execute_instruction(&mut cpu);
        assert_eq!(cpu.registers[A0], 1u64 << 35);

        RV64GCInstruction::Srl(A0 as u8, A0 as u8, A2 as u8).execute_instruction(&mut cpu);
        assert_eq!(cpu.registers[A0], 1);

        cpu.registers[A1] = 0x8000_0000_0000_0000;
        RV64GCInstruction::Sra(A0 as u8, A1 as u8, A2 as u8).execute_instruction(&mut cpu);
        assert_eq!(cpu.registers[A0], 0xffff_ffff_f000_0000);
    }

    #[test]
    fn lwu_zero_extends_loaded_word() {
        let mut cpu = RV64GC::new();
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 4, vec![0xff, 0xff, 0xff, 0xff]))
            .unwrap();
        cpu.registers[A1] = 0x1000;

        RV64GCInstruction::Lwu(A0 as u8, A1 as u8, 0).execute_instruction(&mut cpu);

        assert_eq!(cpu.registers[A0], 0xffff_ffff);
    }

    #[test]
    fn rv64_multiply_instructions_return_architectural_halves() {
        let mut cpu = RV64GC::new();
        cpu.registers[A1] = 0xffff_ffff_ffff_fffe;
        cpu.registers[A2] = 3;

        RV64GCInstruction::Mul(A0 as u8, A1 as u8, A2 as u8).execute_instruction(&mut cpu);
        assert_eq!(cpu.registers[A0], 0xffff_ffff_ffff_fffa);

        RV64GCInstruction::Mulh(A0 as u8, A1 as u8, A2 as u8).execute_instruction(&mut cpu);
        assert_eq!(cpu.registers[A0], 0xffff_ffff_ffff_ffff);

        RV64GCInstruction::Mulhsu(A0 as u8, A1 as u8, A2 as u8).execute_instruction(&mut cpu);
        assert_eq!(cpu.registers[A0], 0xffff_ffff_ffff_ffff);

        cpu.registers[A1] = u64::MAX;
        cpu.registers[A2] = u64::MAX;
        RV64GCInstruction::Mulhu(A0 as u8, A1 as u8, A2 as u8).execute_instruction(&mut cpu);
        assert_eq!(cpu.registers[A0], 0xffff_ffff_ffff_fffe);
    }

    #[test]
    fn branch_offsets_are_thirteen_bit_signed_immediates() {
        let mut cpu = RV64GC::new();
        cpu.registers[Pc] = 0x5000;
        cpu.registers[A0] = 1;
        cpu.registers[A1] = 2;

        RV64GCInstruction::Bne(A0 as u8, A1 as u8, 0x1000).execute_instruction(&mut cpu);

        assert_eq!(cpu.registers[Pc], 0x3ffc);
    }

    #[test]
    fn interpreter_step_discards_writes_to_x0() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(0x0050_0013u32.to_le_bytes().to_vec()); // addi x0, x0, 5

        cpu.step();

        assert_eq!(cpu.registers[Zero], 0);
    }
}
