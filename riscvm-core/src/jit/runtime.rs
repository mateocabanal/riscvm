use crate::cpu::RV64GCRegAbiName::Pc;
use crate::cpu::RV64GC;
use crate::fcsr::{classify_f32, classify_f64, round_f32, round_f64, RoundingMode, FCSR};
use crate::sign_extend;

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryWidth {
    Byte = 1,
    Half = 2,
    Word = 4,
    Double = 8,
}

impl MemoryWidth {
    pub(crate) fn bytes(self) -> u64 {
        self as u64
    }
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeBinaryOp {
    Addw,
    Div,
    Divu,
    Divuw,
    Divw,
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Mulw,
    Rem,
    Remu,
    Remuw,
    Remw,
    Sllw,
    Slt,
    Sltu,
    Sraw,
    Srlw,
    Subw,
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeAtomicOp {
    Amoaddd,
    Amoaddw,
    Amoandd,
    Amoandw,
    Amomaxd,
    Amomaxud,
    Amomaxuw,
    Amomaxw,
    Amomind,
    Amominud,
    Amominuw,
    Amominw,
    Amoord,
    Amoorw,
    Amoswapd,
    Amoswapw,
    Amoxord,
    Amoxorw,
    Lrd,
    Lrw,
    Scd,
    Scw,
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeCsrOp {
    Csrrw,
    Csrrs,
    Csrrc,
    Csrrwi,
    Csrrsi,
    Csrrci,
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeTrapOp {
    Ebreak,
    Cebreak,
    Uret,
    Sret,
    Mret,
    Wfi,
    SfenceVma,
    IllegalInstruction,
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeFloatOp {
    FmaddS,
    FmsubS,
    FnmaddS,
    FnmsubS,
    AddS,
    SubS,
    MulS,
    DivS,
    SqrtS,
    SgnjS,
    SgnjnS,
    SgnjxS,
    MinS,
    MaxS,
    CvtWS,
    CvtWuS,
    CvtLS,
    CvtLuS,
    MvXW,
    EqS,
    LtS,
    LeS,
    ClassS,
    CvtSW,
    CvtSWu,
    CvtSL,
    CvtSLu,
    MvWX,
    FmaddD,
    FmsubD,
    FnmaddD,
    FnmsubD,
    AddD,
    SubD,
    MulD,
    DivD,
    SqrtD,
    SgnjD,
    SgnjnD,
    SgnjxD,
    MinD,
    MaxD,
    EqD,
    LtD,
    LeD,
    ClassD,
    CvtSD,
    CvtDS,
    CvtWD,
    CvtWuD,
    CvtDW,
    CvtDWu,
    MvXD,
}

impl RuntimeFloatOp {
    fn needs_rounding_mode(self) -> bool {
        matches!(
            self,
            Self::FmaddS
                | Self::FmsubS
                | Self::FnmaddS
                | Self::FnmsubS
                | Self::AddS
                | Self::SubS
                | Self::MulS
                | Self::DivS
                | Self::SqrtS
                | Self::CvtWS
                | Self::CvtWuS
                | Self::CvtLS
                | Self::CvtLuS
                | Self::CvtSW
                | Self::CvtSWu
                | Self::CvtSL
                | Self::CvtSLu
                | Self::FmaddD
                | Self::FmsubD
                | Self::FnmaddD
                | Self::FnmsubD
                | Self::AddD
                | Self::SubD
                | Self::MulD
                | Self::DivD
                | Self::SqrtD
                | Self::CvtSD
                | Self::CvtWD
                | Self::CvtWuD
        )
    }
}

#[allow(dead_code)]
pub(crate) unsafe extern "C" fn jit_runtime_load(
    cpu: *mut RV64GC,
    addr: u64,
    width: u64,
    signed: u64,
) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "load") else {
        return 0;
    };
    let Some(value) = read_runtime_memory(cpu, addr, width, "load") else {
        return 0;
    };

    if signed != 0 {
        match width {
            1 => sign_extend(value, 8) as u64,
            2 => sign_extend(value, 16) as u64,
            4 => sign_extend(value, 32) as u64,
            8 => value,
            _ => 0,
        }
    } else {
        value
    }
}

pub(crate) unsafe extern "C" fn jit_runtime_load_u8(cpu: *mut RV64GC, addr: u64) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "load") else {
        return 0;
    };
    read_runtime_memory(cpu, addr, 1, "load").unwrap_or(0)
}

pub(crate) unsafe extern "C" fn jit_runtime_load_i8(cpu: *mut RV64GC, addr: u64) -> u64 {
    sign_extend(unsafe { jit_runtime_load_u8(cpu, addr) }, 8) as u64
}

pub(crate) unsafe extern "C" fn jit_runtime_load_u16(cpu: *mut RV64GC, addr: u64) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "load") else {
        return 0;
    };
    read_runtime_memory(cpu, addr, 2, "load").unwrap_or(0)
}

pub(crate) unsafe extern "C" fn jit_runtime_load_i16(cpu: *mut RV64GC, addr: u64) -> u64 {
    sign_extend(unsafe { jit_runtime_load_u16(cpu, addr) }, 16) as u64
}

pub(crate) unsafe extern "C" fn jit_runtime_load_u32(cpu: *mut RV64GC, addr: u64) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "load") else {
        return 0;
    };
    read_runtime_memory(cpu, addr, 4, "load").unwrap_or(0)
}

pub(crate) unsafe extern "C" fn jit_runtime_load_i32(cpu: *mut RV64GC, addr: u64) -> u64 {
    sign_extend(unsafe { jit_runtime_load_u32(cpu, addr) }, 32) as u64
}

pub(crate) unsafe extern "C" fn jit_runtime_load_u64(cpu: *mut RV64GC, addr: u64) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "load") else {
        return 0;
    };
    read_runtime_memory(cpu, addr, 8, "load").unwrap_or(0)
}

#[allow(dead_code)]
pub(crate) unsafe extern "C" fn jit_runtime_store(
    cpu: *mut RV64GC,
    addr: u64,
    value: u64,
    width: u64,
) {
    let Some(cpu) = runtime_cpu(cpu, "store") else {
        return;
    };
    write_runtime_memory(cpu, addr, value, width, "store");
}

pub(crate) unsafe extern "C" fn jit_runtime_store_u8(cpu: *mut RV64GC, addr: u64, value: u64) {
    let Some(cpu) = runtime_cpu(cpu, "store") else {
        return;
    };
    write_runtime_memory(cpu, addr, value, 1, "store");
}

pub(crate) unsafe extern "C" fn jit_runtime_store_u16(cpu: *mut RV64GC, addr: u64, value: u64) {
    let Some(cpu) = runtime_cpu(cpu, "store") else {
        return;
    };
    write_runtime_memory(cpu, addr, value, 2, "store");
}

pub(crate) unsafe extern "C" fn jit_runtime_store_u32(cpu: *mut RV64GC, addr: u64, value: u64) {
    let Some(cpu) = runtime_cpu(cpu, "store") else {
        return;
    };
    write_runtime_memory(cpu, addr, value, 4, "store");
}

pub(crate) unsafe extern "C" fn jit_runtime_store_u64(cpu: *mut RV64GC, addr: u64, value: u64) {
    let Some(cpu) = runtime_cpu(cpu, "store") else {
        return;
    };
    write_runtime_memory(cpu, addr, value, 8, "store");
}

pub(crate) unsafe extern "C" fn jit_runtime_direct_write_ptr(
    cpu: *mut RV64GC,
    addr: u64,
    len: u64,
) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "direct_write_ptr") else {
        return 0;
    };

    match cpu.ram.direct_write_ptr_range(addr, len) {
        Ok(ptr) => ptr as usize as u64,
        Err(error) => {
            record_runtime_fault(
                cpu,
                format!("direct write pointer failed at 0x{addr:016x} len={len}: {error}"),
            );
            0
        }
    }
}

pub(crate) unsafe extern "C" fn jit_runtime_try_direct_read_ptr(
    cpu: *mut RV64GC,
    addr: u64,
    len: u64,
) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "try_direct_read_ptr") else {
        return 0;
    };

    cpu.ram
        .direct_read_ptr_range(addr, len)
        .map(|ptr| ptr as usize as u64)
        .unwrap_or(0)
}

pub(crate) unsafe extern "C" fn jit_runtime_float_load(
    cpu: *mut RV64GC,
    rd: u64,
    addr: u64,
    width: u64,
) {
    let Some(cpu) = runtime_cpu(cpu, "float load") else {
        return;
    };
    if rd >= 32 {
        record_runtime_fault(cpu, format!("float load invalid register f{rd}"));
        return;
    }
    let Some(raw) = read_runtime_memory(cpu, addr, width, "float load") else {
        return;
    };
    let value = match width {
        4 => nan_box_f32(raw as u32),
        8 => raw,
        _ => return,
    };
    cpu.float_registers[rd as usize] = value;
}

pub(crate) unsafe extern "C" fn jit_runtime_float_store(
    cpu: *mut RV64GC,
    rs2: u64,
    addr: u64,
    width: u64,
) {
    let Some(cpu) = runtime_cpu(cpu, "float store") else {
        return;
    };
    if rs2 >= 32 {
        record_runtime_fault(cpu, format!("float store invalid register f{rs2}"));
        return;
    }
    let value = cpu.float_registers[rs2 as usize];
    write_runtime_memory(cpu, addr, value, width, "float store");
}

pub(crate) unsafe extern "C" fn jit_runtime_float_op(
    cpu: *mut RV64GC,
    op: u64,
    rd: u64,
    rm: u64,
    rs1: u64,
    rs2: u64,
    rs3: u64,
) {
    let Some(cpu) = runtime_cpu(cpu, "float op") else {
        return;
    };
    let Some(op) = runtime_float_op(op) else {
        record_runtime_fault(cpu, format!("unknown float op {op}"));
        return;
    };
    if rd >= 32 || rs1 >= 32 || rs2 >= 32 || rs3 >= 32 {
        record_runtime_fault(
            cpu,
            format!("float op invalid register rd={rd} rs1={rs1} rs2={rs2} rs3={rs3}"),
        );
        return;
    }

    let rounding_mode = match op.needs_rounding_mode() {
        true => match runtime_rounding_mode(cpu, rm) {
            Some(rounding_mode) => Some(rounding_mode),
            None => {
                record_runtime_fault(cpu, format!("float op invalid rounding mode {rm}"));
                return;
            }
        },
        false => None,
    };

    let lhs_bits = cpu.float_registers[rs1 as usize] as u32;
    let rhs_bits = cpu.float_registers[rs2 as usize] as u32;
    let lhs = f32::from_bits(lhs_bits);
    let rhs = f32::from_bits(rhs_bits);

    match op {
        RuntimeFloatOp::FmaddS
        | RuntimeFloatOp::FmsubS
        | RuntimeFloatOp::FnmaddS
        | RuntimeFloatOp::FnmsubS
        | RuntimeFloatOp::AddS
        | RuntimeFloatOp::SubS
        | RuntimeFloatOp::MulS
        | RuntimeFloatOp::DivS
        | RuntimeFloatOp::SqrtS
        | RuntimeFloatOp::CvtSW
        | RuntimeFloatOp::CvtSWu
        | RuntimeFloatOp::CvtSL
        | RuntimeFloatOp::CvtSLu => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            let result = match op {
                RuntimeFloatOp::FmaddS => {
                    let addend = f32::from_bits(cpu.float_registers[rs3 as usize] as u32);
                    (lhs * rhs) + addend
                }
                RuntimeFloatOp::FmsubS => {
                    let subtrahend = f32::from_bits(cpu.float_registers[rs3 as usize] as u32);
                    (lhs * rhs) - subtrahend
                }
                RuntimeFloatOp::FnmaddS => {
                    let addend = f32::from_bits(cpu.float_registers[rs3 as usize] as u32);
                    -(lhs * rhs) - addend
                }
                RuntimeFloatOp::FnmsubS => {
                    let subtrahend = f32::from_bits(cpu.float_registers[rs3 as usize] as u32);
                    -(lhs * rhs) + subtrahend
                }
                RuntimeFloatOp::AddS => lhs + rhs,
                RuntimeFloatOp::SubS => lhs - rhs,
                RuntimeFloatOp::MulS => lhs * rhs,
                RuntimeFloatOp::DivS => lhs / rhs,
                RuntimeFloatOp::SqrtS => lhs.sqrt(),
                RuntimeFloatOp::CvtSW => cpu.registers[rs1 as usize] as i32 as f32,
                RuntimeFloatOp::CvtSWu => cpu.registers[rs1 as usize] as u32 as f32,
                RuntimeFloatOp::CvtSL => cpu.registers[rs1 as usize] as i64 as f32,
                RuntimeFloatOp::CvtSLu => cpu.registers[rs1 as usize] as f32,
                _ => return,
            };
            cpu.float_registers[rd as usize] =
                nan_box_f32(round_f32(result, rounding_mode).to_bits());
        }
        RuntimeFloatOp::SgnjS | RuntimeFloatOp::SgnjnS | RuntimeFloatOp::SgnjxS => {
            let sign = match op {
                RuntimeFloatOp::SgnjS => rhs_bits & 0x8000_0000,
                RuntimeFloatOp::SgnjnS => (!rhs_bits) & 0x8000_0000,
                RuntimeFloatOp::SgnjxS => (lhs_bits ^ rhs_bits) & 0x8000_0000,
                _ => return,
            };
            cpu.float_registers[rd as usize] = nan_box_f32((lhs_bits & 0x7fff_ffff) | sign);
        }
        RuntimeFloatOp::MinS => {
            cpu.float_registers[rd as usize] = nan_box_f32(lhs.min(rhs).to_bits());
        }
        RuntimeFloatOp::MaxS => {
            cpu.float_registers[rd as usize] = nan_box_f32(lhs.max(rhs).to_bits());
        }
        RuntimeFloatOp::CvtWS => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            write_runtime_integer(cpu, rd, round_f32(lhs, rounding_mode) as i64 as u64);
        }
        RuntimeFloatOp::CvtWuS => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            write_runtime_integer(
                cpu,
                rd,
                sign_extend(u64::from(round_f32(lhs, rounding_mode) as u32), 32) as u64,
            );
        }
        RuntimeFloatOp::CvtLS => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            write_runtime_integer(cpu, rd, round_f32(lhs, rounding_mode) as i64 as u64);
        }
        RuntimeFloatOp::CvtLuS => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            write_runtime_integer(cpu, rd, round_f32(lhs, rounding_mode) as u64);
        }
        RuntimeFloatOp::MvXW => {
            write_runtime_integer(cpu, rd, sign_extend(u64::from(lhs_bits), 32) as u64);
        }
        RuntimeFloatOp::EqS | RuntimeFloatOp::LtS | RuntimeFloatOp::LeS => {
            if lhs.is_nan() || rhs.is_nan() {
                cpu.fcsr.set_flag(FCSR::NV);
                write_runtime_integer(cpu, rd, 0);
                return;
            }
            let result = match op {
                RuntimeFloatOp::EqS => lhs == rhs,
                RuntimeFloatOp::LtS => lhs < rhs,
                RuntimeFloatOp::LeS => lhs <= rhs,
                _ => return,
            };
            write_runtime_integer(cpu, rd, u64::from(result));
        }
        RuntimeFloatOp::ClassS => {
            write_runtime_integer(cpu, rd, u64::from(classify_f32(lhs)));
        }
        RuntimeFloatOp::MvWX => {
            cpu.float_registers[rd as usize] = nan_box_f32(cpu.registers[rs1 as usize] as u32);
        }
        RuntimeFloatOp::FmaddD
        | RuntimeFloatOp::FmsubD
        | RuntimeFloatOp::FnmaddD
        | RuntimeFloatOp::FnmsubD
        | RuntimeFloatOp::AddD
        | RuntimeFloatOp::SubD
        | RuntimeFloatOp::MulD
        | RuntimeFloatOp::DivD
        | RuntimeFloatOp::SqrtD => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            let lhs = f64::from_bits(cpu.float_registers[rs1 as usize]);
            let rhs = f64::from_bits(cpu.float_registers[rs2 as usize]);
            let result = match op {
                RuntimeFloatOp::FmaddD => {
                    let addend = f64::from_bits(cpu.float_registers[rs3 as usize]);
                    (lhs * rhs) + addend
                }
                RuntimeFloatOp::FmsubD => {
                    let subtrahend = f64::from_bits(cpu.float_registers[rs3 as usize]);
                    (lhs * rhs) - subtrahend
                }
                RuntimeFloatOp::FnmaddD => {
                    let addend = f64::from_bits(cpu.float_registers[rs3 as usize]);
                    -(lhs * rhs) - addend
                }
                RuntimeFloatOp::FnmsubD => {
                    let subtrahend = f64::from_bits(cpu.float_registers[rs3 as usize]);
                    -(lhs * rhs) + subtrahend
                }
                RuntimeFloatOp::AddD => lhs + rhs,
                RuntimeFloatOp::SubD => lhs - rhs,
                RuntimeFloatOp::MulD => lhs * rhs,
                RuntimeFloatOp::DivD => lhs / rhs,
                RuntimeFloatOp::SqrtD => lhs.sqrt(),
                _ => return,
            };
            cpu.float_registers[rd as usize] = round_f64(result, rounding_mode).to_bits();
        }
        RuntimeFloatOp::SgnjD | RuntimeFloatOp::SgnjnD | RuntimeFloatOp::SgnjxD => {
            let lhs_bits = cpu.float_registers[rs1 as usize];
            let rhs_bits = cpu.float_registers[rs2 as usize];
            let sign = match op {
                RuntimeFloatOp::SgnjD => rhs_bits & 0x8000_0000_0000_0000,
                RuntimeFloatOp::SgnjnD => (!rhs_bits) & 0x8000_0000_0000_0000,
                RuntimeFloatOp::SgnjxD => (lhs_bits ^ rhs_bits) & 0x8000_0000_0000_0000,
                _ => return,
            };
            cpu.float_registers[rd as usize] = (lhs_bits & 0x7fff_ffff_ffff_ffff) | sign;
        }
        RuntimeFloatOp::MinD => {
            let lhs = f64::from_bits(cpu.float_registers[rs1 as usize]);
            let rhs = f64::from_bits(cpu.float_registers[rs2 as usize]);
            cpu.float_registers[rd as usize] = lhs.min(rhs).to_bits();
        }
        RuntimeFloatOp::MaxD => {
            let lhs = f64::from_bits(cpu.float_registers[rs1 as usize]);
            let rhs = f64::from_bits(cpu.float_registers[rs2 as usize]);
            cpu.float_registers[rd as usize] = lhs.max(rhs).to_bits();
        }
        RuntimeFloatOp::EqD | RuntimeFloatOp::LtD | RuntimeFloatOp::LeD => {
            let lhs = f64::from_bits(cpu.float_registers[rs1 as usize]);
            let rhs = f64::from_bits(cpu.float_registers[rs2 as usize]);
            if lhs.is_nan() || rhs.is_nan() {
                cpu.fcsr.set_flag(FCSR::NV);
                write_runtime_integer(cpu, rd, 0);
                return;
            }
            let result = match op {
                RuntimeFloatOp::EqD => lhs == rhs,
                RuntimeFloatOp::LtD => lhs < rhs,
                RuntimeFloatOp::LeD => lhs <= rhs,
                _ => return,
            };
            write_runtime_integer(cpu, rd, u64::from(result));
        }
        RuntimeFloatOp::ClassD => {
            let value = f64::from_bits(cpu.float_registers[rs1 as usize]);
            write_runtime_integer(cpu, rd, u64::from(classify_f64(value)));
        }
        RuntimeFloatOp::CvtSD => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            let value = f64::from_bits(cpu.float_registers[rs1 as usize]);
            cpu.float_registers[rd as usize] =
                nan_box_f32(round_f32(value as f32, rounding_mode).to_bits());
        }
        RuntimeFloatOp::CvtDS => {
            let single = f32::from_bits(cpu.float_registers[rs1 as usize] as u32);
            cpu.float_registers[rd as usize] = f64::from(single).to_bits();
        }
        RuntimeFloatOp::CvtWD => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            let value = f64::from_bits(cpu.float_registers[rs1 as usize]);
            write_runtime_integer(cpu, rd, round_f64(value, rounding_mode) as i64 as u64);
        }
        RuntimeFloatOp::CvtWuD => {
            let Some(rounding_mode) = rounding_mode else {
                return;
            };
            let value = f64::from_bits(cpu.float_registers[rs1 as usize]);
            write_runtime_integer(
                cpu,
                rd,
                sign_extend(u64::from(round_f64(value, rounding_mode) as u32), 32) as u64,
            );
        }
        RuntimeFloatOp::CvtDW => {
            cpu.float_registers[rd as usize] =
                (cpu.registers[rs1 as usize] as i32 as f64).to_bits();
        }
        RuntimeFloatOp::CvtDWu => {
            cpu.float_registers[rd as usize] =
                (cpu.registers[rs1 as usize] as u32 as f64).to_bits();
        }
        RuntimeFloatOp::MvXD => {
            write_runtime_integer(cpu, rd, cpu.float_registers[rs1 as usize]);
        }
    }
}

pub(crate) unsafe extern "C" fn jit_runtime_ecall(cpu: *mut RV64GC) {
    let Some(cpu) = runtime_cpu(cpu, "ecall") else {
        return;
    };
    cpu.syscall_handler();
}

pub(crate) unsafe extern "C" fn jit_runtime_csr(
    cpu: *mut RV64GC,
    op: u64,
    rd: u64,
    rs1_or_uimm: u64,
    csr: u64,
) {
    let Some(cpu) = runtime_cpu(cpu, "csr") else {
        return;
    };
    let Some(op) = runtime_csr_op(op) else {
        record_runtime_fault(cpu, format!("unknown CSR op {op}"));
        return;
    };
    if rd >= 32 || rs1_or_uimm >= 32 || csr > u64::from(u16::MAX) {
        record_runtime_fault(
            cpu,
            format!("csr op invalid operands rd={rd} src={rs1_or_uimm} csr=0x{csr:x}"),
        );
        return;
    }

    let rd = rd as u8;
    let src = rs1_or_uimm as u8;
    let csr = csr as u16;
    let result = match op {
        RuntimeCsrOp::Csrrw => cpu.csrrw(rd, src, csr),
        RuntimeCsrOp::Csrrs => cpu.csrrs(rd, src, csr),
        RuntimeCsrOp::Csrrc => cpu.csrrc(rd, src, csr),
        RuntimeCsrOp::Csrrwi => cpu.csrrwi(rd, u32::from(src), csr),
        RuntimeCsrOp::Csrrsi => cpu.csrrsi(rd, u32::from(src), csr),
        RuntimeCsrOp::Csrrci => cpu.csrrci(rd, u32::from(src), csr),
    };
    if let Err(reason) = result {
        record_runtime_fault(cpu, reason);
    }
}

pub(crate) unsafe extern "C" fn jit_runtime_trap(cpu: *mut RV64GC, op: u64, pc: u64, opcode: u64) {
    let Some(cpu) = runtime_cpu(cpu, "trap") else {
        return;
    };
    cpu.registers[Pc] = pc;
    let Some(op) = runtime_trap_op(op) else {
        record_runtime_fault(cpu, format!("unknown trap op {op} at pc=0x{pc:016x}"));
        return;
    };
    let reason = match op {
        RuntimeTrapOp::Ebreak => format!("ebreak at pc=0x{pc:016x}"),
        RuntimeTrapOp::Cebreak => format!("c.ebreak at pc=0x{pc:016x}"),
        RuntimeTrapOp::Uret => format!("uret is not supported at pc=0x{pc:016x}"),
        RuntimeTrapOp::Sret => format!("sret is not supported at pc=0x{pc:016x}"),
        RuntimeTrapOp::Mret => format!("mret is not supported at pc=0x{pc:016x}"),
        RuntimeTrapOp::Wfi => format!("wfi is not supported at pc=0x{pc:016x}"),
        RuntimeTrapOp::SfenceVma => format!("sfence.vma is not supported at pc=0x{pc:016x}"),
        RuntimeTrapOp::IllegalInstruction => {
            format!("illegal instruction 0x{opcode:08x} at pc=0x{pc:016x}")
        }
    };
    record_runtime_fault(cpu, reason);
}

pub(crate) unsafe extern "C" fn jit_runtime_binary(op: u64, lhs: u64, rhs: u64) -> u64 {
    let Some(op) = runtime_binary_op(op) else {
        return 0;
    };
    match op {
        RuntimeBinaryOp::Addw => {
            sign_extend((lhs as i32).wrapping_add(rhs as i32) as u32 as u64, 32) as u64
        }
        RuntimeBinaryOp::Div => {
            let dividend = lhs as i64;
            let divisor = rhs as i64;
            if divisor == 0 {
                u64::MAX
            } else if dividend == i64::MIN && divisor == -1 {
                dividend as u64
            } else {
                dividend.wrapping_div(divisor) as u64
            }
        }
        RuntimeBinaryOp::Divu => {
            if rhs == 0 {
                u64::MAX
            } else {
                lhs.wrapping_div(rhs)
            }
        }
        RuntimeBinaryOp::Divuw => {
            let dividend = lhs as u32;
            let divisor = rhs as u32;
            if divisor == 0 {
                u64::MAX
            } else {
                sign_extend(u64::from(dividend.wrapping_div(divisor)), 32) as u64
            }
        }
        RuntimeBinaryOp::Divw => {
            let dividend = lhs as i32;
            let divisor = rhs as i32;
            if divisor == 0 {
                u64::MAX
            } else {
                sign_extend(dividend.wrapping_div(divisor) as u32 as u64, 32) as u64
            }
        }
        RuntimeBinaryOp::Mul => lhs.wrapping_mul(rhs),
        RuntimeBinaryOp::Mulh => {
            let multiplicand = lhs as i64 as i128;
            let multiplier = rhs as i64 as i128;
            ((multiplicand * multiplier) >> 64) as u64
        }
        RuntimeBinaryOp::Mulhsu => {
            let multiplicand = lhs as i64 as i128;
            let multiplier = rhs as u128 as i128;
            ((multiplicand * multiplier) >> 64) as u64
        }
        RuntimeBinaryOp::Mulhu => {
            let multiplicand = lhs as u128;
            let multiplier = rhs as u128;
            ((multiplicand * multiplier) >> 64) as u64
        }
        RuntimeBinaryOp::Mulw => sign_extend(
            ((lhs as i64).wrapping_mul(rhs as i64) as u64) & u64::from(u32::MAX),
            32,
        ) as u64,
        RuntimeBinaryOp::Rem => {
            let dividend = lhs as i64;
            let divisor = rhs as i64;
            if divisor == 0 {
                dividend as u64
            } else if dividend == i64::MIN && divisor == -1 {
                0
            } else {
                dividend.wrapping_rem(divisor) as u64
            }
        }
        RuntimeBinaryOp::Remu => {
            if rhs == 0 {
                lhs
            } else {
                lhs.wrapping_rem(rhs)
            }
        }
        RuntimeBinaryOp::Remuw => {
            let dividend = lhs as u32;
            let divisor = rhs as u32;
            if divisor == 0 {
                sign_extend(u64::from(dividend), 32) as u64
            } else {
                sign_extend(u64::from(dividend.wrapping_rem(divisor)), 32) as u64
            }
        }
        RuntimeBinaryOp::Remw => {
            let dividend = lhs as i32;
            let divisor = rhs as i32;
            if divisor == 0 {
                sign_extend(dividend as u32 as u64, 32) as u64
            } else {
                sign_extend(dividend.wrapping_rem(divisor) as u32 as u64, 32) as u64
            }
        }
        RuntimeBinaryOp::Sllw => sign_extend(
            (lhs as u32).wrapping_shl((rhs & 0b1_1111) as u32) as u64,
            32,
        ) as u64,
        RuntimeBinaryOp::Slt => u64::from((lhs as i64) < (rhs as i64)),
        RuntimeBinaryOp::Sltu => u64::from(lhs < rhs),
        RuntimeBinaryOp::Sraw => sign_extend(
            (lhs as i32).wrapping_shr((rhs & 0b1_1111) as u32) as u32 as u64,
            32,
        ) as u64,
        RuntimeBinaryOp::Srlw => sign_extend(
            u64::from((lhs as u32).wrapping_shr((rhs & 0b1_1111) as u32)),
            32,
        ) as u64,
        RuntimeBinaryOp::Subw => {
            sign_extend((lhs as i32).wrapping_sub(rhs as i32) as u32 as u64, 32) as u64
        }
    }
}

pub(crate) unsafe extern "C" fn jit_runtime_atomic(
    cpu: *mut RV64GC,
    op: u64,
    addr: u64,
    value: u64,
) -> u64 {
    let Some(cpu) = runtime_cpu(cpu, "atomic") else {
        return 0;
    };
    let Some(op) = runtime_atomic_op(op) else {
        record_runtime_fault(cpu, format!("invalid atomic op {op}"));
        return 0;
    };
    match op {
        RuntimeAtomicOp::Lrw => {
            let Some(old) = read_runtime_memory(cpu, addr, 4, "atomic load word") else {
                return 0;
            };
            sign_extend(old, 32) as u64
        }
        RuntimeAtomicOp::Scw => {
            write_runtime_memory(cpu, addr, value, 4, "atomic store word");
            0
        }
        RuntimeAtomicOp::Amoswapw => atomic_word(cpu, addr, |old| (old, value as u32)).unwrap_or(0),
        RuntimeAtomicOp::Amoaddw => {
            atomic_word(cpu, addr, |old| (old, old.wrapping_add(value as u32))).unwrap_or(0)
        }
        RuntimeAtomicOp::Amoxorw => {
            atomic_word(cpu, addr, |old| (old, old ^ value as u32)).unwrap_or(0)
        }
        RuntimeAtomicOp::Amoorw => {
            atomic_word(cpu, addr, |old| (old, old | value as u32)).unwrap_or(0)
        }
        RuntimeAtomicOp::Amoandw => {
            atomic_word(cpu, addr, |old| (old, old & value as u32)).unwrap_or(0)
        }
        RuntimeAtomicOp::Amominw => atomic_word(cpu, addr, |old| {
            let new = (old as i32).min(value as i32) as u32;
            (old, new)
        })
        .unwrap_or(0),
        RuntimeAtomicOp::Amomaxw => atomic_word(cpu, addr, |old| {
            let new = (old as i32).max(value as i32) as u32;
            (old, new)
        })
        .unwrap_or(0),
        RuntimeAtomicOp::Amominuw => {
            atomic_word(cpu, addr, |old| (old, old.min(value as u32))).unwrap_or(0)
        }
        RuntimeAtomicOp::Amomaxuw => {
            atomic_word(cpu, addr, |old| (old, old.max(value as u32))).unwrap_or(0)
        }
        RuntimeAtomicOp::Lrd => {
            read_runtime_memory(cpu, addr, 8, "atomic load doubleword").unwrap_or(0)
        }
        RuntimeAtomicOp::Scd => {
            write_runtime_memory(cpu, addr, value, 8, "atomic store doubleword");
            0
        }
        RuntimeAtomicOp::Amoswapd => atomic_doubleword(cpu, addr, |_| value).unwrap_or(0),
        RuntimeAtomicOp::Amoaddd => atomic_doubleword(cpu, addr, |old| {
            (old as i64).wrapping_add(value as i64) as u64
        })
        .unwrap_or(0),
        RuntimeAtomicOp::Amoxord => atomic_doubleword(cpu, addr, |old| old ^ value).unwrap_or(0),
        RuntimeAtomicOp::Amoord => atomic_doubleword(cpu, addr, |old| old | value).unwrap_or(0),
        RuntimeAtomicOp::Amoandd => atomic_doubleword(cpu, addr, |old| old & value).unwrap_or(0),
        RuntimeAtomicOp::Amomind => {
            atomic_doubleword(cpu, addr, |old| (old as i64).min(value as i64) as u64).unwrap_or(0)
        }
        RuntimeAtomicOp::Amomaxd => {
            atomic_doubleword(cpu, addr, |old| (old as i64).max(value as i64) as u64).unwrap_or(0)
        }
        RuntimeAtomicOp::Amominud => {
            atomic_doubleword(cpu, addr, |old| old.min(value)).unwrap_or(0)
        }
        RuntimeAtomicOp::Amomaxud => {
            atomic_doubleword(cpu, addr, |old| old.max(value)).unwrap_or(0)
        }
    }
}

fn nan_box_f32(bits: u32) -> u64 {
    0xffff_ffff_0000_0000 | u64::from(bits)
}

fn write_runtime_integer(cpu: &mut RV64GC, register: u64, value: u64) {
    if register != 0 && register < 32 {
        cpu.registers[register as usize] = value;
    }
}

fn runtime_cpu<'a>(cpu: *mut RV64GC, operation: &str) -> Option<&'a mut RV64GC> {
    let cpu = unsafe { cpu.as_mut() };
    if cpu.is_none() {
        eprintln!("[jit-runtime] {operation} received null CPU pointer");
    }
    cpu
}

fn read_runtime_memory(cpu: &mut RV64GC, addr: u64, width: u64, operation: &str) -> Option<u64> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        record_runtime_fault(cpu, format!("{operation} invalid width {width}"));
        return None;
    }

    match cpu.ram.read_nbytes_cached(addr, width) {
        Ok(value) => Some(value),
        Err(error) => {
            record_runtime_fault(
                cpu,
                format!("{operation} failed at 0x{addr:016x} width={width}: {error}"),
            );
            None
        }
    }
}

fn write_runtime_memory(
    cpu: &mut RV64GC,
    addr: u64,
    value: u64,
    width: u64,
    operation: &str,
) -> Option<()> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        record_runtime_fault(cpu, format!("{operation} invalid width {width}"));
        return None;
    }

    match cpu.ram.write_nbytes_cached(addr, value, width) {
        Ok(()) => Some(()),
        Err(error) => {
            record_runtime_fault(
                cpu,
                format!("{operation} failed at 0x{addr:016x} width={width}: {error}"),
            );
            None
        }
    }
}

fn record_runtime_fault(cpu: &mut RV64GC, reason: String) {
    cpu.set_jit_runtime_fault(reason);
}

fn atomic_word(cpu: &mut RV64GC, addr: u64, update: impl FnOnce(u32) -> (u32, u32)) -> Option<u64> {
    let old = read_runtime_memory(cpu, addr, 4, "atomic read-modify-write word")? as u32;
    let (result, new) = update(old);
    write_runtime_memory(
        cpu,
        addr,
        u64::from(new),
        4,
        "atomic read-modify-write word",
    )?;
    Some(sign_extend(u64::from(result), 32) as u64)
}

fn atomic_doubleword(cpu: &mut RV64GC, addr: u64, update: impl FnOnce(u64) -> u64) -> Option<u64> {
    let old = read_runtime_memory(cpu, addr, 8, "atomic read-modify-write doubleword")?;
    write_runtime_memory(
        cpu,
        addr,
        update(old),
        8,
        "atomic read-modify-write doubleword",
    )?;
    Some(old)
}

fn runtime_binary_op(raw: u64) -> Option<RuntimeBinaryOp> {
    match raw {
        x if x == RuntimeBinaryOp::Addw as u64 => Some(RuntimeBinaryOp::Addw),
        x if x == RuntimeBinaryOp::Div as u64 => Some(RuntimeBinaryOp::Div),
        x if x == RuntimeBinaryOp::Divu as u64 => Some(RuntimeBinaryOp::Divu),
        x if x == RuntimeBinaryOp::Divuw as u64 => Some(RuntimeBinaryOp::Divuw),
        x if x == RuntimeBinaryOp::Divw as u64 => Some(RuntimeBinaryOp::Divw),
        x if x == RuntimeBinaryOp::Mul as u64 => Some(RuntimeBinaryOp::Mul),
        x if x == RuntimeBinaryOp::Mulh as u64 => Some(RuntimeBinaryOp::Mulh),
        x if x == RuntimeBinaryOp::Mulhsu as u64 => Some(RuntimeBinaryOp::Mulhsu),
        x if x == RuntimeBinaryOp::Mulhu as u64 => Some(RuntimeBinaryOp::Mulhu),
        x if x == RuntimeBinaryOp::Mulw as u64 => Some(RuntimeBinaryOp::Mulw),
        x if x == RuntimeBinaryOp::Rem as u64 => Some(RuntimeBinaryOp::Rem),
        x if x == RuntimeBinaryOp::Remu as u64 => Some(RuntimeBinaryOp::Remu),
        x if x == RuntimeBinaryOp::Remuw as u64 => Some(RuntimeBinaryOp::Remuw),
        x if x == RuntimeBinaryOp::Remw as u64 => Some(RuntimeBinaryOp::Remw),
        x if x == RuntimeBinaryOp::Sllw as u64 => Some(RuntimeBinaryOp::Sllw),
        x if x == RuntimeBinaryOp::Slt as u64 => Some(RuntimeBinaryOp::Slt),
        x if x == RuntimeBinaryOp::Sltu as u64 => Some(RuntimeBinaryOp::Sltu),
        x if x == RuntimeBinaryOp::Sraw as u64 => Some(RuntimeBinaryOp::Sraw),
        x if x == RuntimeBinaryOp::Srlw as u64 => Some(RuntimeBinaryOp::Srlw),
        x if x == RuntimeBinaryOp::Subw as u64 => Some(RuntimeBinaryOp::Subw),
        _ => None,
    }
}

fn runtime_float_op(raw: u64) -> Option<RuntimeFloatOp> {
    match raw {
        x if x == RuntimeFloatOp::FmaddS as u64 => Some(RuntimeFloatOp::FmaddS),
        x if x == RuntimeFloatOp::FmsubS as u64 => Some(RuntimeFloatOp::FmsubS),
        x if x == RuntimeFloatOp::FnmaddS as u64 => Some(RuntimeFloatOp::FnmaddS),
        x if x == RuntimeFloatOp::FnmsubS as u64 => Some(RuntimeFloatOp::FnmsubS),
        x if x == RuntimeFloatOp::AddS as u64 => Some(RuntimeFloatOp::AddS),
        x if x == RuntimeFloatOp::SubS as u64 => Some(RuntimeFloatOp::SubS),
        x if x == RuntimeFloatOp::MulS as u64 => Some(RuntimeFloatOp::MulS),
        x if x == RuntimeFloatOp::DivS as u64 => Some(RuntimeFloatOp::DivS),
        x if x == RuntimeFloatOp::SqrtS as u64 => Some(RuntimeFloatOp::SqrtS),
        x if x == RuntimeFloatOp::SgnjS as u64 => Some(RuntimeFloatOp::SgnjS),
        x if x == RuntimeFloatOp::SgnjnS as u64 => Some(RuntimeFloatOp::SgnjnS),
        x if x == RuntimeFloatOp::SgnjxS as u64 => Some(RuntimeFloatOp::SgnjxS),
        x if x == RuntimeFloatOp::MinS as u64 => Some(RuntimeFloatOp::MinS),
        x if x == RuntimeFloatOp::MaxS as u64 => Some(RuntimeFloatOp::MaxS),
        x if x == RuntimeFloatOp::CvtWS as u64 => Some(RuntimeFloatOp::CvtWS),
        x if x == RuntimeFloatOp::CvtWuS as u64 => Some(RuntimeFloatOp::CvtWuS),
        x if x == RuntimeFloatOp::CvtLS as u64 => Some(RuntimeFloatOp::CvtLS),
        x if x == RuntimeFloatOp::CvtLuS as u64 => Some(RuntimeFloatOp::CvtLuS),
        x if x == RuntimeFloatOp::MvXW as u64 => Some(RuntimeFloatOp::MvXW),
        x if x == RuntimeFloatOp::EqS as u64 => Some(RuntimeFloatOp::EqS),
        x if x == RuntimeFloatOp::LtS as u64 => Some(RuntimeFloatOp::LtS),
        x if x == RuntimeFloatOp::LeS as u64 => Some(RuntimeFloatOp::LeS),
        x if x == RuntimeFloatOp::ClassS as u64 => Some(RuntimeFloatOp::ClassS),
        x if x == RuntimeFloatOp::CvtSW as u64 => Some(RuntimeFloatOp::CvtSW),
        x if x == RuntimeFloatOp::CvtSWu as u64 => Some(RuntimeFloatOp::CvtSWu),
        x if x == RuntimeFloatOp::CvtSL as u64 => Some(RuntimeFloatOp::CvtSL),
        x if x == RuntimeFloatOp::CvtSLu as u64 => Some(RuntimeFloatOp::CvtSLu),
        x if x == RuntimeFloatOp::MvWX as u64 => Some(RuntimeFloatOp::MvWX),
        x if x == RuntimeFloatOp::FmaddD as u64 => Some(RuntimeFloatOp::FmaddD),
        x if x == RuntimeFloatOp::FmsubD as u64 => Some(RuntimeFloatOp::FmsubD),
        x if x == RuntimeFloatOp::FnmaddD as u64 => Some(RuntimeFloatOp::FnmaddD),
        x if x == RuntimeFloatOp::FnmsubD as u64 => Some(RuntimeFloatOp::FnmsubD),
        x if x == RuntimeFloatOp::AddD as u64 => Some(RuntimeFloatOp::AddD),
        x if x == RuntimeFloatOp::SubD as u64 => Some(RuntimeFloatOp::SubD),
        x if x == RuntimeFloatOp::MulD as u64 => Some(RuntimeFloatOp::MulD),
        x if x == RuntimeFloatOp::DivD as u64 => Some(RuntimeFloatOp::DivD),
        x if x == RuntimeFloatOp::SqrtD as u64 => Some(RuntimeFloatOp::SqrtD),
        x if x == RuntimeFloatOp::SgnjD as u64 => Some(RuntimeFloatOp::SgnjD),
        x if x == RuntimeFloatOp::SgnjnD as u64 => Some(RuntimeFloatOp::SgnjnD),
        x if x == RuntimeFloatOp::SgnjxD as u64 => Some(RuntimeFloatOp::SgnjxD),
        x if x == RuntimeFloatOp::MinD as u64 => Some(RuntimeFloatOp::MinD),
        x if x == RuntimeFloatOp::MaxD as u64 => Some(RuntimeFloatOp::MaxD),
        x if x == RuntimeFloatOp::EqD as u64 => Some(RuntimeFloatOp::EqD),
        x if x == RuntimeFloatOp::LtD as u64 => Some(RuntimeFloatOp::LtD),
        x if x == RuntimeFloatOp::LeD as u64 => Some(RuntimeFloatOp::LeD),
        x if x == RuntimeFloatOp::ClassD as u64 => Some(RuntimeFloatOp::ClassD),
        x if x == RuntimeFloatOp::CvtSD as u64 => Some(RuntimeFloatOp::CvtSD),
        x if x == RuntimeFloatOp::CvtDS as u64 => Some(RuntimeFloatOp::CvtDS),
        x if x == RuntimeFloatOp::CvtWD as u64 => Some(RuntimeFloatOp::CvtWD),
        x if x == RuntimeFloatOp::CvtWuD as u64 => Some(RuntimeFloatOp::CvtWuD),
        x if x == RuntimeFloatOp::CvtDW as u64 => Some(RuntimeFloatOp::CvtDW),
        x if x == RuntimeFloatOp::CvtDWu as u64 => Some(RuntimeFloatOp::CvtDWu),
        x if x == RuntimeFloatOp::MvXD as u64 => Some(RuntimeFloatOp::MvXD),
        _ => None,
    }
}

fn runtime_rounding_mode(cpu: &RV64GC, raw: u64) -> Option<RoundingMode> {
    match raw {
        0b000 => Some(RoundingMode::Rne),
        0b001 => Some(RoundingMode::Rtz),
        0b010 => Some(RoundingMode::Rdn),
        0b011 => Some(RoundingMode::Rup),
        0b100 => Some(RoundingMode::Rmm),
        0b111 => Some(cpu.fcsr.frm),
        _ => None,
    }
}

fn runtime_csr_op(raw: u64) -> Option<RuntimeCsrOp> {
    match raw {
        x if x == RuntimeCsrOp::Csrrw as u64 => Some(RuntimeCsrOp::Csrrw),
        x if x == RuntimeCsrOp::Csrrs as u64 => Some(RuntimeCsrOp::Csrrs),
        x if x == RuntimeCsrOp::Csrrc as u64 => Some(RuntimeCsrOp::Csrrc),
        x if x == RuntimeCsrOp::Csrrwi as u64 => Some(RuntimeCsrOp::Csrrwi),
        x if x == RuntimeCsrOp::Csrrsi as u64 => Some(RuntimeCsrOp::Csrrsi),
        x if x == RuntimeCsrOp::Csrrci as u64 => Some(RuntimeCsrOp::Csrrci),
        _ => None,
    }
}

fn runtime_trap_op(raw: u64) -> Option<RuntimeTrapOp> {
    match raw {
        x if x == RuntimeTrapOp::Ebreak as u64 => Some(RuntimeTrapOp::Ebreak),
        x if x == RuntimeTrapOp::Cebreak as u64 => Some(RuntimeTrapOp::Cebreak),
        x if x == RuntimeTrapOp::Uret as u64 => Some(RuntimeTrapOp::Uret),
        x if x == RuntimeTrapOp::Sret as u64 => Some(RuntimeTrapOp::Sret),
        x if x == RuntimeTrapOp::Mret as u64 => Some(RuntimeTrapOp::Mret),
        x if x == RuntimeTrapOp::Wfi as u64 => Some(RuntimeTrapOp::Wfi),
        x if x == RuntimeTrapOp::SfenceVma as u64 => Some(RuntimeTrapOp::SfenceVma),
        x if x == RuntimeTrapOp::IllegalInstruction as u64 => {
            Some(RuntimeTrapOp::IllegalInstruction)
        }
        _ => None,
    }
}

fn runtime_atomic_op(raw: u64) -> Option<RuntimeAtomicOp> {
    match raw {
        x if x == RuntimeAtomicOp::Amoaddd as u64 => Some(RuntimeAtomicOp::Amoaddd),
        x if x == RuntimeAtomicOp::Amoaddw as u64 => Some(RuntimeAtomicOp::Amoaddw),
        x if x == RuntimeAtomicOp::Amoandd as u64 => Some(RuntimeAtomicOp::Amoandd),
        x if x == RuntimeAtomicOp::Amoandw as u64 => Some(RuntimeAtomicOp::Amoandw),
        x if x == RuntimeAtomicOp::Amomaxd as u64 => Some(RuntimeAtomicOp::Amomaxd),
        x if x == RuntimeAtomicOp::Amomaxud as u64 => Some(RuntimeAtomicOp::Amomaxud),
        x if x == RuntimeAtomicOp::Amomaxuw as u64 => Some(RuntimeAtomicOp::Amomaxuw),
        x if x == RuntimeAtomicOp::Amomaxw as u64 => Some(RuntimeAtomicOp::Amomaxw),
        x if x == RuntimeAtomicOp::Amomind as u64 => Some(RuntimeAtomicOp::Amomind),
        x if x == RuntimeAtomicOp::Amominud as u64 => Some(RuntimeAtomicOp::Amominud),
        x if x == RuntimeAtomicOp::Amominuw as u64 => Some(RuntimeAtomicOp::Amominuw),
        x if x == RuntimeAtomicOp::Amominw as u64 => Some(RuntimeAtomicOp::Amominw),
        x if x == RuntimeAtomicOp::Amoord as u64 => Some(RuntimeAtomicOp::Amoord),
        x if x == RuntimeAtomicOp::Amoorw as u64 => Some(RuntimeAtomicOp::Amoorw),
        x if x == RuntimeAtomicOp::Amoswapd as u64 => Some(RuntimeAtomicOp::Amoswapd),
        x if x == RuntimeAtomicOp::Amoswapw as u64 => Some(RuntimeAtomicOp::Amoswapw),
        x if x == RuntimeAtomicOp::Amoxord as u64 => Some(RuntimeAtomicOp::Amoxord),
        x if x == RuntimeAtomicOp::Amoxorw as u64 => Some(RuntimeAtomicOp::Amoxorw),
        x if x == RuntimeAtomicOp::Lrd as u64 => Some(RuntimeAtomicOp::Lrd),
        x if x == RuntimeAtomicOp::Lrw as u64 => Some(RuntimeAtomicOp::Lrw),
        x if x == RuntimeAtomicOp::Scd as u64 => Some(RuntimeAtomicOp::Scd),
        x if x == RuntimeAtomicOp::Scw as u64 => Some(RuntimeAtomicOp::Scw),
        _ => None,
    }
}
