use bit::BitIndex;
use tracing::debug;
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing::span;
use tracing::trace;
use tracing::Level;

use crate::opcodes::*;
use crate::ram::MemoryRegion;
use crate::ram::Ram;
use crate::sign_extend;
use crate::sign_extend12;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

use crate::cpu::RV64GCRegAbiName::*;

type Reg = u8;
type Imm = u32;
type Csr = u16;

#[derive(Debug)]
pub struct RV64GC {
    pub registers: RV64GCRegisters,
    pub ram: Ram,
    pub should_quit: bool,
}

impl Default for RV64GC {
    fn default() -> Self {
        Self::new()
    }
}

impl RV64GC {
    pub fn new() -> RV64GC {
        let mut registers = RV64GCRegisters::new();
        let mut ram = Ram::new();
        let stack_size: u64 = 8 * 1024 * 1024; // 8 MB
        let stack_top: u64 = 0x7FFF_FFFF_FFFF_FFFF; // Example top address
        let stack_start = stack_top - stack_size;

        let stack_region = MemoryRegion::new(stack_start, stack_size, vec![0; stack_size as usize]);
        ram.add_region(stack_region).unwrap();

        registers[Sp] = stack_top;

        RV64GC {
            registers,
            ram,
            should_quit: false,
        }
    }

    pub fn load_bin(&mut self, bin: Vec<u8>) {
        let bin_load = MemoryRegion::new(0, bin.len() as u64, bin);
        self.ram.add_region(bin_load).unwrap();
        self.registers[Pc] = 0;
    }

    pub fn load_elf(&mut self, bin: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let span = span!(Level::TRACE, "load_elf");
        let _guard = span.enter();

        let elf = goblin::elf::Elf::parse(&bin)?;
        let entry = elf.entry;
        self.registers[Pc] = entry;
        if elf.header.e_machine != goblin::elf::header::EM_RISCV {
            return Err("Not a RISC-V ELF".into());
        }

        for ph in &elf.program_headers {
            trace!("Reading ph of type: {:#08x}", ph.p_type);
            if ph.p_type == goblin::elf::program_header::PT_LOAD {
                let base_addr = ph.p_vaddr;
                let size = ph.p_memsz;

                let data = &bin[ph.file_range()];

                let memory_region = MemoryRegion::new(base_addr, size, data.to_vec());
                trace!(
                    "adding region, start: {}\t len: {}\toffset: {}",
                    base_addr,
                    size,
                    ph.p_offset
                );
                self.ram.add_region(memory_region)?;
            }
        }

        trace!("mem regions: {}", self.ram);

        Ok(())
    }

    // NOTE: Takes mutable reference, to pass down the call stack
    pub fn start(&mut self) {
        let span = span!(Level::TRACE, "cpu loop");
        let _guard = span.enter();
        // while self.registers[Pc] <= (self.program.len() - 4) as u64 {
        //     self.step();
        // }

        while !self.should_quit {
            self.step();
        }
    }

    pub fn step(&mut self) {
        let span = span!(Level::TRACE, "step");
        let _guard = span.enter();

        trace!("pc: {}", self.registers[Pc]);
        self.execute();
        assert_eq!(self.registers[0], 0);
        self.registers[Pc] += 4;
    }

    pub fn execute(&mut self) {
        let ins = self.find_instruction();
        ins.execute_instruction(self);
    }

    pub fn find_instruction(&self) -> RV64GCInstruction {
        let pc = self.registers[Pc];
        let current_ins = self.ram.read_word(pc).unwrap();

        // Default values
        let rd = current_ins.bit_range(7..12) as Reg;
        let rs1 = current_ins.bit_range(15..20) as Reg;
        let rs2 = current_ins.bit_range(20..25) as Reg;
        let imm = current_ins.bit_range(20..32) as Imm;

        match current_ins {
            i if is_rv64i_add_instruction(i) => RV64GCInstruction::Add(rd, rs1, rs2),

            i if is_rv64i_addi_instruction(i) => RV64GCInstruction::Addi(rd, rs1, imm),

            i if is_rv64i_auipc_instruction(i) => {
                let ov_imm = i.bit_range(12..32);
                RV64GCInstruction::Auipc(rd, ov_imm)
            }

            i if is_rv64i_lui_instruction(i) => {
                let ov_imm = i.bit_range(12..32);
                RV64GCInstruction::Lui(rd, ov_imm)
            }

            i if is_rv64i_slti_instruction(i) => RV64GCInstruction::Slti(rd, rs1, imm),

            i if is_rv64i_sltiu_instruction(i) => RV64GCInstruction::Sltiu(rd, rs1, imm),

            i if is_rv64i_xori_instruction(i) => RV64GCInstruction::Xori(rd, rs1, imm),

            i if is_rv64i_ori_instruction(i) => RV64GCInstruction::Ori(rd, rs1, imm),

            i if is_rv64i_andi_instruction(i) => RV64GCInstruction::Andi(rd, rs1, imm),

            i if is_rv64i_slli_instruction(i) => {
                let shamt = i.bit_range(20..26);
                RV64GCInstruction::Slli(rd, rs1, shamt)
            }

            i if is_rv64i_srli_instruction(i) => {
                let shamt = i.bit_range(20..26);
                RV64GCInstruction::Srli(rd, rs1, shamt)
            }

            i if is_rv64i_srai_instruction(i) => {
                let shamt = i.bit_range(20..26);
                RV64GCInstruction::Srai(rd, rs1, shamt)
            }

            i if is_rv64i_sub_instruction(i) => RV64GCInstruction::Sub(rd, rs1, rs2),

            i if is_rv64i_xor_instruction(i) => RV64GCInstruction::Xor(rd, rs1, rs2),

            i if is_rv64i_ecall_instruction(i) => RV64GCInstruction::Ecall,

            i if is_rv64i_lb_instruction(i) => RV64GCInstruction::Lb(rd, rs1, imm),

            i if is_rv64i_sb_instruction(i) => {
                let lo_offset = i.bit_range(7..12);
                let hi_offset = i.bit_range(25..32);

                let offset = (hi_offset << 5) | lo_offset;

                RV64GCInstruction::Sb(rs1, rs2, offset)
            }

            i if is_rv64i_lw_instruction(i) => RV64GCInstruction::Lw(rd, rs1, imm),

            i if is_rv64i_sw_instruction(i) => {
                let lo_offset = i.bit_range(7..12);
                let hi_offset = i.bit_range(25..32);

                let offset = (hi_offset << 5) | lo_offset;

                RV64GCInstruction::Sw(rs1, rs2, offset)
            }

            i if is_rv64i_ld_instruction(i) => RV64GCInstruction::Ld(rd, rs1, imm),

            i if is_rv64i_sd_instruction(i) => {
                let lo_offset = i.bit_range(7..12);
                let hi_offset = i.bit_range(25..32);

                let offset = (hi_offset << 5) | lo_offset;

                RV64GCInstruction::Sd(rs1, rs2, offset)
            }

            i if is_rv64i_jal_instruction(i) => {
                let offset = (i.bit(31) as u32) << 20
                    | i.bit_range(12..20) << 12
                    | (i.bit(20) as u32) << 11
                    | i.bit_range(21..31) << 1;

                trace!("offset: {offset:#b}");
                trace!("offset: {}", crate::sign_extend(offset as u64, 20));

                RV64GCInstruction::Jal(rd, offset)
            }

            i if is_rv64i_jalr_instruction(i) => RV64GCInstruction::Jalr(rd, rs1, imm),

            i if is_rv64i_bge_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                RV64GCInstruction::Bge(rs1, rs2, offset)
            }

            i if is_rv64i_beq_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                RV64GCInstruction::Beq(rs1, rs2, offset)
            }

            i if is_rv64i_bne_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                RV64GCInstruction::Bne(rs1, rs2, offset)
            }

            i if is_rv64i_blt_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                RV64GCInstruction::Blt(rs1, rs2, offset)
            }

            i if is_rv64i_addiw_instruction(i) => RV64GCInstruction::Addiw(rd, rs1, imm),

            i if is_rv64i_sraiw_instruction(i) => {
                let shamt = i.bit_range(20..25);

                RV64GCInstruction::Sraiw(rd, rs1, shamt)
            }

            i if is_rv64i_addw_instruction(i) => RV64GCInstruction::Addw(rd, rs1, rs2),

            i if is_rv64i_subw_instruction(i) => RV64GCInstruction::Subw(rd, rs1, rs2),

            _ => {
                panic!("instruction not implemented, instruction: {current_ins:08x}",);
            }
        }
    }

    pub fn syscall_handler(&mut self) {
        let span = span!(Level::TRACE, "syscall_handler");
        let _guard = span.enter();

        let syscall_id = self.registers[A7];
        trace!("system call: {syscall_id}");

        match syscall_id {
            64 => {
                trace!("printing...");

                let ptr = self.registers[A1];
                let len = self.registers[A2];

                trace!("ptr: {ptr:#08x}");
                trace!("len: {len}");

                let msg = (ptr..ptr + len)
                    .map(|addr| {
                        self.ram
                            .read_byte(addr)
                            .inspect_err(|e| panic!("{e}"))
                            .unwrap() as char
                    })
                    .collect::<String>();

                print!("{msg}");
            }

            93 => {
                let error_code = self.registers[A0];
                info!("Program exited with code: {error_code}");
                self.should_quit = true;
            }
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RV64GCInstruction {
    Add(Reg, Reg, Reg),
    Addi(Reg, Reg, Imm),
    Auipc(Reg, Imm),
    Lui(Reg, Imm),
    Slti(Reg, Reg, Imm),
    Sltiu(Reg, Reg, Imm),
    Xori(Reg, Reg, Imm),
    Ori(Reg, Reg, Imm),
    Andi(Reg, Reg, Imm),
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
    Csrrwi(Reg, Reg, Csr),
    Csrrsi(Reg, Imm, Csr),
    Csrrci(Reg, Reg, Csr),
    Ecall,
    Ebreak,
    Uret,
    Sret,
    Mret,
    Wfi,
    SfenceVma(Reg, Reg, Reg),
    Lb(Reg, Reg, Imm),
    Lh(Reg, Reg, Imm),
    Lw(Reg, Reg, Imm),
    Lbu(Reg, Reg, Imm),
    Lhu(Reg, Reg, Imm),
    Sb(Reg, Reg, Imm),
    Sh(Reg, Reg, Imm),
    Sw(Reg, Reg, Imm),
    Jal(Reg, Imm),
    Jalr(Reg, Reg, Imm),
    Beq(Reg, Reg, Imm),
    Bne(Reg, Reg, Imm),
    Blt(Reg, Reg, Imm),
    Bge(Reg, Reg, Imm),
    Bltu(Reg, Reg, Imm),
    Bgeu(Reg, Reg, Imm),
    Ld(Reg, Reg, Imm),
    Sd(Reg, Reg, Imm),
    Addiw(Reg, Reg, Imm),
    Sraiw(Reg, Reg, Imm),
    Addw(Reg, Reg, Reg),
    Subw(Reg, Reg, Reg),
}

impl RV64GCInstruction {
    pub fn execute_instruction(&self, cpu: &mut RV64GC) {
        let span = span!(Level::TRACE, "execute_instruction");
        let _guard = span.enter();
        trace!("{}", self);

        match self {
            RV64GCInstruction::Add(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_add(cpu.registers[rs2]);
            }

            RV64GCInstruction::Addi(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                cpu.registers[rd] = (cpu.registers[rs1] as i64).wrapping_add(simm) as u64;
            }

            RV64GCInstruction::Auipc(rd, imm) => {
                cpu.registers[rd] =
                    (cpu.registers[Pc] as i64).wrapping_add(sign_extend12(*imm << 12)) as u64;
            }

            RV64GCInstruction::Lui(rd, imm) => {
                cpu.registers[rd] = sign_extend12(*imm << 12) as u64;
            }

            RV64GCInstruction::Slti(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let rs1 = cpu.registers[rs1] as i64;

                if rs1 < simm {
                    cpu.registers[rd] = 1;
                } else {
                    cpu.registers[rd] = 0;
                }
            }

            RV64GCInstruction::Sltiu(rd, rs1, imm) => {
                let simm = sign_extend12(*imm) as u64;

                if cpu.registers[rs1] < simm {
                    cpu.registers[rd] = 1;
                } else {
                    cpu.registers[rd] = 0;
                }
            }

            RV64GCInstruction::Xori(rd, rs1, imm) => {
                let simm = sign_extend12(*imm) as u64;

                cpu.registers[rd] = cpu.registers[rs1] ^ simm;
            }

            RV64GCInstruction::Ori(rd, rs1, imm) => {
                let simm = sign_extend12(*imm) as u64;

                cpu.registers[rd] = cpu.registers[rs1] | simm;
            }

            RV64GCInstruction::Andi(rd, rs1, imm) => {
                let simm = sign_extend12(*imm) as u64;

                cpu.registers[rd] = cpu.registers[rs1] & simm;
            }

            RV64GCInstruction::Slli(rd, rs1, imm) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_shl(*imm);
            }

            RV64GCInstruction::Srli(rd, rs1, imm) => {
                cpu.registers[rd] = cpu.registers[rs1].wrapping_shr(*imm);
            }

            RV64GCInstruction::Srai(rd, rs1, imm) => {
                cpu.registers[rd] =
                    ((cpu.registers[rs1] as i64).wrapping_shr(sign_extend12(*imm) as u32)) as u64;
            }

            RV64GCInstruction::Sub(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] - cpu.registers[rs2];
            }

            RV64GCInstruction::Sll(rd, rs1, rs2) => {
                cpu.registers[rd] =
                    cpu.registers[rs1].wrapping_shl(cpu.registers[rs2] as u32 & ((1 << 5) - 1));
            }

            RV64GCInstruction::Slt(rd, rs1, rs2) => {
                if (cpu.registers[rs1] as i64) < (cpu.registers[rs2] as i64) {
                    cpu.registers[rd] = 1
                } else {
                    cpu.registers[rd] = 0
                }
            }

            RV64GCInstruction::Sltu(rd, rs1, rs2) => {
                if cpu.registers[rs1] < cpu.registers[rs2] {
                    cpu.registers[rd] = 1
                } else {
                    cpu.registers[rd] = 0
                }
            }

            RV64GCInstruction::Xor(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] ^ cpu.registers[rs2];
            }

            RV64GCInstruction::Srl(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] >> (cpu.registers[rs2] & ((1 << 5) - 1));
            }

            RV64GCInstruction::Sra(rd, rs1, rs2) => {
                cpu.registers[rd] =
                    ((cpu.registers[rs1] as i64) >> (cpu.registers[rs2] & ((1 << 5) - 1))) as u64;
            }

            RV64GCInstruction::Or(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] | cpu.registers[rs2];
            }

            RV64GCInstruction::And(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] & cpu.registers[rs2];
            }

            RV64GCInstruction::Fence(_, _) => {
                todo!()
            }

            RV64GCInstruction::FenceI => todo!(),

            RV64GCInstruction::Uret => todo!(),

            RV64GCInstruction::Sret => todo!(),

            RV64GCInstruction::Mret => todo!(),

            RV64GCInstruction::Wfi => todo!(),

            RV64GCInstruction::SfenceVma(_, _, _) => todo!(),

            RV64GCInstruction::Csrrw(_, _, _) => todo!(),
            RV64GCInstruction::Csrrs(_, _, _) => todo!(),
            RV64GCInstruction::Csrrc(_, _, _) => todo!(),
            RV64GCInstruction::Csrrwi(_, _, _) => todo!(),
            RV64GCInstruction::Csrrsi(_, _, _) => todo!(),
            RV64GCInstruction::Csrrci(_, _, _) => todo!(),

            RV64GCInstruction::Ecall => {
                cpu.syscall_handler();
            }

            RV64GCInstruction::Ebreak => {
                todo!()
            }

            RV64GCInstruction::Lb(rd, rs1, imm) => {
                cpu.registers[rd] = sign_extend12(
                    cpu.ram
                        .read_byte((cpu.registers[rs1] as i64 + sign_extend12(*imm)) as u64)
                        .unwrap()
                        .into(),
                ) as u64;
            }

            RV64GCInstruction::Lbu(rd, rs1, offset) => {
                cpu.registers[rd] = cpu
                    .ram
                    .read_byte(cpu.registers[rs1] + *offset as u64)
                    .unwrap()
                    .into();
            }

            RV64GCInstruction::Lhu(rd, rs1, offset) => {
                cpu.registers[rd] = cpu
                    .ram
                    .read_doubleword(cpu.registers[rs1] + *offset as u64)
                    .unwrap()
            }

            RV64GCInstruction::Sb(rs1, rs2, imm) => {
                let simm = sign_extend12(*imm);
                let addr = (cpu.registers[rs1] as i64).wrapping_add(simm);

                cpu.ram
                    .write_byte(addr as u64, cpu.registers[rs2] as u8)
                    .unwrap();
            }

            RV64GCInstruction::Sh(rs1, rs2, imm) => {
                let simm = sign_extend12(*imm);
                let addr = (cpu.registers[rs1] as i64).wrapping_add(simm);
                let value = cpu.registers[rs2] as u16;

                cpu.ram.write_halfword(addr as u64, value as u64).unwrap();
            }

            RV64GCInstruction::Lh(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let addr = (cpu.registers[rs1] as i64).wrapping_add(simm);

                cpu.registers[rd] = cpu.ram.read_halfword(addr as u64).unwrap();
            }

            RV64GCInstruction::Lw(rd, rs1, imm) => {
                cpu.registers[rd] = sign_extend(
                    cpu.ram
                        .read_word((cpu.registers[rs1] as i64 + sign_extend12(*imm)) as u64)
                        .inspect_err(|e| panic!("{e}"))
                        .unwrap()
                        .into(),
                    32,
                ) as u64;
            }

            RV64GCInstruction::Sw(rs1, rs2, imm) => {
                let simm = sign_extend12(*imm);
                let addr = cpu.registers[rs1] as i64 + simm;

                cpu.ram
                    .write_word(addr as u64, cpu.registers[rs2] as u32)
                    .inspect_err(|e| panic!("{e}"))
                    .unwrap();
            }

            RV64GCInstruction::Ld(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                cpu.registers[rd] = cpu
                    .ram
                    .read_doubleword((cpu.registers[rs1] as i64 + simm) as u64)
                    .inspect_err(|e| panic!("{e}"))
                    .unwrap();
            }

            RV64GCInstruction::Sd(rs1, rs2, imm) => {
                let simm = sign_extend12(*imm);
                let addr = (cpu.registers[rs1] as i64).wrapping_add(simm);
                cpu.ram
                    .write_doubleword(addr as u64, cpu.registers[rs2])
                    .inspect_err(|e| panic!("{e}\nAddress: {:08x}", cpu.registers[rs1]))
                    .unwrap();
            }

            RV64GCInstruction::Jal(rd, imm) => {
                let span = span!(Level::TRACE, "jal");
                let _guard = span.enter();

                let simm = crate::sign_extend(*imm as u64, 20);

                if *rd > 0 {
                    cpu.registers[rd] = cpu.registers[Pc] + 4;
                }
                cpu.registers[Pc] = (cpu.registers[Pc] as i64 + simm) as u64;

                // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                // behaviour
                cpu.registers[Pc] -= 4;
            }

            RV64GCInstruction::Jalr(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let jump_addr = (cpu.registers[rs1] as i64).wrapping_add(simm);

                if *rd > 0 {
                    cpu.registers[rd] = cpu.registers[Pc] + 4;
                }
                cpu.registers[Pc] = (jump_addr as u64) & !1;

                // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                // behaviour
                cpu.registers[Pc] -= 4;
            }

            RV64GCInstruction::Beq(rs1, rs2, imm) => {
                if cpu.registers[rs1] == cpu.registers[rs2] {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            RV64GCInstruction::Bne(rs1, rs2, imm) => {
                if cpu.registers[rs1] != cpu.registers[rs2] {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            RV64GCInstruction::Blt(rs1, rs2, imm) => {
                if (cpu.registers[rs1] as i64) < (cpu.registers[rs2] as i64) {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            RV64GCInstruction::Bge(rs1, rs2, imm) => {
                if (cpu.registers[rs1] as i64) >= (cpu.registers[rs2] as i64) {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            RV64GCInstruction::Bltu(rs1, rs2, imm) => {
                if cpu.registers[rs1] < cpu.registers[rs2] {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            RV64GCInstruction::Bgeu(rs1, rs2, imm) => {
                if cpu.registers[rs1] >= cpu.registers[rs2] {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            RV64GCInstruction::Addiw(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let value = (cpu.registers[rs1] as i64).wrapping_add(simm) & (u32::MAX as i64);

                cpu.registers[rd] = sign_extend(value as u64, 32) as u64;
            }

            RV64GCInstruction::Sraiw(rd, rs1, shamt) => {
                let shifted_reg = cpu.registers[rs1] & (u32::MAX as u64);

                cpu.registers[rd] = sign_extend(shifted_reg >> shamt, 32) as u64;
            }

            RV64GCInstruction::Addw(rd, rs1, rs2) => {
                let rs1_low = cpu.registers[rs1] & (u32::MAX as u64);
                let rs2_low = cpu.registers[rs2] & (u32::MAX as u64);

                cpu.registers[rd] = sign_extend(rs1_low.wrapping_add(rs2_low), 32) as u64;
            }

            RV64GCInstruction::Subw(rd, rs1, rs2) => {
                let rs1_low = cpu.registers[rs1] & (u32::MAX as u64);
                let rs2_low = cpu.registers[rs2] & (u32::MAX as u64);

                cpu.registers[rd] = sign_extend(rs1_low.wrapping_sub(rs2_low), 32) as u64;
            }
        }
    }
}

impl Display for RV64GCInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RV64GCInstruction::Addi(rd, rs1, imm) => {
                write!(f, "addi x{rd}, x{rs1}, {}", *imm as i64)
            }

            RV64GCInstruction::Auipc(rd, imm) => {
                write!(f, "auipc x{rd}, {imm}")
            }

            RV64GCInstruction::Xori(rd, rs1, imm) => {
                write!(f, "xori x{rd}, x{rs1}, {imm}")
            }

            RV64GCInstruction::Srai(rd, rs1, imm) => {
                write!(f, "srai x{rd}, x{rs1}, {imm}")
            }

            RV64GCInstruction::Add(rd, rs1, rs2) => {
                write!(f, "add x{rd}, x{rs1}, x{rs2}")
            }

            RV64GCInstruction::Sub(rd, rs1, rs2) => {
                write!(f, "sub x{rd}, x{rs1}, x{rs2}")
            }

            RV64GCInstruction::Xor(rd, rs1, rs2) => {
                write!(f, "xor x{rd}, x{rs1}, x{rs2}")
            }

            RV64GCInstruction::Ecall => {
                write!(f, "ecall")
            }

            RV64GCInstruction::Sd(rs1, rs2, offset) => {
                write!(f, "sd x{rs2}, {offset}(x{rs1})")
            }

            RV64GCInstruction::Jal(rd, imm) => {
                let simm = crate::sign_extend(*imm as u64, 20);
                write!(f, "jal x{rd}, {simm}")
            }

            RV64GCInstruction::Bne(rs1, rs2, imm) => {
                let simm = crate::sign_extend(*imm as u64, 13);
                write!(f, "bne x{rs1}, x{rs2}, {simm}")
            }

            RV64GCInstruction::Bge(rs1, rs2, imm) => {
                let simm = crate::sign_extend(*imm as u64, 13);
                write!(f, "bge x{rs1}, x{rs2}, {simm}")
            }

            e => write!(f, "{e:?}"),
        }
    }
}

#[derive(Debug)]
pub struct RV64GCRegisters {
    registers: [u64; 33],
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
        };

        write!(f, "{reg}")
    }
}
