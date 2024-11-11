use bit::BitIndex;
use goblin::elf::Elf;
use rand::RngCore;
use tracing::info;
use tracing::span;
use tracing::trace;
use tracing::Level;

use crate::fcsr::classify_f32;
use crate::fcsr::round_f32;
use crate::fcsr::RoundingMode;
use crate::fcsr::FCSR;
use crate::opcodes::*;
use crate::ram::MemoryRegion;
use crate::ram::Ram;
use crate::sign_extend;
use crate::sign_extend12;
use crate::syscalls::*;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

use crate::cpu::RV64GCRegAbiName::*;

type Reg = u8;
type Imm = u32;
type Simm = i64;
type Csr = u16;

const AT_NULL: u64 = 0; // End of auxv
#[allow(dead_code)]
const AT_IGNORE: u64 = 1; // Ignore entry
#[allow(dead_code)]
const AT_EXECFD: u64 = 2; // File descriptor of program
const AT_PHDR: u64 = 3; // Program headers address
const AT_PHENT: u64 = 4; // Size of program header entries
const AT_PHNUM: u64 = 5; // Number of program header entries
const AT_PAGESZ: u64 = 6; // System page size
#[allow(dead_code)]
const AT_BASE: u64 = 7; // Interpreter base address
#[allow(dead_code)]
const AT_FLAGS: u64 = 8; // Flags
const AT_ENTRY: u64 = 9; // Program entry point address
const AT_UID: u64 = 11; // Real user ID
const AT_EUID: u64 = 12; // Effective user ID
const AT_GID: u64 = 13; // Real group ID
const AT_EGID: u64 = 14; // Effective group ID
#[allow(dead_code)]
const AT_PLATFORM: u64 = 15; // Platform string address
#[allow(dead_code)]
const AT_HWCAP: u64 = 16; // Hardware capabilities
const AT_CLKTCK: u64 = 17; // Clock ticks per second
const AT_SECURE: u64 = 23; // Secure mode boolean
const AT_RANDOM: u64 = 25; // Address of random bytes
const AT_EXECFN: u64 = 31; // Filename of executed program

#[derive(Debug)]
pub struct RV64GC {
    pub registers: RV64GCRegisters,
    pub float_registers: RV64GCFloatRegisters,
    pub fcsr: FCSR,
    pub ram: Ram,
    pub should_quit: bool,
    elf_bin: Vec<u8>,
}

impl Default for RV64GC {
    fn default() -> Self {
        Self::new()
    }
}

impl RV64GC {
    fn write_auxv_to_stack(
        &mut self,
        elf: Elf,
        phdr_ptr: Option<u64>,
        rand_ptr: u64,
        execfn_ptr: u64,
    ) -> u64 {
        let mut map = vec![
            (AT_NULL, 0),
            (AT_PHENT, elf.header.e_phentsize.into()),
            (AT_PHNUM, elf.header.e_phnum.into()),
            (AT_PAGESZ, 4096),
            (AT_ENTRY, elf.entry),
            (AT_UID, 1000),
            (AT_EUID, 1000),
            (AT_GID, 1000),
            (AT_EGID, 1000),
            (AT_SECURE, 0),
            (AT_RANDOM, rand_ptr),
            (AT_CLKTCK, 100),
            (AT_EXECFN, execfn_ptr),
        ];

        if let Some(p) = phdr_ptr {
            map.push((AT_PHDR, p))
        }

        let mut sp = self.registers[Sp];
        for (k, v) in map.iter() {
            sp -= 8;
            self.ram.write_doubleword(sp, *v).unwrap();

            sp -= 8;
            self.ram.write_doubleword(sp, *k).unwrap();
        }

        sp
    }

    fn initialize_stack(&mut self, elf: Elf, phdr_addr: Option<u64>) {
        let ram = &mut self.ram;
        let args = std::env::args().collect::<Vec<String>>();
        let mut sp = 0x7FFF_FFFF_FFFF_FFF0;
        let stack_size: u64 = 8 * 1024 * 1024; // 8 MB
        let stack_start = sp - stack_size;

        let stack_region = MemoryRegion::new(stack_start, stack_size, vec![0; stack_size as usize]);
        ram.add_region(stack_region).unwrap();

        let mut rand_bytes = [0u8; 16];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut rand_bytes);

        for b in rand_bytes.iter().rev() {
            sp -= 1;
            ram.write_byte(sp, *b).unwrap();
        }

        let rand_ptr = sp;

        let mut envp_ptrs = vec![];

        // Push host environment vars to stack
        for (k, v) in std::env::vars() {
            for c in format!("{k}={v}\0").bytes().rev() {
                sp -= 1;
                ram.write_byte(sp, c).unwrap();
            }

            envp_ptrs.push(sp);
        }

        let mut argv_ptrs = vec![];

        for arg in args.iter().skip(2).rev() {
            sp -= 1;
            ram.write_byte(sp, 0).unwrap();
            for c in arg.bytes().rev() {
                sp -= 1;
                ram.write_byte(sp, c).unwrap();
            }

            argv_ptrs.push(sp);
        }

        let path_name = args.get(1).unwrap().split('/').last().unwrap();
        sp -= 1;
        ram.write_byte(sp, 0).unwrap();
        for c in path_name.bytes().rev() {
            sp -= 1;
            ram.write_byte(sp, c).unwrap();
        }

        argv_ptrs.push(sp);

        // NOTE: Leave 10kb of extra space for vars
        sp &= !0xF;

        // NOTE: Let go of the mutable borrow for now
        let _ = ram;

        self.registers[Sp] = sp;
        sp = self.write_auxv_to_stack(elf, phdr_addr, rand_ptr, *argv_ptrs.last().unwrap());

        let ram = &mut self.ram;

        // Push NULL terminator for envp
        sp -= 8;
        ram.write_doubleword(sp, 0).unwrap();

        // FIXME: Thows a 'malloc(): corrupted top size'
        // for i in envp_ptrs {
        //     sp -= 8;
        //     ram.write_doubleword(sp, i).unwrap();
        // }

        // Push NULL terminator for argv
        sp -= 8;
        ram.write_doubleword(sp, 0).unwrap();

        for i in argv_ptrs.iter() {
            sp -= 8;
            ram.write_doubleword(sp, *i).unwrap();
        }

        sp -= 8;
        // Push argc
        ram.write_doubleword(sp, argv_ptrs.len() as u64).unwrap();

        self.registers[Sp] = sp;
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
            fcsr: FCSR::new(),
            should_quit: false,
            elf_bin: vec![],
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

        let mut ehdr = None;

        for ph in &elf.program_headers {
            trace!("Reading ph of type: {:#08x}", ph.p_type);
            match ph.p_type {
                goblin::elf::program_header::PT_LOAD => {
                    let v_addr = ph.p_vaddr;
                    if ph.p_offset == 0 {
                        // HACK: Is it guaranteed to start 64 bytes ahead??!!
                        ehdr = Some(v_addr + 64);
                    }
                    let mem_size = ph.p_memsz;

                    let mut data = vec![0u8; mem_size as usize];

                    for (i, byte) in bin[ph.file_range()].iter().enumerate() {
                        data[i] = *byte;
                    }

                    let memory_region =
                        MemoryRegion::new_with_flags(v_addr, mem_size, data, ph.p_flags.into());

                    trace!(
                        "adding region, start: {}\t len: {}\toffset: {}",
                        v_addr,
                        mem_size,
                        ph.p_offset
                    );
                    self.ram.add_region(memory_region)?;
                }

                _ => trace!("skipping over ph type: {:08x}", ph.p_type),
            }
        }

        self.initialize_stack(elf, ehdr);
        self.elf_bin = bin;

        trace!("mem regions: {}", self.ram);

        Ok(())
    }

    pub fn reset(&mut self) {
        self.registers = RV64GCRegisters::new();
        self.registers[Sp] = 0x7FFF_FFFF_FFFF_FFF0;
        self.float_registers = RV64GCFloatRegisters::new();
        self.ram = Ram::new();

        self.load_elf(self.elf_bin.clone()).unwrap();
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

        trace!("pc: {:08x}", self.registers[Pc]);

        // if self.points_to_break.contains(&self.registers[Pc]) {
        //     println!("{}", &self.registers);
        // }

        self.execute();
        assert_eq!(self.registers[0], 0);
    }

    pub fn execute(&mut self) {
        let current_ins = self.ram.read_word(self.registers[Pc]).unwrap();

        if let Ok(env) = std::env::var("DUMP_OPS") {
            if env == "1" {
                trace!("opcode: {current_ins:08x}");
            }
        }

        let ins = self.find_instruction(current_ins);
        ins.execute_instruction(self);

        if let RV64GCInstruction::IllegalInstruction(opcode) = ins {
            panic!(
                "instruction not implemented:\n\tpc: {:08x}\n\tinstruction: {opcode:08x}",
                self.registers[Pc]
            );
        }

        if current_ins & 3 == 3 {
            self.registers[Pc] = self.registers[Pc].wrapping_add(4);
        } else {
            self.registers[Pc] = self.registers[Pc].wrapping_add(2);
        }
    }

    pub fn find_instruction(&self, current_ins: u32) -> RV64GCInstruction {
        use RV64GCInstruction::*;

        // Default values
        let rd = current_ins.bit_range(7..12) as Reg;
        let rs1 = current_ins.bit_range(15..20) as Reg;
        let rs2 = current_ins.bit_range(20..25) as Reg;
        let rs3 = current_ins.bit_range(27..32) as Reg;
        let imm = current_ins.bit_range(20..32) as Imm;

        let rm = current_ins.bit_range(12..15) as Reg;

        if current_ins & 0b11 != 0b11 {
            let c_ins = current_ins as u16;
            let c_rs1 = c_ins.bit_range(7..12) as Reg;
            let x_rs1 = c_ins.bit_range(7..10) as Reg;
            let x2_rs1 = c_ins.bit_range(2..5) as Reg;
            let c_rs2 = c_ins.bit_range(2..7) as Reg;

            return match c_ins {
                i if is_rv64c_nop_instruction(i) => Cnop,
                i if is_rv64c_ebreak_instruction(i) => Cebreak,
                i if is_rv64c_jalr_instruction(i) => Cjalr(c_rs1),
                i if is_rv64c_add_instruction(i) => Cadd(c_rs1, c_rs2),

                i if is_rv64c_jr_instruction(i) => Cjr(c_rs1),
                i if is_rv64c_mv_instruction(i) => {
                    trace!("c.mv opcode: {i:04x}");
                    Cmv(c_rs1, c_rs2)
                }

                i if is_rv64c_addi_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 5 | (i.bit_range(2..7) as u32);
                    let simm = sign_extend(imm.into(), 6);

                    Caddi(c_rs1, simm)
                }

                i if is_rv64c_addiw_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 5 | (i.bit_range(2..7) as u32);
                    let simm = sign_extend(imm.into(), 6);

                    Caddiw(c_rs1, simm)
                }

                i if is_rv64c_addi16sp_instruction(i) => {
                    let imm = (i.bit(12) as u16) << 9
                        | i.bit_range(3..5) << 7
                        | (i.bit(5) as u16) << 6
                        | (i.bit(2) as u16) << 5
                        | (i.bit(6) as u16) << 4;

                    Caddi16sp(sign_extend(imm as u64, 10))
                }

                i if is_rv64c_lui_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 17 | u32::from(i.bit_range(2..7)) << 12;

                    trace!("c.lui imm: {}", sign_extend(imm as u64, 18));

                    Clui(c_rs1, imm)
                }

                i if is_rv64c_andi_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 5 | u32::from(i.bit_range(2..7));
                    let simm = sign_extend(imm.into(), 6);
                    Candi(x_rs1 + 8, simm)
                }

                i if is_rv64c_ldsp_instruction(i) => {
                    let imm = (i.bit_range(2..5) as u32) << 6
                        | (i.bit(12) as u32) << 5
                        | (i.bit_range(5..7) as u32) << 3;

                    trace!("c.ldsp opcode: {i:04x}");

                    Cldsp(c_rs1, imm)
                }

                i if is_rv64c_lwsp_instruction(i) => {
                    let imm = (i.bit_range(2..4) as u32) << 6
                        | (i.bit(12) as u32) << 5
                        | (i.bit_range(4..7) as u32) << 2;

                    trace!("c.lwsp opcode: {i:04x}");

                    Clwsp(c_rs1, imm)
                }

                i if is_rv64c_swsp_instruction(i) => {
                    let imm = i.bit_range(7..9) << 6 | i.bit_range(9..13) << 2;

                    Cswsp(c_rs2, imm.into())
                }

                i if is_rv64c_addi4spn_instruction(i) => {
                    let imm = u32::from(i.bit_range(7..11)) << 6
                        | u32::from(i.bit_range(11..13)) << 4
                        | (i.bit(5) as u32) << 3
                        | (i.bit(6) as u32) << 2;

                    trace!("c.addi4spn opcode: {i:04x}");

                    Caddi4spn(x2_rs1 + 8, imm)
                }

                i if is_rv64c_li_instruction(i) => {
                    trace!("c.li instruction: {:04x}", i);
                    let imm = (i.bit(12) as u32) << 5 | u32::from(i.bit_range(2..7));
                    Cli(c_rs1, imm)
                }

                i if is_rv64c_slli_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 5 | u32::from(i.bit_range(2..7));
                    Cslli(c_rs1, imm)
                }

                i if is_rv64c_srli_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 5 | u32::from(i.bit_range(2..7));
                    Csrli(x_rs1 + 8, imm)
                }

                i if is_rv64c_srai_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 5 | u32::from(i.bit_range(2..7));
                    Csrai(x_rs1 + 8, imm)
                }

                i if is_rv64c_sdsp_instruction(i) => {
                    let imm = (i.bit_range(7..10) as u32) << 6 | (i.bit_range(10..13) as u32) << 3;

                    Csdsp(c_rs2, imm)
                }

                i if is_rv64c_ld_instruction(i) => {
                    let imm = (i.bit_range(5..7) as u32) << 6 | (i.bit_range(10..13) as u32) << 3;

                    Cld(x2_rs1 + 8, x_rs1 + 8, imm)
                }

                i if is_rv64c_beqz_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 8
                        | (i.bit_range(5..7) as u32) << 6
                        | (i.bit(2) as u32) << 5
                        | (i.bit_range(10..12) as u32) << 3
                        | (i.bit_range(3..5) as u32) << 1;

                    trace!("beqz imm: {imm}");

                    Cbeqz(x_rs1 + 8, imm)
                }

                i if is_rv64c_bnez_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 8
                        | (i.bit_range(5..7) as u32) << 6
                        | (i.bit(2) as u32) << 5
                        | (i.bit_range(10..12) as u32) << 3
                        | (i.bit_range(3..5) as u32) << 1;

                    Cbnez(x_rs1 + 8, imm)
                }

                i if is_rv64c_sd_instruction(i) => {
                    let imm = (i.bit_range(5..7) as u32) << 6 | (i.bit_range(10..13) as u32) << 3;

                    Csd(x_rs1 + 8, x2_rs1 + 8, imm)
                }

                // Wtf is this bit layout????
                // [ 11 | 4 | 9 | 8 | 10 | 6 | 7 | 3 | 2 | 1 | 5 ]
                //   12  11  10   9    8   7   6   5   4   3   2
                i if is_rv64c_j_instruction(i) => {
                    let imm = (i.bit(12) as u32) << 11
                        | (i.bit(8) as u32) << 10
                        | (i.bit_range(9..11) as u32) << 8
                        | (i.bit(6) as u32) << 7
                        | (i.bit(7) as u32) << 6
                        | (i.bit(2) as u32) << 5
                        | (i.bit(11) as u32) << 4
                        | (i.bit_range(3..6) as u32) << 1;

                    Cj(imm)
                }

                i if is_rv64c_sw_instruction(i) => {
                    let imm = (i.bit(5) as u32) << 6
                        | (i.bit_range(10..13) as u32) << 3
                        | (i.bit(6) as u32) << 2;

                    Csw(x_rs1 + 8, x2_rs1 + 8, imm)
                }

                i if is_rv64c_lw_instruction(i) => {
                    let imm = (i.bit(5) as u32) << 6
                        | (i.bit_range(10..13) as u32) << 3
                        | (i.bit(6) as u32) << 2;

                    // NOTE: Rd is swapped for some reason
                    Clw(x2_rs1 + 8, x_rs1 + 8, imm)
                }

                i if is_rv64c_or_instruction(i) => Cor(x_rs1 + 8, x2_rs1 + 8),

                i if is_rv64c_and_instruction(i) => Cand(x_rs1 + 8, x2_rs1 + 8),

                i if is_rv64c_xor_instruction(i) => Cxor(x_rs1 + 8, x2_rs1 + 8),

                i if is_rv64c_sub_instruction(i) => Csub(x_rs1 + 8, x2_rs1 + 8),

                i if is_rv64c_addw_instruction(i) => Caddw(x_rs1 + 8, x2_rs1 + 8),

                i if is_rv64c_subw_instruction(i) => Csubw(x_rs1 + 8, x2_rs1 + 8),

                i if is_rv64c_fsd_instruction(i) => {
                    let imm = i.bit_range(5..7) << 6 | i.bit_range(10..13) << 3;
                    Cfsd(x_rs1 + 8, x2_rs1 + 8, imm.into())
                }

                i if is_rv64c_fsdsp_instruction(i) => {
                    let imm = i.bit_range(7..10) << 6 | i.bit_range(10..13) << 3;

                    Cfsdsp(c_rs2, imm.into())
                }

                _ => IllegalInstruction(c_ins.into()),
            };
        }
        match current_ins {
            i if is_rv64i_add_instruction(i) => Add(rd, rs1, rs2),

            i if is_rv64i_addi_instruction(i) => Addi(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_auipc_instruction(i) => {
                let ov_imm = i.bit_range(12..32) << 12;
                Auipc(rd, sign_extend(ov_imm.into(), 32))
            }

            i if is_rv64i_lui_instruction(i) => {
                let ov_imm = i.bit_range(12..32) << 12;
                Lui(rd, sign_extend(ov_imm.into(), 32))
            }

            i if is_rv64i_slti_instruction(i) => Slti(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_sltiu_instruction(i) => Sltiu(rd, rs1, imm),

            i if is_rv64i_xori_instruction(i) => Xori(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_ori_instruction(i) => Ori(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_andi_instruction(i) => Andi(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_slli_instruction(i) => {
                let shamt = i.bit_range(20..26);
                Slli(rd, rs1, shamt)
            }

            i if is_rv64i_srli_instruction(i) => {
                let shamt = i.bit_range(20..26);
                Srli(rd, rs1, shamt)
            }

            i if is_rv64i_srai_instruction(i) => {
                let shamt = i.bit_range(20..26);
                Srai(rd, rs1, shamt)
            }

            i if is_rv64i_sll_instruction(i) => Sll(rd, rs1, rs2),

            i if is_rv64i_srl_instruction(i) => Srl(rd, rs1, rs2),

            i if is_rv64i_slt_instruction(i) => Slt(rd, rs1, rs2),

            i if is_rv64i_sltu_instruction(i) => Sltu(rd, rs1, rs2),

            i if is_rv64i_sub_instruction(i) => Sub(rd, rs1, rs2),

            i if is_rv64i_xor_instruction(i) => Xor(rd, rs1, rs2),

            i if is_rv64i_and_instruction(i) => And(rd, rs1, rs2),

            i if is_rv64i_or_instruction(i) => Or(rd, rs1, rs2),

            i if is_rv64i_ecall_instruction(i) => Ecall,

            i if is_rv64i_fence_instruction(i) => {
                Fence(i.bit_range(20..24) as u8, i.bit_range(24..28) as u8)
            }

            i if is_rv64i_lb_instruction(i) => Lb(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_lbu_instruction(i) => Lbu(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_lhu_instruction(i) => Lhu(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_sb_instruction(i) => {
                let lo_offset = i.bit_range(7..12);
                let hi_offset = i.bit_range(25..32);

                let offset = (hi_offset << 5) | lo_offset;

                Sb(rs1, rs2, sign_extend12(offset))
            }

            i if is_rv64i_sh_instruction(i) => {
                let lo_offset = i.bit_range(7..12);
                let hi_offset = i.bit_range(25..32);

                let offset = (hi_offset << 5) | lo_offset;

                Sh(rs1, rs2, sign_extend12(offset))
            }

            i if is_rv64i_lw_instruction(i) => Lw(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_sw_instruction(i) => {
                let lo_offset = i.bit_range(7..12);
                let hi_offset = i.bit_range(25..32);

                let offset = (hi_offset << 5) | lo_offset;

                Sw(rs1, rs2, sign_extend12(offset))
            }

            i if is_rv64i_ld_instruction(i) => Ld(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_sd_instruction(i) => {
                let lo_offset = i.bit_range(7..12);
                let hi_offset = i.bit_range(25..32);

                let offset = (hi_offset << 5) | lo_offset;

                Sd(rs1, rs2, sign_extend12(offset))
            }

            i if is_rv64i_jal_instruction(i) => {
                let offset = (i.bit(31) as u32) << 20
                    | i.bit_range(12..20) << 12
                    | (i.bit(20) as u32) << 11
                    | i.bit_range(21..31) << 1;

                let s_offset = sign_extend(offset.into(), 21);

                trace!("offset: {:#020b}", offset);
                Jal(rd, s_offset)
            }

            i if is_rv64i_jalr_instruction(i) => Jalr(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_bge_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                Bge(rs1, rs2, offset)
            }

            i if is_rv64i_bgeu_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                Bgeu(rs1, rs2, offset)
            }

            i if is_rv64i_beq_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                Beq(rs1, rs2, offset)
            }

            i if is_rv64i_bne_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                Bne(rs1, rs2, offset)
            }

            i if is_rv64i_blt_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                Blt(rs1, rs2, offset)
            }

            i if is_rv64i_bltu_instruction(i) => {
                let offset = (i.bit(31) as u32) << 12
                    | (i.bit(7) as u32) << 11
                    | i.bit_range(25..31) << 5
                    | i.bit_range(8..12) << 1;

                Bltu(rs1, rs2, offset)
            }

            i if is_rv64i_addiw_instruction(i) => Addiw(rd, rs1, imm),

            i if is_rv64i_slliw_instruction(i) => {
                let shamt = current_ins.bit_range(20..26);

                Slliw(rd, rs1, shamt)
            }

            i if is_rv64i_srliw_instruction(i) => {
                let shamt = current_ins.bit_range(20..26);

                Srliw(rd, rs1, shamt)
            }

            i if is_rv64i_sraiw_instruction(i) => {
                let shamt = i.bit_range(20..25);

                Sraiw(rd, rs1, shamt)
            }

            i if is_rv64i_addw_instruction(i) => Addw(rd, rs1, rs2),

            i if is_rv64i_subw_instruction(i) => Subw(rd, rs1, rs2),

            i if is_rv64i_sllw_instruction(i) => Sllw(rd, rs1, rs2),

            i if is_rv64i_srlw_instruction(i) => Srlw(rd, rs1, rs2),

            i if is_rv64i_sraw_instruction(i) => Sraw(rd, rs1, rs2),

            i if is_rv64i_lwu_instruction(i) => Lwu(rd, rs1, imm),

            i if is_rv64m_mul_instruction(i) => Mul(rd, rs1, rs2),

            i if is_rv64m_mulh_instruction(i) => Mulh(rd, rs1, rs2),

            i if is_rv64m_mulhsu_instruction(i) => Mulhsu(rd, rs1, rs2),

            i if is_rv64m_mulhu_instruction(i) => Mulhu(rd, rs1, rs2),

            i if is_rv64m_div_instruction(i) => Div(rd, rs1, rs2),

            i if is_rv64m_divu_instruction(i) => Divu(rd, rs1, rs2),

            i if is_rv64m_rem_instruction(i) => Rem(rd, rs1, rs2),

            i if is_rv64m_remu_instruction(i) => Remu(rd, rs1, rs2),

            i if is_rv64m_mulw_instruction(i) => Mulw(rd, rs1, rs2),

            i if is_rv64m_divw_instruction(i) => Divw(rd, rs1, rs2),

            i if is_rv64m_divuw_instruction(i) => Divuw(rd, rs1, rs2),

            i if is_rv64m_remw_instruction(i) => Remw(rd, rs1, rs2),

            i if is_rv64m_remuw_instruction(i) => Remuw(rd, rs1, rs2),

            i if is_rv64a_lrw_instruction(i) => Lrw(rd, rs1),

            i if is_rv64a_lrd_instruction(i) => Lrd(rd, rs1),

            i if is_rv64a_scw_instruction(i) => Scw(rd, rs1, rs2),

            i if is_rv64a_scd_instruction(i) => Scd(rd, rs1, rs2),

            i if is_rv64a_amoswapw_instruction(i) => Amoswapw(rd, rs1, rs2),

            i if is_rv64a_amoaddw_instruction(i) => Amoaddw(rd, rs1, rs2),

            i if is_rv64a_amoxorw_instruction(i) => Amoxorw(rd, rs1, rs2),

            i if is_rv64a_amoandw_instruction(i) => Amoandw(rd, rs1, rs2),

            i if is_rv64a_amoorw_instruction(i) => Amoorw(rd, rs1, rs2),

            i if is_rv64a_amominw_instruction(i) => Amominw(rd, rs1, rs2),

            i if is_rv64a_amomaxw_instruction(i) => Amomaxw(rd, rs1, rs2),

            i if is_rv64a_amominuw_instruction(i) => Amominuw(rd, rs1, rs2),

            i if is_rv64a_amomaxuw_instruction(i) => Amomaxuw(rd, rs1, rs2),

            i if is_rv64a_amoswapd_instruction(i) => Amoswapd(rd, rs1, rs2),

            i if is_rv64a_amoaddd_instruction(i) => Amoaddd(rd, rs1, rs2),

            i if is_rv64a_amoxord_instruction(i) => Amoxord(rd, rs1, rs2),

            i if is_rv64a_amoandd_instruction(i) => Amoandd(rd, rs1, rs2),

            i if is_rv64a_amoord_instruction(i) => Amoord(rd, rs1, rs2),

            i if is_rv64a_amomind_instruction(i) => Amomind(rd, rs1, rs2),

            i if is_rv64a_amomaxd_instruction(i) => Amomaxd(rd, rs1, rs2),

            i if is_rv64a_amominud_instruction(i) => Amominud(rd, rs1, rs2),

            i if is_rv64a_amomaxud_instruction(i) => Amomaxud(rd, rs1, rs2),

            // RV64F
            i if is_rv64f_fmadds_instruction(i) => Fmadds(rd, rm, rs1, rs2, rs3),
            i if is_rv64f_fmsubs_instruction(i) => Fmsubs(rd, rm, rs1, rs2, rs3),
            i if is_rv64f_fnmadds_instruction(i) => Fnmadds(rd, rm, rs1, rs2, rs3),
            i if is_rv64f_fnmsubs_instruction(i) => Fnmsubs(rd, rm, rs1, rs2, rs3),

            i if is_rv64f_fadds_instruction(i) => Fadds(rd, rm, rs1, rs2),
            i if is_rv64f_fsubs_instruction(i) => Fsubs(rd, rm, rs1, rs2),
            i if is_rv64f_fmuls_instruction(i) => Fmuls(rd, rm, rs1, rs2),
            i if is_rv64f_fdivs_instruction(i) => Fdivs(rd, rm, rs1, rs2),
            i if is_rv64f_fsqrts_instruction(i) => Fsqrts(rd, rm, rs1),

            i if is_rv64f_fsgnjs_instruction(i) => Fsgnjs(rd, rs1, rs2),
            i if is_rv64f_fsgnjns_instruction(i) => Fsgnjs(rd, rs1, rs2),
            i if is_rv64f_fsgnjxs_instruction(i) => Fsgnjs(rd, rs1, rs2),

            i if is_rv64f_fmins_instruction(i) => Fmins(rd, rs1, rs2),
            i if is_rv64f_fmaxs_instruction(i) => Fmaxs(rd, rs1, rs2),

            i if is_rv64f_fcvtws_instruction(i) => Fcvtws(rd, rm, rs1),
            i if is_rv64f_fcvtwus_instruction(i) => Fcvtws(rd, rm, rs1),
            i if is_rv64f_fmvxw_instruction(i) => Fmvxw(rd, rs1),
            i if is_rv64f_fmvwx_instruction(i) => Fmvwx(rd, rs1),

            i if is_rv64f_feqs_instruction(i) => Feqs(rd, rs1, rs2),
            i if is_rv64f_flts_instruction(i) => Flts(rd, rs1, rs2),
            i if is_rv64f_fles_instruction(i) => Fles(rd, rs1, rs2),

            i if is_rv64f_fclasss_instruction(i) => Fclasss(rd, rs1),

            i if is_rv64f_fcvtsw_instruction(i) => Fcvtsw(rd, rm, rs1),
            i if is_rv64f_fcvtswu_instruction(i) => Fcvtswu(rd, rm, rs1),

            i if is_rv64f_flw_instruction(i) => {
                trace!("flw: {i:08x}");
                Flw(rd, rs1, imm)
            }
            i if is_rv64f_fsw_instruction(i) => {
                let imm = i.bit_range(25..32) << 5 | i.bit_range(7..12);
                trace!("fsw: {i:08x}");
                trace!("imm: {}", sign_extend12(imm));

                Fsw(rs1, rs2, imm)
            }

            // RV64D
            i if is_rv64f_fsd_instruction(i) => {
                let imm = i.bit_range(25..32) << 5 | i.bit_range(7..12);
                let simm = sign_extend12(imm);

                Fsd(rs1, rs2, simm)
            }

            i if is_rv64f_fsgnjd_instruction(i) => Fsgnjd(rd, rs1, rs2),

            i if is_rv64f_fcvtds_instruction(i) => Fcvtds(rd, rm, rs1),

            i if is_rv64f_fmvxd_instruction(i) => Fmvxd(rd, rs1),

            _ => IllegalInstruction(current_ins),
        }
    }

    pub fn syscall_handler(&mut self) {
        let span = span!(Level::TRACE, "syscall_handler");
        let _guard = span.enter();

        let syscall_id = self.registers[A7];
        trace!("system call: {syscall_id}");

        match syscall_id {
            62 => lseek(self),

            63 => read(self),

            64 => write(self),

            66 => writev(self),

            78 => readlink(self),

            80 => lstat(self),

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

            // NOTE: set_tid
            96 => {
                // PID
                self.registers[A0] = 0;
            }

            98 => futex(self),

            // NOTE: set_robust_list
            99 => {
                self.registers[A0] = 0;
            }

            113 => clock_gettime(self),

            131 => tgkill(self),

            134 => sig_action(self),

            135 => rt_sigprocmask(self),

            172 => getpid(self),
            178 => gettid(self),

            214 => brk(self),

            222 => mmap(self),

            226 => mprotect(self),

            258 => riscv_hwprobe(self),

            261 => prlimit64(self),

            278 => getrandom(self),

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

            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
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
    Fmvxw(Reg, Reg),
    Feqs(Reg, Reg, Reg),
    Flts(Reg, Reg, Reg),
    Fles(Reg, Reg, Reg),
    Fclasss(Reg, Reg),
    Fcvtsw(Reg, Reg, Reg),
    Fcvtswu(Reg, Reg, Reg),
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
    Fcvtsd(Reg, Reg),
    Fcvtds(Reg, Reg, Reg),
    Fcvtwd(Reg, Reg),
    Fcvtwud(Reg, Reg),
    Fcvtdwu(Reg, Reg),
    Fcvtdw(Reg, Reg),
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

        let span = span!(Level::TRACE, "execute_instruction");
        let _guard = span.enter();
        trace!("{}", self);

        match self {
            IllegalInstruction(i) => {
                panic!("Not a valid instruction: 0x{i:08x}");
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
                let simm = sign_extend12(*imm) as u64;

                if cpu.registers[rs1] < simm {
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
                    cpu.registers[rs1].wrapping_shl(cpu.registers[rs2] as u32 & ((1 << 5) - 1));
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
                cpu.registers[rd] = cpu.registers[rs1] >> (cpu.registers[rs2] & ((1 << 5) - 1));
            }

            Sra(rd, rs1, rs2) => {
                cpu.registers[rd] =
                    ((cpu.registers[rs1] as i64) >> (cpu.registers[rs2] & ((1 << 5) - 1))) as u64;
            }

            Or(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] | cpu.registers[rs2];
            }

            And(rd, rs1, rs2) => {
                cpu.registers[rd] = cpu.registers[rs1] & cpu.registers[rs2];
            }

            Fence(_, _) => {}

            FenceI => todo!(),

            Uret => todo!(),

            Sret => todo!(),

            Mret => todo!(),

            Wfi => todo!(),

            SfenceVma(_, _, _) => todo!(),

            Csrrw(_, _, _) => todo!(),
            Csrrs(_, _, _) => todo!(),
            Csrrc(_, _, _) => todo!(),
            Csrrwi(_, _, _) => todo!(),
            Csrrsi(_, _, _) => todo!(),
            Csrrci(_, _, _) => todo!(),

            Ecall => {
                cpu.syscall_handler();
            }

            Ebreak => {
                todo!()
            }

            // This was previously checking the sign bit at the 4th bit,
            // absolutely stupid...
            Lb(rd, rs1, simm) => {
                cpu.registers[rd] = sign_extend(
                    cpu.ram
                        .read_byte((cpu.registers[rs1] as i64 + simm) as u64)
                        .unwrap()
                        .into(),
                    8,
                ) as u64;
            }

            Lbu(rd, rs1, simm) => {
                trace!("lbu simm: {simm}");
                cpu.registers[rd] = cpu
                    .ram
                    .read_byte(cpu.registers[rs1].wrapping_add_signed(*simm))
                    .unwrap()
                    .into();
            }

            Lhu(rd, rs1, offset) => {
                cpu.registers[rd] = cpu
                    .ram
                    .read_halfword(cpu.registers[rs1] + *offset as u64)
                    .unwrap()
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
                let addr = (cpu.registers[rs1] as i64).wrapping_add(*simm);

                cpu.registers[rd] =
                    sign_extend(cpu.ram.read_halfword(addr as u64).unwrap(), 16) as u64;
            }

            Lw(rd, rs1, simm) => {
                cpu.registers[rd] = sign_extend(
                    cpu.ram
                        .read_word((cpu.registers[rs1] as i64 + simm) as u64)
                        .inspect_err(|e| panic!("{e}"))
                        .unwrap()
                        .into(),
                    32,
                ) as u64;
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
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bne(rs1, rs2, imm) => {
                if cpu.registers[rs1] != cpu.registers[rs2] {
                    let simm = sign_extend12(*imm);
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
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bge(rs1, rs2, imm) => {
                if (cpu.registers[rs1] as i64) >= (cpu.registers[rs2] as i64) {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bltu(rs1, rs2, imm) => {
                if cpu.registers[rs1] < cpu.registers[rs2] {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Bgeu(rs1, rs2, imm) => {
                if cpu.registers[rs1] >= cpu.registers[rs2] {
                    let simm = sign_extend12(*imm);
                    cpu.registers[Pc] = (cpu.registers[Pc] as i64).wrapping_add(simm) as u64;

                    // Due to adding 4 to the PC every step, we must decrement by 4 to revert this
                    // behaviour
                    cpu.registers[Pc] -= 4;
                }
            }

            Addiw(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let value = (cpu.registers[rs1] as i64).wrapping_add(simm) & (u32::MAX as i64);

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
                let rs1_low = cpu.registers[rs1] & (u32::MAX as u64);
                let rs2_low = cpu.registers[rs2] & (u32::MAX as u64);

                cpu.registers[rd] = sign_extend(rs1_low.wrapping_add(rs2_low), 32) as u64;
            }

            Subw(rd, rs1, rs2) => {
                let rs1_low = cpu.registers[rs1] & (u32::MAX as u64);
                let rs2_low = cpu.registers[rs2] & (u32::MAX as u64);

                cpu.registers[rd] = sign_extend(rs1_low.wrapping_sub(rs2_low), 32) as u64;
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

                cpu.registers[rd] = sign_extend(u64::from(mem), 32) as u64;
            }

            Mul(rd, rs1, rs2) => {
                let value = (cpu.registers[rs1] as i64).wrapping_mul(cpu.registers[rs2] as i64);

                cpu.registers[rd] = (value & i32::MAX as i64) as u64;
            }

            Mulh(rd, rs1, rs2) => {
                let h_multiplicand = (cpu.registers[rs1] >> 32) as i64;
                let h_multiplier = (cpu.registers[rs2] >> 32) as i64;

                cpu.registers[rd] = h_multiplicand.wrapping_mul(h_multiplier) as u64;
            }

            Mulhsu(rd, rs1, rs2) => {
                let h_multiplicand = cpu.registers[rs1] >> 32;
                let h_multiplier = (cpu.registers[rs2] >> 32) as i64;
                let result: u64 = if h_multiplier.is_negative() {
                    (h_multiplicand as i64).wrapping_mul(h_multiplier) as u64
                } else {
                    h_multiplicand.wrapping_mul(h_multiplier as u64)
                };

                cpu.registers[rd] = result;
            }

            Mulhu(rd, rs1, rs2) => {
                let h_multiplicand = cpu.registers[rs1] >> 32;
                let h_multiplier = cpu.registers[rs2] >> 32;
                cpu.registers[rd] = h_multiplicand.wrapping_mul(h_multiplier);
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
                    cpu.registers[rd] = u64::MAX;
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
                    cpu.registers[rd] = u64::MAX;
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
                    cpu.registers[rd] = u64::MAX;
                    return;
                }

                cpu.registers[rd] =
                    sign_extend(signed_rs1.wrapping_rem(signed_rs2) as u64, 32) as u64;
            }

            Remuw(rd, rs1, rs2) => {
                let unsigned_rs1 = (cpu.registers[rs1] & (u32::MAX as u64)) as u32;
                let unsigned_rs2 = (cpu.registers[rs2] & (u32::MAX as u64)) as u32;

                if unsigned_rs2 == 0 {
                    cpu.registers[rd] = u64::MAX;
                    return;
                }

                cpu.registers[rd] =
                    sign_extend(unsigned_rs1.wrapping_rem(unsigned_rs2) as u64, 32) as u64;
            }

            // WARNING: RV64A
            // TODO: Properly implement RV64A for multithreading
            Lrw(rd, rs1) => {
                cpu.registers[rd] = cpu.ram.read_word(cpu.registers[rs1]).unwrap() as u64;
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

                cpu.float_registers[rd] = u64::from(round_f32((rs1 * rs2) + rs3, rm).to_bits());
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

                cpu.float_registers[rd] = u64::from(round_f32((rs1 * rs2) - rs3, rm).to_bits());
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

                cpu.float_registers[rd] = u64::from(round_f32(-(rs1 * rs2) + rs3, rm).to_bits());
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

                cpu.float_registers[rd] = u64::from(round_f32(-(rs1 * rs2) - rs3, rm).to_bits());
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

                cpu.float_registers[rd] = u64::from(res.to_bits());
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

                cpu.float_registers[rd] = u64::from(res.to_bits());
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

                cpu.float_registers[rd] = u64::from(res.to_bits());
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

                cpu.float_registers[rd] = u64::from(res.to_bits());
            }

            Fsqrts(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);

                let res = round_f32(rs1.sqrt(), rm);

                cpu.float_registers[rd] = u64::from(res.to_bits());
            }

            Fsgnjs(rd, rs1, rs2) => {
                let rs1 = cpu.float_registers[rs1] as u32;
                let rs2 = cpu.float_registers[rs2] as u32;

                let sign_bit = rs1.bit(31);
                let res = *rs2.bit_range(0..31).set_bit(31, sign_bit);

                cpu.float_registers[rd] = u64::from(res);
            }

            Fsgnjns(rd, rs1, rs2) => {
                let rs1 = cpu.float_registers[rs1] as u32;
                let rs2 = cpu.float_registers[rs2] as u32;

                let sign_bit = rs1.bit(31);
                let res = *rs2.bit_range(0..31).set_bit(31, !sign_bit);

                cpu.float_registers[rd] = u64::from(res);
            }

            Fsgnjxs(rd, rs1, rs2) => {
                let rs1 = cpu.float_registers[rs1] as u32;
                let rs2 = cpu.float_registers[rs2] as u32;

                let rs1_sign_bit = rs1.bit(31);
                let rs2_sign_bit = rs2.bit(31);
                let res = *rs2
                    .bit_range(0..31)
                    .set_bit(31, rs1_sign_bit ^ rs2_sign_bit);

                cpu.float_registers[rd] = u64::from(res);
            }

            Fmins(rd, rs1, rs2) => {
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                cpu.float_registers[rd] = u64::from(rs1.min(rs2).to_bits())
            }

            Fmaxs(rd, rs1, rs2) => {
                let rs1 = f32::from_bits(cpu.float_registers[rs1] as u32);
                let rs2 = f32::from_bits(cpu.float_registers[rs2] as u32);

                cpu.float_registers[rd] = u64::from(rs1.max(rs2).to_bits())
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

                cpu.registers[rd] = val as u64
            }

            Fmvxw(rd, rs1) => cpu.registers[rd] = u64::from(cpu.float_registers[rs1] as u32),

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
                cpu.float_registers[rd] = u64::from(round_f32(int as f32, rm).to_bits());
            }

            Fcvtswu(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };
                let int = cpu.registers[rs1] as u32;
                cpu.float_registers[rd] = u64::from(round_f32(int as f32, rm).to_bits());
            }

            Fmvwx(rd, rs1) => cpu.float_registers[rd] = cpu.registers[rs1] & u32::MAX as u64,

            Flw(rd, rs1, imm) => {
                let simm = sign_extend12(*imm);
                let addr = (cpu.registers[rs1] as i64).wrapping_add(simm) as u64;
                trace!("simm: {simm}");
                let value = f32::from_bits(cpu.ram.read_word(addr).unwrap());

                trace!("flw addr: {addr:08x}");

                cpu.float_registers[rd] = (value as f32).to_bits() as u64;
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

            Fsd(rs1, rs2, simm) => {
                let value = cpu.float_registers[rs2];

                cpu.ram
                    .write_doubleword(cpu.registers[rs1].wrapping_add_signed(*simm), value)
                    .unwrap();
            }

            Fcvtds(rd, rm, rs1) => {
                let rm = if *rm == 0b111 {
                    cpu.fcsr.frm
                } else {
                    RoundingMode::from(rm)
                };

                let dbl_precision = f64::from_bits(cpu.float_registers[rs1]);
                cpu.float_registers[rd] = (round_f32(dbl_precision as f32, rm)).to_bits() as u64;
            }

            Fmvxd(rd, rs1) => cpu.registers[rd] = cpu.registers[rs1],

            Fsgnjd(rd, rs1, rs2) => {
                let sign_bit_rem = cpu.float_registers[rs1] & 0x8000000000000000;
                cpu.float_registers[rd] = sign_bit_rem & cpu.float_registers[rs2];
            }

            // NOTE: RV64C
            Cebreak => panic!("c.ebreak not implemented!"),

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

            Cldsp(rs1, imm) => {
                let addr = cpu.registers[Sp] + (*imm) as u64;
                trace!("c.ldsp: {addr:08x}");
                cpu.registers[rs1] = cpu.ram.read_doubleword(addr).unwrap();
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
                let addr = cpu.registers[Sp] + u64::from(*imm);
                cpu.ram.write_doubleword(addr, cpu.registers[rs1]).unwrap();
            }

            Cld(rd, rs1, imm) => {
                let val = cpu
                    .ram
                    .read_doubleword(cpu.registers[rs1] + u64::from(*imm))
                    .unwrap();

                cpu.registers[rd] = val;
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
                cpu.ram
                    .write_doubleword(
                        cpu.registers[rs1].wrapping_add(*imm as u64),
                        cpu.registers[rs2],
                    )
                    .unwrap();
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
                cpu.ram
                    .write_word(cpu.registers[rs1] + *imm as u64, cpu.registers[rs2] as u32)
                    .unwrap();
            }

            Csrli(rd, imm) => {
                cpu.registers[rd] = cpu.registers[rd].wrapping_shr(*imm);
            }

            Csrai(rd, imm) => {
                cpu.registers[rd] = (cpu.registers[rd] as i64).wrapping_shr(*imm) as u64;
            }

            Caddiw(rd, simm) => {
                let rd_val = cpu.registers[rd] as i64 as i32;

                cpu.registers[rd] = rd_val.wrapping_add(*simm as i32) as u64;
            }

            Clwsp(rd, imm) => {
                let val = sign_extend(
                    cpu.ram
                        .read_word(cpu.registers[Sp] + *imm as u64)
                        .unwrap()
                        .into(),
                    32,
                );
                cpu.registers[rd] = val as u64;
            }

            Cnop => {}

            Csub(rd, rs1) => {
                cpu.registers[rd] = cpu.registers[rd].wrapping_sub(cpu.registers[rs1]);
            }

            Clw(rd, rs1, imm) => {
                let offset = cpu.registers[rs1] + *imm as u64;

                cpu.registers[rd] =
                    sign_extend(cpu.ram.read_word(offset).unwrap().into(), 32) as u64;
            }

            Caddw(rd, rs1) => {
                let res = cpu.registers[rd].wrapping_add(cpu.registers[rs1]);

                cpu.registers[rd] = sign_extend(res & u32::MAX as u64, 32) as u64;
            }

            Csubw(rd, rs1) => {
                let res = cpu.registers[rd].wrapping_sub(cpu.registers[rs1]);

                cpu.registers[rd] = sign_extend(res & u32::MAX as u64, 32) as u64;
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

            _ => todo!(),
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

#[derive(Debug)]
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

    pub const fn float_reg(value: u8) -> u8 {
        value + 33
    }
}

#[derive(Debug)]
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
