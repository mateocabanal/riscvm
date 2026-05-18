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
use crate::opcodes::*;
use crate::ram::MemoryRegion;
use crate::ram::Ram;
use crate::sign_extend;
use crate::sign_extend12;
use crate::syscalls::*;
use crate::tracer::{ExecutionEngine, ExecutionTracer, TraceOptions};
use std::collections::BTreeSet;
use std::fmt::Display;
use std::ops::{Index, IndexMut};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::cpu::RV64GCRegAbiName::*;

type Reg = u8;
type Imm = u32;
type Simm = i64;
type Csr = u16;

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

#[derive(Debug)]
pub struct RV64GC {
    pub registers: RV64GCRegisters,
    pub float_registers: RV64GCFloatRegisters,
    pub fcsr: FCSR,
    pub ram: Ram,
    pub filesystem: GuestFileSystem,
    tracer: Option<ExecutionTracer>,
    pub should_quit: bool,
    executable_path: Option<PathBuf>,
    argv: Vec<String>,
    elf_bin: Vec<u8>,
    decode_cache: Vec<Option<DecodedInstruction>>,
    jit_runtime_fault: Option<String>,
}

impl Default for RV64GC {
    fn default() -> Self {
        Self::new()
    }
}

impl RV64GC {
    fn initialize_stack_with_ext_lib(&mut self, elf: Elf, phdr_addr: Option<u64>) {
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

        // let envp_vec = std::env::vars()
        //     .map(|(k, v)| format!("{k}={v}"))
        //     .collect::<Vec<String>>();
        // for s in envp_vec.iter() {
        //     builder.env_v.push(s);
        // }

        let mut rand_bytes = [0u8; 16];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut rand_bytes);
        let auxv = [
            AuxVar::Phdr(phdr_addr.unwrap() as *const u8),
            AuxVar::Phent(elf.header.e_phentsize.into()),
            AuxVar::Phnum(elf.header.e_phnum.into()),
            AuxVar::Pagesz(4096),
            AuxVar::Entry(elf.header.e_entry as *const u8),
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
            executable_path: None,
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
        let entry = elf.entry;
        self.registers[Pc] = entry;
        let mut program_break_base = 0;

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
                        ehdr = Some(v_addr + elf.header.e_phoff);
                    }
                    let mem_size = ph.p_memsz;
                    program_break_base = program_break_base.max(v_addr + mem_size);

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

        self.apply_dynamic_relocations(&elf)?;
        self.ram.set_program_break_base(program_break_base);
        self.initialize_stack_with_ext_lib(elf, ehdr);
        self.elf_bin = bin;

        trace!("mem regions: {}", self.ram);

        Ok(())
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
        jit.run(self)
    }

    pub fn step(&mut self) {
        trace!("pc: {:08x}", self.registers[Pc]);

        // if self.points_to_break.contains(&self.registers[Pc]) {
        //     println!("{}", &self.registers);
        // }

        self.execute();
        self.registers[Zero] = 0;
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

                i if is_rv64c_fldsp_instruction(i) => {
                    let imm = (i.bit_range(2..5) as u32) << 6
                        | (i.bit(12) as u32) << 5
                        | (i.bit_range(5..7) as u32) << 3;

                    Cfldsp(c_rs1, imm)
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

                i if is_rv64c_fld_instruction(i) => {
                    let imm = (i.bit_range(5..7) as u32) << 6 | (i.bit_range(10..13) as u32) << 3;

                    Cfld(x2_rs1 + 8, x_rs1 + 8, imm)
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

            i if is_rv64i_sra_instruction(i) => Sra(rd, rs1, rs2),

            i if is_rv64i_slt_instruction(i) => Slt(rd, rs1, rs2),

            i if is_rv64i_sltu_instruction(i) => Sltu(rd, rs1, rs2),

            i if is_rv64i_sub_instruction(i) => Sub(rd, rs1, rs2),

            i if is_rv64i_xor_instruction(i) => Xor(rd, rs1, rs2),

            i if is_rv64i_and_instruction(i) => And(rd, rs1, rs2),

            i if is_rv64i_or_instruction(i) => Or(rd, rs1, rs2),

            i if is_rv64i_ecall_instruction(i) => Ecall,

            i if is_rv64i_ebreak_instruction(i) => Ebreak,

            i if is_rv64i_fence_instruction(i) => {
                Fence(i.bit_range(20..24) as u8, i.bit_range(24..28) as u8)
            }

            i if is_rv64i_fencei_instruction(i) => FenceI,

            i if is_rv64i_csrrw_instruction(i) => Csrrw(rd, rs1, imm as Csr),
            i if is_rv64i_csrrs_instruction(i) => Csrrs(rd, rs1, imm as Csr),
            i if is_rv64i_csrrc_instruction(i) => Csrrc(rd, rs1, imm as Csr),
            i if is_rv64i_csrrwi_instruction(i) => Csrrwi(rd, rs1 as Imm, imm as Csr),
            i if is_rv64i_csrrsi_instruction(i) => Csrrsi(rd, rs1 as Imm, imm as Csr),
            i if is_rv64i_csrrci_instruction(i) => Csrrci(rd, rs1 as Imm, imm as Csr),

            i if is_rv64i_lb_instruction(i) => Lb(rd, rs1, sign_extend12(imm)),

            i if is_rv64i_lh_instruction(i) => Lh(rd, rs1, sign_extend12(imm)),

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
            i if is_rv64f_fsgnjns_instruction(i) => Fsgnjns(rd, rs1, rs2),
            i if is_rv64f_fsgnjxs_instruction(i) => Fsgnjxs(rd, rs1, rs2),

            i if is_rv64f_fmins_instruction(i) => Fmins(rd, rs1, rs2),
            i if is_rv64f_fmaxs_instruction(i) => Fmaxs(rd, rs1, rs2),

            i if is_rv64f_fcvtws_instruction(i) => Fcvtws(rd, rm, rs1),
            i if is_rv64f_fcvtwus_instruction(i) => Fcvtwus(rd, rm, rs1),
            i if is_rv64f_fcvtls_instruction(i) => Fcvtls(rd, rm, rs1),
            i if is_rv64f_fcvtlus_instruction(i) => Fcvtlus(rd, rm, rs1),
            i if is_rv64f_fmvxw_instruction(i) => Fmvxw(rd, rs1),
            i if is_rv64f_fmvwx_instruction(i) => Fmvwx(rd, rs1),

            i if is_rv64f_feqs_instruction(i) => Feqs(rd, rs1, rs2),
            i if is_rv64f_flts_instruction(i) => Flts(rd, rs1, rs2),
            i if is_rv64f_fles_instruction(i) => Fles(rd, rs1, rs2),

            i if is_rv64f_fclasss_instruction(i) => Fclasss(rd, rs1),

            i if is_rv64f_fcvtsw_instruction(i) => Fcvtsw(rd, rm, rs1),
            i if is_rv64f_fcvtswu_instruction(i) => Fcvtswu(rd, rm, rs1),
            i if is_rv64f_fcvtsl_instruction(i) => Fcvtsl(rd, rm, rs1),
            i if is_rv64f_fcvtslu_instruction(i) => Fcvtslu(rd, rm, rs1),

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
            i if is_rv64f_fmaddd_instruction(i) => Fmaddd(rd, rm, rs1, rs2, rs3),
            i if is_rv64f_fmsubd_instruction(i) => Fmsubd(rd, rm, rs1, rs2, rs3),
            i if is_rv64f_fnmaddd_instruction(i) => Fnmaddd(rd, rm, rs1, rs2, rs3),
            i if is_rv64f_fnmsubd_instruction(i) => Fnmsubd(rd, rm, rs1, rs2, rs3),

            i if is_rv64f_faddd_instruction(i) => Faddd(rd, rm, rs1, rs2),
            i if is_rv64f_fsubd_instruction(i) => Fsubd(rd, rm, rs1, rs2),
            i if is_rv64f_fmuld_instruction(i) => Fmuld(rd, rm, rs1, rs2),
            i if is_rv64f_fdivd_instruction(i) => Fdivd(rd, rm, rs1, rs2),
            i if is_rv64f_fsqrtd_instruction(i) => Fsqrtd(rd, rm, rs1),

            i if is_rv64f_fld_instruction(i) => Fld(rd, rs1, imm),

            i if is_rv64f_fsd_instruction(i) => {
                let imm = i.bit_range(25..32) << 5 | i.bit_range(7..12);
                let simm = sign_extend12(imm);

                Fsd(rs1, rs2, simm)
            }

            i if is_rv64f_fsgnjd_instruction(i) => Fsgnjd(rd, rs1, rs2),

            i if is_rv64f_fsgnjnd_instruction(i) => Fsgnjnd(rd, rs1, rs2),

            i if is_rv64f_fsgnjxd_instruction(i) => Fsgnjxd(rd, rs1, rs2),

            i if is_rv64f_fmind_instruction(i) => Fmind(rd, rs1, rs2),
            i if is_rv64f_fmaxd_instruction(i) => Fmaxd(rd, rs1, rs2),

            i if is_rv64f_feqd_instruction(i) => Feqd(rd, rs1, rs2),
            i if is_rv64f_fltd_instruction(i) => Fltd(rd, rs1, rs2),
            i if is_rv64f_fled_instruction(i) => Fled(rd, rs1, rs2),
            i if is_rv64f_fclassd_instruction(i) => Fclassd(rd, rs1),

            i if is_rv64f_fcvtsd_instruction(i) => Fcvtsd(rd, rm, rs1),
            i if is_rv64f_fcvtds_instruction(i) => Fcvtds(rd, rm, rs1),
            i if is_rv64f_fcvtwd_instruction(i) => Fcvtwd(rd, rm, rs1),
            i if is_rv64f_fcvtwud_instruction(i) => Fcvtwud(rd, rm, rs1),
            i if is_rv64f_fcvtdw_instruction(i) => Fcvtdw(rd, rm, rs1),
            i if is_rv64f_fcvtdwu_instruction(i) => Fcvtdwu(rd, rm, rs1),

            i if is_rv64f_fmvxd_instruction(i) => Fmvxd(rd, rs1),

            _ => IllegalInstruction(current_ins),
        }
    }

    pub fn syscall_handler(&mut self) {
        let span = span!(Level::TRACE, "syscall_handler");
        let _guard = span.enter();

        let syscall_id = self.registers[A7];
        debug!("system call: {syscall_id}");
        self.trace_syscall(syscall_id);

        match syscall_id {
            17 => getcwd(self),

            23 => dup(self),

            24 => dup3(self),

            25 => fcntl(self),

            29 => ioctl(self),

            48 => faccessat(self),

            56 => openat(self),

            57 => close(self),

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

            214 => brk(self),

            215 => munmap(self),

            222 => mmap(self),

            226 => mprotect(self),

            233 => madvise(self),

            258 => riscv_hwprobe(self),

            261 => prlimit64(self),

            278 => getrandom(self),

            293 => rseq(self),

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
    Fcvtdwu(Reg, Reg, Reg),
    Fcvtdw(Reg, Reg, Reg),
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

            Fcvtdw(rd, _, rs1) => {
                cpu.float_registers[rd] = (cpu.registers[rs1] as i32 as f64).to_bits();
            }

            Fcvtdwu(rd, _, rs1) => {
                cpu.float_registers[rd] = (cpu.registers[rs1] as u32 as f64).to_bits();
            }

            Fmvxd(rd, rs1) => cpu.registers[rd] = cpu.float_registers[rs1],

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

    #[cfg_attr(not(all(target_arch = "aarch64", unix)), allow(dead_code))]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u64 {
        self.registers.as_mut_ptr()
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
