use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

use crate::reverse::{
    format_reverse_compiled_with_style, reverse_compile_executable, reverse_compile_from,
    ReverseOutputStyle,
};
use riscvm_core::cpu::{RV64GCRegAbiName, RV64GC};
use riscvm_core::jit::{JitExecutionMode, JitOptions};
use riscvm_core::tracer::TraceOptions;

const PC_REGISTER: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub message: String,
    pub should_quit: bool,
}

impl CommandOutput {
    fn message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            should_quit: false,
        }
    }

    fn quit(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            should_quit: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisassembledInstruction {
    pub address: u64,
    pub opcode: u32,
    pub text: String,
    pub len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Breakpoint {
        pc: u64,
    },
    Exited {
        pc: u64,
    },
    Stepped {
        pc: u64,
        steps: u64,
    },
    StepLimit {
        pc: u64,
        steps: u64,
    },
    Target {
        pc: u64,
    },
    Watchpoint {
        id: usize,
        description: String,
        old: u64,
        new: u64,
    },
}

impl StopReason {
    pub fn message(&self) -> String {
        match self {
            Self::Breakpoint { pc } => format!("breakpoint hit at 0x{pc:016x}"),
            Self::Exited { pc } => format!("program exited at pc=0x{pc:016x}"),
            Self::Stepped { pc, steps } => {
                format!("stepped {steps} instruction(s), pc=0x{pc:016x}")
            }
            Self::StepLimit { pc, steps } => {
                format!("step limit reached after {steps} steps at pc=0x{pc:016x}")
            }
            Self::Target { pc } => format!("reached 0x{pc:016x}"),
            Self::Watchpoint {
                id,
                description,
                old,
                new,
            } => {
                format!("watchpoint #{id} hit: {description} changed 0x{old:016x} -> 0x{new:016x}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchTarget {
    Register { register: usize },
    Memory { address: u64, width: MemoryWidth },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Watchpoint {
    pub id: usize,
    pub target: WatchTarget,
    last_value: u64,
}

impl Watchpoint {
    fn description(&self) -> String {
        match self.target {
            WatchTarget::Register { register } => format_register(register),
            WatchTarget::Memory { address, width } => {
                format!("{} @ 0x{address:016x}", width.name())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryWidth {
    Byte,
    Halfword,
    Word,
    Doubleword,
}

impl MemoryWidth {
    fn bytes(self) -> u64 {
        match self {
            Self::Byte => 1,
            Self::Halfword => 2,
            Self::Word => 4,
            Self::Doubleword => 8,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Byte => "byte",
            Self::Halfword => "halfword",
            Self::Word => "word",
            Self::Doubleword => "doubleword",
        }
    }
}

pub struct Debugger {
    cpu: RV64GC,
    breakpoints: BTreeSet<u64>,
    watchpoints: Vec<Watchpoint>,
    next_watchpoint_id: usize,
    mount_root: Option<PathBuf>,
    executable_dir: Option<PathBuf>,
    linux_sysroot: Option<PathBuf>,
    last_stop: Option<StopReason>,
}

impl Debugger {
    pub fn from_cpu(cpu: RV64GC) -> Self {
        Self {
            cpu,
            breakpoints: BTreeSet::new(),
            watchpoints: Vec::new(),
            next_watchpoint_id: 1,
            mount_root: None,
            executable_dir: None,
            linux_sysroot: None,
            last_stop: None,
        }
    }

    pub fn load(path: &str, guest_args: Vec<String>, sysroot: Option<String>) -> io::Result<Self> {
        let executable_path = absolute_executable_path(path);
        let mut cpu = RV64GC::new();
        let mount_root = std::env::current_dir()?;
        let executable_dir = executable_path.parent().map(PathBuf::from);
        let linux_sysroot = sysroot.map(PathBuf::from);

        if let Some(executable_dir) = executable_dir.as_ref() {
            let guest_prefix = executable_dir.to_string_lossy();
            cpu.mount_host_directory_at(&guest_prefix, executable_dir)
                .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{error:?}")))?;
        }
        if let Some(sysroot) = linux_sysroot.as_ref() {
            cpu.set_linux_sysroot(sysroot);
            cpu.mount_host_directory(sysroot)
                .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{error:?}")))?;
        }
        cpu.mount_host_directory(&mount_root)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("{error:?}")))?;
        cpu.set_executable_path(&executable_path);

        let mut argv = vec![executable_path.to_string_lossy().into_owned()];
        argv.extend(guest_args);
        cpu.set_argv(argv);

        let mut buf = Vec::new();
        File::open(path)?.read_to_end(&mut buf)?;
        if path.ends_with(".bin") {
            cpu.load_bin(buf);
        } else {
            cpu.load_elf(buf)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        }

        let mut debugger = Self::from_cpu(cpu);
        debugger.mount_root = Some(mount_root);
        debugger.executable_dir = executable_dir;
        debugger.linux_sysroot = linux_sysroot;
        Ok(debugger)
    }

    pub fn cpu(&self) -> &RV64GC {
        &self.cpu
    }

    pub fn cpu_mut(&mut self) -> &mut RV64GC {
        &mut self.cpu
    }

    pub fn pc(&self) -> u64 {
        self.cpu.registers[PC_REGISTER]
    }

    pub fn breakpoints(&self) -> &BTreeSet<u64> {
        &self.breakpoints
    }

    pub fn watchpoints(&self) -> &[Watchpoint] {
        &self.watchpoints
    }

    pub fn last_stop(&self) -> Option<&StopReason> {
        self.last_stop.as_ref()
    }

    pub fn set_breakpoint(&mut self, address: u64) -> bool {
        self.breakpoints.insert(address)
    }

    pub fn delete_breakpoint(&mut self, address: u64) -> bool {
        self.breakpoints.remove(&address)
    }

    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    pub fn step(&mut self, count: u64) -> StopReason {
        let count = count.max(1);
        for _ in 0..count {
            if self.cpu.should_quit {
                let stop = StopReason::Exited { pc: self.pc() };
                self.last_stop = Some(stop.clone());
                return stop;
            }

            let before = self.watch_values();
            self.cpu.step();

            if let Some(stop) = self.changed_watchpoint(before) {
                self.last_stop = Some(stop.clone());
                return stop;
            }
        }

        let stop = StopReason::Stepped {
            pc: self.pc(),
            steps: count,
        };
        self.last_stop = Some(stop.clone());
        stop
    }

    pub fn continue_execution(&mut self) -> StopReason {
        self.continue_inner(None, None)
    }

    pub fn run_until_pc(&mut self, target: u64) -> StopReason {
        self.continue_inner(Some(target), None)
    }

    pub fn continue_with_limit(&mut self, limit: u64) -> StopReason {
        self.continue_inner(None, Some(limit))
    }

    pub fn disassemble_from(
        &self,
        mut address: u64,
        count: usize,
    ) -> Result<Vec<DisassembledInstruction>, String> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            let opcode = self
                .cpu
                .ram
                .read_word(address)
                .map_err(|error| error.to_string())?;
            let instruction = self.cpu.find_instruction(opcode);
            let len = instruction_len(opcode);
            result.push(DisassembledInstruction {
                address,
                opcode,
                text: instruction.to_string(),
                len,
            });
            address = address.wrapping_add(len);
        }
        Ok(result)
    }

    pub fn execute_command(&mut self, line: &str) -> CommandOutput {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            return CommandOutput::message("empty command");
        }

        match tokens[0] {
            "help" | "h" | "?" => CommandOutput::message(help_text()),
            "quit" | "q" => CommandOutput::quit("quitting debugger"),
            "status" | "st" => CommandOutput::message(self.status()),
            "pc" => CommandOutput::message(format!("pc = 0x{:016x}", self.pc())),
            "reset" | "run" | "r" => match self.reset() {
                Ok(()) => CommandOutput::message(format!("reset pc=0x{:016x}", self.pc())),
                Err(error) => CommandOutput::message(format!("reset failed: {error}")),
            },
            "step" | "si" | "s" => self.command_step(&tokens),
            "continue" | "cont" | "c" => self.command_continue(&tokens),
            "break" | "breakpoint" | "b" => self.command_breakpoint(&tokens),
            "delete" | "d" => self.command_delete(&tokens),
            "clear" => {
                self.clear_breakpoints();
                CommandOutput::message("all breakpoints cleared")
            }
            "breakpoints" | "info" => self.command_info(&tokens),
            "regs" | "registers" => CommandOutput::message(self.format_registers()),
            "reg" | "register" => self.command_register(&tokens),
            "set" => self.command_set(&tokens),
            "x" | "mem" | "memory" => self.command_memory(&tokens),
            "disasm" | "disassemble" | "u" => self.command_disassemble(&tokens),
            "decompile" | "reverse" | "rc" => self.command_reverse_compile(&tokens),
            "watch" | "w" => self.command_watch(&tokens),
            "jit" => self.command_jit(&tokens),
            "profile" => self.command_profile(),
            _ => CommandOutput::message(format!("unknown command: {}", tokens[0])),
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        self.cpu.reset();
        if let Some(executable_dir) = &self.executable_dir {
            let guest_prefix = executable_dir.to_string_lossy();
            self.cpu
                .mount_host_directory_at(&guest_prefix, executable_dir)
                .map_err(|error| format!("{error:?}"))?;
        }
        if let Some(sysroot) = &self.linux_sysroot {
            self.cpu.set_linux_sysroot(sysroot);
            self.cpu
                .mount_host_directory(sysroot)
                .map_err(|error| format!("{error:?}"))?;
        }
        if let Some(root) = &self.mount_root {
            self.cpu
                .mount_host_directory(root)
                .map_err(|error| format!("{error:?}"))?;
        }
        for index in 0..self.watchpoints.len() {
            let value = self.read_watch_target(&self.watchpoints[index].target)?;
            self.watchpoints[index].last_value = value;
        }
        self.last_stop = None;
        Ok(())
    }

    fn continue_inner(&mut self, target: Option<u64>, limit: Option<u64>) -> StopReason {
        let start_pc = self.pc();
        let mut first_iteration = true;
        let mut steps = 0;

        loop {
            let pc = self.pc();
            if self.cpu.should_quit {
                let stop = StopReason::Exited { pc };
                self.last_stop = Some(stop.clone());
                return stop;
            }
            if target == Some(pc) {
                let stop = StopReason::Target { pc };
                self.last_stop = Some(stop.clone());
                return stop;
            }
            if (!first_iteration || pc != start_pc) && self.breakpoints.contains(&pc) {
                let stop = StopReason::Breakpoint { pc };
                self.last_stop = Some(stop.clone());
                return stop;
            }
            if limit.is_some_and(|limit| steps >= limit) {
                let stop = StopReason::StepLimit { pc, steps };
                self.last_stop = Some(stop.clone());
                return stop;
            }

            first_iteration = false;
            let before = self.watch_values();
            self.cpu.step();
            steps += 1;

            if let Some(stop) = self.changed_watchpoint(before) {
                self.last_stop = Some(stop.clone());
                return stop;
            }
        }
    }

    fn command_step(&mut self, tokens: &[&str]) -> CommandOutput {
        let count = match tokens.get(1) {
            Some(value) => match parse_u64(value) {
                Ok(count) => count,
                Err(error) => return CommandOutput::message(error),
            },
            None => 1,
        };
        let stop = self.step(count);
        CommandOutput::message(stop.message())
    }

    fn command_continue(&mut self, tokens: &[&str]) -> CommandOutput {
        let mut target = None;
        let mut limit = None;
        let mut idx = 1;

        while idx < tokens.len() {
            match tokens[idx] {
                "to" => {
                    let Some(value) = tokens.get(idx + 1) else {
                        return CommandOutput::message("continue to requires an address");
                    };
                    target = match self.resolve_address(value) {
                        Ok(address) => Some(address),
                        Err(error) => return CommandOutput::message(error),
                    };
                    idx += 2;
                }
                "limit" => {
                    let Some(value) = tokens.get(idx + 1) else {
                        return CommandOutput::message("continue limit requires a count");
                    };
                    limit = match parse_u64(value) {
                        Ok(limit) => Some(limit),
                        Err(error) => return CommandOutput::message(error),
                    };
                    idx += 2;
                }
                value => {
                    target = match self.resolve_address(value) {
                        Ok(address) => Some(address),
                        Err(error) => return CommandOutput::message(error),
                    };
                    idx += 1;
                }
            }
        }

        let stop = self.continue_inner(target, limit);
        CommandOutput::message(stop.message())
    }

    fn command_breakpoint(&mut self, tokens: &[&str]) -> CommandOutput {
        match tokens.get(1).copied() {
            None | Some("list") | Some("l") => CommandOutput::message(self.format_breakpoints()),
            Some("clear") | Some("c") => {
                self.clear_breakpoints();
                CommandOutput::message("all breakpoints cleared")
            }
            Some("delete") | Some("del") | Some("d") => {
                let Some(value) = tokens.get(2) else {
                    return CommandOutput::message("breakpoint delete requires an address");
                };
                let address = match self.resolve_address(value) {
                    Ok(address) => address,
                    Err(error) => return CommandOutput::message(error),
                };
                if self.delete_breakpoint(address) {
                    CommandOutput::message(format!("breakpoint deleted at 0x{address:016x}"))
                } else {
                    CommandOutput::message(format!("no breakpoint at 0x{address:016x}"))
                }
            }
            Some("set") | Some("s") => {
                let Some(value) = tokens.get(2) else {
                    return CommandOutput::message("breakpoint set requires an address");
                };
                self.set_breakpoint_command(value)
            }
            Some(value) => self.set_breakpoint_command(value),
        }
    }

    fn command_delete(&mut self, tokens: &[&str]) -> CommandOutput {
        let Some(value) = tokens.get(1) else {
            return CommandOutput::message("delete requires an address");
        };
        let address = match self.resolve_address(value) {
            Ok(address) => address,
            Err(error) => return CommandOutput::message(error),
        };
        if self.delete_breakpoint(address) {
            CommandOutput::message(format!("breakpoint deleted at 0x{address:016x}"))
        } else {
            CommandOutput::message(format!("no breakpoint at 0x{address:016x}"))
        }
    }

    fn set_breakpoint_command(&mut self, value: &str) -> CommandOutput {
        let address = match self.resolve_address(value) {
            Ok(address) => address,
            Err(error) => return CommandOutput::message(error),
        };
        if self.set_breakpoint(address) {
            CommandOutput::message(format!("breakpoint set at 0x{address:016x}"))
        } else {
            CommandOutput::message(format!("breakpoint already set at 0x{address:016x}"))
        }
    }

    fn command_info(&self, tokens: &[&str]) -> CommandOutput {
        match tokens.get(1).copied() {
            None | Some("breakpoints") | Some("b") => {
                CommandOutput::message(self.format_breakpoints())
            }
            Some("watch") | Some("watchpoints") => {
                CommandOutput::message(self.format_watchpoints())
            }
            Some("registers") | Some("regs") => CommandOutput::message(self.format_registers()),
            Some(other) => CommandOutput::message(format!("unknown info topic: {other}")),
        }
    }

    fn command_register(&self, tokens: &[&str]) -> CommandOutput {
        let Some(name) = tokens.get(1) else {
            return CommandOutput::message(self.format_registers());
        };
        let register = match parse_register(name) {
            Ok(register) => register,
            Err(error) => return CommandOutput::message(error),
        };
        CommandOutput::message(format!(
            "{} = 0x{:016x}",
            format_register(register),
            self.cpu.registers[register]
        ))
    }

    fn command_set(&mut self, tokens: &[&str]) -> CommandOutput {
        match tokens.get(1).copied() {
            Some("reg") | Some("register") => {
                let (Some(register), Some(value)) = (tokens.get(2), tokens.get(3)) else {
                    return CommandOutput::message("set reg requires a register and value");
                };
                self.set_register_command(register, value)
            }
            Some("mem") | Some("memory") => {
                let (Some(address), Some(value)) = (tokens.get(2), tokens.get(3)) else {
                    return CommandOutput::message("set mem requires an address and value");
                };
                let width = tokens
                    .get(4)
                    .map_or(Ok(MemoryWidth::Doubleword), |value| parse_width(value));
                self.write_memory_command(address, value, width)
            }
            Some(register) if parse_register(register).is_ok() => {
                let Some(value) = tokens.get(2) else {
                    return CommandOutput::message("set requires a value");
                };
                self.set_register_command(register, value)
            }
            Some(other) => CommandOutput::message(format!("unknown set target: {other}")),
            None => CommandOutput::message("set requires a target"),
        }
    }

    fn set_register_command(&mut self, register: &str, value: &str) -> CommandOutput {
        let register = match parse_register(register) {
            Ok(register) => register,
            Err(error) => return CommandOutput::message(error),
        };
        let value = match parse_u64(value) {
            Ok(value) => value,
            Err(error) => return CommandOutput::message(error),
        };
        self.cpu.registers[register] = value;
        if register == 0 {
            self.cpu.registers[0usize] = 0;
        }
        CommandOutput::message(format!("{} = 0x{value:016x}", format_register(register)))
    }

    fn command_memory(&mut self, tokens: &[&str]) -> CommandOutput {
        if tokens[0] == "mem" || tokens[0] == "memory" {
            match tokens.get(1).copied() {
                Some("read") | Some("r") => {
                    let Some(address) = tokens.get(2) else {
                        return CommandOutput::message("mem read requires an address");
                    };
                    let count = tokens.get(3).map_or(Ok(1), |value| parse_usize(value));
                    let width = tokens
                        .get(4)
                        .map_or(Ok(MemoryWidth::Doubleword), |value| parse_width(value));
                    return self.read_memory_command(address, count, width);
                }
                Some("write") | Some("w") => {
                    let (Some(address), Some(value)) = (tokens.get(2), tokens.get(3)) else {
                        return CommandOutput::message("mem write requires an address and value");
                    };
                    let width = tokens
                        .get(4)
                        .map_or(Ok(MemoryWidth::Doubleword), |value| parse_width(value));
                    return self.write_memory_command(address, value, width);
                }
                Some(other) => {
                    return CommandOutput::message(format!("unknown memory operation: {other}"))
                }
                None => return CommandOutput::message("memory requires read or write"),
            }
        }

        let Some(address) = tokens.get(1) else {
            return CommandOutput::message("x requires an address");
        };
        let count = tokens.get(2).map_or(Ok(1), |value| parse_usize(value));
        let width = tokens
            .get(3)
            .map_or(Ok(MemoryWidth::Doubleword), |value| parse_width(value));
        self.read_memory_command(address, count, width)
    }

    fn read_memory_command(
        &self,
        address: &str,
        count: Result<usize, String>,
        width: Result<MemoryWidth, String>,
    ) -> CommandOutput {
        let address = match self.resolve_address(address) {
            Ok(address) => address,
            Err(error) => return CommandOutput::message(error),
        };
        let count = match count {
            Ok(count) => count.max(1),
            Err(error) => return CommandOutput::message(error),
        };
        let width = match width {
            Ok(width) => width,
            Err(error) => return CommandOutput::message(error),
        };

        let mut lines = Vec::with_capacity(count);
        for idx in 0..count {
            let addr = address.wrapping_add((idx as u64) * width.bytes());
            match self.read_memory(addr, width) {
                Ok(value) => lines.push(format!("0x{addr:016x}: 0x{value:016x}")),
                Err(error) => return CommandOutput::message(error),
            }
        }
        CommandOutput::message(lines.join("\n"))
    }

    fn write_memory_command(
        &mut self,
        address: &str,
        value: &str,
        width: Result<MemoryWidth, String>,
    ) -> CommandOutput {
        let address = match self.resolve_address(address) {
            Ok(address) => address,
            Err(error) => return CommandOutput::message(error),
        };
        let value = match parse_u64(value) {
            Ok(value) => value,
            Err(error) => return CommandOutput::message(error),
        };
        let width = match width {
            Ok(width) => width,
            Err(error) => return CommandOutput::message(error),
        };

        match self.write_memory(address, value, width) {
            Ok(()) => CommandOutput::message(format!(
                "{} at 0x{address:016x} = 0x{value:016x}",
                width.name()
            )),
            Err(error) => CommandOutput::message(error),
        }
    }

    fn command_disassemble(&self, tokens: &[&str]) -> CommandOutput {
        let address = match tokens.get(1) {
            Some(value) => match self.resolve_address(value) {
                Ok(address) => address,
                Err(error) => return CommandOutput::message(error),
            },
            None => self.pc(),
        };
        let count = match tokens.get(2).map_or(Ok(10), |value| parse_usize(value)) {
            Ok(count) => count,
            Err(error) => return CommandOutput::message(error),
        };
        match self.disassemble_from(address, count) {
            Ok(items) => CommandOutput::message(
                items
                    .into_iter()
                    .map(|item| {
                        format!(
                            "0x{:016x}: 0x{:08x}  {}",
                            item.address, item.opcode, item.text
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            Err(error) => CommandOutput::message(error),
        }
    }

    fn command_reverse_compile(&self, tokens: &[&str]) -> CommandOutput {
        let (style, args) = reverse_output_style_and_args(tokens);
        match args.first().copied() {
            Some("all") => {
                let limit = match args.get(1).map_or(Ok(256), |value| parse_usize(value)) {
                    Ok(limit) => limit,
                    Err(error) => return CommandOutput::message(error),
                };
                match reverse_compile_executable(&self.cpu, limit) {
                    Ok((instructions, truncated)) => CommandOutput::message(
                        format_reverse_compiled_with_style(&instructions, truncated, style),
                    ),
                    Err(error) => CommandOutput::message(error),
                }
            }
            Some("pc") | Some("current") => {
                let count = match args.get(1).map_or(Ok(16), |value| parse_usize(value)) {
                    Ok(count) => count,
                    Err(error) => return CommandOutput::message(error),
                };
                self.reverse_compile_range(self.pc(), count, style)
            }
            Some(value) => {
                let address = match self.resolve_address(value) {
                    Ok(address) => address,
                    Err(error) => return CommandOutput::message(error),
                };
                let count = match args.get(1).map_or(Ok(16), |value| parse_usize(value)) {
                    Ok(count) => count,
                    Err(error) => return CommandOutput::message(error),
                };
                self.reverse_compile_range(address, count, style)
            }
            None => self.reverse_compile_range(self.pc(), 16, style),
        }
    }

    fn reverse_compile_range(
        &self,
        address: u64,
        count: usize,
        style: ReverseOutputStyle,
    ) -> CommandOutput {
        match reverse_compile_from(&self.cpu, address, count) {
            Ok(instructions) => CommandOutput::message(format_reverse_compiled_with_style(
                &instructions,
                false,
                style,
            )),
            Err(error) => CommandOutput::message(error),
        }
    }

    fn command_watch(&mut self, tokens: &[&str]) -> CommandOutput {
        match tokens.get(1).copied() {
            None | Some("list") | Some("l") => CommandOutput::message(self.format_watchpoints()),
            Some("clear") | Some("c") => {
                self.watchpoints.clear();
                CommandOutput::message("all watchpoints cleared")
            }
            Some("delete") | Some("d") => {
                let Some(value) = tokens.get(2) else {
                    return CommandOutput::message("watch delete requires an id");
                };
                let id = match parse_usize(value) {
                    Ok(id) => id,
                    Err(error) => return CommandOutput::message(error),
                };
                if let Some(index) = self.watchpoints.iter().position(|watch| watch.id == id) {
                    self.watchpoints.remove(index);
                    CommandOutput::message(format!("watchpoint #{id} deleted"))
                } else {
                    CommandOutput::message(format!("no watchpoint #{id}"))
                }
            }
            Some("reg") | Some("register") => {
                let Some(register) = tokens.get(2) else {
                    return CommandOutput::message("watch reg requires a register");
                };
                let register = match parse_register(register) {
                    Ok(register) => register,
                    Err(error) => return CommandOutput::message(error),
                };
                self.add_watchpoint(WatchTarget::Register { register })
            }
            Some("mem") | Some("memory") => {
                let Some(address) = tokens.get(2) else {
                    return CommandOutput::message("watch mem requires an address");
                };
                let address = match self.resolve_address(address) {
                    Ok(address) => address,
                    Err(error) => return CommandOutput::message(error),
                };
                let width = match tokens
                    .get(3)
                    .map_or(Ok(MemoryWidth::Doubleword), |value| parse_width(value))
                {
                    Ok(width) => width,
                    Err(error) => return CommandOutput::message(error),
                };
                self.add_watchpoint(WatchTarget::Memory { address, width })
            }
            Some(other) => CommandOutput::message(format!("unknown watch operation: {other}")),
        }
    }

    fn command_jit(&mut self, tokens: &[&str]) -> CommandOutput {
        let Some(mode) = tokens.get(1).copied() else {
            return CommandOutput::message(
                "jit requires an engine: jit run <hybrid|jit|aot> [profile] [top N] [interval N]",
            );
        };
        if mode != "run" {
            return CommandOutput::message(format!("unknown jit operation: {mode}"));
        }

        let Some(engine) = tokens.get(2).copied() else {
            return CommandOutput::message("jit run requires hybrid, jit, or aot");
        };
        let execution_mode = match engine {
            "hybrid" => JitExecutionMode::Hybrid,
            "jit" => JitExecutionMode::Jit,
            "aot" => JitExecutionMode::Aot,
            _ => return CommandOutput::message(format!("unknown jit engine: {engine}")),
        };

        let mut profile = false;
        let mut top_limit = TraceOptions::default().top_limit;
        let mut profile_interval = None;
        let mut idx = 3;
        while idx < tokens.len() {
            match tokens[idx] {
                "profile" => {
                    profile = true;
                    idx += 1;
                }
                "top" => {
                    let Some(value) = tokens.get(idx + 1) else {
                        return CommandOutput::message("jit run top requires a count");
                    };
                    top_limit = match parse_usize(value) {
                        Ok(value) => value,
                        Err(error) => return CommandOutput::message(error),
                    };
                    profile = true;
                    idx += 2;
                }
                "interval" => {
                    let Some(value) = tokens.get(idx + 1) else {
                        return CommandOutput::message("jit run interval requires a count");
                    };
                    profile_interval = match parse_u64(value) {
                        Ok(value) => Some(value.max(1)),
                        Err(error) => return CommandOutput::message(error),
                    };
                    profile = true;
                    idx += 2;
                }
                other => return CommandOutput::message(format!("unknown jit run option: {other}")),
            }
        }

        if profile {
            self.cpu.set_trace_options(TraceOptions {
                profile: true,
                top_limit,
                profile_interval,
                ..TraceOptions::default()
            });
        }

        let mut options = JitOptions {
            execution_mode,
            ..JitOptions::default()
        };
        if self.linux_sysroot.is_some() {
            options.host_libc_plt_stdio = false;
        }

        match self.cpu.start_jit_with_options(options) {
            Ok(()) => {
                let stop = StopReason::Exited { pc: self.pc() };
                self.last_stop = Some(stop.clone());
                let mut message = format!("jit {engine} completed: {}", stop.message());
                if profile {
                    if let Some(report) = self.cpu.trace_report() {
                        message.push('\n');
                        message.push_str(&report);
                    }
                }
                CommandOutput::message(message)
            }
            Err(error) => CommandOutput::message(format!("jit {engine} failed: {error}")),
        }
    }

    fn command_profile(&self) -> CommandOutput {
        match self.cpu.trace_report() {
            Some(report) => CommandOutput::message(report),
            None => CommandOutput::message("profiling is not enabled"),
        }
    }

    fn add_watchpoint(&mut self, target: WatchTarget) -> CommandOutput {
        let last_value = match self.read_watch_target(&target) {
            Ok(value) => value,
            Err(error) => return CommandOutput::message(error),
        };
        let id = self.next_watchpoint_id;
        self.next_watchpoint_id += 1;
        let watchpoint = Watchpoint {
            id,
            target,
            last_value,
        };
        let description = watchpoint.description();
        self.watchpoints.push(watchpoint);
        CommandOutput::message(format!("watchpoint #{id} set on {description}"))
    }

    fn read_memory(&self, address: u64, width: MemoryWidth) -> Result<u64, String> {
        match width {
            MemoryWidth::Byte => self
                .cpu
                .ram
                .read_byte(address)
                .map(u64::from)
                .map_err(|error| error.to_string()),
            MemoryWidth::Halfword => self
                .cpu
                .ram
                .read_halfword(address)
                .map_err(|error| error.to_string()),
            MemoryWidth::Word => self
                .cpu
                .ram
                .read_word(address)
                .map(u64::from)
                .map_err(|error| error.to_string()),
            MemoryWidth::Doubleword => self
                .cpu
                .ram
                .read_doubleword(address)
                .map_err(|error| error.to_string()),
        }
    }

    fn write_memory(&mut self, address: u64, value: u64, width: MemoryWidth) -> Result<(), String> {
        match width {
            MemoryWidth::Byte => self.cpu.ram.write_byte(address, value as u8),
            MemoryWidth::Halfword => self.cpu.ram.write_halfword(address, value),
            MemoryWidth::Word => self.cpu.ram.write_word(address, value as u32),
            MemoryWidth::Doubleword => self.cpu.ram.write_doubleword(address, value),
        }
        .map_err(|error| error.to_string())
    }

    fn watch_values(&self) -> Vec<Option<u64>> {
        self.watchpoints
            .iter()
            .map(|watchpoint| self.read_watch_target(&watchpoint.target).ok())
            .collect()
    }

    fn changed_watchpoint(&mut self, before: Vec<Option<u64>>) -> Option<StopReason> {
        for idx in 0..self.watchpoints.len() {
            let Ok(new) = self.read_watch_target(&self.watchpoints[idx].target) else {
                continue;
            };
            let old = before
                .get(idx)
                .and_then(|value| *value)
                .unwrap_or(self.watchpoints[idx].last_value);
            self.watchpoints[idx].last_value = new;
            if old != new {
                return Some(StopReason::Watchpoint {
                    id: self.watchpoints[idx].id,
                    description: self.watchpoints[idx].description(),
                    old,
                    new,
                });
            }
        }
        None
    }

    fn read_watch_target(&self, target: &WatchTarget) -> Result<u64, String> {
        match *target {
            WatchTarget::Register { register } => Ok(self.cpu.registers[register]),
            WatchTarget::Memory { address, width } => self.read_memory(address, width),
        }
    }

    fn resolve_address(&self, value: &str) -> Result<u64, String> {
        if let Ok(register) = parse_register(value) {
            return Ok(self.cpu.registers[register]);
        }
        parse_u64(value)
    }

    fn status(&self) -> String {
        let stop = self
            .last_stop
            .as_ref()
            .map_or_else(|| "not stopped".to_string(), StopReason::message);
        format!(
            "pc=0x{:016x}\nbreakpoints={}\nwatchpoints={}\n{}",
            self.pc(),
            self.breakpoints.len(),
            self.watchpoints.len(),
            stop
        )
    }

    fn format_registers(&self) -> String {
        let mut lines = Vec::new();
        for row in 0..8 {
            let mut parts = Vec::new();
            for col in 0..4 {
                let register = row + col * 8;
                parts.push(format!(
                    "{:<4}=0x{:016x}",
                    format_register(register),
                    self.cpu.registers[register]
                ));
            }
            lines.push(parts.join("  "));
        }
        lines.push(format!("pc  =0x{:016x}", self.pc()));
        lines.join("\n")
    }

    fn format_breakpoints(&self) -> String {
        if self.breakpoints.is_empty() {
            return "no breakpoints".to_string();
        }
        self.breakpoints
            .iter()
            .enumerate()
            .map(|(idx, address)| format!("#{idx}: 0x{address:016x}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_watchpoints(&self) -> String {
        if self.watchpoints.is_empty() {
            return "no watchpoints".to_string();
        }
        self.watchpoints
            .iter()
            .map(|watch| {
                format!(
                    "#{}: {} = 0x{:016x}",
                    watch.id,
                    watch.description(),
                    watch.last_value
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub fn absolute_executable_path(path: &str) -> PathBuf {
    let raw_path = PathBuf::from(path);
    if let Ok(canonical) = std::fs::canonicalize(&raw_path) {
        return canonical;
    }
    if raw_path.is_absolute() {
        return raw_path;
    }

    std::env::current_dir()
        .map(|cwd| cwd.join(raw_path))
        .unwrap_or_else(|_| PathBuf::from("/").join(path))
}

pub fn load_debugger_from_path(
    path: &str,
    guest_args: Vec<String>,
    sysroot: Option<String>,
) -> io::Result<Debugger> {
    Debugger::load(path, guest_args, sysroot)
}

fn instruction_len(opcode: u32) -> u64 {
    if opcode & 0b11 == 0b11 {
        4
    } else {
        2
    }
}

fn parse_u64(value: &str) -> Result<u64, String> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|_| format!("invalid hex value: {value}"))
    } else {
        value
            .parse::<u64>()
            .map_err(|_| format!("invalid integer value: {value}"))
    }
}

fn parse_usize(value: &str) -> Result<usize, String> {
    parse_u64(value).and_then(|value| {
        usize::try_from(value).map_err(|_| format!("value is too large: {value}"))
    })
}

fn parse_width(value: &str) -> Result<MemoryWidth, String> {
    match value {
        "b" | "byte" | "1" => Ok(MemoryWidth::Byte),
        "h" | "half" | "halfword" | "2" => Ok(MemoryWidth::Halfword),
        "w" | "word" | "4" => Ok(MemoryWidth::Word),
        "g" | "d" | "double" | "doubleword" | "8" => Ok(MemoryWidth::Doubleword),
        _ => Err(format!("invalid memory width: {value}")),
    }
}

fn reverse_output_style_and_args<'a>(tokens: &'a [&'a str]) -> (ReverseOutputStyle, Vec<&'a str>) {
    let mut style = ReverseOutputStyle::Annotated;
    let mut args = Vec::new();

    for token in tokens.iter().skip(1).copied() {
        match token {
            "c" | "-c" | "--c" | "--style=c" => style = ReverseOutputStyle::C,
            "annotated" | "listing" | "--annotated" | "--style=annotated" => {
                style = ReverseOutputStyle::Annotated
            }
            _ => args.push(token),
        }
    }

    (style, args)
}

fn parse_register(value: &str) -> Result<usize, String> {
    let value = value.trim();
    if value == "pc" {
        return Ok(PC_REGISTER);
    }
    if let Some(raw) = value.strip_prefix('x') {
        let register = raw
            .parse::<usize>()
            .map_err(|_| format!("invalid register: {value}"))?;
        if register < 32 {
            return Ok(register);
        }
        return Err(format!("register out of range: {value}"));
    }

    let register = match value {
        "zero" => 0,
        "ra" => 1,
        "sp" => 2,
        "gp" => 3,
        "tp" => 4,
        "t0" => 5,
        "t1" => 6,
        "t2" => 7,
        "s0" | "fp" => 8,
        "s1" => 9,
        "a0" => 10,
        "a1" => 11,
        "a2" => 12,
        "a3" => 13,
        "a4" => 14,
        "a5" => 15,
        "a6" => 16,
        "a7" => 17,
        "s2" => 18,
        "s3" => 19,
        "s4" => 20,
        "s5" => 21,
        "s6" => 22,
        "s7" => 23,
        "s8" => 24,
        "s9" => 25,
        "s10" => 26,
        "s11" => 27,
        "t3" => 28,
        "t4" => 29,
        "t5" => 30,
        "t6" => 31,
        _ => return Err(format!("unknown register: {value}")),
    };

    Ok(register)
}

fn format_register(register: usize) -> String {
    if register == PC_REGISTER {
        return "pc".to_string();
    }

    let abi = match register {
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
    };
    format!("x{register}/{abi}")
}

fn help_text() -> &'static str {
    "Commands:
  step|s [n]                 step one or n guest instructions
  continue|c [addr]          run until exit, breakpoint, watchpoint, or addr
  continue to <addr>         run until addr
  continue limit <n>         run at most n instructions
  break|b <addr>             set breakpoint
  b list | b delete <addr>   list or delete breakpoints
  clear                      clear all breakpoints
  watch reg <reg>            stop when a register changes
  watch mem <addr> [width]   stop when memory changes
  watch list|delete|clear    manage watchpoints
  regs | reg <reg>           inspect registers
  set reg <reg> <value>      edit register
  x <addr> [count] [width]   read memory
  mem write <addr> <value> [width]
  disasm|u [addr] [count]    disassemble guest instructions
  decompile|reverse [addr] [count]
                             reverse compile guest code to pseudo-C
  decompile c [addr] [count] emit C-style output with declarations
  decompile all [limit]      reverse compile executable regions
  decompile c all [limit]    reverse compile executable regions as C
  jit run <hybrid|jit|aot> [profile] [top N] [interval N]
                             run a JIT-family engine and optionally print a profile
  profile                    print the current trace/profile report
  status | reset | quit"
}

#[allow(dead_code)]
fn _assert_register_enum_layout() {
    let _ = RV64GCRegAbiName::Pc as usize == PC_REGISTER;
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscvm_core::ram::MemoryRegion;

    fn rv64_word(value: u32) -> [u8; 4] {
        value.to_le_bytes()
    }

    fn test_debugger(words: &[u32]) -> Debugger {
        let mut bin = Vec::new();
        for word in words {
            bin.extend(rv64_word(*word));
        }
        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        Debugger::from_cpu(cpu)
    }

    #[test]
    fn step_command_advances_guest_instructions() {
        let mut debugger = test_debugger(&[
            0x0010_0093, // addi x1, x0, 1
            0x0010_8093, // addi x1, x1, 1
        ]);

        let output = debugger.execute_command("step 2");

        assert!(output.message.contains("stepped 2 instruction"));
        assert!(output.message.contains("pc=0x0000000000000008"));
        assert_eq!(debugger.cpu().registers[1usize], 2);
        assert_eq!(debugger.pc(), 8);
    }

    #[test]
    fn continue_stops_before_executing_breakpoint_address() {
        let mut debugger = test_debugger(&[
            0x0010_0093, // addi x1, x0, 1
            0x0010_8093, // addi x1, x1, 1
        ]);
        debugger.execute_command("break 0x4");

        let output = debugger.execute_command("continue");

        assert!(output.message.contains("breakpoint hit"));
        assert_eq!(debugger.pc(), 4);
        assert_eq!(debugger.cpu().registers[1usize], 1);
    }

    #[test]
    fn continue_without_breakpoints_can_run_to_exit() {
        let mut debugger = test_debugger(&[
            0x0000_0513, // addi a0, x0, 0
            0x05d0_0893, // addi a7, x0, 93
            0x0000_0073, // ecall
        ]);

        let output = debugger.execute_command("continue");

        assert!(output.message.contains("program exited"));
        assert!(debugger.cpu().should_quit);
    }

    #[test]
    fn register_commands_read_and_write_abi_names() {
        let mut debugger = test_debugger(&[]);

        let output = debugger.execute_command("set reg a0 0x2a");
        assert!(output.message.contains("0x000000000000002a"));

        let output = debugger.execute_command("reg x10");
        assert!(output.message.contains("0x000000000000002a"));
    }

    #[test]
    fn memory_commands_read_and_write_sized_values() {
        let mut cpu = RV64GC::new();
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 0x100, vec![0; 0x100]))
            .unwrap();
        let mut debugger = Debugger::from_cpu(cpu);

        let output = debugger.execute_command("mem write 0x1008 0xaabbccdd 4");
        assert!(output.message.contains("word"));

        let output = debugger.execute_command("x 0x1008 1 4");
        assert!(output.message.contains("0x00000000aabbccdd"));
    }

    #[test]
    fn disassemble_command_reports_addresses_and_text() {
        let mut debugger = test_debugger(&[0x0010_0093]);

        let output = debugger.execute_command("disasm 0 1");

        assert!(output.message.contains("0x0000000000000000"));
        assert!(output.message.contains("addi"));
    }

    #[test]
    fn reverse_compile_command_reports_pseudocode_labels_and_source() {
        let mut debugger = test_debugger(&[
            0x0010_0093, // addi x1, x0, 1
            0x0080_006f, // jal x0, 8
            0x0020_0113, // addi x2, x0, 2
            0x0000_0073, // ecall
        ]);

        let output = debugger.execute_command("decompile 0 4");

        assert!(output.message.contains("void guest_code_0000000000000000"));
        assert!(output.message.contains("ra = 0x1;"));
        assert!(output.message.contains("goto L_000000000000000c;"));
        assert!(output.message.contains("L_000000000000000c:"));
        assert!(output.message.contains("syscall(a7);"));
        assert!(output.message.contains("// 0x0000000000000004:"));
    }

    #[test]
    fn reverse_compile_all_uses_executable_binary_regions() {
        let mut debugger = test_debugger(&[
            0x0010_0093, // addi x1, x0, 1
            0x0020_0113, // addi x2, x0, 2
        ]);

        let output = debugger.execute_command("reverse all 1");

        assert!(output.message.contains("void guest_code_0000000000000000"));
        assert!(output.message.contains("ra = 0x1;"));
        assert!(output.message.contains("truncated"));
    }

    #[test]
    fn reverse_compile_c_mode_emits_c_prelude_and_memory_macros() {
        let mut debugger = test_debugger(&[
            0x0010_0093, // addi x1, x0, 1
            0x0011_3423, // sd x1, 8(x2)
            0x0000_0073, // ecall
        ]);

        let output = debugger.execute_command("decompile c 0 3");

        assert!(output.message.contains("#include <stdint.h>"));
        assert!(output.message.contains("extern uint8_t guest_memory[];"));
        assert!(output.message.contains("uint64_t zero = 0"));
        assert!(output.message.contains("ra = 0x1;"));
        assert!(output.message.contains("MEM_U64(sp + 8) = ra;"));
        assert!(output.message.contains("syscall(a7);"));
    }

    #[test]
    fn register_watchpoint_stops_on_change() {
        let mut debugger = test_debugger(&[
            0x0010_0093, // addi x1, x0, 1
            0x0010_8093, // addi x1, x1, 1
        ]);
        debugger.execute_command("watch reg x1");

        let output = debugger.execute_command("continue");

        assert!(output.message.contains("watchpoint #1 hit"));
        assert_eq!(debugger.pc(), 4);
        assert_eq!(debugger.cpu().registers[1usize], 1);
    }

    #[test]
    fn breakpoint_management_is_idempotent_and_sorted() {
        let mut debugger = test_debugger(&[]);

        assert!(debugger.execute_command("b 0x20").message.contains("set"));
        assert!(debugger
            .execute_command("break set 0x10")
            .message
            .contains("set"));
        assert!(debugger
            .execute_command("break 0x10")
            .message
            .contains("already"));

        let output = debugger.execute_command("break list");
        let first = output.message.find("0x0000000000000010").unwrap();
        let second = output.message.find("0x0000000000000020").unwrap();
        assert!(first < second);
    }
}
