use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;
#[cfg(unix)]
use std::os::raw::{c_char, c_int, c_void};
#[cfg(unix)]
use std::ptr;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::cpu::RV64GCRegAbiName::*;
use crate::cpu::{RV64GCInstruction, RV64GC};
use crate::debug;
use crate::tracer::{ExecutionEngine, InstructionTrace};
use crate::{sign_extend, sign_extend12};

#[cfg(all(target_arch = "aarch64", unix))]
mod aarch64;
#[cfg_attr(not(all(target_arch = "aarch64", unix)), allow(dead_code))]
mod background;
mod optimizer;
mod profile;
#[cfg_attr(not(all(target_arch = "aarch64", unix)), allow(dead_code))]
mod runtime;
mod tier;
mod trace;

use background::{BackgroundCompileResult, BackgroundCompiler, BackgroundEnqueueError};
use optimizer::{optimize_plan, OptimizationReport};
pub use profile::{
    format_startup_profile, normalize_startup_profile, parse_startup_profile,
    JitStartupProfileEntry, JitStartupProfileParseError,
};
use tier::{JitTier, DEFAULT_HOT_THRESHOLD};

#[cfg(all(target_arch = "aarch64", unix))]
pub(crate) use runtime::{
    jit_runtime_atomic, jit_runtime_binary, jit_runtime_csr, jit_runtime_direct_write_ptr,
    jit_runtime_ecall, jit_runtime_float_load, jit_runtime_float_op, jit_runtime_float_store,
    jit_runtime_load_i16, jit_runtime_load_i32, jit_runtime_load_i8, jit_runtime_load_u16,
    jit_runtime_load_u32, jit_runtime_load_u64, jit_runtime_load_u8, jit_runtime_store_u16,
    jit_runtime_store_u32, jit_runtime_store_u64, jit_runtime_store_u8, jit_runtime_trap,
    jit_runtime_try_direct_byte_copy, jit_runtime_try_direct_read_ptr,
    jit_runtime_try_direct_write_ptr,
};
pub(crate) use runtime::{
    MemoryWidth, RuntimeAtomicOp, RuntimeBinaryOp, RuntimeCsrOp, RuntimeFloatOp, RuntimeTrapOp,
};

const MAX_BLOCK_INSTRUCTIONS: usize = 64;
const MAX_CACHED_BLOCK_CHAIN: usize = 1024;
const DISPATCH_CACHE_ENTRIES: usize = 4096;
const CANDIDATE_BYTE_SCAN_CACHE_ENTRIES: usize = 1024;
const BASELINE_OPTIMIZER_MIN_INSTRUCTIONS: usize = MAX_BLOCK_INSTRUCTIONS + 1;
const MAX_REGALLOC_GUEST_REGISTERS: usize = 16;
const MAX_HOST_LIBC_BYTES: usize = 16 * 1024 * 1024;
const MAX_HOST_LIBC_C_STRING: usize = 1024 * 1024;
const HOST_LIBC_EXIT_TRAMPOLINE: u64 = 0xffff_ffff_ff00_0000;

type FastU64Hasher = BuildHasherDefault<U64IdentityHasher>;
type FastU64Map<V> = HashMap<u64, V, FastU64Hasher>;
type FastU64Set = HashSet<u64, FastU64Hasher>;

#[derive(Clone, Copy)]
struct DispatchCacheEntry {
    pc: u64,
    block: *mut CompiledBlock,
}

#[derive(Clone, Copy)]
struct CandidateByteScanCacheEntry {
    pc: u64,
    code_version: u64,
    result: CandidateByteScanCacheResult,
}

#[derive(Clone, Copy)]
enum CandidateByteScanCacheResult {
    Empty,
    NotCandidate,
    Candidate(CandidateByteScanLoop),
}

#[derive(Clone, Copy)]
struct CandidateByteScanLoop {
    index_register: u8,
    shadow_register: u8,
    compare_offset_register: u8,
    limit_register: u8,
    exhausted_pc: u64,
    table_base_register: u8,
    entry_address_register: u8,
    candidate_offset_register: u8,
    candidate_base_register: u8,
    candidate_pointer_register: u8,
    lhs_address_register: u8,
    rhs_address_register: u8,
    rhs_base_register: u8,
    lhs_value_register: u8,
    rhs_value_register: u8,
    match_pc: u64,
}

impl DispatchCacheEntry {
    const fn empty() -> Self {
        Self {
            pc: 0,
            block: std::ptr::null_mut(),
        }
    }
}

impl CandidateByteScanCacheEntry {
    const fn empty() -> Self {
        Self {
            pc: 0,
            code_version: u64::MAX,
            result: CandidateByteScanCacheResult::Empty,
        }
    }

    fn new(pc: u64, code_version: u64, pattern: Option<CandidateByteScanLoop>) -> Self {
        let result = match pattern {
            Some(pattern) => CandidateByteScanCacheResult::Candidate(pattern),
            None => CandidateByteScanCacheResult::NotCandidate,
        };
        Self {
            pc,
            code_version,
            result,
        }
    }
}

#[derive(Default)]
struct U64IdentityHasher {
    value: u64,
}

impl Hasher for U64IdentityHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            value ^= u64::from(*byte);
            value = value.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.value = value;
    }

    fn write_u64(&mut self, value: u64) {
        self.value = value;
    }

    fn write_usize(&mut self, value: usize) {
        self.value = value as u64;
    }

    fn finish(&self) -> u64 {
        self.value
    }
}

#[cfg(all(target_arch = "aarch64", unix))]
type NativeBackend = aarch64::AArch64Backend;

#[cfg(not(all(target_arch = "aarch64", unix)))]
struct NativeBackend;

#[cfg(not(all(target_arch = "aarch64", unix)))]
impl NativeBackend {
    #[allow(dead_code)]
    fn new() -> Self {
        Self
    }

    fn compile(
        &mut self,
        _plan: &BlockPlan,
        _tier: JitTier,
        _include_listing: bool,
    ) -> Result<CompiledBlock, JitError> {
        Err(JitError::UnsupportedHost)
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn memcmp(lhs: *const c_void, rhs: *const c_void, count: usize) -> c_int;
    fn memcpy(dest: *mut c_void, src: *const c_void, count: usize) -> *mut c_void;
    fn memmove(dest: *mut c_void, src: *const c_void, count: usize) -> *mut c_void;
    fn memset(dest: *mut c_void, value: c_int, count: usize) -> *mut c_void;
    fn strcmp(lhs: *const c_char, rhs: *const c_char) -> c_int;
    fn strlen(value: *const c_char) -> usize;
    fn strncmp(lhs: *const c_char, rhs: *const c_char, count: usize) -> c_int;
}

#[derive(Debug)]
pub enum JitError {
    UnsupportedHost,
    AllocationFailed,
    PermissionFailed,
    BackgroundCompilerStopped,
    BackgroundCompilerPoisoned,
    BackgroundThreadSpawnFailed { reason: String },
    AotMiss { pc: u64 },
    AotUnsupportedBlock { pc: u64 },
    BlockPlanningFailed { pc: u64, reason: String },
    CompiledBlockMissing { pc: u64 },
    InterpreterFallbackDisabled { pc: u64 },
    HostLibcFailed,
    RuntimeFault { pc: u64, reason: String },
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedHost => write!(f, "JIT is only available on Unix AArch64 hosts"),
            Self::AllocationFailed => write!(f, "failed to allocate executable JIT memory"),
            Self::PermissionFailed => write!(f, "failed to mark JIT memory executable"),
            Self::BackgroundCompilerStopped => write!(f, "background JIT compiler stopped"),
            Self::BackgroundCompilerPoisoned => {
                write!(f, "background JIT compiler queue mutex was poisoned")
            }
            Self::BackgroundThreadSpawnFailed { reason } => {
                write!(f, "failed to spawn background JIT compiler thread: {reason}")
            }
            Self::AotMiss { pc } => write!(
                f,
                "AOT execution reached pc=0x{pc:016x}, which was not precompiled"
            ),
            Self::AotUnsupportedBlock { pc } => write!(
                f,
                "AOT cannot compile the block at pc=0x{pc:016x} without interpreter fallback"
            ),
            Self::BlockPlanningFailed { pc, reason } => {
                write!(
                    f,
                    "JIT could not create a native block at pc=0x{pc:016x}: {reason}"
                )
            }
            Self::CompiledBlockMissing { pc } => {
                write!(f, "JIT compiled block missing for pc=0x{pc:016x}")
            }
            Self::InterpreterFallbackDisabled { pc } => write!(
                f,
                "JIT block at pc=0x{pc:016x} requires interpreter fallback, but fallback is disabled"
            ),
            Self::HostLibcFailed => write!(f, "host libc call failed"),
            Self::RuntimeFault { pc, reason } => {
                write!(f, "JIT runtime fault in block pc=0x{pc:016x}: {reason}")
            }
        }
    }
}

impl std::error::Error for JitError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostLibcFunction {
    Exit,
    LibcStartMain,
    Memcmp,
    Memcpy,
    Memmove,
    Memset,
    Puts,
    Printf,
    Strcmp,
    Strlen,
    Strncmp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitStep {
    Native { pc: u64, instructions: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitExecutionMode {
    Jit,
    Hybrid,
    Aot,
}

impl JitExecutionMode {
    fn trace_engine(self) -> ExecutionEngine {
        match self {
            Self::Jit => ExecutionEngine::Jit,
            Self::Hybrid => ExecutionEngine::Hybrid,
            Self::Aot => ExecutionEngine::Aot,
        }
    }

    fn precompiles_before_execution(self) -> bool {
        matches!(self, Self::Aot)
    }

    fn allows_runtime_compilation(self) -> bool {
        matches!(self, Self::Jit | Self::Hybrid)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitOptions {
    pub execution_mode: JitExecutionMode,
    pub host_libc: bool,
    pub host_libc_plt_stdio: bool,
    pub libc_start_main_shortcut: bool,
    pub debug_log: bool,
    pub dump_instructions: bool,
    pub dynamic_recompilation: bool,
    pub trace_compilation: bool,
    pub hot_threshold: u64,
    pub tier_budgeting: bool,
    pub non_loop_hot_threshold_multiplier: u64,
    pub min_optimized_block_instructions: usize,
    pub background_compilation: bool,
    pub compiler_threads: usize,
    pub compile_queue_limit: usize,
    pub aot_compile_misses: bool,
    pub aot_symbol_entries: bool,
    pub aot_linear_sweep: bool,
}

impl Default for JitOptions {
    fn default() -> Self {
        Self {
            execution_mode: JitExecutionMode::Hybrid,
            host_libc: true,
            host_libc_plt_stdio: true,
            libc_start_main_shortcut: false,
            debug_log: false,
            dump_instructions: false,
            dynamic_recompilation: true,
            trace_compilation: true,
            hot_threshold: DEFAULT_HOT_THRESHOLD,
            tier_budgeting: true,
            non_loop_hot_threshold_multiplier: 4096,
            min_optimized_block_instructions: 2,
            background_compilation: false,
            compiler_threads: 1,
            compile_queue_limit: 64,
            aot_compile_misses: true,
            aot_symbol_entries: false,
            aot_linear_sweep: false,
        }
    }
}

pub struct JitEngine {
    backend: NativeBackend,
    cache: FastU64Map<Box<CompiledBlock>>,
    dispatch_cache: Vec<DispatchCacheEntry>,
    candidate_byte_scan_cache: Vec<CandidateByteScanCacheEntry>,
    pending_optimized_compiles: FastU64Set,
    background_compiler: Option<BackgroundCompiler>,
    options: JitOptions,
    precompiled_entry: bool,
    precompiled_startup_profile: bool,
    startup_profile: Vec<JitStartupProfileEntry>,
    host_libc_symbols: FastU64Map<HostLibcFunction>,
    host_libc_symbol_range: Option<(u64, u64)>,
    host_libc_initialized: bool,
}

impl JitEngine {
    #[cfg(all(target_arch = "aarch64", unix))]
    pub fn new() -> Result<Self, JitError> {
        Self::with_options(JitOptions::default())
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    pub fn with_options(options: JitOptions) -> Result<Self, JitError> {
        Self::with_startup_profile(options, [])
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    pub fn with_startup_profile<I>(
        mut options: JitOptions,
        startup_profile: I,
    ) -> Result<Self, JitError>
    where
        I: IntoIterator<Item = JitStartupProfileEntry>,
    {
        options.hot_threshold = options.hot_threshold.max(1);
        options.non_loop_hot_threshold_multiplier =
            options.non_loop_hot_threshold_multiplier.max(1);
        options.min_optimized_block_instructions = options.min_optimized_block_instructions.max(1);
        options.compiler_threads = options.compiler_threads.max(1);
        options.compile_queue_limit = options.compile_queue_limit.max(1);
        if options.execution_mode == JitExecutionMode::Aot {
            options.dynamic_recompilation = false;
            options.background_compilation = false;
        }
        let background_compiler = if options.background_compilation && options.dynamic_recompilation
        {
            Some(BackgroundCompiler::new(
                options.compiler_threads,
                options.compile_queue_limit,
            )?)
        } else {
            None
        };
        Ok(Self {
            backend: NativeBackend::new(),
            cache: FastU64Map::default(),
            dispatch_cache: vec![DispatchCacheEntry::empty(); DISPATCH_CACHE_ENTRIES],
            candidate_byte_scan_cache: vec![
                CandidateByteScanCacheEntry::empty();
                CANDIDATE_BYTE_SCAN_CACHE_ENTRIES
            ],
            pending_optimized_compiles: FastU64Set::default(),
            background_compiler,
            options,
            precompiled_entry: false,
            precompiled_startup_profile: false,
            startup_profile: normalize_startup_profile(startup_profile),
            host_libc_symbols: FastU64Map::default(),
            host_libc_symbol_range: None,
            host_libc_initialized: false,
        })
    }

    #[cfg(not(all(target_arch = "aarch64", unix)))]
    pub fn new() -> Result<Self, JitError> {
        Err(JitError::UnsupportedHost)
    }

    #[cfg(not(all(target_arch = "aarch64", unix)))]
    pub fn with_options(_options: JitOptions) -> Result<Self, JitError> {
        Err(JitError::UnsupportedHost)
    }

    #[cfg(not(all(target_arch = "aarch64", unix)))]
    pub fn with_startup_profile<I>(
        _options: JitOptions,
        _startup_profile: I,
    ) -> Result<Self, JitError>
    where
        I: IntoIterator<Item = JitStartupProfileEntry>,
    {
        Err(JitError::UnsupportedHost)
    }

    pub fn run(&mut self, cpu: &mut RV64GC) -> Result<(), JitError> {
        cpu.trace_start(self.options.execution_mode.trace_engine());
        self.ensure_precompiled(cpu)?;
        self.log(format_args!(
            "run start pc=0x{:016x} cache_entries={}",
            cpu.registers[Pc],
            self.cache.len()
        ));
        while !cpu.should_quit {
            if debug::termination_requested() {
                cpu.should_quit = true;
                break;
            }
            self.step_precompiled(cpu)?;
            self.execute_cached_block_chain(cpu)?;
            crate::syscalls::run_ready_synthetic_threads(cpu);
            if let Some(reason) = cpu.take_jit_runtime_fault() {
                return Err(JitError::RuntimeFault {
                    pc: cpu.registers[Pc],
                    reason,
                });
            }
            assert_eq!(cpu.registers[Zero], 0);
        }
        self.log(format_args!(
            "run stop pc=0x{:016x} cache_entries={}",
            cpu.registers[Pc],
            self.cache.len()
        ));
        cpu.trace_finish();

        Ok(())
    }

    pub fn step(&mut self, cpu: &mut RV64GC) -> Result<JitStep, JitError> {
        self.ensure_precompiled(cpu)?;
        self.step_precompiled(cpu)
    }

    fn step_precompiled(&mut self, cpu: &mut RV64GC) -> Result<JitStep, JitError> {
        self.drain_background_compiler(cpu)?;

        let pc = cpu.registers[Pc];
        if pc == HOST_LIBC_EXIT_TRAMPOLINE {
            cpu.should_quit = true;
            self.log(format_args!(
                "host libc exit trampoline status={}",
                cpu.registers[A0]
            ));
            return Ok(JitStep::Native {
                pc,
                instructions: 1,
            });
        }
        if let Some(instructions) = self.try_candidate_byte_scan_loop(cpu, pc) {
            return Ok(JitStep::Native { pc, instructions });
        }
        if self.try_host_libc(cpu, pc)? {
            return Ok(JitStep::Native {
                pc,
                instructions: 1,
            });
        }

        let options = self.options;
        let compile_decision = if let Some(block) = self.cache.get_mut(&pc) {
            if block.is_stale(cpu) {
                if options.execution_mode.allows_runtime_compilation() {
                    CompileDecision::Synchronous {
                        tier: JitTier::Baseline,
                        reason: "stale",
                        preserved_execution_count: 0,
                    }
                } else if options.execution_mode == JitExecutionMode::Aot
                    && options.aot_compile_misses
                {
                    CompileDecision::AotRuntime {
                        reason: "aot-stale",
                    }
                } else {
                    CompileDecision::AotMiss
                }
            } else if options.dynamic_recompilation
                && block.tier == JitTier::Baseline
                && block.execution_count >= options.hot_threshold
                && block.execution_count >= block.next_optimized_attempt_count
            {
                match optimized_tier_decision(block, options) {
                    OptimizedTierDecision::Ready { reason } => {
                        let tier = if options.trace_compilation {
                            JitTier::Trace
                        } else {
                            JitTier::Optimized
                        };
                        if self.background_compiler.is_some() {
                            CompileDecision::BackgroundOptimized {
                                tier,
                                reason,
                                preserved_execution_count: block.execution_count,
                            }
                        } else {
                            CompileDecision::Synchronous {
                                tier,
                                reason,
                                preserved_execution_count: block.execution_count,
                            }
                        }
                    }
                    OptimizedTierDecision::RetryAt { execution_count } => {
                        block.next_optimized_attempt_count = execution_count;
                        CompileDecision::None
                    }
                }
            } else {
                CompileDecision::None
            }
        } else {
            if options.execution_mode.allows_runtime_compilation() {
                CompileDecision::Synchronous {
                    tier: JitTier::Baseline,
                    reason: "cold",
                    preserved_execution_count: 0,
                }
            } else if options.execution_mode == JitExecutionMode::Aot && options.aot_compile_misses
            {
                CompileDecision::AotRuntime { reason: "aot-miss" }
            } else {
                CompileDecision::AotMiss
            }
        };

        let cache_hit = match compile_decision {
            CompileDecision::AotMiss => return Err(JitError::AotMiss { pc }),
            CompileDecision::AotRuntime { reason } => {
                let outcome = self.compile_aot_runtime_block(cpu, pc, reason)?;
                outcome == CompileOutcome::PromotedWithoutCompile
            }
            CompileDecision::Synchronous {
                tier,
                reason,
                preserved_execution_count,
            } => {
                let outcome =
                    self.compile_block(cpu, pc, tier, reason, preserved_execution_count)?;
                outcome == CompileOutcome::PromotedWithoutCompile
            }
            CompileDecision::BackgroundOptimized {
                tier,
                reason,
                preserved_execution_count,
            } => {
                self.queue_background_optimization(
                    cpu,
                    pc,
                    tier,
                    reason,
                    preserved_execution_count,
                )?;
                true
            }
            CompileDecision::None => true,
        };

        if cache_hit {
            if let Some(tracer) = cpu.tracer_mut() {
                tracer.record_jit_cache_hit();
            }
        }

        let options = self.options;
        let Some(block) = self.dispatch_block_mut(pc) else {
            return Err(JitError::CompiledBlockMissing { pc });
        };
        execute_compiled_block(options, cpu, pc, block)
    }

    fn execute_cached_block_chain(&mut self, cpu: &mut RV64GC) -> Result<(), JitError> {
        for _ in 0..MAX_CACHED_BLOCK_CHAIN {
            if cpu.should_quit || debug::termination_requested() {
                if debug::termination_requested() {
                    cpu.should_quit = true;
                }
                return Ok(());
            }

            let pc = cpu.registers[Pc];
            if pc == HOST_LIBC_EXIT_TRAMPOLINE || self.pc_may_be_host_libc(pc) {
                return Ok(());
            }
            if self.try_candidate_byte_scan_loop(cpu, pc).is_some() {
                continue;
            }

            let options = self.options;
            let Some(block) = self.dispatch_block_mut(pc) else {
                return Ok(());
            };
            if block.is_stale(cpu) || block_needs_compile_check(block, options) {
                return Ok(());
            }

            if options.debug_log || cpu.tracer_enabled() {
                execute_compiled_block(options, cpu, pc, block)?;
            } else {
                execute_compiled_block_fast(cpu, pc, block)?;
            }
        }

        Ok(())
    }

    fn dispatch_block_mut(&mut self, pc: u64) -> Option<&mut CompiledBlock> {
        let slot = dispatch_cache_slot(pc);
        let entry = self.dispatch_cache[slot];
        if entry.pc == pc && !entry.block.is_null() {
            // Compiled blocks live behind stable Boxes. Entries are cleared whenever a
            // block is replaced, so a matching pointer is valid for this engine.
            return Some(unsafe { &mut *entry.block });
        }

        let block = self.cache.get_mut(&pc)?;
        let block = block.as_mut();
        self.dispatch_cache[slot] = DispatchCacheEntry {
            pc,
            block: block as *mut CompiledBlock,
        };
        Some(block)
    }

    fn invalidate_dispatch_cache(&mut self) {
        self.dispatch_cache.fill(DispatchCacheEntry::empty());
    }

    fn pc_may_be_host_libc(&self, pc: u64) -> bool {
        if !self.options.host_libc {
            return false;
        }
        self.host_libc_symbol_range
            .is_some_and(|(min_pc, max_pc)| pc >= min_pc && pc <= max_pc)
    }

    fn try_candidate_byte_scan_loop(&mut self, cpu: &mut RV64GC, pc: u64) -> Option<u64> {
        let slot = candidate_byte_scan_cache_slot(pc);
        let code_version = cpu.ram.code_version();
        let entry = self.candidate_byte_scan_cache[slot];
        let pattern = if entry.pc == pc && entry.code_version == code_version {
            match entry.result {
                CandidateByteScanCacheResult::Empty
                | CandidateByteScanCacheResult::NotCandidate => None,
                CandidateByteScanCacheResult::Candidate(pattern) => Some(pattern),
            }
        } else {
            let pattern = candidate_byte_scan_loop_from_cpu(cpu, pc);
            self.candidate_byte_scan_cache[slot] =
                CandidateByteScanCacheEntry::new(pc, code_version, pattern);
            pattern
        }?;

        execute_candidate_byte_scan_loop(cpu, pattern)
    }

    fn ensure_precompiled(&mut self, cpu: &mut RV64GC) -> Result<(), JitError> {
        self.ensure_host_libc_symbols(cpu);
        self.ensure_startup_profile_precompiled(cpu)?;
        if self.precompiled_entry || !self.options.execution_mode.precompiles_before_execution() {
            return Ok(());
        }

        let entry_pc = cpu.registers[Pc];
        let compiled_blocks = self.precompile_reachable_blocks(cpu, entry_pc)?;
        self.precompiled_entry = true;
        self.log(format_args!(
            "aot precompile complete entry=0x{entry_pc:016x} blocks={compiled_blocks} cache_entries={}",
            self.cache.len()
        ));
        Ok(())
    }

    fn ensure_startup_profile_precompiled(&mut self, cpu: &mut RV64GC) -> Result<(), JitError> {
        if self.precompiled_startup_profile || self.startup_profile.is_empty() {
            return Ok(());
        }

        let entries = self.startup_profile.clone();
        let mut compiled_blocks = 0usize;
        for entry in entries {
            if self.cache.contains_key(&entry.pc) {
                continue;
            }

            let (planned, tier) = startup_profile_plan_for_tier(cpu, entry.pc, self.options);
            let Some(plan) = planned.plan else {
                self.log(format_args!(
                    "skip startup-profile pc=0x{:016x} count={} cause={}",
                    entry.pc, entry.count, planned.stop
                ));
                continue;
            };

            if !plan.can_run_without_interpreter_fallback() {
                self.log(format_args!(
                    "skip startup-profile pc=0x{:016x} count={} cause=interpreter-fallback",
                    entry.pc, entry.count
                ));
                continue;
            }

            self.compile_plan(cpu, entry.pc, plan, tier, "startup-profile", 0)?;
            compiled_blocks += 1;
        }

        self.precompiled_startup_profile = true;
        self.log(format_args!(
            "startup profile precompile complete blocks={compiled_blocks} cache_entries={}",
            self.cache.len()
        ));
        Ok(())
    }

    fn ensure_host_libc_symbols(&mut self, cpu: &RV64GC) {
        if self.host_libc_initialized {
            return;
        }
        self.host_libc_initialized = true;
        if !self.options.host_libc {
            return;
        }

        for (address, name) in cpu.elf_symbol_names() {
            let function = if self.options.libc_start_main_shortcut {
                host_libc_start_main_shortcut_for_defined_symbol(&name)
                    .or_else(|| host_libc_function_for_defined_symbol(&name))
            } else {
                host_libc_function_for_defined_symbol(&name)
            };
            let Some(function) = function else {
                continue;
            };
            self.insert_host_libc_symbol(address, function);
        }
        for (address, name) in cpu.elf_plt_symbol_names() {
            let Some(function) =
                host_libc_function_for_plt_name(&name, self.options.host_libc_plt_stdio)
            else {
                continue;
            };
            self.insert_host_libc_symbol(address, function);
        }
    }

    fn insert_host_libc_symbol(&mut self, address: u64, function: HostLibcFunction) {
        self.host_libc_symbols.entry(address).or_insert(function);
        self.host_libc_symbol_range = Some(
            self.host_libc_symbol_range
                .map_or((address, address), |(min, max)| {
                    (min.min(address), max.max(address))
                }),
        );
    }

    fn try_host_libc(&mut self, cpu: &mut RV64GC, pc: u64) -> Result<bool, JitError> {
        if !self.options.host_libc {
            return Ok(false);
        }
        if !self.host_libc_initialized {
            self.ensure_host_libc_symbols(cpu);
        }
        let Some((min_pc, max_pc)) = self.host_libc_symbol_range else {
            return Ok(false);
        };
        if pc < min_pc || pc > max_pc {
            return Ok(false);
        }

        let Some(function) = self.host_libc_symbols.get(&pc).copied() else {
            return Ok(false);
        };
        if !host_libc_call_has_return_address(cpu, pc) {
            return Ok(false);
        }

        self.log(format_args!(
            "host libc call pc=0x{pc:016x} function={function:?} ra=0x{:016x}",
            cpu.registers[Ra]
        ));
        let handled = match function {
            HostLibcFunction::Exit => execute_host_exit(cpu),
            HostLibcFunction::LibcStartMain => execute_host_libc_start_main(cpu),
            HostLibcFunction::Memcmp => execute_host_memcmp(cpu),
            HostLibcFunction::Memcpy => execute_host_memcpy(cpu),
            HostLibcFunction::Memmove => execute_host_memmove(cpu),
            HostLibcFunction::Memset => execute_host_memset(cpu),
            HostLibcFunction::Puts => execute_host_puts(cpu),
            HostLibcFunction::Printf => execute_host_printf(cpu),
            HostLibcFunction::Strcmp => execute_host_strcmp(cpu),
            HostLibcFunction::Strlen => execute_host_strlen(cpu),
            HostLibcFunction::Strncmp => execute_host_strncmp(cpu),
        }?;
        self.log(format_args!(
            "host libc result pc=0x{pc:016x} handled={handled} next_pc=0x{:016x}",
            cpu.registers[Pc]
        ));
        Ok(handled)
    }

    fn precompile_reachable_blocks(
        &mut self,
        cpu: &mut RV64GC,
        entry_pc: u64,
    ) -> Result<usize, JitError> {
        let aot_mode = self.options.execution_mode == JitExecutionMode::Aot;
        let exhaustive_precompile = aot_mode
            && (!self.options.aot_compile_misses
                || self.options.aot_symbol_entries
                || self.options.aot_linear_sweep);
        let mut worklist = VecDeque::from([entry_pc]);
        if aot_mode {
            if self.options.aot_symbol_entries || !self.options.aot_compile_misses {
                for aot_pc in cpu.elf_aot_entry_points() {
                    worklist.push_back(aot_pc);
                }
            }
            if self.options.aot_linear_sweep {
                self.seed_linear_aot_block_starts(cpu, &mut worklist);
            }
        }
        let mut visited = FastU64Set::default();
        let mut compiled_blocks = 0;

        while let Some(pc) = worklist.pop_front() {
            if !visited.insert(pc) || self.cache.contains_key(&pc) {
                continue;
            }

            let (planned, tier) = if aot_mode {
                aot_precompile_plan_for_tier(cpu, pc, self.options)
            } else {
                (BlockPlan::from_cpu(cpu, pc), JitTier::Baseline)
            };
            let Some(plan) = planned.plan else {
                if aot_mode && pc == entry_pc {
                    return Err(JitError::AotUnsupportedBlock { pc });
                }
                continue;
            };

            if aot_mode && !plan.can_run_without_interpreter_fallback() {
                if pc == entry_pc {
                    return Err(JitError::AotUnsupportedBlock { pc });
                }
                self.log(format_args!(
                    "skip aot precompile pc=0x{pc:016x} cause=interpreter-fallback"
                ));
                continue;
            }

            let successors = if exhaustive_precompile {
                plan.successors()
            } else {
                Vec::new()
            };
            self.compile_plan(cpu, pc, plan, tier, "aot", 0)?;
            compiled_blocks += 1;

            for successor in successors {
                if !visited.contains(&successor) {
                    worklist.push_back(successor);
                }
            }
        }

        Ok(compiled_blocks)
    }

    fn seed_linear_aot_block_starts(&self, cpu: &RV64GC, worklist: &mut VecDeque<u64>) {
        for (start, end) in cpu.elf_executable_ranges() {
            worklist.push_back(start);
            let mut pc = start;
            while pc < end {
                let Ok(opcode) = cpu.ram.read_word(pc) else {
                    break;
                };
                let instruction_len = instruction_len(opcode);
                let next_pc = pc.wrapping_add(instruction_len);
                let decoded = cpu.find_instruction(opcode);
                let Some(native) = NativeInstruction::lower(pc, decoded) else {
                    worklist.push_back(next_pc);
                    pc = next_pc;
                    continue;
                };

                if native.terminates_block() {
                    worklist.push_back(next_pc);
                    seed_native_successors(&native, worklist);
                }

                pc = next_pc;
            }
        }
    }

    fn drain_background_compiler(&mut self, cpu: &mut RV64GC) -> Result<(), JitError> {
        let Some(compiler) = self.background_compiler.as_mut() else {
            return Ok(());
        };

        let results = compiler.drain_ready();

        for result in results {
            self.apply_background_compile_result(cpu, result)?;
        }

        Ok(())
    }

    fn apply_background_compile_result(
        &mut self,
        cpu: &mut RV64GC,
        result: BackgroundCompileResult,
    ) -> Result<(), JitError> {
        match result {
            BackgroundCompileResult::Compiled {
                pc,
                tier,
                reason,
                preserved_execution_count,
                plan,
                optimization_report,
                mut block,
                compile_duration,
            } => {
                self.pending_optimized_compiles.remove(&pc);
                let stop_text = plan.stop.to_string();
                let current_execution_count = {
                    let Some(current) = self.cache.get_mut(&pc) else {
                        self.log(format_args!(
                            "discard background compile pc=0x{pc:016x} tier={tier} reason={reason} cause=missing-block"
                        ));
                        if let Some(tracer) = cpu.tracer_mut() {
                            tracer.record_jit_background_discard();
                        }
                        return Ok(());
                    };

                    if current.is_stale(cpu) || block.is_stale(cpu) {
                        self.log(format_args!(
                            "discard background compile pc=0x{pc:016x} tier={tier} reason={reason} cause=stale"
                        ));
                        if let Some(tracer) = cpu.tracer_mut() {
                            tracer.record_jit_background_discard();
                        }
                        return Ok(());
                    }

                    if current.tier != JitTier::Baseline {
                        self.log(format_args!(
                            "discard background compile pc=0x{pc:016x} tier={tier} reason={reason} cause=already-promoted"
                        ));
                        if let Some(tracer) = cpu.tracer_mut() {
                            tracer.record_jit_background_discard();
                        }
                        return Ok(());
                    }

                    current.execution_count.max(preserved_execution_count)
                };

                block.tier = tier;
                block.execution_count = current_execution_count;
                block.next_optimized_attempt_count = current_execution_count;
                let instruction_count = plan.guest_instruction_count;
                let emitted_operations = plan.operations.len();
                let end_pc = plan.end_pc;
                let stop = plan.stop;
                let code_len = block.code_len;
                let native_entry = block.native_entry_address();
                self.log(format_args!(
                    "compile pc=0x{pc:016x} tier={tier} reason={reason} background=true native=0x{native_entry:016x} end=0x{end_pc:016x} guest_instructions={instruction_count} emitted_operations={emitted_operations} code_bytes={code_len} stop={stop} optimizations={optimization_report}"
                ));
                self.dump_block(&plan, &block, optimization_report);
                if let Some(tracer) = cpu.tracer_mut() {
                    tracer.record_jit_compile(
                        pc,
                        instruction_count,
                        code_len,
                        &stop_text,
                        compile_duration,
                    );
                    tracer.record_jit_background_adoption();
                }
                append_jit_map(pc, tier, native_entry, code_len, instruction_count);
                self.cache.insert(pc, Box::new(block));
                self.invalidate_dispatch_cache();
            }
            BackgroundCompileResult::PromotedWithoutCompile {
                pc,
                tier,
                reason,
                optimization_report,
            } => {
                self.pending_optimized_compiles.remove(&pc);
                let Some(block) = self.cache.get_mut(&pc) else {
                    return Ok(());
                };
                if block.is_stale(cpu) || block.tier != JitTier::Baseline {
                    if let Some(tracer) = cpu.tracer_mut() {
                        tracer.record_jit_background_discard();
                    }
                    return Ok(());
                }
                block.tier = tier;
                block.next_optimized_attempt_count = block.execution_count;
                self.log(format_args!(
                    "promote pc=0x{pc:016x} tier={tier} reason={reason} background=true skipped_compile=true optimizations={optimization_report}"
                ));
                if let Some(tracer) = cpu.tracer_mut() {
                    tracer.record_jit_background_adoption();
                }
            }
            BackgroundCompileResult::Failed {
                pc,
                tier,
                reason,
                error,
            } => {
                self.pending_optimized_compiles.remove(&pc);
                self.log(format_args!(
                    "background compile failed pc=0x{pc:016x} tier={tier} reason={reason}: {error}"
                ));
                return Err(error);
            }
            BackgroundCompileResult::WorkerFailed { error } => return Err(error),
        }

        Ok(())
    }

    fn queue_background_optimization(
        &mut self,
        cpu: &mut RV64GC,
        pc: u64,
        tier: JitTier,
        reason: &'static str,
        preserved_execution_count: u64,
    ) -> Result<(), JitError> {
        if !self.pending_optimized_compiles.insert(pc) {
            return Ok(());
        }

        let (planned, tier) = plan_for_tier(cpu, pc, tier);
        let Some(plan) = planned.plan else {
            return Err(JitError::BlockPlanningFailed {
                pc,
                reason: planned.stop.to_string(),
            });
        };

        let Some(compiler) = self.background_compiler.as_ref() else {
            self.pending_optimized_compiles.remove(&pc);
            return Ok(());
        };

        match compiler.try_enqueue(background::BackgroundCompileJob {
            pc,
            plan,
            tier,
            reason,
            preserved_execution_count,
            include_listing: self.options.dump_instructions,
        }) {
            Ok(()) => {
                self.log(format_args!(
                    "queue compile pc=0x{pc:016x} tier={tier} reason={reason} background=true"
                ));
                if let Some(tracer) = cpu.tracer_mut() {
                    tracer.record_jit_background_queue();
                }
                Ok(())
            }
            Err(BackgroundEnqueueError::Full) => {
                self.pending_optimized_compiles.remove(&pc);
                self.log(format_args!(
                    "skip background compile pc=0x{pc:016x} tier={tier} reason={reason} cause=queue-full"
                ));
                if let Some(tracer) = cpu.tracer_mut() {
                    tracer.record_jit_background_queue_full();
                }
                Ok(())
            }
            Err(BackgroundEnqueueError::Disconnected) => {
                self.pending_optimized_compiles.remove(&pc);
                Err(JitError::BackgroundCompilerStopped)
            }
        }
    }

    fn compile_block(
        &mut self,
        cpu: &mut RV64GC,
        pc: u64,
        tier: JitTier,
        reason: &str,
        preserved_execution_count: u64,
    ) -> Result<CompileOutcome, JitError> {
        let (planned, tier) = plan_for_tier(cpu, pc, tier);
        let Some(plan) = planned.plan else {
            return Err(JitError::BlockPlanningFailed {
                pc,
                reason: planned.stop.to_string(),
            });
        };

        if !plan.can_run_without_interpreter_fallback() {
            return Err(JitError::InterpreterFallbackDisabled { pc });
        }

        self.compile_plan(cpu, pc, plan, tier, reason, preserved_execution_count)
    }

    fn compile_aot_runtime_block(
        &mut self,
        cpu: &mut RV64GC,
        pc: u64,
        reason: &str,
    ) -> Result<CompileOutcome, JitError> {
        let planned = BlockPlan::optimized_from_cpu(cpu, pc);
        let Some(plan) = planned.plan else {
            return Err(JitError::BlockPlanningFailed {
                pc,
                reason: planned.stop.to_string(),
            });
        };

        if !plan.can_run_without_interpreter_fallback() {
            return Err(JitError::InterpreterFallbackDisabled { pc });
        }

        if plan.contains_compiler_region() {
            return self.compile_plan(cpu, pc, plan, JitTier::Optimized, reason, 0);
        }

        if let Some(traced) = aot_trace_plan_from_cpu(cpu, pc, self.options) {
            let Some(trace_plan) = traced.plan else {
                unreachable!("aot trace helper only returns planned traces")
            };
            if !trace_plan.can_run_without_interpreter_fallback() {
                return Err(JitError::InterpreterFallbackDisabled { pc });
            }
            return self.compile_plan(cpu, pc, trace_plan, JitTier::Trace, reason, 0);
        }

        if aot_runtime_plan_should_optimize(&plan) {
            return self.compile_plan(cpu, pc, plan, JitTier::Optimized, reason, 0);
        }

        self.compile_plan(cpu, pc, plan, JitTier::Baseline, reason, 0)
    }

    fn compile_plan(
        &mut self,
        cpu: &mut RV64GC,
        pc: u64,
        plan: BlockPlan,
        tier: JitTier,
        reason: &str,
        preserved_execution_count: u64,
    ) -> Result<CompileOutcome, JitError> {
        let prepared = prepare_plan_for_tier(plan, tier);
        if prepared.skipped_compile {
            if let Some(block) = self.cache.get_mut(&pc) {
                block.tier = JitTier::Optimized;
                self.log(format_args!(
                    "promote pc=0x{pc:016x} tier={tier} reason={reason} skipped_compile=true optimizations={}",
                    prepared.optimization_report
                ));
                return Ok(CompileOutcome::PromotedWithoutCompile);
            }
        }

        let plan = prepared.plan;
        let optimization_report = prepared.optimization_report;

        let instruction_count = plan.guest_instruction_count;
        let emitted_operations = plan.operations.len();
        let end_pc = plan.end_pc;
        let stop = plan.stop;
        let stop_text = stop.to_string();
        let compile_start = Instant::now();
        let mut block = self
            .backend
            .compile(&plan, tier, self.options.dump_instructions)?;
        let compile_duration = compile_start.elapsed();
        block.tier = tier;
        block.execution_count = preserved_execution_count;
        block.next_optimized_attempt_count = preserved_execution_count;
        let code_len = block.code_len;
        let native_entry = block.native_entry_address();
        self.log(format_args!(
            "compile pc=0x{pc:016x} tier={tier} reason={reason} native=0x{native_entry:016x} end=0x{end_pc:016x} guest_instructions={instruction_count} emitted_operations={emitted_operations} code_bytes={code_len} stop={stop} optimizations={optimization_report}"
        ));
        self.dump_block(&plan, &block, optimization_report);
        if let Some(tracer) = cpu.tracer_mut() {
            tracer.record_jit_compile(
                pc,
                instruction_count,
                code_len,
                &stop_text,
                compile_duration,
            );
        }
        append_jit_map(pc, tier, native_entry, code_len, instruction_count);
        self.cache.insert(pc, Box::new(block));
        self.invalidate_dispatch_cache();

        Ok(CompileOutcome::Compiled)
    }

    fn log(&self, args: fmt::Arguments<'_>) {
        if self.options.debug_log {
            debug::line(format_args!("[jit] {args}"));
        }
    }

    fn dump_block(
        &self,
        plan: &BlockPlan,
        block: &CompiledBlock,
        optimization_report: OptimizationReport,
    ) {
        if !self.options.dump_instructions {
            return;
        }

        let start_pc = plan.fingerprint.first().map(|(pc, _)| *pc).unwrap_or(0);
        if let Some(filter_pc) = jit_dump_pc_filter() {
            if start_pc != filter_pc {
                return;
            }
        }
        debug::line(format_args!(
            "[jit-dump] block pc=0x{start_pc:016x} tier={} end=0x{:016x} guest_instructions={} emitted_operations={} code_bytes={} stop={} optimizations={}",
            block.tier,
            plan.end_pc,
            plan.guest_instruction_count,
            plan.operations.len(),
            block.code_len,
            plan.stop,
            optimization_report
        ));
        debug::line(format_args!("[jit-dump] rv64:"));
        for operation in &plan.operations {
            debug::line(format_args!(
                "[jit-dump]   0x{:016x}: 0x{:08x}  {}",
                operation.pc(),
                operation.opcode(),
                operation
            ));
        }
        debug::line(format_args!("[jit-dump] aarch64:"));
        for emission in &block.native_listing {
            debug::line(format_args!(
                "[jit-dump]   +0x{:04x}: 0x{:08x}  {}",
                emission.offset, emission.word, emission.text
            ));
        }
    }
}

fn jit_dump_pc_filter() -> Option<u64> {
    static FILTER: OnceLock<Option<u64>> = OnceLock::new();
    *FILTER.get_or_init(|| {
        let raw = std::env::var("RISCVM_JIT_DUMP_PC").ok()?;
        parse_u64_env_literal(raw.trim())
    })
}

fn parse_u64_env_literal(value: &str) -> Option<u64> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}

fn candidate_byte_scan_loop_from_cpu(cpu: &RV64GC, pc: u64) -> Option<CandidateByteScanLoop> {
    let increment = lower_native_at(cpu, pc)?;
    let NativeInstruction::Addi {
        rd: index_register,
        rs1: index_source,
        imm: 1,
    } = increment.instruction
    else {
        return None;
    };
    if index_register == 0 || index_source != index_register {
        return None;
    }

    let shadow_copy = lower_native_at(cpu, increment.next_pc)?;
    let (shadow_register, compare_offset_register) =
        copy_instruction_registers(shadow_copy.instruction)?;

    let loop_branch = lower_native_at(cpu, shadow_copy.next_pc)?;
    let (rs1, rs2, head_pc, exhausted_pc) = match loop_branch.instruction {
        NativeInstruction::Bne {
            rs1,
            rs2,
            target: head_pc,
            fallthrough: exhausted_pc,
        } => (rs1, rs2, head_pc, exhausted_pc),
        NativeInstruction::Beq {
            rs1,
            rs2,
            target: exhausted_pc,
            fallthrough: head_pc,
        } => (rs1, rs2, head_pc, exhausted_pc),
        _ => return None,
    };
    let limit_register = if rs1 == index_register {
        rs2
    } else if rs2 == index_register {
        rs1
    } else {
        return None;
    };
    if limit_register == 0 || head_pc <= loop_branch.pc {
        return None;
    }

    let scale = lower_native_at(cpu, head_pc)?;
    let NativeInstruction::Slli {
        rd: entry_address_register,
        rs1: scaled_index_source,
        shamt: 2,
    } = scale.instruction
    else {
        return None;
    };
    if entry_address_register == 0 || scaled_index_source != index_register {
        return None;
    }

    let table_add = lower_native_at(cpu, scale.next_pc)?;
    let NativeInstruction::Add {
        rd: table_entry_register,
        rs1: table_lhs,
        rs2: table_rhs,
    } = table_add.instruction
    else {
        return None;
    };
    if table_entry_register != entry_address_register {
        return None;
    }
    let table_base_register = if table_lhs == entry_address_register {
        table_rhs
    } else if table_rhs == entry_address_register {
        table_lhs
    } else {
        return None;
    };
    if table_base_register == 0 {
        return None;
    }

    let candidate_load = lower_native_at(cpu, table_add.next_pc)?;
    let NativeInstruction::Load {
        rd: candidate_offset_register,
        rs1: candidate_load_base,
        imm: 0,
        width: MemoryWidth::Word,
        signed: false,
    } = candidate_load.instruction
    else {
        return None;
    };
    if candidate_offset_register == 0 || candidate_load_base != entry_address_register {
        return None;
    }

    let candidate_add = lower_native_at(cpu, candidate_load.next_pc)?;
    let NativeInstruction::Add {
        rd: candidate_pointer_register,
        rs1: candidate_lhs,
        rs2: candidate_rhs,
    } = candidate_add.instruction
    else {
        return None;
    };
    let candidate_base_register = if candidate_lhs == candidate_offset_register {
        candidate_rhs
    } else if candidate_rhs == candidate_offset_register {
        candidate_lhs
    } else {
        return None;
    };
    if candidate_pointer_register == 0 || candidate_base_register == 0 {
        return None;
    }

    let lhs_address_add = lower_native_at(cpu, candidate_add.next_pc)?;
    let NativeInstruction::Add {
        rd: lhs_address_register,
        rs1: lhs_address_lhs,
        rs2: lhs_address_rhs,
    } = lhs_address_add.instruction
    else {
        return None;
    };
    if lhs_address_register == 0
        || !((lhs_address_lhs == candidate_pointer_register
            && lhs_address_rhs == compare_offset_register)
            || (lhs_address_rhs == candidate_pointer_register
                && lhs_address_lhs == compare_offset_register))
    {
        return None;
    }

    let rhs_address_add = lower_native_at(cpu, lhs_address_add.next_pc)?;
    let NativeInstruction::Add {
        rd: rhs_address_register,
        rs1: rhs_address_lhs,
        rs2: rhs_address_rhs,
    } = rhs_address_add.instruction
    else {
        return None;
    };
    let rhs_base_register = if rhs_address_lhs == compare_offset_register {
        rhs_address_rhs
    } else if rhs_address_rhs == compare_offset_register {
        rhs_address_lhs
    } else {
        return None;
    };
    if rhs_address_register == 0 || rhs_base_register == 0 {
        return None;
    }

    let lhs_load = lower_native_at(cpu, rhs_address_add.next_pc)?;
    let NativeInstruction::Load {
        rd: lhs_value_register,
        rs1: lhs_load_base,
        imm: 0,
        width: MemoryWidth::Byte,
        signed: false,
    } = lhs_load.instruction
    else {
        return None;
    };
    if lhs_value_register == 0 || lhs_load_base != lhs_address_register {
        return None;
    }

    let rhs_load = lower_native_at(cpu, lhs_load.next_pc)?;
    let NativeInstruction::Load {
        rd: rhs_value_register,
        rs1: rhs_load_base,
        imm: 0,
        width: MemoryWidth::Byte,
        signed: false,
    } = rhs_load.instruction
    else {
        return None;
    };
    if rhs_value_register == 0 || rhs_load_base != rhs_address_register {
        return None;
    }

    let compare_branch = lower_native_at(cpu, rhs_load.next_pc)?;
    let NativeInstruction::Bne {
        rs1: compare_lhs,
        rs2: compare_rhs,
        target: loop_pc,
        fallthrough: match_pc,
    } = compare_branch.instruction
    else {
        return None;
    };
    if loop_pc != pc
        || !((compare_lhs == lhs_value_register && compare_rhs == rhs_value_register)
            || (compare_lhs == rhs_value_register && compare_rhs == lhs_value_register))
    {
        return None;
    }

    Some(CandidateByteScanLoop {
        index_register,
        shadow_register,
        compare_offset_register,
        limit_register,
        exhausted_pc,
        table_base_register,
        entry_address_register,
        candidate_offset_register,
        candidate_base_register,
        candidate_pointer_register,
        lhs_address_register,
        rhs_address_register,
        rhs_base_register,
        lhs_value_register,
        rhs_value_register,
        match_pc,
    })
}

fn execute_candidate_byte_scan_loop(
    cpu: &mut RV64GC,
    pattern: CandidateByteScanLoop,
) -> Option<u64> {
    let mut executed = 0u64;
    loop {
        let index = cpu.registers[pattern.index_register as usize].wrapping_add(1);
        cpu.registers[pattern.index_register as usize] = index;
        cpu.registers[pattern.shadow_register as usize] =
            cpu.registers[pattern.compare_offset_register as usize];
        executed = executed.saturating_add(3);

        if index == cpu.registers[pattern.limit_register as usize] {
            cpu.registers[Pc] = pattern.exhausted_pc;
            cpu.registers[Zero] = 0;
            return Some(executed);
        }

        let table_entry =
            cpu.registers[pattern.table_base_register as usize].wrapping_add(index << 2);
        let candidate_offset = u64::from(cpu.ram.read_u32_cached(table_entry).ok()?);
        let candidate_pointer =
            cpu.registers[pattern.candidate_base_register as usize].wrapping_add(candidate_offset);
        let compare_offset = cpu.registers[pattern.compare_offset_register as usize];
        let lhs_address = candidate_pointer.wrapping_add(compare_offset);
        let rhs_address =
            cpu.registers[pattern.rhs_base_register as usize].wrapping_add(compare_offset);
        let lhs_value = u64::from(cpu.ram.read_u8_cached(lhs_address).ok()?);
        let rhs_value = u64::from(cpu.ram.read_u8_cached(rhs_address).ok()?);

        cpu.registers[pattern.entry_address_register as usize] = table_entry;
        cpu.registers[pattern.candidate_offset_register as usize] = candidate_offset;
        cpu.registers[pattern.candidate_pointer_register as usize] = candidate_pointer;
        cpu.registers[pattern.lhs_address_register as usize] = lhs_address;
        cpu.registers[pattern.rhs_address_register as usize] = rhs_address;
        cpu.registers[pattern.lhs_value_register as usize] = lhs_value;
        cpu.registers[pattern.rhs_value_register as usize] = rhs_value;
        executed = executed.saturating_add(9);

        if lhs_value == rhs_value {
            cpu.registers[Pc] = pattern.match_pc;
            cpu.registers[Zero] = 0;
            return Some(executed);
        }
    }
}

fn copy_instruction_registers(instruction: NativeInstruction) -> Option<(u8, u8)> {
    match instruction {
        NativeInstruction::Addi { rd, rs1, imm: 0 } if rd != 0 => Some((rd, rs1)),
        NativeInstruction::Add { rd, rs1, rs2 } if rd != 0 && rs1 == 0 => Some((rd, rs2)),
        NativeInstruction::Add { rd, rs1, rs2 } if rd != 0 && rs2 == 0 => Some((rd, rs1)),
        _ => None,
    }
}

#[inline(always)]
fn execute_compiled_block(
    options: JitOptions,
    cpu: &mut RV64GC,
    pc: u64,
    block: &mut CompiledBlock,
) -> Result<JitStep, JitError> {
    let instructions = block.instruction_count;
    let code_len = block.code_len;
    let tier = block.tier;
    if options.debug_log {
        debug::line(format_args!(
            "[jit] execute pc=0x{pc:016x} tier={tier} instructions={instructions}"
        ));
    }
    let execute_start = cpu.tracer_enabled().then(Instant::now);
    let executed_instructions = block.execute(cpu);
    cpu.registers[Zero] = 0;
    if let Some(reason) = cpu.take_jit_runtime_fault() {
        return Err(JitError::RuntimeFault { pc, reason });
    }
    block.execution_count = block.execution_count.saturating_add(1);
    if options.debug_log {
        debug::line(format_args!(
            "[jit] execute complete pc=0x{pc:016x} tier={tier} next_pc=0x{:016x}",
            cpu.registers[Pc]
        ));
    }
    let next_pc = cpu.registers[Pc];
    if let Some(tracer) = cpu.tracer_mut() {
        let execute_duration = execute_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        tracer.record_jit_block(
            pc,
            instructions,
            executed_instructions,
            code_len,
            next_pc,
            execute_duration,
            &block.profile_instructions,
        );
    }

    Ok(JitStep::Native {
        pc,
        instructions: executed_instructions,
    })
}

#[inline(always)]
fn execute_compiled_block_fast(
    cpu: &mut RV64GC,
    pc: u64,
    block: &mut CompiledBlock,
) -> Result<(), JitError> {
    block.execute(cpu);
    if let Some(reason) = cpu.take_jit_runtime_fault() {
        return Err(JitError::RuntimeFault { pc, reason });
    }
    if block.tier == JitTier::Baseline {
        block.execution_count = block.execution_count.saturating_add(1);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompileOutcome {
    Compiled,
    PromotedWithoutCompile,
}

enum CompileDecision {
    AotMiss,
    AotRuntime {
        reason: &'static str,
    },
    Synchronous {
        tier: JitTier,
        reason: &'static str,
        preserved_execution_count: u64,
    },
    BackgroundOptimized {
        tier: JitTier,
        reason: &'static str,
        preserved_execution_count: u64,
    },
    None,
}

fn aot_runtime_plan_should_optimize(plan: &BlockPlan) -> bool {
    plan.contains_compiler_region()
        || plan.ends_with_self_loop_branch()
        || plan.ends_with_loop_back_edge()
}

fn startup_profile_plan_for_tier(
    cpu: &RV64GC,
    pc: u64,
    options: JitOptions,
) -> (BlockPlanResult, JitTier) {
    if options.execution_mode == JitExecutionMode::Aot {
        return aot_precompile_plan_for_tier(cpu, pc, options);
    }

    (BlockPlan::from_cpu(cpu, pc), JitTier::Baseline)
}

fn aot_precompile_plan_for_tier(
    cpu: &RV64GC,
    pc: u64,
    options: JitOptions,
) -> (BlockPlanResult, JitTier) {
    let optimized = BlockPlan::optimized_from_cpu(cpu, pc);
    if optimized
        .plan
        .as_ref()
        .is_some_and(BlockPlan::contains_compiler_region)
    {
        return (optimized, JitTier::Optimized);
    }

    if let Some(traced) = aot_trace_plan_from_cpu(cpu, pc, options) {
        return (traced, JitTier::Trace);
    }

    (optimized, JitTier::Optimized)
}

fn aot_trace_plan_from_cpu(cpu: &RV64GC, pc: u64, options: JitOptions) -> Option<BlockPlanResult> {
    if !options.trace_compilation || !options.aot_compile_misses {
        return None;
    }

    let traced = BlockPlan::trace_from_cpu(cpu, pc);
    if traced
        .plan
        .as_ref()
        .is_some_and(aot_trace_plan_should_optimize)
    {
        Some(traced)
    } else {
        None
    }
}

fn aot_trace_plan_should_optimize(plan: &BlockPlan) -> bool {
    plan.operations.last().is_some_and(|operation| {
        matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::TraceLoopGuard { .. })
        )
    })
}

fn plan_for_tier(cpu: &RV64GC, pc: u64, tier: JitTier) -> (BlockPlanResult, JitTier) {
    match tier {
        JitTier::Baseline => (BlockPlan::from_cpu(cpu, pc), JitTier::Baseline),
        JitTier::Optimized => (BlockPlan::optimized_from_cpu(cpu, pc), JitTier::Optimized),
        JitTier::Trace => {
            let optimized = BlockPlan::optimized_from_cpu(cpu, pc);
            if optimized
                .plan
                .as_ref()
                .is_some_and(BlockPlan::contains_compiler_region)
            {
                return (optimized, JitTier::Optimized);
            }

            let trace = BlockPlan::trace_from_cpu(cpu, pc);
            if trace.plan.is_some() {
                (trace, JitTier::Trace)
            } else {
                (optimized, JitTier::Optimized)
            }
        }
    }
}

fn host_libc_function_for_name(name: &str) -> Option<HostLibcFunction> {
    match name {
        "exit" | "__GI_exit" => Some(HostLibcFunction::Exit),
        "memcmp" | "__memcmp" => Some(HostLibcFunction::Memcmp),
        "memcpy" | "__memcpy" | "__memcpy_generic" => Some(HostLibcFunction::Memcpy),
        "memmove" | "__memmove" => Some(HostLibcFunction::Memmove),
        "memset" | "__memset" | "__memset_generic" => Some(HostLibcFunction::Memset),
        "printf" | "__printf" | "_IO_printf" => Some(HostLibcFunction::Printf),
        "puts" | "_IO_puts" => Some(HostLibcFunction::Puts),
        "strcmp" | "__strcmp" => Some(HostLibcFunction::Strcmp),
        "strlen" | "__strlen" | "__strlen_generic" => Some(HostLibcFunction::Strlen),
        "strncmp" | "__strncmp" => Some(HostLibcFunction::Strncmp),
        _ => None,
    }
}

fn host_libc_function_for_plt_name(name: &str, allow_stdio: bool) -> Option<HostLibcFunction> {
    match host_libc_function_for_name(name) {
        Some(HostLibcFunction::Exit | HostLibcFunction::Printf | HostLibcFunction::Puts)
            if !allow_stdio =>
        {
            None
        }
        function => function,
    }
}

fn host_libc_function_for_defined_symbol(name: &str) -> Option<HostLibcFunction> {
    match host_libc_function_for_name(name) {
        Some(
            HostLibcFunction::Exit
            | HostLibcFunction::LibcStartMain
            | HostLibcFunction::Printf
            | HostLibcFunction::Puts,
        ) => None,
        function => function,
    }
}

fn host_libc_start_main_shortcut_for_defined_symbol(name: &str) -> Option<HostLibcFunction> {
    if name == "__libc_start_main" {
        return Some(HostLibcFunction::LibcStartMain);
    }

    match host_libc_function_for_name(name) {
        Some(HostLibcFunction::Exit | HostLibcFunction::Printf | HostLibcFunction::Puts) => {
            host_libc_function_for_name(name)
        }
        _ => None,
    }
}

fn execute_host_exit(cpu: &mut RV64GC) -> Result<bool, JitError> {
    cpu.should_quit = true;
    cpu.registers[Pc] = cpu.registers[Ra];
    Ok(true)
}

fn execute_host_libc_start_main(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let main = cpu.registers[A0];
    if main == 0 {
        return Ok(false);
    }

    let argc = cpu.registers[A1];
    let argv = cpu.registers[A2];
    let envp = argv.wrapping_add(argc.wrapping_add(1).wrapping_mul(8));
    cpu.registers[A0] = argc;
    cpu.registers[A1] = argv;
    cpu.registers[A2] = envp;
    cpu.registers[Ra] = HOST_LIBC_EXIT_TRAMPOLINE;
    cpu.registers[Pc] = main;
    Ok(true)
}

fn execute_host_puts(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let Some(bytes) = read_guest_c_string(cpu, cpu.registers[A0], MAX_HOST_LIBC_C_STRING) else {
        return Ok(false);
    };
    let Some(bytes) = bytes.strip_suffix(&[0]) else {
        return Ok(false);
    };

    let mut output = Vec::with_capacity(bytes.len() + 1);
    output.extend_from_slice(bytes);
    output.push(b'\n');
    write_host_stdout_output(cpu, &output)?;
    cpu.registers[A0] = output.len() as u64;
    cpu.registers[Pc] = cpu.registers[Ra];
    Ok(true)
}

#[cfg(unix)]
fn execute_host_memcmp(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let count = cpu.registers[A2] as usize;
    if count > MAX_HOST_LIBC_BYTES {
        return Ok(false);
    }
    if count == 0 {
        finish_host_libc_call(cpu, 0);
        return Ok(true);
    }
    if let (Ok(lhs), Ok(rhs)) = (
        cpu.ram
            .direct_read_ptr_range(cpu.registers[A0], count as u64),
        cpu.ram
            .direct_read_ptr_range(cpu.registers[A1], count as u64),
    ) {
        let result = unsafe { memcmp(lhs.cast::<c_void>(), rhs.cast::<c_void>(), count) };
        finish_host_libc_call(cpu, result as i64 as u64);
        return Ok(true);
    }

    let Some(lhs) = read_guest_bytes(cpu, cpu.registers[A0], count) else {
        return Ok(false);
    };
    let Some(rhs) = read_guest_bytes(cpu, cpu.registers[A1], count) else {
        return Ok(false);
    };

    let result = unsafe {
        memcmp(
            lhs.as_ptr().cast::<c_void>(),
            rhs.as_ptr().cast::<c_void>(),
            count,
        )
    };
    finish_host_libc_call(cpu, result as i64 as u64);
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_memcmp(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

#[cfg(unix)]
fn execute_host_memcpy(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let dest = cpu.registers[A0];
    let src = cpu.registers[A1];
    let count = cpu.registers[A2] as usize;
    if count > MAX_HOST_LIBC_BYTES {
        return Ok(false);
    }
    if count == 0 {
        finish_host_libc_call(cpu, dest);
        return Ok(true);
    }
    if !guest_byte_ranges_overlap(dest, src, count) {
        if let Ok(src_ptr) = cpu.ram.direct_read_ptr_range(src, count as u64) {
            if let Ok(dest_ptr) = cpu.ram.direct_write_ptr_range(dest, count as u64) {
                unsafe {
                    ptr::copy_nonoverlapping(src_ptr, dest_ptr, count);
                }
                finish_host_libc_call(cpu, dest);
                return Ok(true);
            }
        }
    }

    let Some(src_bytes) = read_guest_bytes(cpu, src, count) else {
        return Ok(false);
    };

    let mut dest_bytes = vec![0; count];
    unsafe {
        memcpy(
            dest_bytes.as_mut_ptr().cast::<c_void>(),
            src_bytes.as_ptr().cast::<c_void>(),
            count,
        );
    }
    write_guest_bytes(cpu, dest, &dest_bytes)?;
    finish_host_libc_call(cpu, dest);
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_memcpy(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

#[cfg(unix)]
fn execute_host_memmove(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let dest = cpu.registers[A0];
    let src = cpu.registers[A1];
    let count = cpu.registers[A2] as usize;
    if count > MAX_HOST_LIBC_BYTES {
        return Ok(false);
    }
    if count == 0 {
        finish_host_libc_call(cpu, dest);
        return Ok(true);
    }
    if let Ok(src_ptr) = cpu.ram.direct_read_ptr_range(src, count as u64) {
        if let Ok(dest_ptr) = cpu.ram.direct_write_ptr_range(dest, count as u64) {
            unsafe {
                ptr::copy(src_ptr, dest_ptr, count);
            }
            finish_host_libc_call(cpu, dest);
            return Ok(true);
        }
    }

    let Some(src_bytes) = read_guest_bytes(cpu, src, count) else {
        return Ok(false);
    };

    let mut dest_bytes = vec![0; count];
    unsafe {
        memmove(
            dest_bytes.as_mut_ptr().cast::<c_void>(),
            src_bytes.as_ptr().cast::<c_void>(),
            count,
        );
    }
    write_guest_bytes(cpu, dest, &dest_bytes)?;
    finish_host_libc_call(cpu, dest);
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_memmove(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

#[cfg(unix)]
fn execute_host_memset(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let dest = cpu.registers[A0];
    let value = cpu.registers[A1] as c_int;
    let count = cpu.registers[A2] as usize;
    if count > MAX_HOST_LIBC_BYTES {
        return Ok(false);
    }
    if count == 0 {
        finish_host_libc_call(cpu, dest);
        return Ok(true);
    }
    if let Ok(dest_bytes) = cpu.ram.write_slice_range(dest, count as u64) {
        dest_bytes.fill(value as u8);
        finish_host_libc_call(cpu, dest);
        return Ok(true);
    }

    let mut bytes = vec![0; count];
    unsafe {
        memset(bytes.as_mut_ptr().cast::<c_void>(), value, count);
    }
    write_guest_bytes(cpu, dest, &bytes)?;
    finish_host_libc_call(cpu, dest);
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_memset(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

#[cfg(unix)]
fn execute_host_printf(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let Some(format) = read_guest_c_string(cpu, cpu.registers[A0], 4096) else {
        return Ok(false);
    };
    let Some(format_without_nul) = format.strip_suffix(&[0]) else {
        return Ok(false);
    };
    let Ok(format_text) = std::str::from_utf8(format_without_nul) else {
        return Ok(false);
    };

    let args = [
        cpu.registers[A1],
        cpu.registers[A2],
        cpu.registers[A3],
        cpu.registers[A4],
        cpu.registers[A5],
        cpu.registers[A6],
        cpu.registers[A7],
    ];
    let Some(output) = format_host_printf(cpu, format_text, &args) else {
        return Ok(false);
    };

    write_host_stdout_output(cpu, &output)?;
    let output_len = output.len();
    cpu.registers[A0] = output_len as u64;
    cpu.registers[Pc] = cpu.registers[Ra];
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_printf(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

fn format_host_printf(cpu: &RV64GC, format: &str, args: &[u64]) -> Option<Vec<u8>> {
    let mut output = Vec::new();
    let mut arg_index = 0;
    let bytes = format.as_bytes();
    let mut offset = 0;

    while offset < bytes.len() {
        let byte = bytes[offset];
        offset += 1;
        if byte != b'%' {
            output.push(byte);
            continue;
        }

        if offset >= bytes.len() {
            return None;
        }
        if bytes[offset] == b'%' {
            output.push(b'%');
            offset += 1;
            continue;
        }

        let mut long_count = 0;
        while offset < bytes.len() && bytes[offset] == b'l' {
            long_count += 1;
            if long_count > 2 {
                return None;
            }
            offset += 1;
        }
        if offset >= bytes.len() {
            return None;
        }

        let specifier = bytes[offset];
        offset += 1;
        let raw_arg = *args.get(arg_index)?;
        arg_index += 1;

        match specifier {
            b'd' | b'i' => {
                let value = match long_count {
                    0 => i64::from(raw_arg as u32 as i32),
                    1 | 2 => raw_arg as i64,
                    _ => return None,
                };
                output.extend_from_slice(value.to_string().as_bytes());
            }
            b'u' => {
                let value = match long_count {
                    0 => u64::from(raw_arg as u32),
                    1 | 2 => raw_arg,
                    _ => return None,
                };
                output.extend_from_slice(value.to_string().as_bytes());
            }
            b'x' | b'X' => {
                let value = match long_count {
                    0 => u64::from(raw_arg as u32),
                    1 | 2 => raw_arg,
                    _ => return None,
                };
                let text = if specifier == b'x' {
                    format!("{value:x}")
                } else {
                    format!("{value:X}")
                };
                output.extend_from_slice(text.as_bytes());
            }
            b'c' if long_count == 0 => output.push(raw_arg as u8),
            b's' if long_count == 0 => {
                let value = read_guest_c_string(cpu, raw_arg, MAX_HOST_LIBC_C_STRING)?;
                let value = value.strip_suffix(&[0])?;
                output.extend_from_slice(value);
            }
            _ => return None,
        }
    }

    Some(output)
}

#[cfg(unix)]
fn execute_host_strcmp(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let Some(lhs) = read_guest_c_string(cpu, cpu.registers[A0], MAX_HOST_LIBC_C_STRING) else {
        return Ok(false);
    };
    let Some(rhs) = read_guest_c_string(cpu, cpu.registers[A1], MAX_HOST_LIBC_C_STRING) else {
        return Ok(false);
    };

    let result = unsafe { strcmp(lhs.as_ptr().cast(), rhs.as_ptr().cast()) };
    finish_host_libc_call(cpu, result as i64 as u64);
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_strcmp(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

#[cfg(unix)]
fn execute_host_strlen(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let Some(value) = read_guest_c_string(cpu, cpu.registers[A0], MAX_HOST_LIBC_C_STRING) else {
        return Ok(false);
    };

    let result = unsafe { strlen(value.as_ptr().cast()) };
    finish_host_libc_call(cpu, result as u64);
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_strlen(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

#[cfg(unix)]
fn execute_host_strncmp(cpu: &mut RV64GC) -> Result<bool, JitError> {
    let count = cpu.registers[A2] as usize;
    if count > MAX_HOST_LIBC_C_STRING {
        return Ok(false);
    }
    if count == 0 {
        finish_host_libc_call(cpu, 0);
        return Ok(true);
    }

    let Some(lhs) = read_guest_string_prefix(cpu, cpu.registers[A0], count) else {
        return Ok(false);
    };
    let Some(rhs) = read_guest_string_prefix(cpu, cpu.registers[A1], count) else {
        return Ok(false);
    };

    let result = unsafe { strncmp(lhs.as_ptr().cast(), rhs.as_ptr().cast(), count) };
    finish_host_libc_call(cpu, result as i64 as u64);
    Ok(true)
}

#[cfg(not(unix))]
fn execute_host_strncmp(_cpu: &mut RV64GC) -> Result<bool, JitError> {
    Ok(false)
}

fn host_libc_call_has_return_address(cpu: &RV64GC, target_pc: u64) -> bool {
    let ra = cpu.registers[Ra];

    if let Some(call_pc) = ra.checked_sub(4) {
        if let Ok(instruction) = cpu.ram.read_word(call_pc) {
            let opcode = instruction & 0x7f;
            let rd = (instruction >> 7) & 0x1f;
            if instruction & 0b11 == 0b11 && rd == Ra as u32 {
                return match opcode {
                    0x6f => jal_target(call_pc, instruction) == target_pc,
                    0x67 => true,
                    _ => false,
                };
            }
        }
    }

    let Some(compressed_call_pc) = ra.checked_sub(2) else {
        return false;
    };
    if let Ok(instruction) = cpu.ram.read_halfword(compressed_call_pc) {
        let instruction = instruction as u16;
        return instruction & 0b1111_0000_0111_1111 == 0b1001_0000_0000_0010;
    }
    false
}

fn jal_target(pc: u64, instruction: u32) -> u64 {
    let offset = ((instruction as u64 >> 31) & 0x1) << 20
        | ((instruction as u64 >> 12) & 0xff) << 12
        | ((instruction as u64 >> 20) & 0x1) << 11
        | ((instruction as u64 >> 21) & 0x3ff) << 1;
    (pc as i64).wrapping_add(sign_extend(offset, 21)) as u64
}

fn finish_host_libc_call(cpu: &mut RV64GC, result: u64) {
    cpu.registers[A0] = result;
    cpu.registers[Pc] = cpu.registers[Ra];
}

fn read_guest_bytes(cpu: &RV64GC, address: u64, len: usize) -> Option<Vec<u8>> {
    if let Ok(bytes) = cpu.ram.read_slice_range(address, len as u64) {
        return Some(bytes.to_vec());
    }

    let mut bytes = Vec::with_capacity(len);
    for offset in 0..len {
        bytes.push(
            cpu.ram
                .read_byte(address.wrapping_add(offset as u64))
                .ok()?,
        );
    }
    Some(bytes)
}

fn read_guest_c_string(cpu: &RV64GC, address: u64, max_len: usize) -> Option<Vec<u8>> {
    if let Ok(bytes) = cpu.ram.read_slice_range(address, max_len as u64) {
        let nul = bytes.iter().position(|byte| *byte == 0)?;
        return Some(bytes[..=nul].to_vec());
    }

    let mut bytes = Vec::new();
    for offset in 0..max_len {
        let byte = cpu
            .ram
            .read_byte(address.wrapping_add(offset as u64))
            .ok()?;
        bytes.push(byte);
        if byte == 0 {
            return Some(bytes);
        }
    }
    None
}

fn read_guest_string_prefix(cpu: &RV64GC, address: u64, max_len: usize) -> Option<Vec<u8>> {
    if let Ok(bytes) = cpu.ram.read_slice_range(address, max_len as u64) {
        let mut prefix = match bytes.iter().position(|byte| *byte == 0) {
            Some(nul) => bytes[..=nul].to_vec(),
            None => {
                let mut bytes = bytes.to_vec();
                bytes.push(0);
                bytes
            }
        };
        if prefix.is_empty() {
            prefix.push(0);
        }
        return Some(prefix);
    }

    let mut bytes = Vec::new();
    for offset in 0..max_len {
        let byte = cpu
            .ram
            .read_byte(address.wrapping_add(offset as u64))
            .ok()?;
        bytes.push(byte);
        if byte == 0 {
            return Some(bytes);
        }
    }
    bytes.push(0);
    Some(bytes)
}

fn write_guest_bytes(cpu: &mut RV64GC, address: u64, bytes: &[u8]) -> Result<(), JitError> {
    if let Ok(dest) = cpu.ram.write_slice_range(address, bytes.len() as u64) {
        dest.copy_from_slice(bytes);
        return Ok(());
    }

    for (offset, byte) in bytes.iter().copied().enumerate() {
        cpu.ram
            .write_byte(address.wrapping_add(offset as u64), byte)
            .map_err(|_| JitError::HostLibcFailed)?;
    }
    Ok(())
}

fn write_host_stdout_output(cpu: &mut RV64GC, bytes: &[u8]) -> Result<(), JitError> {
    cpu.filesystem
        .write(1, bytes)
        .map(|_| ())
        .map_err(|_| JitError::HostLibcFailed)
}

#[cfg(unix)]
fn guest_byte_ranges_overlap(lhs: u64, rhs: u64, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let len = len as u64;
    let Some(lhs_end) = lhs.checked_add(len) else {
        return true;
    };
    let Some(rhs_end) = rhs.checked_add(len) else {
        return true;
    };
    lhs < rhs_end && rhs < lhs_end
}

enum OptimizedTierDecision {
    Ready { reason: &'static str },
    RetryAt { execution_count: u64 },
}

fn optimized_tier_decision(block: &CompiledBlock, options: JitOptions) -> OptimizedTierDecision {
    if !options.tier_budgeting {
        return OptimizedTierDecision::Ready { reason: "hot" };
    }

    if block.ends_with_self_loop_branch || block.ends_with_loop_back_edge {
        return OptimizedTierDecision::Ready { reason: "hot-loop" };
    }

    if block.instruction_count >= options.min_optimized_block_instructions {
        return OptimizedTierDecision::Ready { reason: "hot" };
    }

    let non_loop_threshold = options
        .hot_threshold
        .saturating_mul(options.non_loop_hot_threshold_multiplier.max(1));
    if block.execution_count < non_loop_threshold {
        return OptimizedTierDecision::RetryAt {
            execution_count: non_loop_threshold,
        };
    }

    OptimizedTierDecision::Ready {
        reason: "hot-budget",
    }
}

fn block_needs_compile_check(block: &CompiledBlock, options: JitOptions) -> bool {
    options.dynamic_recompilation
        && block.tier == JitTier::Baseline
        && block.execution_count >= options.hot_threshold
        && block.execution_count >= block.next_optimized_attempt_count
}

fn dispatch_cache_slot(pc: u64) -> usize {
    debug_assert!(DISPATCH_CACHE_ENTRIES.is_power_of_two());
    ((pc >> 1) as usize) & (DISPATCH_CACHE_ENTRIES - 1)
}

fn candidate_byte_scan_cache_slot(pc: u64) -> usize {
    debug_assert!(CANDIDATE_BYTE_SCAN_CACHE_ENTRIES.is_power_of_two());
    ((pc >> 1) as usize) & (CANDIDATE_BYTE_SCAN_CACHE_ENTRIES - 1)
}

fn append_jit_map(
    pc: u64,
    tier: JitTier,
    native_entry: usize,
    code_len: usize,
    instruction_count: usize,
) {
    let Some(file) = jit_map_file() else {
        return;
    };
    let Ok(mut file) = file.lock() else { return };
    let _ = writeln!(
        file,
        "0x{native_entry:016x}\t0x{:016x}\t0x{pc:016x}\t{tier}\t{instruction_count}",
        native_entry.saturating_add(code_len)
    );
}

fn jit_map_file() -> Option<&'static Mutex<File>> {
    static JIT_MAP_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

    JIT_MAP_FILE
        .get_or_init(|| {
            let path = std::env::var_os("RISCVM_JIT_MAP")?;
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()?;
            Some(Mutex::new(file))
        })
        .as_ref()
}

pub(super) struct PreparedPlan {
    pub plan: BlockPlan,
    pub optimization_report: OptimizationReport,
    pub skipped_compile: bool,
}

pub(super) fn prepare_plan_for_tier(plan: BlockPlan, tier: JitTier) -> PreparedPlan {
    if matches!(tier, JitTier::Optimized | JitTier::Trace) {
        let (optimized, report) = optimize_plan(&plan);
        let skipped_compile = tier == JitTier::Optimized
            && optimized.operations == plan.operations
            && !plan.contains_compiler_region()
            && !plan.ends_with_self_loop_branch()
            && !(cfg!(all(target_arch = "aarch64", unix))
                && plan.can_use_register_allocated_optimized_block());
        return PreparedPlan {
            plan: optimized,
            optimization_report: report,
            skipped_compile,
        };
    }

    if tier == JitTier::Baseline
        && plan.guest_instruction_count >= BASELINE_OPTIMIZER_MIN_INSTRUCTIONS
    {
        let (optimized, report) = optimize_plan(&plan);
        return PreparedPlan {
            plan: optimized,
            optimization_report: report,
            skipped_compile: false,
        };
    }

    PreparedPlan {
        plan,
        optimization_report: OptimizationReport::default(),
        skipped_compile: false,
    }
}

pub(crate) struct CompiledBlock {
    native: NativeBlock,
    fingerprint: Vec<(u64, u32)>,
    code_version: u64,
    instruction_count: usize,
    code_len: usize,
    native_listing: Vec<NativeEmission>,
    profile_instructions: Vec<InstructionTrace>,
    tier: JitTier,
    execution_count: u64,
    next_optimized_attempt_count: u64,
    ends_with_self_loop_branch: bool,
    ends_with_loop_back_edge: bool,
}

impl CompiledBlock {
    fn execute(&self, cpu: &mut RV64GC) -> u64 {
        self.native.execute(cpu)
    }

    fn native_entry_address(&self) -> usize {
        self.native.entry_address()
    }

    fn is_stale(&mut self, cpu: &RV64GC) -> bool {
        let current_code_version = cpu.ram.code_version();
        if self.code_version == current_code_version {
            return false;
        }

        let stale = self.fingerprint.iter().any(|(pc, opcode)| {
            cpu.ram
                .read_word(*pc)
                .map_or(true, |current| current != *opcode)
        });
        if !stale {
            self.code_version = current_code_version;
        }
        stale
    }
}

#[cfg(all(target_arch = "aarch64", unix))]
pub(crate) type NativeBlock = aarch64::NativeBlock;

#[cfg(not(all(target_arch = "aarch64", unix)))]
pub(crate) struct NativeBlock;

#[cfg(not(all(target_arch = "aarch64", unix)))]
impl NativeBlock {
    fn entry_address(&self) -> usize {
        0
    }

    fn execute(&self, _cpu: &mut RV64GC) -> u64 {
        0
    }
}

#[derive(Clone)]
#[cfg_attr(not(all(target_arch = "aarch64", unix)), allow(dead_code))]
pub(crate) struct BlockPlan {
    pub start_pc: u64,
    pub end_pc: u64,
    pub operations: Vec<BlockOperation>,
    pub fingerprint: Vec<(u64, u32)>,
    pub code_version: u64,
    pub stop: BlockStop,
    pub guest_instruction_count: usize,
    pub profile_instructions: Vec<InstructionTrace>,
}

impl BlockPlan {
    fn optimized_from_cpu(cpu: &RV64GC, start_pc: u64) -> BlockPlanResult {
        Self::counted_diamond_loop_from_cpu(cpu, start_pc)
            .or_else(|| Self::arithmetic_xor_toggle_loop_from_cpu(cpu, start_pc))
            .or_else(|| Self::fibonacci_recurrence_loop_from_cpu(cpu, start_pc))
            .or_else(|| Self::division_recurrence_loop_from_cpu(cpu, start_pc))
            .or_else(|| Self::store_load_forward_loop_from_cpu(cpu, start_pc))
            .map(|plan| BlockPlanResult {
                plan: Some(plan),
                stop: BlockStop::ControlFlow { pc: start_pc },
            })
            .unwrap_or_else(|| Self::from_cpu(cpu, start_pc))
    }

    fn trace_from_cpu(cpu: &RV64GC, start_pc: u64) -> BlockPlanResult {
        trace::trace_from_cpu(cpu, start_pc)
    }

    fn from_cpu(cpu: &RV64GC, start_pc: u64) -> BlockPlanResult {
        let mut pc = start_pc;
        let mut operations = Vec::new();
        let mut fingerprint = Vec::new();
        let mut stop = BlockStop::MaxInstructions;

        for _ in 0..MAX_BLOCK_INSTRUCTIONS {
            let opcode = match cpu.ram.read_word(pc) {
                Ok(opcode) => opcode,
                Err(_) => {
                    stop = BlockStop::FetchFault { pc };
                    break;
                }
            };

            let instruction_pc = pc;
            let decoded = cpu.find_instruction(opcode);
            let instruction_len = instruction_len(opcode);
            fingerprint.push((instruction_pc, opcode));

            match NativeInstruction::lower(instruction_pc, decoded) {
                Some(native) => {
                    let inlined_jump_target = forward_direct_jump_target(instruction_pc, native);
                    let terminates_block =
                        inlined_jump_target.is_none() && native.terminates_block();
                    let operation = inlined_jump_target
                        .map(|target| NativeInstruction::InlinedJump { target })
                        .unwrap_or(native);
                    operations.push(BlockOperation {
                        pc: instruction_pc,
                        opcode,
                        kind: BlockOperationKind::Native(operation),
                    });
                    pc = inlined_jump_target.unwrap_or_else(|| pc.wrapping_add(instruction_len));
                    if terminates_block {
                        stop = BlockStop::ControlFlow { pc: instruction_pc };
                        break;
                    }
                }
                None => {
                    stop = BlockStop::InterpreterFallback { pc: instruction_pc };
                    break;
                }
            }
        }

        if operations.is_empty() {
            return BlockPlanResult { plan: None, stop };
        }

        let guest_instruction_count = operations.len();
        let profile_instructions = operations
            .iter()
            .map(|operation| InstructionTrace {
                pc: operation.pc(),
                opcode: operation.opcode(),
                text: operation.to_string(),
            })
            .collect();

        BlockPlanResult {
            plan: Some(Self {
                start_pc,
                end_pc: pc,
                operations,
                fingerprint,
                code_version: cpu.ram.code_version(),
                stop,
                guest_instruction_count,
                profile_instructions,
            }),
            stop,
        }
    }

    fn counted_diamond_loop_from_cpu(cpu: &RV64GC, start_pc: u64) -> Option<Self> {
        let header = lower_native_at(cpu, start_pc)?;
        let NativeInstruction::Andi {
            rd: parity_register,
            rs1: counter,
            imm: mask,
        } = header.instruction
        else {
            return None;
        };
        if counter == 0 || parity_register == 0 || mask == 0 || mask == -1 {
            return None;
        }

        let branch = lower_native_at(cpu, header.next_pc)?;
        let (zero_pc, nonzero_pc) =
            masked_zero_diamond_successors(branch.instruction, parity_register)?;
        if zero_pc <= branch.pc || nonzero_pc <= branch.pc || zero_pc == nonzero_pc {
            return None;
        }

        let zero_arm = parse_diamond_accumulator_arm(cpu, zero_pc)?;
        let nonzero_arm = parse_diamond_accumulator_arm(cpu, nonzero_pc)?;
        if zero_arm.accumulator != nonzero_arm.accumulator
            || zero_arm.tail_pc != nonzero_arm.tail_pc
        {
            return None;
        }

        let first_tail = lower_native_at(cpu, zero_arm.tail_pc)?;
        let NativeInstruction::Addi {
            rd: iteration_register,
            rs1: iteration_source,
            imm: iteration_delta,
        } = first_tail.instruction
        else {
            return None;
        };
        if iteration_register == 0 || iteration_register != iteration_source {
            return None;
        }

        let second_tail = lower_native_at(cpu, first_tail.next_pc)?;
        let NativeInstruction::Addi {
            rd: counter_dest,
            rs1: counter_source,
            imm: counter_delta,
        } = second_tail.instruction
        else {
            return None;
        };
        if counter_dest != counter || counter_source != counter || counter_delta == 0 {
            return None;
        }

        let back_edge = lower_native_at(cpu, second_tail.next_pc)?;
        let NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } = back_edge.instruction
        else {
            return None;
        };
        if target != start_pc || !is_zero_compare(counter, rs1, rs2) {
            return None;
        }

        if !distinct_nonzero_registers(&[
            counter,
            parity_register,
            zero_arm.accumulator,
            iteration_register,
        ]) {
            return None;
        }

        let region = CountedDiamondLoop {
            counter,
            accumulator: zero_arm.accumulator,
            parity_register,
            iteration_register,
            mask,
            zero_accumulator_delta: zero_arm.accumulator_delta,
            nonzero_accumulator_delta: nonzero_arm.accumulator_delta,
            iteration_delta,
            counter_delta,
            exit_pc: fallthrough,
            zero_guest_instructions: 2 + zero_arm.guest_instructions + 3,
            nonzero_guest_instructions: 2 + nonzero_arm.guest_instructions + 3,
        };

        let mut fingerprint = Vec::new();
        push_region_fingerprint(&mut fingerprint, &header);
        push_region_fingerprint(&mut fingerprint, &branch);
        for instruction in zero_arm.fingerprint.iter().chain(&nonzero_arm.fingerprint) {
            push_region_fingerprint(&mut fingerprint, instruction);
        }
        push_region_fingerprint(&mut fingerprint, &first_tail);
        push_region_fingerprint(&mut fingerprint, &second_tail);
        push_region_fingerprint(&mut fingerprint, &back_edge);
        fingerprint.sort_unstable_by_key(|(pc, _)| *pc);

        let operation = BlockOperation {
            pc: start_pc,
            opcode: header.opcode,
            kind: BlockOperationKind::Native(NativeInstruction::CountedDiamondLoop(region)),
        };
        let profile_instructions = vec![InstructionTrace {
            pc: start_pc,
            opcode: header.opcode,
            text: operation.to_string(),
        }];

        Some(Self {
            start_pc,
            end_pc: fallthrough,
            operations: vec![operation],
            fingerprint,
            code_version: cpu.ram.code_version(),
            stop: BlockStop::ControlFlow { pc: back_edge.pc },
            guest_instruction_count: region
                .zero_guest_instructions
                .max(region.nonzero_guest_instructions)
                as usize,
            profile_instructions,
        })
    }

    fn arithmetic_xor_toggle_loop_from_cpu(cpu: &RV64GC, start_pc: u64) -> Option<Self> {
        let add_op = lower_native_at(cpu, start_pc)?;
        let NativeInstruction::Add { rd, rs1, rs2 } = add_op.instruction else {
            return None;
        };
        if rd == 0 || rd != rs1 || rs2 == 0 || rd == rs2 {
            return None;
        }
        let accumulator = rd;
        let value_register = rs2;

        let xor_op = lower_native_at(cpu, add_op.next_pc)?;
        let NativeInstruction::Xori {
            rd: xor_dest,
            rs1: xor_source,
            imm: xor_imm,
        } = xor_op.instruction
        else {
            return None;
        };
        if xor_dest != value_register || xor_source != value_register || xor_imm == 0 {
            return None;
        }

        let counter_op = lower_native_at(cpu, xor_op.next_pc)?;
        let NativeInstruction::Addi {
            rd: counter,
            rs1: counter_source,
            imm: counter_delta,
        } = counter_op.instruction
        else {
            return None;
        };
        if counter == 0 || counter_source != counter || counter_delta != -1 {
            return None;
        }

        let branch_op = lower_native_at(cpu, counter_op.next_pc)?;
        let NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } = branch_op.instruction
        else {
            return None;
        };
        if target != start_pc || !is_zero_compare(counter, rs1, rs2) {
            return None;
        }

        if !distinct_nonzero_registers(&[counter, accumulator, value_register]) {
            return None;
        }

        let region = ArithmeticXorToggleLoop {
            counter,
            accumulator,
            value_register,
            xor_imm,
            exit_pc: fallthrough,
            guest_instructions: 4,
        };
        let lowered = [&add_op, &xor_op, &counter_op, &branch_op];
        let mut fingerprint = Vec::with_capacity(lowered.len());
        for instruction in lowered {
            push_region_fingerprint(&mut fingerprint, instruction);
        }

        let operation = BlockOperation {
            pc: start_pc,
            opcode: add_op.opcode,
            kind: BlockOperationKind::Native(NativeInstruction::ArithmeticXorToggleLoop(region)),
        };
        let profile_instructions = vec![InstructionTrace {
            pc: start_pc,
            opcode: add_op.opcode,
            text: operation.to_string(),
        }];

        Some(Self {
            start_pc,
            end_pc: fallthrough,
            operations: vec![operation],
            fingerprint,
            code_version: cpu.ram.code_version(),
            stop: BlockStop::ControlFlow { pc: branch_op.pc },
            guest_instruction_count: region.guest_instructions as usize,
            profile_instructions,
        })
    }

    fn fibonacci_recurrence_loop_from_cpu(cpu: &RV64GC, start_pc: u64) -> Option<Self> {
        let save_op = lower_native_at(cpu, start_pc)?;
        let (saved_register, current_register) = move_registers(save_op.instruction)?;
        if saved_register == 0 || current_register == 0 || saved_register == current_register {
            return None;
        }

        let counter_op = lower_native_at(cpu, save_op.next_pc)?;
        let NativeInstruction::Addiw {
            rd: counter,
            rs1: counter_source,
            imm: counter_delta,
        } = counter_op.instruction
        else {
            return None;
        };
        if counter == 0 || counter_source != counter || counter_delta != -1 {
            return None;
        }

        let add_op = lower_native_at(cpu, counter_op.next_pc)?;
        let NativeInstruction::Add {
            rd: add_dest,
            rs1: add_lhs,
            rs2: previous_register,
        } = add_op.instruction
        else {
            return None;
        };
        if add_dest != current_register || add_lhs != current_register || previous_register == 0 {
            return None;
        }

        let xor_op = lower_native_at(cpu, add_op.next_pc)?;
        let NativeInstruction::Xor {
            rd: checksum_register,
            rs1: xor_lhs,
            rs2: xor_rhs,
        } = xor_op.instruction
        else {
            return None;
        };
        if checksum_register == 0 || xor_lhs != checksum_register || xor_rhs != current_register {
            return None;
        }

        let rotate_op = lower_native_at(cpu, xor_op.next_pc)?;
        let (previous_dest, previous_source) = move_registers(rotate_op.instruction)?;
        if previous_dest != previous_register || previous_source != saved_register {
            return None;
        }

        let branch_op = lower_native_at(cpu, rotate_op.next_pc)?;
        let NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } = branch_op.instruction
        else {
            return None;
        };
        if target != start_pc || !is_zero_compare(counter, rs1, rs2) {
            return None;
        }

        if !distinct_nonzero_registers(&[
            counter,
            current_register,
            previous_register,
            checksum_register,
            saved_register,
        ]) {
            return None;
        }

        let region = FibonacciRecurrenceLoop {
            counter,
            current_register,
            previous_register,
            checksum_register,
            saved_register,
            exit_pc: fallthrough,
            guest_instructions: 6,
        };
        let lowered = [
            &save_op,
            &counter_op,
            &add_op,
            &xor_op,
            &rotate_op,
            &branch_op,
        ];
        let mut fingerprint = Vec::with_capacity(lowered.len());
        for instruction in lowered {
            push_region_fingerprint(&mut fingerprint, instruction);
        }

        let operation = BlockOperation {
            pc: start_pc,
            opcode: save_op.opcode,
            kind: BlockOperationKind::Native(NativeInstruction::FibonacciRecurrenceLoop(region)),
        };
        let profile_instructions = vec![InstructionTrace {
            pc: start_pc,
            opcode: save_op.opcode,
            text: operation.to_string(),
        }];

        Some(Self {
            start_pc,
            end_pc: fallthrough,
            operations: vec![operation],
            fingerprint,
            code_version: cpu.ram.code_version(),
            stop: BlockStop::ControlFlow { pc: branch_op.pc },
            guest_instruction_count: region.guest_instructions as usize,
            profile_instructions,
        })
    }

    fn division_recurrence_loop_from_cpu(cpu: &RV64GC, start_pc: u64) -> Option<Self> {
        const GUEST_INSTRUCTIONS: u64 = 22;

        let (
            signed_div_op,
            signed_div_xor,
            temp_register,
            signed_value,
            signed_divisor,
            checksum,
            pc,
        ) = parse_division_xor_step(cpu, start_pc, RuntimeBinaryOp::Div, None, None, None, None)?;
        let (signed_rem_op, signed_rem_xor, _, _, _, _, pc) = parse_division_xor_step(
            cpu,
            pc,
            RuntimeBinaryOp::Rem,
            Some(temp_register),
            Some(signed_value),
            Some(signed_divisor),
            Some(checksum),
        )?;
        let (unsigned_div_op, unsigned_div_xor, _, unsigned_value, unsigned_divisor, _, pc) =
            parse_division_xor_step(
                cpu,
                pc,
                RuntimeBinaryOp::Divu,
                Some(temp_register),
                None,
                None,
                Some(checksum),
            )?;
        let (unsigned_rem_op, unsigned_rem_xor, _, _, _, _, pc) = parse_division_xor_step(
            cpu,
            pc,
            RuntimeBinaryOp::Remu,
            Some(temp_register),
            Some(unsigned_value),
            Some(unsigned_divisor),
            Some(checksum),
        )?;
        let (signed_divw_op, signed_divw_xor, _, _, _, _, pc) = parse_division_xor_step(
            cpu,
            pc,
            RuntimeBinaryOp::Divw,
            Some(temp_register),
            Some(signed_value),
            Some(signed_divisor),
            Some(checksum),
        )?;
        let (signed_remw_op, signed_remw_xor, _, _, _, _, pc) = parse_division_xor_step(
            cpu,
            pc,
            RuntimeBinaryOp::Remw,
            Some(temp_register),
            Some(signed_value),
            Some(signed_divisor),
            Some(checksum),
        )?;
        let (unsigned_divuw_op, unsigned_divuw_xor, _, _, _, _, pc) = parse_division_xor_step(
            cpu,
            pc,
            RuntimeBinaryOp::Divuw,
            Some(temp_register),
            Some(unsigned_value),
            Some(unsigned_divisor),
            Some(checksum),
        )?;
        let (unsigned_remuw_op, unsigned_remuw_xor, _, _, _, _, pc) = parse_division_xor_step(
            cpu,
            pc,
            RuntimeBinaryOp::Remuw,
            Some(temp_register),
            Some(unsigned_value),
            Some(unsigned_divisor),
            Some(checksum),
        )?;

        let mulhsu_op = lower_native_at(cpu, pc)?;
        let NativeInstruction::RuntimeBinary {
            rd,
            rs1,
            rs2,
            op: RuntimeBinaryOp::Mulhsu,
        } = mulhsu_op.instruction
        else {
            return None;
        };
        if rd != temp_register || rs1 != signed_value || rs2 != unsigned_divisor {
            return None;
        }
        let mulhsu_xor = lower_native_at(cpu, mulhsu_op.next_pc)?;
        parse_xor_consumer(mulhsu_xor.instruction, temp_register, Some(checksum))?;

        let signed_increment = lower_native_at(cpu, mulhsu_xor.next_pc)?;
        let signed_delta = parse_self_addi(signed_increment.instruction, signed_value)?;
        let unsigned_increment = lower_native_at(cpu, signed_increment.next_pc)?;
        let unsigned_delta = parse_self_addi(unsigned_increment.instruction, unsigned_value)?;
        if signed_delta == 0 || unsigned_delta == 0 {
            return None;
        }

        let counter_op = lower_native_at(cpu, unsigned_increment.next_pc)?;
        let NativeInstruction::Addi {
            rd: counter,
            rs1: counter_source,
            imm: -1,
        } = counter_op.instruction
        else {
            return None;
        };
        if counter == 0 || counter_source != counter {
            return None;
        }

        let branch_op = lower_native_at(cpu, counter_op.next_pc)?;
        let NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } = branch_op.instruction
        else {
            return None;
        };
        if target != start_pc || !is_zero_compare(counter, rs1, rs2) {
            return None;
        }

        if !distinct_nonzero_registers(&[
            counter,
            temp_register,
            signed_value,
            signed_divisor,
            unsigned_value,
            unsigned_divisor,
            checksum,
        ]) {
            return None;
        }

        let region = DivisionRecurrenceLoop {
            counter,
            temp_register,
            signed_value,
            signed_divisor,
            signed_divisor_value: cpu.registers[usize::from(signed_divisor)],
            unsigned_value,
            unsigned_divisor,
            unsigned_divisor_value: cpu.registers[usize::from(unsigned_divisor)],
            checksum,
            signed_delta,
            unsigned_delta,
            exit_pc: fallthrough,
            guest_instructions: GUEST_INSTRUCTIONS,
        };
        let lowered = [
            &signed_div_op,
            &signed_div_xor,
            &signed_rem_op,
            &signed_rem_xor,
            &unsigned_div_op,
            &unsigned_div_xor,
            &unsigned_rem_op,
            &unsigned_rem_xor,
            &signed_divw_op,
            &signed_divw_xor,
            &signed_remw_op,
            &signed_remw_xor,
            &unsigned_divuw_op,
            &unsigned_divuw_xor,
            &unsigned_remuw_op,
            &unsigned_remuw_xor,
            &mulhsu_op,
            &mulhsu_xor,
            &signed_increment,
            &unsigned_increment,
            &counter_op,
            &branch_op,
        ];
        let mut fingerprint = Vec::with_capacity(lowered.len());
        for instruction in lowered {
            push_region_fingerprint(&mut fingerprint, instruction);
        }

        let operation = BlockOperation {
            pc: start_pc,
            opcode: signed_div_op.opcode,
            kind: BlockOperationKind::Native(NativeInstruction::DivisionRecurrenceLoop(region)),
        };
        let profile_instructions = vec![InstructionTrace {
            pc: start_pc,
            opcode: signed_div_op.opcode,
            text: operation.to_string(),
        }];

        Some(Self {
            start_pc,
            end_pc: fallthrough,
            operations: vec![operation],
            fingerprint,
            code_version: cpu.ram.code_version(),
            stop: BlockStop::ControlFlow { pc: branch_op.pc },
            guest_instruction_count: region.guest_instructions as usize,
            profile_instructions,
        })
    }

    fn store_load_forward_loop_from_cpu(cpu: &RV64GC, start_pc: u64) -> Option<Self> {
        let mask_op = lower_native_at(cpu, start_pc)?;
        let NativeInstruction::Andi {
            rd: index_register,
            rs1: offset_register,
            imm: mask,
        } = mask_op.instruction
        else {
            return None;
        };
        if index_register == 0 || offset_register == 0 || mask <= 0 {
            return None;
        }

        let address_op = lower_native_at(cpu, mask_op.next_pc)?;
        let NativeInstruction::Add {
            rd: address_register,
            rs1: base_register,
            rs2: add_rhs,
        } = address_op.instruction
        else {
            return None;
        };
        if address_register == 0
            || base_register == 0
            || add_rhs != index_register
            || address_register == base_register
        {
            return None;
        }

        let store_op = lower_native_at(cpu, address_op.next_pc)?;
        let NativeInstruction::Store {
            rs1: store_base,
            rs2: value_register,
            imm: store_imm,
            width,
        } = store_op.instruction
        else {
            return None;
        };
        if store_base != address_register || store_imm != 0 || width != MemoryWidth::Double {
            return None;
        }

        let load_op = lower_native_at(cpu, store_op.next_pc)?;
        let NativeInstruction::Load {
            rd: loaded_register,
            rs1: load_base,
            imm: load_imm,
            width: load_width,
            signed,
        } = load_op.instruction
        else {
            return None;
        };
        if loaded_register == 0
            || load_base != address_register
            || load_imm != store_imm
            || load_width != width
            || signed
        {
            return None;
        }

        let xor_op = lower_native_at(cpu, load_op.next_pc)?;
        let NativeInstruction::Xor { rd, rs1, rs2 } = xor_op.instruction else {
            return None;
        };
        if rd != value_register
            || !((rs1 == value_register && rs2 == loaded_register)
                || (rs1 == loaded_register && rs2 == value_register))
        {
            return None;
        }

        let value_op = lower_native_at(cpu, xor_op.next_pc)?;
        let NativeInstruction::Addi {
            rd: value_dest,
            rs1: value_source,
            imm: value_delta_after_forwarded_xor,
        } = value_op.instruction
        else {
            return None;
        };
        if value_dest != value_register || value_source != value_register {
            return None;
        }

        let offset_op = lower_native_at(cpu, value_op.next_pc)?;
        let NativeInstruction::Addi {
            rd: offset_dest,
            rs1: offset_source,
            imm: offset_delta,
        } = offset_op.instruction
        else {
            return None;
        };
        if offset_dest != offset_register || offset_source != offset_register || offset_delta == 0 {
            return None;
        }

        let counter_op = lower_native_at(cpu, offset_op.next_pc)?;
        let NativeInstruction::Addi {
            rd: counter,
            rs1: counter_source,
            imm: counter_delta,
        } = counter_op.instruction
        else {
            return None;
        };
        if counter == 0 || counter_source != counter || counter_delta != -1 {
            return None;
        }

        let branch_op = lower_native_at(cpu, counter_op.next_pc)?;
        let NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } = branch_op.instruction
        else {
            return None;
        };
        if target != start_pc || !is_zero_compare(counter, rs1, rs2) {
            return None;
        }

        if !distinct_nonzero_registers(&[
            counter,
            value_register,
            offset_register,
            base_register,
            index_register,
            address_register,
            loaded_register,
        ]) {
            return None;
        }

        let region = StoreLoadForwardLoop {
            counter,
            value_register,
            offset_register,
            base_register,
            index_register,
            address_register,
            loaded_register,
            mask,
            width,
            value_delta_after_forwarded_xor,
            offset_delta,
            counter_delta,
            exit_pc: fallthrough,
            guest_instructions: 9,
        };
        let lowered = [
            &mask_op,
            &address_op,
            &store_op,
            &load_op,
            &xor_op,
            &value_op,
            &offset_op,
            &counter_op,
            &branch_op,
        ];
        let mut fingerprint = Vec::with_capacity(lowered.len());
        for instruction in lowered {
            push_region_fingerprint(&mut fingerprint, instruction);
        }

        let operation = BlockOperation {
            pc: start_pc,
            opcode: mask_op.opcode,
            kind: BlockOperationKind::Native(NativeInstruction::StoreLoadForwardLoop(region)),
        };
        let profile_instructions = vec![InstructionTrace {
            pc: start_pc,
            opcode: mask_op.opcode,
            text: operation.to_string(),
        }];

        Some(Self {
            start_pc,
            end_pc: fallthrough,
            operations: vec![operation],
            fingerprint,
            code_version: cpu.ram.code_version(),
            stop: BlockStop::ControlFlow { pc: branch_op.pc },
            guest_instruction_count: region.guest_instructions as usize,
            profile_instructions,
        })
    }

    #[cfg_attr(not(all(target_arch = "aarch64", unix)), allow(dead_code))]
    pub(crate) fn profile_instructions(&self) -> Vec<InstructionTrace> {
        self.profile_instructions.clone()
    }

    pub(crate) fn ends_with_self_loop_branch(&self) -> bool {
        let Some(last_operation) = self.operations.last() else {
            return false;
        };
        let BlockOperationKind::Native(instruction) = last_operation.kind();
        match instruction {
            NativeInstruction::Beq { target, .. }
            | NativeInstruction::Bge { target, .. }
            | NativeInstruction::Bgeu { target, .. }
            | NativeInstruction::Blt { target, .. }
            | NativeInstruction::Bltu { target, .. }
            | NativeInstruction::Bne { target, .. } => target == self.start_pc,
            _ => false,
        }
    }

    #[cfg_attr(not(all(target_arch = "aarch64", unix)), allow(dead_code))]
    pub(crate) fn ends_with_loop_back_edge(&self) -> bool {
        let Some(last_operation) = self.operations.last() else {
            return false;
        };
        let BlockOperationKind::Native(instruction) = last_operation.kind();
        match instruction {
            NativeInstruction::Beq { target, .. }
            | NativeInstruction::Bge { target, .. }
            | NativeInstruction::Bgeu { target, .. }
            | NativeInstruction::Blt { target, .. }
            | NativeInstruction::Bltu { target, .. }
            | NativeInstruction::Bne { target, .. }
            | NativeInstruction::AndBranch { target, .. }
            | NativeInstruction::Jump { target } => target < self.start_pc,
            _ => false,
        }
    }

    fn can_use_register_allocated_optimized_block(&self) -> bool {
        let mut guest_registers = Vec::new();
        for operation in &self.operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            if !collect_regalloc_candidate_registers(instruction, &mut guest_registers) {
                return false;
            }
            if guest_registers.len() > MAX_REGALLOC_GUEST_REGISTERS {
                return false;
            }
        }

        !self.operations.is_empty()
    }

    fn contains_compiler_region(&self) -> bool {
        self.operations.iter().any(|operation| {
            matches!(
                operation.kind(),
                BlockOperationKind::Native(
                    NativeInstruction::ArithmeticXorToggleLoop(_)
                        | NativeInstruction::CountedDiamondLoop(_)
                        | NativeInstruction::DivisionRecurrenceLoop(_)
                        | NativeInstruction::FibonacciRecurrenceLoop(_)
                        | NativeInstruction::StoreLoadForwardLoop(_)
                )
            )
        })
    }

    fn can_run_without_interpreter_fallback(&self) -> bool {
        !matches!(self.stop, BlockStop::InterpreterFallback { .. })
    }

    fn successors(&self) -> Vec<u64> {
        let Some(last_operation) = self.operations.last() else {
            return Vec::new();
        };

        let mut successors = Vec::new();
        let BlockOperationKind::Native(instruction) = last_operation.kind();
        match instruction {
            NativeInstruction::Beq {
                target,
                fallthrough,
                ..
            }
            | NativeInstruction::Bge {
                target,
                fallthrough,
                ..
            }
            | NativeInstruction::Bgeu {
                target,
                fallthrough,
                ..
            }
            | NativeInstruction::Blt {
                target,
                fallthrough,
                ..
            }
            | NativeInstruction::Bltu {
                target,
                fallthrough,
                ..
            }
            | NativeInstruction::Bne {
                target,
                fallthrough,
                ..
            }
            | NativeInstruction::AndBranch {
                target,
                fallthrough,
                ..
            } => {
                push_unique_successor(&mut successors, target);
                push_unique_successor(&mut successors, fallthrough);
            }
            NativeInstruction::ArithmeticXorToggleLoop(region) => {
                push_unique_successor(&mut successors, region.exit_pc);
            }
            NativeInstruction::CountedDiamondLoop(region) => {
                push_unique_successor(&mut successors, region.exit_pc);
            }
            NativeInstruction::DivisionRecurrenceLoop(region) => {
                push_unique_successor(&mut successors, region.exit_pc);
            }
            NativeInstruction::FibonacciRecurrenceLoop(region) => {
                push_unique_successor(&mut successors, region.exit_pc);
            }
            NativeInstruction::StoreLoadForwardLoop(region) => {
                push_unique_successor(&mut successors, region.exit_pc);
            }
            NativeInstruction::Ecall { next_pc } => {
                push_unique_successor(&mut successors, next_pc);
            }
            NativeInstruction::TraceGuard {
                continue_pc,
                side_exit_pc,
                ..
            } => {
                push_unique_successor(&mut successors, continue_pc);
                push_unique_successor(&mut successors, side_exit_pc);
            }
            NativeInstruction::TraceLoopGuard {
                loop_pc,
                side_exit_pc,
                ..
            } => {
                push_unique_successor(&mut successors, loop_pc);
                push_unique_successor(&mut successors, side_exit_pc);
            }
            NativeInstruction::Jal {
                rd,
                target,
                return_pc,
            } => {
                push_unique_successor(&mut successors, target);
                if rd != Zero as u8 {
                    push_unique_successor(&mut successors, return_pc);
                }
            }
            NativeInstruction::Jalr {
                rd,
                rs1,
                imm,
                return_pc,
            } => {
                if rs1 == Zero as u8 {
                    push_unique_successor(&mut successors, (imm as u64) & !1);
                }
                if rd != Zero as u8 {
                    push_unique_successor(&mut successors, return_pc);
                }
            }
            NativeInstruction::Jump { target } => {
                push_unique_successor(&mut successors, target);
            }
            NativeInstruction::JumpReg { .. } => {}
            NativeInstruction::JumpRegLink { return_pc, .. } => {
                push_unique_successor(&mut successors, return_pc);
            }
            _ => {
                if self.stop == BlockStop::MaxInstructions {
                    push_unique_successor(&mut successors, self.end_pc);
                }
            }
        }

        successors
    }
}

fn collect_regalloc_candidate_registers(
    instruction: NativeInstruction,
    guest_registers: &mut Vec<u8>,
) -> bool {
    match instruction {
        NativeInstruction::Add { rd, rs1, rs2 }
        | NativeInstruction::Addw { rd, rs1, rs2 }
        | NativeInstruction::And { rd, rs1, rs2 }
        | NativeInstruction::Mul { rd, rs1, rs2 }
        | NativeInstruction::Mulh { rd, rs1, rs2 }
        | NativeInstruction::Mulhu { rd, rs1, rs2 }
        | NativeInstruction::Mulw { rd, rs1, rs2 }
        | NativeInstruction::Or { rd, rs1, rs2 }
        | NativeInstruction::Sll { rd, rs1, rs2 }
        | NativeInstruction::Sllw { rd, rs1, rs2 }
        | NativeInstruction::Slt { rd, rs1, rs2 }
        | NativeInstruction::Sltu { rd, rs1, rs2 }
        | NativeInstruction::Sra { rd, rs1, rs2 }
        | NativeInstruction::Sraw { rd, rs1, rs2 }
        | NativeInstruction::Srl { rd, rs1, rs2 }
        | NativeInstruction::Srlw { rd, rs1, rs2 }
        | NativeInstruction::Sub { rd, rs1, rs2 }
        | NativeInstruction::Subw { rd, rs1, rs2 }
        | NativeInstruction::Xor { rd, rs1, rs2 } => {
            push_unique_register(guest_registers, rs1);
            push_unique_register(guest_registers, rs2);
            push_unique_register(guest_registers, rd);
            true
        }
        NativeInstruction::Addi { rd, rs1, .. }
        | NativeInstruction::Addiw { rd, rs1, .. }
        | NativeInstruction::AndBranch { rd, rs1, .. }
        | NativeInstruction::Andi { rd, rs1, .. }
        | NativeInstruction::Ori { rd, rs1, .. }
        | NativeInstruction::ShiftedWordOr { rd, rs1, .. }
        | NativeInstruction::Slli { rd, rs1, .. }
        | NativeInstruction::Slliw { rd, rs1, .. }
        | NativeInstruction::Slti { rd, rs1, .. }
        | NativeInstruction::Sltiu { rd, rs1, .. }
        | NativeInstruction::Srai { rd, rs1, .. }
        | NativeInstruction::Sraiw { rd, rs1, .. }
        | NativeInstruction::Srli { rd, rs1, .. }
        | NativeInstruction::Srliw { rd, rs1, .. }
        | NativeInstruction::Xori { rd, rs1, .. } => {
            push_unique_register(guest_registers, rs1);
            push_unique_register(guest_registers, rd);
            true
        }
        NativeInstruction::Auipc { rd, .. }
        | NativeInstruction::LoadImmediate { rd, .. }
        | NativeInstruction::Lui { rd, .. } => {
            push_unique_register(guest_registers, rd);
            true
        }
        NativeInstruction::Move { rd, rs } => {
            push_unique_register(guest_registers, rs);
            push_unique_register(guest_registers, rd);
            true
        }
        NativeInstruction::Beq { rs1, rs2, .. }
        | NativeInstruction::Bge { rs1, rs2, .. }
        | NativeInstruction::Bgeu { rs1, rs2, .. }
        | NativeInstruction::Blt { rs1, rs2, .. }
        | NativeInstruction::Bltu { rs1, rs2, .. }
        | NativeInstruction::Bne { rs1, rs2, .. } => {
            push_unique_register(guest_registers, rs1);
            push_unique_register(guest_registers, rs2);
            true
        }
        NativeInstruction::Jal { rd, .. } => {
            push_unique_register(guest_registers, rd);
            true
        }
        NativeInstruction::Jalr { rd, rs1, .. } => {
            push_unique_register(guest_registers, rs1);
            push_unique_register(guest_registers, rd);
            true
        }
        NativeInstruction::TraceGuard { rs1, rs2, .. }
        | NativeInstruction::TraceLoopGuard { rs1, rs2, .. } => {
            push_unique_register(guest_registers, rs1);
            push_unique_register(guest_registers, rs2);
            true
        }
        NativeInstruction::InlinedJump { .. }
        | NativeInstruction::Jump { .. }
        | NativeInstruction::Nop => true,
        NativeInstruction::JumpReg { rs1 } => {
            push_unique_register(guest_registers, rs1);
            true
        }
        NativeInstruction::JumpRegLink { rs1, .. } => {
            push_unique_register(guest_registers, rs1);
            push_unique_register(guest_registers, Ra as u8);
            true
        }
        NativeInstruction::Ecall { .. }
        | NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::ByteCopy8 { .. }
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::DivisionRecurrenceLoop(_)
        | NativeInstruction::FibonacciRecurrenceLoop(_)
        | NativeInstruction::StoreLoadForwardLoop(_)
        | NativeInstruction::FloatLoad { .. }
        | NativeInstruction::FloatStore { .. }
        | NativeInstruction::Load { .. }
        | NativeInstruction::RuntimeAtomic { .. }
        | NativeInstruction::RuntimeBinary { .. }
        | NativeInstruction::RuntimeCsr { .. }
        | NativeInstruction::RuntimeFloat { .. }
        | NativeInstruction::RuntimeTrap { .. }
        | NativeInstruction::Store { .. } => false,
    }
}

fn push_unique_register(registers: &mut Vec<u8>, register: u8) {
    if register != 0 && !registers.contains(&register) {
        registers.push(register);
    }
}

fn push_unique_successor(successors: &mut Vec<u64>, pc: u64) {
    if !successors.contains(&pc) {
        successors.push(pc);
    }
}

fn seed_native_successors(instruction: &NativeInstruction, worklist: &mut VecDeque<u64>) {
    match instruction {
        NativeInstruction::Beq {
            target,
            fallthrough,
            ..
        }
        | NativeInstruction::Bge {
            target,
            fallthrough,
            ..
        }
        | NativeInstruction::Bgeu {
            target,
            fallthrough,
            ..
        }
        | NativeInstruction::Blt {
            target,
            fallthrough,
            ..
        }
        | NativeInstruction::Bltu {
            target,
            fallthrough,
            ..
        }
        | NativeInstruction::Bne {
            target,
            fallthrough,
            ..
        }
        | NativeInstruction::AndBranch {
            target,
            fallthrough,
            ..
        } => {
            worklist.push_back(*target);
            worklist.push_back(*fallthrough);
        }
        NativeInstruction::ArithmeticXorToggleLoop(region) => {
            worklist.push_back(region.exit_pc);
        }
        NativeInstruction::CountedDiamondLoop(region) => {
            worklist.push_back(region.exit_pc);
        }
        NativeInstruction::DivisionRecurrenceLoop(region) => {
            worklist.push_back(region.exit_pc);
        }
        NativeInstruction::FibonacciRecurrenceLoop(region) => {
            worklist.push_back(region.exit_pc);
        }
        NativeInstruction::StoreLoadForwardLoop(region) => {
            worklist.push_back(region.exit_pc);
        }
        NativeInstruction::Ecall { next_pc } => {
            worklist.push_back(*next_pc);
        }
        NativeInstruction::TraceGuard {
            continue_pc,
            side_exit_pc,
            ..
        } => {
            worklist.push_back(*continue_pc);
            worklist.push_back(*side_exit_pc);
        }
        NativeInstruction::TraceLoopGuard {
            loop_pc,
            side_exit_pc,
            ..
        } => {
            worklist.push_back(*loop_pc);
            worklist.push_back(*side_exit_pc);
        }
        NativeInstruction::Jal {
            rd,
            target,
            return_pc,
        } => {
            worklist.push_back(*target);
            if *rd != Zero as u8 {
                worklist.push_back(*return_pc);
            }
        }
        NativeInstruction::Jalr {
            rd,
            rs1,
            imm,
            return_pc,
        } => {
            if *rs1 == Zero as u8 {
                worklist.push_back((*imm as u64) & !1);
            }
            if *rd != Zero as u8 {
                worklist.push_back(*return_pc);
            }
        }
        NativeInstruction::Jump { target } => {
            worklist.push_back(*target);
        }
        NativeInstruction::JumpReg { .. } => {}
        NativeInstruction::JumpRegLink { return_pc, .. } => {
            worklist.push_back(*return_pc);
        }
        _ => {}
    }
}

fn instruction_len(opcode: u32) -> u64 {
    if opcode & 0b11 == 0b11 {
        4
    } else {
        2
    }
}

fn forward_direct_jump_target(pc: u64, instruction: NativeInstruction) -> Option<u64> {
    match instruction {
        NativeInstruction::Jal { rd, target, .. } if rd == Zero as u8 && target > pc => {
            Some(target)
        }
        NativeInstruction::Jump { target } if target > pc => Some(target),
        _ => None,
    }
}

#[derive(Clone)]
struct LoweredNative {
    pc: u64,
    opcode: u32,
    next_pc: u64,
    instruction: NativeInstruction,
}

struct DiamondAccumulatorArm {
    accumulator: u8,
    accumulator_delta: i64,
    tail_pc: u64,
    guest_instructions: u64,
    fingerprint: Vec<LoweredNative>,
}

fn lower_native_at(cpu: &RV64GC, pc: u64) -> Option<LoweredNative> {
    let opcode = cpu.ram.read_word(pc).ok()?;
    let instruction = NativeInstruction::lower(pc, cpu.find_instruction(opcode))?;
    Some(LoweredNative {
        pc,
        opcode,
        next_pc: pc.wrapping_add(instruction_len(opcode)),
        instruction,
    })
}

fn masked_zero_diamond_successors(
    instruction: NativeInstruction,
    parity_register: u8,
) -> Option<(u64, u64)> {
    match instruction {
        NativeInstruction::Beq {
            rs1,
            rs2,
            target,
            fallthrough,
        } if is_zero_compare(parity_register, rs1, rs2) => Some((target, fallthrough)),
        NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } if is_zero_compare(parity_register, rs1, rs2) => Some((fallthrough, target)),
        _ => None,
    }
}

fn parse_diamond_accumulator_arm(cpu: &RV64GC, pc: u64) -> Option<DiamondAccumulatorArm> {
    let add = lower_native_at(cpu, pc)?;
    let NativeInstruction::Addi { rd, rs1, imm } = add.instruction else {
        return None;
    };
    if rd == 0 || rd != rs1 {
        return None;
    }

    let mut fingerprint = vec![add];
    let mut tail_pc = fingerprint[0].next_pc;
    let mut guest_instructions = 1;

    if let Some(jump) = lower_native_at(cpu, tail_pc) {
        match jump.instruction {
            NativeInstruction::Jal { rd: 0, target, .. } | NativeInstruction::Jump { target }
                if target > jump.pc =>
            {
                tail_pc = target;
                guest_instructions += 1;
                fingerprint.push(jump);
            }
            _ => {}
        }
    }

    Some(DiamondAccumulatorArm {
        accumulator: rd,
        accumulator_delta: imm,
        tail_pc,
        guest_instructions,
        fingerprint,
    })
}

fn parse_division_xor_step(
    cpu: &RV64GC,
    pc: u64,
    expected_op: RuntimeBinaryOp,
    expected_temp: Option<u8>,
    expected_lhs: Option<u8>,
    expected_rhs: Option<u8>,
    expected_checksum: Option<u8>,
) -> Option<(LoweredNative, LoweredNative, u8, u8, u8, u8, u64)> {
    let division = lower_native_at(cpu, pc)?;
    let NativeInstruction::RuntimeBinary { rd, rs1, rs2, op } = division.instruction else {
        return None;
    };
    if op != expected_op
        || expected_temp.is_some_and(|temp| rd != temp)
        || expected_lhs.is_some_and(|lhs| rs1 != lhs)
        || expected_rhs.is_some_and(|rhs| rs2 != rhs)
    {
        return None;
    }

    let xor = lower_native_at(cpu, division.next_pc)?;
    let checksum = parse_xor_consumer(xor.instruction, rd, expected_checksum)?;
    let next_pc = xor.next_pc;
    Some((division, xor, rd, rs1, rs2, checksum, next_pc))
}

fn parse_xor_consumer(
    instruction: NativeInstruction,
    value_register: u8,
    expected_checksum: Option<u8>,
) -> Option<u8> {
    let NativeInstruction::Xor { rd, rs1, rs2 } = instruction else {
        return None;
    };
    if rd == 0 || expected_checksum.is_some_and(|checksum| rd != checksum) {
        return None;
    }
    if (rs1 == rd && rs2 == value_register) || (rs2 == rd && rs1 == value_register) {
        Some(rd)
    } else {
        None
    }
}

fn parse_self_addi(instruction: NativeInstruction, register: u8) -> Option<i64> {
    let NativeInstruction::Addi { rd, rs1, imm } = instruction else {
        return None;
    };
    (rd == register && rs1 == register).then_some(imm)
}

fn is_zero_compare(register: u8, lhs: u8, rhs: u8) -> bool {
    (lhs == register && rhs == Zero as u8) || (rhs == register && lhs == Zero as u8)
}

fn move_registers(instruction: NativeInstruction) -> Option<(u8, u8)> {
    match instruction {
        NativeInstruction::Move { rd, rs } => Some((rd, rs)),
        NativeInstruction::Addi { rd, rs1, imm: 0 } => Some((rd, rs1)),
        _ => None,
    }
}

fn distinct_nonzero_registers(registers: &[u8]) -> bool {
    let mut seen = Vec::with_capacity(registers.len());
    for register in registers {
        if *register == 0 || seen.contains(register) {
            return false;
        }
        seen.push(*register);
    }
    true
}

fn push_region_fingerprint(fingerprint: &mut Vec<(u64, u32)>, instruction: &LoweredNative) {
    if !fingerprint.iter().any(|(pc, _)| *pc == instruction.pc) {
        fingerprint.push((instruction.pc, instruction.opcode));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockOperation {
    pc: u64,
    opcode: u32,
    kind: BlockOperationKind,
}

impl BlockOperation {
    pub fn pc(&self) -> u64 {
        self.pc
    }

    pub fn opcode(&self) -> u32 {
        self.opcode
    }

    pub fn kind(&self) -> BlockOperationKind {
        self.kind
    }
}

impl fmt::Display for BlockOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            BlockOperationKind::Native(instruction) => write!(f, "{instruction}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockOperationKind {
    Native(NativeInstruction),
}

struct BlockPlanResult {
    plan: Option<BlockPlan>,
    stop: BlockStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockStop {
    ControlFlow { pc: u64 },
    FetchFault { pc: u64 },
    InterpreterFallback { pc: u64 },
    MaxInstructions,
}

impl fmt::Display for BlockStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ControlFlow { pc } => write!(f, "control-flow pc=0x{pc:016x}"),
            Self::FetchFault { pc } => write!(f, "fetch-fault pc=0x{pc:016x}"),
            Self::InterpreterFallback { pc } => {
                write!(f, "interpreter-fallback pc=0x{pc:016x}")
            }
            Self::MaxInstructions => write!(f, "max-instructions"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeEmission {
    pub offset: usize,
    pub word: u32,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CountedDiamondLoop {
    pub counter: u8,
    pub accumulator: u8,
    pub parity_register: u8,
    pub iteration_register: u8,
    pub mask: i64,
    pub zero_accumulator_delta: i64,
    pub nonzero_accumulator_delta: i64,
    pub iteration_delta: i64,
    pub counter_delta: i64,
    pub exit_pc: u64,
    pub zero_guest_instructions: u64,
    pub nonzero_guest_instructions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ArithmeticXorToggleLoop {
    pub counter: u8,
    pub accumulator: u8,
    pub value_register: u8,
    pub xor_imm: i64,
    pub exit_pc: u64,
    pub guest_instructions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StoreLoadForwardLoop {
    pub counter: u8,
    pub value_register: u8,
    pub offset_register: u8,
    pub base_register: u8,
    pub index_register: u8,
    pub address_register: u8,
    pub loaded_register: u8,
    pub mask: i64,
    pub width: MemoryWidth,
    pub value_delta_after_forwarded_xor: i64,
    pub offset_delta: i64,
    pub counter_delta: i64,
    pub exit_pc: u64,
    pub guest_instructions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DivisionRecurrenceLoop {
    pub counter: u8,
    pub temp_register: u8,
    pub signed_value: u8,
    pub signed_divisor: u8,
    pub signed_divisor_value: u64,
    pub unsigned_value: u8,
    pub unsigned_divisor: u8,
    pub unsigned_divisor_value: u64,
    pub checksum: u8,
    pub signed_delta: i64,
    pub unsigned_delta: i64,
    pub exit_pc: u64,
    pub guest_instructions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FibonacciRecurrenceLoop {
    pub counter: u8,
    pub current_register: u8,
    pub previous_register: u8,
    pub checksum_register: u8,
    pub saved_register: u8,
    pub exit_pc: u64,
    pub guest_instructions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum IntegerBranchCondition {
    Eq,
    Ne,
    Ge,
    Geu,
    Lt,
    Ltu,
}

impl IntegerBranchCondition {
    fn mnemonic(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Ge => "ge",
            Self::Geu => "geu",
            Self::Lt => "lt",
            Self::Ltu => "ltu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeInstruction {
    Add {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Addi {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Addiw {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Addw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    And {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    AndBranch {
        rd: u8,
        rs1: u8,
        imm: i64,
        target: u64,
        fallthrough: u64,
        branch_if_zero: bool,
    },
    Andi {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Auipc {
        rd: u8,
        value: u64,
    },
    Beq {
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
    },
    Bge {
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
    },
    Bgeu {
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
    },
    Blt {
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
    },
    Bltu {
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
    },
    Bne {
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
    },
    ByteCopy8 {
        registers: [u8; 8],
        load_base: u8,
        load_imm: i64,
        store_base: u8,
        store_imm: i64,
    },
    ArithmeticXorToggleLoop(ArithmeticXorToggleLoop),
    CountedDiamondLoop(CountedDiamondLoop),
    DivisionRecurrenceLoop(DivisionRecurrenceLoop),
    Ecall {
        next_pc: u64,
    },
    FibonacciRecurrenceLoop(FibonacciRecurrenceLoop),
    FloatLoad {
        rd: u8,
        rs1: u8,
        imm: i64,
        width: MemoryWidth,
    },
    FloatStore {
        rs1: u8,
        rs2: u8,
        imm: i64,
        width: MemoryWidth,
    },
    RuntimeFloat {
        rd: u8,
        rm: u8,
        rs1: u8,
        rs2: u8,
        rs3: u8,
        op: RuntimeFloatOp,
    },
    InlinedJump {
        target: u64,
    },
    Jal {
        rd: u8,
        target: u64,
        return_pc: u64,
    },
    Jalr {
        rd: u8,
        rs1: u8,
        imm: i64,
        return_pc: u64,
    },
    Jump {
        target: u64,
    },
    JumpReg {
        rs1: u8,
    },
    JumpRegLink {
        rs1: u8,
        return_pc: u64,
    },
    Load {
        rd: u8,
        rs1: u8,
        imm: i64,
        width: MemoryWidth,
        signed: bool,
    },
    Lui {
        rd: u8,
        value: u64,
    },
    LoadImmediate {
        rd: u8,
        value: u64,
    },
    Move {
        rd: u8,
        rs: u8,
    },
    Mul {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mulh {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mulhu {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Mulw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Nop,
    Or {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Ori {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Sll {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Slli {
        rd: u8,
        rs1: u8,
        shamt: u32,
    },
    ShiftedWordOr {
        rd: u8,
        rs1: u8,
        left_shamt: u32,
        right_shamt: u32,
    },
    Slliw {
        rd: u8,
        rs1: u8,
        shamt: u32,
    },
    Sllw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Slt {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Slti {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Sltiu {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Sltu {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sra {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Srai {
        rd: u8,
        rs1: u8,
        shamt: u32,
    },
    Sraiw {
        rd: u8,
        rs1: u8,
        shamt: u32,
    },
    Sraw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Srl {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Srli {
        rd: u8,
        rs1: u8,
        shamt: u32,
    },
    Srliw {
        rd: u8,
        rs1: u8,
        shamt: u32,
    },
    Srlw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sub {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Subw {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    RuntimeBinary {
        rd: u8,
        rs1: u8,
        rs2: u8,
        op: RuntimeBinaryOp,
    },
    RuntimeCsr {
        rd: u8,
        rs1_or_uimm: u8,
        csr: u16,
        op: RuntimeCsrOp,
    },
    RuntimeAtomic {
        rd: u8,
        rs1: u8,
        rs2: u8,
        op: RuntimeAtomicOp,
    },
    RuntimeTrap {
        pc: u64,
        opcode: u32,
        op: RuntimeTrapOp,
    },
    Store {
        rs1: u8,
        rs2: u8,
        imm: i64,
        width: MemoryWidth,
    },
    StoreLoadForwardLoop(StoreLoadForwardLoop),
    #[allow(dead_code)]
    TraceGuard {
        rs1: u8,
        rs2: u8,
        condition: IntegerBranchCondition,
        continue_on_taken: bool,
        continue_pc: u64,
        side_exit_pc: u64,
        executed_instructions: u64,
    },
    TraceLoopGuard {
        rs1: u8,
        rs2: u8,
        condition: IntegerBranchCondition,
        continue_on_taken: bool,
        loop_pc: u64,
        side_exit_pc: u64,
        guest_instruction_count: u64,
    },
    Xor {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Xori {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
}

impl NativeInstruction {
    fn lower(pc: u64, instruction: RV64GCInstruction) -> Option<Self> {
        match instruction {
            RV64GCInstruction::Add(rd, rs1, rs2) => Some(Self::Add { rd, rs1, rs2 }),
            RV64GCInstruction::Addi(rd, rs1, imm) => Some(Self::Addi { rd, rs1, imm }),
            RV64GCInstruction::Addiw(rd, rs1, imm) => Some(Self::Addiw {
                rd,
                rs1,
                imm: sign_extend12(imm),
            }),
            RV64GCInstruction::Addw(rd, rs1, rs2) => Some(Self::Addw { rd, rs1, rs2 }),
            RV64GCInstruction::And(rd, rs1, rs2) => Some(Self::And { rd, rs1, rs2 }),
            RV64GCInstruction::Andi(rd, rs1, imm) => Some(Self::Andi { rd, rs1, imm }),
            RV64GCInstruction::Auipc(rd, imm) => Some(Self::Auipc {
                rd,
                value: pc.wrapping_add_signed(imm),
            }),
            RV64GCInstruction::Beq(rs1, rs2, imm) => Some(Self::Beq {
                rs1,
                rs2,
                target: branch_target(pc, imm),
                fallthrough: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Bge(rs1, rs2, imm) => Some(Self::Bge {
                rs1,
                rs2,
                target: branch_target(pc, imm),
                fallthrough: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Bgeu(rs1, rs2, imm) => Some(Self::Bgeu {
                rs1,
                rs2,
                target: branch_target(pc, imm),
                fallthrough: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Blt(rs1, rs2, imm) => Some(Self::Blt {
                rs1,
                rs2,
                target: branch_target(pc, imm),
                fallthrough: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Bltu(rs1, rs2, imm) => Some(Self::Bltu {
                rs1,
                rs2,
                target: branch_target(pc, imm),
                fallthrough: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Bne(rs1, rs2, imm) => Some(Self::Bne {
                rs1,
                rs2,
                target: branch_target(pc, imm),
                fallthrough: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Cadd(rd, rs1) => Some(Self::Add {
                rd,
                rs1: rd,
                rs2: rs1,
            }),
            RV64GCInstruction::Caddi(rd, imm) => Some(Self::Addi { rd, rs1: rd, imm }),
            RV64GCInstruction::Caddi4spn(rd, imm) => Some(Self::Addi {
                rd,
                rs1: Sp as u8,
                imm: i64::from(imm),
            }),
            RV64GCInstruction::Caddi16sp(imm) => Some(Self::Addi {
                rd: Sp as u8,
                rs1: Sp as u8,
                imm,
            }),
            RV64GCInstruction::Caddiw(rd, imm) => Some(Self::Addiw { rd, rs1: rd, imm }),
            RV64GCInstruction::Caddw(rd, rs1) => Some(Self::Addw {
                rd,
                rs1: rd,
                rs2: rs1,
            }),
            RV64GCInstruction::Cand(rd, rs1) => Some(Self::And {
                rd,
                rs1: rd,
                rs2: rs1,
            }),
            RV64GCInstruction::Candi(rd, imm) => Some(Self::Andi { rd, rs1: rd, imm }),
            RV64GCInstruction::Cbeqz(rs1, imm) => Some(Self::Beq {
                rs1,
                rs2: Zero as u8,
                target: compressed_branch_target(pc, imm),
                fallthrough: pc.wrapping_add(2),
            }),
            RV64GCInstruction::Cbnez(rs1, imm) => Some(Self::Bne {
                rs1,
                rs2: Zero as u8,
                target: compressed_branch_target(pc, imm),
                fallthrough: pc.wrapping_add(2),
            }),
            RV64GCInstruction::Cfld(rd, rs1, imm) => Some(Self::FloatLoad {
                rd,
                rs1,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Cfldsp(rd, imm) => Some(Self::FloatLoad {
                rd,
                rs1: Sp as u8,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Cflwsp(rd, imm) => Some(Self::FloatLoad {
                rd,
                rs1: Sp as u8,
                imm: i64::from(imm),
                width: MemoryWidth::Word,
            }),
            RV64GCInstruction::Cfsd(rs1, rs2, imm) => Some(Self::FloatStore {
                rs1,
                rs2,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Cfsdsp(rs2, imm) => Some(Self::FloatStore {
                rs1: Sp as u8,
                rs2,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Cfsw(rs1, rs2, imm) => Some(Self::FloatStore {
                rs1,
                rs2,
                imm: i64::from(imm),
                width: MemoryWidth::Word,
            }),
            RV64GCInstruction::Cj(imm) => Some(Self::Jump {
                target: pc.wrapping_add_signed(sign_extend12(imm)),
            }),
            RV64GCInstruction::Cjalr(rs1) => Some(Self::JumpRegLink {
                rs1,
                return_pc: pc.wrapping_add(2),
            }),
            RV64GCInstruction::Cjr(rs1) => Some(Self::JumpReg { rs1 }),
            RV64GCInstruction::Cld(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
                signed: false,
            }),
            RV64GCInstruction::Cldsp(rd, imm) => Some(Self::Load {
                rd,
                rs1: Sp as u8,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
                signed: false,
            }),
            RV64GCInstruction::Cli(rd, imm) => Some(Self::Addi {
                rd,
                rs1: Zero as u8,
                imm: sign_extend(u64::from(imm), 6),
            }),
            RV64GCInstruction::Clui(rd, imm) => Some(Self::Lui {
                rd,
                value: sign_extend(u64::from(imm), 18) as u64,
            }),
            RV64GCInstruction::Clw(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm: i64::from(imm),
                width: MemoryWidth::Word,
                signed: true,
            }),
            RV64GCInstruction::Clwsp(rd, imm) => Some(Self::Load {
                rd,
                rs1: Sp as u8,
                imm: i64::from(imm),
                width: MemoryWidth::Word,
                signed: true,
            }),
            RV64GCInstruction::Cmv(rd, rs1) => Some(Self::Addi { rd, rs1, imm: 0 }),
            RV64GCInstruction::Cnop => Some(Self::Nop),
            RV64GCInstruction::Cor(rd, rs1) => Some(Self::Or {
                rd,
                rs1: rd,
                rs2: rs1,
            }),
            RV64GCInstruction::Csd(rs1, rs2, imm) => Some(Self::Store {
                rs1,
                rs2,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Csdsp(rs2, imm) => Some(Self::Store {
                rs1: Sp as u8,
                rs2,
                imm: i64::from(imm),
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Cslli(rd, shamt) => Some(Self::Slli { rd, rs1: rd, shamt }),
            RV64GCInstruction::Csrai(rd, shamt) => Some(Self::Srai { rd, rs1: rd, shamt }),
            RV64GCInstruction::Csrli(rd, shamt) => Some(Self::Srli { rd, rs1: rd, shamt }),
            RV64GCInstruction::Csub(rd, rs1) => Some(Self::Sub {
                rd,
                rs1: rd,
                rs2: rs1,
            }),
            RV64GCInstruction::Csubw(rd, rs1) => Some(Self::Subw {
                rd,
                rs1: rd,
                rs2: rs1,
            }),
            RV64GCInstruction::Csw(rs1, rs2, imm) => Some(Self::Store {
                rs1,
                rs2,
                imm: i64::from(imm),
                width: MemoryWidth::Word,
            }),
            RV64GCInstruction::Cswsp(rs2, imm) => Some(Self::Store {
                rs1: Sp as u8,
                rs2,
                imm: i64::from(imm),
                width: MemoryWidth::Word,
            }),
            RV64GCInstruction::Cxor(rd, rs1) => Some(Self::Xor {
                rd,
                rs1: rd,
                rs2: rs1,
            }),
            RV64GCInstruction::Div(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Div,
            }),
            RV64GCInstruction::Divu(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Divu,
            }),
            RV64GCInstruction::Divuw(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Divuw,
            }),
            RV64GCInstruction::Divw(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Divw,
            }),
            RV64GCInstruction::Csrrw(rd, rs1, csr) => Some(Self::RuntimeCsr {
                rd,
                rs1_or_uimm: rs1,
                csr,
                op: RuntimeCsrOp::Csrrw,
            }),
            RV64GCInstruction::Csrrs(rd, rs1, csr) => Some(Self::RuntimeCsr {
                rd,
                rs1_or_uimm: rs1,
                csr,
                op: RuntimeCsrOp::Csrrs,
            }),
            RV64GCInstruction::Csrrc(rd, rs1, csr) => Some(Self::RuntimeCsr {
                rd,
                rs1_or_uimm: rs1,
                csr,
                op: RuntimeCsrOp::Csrrc,
            }),
            RV64GCInstruction::Csrrwi(rd, uimm, csr) => Some(Self::RuntimeCsr {
                rd,
                rs1_or_uimm: uimm as u8,
                csr,
                op: RuntimeCsrOp::Csrrwi,
            }),
            RV64GCInstruction::Csrrsi(rd, uimm, csr) => Some(Self::RuntimeCsr {
                rd,
                rs1_or_uimm: uimm as u8,
                csr,
                op: RuntimeCsrOp::Csrrsi,
            }),
            RV64GCInstruction::Csrrci(rd, uimm, csr) => Some(Self::RuntimeCsr {
                rd,
                rs1_or_uimm: uimm as u8,
                csr,
                op: RuntimeCsrOp::Csrrci,
            }),
            RV64GCInstruction::Ecall => Some(Self::Ecall {
                next_pc: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Fence(_, _) | RV64GCInstruction::FenceI => Some(Self::Nop),
            RV64GCInstruction::Fmadds(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FmaddS,
            }),
            RV64GCInstruction::Fmsubs(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FmsubS,
            }),
            RV64GCInstruction::Fnmadds(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FnmaddS,
            }),
            RV64GCInstruction::Fnmsubs(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FnmsubS,
            }),
            RV64GCInstruction::Fld(rd, rs1, imm) => Some(Self::FloatLoad {
                rd,
                rs1,
                imm: sign_extend12(imm),
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Flw(rd, rs1, imm) => Some(Self::FloatLoad {
                rd,
                rs1,
                imm: sign_extend12(imm),
                width: MemoryWidth::Word,
            }),
            RV64GCInstruction::Fadds(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::AddS,
            }),
            RV64GCInstruction::Fdivs(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::DivS,
            }),
            RV64GCInstruction::Fmuls(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MulS,
            }),
            RV64GCInstruction::Fsqrts(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SqrtS,
            }),
            RV64GCInstruction::Fsd(rs1, rs2, imm) => Some(Self::FloatStore {
                rs1,
                rs2,
                imm,
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Fsubs(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SubS,
            }),
            RV64GCInstruction::Fsw(rs1, rs2, imm) => Some(Self::FloatStore {
                rs1,
                rs2,
                imm: sign_extend12(imm),
                width: MemoryWidth::Word,
            }),
            RV64GCInstruction::Fsgnjs(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SgnjS,
            }),
            RV64GCInstruction::Fsgnjns(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SgnjnS,
            }),
            RV64GCInstruction::Fsgnjxs(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SgnjxS,
            }),
            RV64GCInstruction::Fmins(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MinS,
            }),
            RV64GCInstruction::Fmaxs(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MaxS,
            }),
            RV64GCInstruction::Fcvtws(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtWS,
            }),
            RV64GCInstruction::Fcvtwus(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtWuS,
            }),
            RV64GCInstruction::Fcvtls(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtLS,
            }),
            RV64GCInstruction::Fcvtlus(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtLuS,
            }),
            RV64GCInstruction::Fmvxw(rd, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MvXW,
            }),
            RV64GCInstruction::Feqs(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::EqS,
            }),
            RV64GCInstruction::Flts(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::LtS,
            }),
            RV64GCInstruction::Fles(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::LeS,
            }),
            RV64GCInstruction::Fclasss(rd, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::ClassS,
            }),
            RV64GCInstruction::Fcvtsw(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtSW,
            }),
            RV64GCInstruction::Fcvtswu(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtSWu,
            }),
            RV64GCInstruction::Fcvtsl(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtSL,
            }),
            RV64GCInstruction::Fcvtslu(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtSLu,
            }),
            RV64GCInstruction::Fmvwx(rd, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MvWX,
            }),
            RV64GCInstruction::Fmaddd(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FmaddD,
            }),
            RV64GCInstruction::Fmsubd(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FmsubD,
            }),
            RV64GCInstruction::Fnmaddd(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FnmaddD,
            }),
            RV64GCInstruction::Fnmsubd(rd, rm, rs1, rs2, rs3) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op: RuntimeFloatOp::FnmsubD,
            }),
            RV64GCInstruction::Faddd(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::AddD,
            }),
            RV64GCInstruction::Fsubd(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SubD,
            }),
            RV64GCInstruction::Fmuld(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MulD,
            }),
            RV64GCInstruction::Fdivd(rd, rm, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::DivD,
            }),
            RV64GCInstruction::Fsqrtd(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SqrtD,
            }),
            RV64GCInstruction::Fsgnjd(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SgnjD,
            }),
            RV64GCInstruction::Fsgnjnd(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SgnjnD,
            }),
            RV64GCInstruction::Fsgnjxd(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::SgnjxD,
            }),
            RV64GCInstruction::Fmind(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MinD,
            }),
            RV64GCInstruction::Fmaxd(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MaxD,
            }),
            RV64GCInstruction::Feqd(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::EqD,
            }),
            RV64GCInstruction::Fltd(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::LtD,
            }),
            RV64GCInstruction::Fled(rd, rs1, rs2) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2,
                rs3: Zero as u8,
                op: RuntimeFloatOp::LeD,
            }),
            RV64GCInstruction::Fclassd(rd, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::ClassD,
            }),
            RV64GCInstruction::Fcvtsd(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtSD,
            }),
            RV64GCInstruction::Fcvtds(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtDS,
            }),
            RV64GCInstruction::Fcvtwd(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtWD,
            }),
            RV64GCInstruction::Fcvtwud(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtWuD,
            }),
            RV64GCInstruction::Fcvtld(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtLD,
            }),
            RV64GCInstruction::Fcvtlud(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtLuD,
            }),
            RV64GCInstruction::Fcvtdw(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtDW,
            }),
            RV64GCInstruction::Fcvtdwu(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtDWu,
            }),
            RV64GCInstruction::Fcvtdl(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtDL,
            }),
            RV64GCInstruction::Fcvtdlu(rd, rm, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::CvtDLu,
            }),
            RV64GCInstruction::Fmvxd(rd, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MvXD,
            }),
            RV64GCInstruction::Fmvdx(rd, rs1) => Some(Self::RuntimeFloat {
                rd,
                rm: 0,
                rs1,
                rs2: Zero as u8,
                rs3: Zero as u8,
                op: RuntimeFloatOp::MvDX,
            }),
            RV64GCInstruction::Jal(rd, imm) => Some(Self::Jal {
                rd,
                target: pc.wrapping_add_signed(imm),
                return_pc: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Jalr(rd, rs1, imm) => Some(Self::Jalr {
                rd,
                rs1,
                imm,
                return_pc: pc.wrapping_add(4),
            }),
            RV64GCInstruction::Lb(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm,
                width: MemoryWidth::Byte,
                signed: true,
            }),
            RV64GCInstruction::Lbu(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm,
                width: MemoryWidth::Byte,
                signed: false,
            }),
            RV64GCInstruction::Ld(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm,
                width: MemoryWidth::Double,
                signed: false,
            }),
            RV64GCInstruction::Lh(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm,
                width: MemoryWidth::Half,
                signed: true,
            }),
            RV64GCInstruction::Lhu(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm,
                width: MemoryWidth::Half,
                signed: false,
            }),
            RV64GCInstruction::Lui(rd, imm) => Some(Self::Lui {
                rd,
                value: imm as u64,
            }),
            RV64GCInstruction::Lw(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm,
                width: MemoryWidth::Word,
                signed: true,
            }),
            RV64GCInstruction::Lwu(rd, rs1, imm) => Some(Self::Load {
                rd,
                rs1,
                imm: sign_extend12(imm),
                width: MemoryWidth::Word,
                signed: false,
            }),
            RV64GCInstruction::Mul(rd, rs1, rs2) => Some(Self::Mul { rd, rs1, rs2 }),
            RV64GCInstruction::Mulh(rd, rs1, rs2) => Some(Self::Mulh { rd, rs1, rs2 }),
            RV64GCInstruction::Mulhsu(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Mulhsu,
            }),
            RV64GCInstruction::Mulhu(rd, rs1, rs2) => Some(Self::Mulhu { rd, rs1, rs2 }),
            RV64GCInstruction::Mulw(rd, rs1, rs2) => Some(Self::Mulw { rd, rs1, rs2 }),
            RV64GCInstruction::Or(rd, rs1, rs2) => Some(Self::Or { rd, rs1, rs2 }),
            RV64GCInstruction::Ori(rd, rs1, imm) => Some(Self::Ori { rd, rs1, imm }),
            RV64GCInstruction::Rem(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Rem,
            }),
            RV64GCInstruction::Remu(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Remu,
            }),
            RV64GCInstruction::Remuw(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Remuw,
            }),
            RV64GCInstruction::Remw(rd, rs1, rs2) => Some(Self::RuntimeBinary {
                rd,
                rs1,
                rs2,
                op: RuntimeBinaryOp::Remw,
            }),
            RV64GCInstruction::Sb(rs1, rs2, imm) => Some(Self::Store {
                rs1,
                rs2,
                imm,
                width: MemoryWidth::Byte,
            }),
            RV64GCInstruction::Sd(rs1, rs2, imm) => Some(Self::Store {
                rs1,
                rs2,
                imm,
                width: MemoryWidth::Double,
            }),
            RV64GCInstruction::Sh(rs1, rs2, imm) => Some(Self::Store {
                rs1,
                rs2,
                imm,
                width: MemoryWidth::Half,
            }),
            RV64GCInstruction::Sll(rd, rs1, rs2) => Some(Self::Sll { rd, rs1, rs2 }),
            RV64GCInstruction::Slli(rd, rs1, shamt) => Some(Self::Slli { rd, rs1, shamt }),
            RV64GCInstruction::Slliw(rd, rs1, shamt) => Some(Self::Slliw { rd, rs1, shamt }),
            RV64GCInstruction::Sllw(rd, rs1, rs2) => Some(Self::Sllw { rd, rs1, rs2 }),
            RV64GCInstruction::Slt(rd, rs1, rs2) => Some(Self::Slt { rd, rs1, rs2 }),
            RV64GCInstruction::Slti(rd, rs1, imm) => Some(Self::Slti { rd, rs1, imm }),
            RV64GCInstruction::Sltiu(rd, rs1, imm) => Some(Self::Sltiu {
                rd,
                rs1,
                imm: sign_extend12(imm),
            }),
            RV64GCInstruction::Sltu(rd, rs1, rs2) => Some(Self::Sltu { rd, rs1, rs2 }),
            RV64GCInstruction::Sra(rd, rs1, rs2) => Some(Self::Sra { rd, rs1, rs2 }),
            RV64GCInstruction::Srai(rd, rs1, shamt) => Some(Self::Srai { rd, rs1, shamt }),
            RV64GCInstruction::Sraiw(rd, rs1, shamt) => Some(Self::Sraiw { rd, rs1, shamt }),
            RV64GCInstruction::Sraw(rd, rs1, rs2) => Some(Self::Sraw { rd, rs1, rs2 }),
            RV64GCInstruction::Srl(rd, rs1, rs2) => Some(Self::Srl { rd, rs1, rs2 }),
            RV64GCInstruction::Srli(rd, rs1, shamt) => Some(Self::Srli { rd, rs1, shamt }),
            RV64GCInstruction::Srliw(rd, rs1, shamt) => Some(Self::Srliw { rd, rs1, shamt }),
            RV64GCInstruction::Srlw(rd, rs1, rs2) => Some(Self::Srlw { rd, rs1, rs2 }),
            RV64GCInstruction::Sub(rd, rs1, rs2) => Some(Self::Sub { rd, rs1, rs2 }),
            RV64GCInstruction::Subw(rd, rs1, rs2) => Some(Self::Subw { rd, rs1, rs2 }),
            RV64GCInstruction::Sw(rs1, rs2, imm) => Some(Self::Store {
                rs1,
                rs2,
                imm,
                width: MemoryWidth::Word,
            }),
            RV64GCInstruction::Xor(rd, rs1, rs2) => Some(Self::Xor { rd, rs1, rs2 }),
            RV64GCInstruction::Xori(rd, rs1, imm) => Some(Self::Xori { rd, rs1, imm }),
            RV64GCInstruction::Amoaddd(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoaddd,
            }),
            RV64GCInstruction::Amoaddw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoaddw,
            }),
            RV64GCInstruction::Amoandd(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoandd,
            }),
            RV64GCInstruction::Amoandw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoandw,
            }),
            RV64GCInstruction::Amomaxd(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amomaxd,
            }),
            RV64GCInstruction::Amomaxud(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amomaxud,
            }),
            RV64GCInstruction::Amomaxuw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amomaxuw,
            }),
            RV64GCInstruction::Amomaxw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amomaxw,
            }),
            RV64GCInstruction::Amomind(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amomind,
            }),
            RV64GCInstruction::Amominud(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amominud,
            }),
            RV64GCInstruction::Amominuw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amominuw,
            }),
            RV64GCInstruction::Amominw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amominw,
            }),
            RV64GCInstruction::Amoord(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoord,
            }),
            RV64GCInstruction::Amoorw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoorw,
            }),
            RV64GCInstruction::Amoswapd(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoswapd,
            }),
            RV64GCInstruction::Amoswapw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoswapw,
            }),
            RV64GCInstruction::Amoxord(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoxord,
            }),
            RV64GCInstruction::Amoxorw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Amoxorw,
            }),
            RV64GCInstruction::Lrd(rd, rs1) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2: Zero as u8,
                op: RuntimeAtomicOp::Lrd,
            }),
            RV64GCInstruction::Lrw(rd, rs1) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2: Zero as u8,
                op: RuntimeAtomicOp::Lrw,
            }),
            RV64GCInstruction::Scd(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Scd,
            }),
            RV64GCInstruction::Scw(rd, rs1, rs2) => Some(Self::RuntimeAtomic {
                rd,
                rs1,
                rs2,
                op: RuntimeAtomicOp::Scw,
            }),
            RV64GCInstruction::Ebreak => Some(Self::RuntimeTrap {
                pc,
                opcode: 0x0010_0073,
                op: RuntimeTrapOp::Ebreak,
            }),
            RV64GCInstruction::Cebreak => Some(Self::RuntimeTrap {
                pc,
                opcode: 0x0000_9002,
                op: RuntimeTrapOp::Cebreak,
            }),
            RV64GCInstruction::Uret => Some(Self::RuntimeTrap {
                pc,
                opcode: 0x0020_0073,
                op: RuntimeTrapOp::Uret,
            }),
            RV64GCInstruction::Sret => Some(Self::RuntimeTrap {
                pc,
                opcode: 0x1020_0073,
                op: RuntimeTrapOp::Sret,
            }),
            RV64GCInstruction::Mret => Some(Self::RuntimeTrap {
                pc,
                opcode: 0x3020_0073,
                op: RuntimeTrapOp::Mret,
            }),
            RV64GCInstruction::Wfi => Some(Self::RuntimeTrap {
                pc,
                opcode: 0x1050_0073,
                op: RuntimeTrapOp::Wfi,
            }),
            RV64GCInstruction::SfenceVma(_, _, _) => Some(Self::RuntimeTrap {
                pc,
                opcode: 0,
                op: RuntimeTrapOp::SfenceVma,
            }),
            RV64GCInstruction::IllegalInstruction(opcode) => Some(Self::RuntimeTrap {
                pc,
                opcode,
                op: RuntimeTrapOp::IllegalInstruction,
            }),
        }
    }

    pub(crate) fn terminates_block(&self) -> bool {
        matches!(
            self,
            Self::Beq { .. }
                | Self::AndBranch { .. }
                | Self::Bge { .. }
                | Self::Bgeu { .. }
                | Self::Blt { .. }
                | Self::Bltu { .. }
                | Self::Bne { .. }
                | Self::ArithmeticXorToggleLoop(_)
                | Self::CountedDiamondLoop(_)
                | Self::DivisionRecurrenceLoop(_)
                | Self::FibonacciRecurrenceLoop(_)
                | Self::StoreLoadForwardLoop(_)
                | Self::Ecall { .. }
                | Self::Jal { .. }
                | Self::Jalr { .. }
                | Self::Jump { .. }
                | Self::JumpReg { .. }
                | Self::JumpRegLink { .. }
                | Self::RuntimeTrap { .. }
                | Self::TraceLoopGuard { .. }
        )
    }
}

fn branch_target(pc: u64, imm: u32) -> u64 {
    pc.wrapping_add_signed(sign_extend(u64::from(imm), 13))
}

fn compressed_branch_target(pc: u64, imm: u32) -> u64 {
    pc.wrapping_add_signed(sign_extend(u64::from(imm), 9))
}

impl fmt::Display for NativeInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add { rd, rs1, rs2 } => write!(f, "add x{rd}, x{rs1}, x{rs2}"),
            Self::Addi { rd, rs1, imm } => write!(f, "addi x{rd}, x{rs1}, {imm}"),
            Self::Addiw { rd, rs1, imm } => write!(f, "addiw x{rd}, x{rs1}, {imm}"),
            Self::Addw { rd, rs1, rs2 } => write!(f, "addw x{rd}, x{rs1}, x{rs2}"),
            Self::And { rd, rs1, rs2 } => write!(f, "and x{rd}, x{rs1}, x{rs2}"),
            Self::AndBranch {
                rd,
                rs1,
                imm,
                target,
                fallthrough,
                branch_if_zero,
            } => write!(
                f,
                "andi+{} x{rd}, x{rs1}, {imm}, target=0x{target:016x}, fallthrough=0x{fallthrough:016x}",
                if *branch_if_zero { "beqz" } else { "bnez" }
            ),
            Self::Andi { rd, rs1, imm } => write!(f, "andi x{rd}, x{rs1}, {imm}"),
            Self::Auipc { rd, value } => write!(f, "auipc x{rd}, resolved=0x{value:016x}"),
            Self::Beq {
                rs1, rs2, target, ..
            } => {
                write!(f, "beq x{rs1}, x{rs2}, target=0x{target:016x}")
            }
            Self::Bge {
                rs1, rs2, target, ..
            } => {
                write!(f, "bge x{rs1}, x{rs2}, target=0x{target:016x}")
            }
            Self::Bgeu {
                rs1, rs2, target, ..
            } => {
                write!(f, "bgeu x{rs1}, x{rs2}, target=0x{target:016x}")
            }
            Self::Blt {
                rs1, rs2, target, ..
            } => {
                write!(f, "blt x{rs1}, x{rs2}, target=0x{target:016x}")
            }
            Self::Bltu {
                rs1, rs2, target, ..
            } => {
                write!(f, "bltu x{rs1}, x{rs2}, target=0x{target:016x}")
            }
            Self::Bne {
                rs1, rs2, target, ..
            } => {
                write!(f, "bne x{rs1}, x{rs2}, target=0x{target:016x}")
            }
            Self::ByteCopy8 {
                registers,
                load_base,
                load_imm,
                store_base,
                store_imm,
            } => write!(
                f,
                "byte-copy8 regs={registers:?}, {load_imm}(x{load_base}) -> {store_imm}(x{store_base})"
            ),
            Self::ArithmeticXorToggleLoop(region) => write!(
                f,
                "arithmetic-xor-toggle-loop counter=x{} acc=x{} value=x{} xor={} exit=0x{:016x}",
                region.counter,
                region.accumulator,
                region.value_register,
                region.xor_imm,
                region.exit_pc
            ),
            Self::CountedDiamondLoop(region) => write!(
                f,
                "counted-diamond-loop counter=x{} acc=x{} parity=x{} iter=x{} mask={} zero_delta={} nonzero_delta={} exit=0x{:016x}",
                region.counter,
                region.accumulator,
                region.parity_register,
                region.iteration_register,
                region.mask,
                region.zero_accumulator_delta,
                region.nonzero_accumulator_delta,
                region.exit_pc
            ),
            Self::DivisionRecurrenceLoop(region) => write!(
                f,
                "division-recurrence-loop counter=x{} temp=x{} signed=x{}/x{}=0x{:016x} unsigned=x{}/x{}=0x{:016x} checksum=x{} exit=0x{:016x}",
                region.counter,
                region.temp_register,
                region.signed_value,
                region.signed_divisor,
                region.signed_divisor_value,
                region.unsigned_value,
                region.unsigned_divisor,
                region.unsigned_divisor_value,
                region.checksum,
                region.exit_pc
            ),
            Self::Ecall { .. } => write!(f, "ecall"),
            Self::FibonacciRecurrenceLoop(region) => write!(
                f,
                "fibonacci-recurrence-loop counter=x{} current=x{} previous=x{} checksum=x{} saved=x{} exit=0x{:016x}",
                region.counter,
                region.current_register,
                region.previous_register,
                region.checksum_register,
                region.saved_register,
                region.exit_pc
            ),
            Self::FloatLoad {
                rd,
                rs1,
                imm,
                width,
            } => write!(f, "fload{} f{rd}, {imm}(x{rs1})", width.bytes()),
            Self::FloatStore {
                rs1,
                rs2,
                imm,
                width,
            } => write!(f, "fstore{} f{rs2}, {imm}(x{rs1})", width.bytes()),
            Self::InlinedJump { target } => write!(f, "j target=0x{target:016x} ; inlined"),
            Self::Jal { rd, target, .. } => write!(f, "jal x{rd}, target=0x{target:016x}"),
            Self::Jalr { rd, rs1, imm, .. } => write!(f, "jalr x{rd}, x{rs1}, {imm}"),
            Self::Jump { target } => write!(f, "j target=0x{target:016x}"),
            Self::JumpReg { rs1 } => write!(f, "jr x{rs1}"),
            Self::JumpRegLink { rs1, .. } => write!(f, "jalr x1, x{rs1}, 0"),
            Self::Load {
                rd,
                rs1,
                imm,
                width,
                signed,
            } => write!(
                f,
                "load{}{} x{rd}, {imm}(x{rs1})",
                width.bytes(),
                if *signed { "s" } else { "u" }
            ),
            Self::Lui { rd, value } => write!(f, "lui x{rd}, value=0x{value:016x}"),
            Self::LoadImmediate { rd, value } => write!(f, "li x{rd}, 0x{value:016x}"),
            Self::Move { rd, rs } => write!(f, "mv x{rd}, x{rs}"),
            Self::Mul { rd, rs1, rs2 } => write!(f, "mul x{rd}, x{rs1}, x{rs2}"),
            Self::Mulh { rd, rs1, rs2 } => write!(f, "mulh x{rd}, x{rs1}, x{rs2}"),
            Self::Mulhu { rd, rs1, rs2 } => write!(f, "mulhu x{rd}, x{rs1}, x{rs2}"),
            Self::Mulw { rd, rs1, rs2 } => write!(f, "mulw x{rd}, x{rs1}, x{rs2}"),
            Self::Nop => write!(f, "nop"),
            Self::Or { rd, rs1, rs2 } => write!(f, "or x{rd}, x{rs1}, x{rs2}"),
            Self::Ori { rd, rs1, imm } => write!(f, "ori x{rd}, x{rs1}, {imm}"),
            Self::Sll { rd, rs1, rs2 } => write!(f, "sll x{rd}, x{rs1}, x{rs2}"),
            Self::Slli { rd, rs1, shamt } => write!(f, "slli x{rd}, x{rs1}, {shamt}"),
            Self::ShiftedWordOr {
                rd,
                rs1,
                left_shamt,
                right_shamt,
            } => write!(
                f,
                "shifted-word-or x{rd}, x{rs1}, left={left_shamt}, rightw={right_shamt}"
            ),
            Self::Slliw { rd, rs1, shamt } => write!(f, "slliw x{rd}, x{rs1}, {shamt}"),
            Self::Sllw { rd, rs1, rs2 } => write!(f, "sllw x{rd}, x{rs1}, x{rs2}"),
            Self::Slt { rd, rs1, rs2 } => write!(f, "slt x{rd}, x{rs1}, x{rs2}"),
            Self::Slti { rd, rs1, imm } => write!(f, "slti x{rd}, x{rs1}, {imm}"),
            Self::Sltiu { rd, rs1, imm } => write!(f, "sltiu x{rd}, x{rs1}, {imm}"),
            Self::Sltu { rd, rs1, rs2 } => write!(f, "sltu x{rd}, x{rs1}, x{rs2}"),
            Self::Sra { rd, rs1, rs2 } => write!(f, "sra x{rd}, x{rs1}, x{rs2}"),
            Self::Srai { rd, rs1, shamt } => write!(f, "srai x{rd}, x{rs1}, {shamt}"),
            Self::Sraiw { rd, rs1, shamt } => write!(f, "sraiw x{rd}, x{rs1}, {shamt}"),
            Self::Sraw { rd, rs1, rs2 } => write!(f, "sraw x{rd}, x{rs1}, x{rs2}"),
            Self::Srl { rd, rs1, rs2 } => write!(f, "srl x{rd}, x{rs1}, x{rs2}"),
            Self::Srli { rd, rs1, shamt } => write!(f, "srli x{rd}, x{rs1}, {shamt}"),
            Self::Srliw { rd, rs1, shamt } => write!(f, "srliw x{rd}, x{rs1}, {shamt}"),
            Self::Srlw { rd, rs1, rs2 } => write!(f, "srlw x{rd}, x{rs1}, x{rs2}"),
            Self::Sub { rd, rs1, rs2 } => write!(f, "sub x{rd}, x{rs1}, x{rs2}"),
            Self::Subw { rd, rs1, rs2 } => write!(f, "subw x{rd}, x{rs1}, x{rs2}"),
            Self::RuntimeBinary { rd, rs1, rs2, op } => {
                write!(f, "{op:?} x{rd}, x{rs1}, x{rs2}")
            }
            Self::RuntimeCsr {
                rd,
                rs1_or_uimm,
                csr,
                op,
            } => write!(f, "{op:?} x{rd}, {rs1_or_uimm}, 0x{csr:03x}"),
            Self::RuntimeAtomic { rd, rs1, rs2, op } => {
                write!(f, "{op:?} x{rd}, (x{rs1}), x{rs2}")
            }
            Self::RuntimeTrap { pc, opcode, op } => {
                write!(f, "trap {op:?} pc=0x{pc:016x} opcode=0x{opcode:08x}")
            }
            Self::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op,
            } => write!(
                f,
                "{op:?} rd={rd}, rs1={rs1}, rs2={rs2}, rs3={rs3}, rm={rm}"
            ),
            Self::Store {
                rs1,
                rs2,
                imm,
                width,
            } => write!(f, "store{} x{rs2}, {imm}(x{rs1})", width.bytes()),
            Self::StoreLoadForwardLoop(region) => write!(
                f,
                "store-load-forward-loop counter=x{} value=x{} offset=x{} base=x{} index=x{} addr=x{} loaded=x{} mask={} width={} exit=0x{:016x}",
                region.counter,
                region.value_register,
                region.offset_register,
                region.base_register,
                region.index_register,
                region.address_register,
                region.loaded_register,
                region.mask,
                region.width.bytes(),
                region.exit_pc
            ),
            Self::TraceGuard {
                rs1,
                rs2,
                condition,
                continue_on_taken,
                continue_pc,
                side_exit_pc,
                executed_instructions,
            } => write!(
                f,
                "trace-guard {} x{rs1}, x{rs2}, continue_{}=0x{continue_pc:016x}, side_exit=0x{side_exit_pc:016x}, executed={executed_instructions}",
                condition.mnemonic(),
                if *continue_on_taken { "taken" } else { "not_taken" }
            ),
            Self::TraceLoopGuard {
                rs1,
                rs2,
                condition,
                continue_on_taken,
                loop_pc,
                side_exit_pc,
                guest_instruction_count,
            } => write!(
                f,
                "trace-loop-guard {} x{rs1}, x{rs2}, continue_{}=0x{loop_pc:016x}, side_exit=0x{side_exit_pc:016x}, guest_instructions={guest_instruction_count}",
                condition.mnemonic(),
                if *continue_on_taken { "taken" } else { "not_taken" }
            ),
            Self::Xor { rd, rs1, rs2 } => write!(f, "xor x{rd}, x{rs1}, x{rs2}"),
            Self::Xori { rd, rs1, imm } => write!(f, "xori x{rd}, x{rs1}, {imm}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ram::MemoryRegion;

    fn rv64_word(value: u32) -> [u8; 4] {
        value.to_le_bytes()
    }

    fn rv64_halfword(value: u16) -> [u8; 2] {
        value.to_le_bytes()
    }

    fn f32_box(value: f32) -> u64 {
        0xffff_ffff_0000_0000 | u64::from(value.to_bits())
    }

    fn rv64_beq(rs1: u8, rs2: u8, offset: u32) -> u32 {
        ((offset >> 12) & 0x1) << 31
            | ((offset >> 5) & 0x3f) << 25
            | (u32::from(rs2) << 20)
            | (u32::from(rs1) << 15)
            | ((offset >> 1) & 0xf) << 8
            | ((offset >> 11) & 0x1) << 7
            | 0x63
    }

    fn rv64_bne(rs1: u8, rs2: u8, offset: u32) -> u32 {
        rv64_beq(rs1, rs2, offset) | 0x1000
    }

    fn rv64_jal(rd: u8, offset: u32) -> u32 {
        ((offset >> 20) & 0x1) << 31
            | ((offset >> 1) & 0x3ff) << 21
            | ((offset >> 11) & 0x1) << 20
            | ((offset >> 12) & 0xff) << 12
            | (u32::from(rd) << 7)
            | 0x6f
    }

    fn rv64_s(funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
        let imm = imm as u32 & 0xfff;
        ((imm >> 5) << 25)
            | (u32::from(rs2) << 20)
            | (u32::from(rs1) << 15)
            | (funct3 << 12)
            | ((imm & 0x1f) << 7)
            | 0x23
    }

    fn rv64_i(opcode: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
        ((imm as u32 & 0xfff) << 20)
            | (u32::from(rs1) << 15)
            | (funct3 << 12)
            | (u32::from(rd) << 7)
            | opcode
    }

    fn rv64_r(opcode: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
        (funct7 << 25)
            | (u32::from(rs2) << 20)
            | (u32::from(rs1) << 15)
            | (funct3 << 12)
            | (u32::from(rd) << 7)
            | opcode
    }

    fn division_recurrence_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 4, 1, 7, 10, 11))); // div x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x33, 6, 1, 7, 10, 11))); // rem x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x33, 5, 1, 7, 12, 13))); // divu x7, x12, x13
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x33, 7, 1, 7, 12, 13))); // remu x7, x12, x13
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x3b, 4, 1, 7, 10, 11))); // divw x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x3b, 6, 1, 7, 10, 11))); // remw x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x3b, 5, 1, 7, 12, 13))); // divuw x7, x12, x13
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x3b, 7, 1, 7, 12, 13))); // remuw x7, x12, x13
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x33, 2, 1, 7, 10, 13))); // mulhsu x7, x10, x13
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_i(0x13, 0, 10, 10, 3))); // addi x10, x10, 3
        bin.extend(rv64_word(rv64_i(0x13, 0, 12, 12, 5))); // addi x12, x12, 5
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fac))); // bne x5, x0, loop
        bin
    }

    fn division_recurrence_cpu(
        bin: &[u8],
        count: u64,
        signed: u64,
        signed_divisor: u64,
        unsigned_divisor: u64,
    ) -> RV64GC {
        let mut cpu = RV64GC::new();
        cpu.load_bin(bin.to_vec());
        cpu.registers[5usize] = count;
        cpu.registers[10usize] = signed;
        cpu.registers[11usize] = signed_divisor;
        cpu.registers[12usize] = 0xabcd_ef01_2345_6789;
        cpu.registers[13usize] = unsigned_divisor;
        cpu.registers[20usize] = 0;
        cpu
    }

    fn assert_division_recurrence_registers(
        cpu: &RV64GC,
        count: u64,
        signed: u64,
        signed_divisor: u64,
        unsigned_divisor: u64,
    ) {
        let (counter, temp, signed_value, unsigned_value, checksum) = division_recurrence_reference(
            count,
            signed,
            signed_divisor,
            0xabcd_ef01_2345_6789,
            unsigned_divisor,
        );
        assert_eq!(cpu.registers[5usize], counter);
        assert_eq!(cpu.registers[7usize], temp);
        assert_eq!(cpu.registers[10usize], signed_value);
        assert_eq!(cpu.registers[12usize], unsigned_value);
        assert_eq!(cpu.registers[20usize], checksum);
        assert_eq!(cpu.registers[Pc], 88);
    }

    fn division_recurrence_reference(
        mut count: u64,
        mut signed: u64,
        signed_divisor: u64,
        mut unsigned: u64,
        unsigned_divisor: u64,
    ) -> (u64, u64, u64, u64, u64) {
        let mut checksum = 0u64;
        let mut temp = 0u64;
        while count != 0 {
            let div = riscv_div(signed, signed_divisor);
            let rem = riscv_rem(signed, signed_divisor);
            let divu = riscv_divu(unsigned, unsigned_divisor);
            let remu = riscv_remu(unsigned, unsigned_divisor);
            let divw = riscv_divw(signed, signed_divisor);
            let remw = riscv_remw(signed, signed_divisor);
            let divuw = riscv_divuw(unsigned, unsigned_divisor);
            let remuw = riscv_remuw(unsigned, unsigned_divisor);
            temp = riscv_mulhsu(signed, unsigned_divisor);
            checksum ^= div ^ rem ^ divu ^ remu ^ divw ^ remw ^ divuw ^ remuw ^ temp;
            signed = signed.wrapping_add(3);
            unsigned = unsigned.wrapping_add(5);
            count = count.wrapping_sub(1);
        }
        (count, temp, signed, unsigned, checksum)
    }

    fn riscv_div(lhs: u64, rhs: u64) -> u64 {
        if rhs == 0 {
            u64::MAX
        } else {
            (lhs as i64).wrapping_div(rhs as i64) as u64
        }
    }

    fn riscv_rem(lhs: u64, rhs: u64) -> u64 {
        if rhs == 0 {
            lhs
        } else {
            (lhs as i64).wrapping_rem(rhs as i64) as u64
        }
    }

    fn riscv_divu(lhs: u64, rhs: u64) -> u64 {
        if rhs == 0 {
            u64::MAX
        } else {
            lhs / rhs
        }
    }

    fn riscv_remu(lhs: u64, rhs: u64) -> u64 {
        if rhs == 0 {
            lhs
        } else {
            lhs % rhs
        }
    }

    fn riscv_divw(lhs: u64, rhs: u64) -> u64 {
        let lhs = lhs as u32 as i32;
        let rhs = rhs as u32 as i32;
        let quotient = if rhs == 0 { -1 } else { lhs.wrapping_div(rhs) };
        sign_extend(u64::from(quotient as u32), 32) as u64
    }

    fn riscv_remw(lhs: u64, rhs: u64) -> u64 {
        let lhs = lhs as u32 as i32;
        let rhs = rhs as u32 as i32;
        let remainder = if rhs == 0 { lhs } else { lhs.wrapping_rem(rhs) };
        sign_extend(u64::from(remainder as u32), 32) as u64
    }

    fn riscv_divuw(lhs: u64, rhs: u64) -> u64 {
        let lhs = lhs as u32;
        let rhs = rhs as u32;
        let quotient = if rhs == 0 { u32::MAX } else { lhs / rhs };
        sign_extend(u64::from(quotient), 32) as u64
    }

    fn riscv_remuw(lhs: u64, rhs: u64) -> u64 {
        let lhs = lhs as u32;
        let rhs = rhs as u32;
        let remainder = if rhs == 0 { lhs } else { lhs % rhs };
        sign_extend(u64::from(remainder), 32) as u64
    }

    fn riscv_mulhsu(lhs: u64, rhs: u64) -> u64 {
        (((lhs as i64 as i128) * (rhs as u128 as i128)) >> 64) as u64
    }

    fn rv64_r4(opcode: u32, rm: u32, rd: u8, rs1: u8, rs2: u8, rs3: u8) -> u32 {
        rv64_r4_fmt(opcode, 0, rm, rd, rs1, rs2, rs3)
    }

    fn rv64_r4_fmt(opcode: u32, fmt: u32, rm: u32, rd: u8, rs1: u8, rs2: u8, rs3: u8) -> u32 {
        (u32::from(rs3) << 27)
            | (fmt << 25)
            | (u32::from(rs2) << 20)
            | (u32::from(rs1) << 15)
            | (rm << 12)
            | (u32::from(rd) << 7)
            | opcode
    }

    fn write_guest_c_string(cpu: &mut RV64GC, address: u64, value: &str) {
        for (offset, byte) in value.bytes().chain([0]).enumerate() {
            cpu.ram.write_byte(address + offset as u64, byte).unwrap();
        }
    }

    fn trace_diamond_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 7, 7, 5, 1))); // andi x7, x5, 1
        bin.extend(rv64_word(rv64_beq(7, 0, 12))); // beq x7, x0, +12
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 3))); // addi x6, x6, 3
        bin.extend(rv64_word(rv64_jal(0, 8))); // jal x0, +8
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 7))); // addi x6, x6, 7
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fe8))); // bne x5, x0, -24
        bin.extend(rv64_word(rv64_i(0x13, 0, 8, 0, 1))); // addi x8, x0, 1
        bin
    }

    fn trace_memory_branch_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x03, 3, 7, 10, 0))); // ld x7, 0(x10)
        bin.extend(rv64_word(rv64_beq(7, 0, 16))); // beq x7, x0, +16
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 1))); // addi x6, x6, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 10, 10, 8))); // addi x10, x10, 8
        bin.extend(rv64_word(rv64_bne(6, 5, 0x1ff0))); // bne x6, x5, -16
        bin.extend(rv64_word(rv64_i(0x13, 0, 8, 0, 1))); // addi x8, x0, 1
        bin
    }

    fn trace_byte_memory_branch_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x03, 4, 7, 10, 0))); // lbu x7, 0(x10)
        bin.extend(rv64_word(rv64_beq(7, 0, 16))); // beq x7, x0, +16
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 1))); // addi x6, x6, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 10, 10, 1))); // addi x10, x10, 1
        bin.extend(rv64_word(rv64_bne(6, 5, 0x1ff0))); // bne x6, x5, -16
        bin.extend(rv64_word(rv64_i(0x13, 0, 8, 0, 1))); // addi x8, x0, 1
        bin
    }

    fn trace_store_load_branch_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_s(3, 10, 11, 0))); // sd x11, 0(x10)
        bin.extend(rv64_word(rv64_i(0x03, 3, 7, 10, 0))); // ld x7, 0(x10)
        bin.extend(rv64_word(rv64_beq(7, 0, 16))); // beq x7, x0, +16
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 1))); // addi x6, x6, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 10, 10, 8))); // addi x10, x10, 8
        bin.extend(rv64_word(rv64_bne(6, 5, 0x1fec))); // bne x6, x5, -20
        bin.extend(rv64_word(rv64_i(0x13, 0, 8, 0, 1))); // addi x8, x0, 1
        bin
    }

    #[test]
    fn host_printf_formats_repeated_signed_ints() {
        let cpu = RV64GC::new();

        let output = format_host_printf(&cpu, "fib(%i) = %i\n", &[10, 55]).unwrap();

        assert_eq!(output, b"fib(10) = 55\n");
    }

    #[test]
    fn host_printf_keeps_unsigned_long_long_support() {
        let cpu = RV64GC::new();

        let output =
            format_host_printf(&cpu, "fibonacci checksum: %llu\n", &[12_345_678_901]).unwrap();

        assert_eq!(output, b"fibonacci checksum: 12345678901\n");
    }

    #[test]
    fn host_printf_formats_guest_strings_hex_and_percent_literals() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(vec![0; 0x2000]);
        write_guest_c_string(&mut cpu, 0x1000, "riscvm");

        let output = format_host_printf(&cpu, "%s: 0x%x %%\n", &[0x1000, 0x2a]).unwrap();

        assert_eq!(output, b"riscvm: 0x2a %\n");
    }

    #[test]
    fn host_printf_rejects_unsupported_format_flags() {
        let cpu = RV64GC::new();

        assert!(format_host_printf(&cpu, "%02i\n", &[7]).is_none());
    }

    #[test]
    fn host_libc_resolves_dynamic_runtime_symbols() {
        assert_eq!(host_libc_function_for_name("__libc_start_main"), None);
        assert_eq!(
            host_libc_function_for_name("printf"),
            Some(HostLibcFunction::Printf)
        );
        assert_eq!(
            host_libc_function_for_name("puts"),
            Some(HostLibcFunction::Puts)
        );
        assert_eq!(host_libc_function_for_name("malloc"), None);
    }

    #[test]
    fn host_libc_can_disable_plt_stdio_shortcuts() {
        assert_eq!(
            host_libc_function_for_plt_name("memcpy", false),
            Some(HostLibcFunction::Memcpy)
        );
        assert_eq!(host_libc_function_for_plt_name("printf", false), None);
        assert_eq!(host_libc_function_for_plt_name("puts", false), None);
        assert_eq!(host_libc_function_for_plt_name("exit", false), None);
        assert_eq!(
            host_libc_function_for_plt_name("puts", true),
            Some(HostLibcFunction::Puts)
        );
    }

    #[test]
    fn host_libc_does_not_replace_static_runtime_entry_points_by_default() {
        assert_eq!(
            host_libc_function_for_defined_symbol("__libc_start_main"),
            None
        );
        assert_eq!(host_libc_function_for_defined_symbol("exit"), None);
        assert_eq!(host_libc_function_for_defined_symbol("puts"), None);
        assert_eq!(host_libc_function_for_defined_symbol("printf"), None);
        assert_eq!(
            host_libc_function_for_defined_symbol("memcpy"),
            Some(HostLibcFunction::Memcpy)
        );
    }

    #[test]
    fn host_libc_has_explicit_static_start_main_shortcut() {
        assert_eq!(
            host_libc_start_main_shortcut_for_defined_symbol("__libc_start_main"),
            Some(HostLibcFunction::LibcStartMain)
        );
        assert_eq!(
            host_libc_start_main_shortcut_for_defined_symbol("printf"),
            Some(HostLibcFunction::Printf)
        );
        assert_eq!(
            host_libc_start_main_shortcut_for_defined_symbol("puts"),
            Some(HostLibcFunction::Puts)
        );
        assert_eq!(
            host_libc_start_main_shortcut_for_defined_symbol("exit"),
            Some(HostLibcFunction::Exit)
        );
    }

    #[test]
    fn host_libc_start_main_enters_guest_main_with_standard_arguments() {
        let mut cpu = RV64GC::new();
        cpu.registers[A0] = 0x1234;
        cpu.registers[A1] = 2;
        cpu.registers[A2] = 0x8000;

        assert!(execute_host_libc_start_main(&mut cpu).unwrap());

        assert_eq!(cpu.registers[Pc], 0x1234);
        assert_eq!(cpu.registers[Ra], HOST_LIBC_EXIT_TRAMPOLINE);
        assert_eq!(cpu.registers[A0], 2);
        assert_eq!(cpu.registers[A1], 0x8000);
        assert_eq!(cpu.registers[A2], 0x8018);
    }

    #[test]
    fn host_puts_writes_guest_string_with_newline() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(vec![0; 0x2000]);
        cpu.filesystem.set_output_mirroring(false);
        write_guest_c_string(&mut cpu, 0x1000, "dynamic");
        cpu.registers[A0] = 0x1000;
        cpu.registers[Ra] = 0x88;

        assert!(execute_host_puts(&mut cpu).unwrap());

        assert_eq!(cpu.stdout(), b"dynamic\n");
        assert_eq!(cpu.registers[A0], 8);
        assert_eq!(cpu.registers[Pc], 0x88);
    }

    #[cfg(unix)]
    #[test]
    fn host_memset_writes_contiguous_guest_memory_directly() {
        let mut cpu = RV64GC::new();
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 32, vec![0; 32]))
            .unwrap();
        cpu.registers[A0] = 0x1004;
        cpu.registers[A1] = 0xab;
        cpu.registers[A2] = 12;
        cpu.registers[Ra] = 0x44;

        assert!(execute_host_memset(&mut cpu).unwrap());

        assert_eq!(cpu.registers[A0], 0x1004);
        assert_eq!(cpu.registers[Pc], 0x44);
        assert_eq!(
            cpu.ram.read_slice_range(0x1000, 20).unwrap(),
            &[
                0, 0, 0, 0, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab,
                0, 0, 0, 0
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn host_memmove_direct_path_preserves_overlap_semantics() {
        let mut cpu = RV64GC::new();
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 16, (0u8..16).collect::<Vec<_>>()))
            .unwrap();
        cpu.registers[A0] = 0x1004;
        cpu.registers[A1] = 0x1000;
        cpu.registers[A2] = 8;
        cpu.registers[Ra] = 0x48;

        assert!(execute_host_memmove(&mut cpu).unwrap());

        assert_eq!(cpu.registers[A0], 0x1004);
        assert_eq!(cpu.registers[Pc], 0x48);
        assert_eq!(
            cpu.ram.read_slice_range(0x1000, 16).unwrap(),
            &[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
        );
    }

    #[test]
    fn host_exit_stops_cpu_and_preserves_status() {
        let mut cpu = RV64GC::new();
        cpu.registers[A0] = 42;
        cpu.registers[Ra] = 0x90;

        assert!(execute_host_exit(&mut cpu).unwrap());

        assert!(cpu.should_quit);
        assert_eq!(cpu.registers[A0], 42);
        assert_eq!(cpu.registers[Pc], 0x90);
    }

    #[test]
    fn jit_executes_a_straight_line_integer_block() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0050_0093)); // addi x1, x0, 5
        bin.extend(rv64_word(0x0070_8113)); // addi x2, x1, 7
        bin.extend(rv64_word(0x0020_81b3)); // add x3, x1, x2

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let Ok(mut engine) = JitEngine::new() else {
            return;
        };

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 3
            }
        );
        assert_eq!(cpu.registers[1usize], 5);
        assert_eq!(cpu.registers[2usize], 12);
        assert_eq!(cpu.registers[3usize], 17);
        assert_eq!(cpu.registers[Pc], 12);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_options_enable_dynamic_recompilation_by_default() {
        assert_eq!(
            JitOptions::default().execution_mode,
            JitExecutionMode::Hybrid
        );
        assert!(!JitExecutionMode::Jit.precompiles_before_execution());
        assert!(!JitExecutionMode::Hybrid.precompiles_before_execution());
        assert!(JitExecutionMode::Aot.precompiles_before_execution());
        assert!(JitOptions::default().host_libc);
        assert!(!JitOptions::default().libc_start_main_shortcut);
        assert!(JitOptions::default().dynamic_recompilation);
        assert!(JitOptions::default().trace_compilation);
        assert_eq!(JitOptions::default().hot_threshold, 8);
        assert!(JitOptions::default().tier_budgeting);
        assert_eq!(
            JitOptions::default().non_loop_hot_threshold_multiplier,
            4096
        );
        assert_eq!(JitOptions::default().min_optimized_block_instructions, 2);
        assert!(!JitOptions::default().background_compilation);
        assert_eq!(JitOptions::default().compiler_threads, 1);
        assert_eq!(JitOptions::default().compile_queue_limit, 64);
        assert!(JitOptions::default().aot_compile_misses);
        assert!(!JitOptions::default().aot_symbol_entries);
        assert!(!JitOptions::default().aot_linear_sweep);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_caches_negative_byte_scan_candidates() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(rv64_word(0x0000_0013).to_vec());
        let mut engine = JitEngine::new().unwrap();

        assert_eq!(engine.try_candidate_byte_scan_loop(&mut cpu, 0), None);

        let entry = engine.candidate_byte_scan_cache[candidate_byte_scan_cache_slot(0)];
        assert_eq!(entry.pc, 0);
        assert_eq!(entry.code_version, cpu.ram.code_version());
        assert!(matches!(
            entry.result,
            CandidateByteScanCacheResult::NotCandidate
        ));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_startup_profile_precompiles_profiled_blocks() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_beq(0, 0, 8))); // beq x0, x0, +8
        bin.extend(rv64_word(0x0000_0013)); // addi x0, x0, 0
        bin.extend(rv64_word(0x0010_0113)); // addi x2, x0, 1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_startup_profile(
            JitOptions::default(),
            [JitStartupProfileEntry::new(8, 10)],
        )
        .unwrap();

        engine.step(&mut cpu).unwrap();

        let profiled = engine.cache.get(&8).expect("profiled block");
        assert_eq!(profiled.tier, JitTier::Baseline);
        assert_eq!(cpu.registers[Pc], 8);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_word_integer_and_compare_operations_natively() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x1b, 0, 3, 0, -1))); // addiw x3, x0, -1
        bin.extend(rv64_word(rv64_i(0x1b, 1, 4, 3, 1))); // slliw x4, x3, 1
        bin.extend(rv64_word(rv64_i(0x1b, 5, 5, 3, 1))); // srliw x5, x3, 1
        bin.extend(rv64_word(rv64_i(0x1b, 5, 6, 3, 0x401))); // sraiw x6, x3, 1
        bin.extend(rv64_word(rv64_r(0x33, 2, 0, 7, 3, 0))); // slt x7, x3, x0
        bin.extend(rv64_word(rv64_r(0x33, 3, 0, 8, 3, 0))); // sltu x8, x3, x0
        bin.extend(rv64_word(rv64_i(0x13, 2, 9, 0, 1))); // slti x9, x0, 1
        bin.extend(rv64_word(rv64_i(0x13, 3, 10, 0, -1))); // sltiu x10, x0, -1
        bin.extend(rv64_word(rv64_r(0x3b, 0, 0, 11, 3, 3))); // addw x11, x3, x3
        bin.extend(rv64_word(rv64_r(0x3b, 0, 0x20, 12, 0, 3))); // subw x12, x0, x3
        bin.extend(rv64_word(rv64_r(0x3b, 1, 0, 13, 3, 9))); // sllw x13, x3, x9
        bin.extend(rv64_word(rv64_r(0x3b, 5, 0, 14, 3, 9))); // srlw x14, x3, x9
        bin.extend(rv64_word(rv64_r(0x3b, 5, 0x20, 15, 3, 9))); // sraw x15, x3, x9
        bin.extend(rv64_word(rv64_i(0x13, 2, 16, 3, 0))); // slti x16, x3, 0
        bin.extend(rv64_word(rv64_i(0x13, 3, 17, 3, -1))); // sltiu x17, x3, -1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[3usize], u64::MAX);
        assert_eq!(cpu.registers[4usize], u64::MAX - 1);
        assert_eq!(cpu.registers[5usize], 0x7fff_ffff);
        assert_eq!(cpu.registers[6usize], u64::MAX);
        assert_eq!(cpu.registers[7usize], 1);
        assert_eq!(cpu.registers[8usize], 0);
        assert_eq!(cpu.registers[9usize], 1);
        assert_eq!(cpu.registers[10usize], 1);
        assert_eq!(cpu.registers[11usize], u64::MAX - 1);
        assert_eq!(cpu.registers[12usize], 1);
        assert_eq!(cpu.registers[13usize], u64::MAX - 1);
        assert_eq!(cpu.registers[14usize], 0x7fff_ffff);
        assert_eq!(cpu.registers[15usize], u64::MAX);
        assert_eq!(cpu.registers[16usize], 1);
        assert_eq!(cpu.registers[17usize], 0);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.matches("cset x9").count() >= 4);
        assert!(listing.contains("cmp x9, #0 ; slti"));
        assert!(listing.contains("cmn x9, #1 ; sltiu"));
        assert!(!listing.contains("csel x9"));
        assert!(!listing.contains("movz x10, #0xffff"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_shift_immediates_to_single_host_instructions() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 0, -1))); // addi x1, x0, -1
        bin.extend(rv64_word(rv64_i(0x13, 1, 2, 1, 3))); // slli x2, x1, 3
        bin.extend(rv64_word(rv64_i(0x13, 5, 3, 1, 4))); // srli x3, x1, 4
        bin.extend(rv64_word(rv64_i(0x13, 5, 4, 1, 0x404))); // srai x4, x1, 4
        bin.extend(rv64_word(rv64_i(0x1b, 0, 5, 0, -1))); // addiw x5, x0, -1
        bin.extend(rv64_word(rv64_i(0x1b, 1, 6, 5, 3))); // slliw x6, x5, 3
        bin.extend(rv64_word(rv64_i(0x1b, 5, 7, 5, 4))); // srliw x7, x5, 4
        bin.extend(rv64_word(rv64_i(0x1b, 5, 8, 5, 0x404))); // sraiw x8, x5, 4

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[2usize], u64::MAX << 3);
        assert_eq!(cpu.registers[3usize], u64::MAX >> 4);
        assert_eq!(cpu.registers[4usize], u64::MAX);
        assert_eq!(cpu.registers[6usize], sign_extend(0xffff_fff8, 32) as u64);
        assert_eq!(cpu.registers[7usize], 0x0fff_ffff);
        assert_eq!(cpu.registers[8usize], u64::MAX);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("lsl x"));
        assert!(listing.contains("lsr x"));
        assert!(listing.contains("asr x"));
        assert!(listing.contains("lsl w"));
        assert!(listing.contains("lsr w"));
        assert!(listing.contains("asr w"));
        assert!(!listing.contains("lslv"));
        assert!(!listing.contains("lsrv"));
        assert!(!listing.contains("asrv"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_logical_immediates_to_aarch64_bitmasks() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 0, -1))); // addi x1, x0, -1
        bin.extend(rv64_word(rv64_i(0x13, 7, 2, 1, 0xff))); // andi x2, x1, 0xff
        bin.extend(rv64_word(rv64_i(0x13, 6, 3, 0, 0x7f))); // ori x3, x0, 0x7f
        bin.extend(rv64_word(rv64_i(0x13, 4, 4, 0, 0x3f))); // xori x4, x0, 0x3f
        bin.extend(rv64_word(rv64_i(0x13, 7, 5, 1, -1))); // andi x5, x1, -1
        bin.extend(rv64_word(rv64_i(0x13, 4, 6, 1, -1))); // xori x6, x1, -1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[2usize], 0xff);
        assert_eq!(cpu.registers[3usize], 0x7f);
        assert_eq!(cpu.registers[4usize], 0x3f);
        assert_eq!(cpu.registers[5usize], u64::MAX);
        assert_eq!(cpu.registers[6usize], 0);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("and x9, x9, #0x00000000000000ff"));
        assert!(listing.contains("orr x9, x31, #0x000000000000007f"));
        assert!(listing.contains("eor x9, x31, #0x000000000000003f"));
        assert!(listing.contains("mvn x9, x9"));
        assert!(!listing.contains("movz x10, #0x00ff"));
        assert!(!listing.contains("movz x10, #0x007f"));
        assert!(!listing.contains("movz x10, #0x003f"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_common_multiply_operations_natively() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 0, -3))); // addi x1, x0, -3
        bin.extend(rv64_word(rv64_i(0x13, 0, 2, 0, 7))); // addi x2, x0, 7
        bin.extend(rv64_word(rv64_r(0x33, 0, 1, 3, 1, 2))); // mul x3, x1, x2
        bin.extend(rv64_word(rv64_r(0x3b, 0, 1, 4, 1, 2))); // mulw x4, x1, x2
        bin.extend(rv64_word(rv64_r(0x33, 1, 1, 5, 1, 2))); // mulh x5, x1, x2
        bin.extend(rv64_word(rv64_r(0x33, 3, 1, 6, 1, 2))); // mulhu x6, x1, x2

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::new().unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[3usize], u64::MAX - 20);
        assert_eq!(cpu.registers[4usize], u64::MAX - 20);
        assert_eq!(cpu.registers[5usize], u64::MAX);
        assert_eq!(cpu.registers[6usize], 6);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_division_remainder_and_mulhsu_natively() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 4, 1, 3, 1, 2))); // div x3, x1, x2
        bin.extend(rv64_word(rv64_r(0x33, 6, 1, 4, 1, 2))); // rem x4, x1, x2
        bin.extend(rv64_word(rv64_r(0x33, 5, 1, 5, 1, 0))); // divu x5, x1, x0
        bin.extend(rv64_word(rv64_r(0x33, 7, 1, 6, 1, 0))); // remu x6, x1, x0
        bin.extend(rv64_word(rv64_r(0x3b, 4, 1, 7, 8, 9))); // divw x7, x8, x9
        bin.extend(rv64_word(rv64_r(0x3b, 6, 1, 10, 8, 9))); // remw x10, x8, x9
        bin.extend(rv64_word(rv64_r(0x3b, 5, 1, 11, 8, 0))); // divuw x11, x8, x0
        bin.extend(rv64_word(rv64_r(0x3b, 7, 1, 12, 8, 0))); // remuw x12, x8, x0
        bin.extend(rv64_word(rv64_r(0x33, 2, 1, 13, 14, 15))); // mulhsu x13, x14, x15
        bin.extend(rv64_word(rv64_r(0x33, 4, 1, 18, 16, 17))); // div x18, x16, x17
        bin.extend(rv64_word(rv64_r(0x33, 6, 1, 19, 16, 17))); // rem x19, x16, x17
        bin.extend(rv64_word(rv64_r(0x3b, 4, 1, 20, 16, 17))); // divw x20, x16, x17
        bin.extend(rv64_word(rv64_r(0x3b, 6, 1, 21, 16, 17))); // remw x21, x16, x17
        bin.extend(rv64_word(rv64_r(0x33, 5, 1, 24, 22, 23))); // divu x24, x22, x23
        bin.extend(rv64_word(rv64_r(0x33, 7, 1, 25, 22, 23))); // remu x25, x22, x23
        bin.extend(rv64_word(rv64_r(0x3b, 5, 1, 26, 22, 23))); // divuw x26, x22, x23
        bin.extend(rv64_word(rv64_r(0x3b, 7, 1, 27, 22, 23))); // remuw x27, x22, x23

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = i64::MIN as u64;
        cpu.registers[2usize] = u64::MAX;
        cpu.registers[8usize] = 0x8000_0000;
        cpu.registers[9usize] = 0xffff_ffff;
        cpu.registers[14usize] = (-3i64) as u64;
        cpu.registers[15usize] = 7;
        cpu.registers[16usize] = (-37i64) as u64;
        cpu.registers[17usize] = 5;
        cpu.registers[22usize] = 100;
        cpu.registers[23usize] = 9;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[3usize], i64::MIN as u64);
        assert_eq!(cpu.registers[4usize], 0);
        assert_eq!(cpu.registers[5usize], u64::MAX);
        assert_eq!(cpu.registers[6usize], i64::MIN as u64);
        assert_eq!(cpu.registers[7usize], sign_extend(0x8000_0000, 32) as u64);
        assert_eq!(cpu.registers[10usize], 0);
        assert_eq!(cpu.registers[11usize], u64::MAX);
        assert_eq!(cpu.registers[12usize], sign_extend(0x8000_0000, 32) as u64);
        assert_eq!(cpu.registers[13usize], u64::MAX);
        assert_eq!(cpu.registers[18usize], (-7i64) as u64);
        assert_eq!(cpu.registers[19usize], (-2i64) as u64);
        assert_eq!(cpu.registers[20usize], (-7i64) as u64);
        assert_eq!(cpu.registers[21usize], (-2i64) as u64);
        assert_eq!(cpu.registers[24usize], 11);
        assert_eq!(cpu.registers[25usize], 1);
        assert_eq!(cpu.registers[26usize], 11);
        assert_eq!(cpu.registers[27usize], 1);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("sdiv x"));
        assert!(listing.contains("udiv x"));
        assert!(listing.contains("sdiv w"));
        assert!(listing.contains("udiv w"));
        assert!(listing.contains("msub x"));
        assert!(listing.contains("msub w"));
        assert!(listing.contains("umulh x"));
        assert!(listing.contains("mulhsu lhs sign mask"));
        assert!(!listing.contains("csel x11, x12, x11, lt ; mulhsu signed high"));
        assert!(!listing.contains("jit_runtime_binary"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_fuses_adjacent_div_rem_pairs_in_register_allocated_loops() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 4, 1, 7, 10, 11))); // div x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 6, 1, 8, 10, 11))); // rem x8, x10, x11
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1ff4))); // bne x5, x0, loop

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = (-37i64) as u64;
        cpu.registers[11usize] = 5;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[7usize], (-7i64) as u64);
        assert_eq!(cpu.registers[8usize], (-2i64) as u64);
        assert_eq!(cpu.registers[Pc], 16);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("fused div/rem"));
        assert_eq!(listing.matches("sdiv x").count(), 2);
        assert!(!listing.contains("stp x29, x30"));
        assert!(!listing.contains("stp x19, x20"));
        assert!(!listing.contains("selective loop save"));
        assert!(listing.contains("fused quotient"));
        assert!(listing.contains("invariant divisor x11 zero mask"));
        assert!(listing.contains("fused quotient nonzero divisor"));
        assert!(listing.contains("fused quotient div divisor mask"));
        assert!(listing.contains("fused remainder"));
        assert!(listing.contains("regalloc final loop decrement"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_fuses_div_rem_pairs_across_pure_integer_consumers() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 4, 1, 7, 10, 11))); // div x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x33, 6, 1, 7, 10, 11))); // rem x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_i(0x13, 0, 7, 0, 123))); // addi x7, x0, 123
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fe8))); // bne x5, x0, loop

        let quotient = (-7i64) as u64;
        let remainder = (-2i64) as u64;
        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = (-37i64) as u64;
        cpu.registers[11usize] = 5;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[7usize], 123);
        assert_eq!(cpu.registers[20usize], quotient ^ remainder);
        assert_eq!(cpu.registers[Pc], 28);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("fused div/rem"));
        assert_eq!(listing.matches("sdiv x").count(), 2);
        assert!(!listing.contains("stp x29, x30"));
        assert!(!listing.contains("stp x19, x20"));
        assert!(!listing.contains("selective loop save"));
        assert!(listing.contains("fused quotient"));
        assert!(listing.contains("invariant divisor x11 zero mask"));
        assert!(listing.contains("regalloc fused quotient consumer"));
        assert!(listing.contains("fused quotient div divisor mask"));
        assert!(listing.contains("regalloc fused remainder consumer"));
        assert_eq!(listing.matches("regalloc final loop decrement").count(), 2);
        assert!(!listing.contains("fused deferred remainder"));
        assert!(!listing.contains("fused quotient nonzero divisor"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_handles_zero_divisor_with_invariant_masks() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 4, 1, 7, 10, 11))); // div x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_r(0x33, 6, 1, 7, 10, 11))); // rem x7, x10, x11
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_i(0x13, 0, 7, 0, 123))); // addi x7, x0, 123
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fe8))); // bne x5, x0, loop

        let quotient = u64::MAX;
        let remainder = (-37i64) as u64;
        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = remainder;
        cpu.registers[11usize] = 0;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[7usize], 123);
        assert_eq!(cpu.registers[20usize], quotient ^ remainder);
        assert_eq!(cpu.registers[Pc], 28);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("invariant divisor x11 zero mask"));
        assert!(listing.contains("fused quotient div divisor mask"));
        assert!(listing.contains("jit_masked_divisor_loop"));
        assert!(!listing.contains("stp x29, x30"));
        assert!(!listing.contains("stp x19, x20"));
        assert!(!listing.contains("selective loop save"));
        assert!(listing.contains("orr x"));
        assert!(!listing.contains("fused div divisor zero"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_schedules_division_recurrence_loop() {
        let bin = division_recurrence_loop_bin();
        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        let mut cpu = division_recurrence_cpu(&bin, 3, (-123_456_789i64) as u64, 37, 97);
        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 66
            }
        );
        assert_division_recurrence_registers(&cpu, 3, (-123_456_789i64) as u64, 37, 97);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("division recurrence reciprocal signed guard"));
        assert!(listing.contains("division recurrence reciprocal div quotient"));
        assert!(listing.contains("division recurrence reciprocal divu quotient"));
        assert!(listing.contains("division recurrence scheduled div"));
        assert!(listing.contains("division recurrence masked"));
        assert!(!listing.contains("fused div/rem"));
        assert!(!listing.contains("selective loop save"));
        assert!(!listing.contains("stp x29, x30"));

        let mut zero_cpu = division_recurrence_cpu(&bin, 3, (-123_456_789i64) as u64, 0, 97);
        assert_eq!(
            engine.step(&mut zero_cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 66
            }
        );
        assert_division_recurrence_registers(&zero_cpu, 3, (-123_456_789i64) as u64, 0, 97);

        let mut changed_cpu = division_recurrence_cpu(&bin, 3, (-123_456_789i64) as u64, 41, 89);
        assert_eq!(
            engine.step(&mut changed_cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 66
            }
        );
        assert_division_recurrence_registers(&changed_cpu, 3, (-123_456_789i64) as u64, 41, 89);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_writes_mulhsu_directly_when_destination_does_not_alias_sources() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 2, 1, 7, 10, 13))); // mulhsu x7, x10, x13
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 20, 20, 7))); // xor x20, x20, x7
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1ff4))); // bne x5, x0, loop

        let lhs = (-123_456_789i64) as u64;
        let rhs = 97;
        let unsigned_high = (((lhs as u128) * (rhs as u128)) >> 64) as u64;
        let correction = if (lhs as i64) < 0 { rhs } else { 0 };
        let expected = unsigned_high.wrapping_sub(correction);

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = lhs;
        cpu.registers[13usize] = rhs;
        cpu.registers[20usize] = 0;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 12,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[7usize], expected);
        assert_eq!(cpu.registers[20usize], expected);
        assert_eq!(cpu.registers[Pc], 16);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("umulh x4, x2, x3 ; mulhsu unsigned high"));
        assert!(listing.contains("; initial loop counter x5"));
        assert!(listing.contains("subs x6, x6, #1 ; regalloc final loop decrement"));
        assert!(listing.contains("lsl x0, x10, #2 ; return executed instructions"));
        assert!(!listing.contains("cmp x6, x31"));
        assert!(!listing.contains("mov x4, x11 ; mulhsu result"));
        assert!(!listing.contains("selective loop save"));
        assert!(!listing.contains("stp x29, x30"));
        assert!(!listing.contains("stp x19, x20"));
        assert!(!listing.contains("stp x21"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_guest_traps_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0050_0093)); // addi x1, x0, 5
        bin.extend(rv64_word(0x0010_0073)); // ebreak

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let plan = BlockPlan::from_cpu(&cpu, 0).plan.unwrap();
        assert_eq!(plan.guest_instruction_count, 2);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 4 }));
        assert!(plan.profile_instructions[1]
            .text
            .contains("trap Ebreak pc=0x0000000000000004"));
        assert!(!plan
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_single_precision_arithmetic_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x00, 4, 1, 2))); // fadd.s f4, f1, f2, rne
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x04, 5, 2, 1))); // fsub.s f5, f2, f1, rne
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x08, 6, 1, 2))); // fmul.s f6, f1, f2, rne
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x0c, 7, 2, 1))); // fdiv.s f7, f2, f1, rne
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x2c, 8, 9, 0))); // fsqrt.s f8, f9, rne

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.float_registers[1usize] = 0xffff_ffff_0000_0000 | u64::from(1.25f32.to_bits());
        cpu.float_registers[2usize] = 0xffff_ffff_0000_0000 | u64::from(2.5f32.to_bits());
        cpu.float_registers[9usize] = 0xffff_ffff_0000_0000 | u64::from(16.0f32.to_bits());

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 5
            }
        );
        assert_eq!(cpu.float_registers[4usize] as u32, 3.75f32.to_bits());
        assert_eq!(cpu.float_registers[5usize] as u32, 1.25f32.to_bits());
        assert_eq!(cpu.float_registers[6usize] as u32, 3.125f32.to_bits());
        assert_eq!(cpu.float_registers[7usize] as u32, 2.0f32.to_bits());
        assert_eq!(cpu.float_registers[8usize] as u32, 4.0f32.to_bits());
        assert_eq!(cpu.registers[Pc], 20);
        assert!(!engine
            .cache
            .get(&0)
            .unwrap()
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_remaining_single_precision_float_operations_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r4(0x43, 0, 4, 1, 2, 3))); // fmadd.s f4, f1, f2, f3
        bin.extend(rv64_word(rv64_r4(0x47, 0, 5, 1, 2, 3))); // fmsub.s f5, f1, f2, f3
        bin.extend(rv64_word(rv64_r4(0x4b, 0, 6, 1, 2, 3))); // fnmsub.s f6, f1, f2, f3
        bin.extend(rv64_word(rv64_r4(0x4f, 0, 7, 1, 2, 3))); // fnmadd.s f7, f1, f2, f3
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x10, 12, 10, 11))); // fsgnj.s f12, f10, f11
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x10, 13, 10, 11))); // fsgnjn.s f13, f10, f11
        bin.extend(rv64_word(rv64_r(0x53, 2, 0x10, 14, 10, 11))); // fsgnjx.s f14, f10, f11
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x14, 15, 10, 11))); // fmin.s f15, f10, f11
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x14, 16, 10, 11))); // fmax.s f16, f10, f11
        bin.extend(rv64_word(rv64_r(0x53, 2, 0x50, 5, 10, 10))); // feq.s x5, f10, f10
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x50, 6, 11, 10))); // flt.s x6, f11, f10
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x50, 7, 11, 11))); // fle.s x7, f11, f11
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x70, 8, 10, 0))); // fclass.s x8, f10
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x70, 9, 11, 0))); // fmv.x.w x9, f11
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x78, 17, 20, 0))); // fmv.w.x f17, x20
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x60, 18, 18, 0))); // fcvt.w.s x18, f18, rtz
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x60, 19, 19, 1))); // fcvt.wu.s x19, f19, rtz
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x60, 21, 18, 2))); // fcvt.l.s x21, f18, rtz
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x60, 22, 19, 3))); // fcvt.lu.s x22, f19, rtz
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x68, 23, 23, 0))); // fcvt.s.w f23, x23
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x68, 24, 24, 1))); // fcvt.s.wu f24, x24
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x68, 25, 25, 2))); // fcvt.s.l f25, x25
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x68, 26, 26, 3))); // fcvt.s.lu f26, x26

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        let f32_box = |value: f32| 0xffff_ffff_0000_0000 | u64::from(value.to_bits());
        cpu.float_registers[1usize] = f32_box(2.0);
        cpu.float_registers[2usize] = f32_box(-3.0);
        cpu.float_registers[3usize] = f32_box(4.0);
        cpu.float_registers[10usize] = f32_box(1.5);
        cpu.float_registers[11usize] = f32_box(-2.0);
        cpu.float_registers[18usize] = f32_box(-42.75);
        cpu.float_registers[19usize] = f32_box(42.75);
        cpu.registers[20usize] = 0xbf80_0000;
        cpu.registers[23usize] = (-7i64) as u64;
        cpu.registers[24usize] = 7;
        cpu.registers[25usize] = (-7_000_000_000i64) as u64;
        cpu.registers[26usize] = 1u64 << 40;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 23
            }
        );
        assert_eq!(cpu.float_registers[4usize], f32_box(-2.0));
        assert_eq!(cpu.float_registers[5usize], f32_box(-10.0));
        assert_eq!(cpu.float_registers[6usize], f32_box(10.0));
        assert_eq!(cpu.float_registers[7usize], f32_box(2.0));
        assert_eq!(cpu.float_registers[12usize], f32_box(-1.5));
        assert_eq!(cpu.float_registers[13usize], f32_box(1.5));
        assert_eq!(cpu.float_registers[14usize], f32_box(-1.5));
        assert_eq!(cpu.float_registers[15usize], f32_box(-2.0));
        assert_eq!(cpu.float_registers[16usize], f32_box(1.5));
        assert_eq!(cpu.registers[5usize], 1);
        assert_eq!(cpu.registers[6usize], 1);
        assert_eq!(cpu.registers[7usize], 1);
        assert_eq!(cpu.registers[8usize], 1 << 6);
        assert_eq!(
            cpu.registers[9usize],
            sign_extend(u64::from((-2.0f32).to_bits()), 32) as u64
        );
        assert_eq!(cpu.float_registers[17usize], f32_box(-1.0));
        assert_eq!(cpu.registers[18usize], (-42i64) as u64);
        assert_eq!(cpu.registers[19usize], 42);
        assert_eq!(cpu.registers[21usize], (-42i64) as u64);
        assert_eq!(cpu.registers[22usize], 42);
        assert_eq!(cpu.float_registers[23usize], f32_box(-7.0));
        assert_eq!(cpu.float_registers[24usize], f32_box(7.0));
        assert_eq!(
            cpu.float_registers[25usize],
            f32_box(-7_000_000_000i64 as f32)
        );
        assert_eq!(cpu.float_registers[26usize], f32_box((1u64 << 40) as f32));
        assert_eq!(cpu.registers[Pc], 92);
        assert!(!engine
            .cache
            .get(&0)
            .unwrap()
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_implemented_double_precision_transfers_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x11, 4, 1, 2))); // fsgnj.d f4, f1, f2
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x11, 5, 1, 2))); // fsgnjn.d f5, f1, f2
        bin.extend(rv64_word(rv64_r(0x53, 2, 0x11, 6, 1, 2))); // fsgnjx.d f6, f1, f2
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x21, 7, 3, 0))); // fcvt.d.s f7, f3
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x71, 8, 2, 0))); // fmv.x.d x8, f2
        bin.extend(rv64_word(rv64_i(0x07, 3, 9, 9, 0))); // fld f9, 0(x9)
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x79, 10, 10, 0))); // fmv.d.x f10, x10

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.ram
            .add_region(MemoryRegion::new(
                0x100,
                8,
                4.25f64.to_bits().to_le_bytes().to_vec(),
            ))
            .unwrap();
        cpu.registers[9usize] = 0x100;
        cpu.float_registers[1usize] = 1.5f64.to_bits();
        cpu.float_registers[2usize] = (-2.0f64).to_bits();
        cpu.float_registers[3usize] = 0xffff_ffff_0000_0000 | u64::from(1.25f32.to_bits());
        cpu.registers[10usize] = 3.5f64.to_bits();

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 7
            }
        );
        assert_eq!(cpu.float_registers[4usize], (-1.5f64).to_bits());
        assert_eq!(cpu.float_registers[5usize], 1.5f64.to_bits());
        assert_eq!(cpu.float_registers[6usize], (-1.5f64).to_bits());
        assert_eq!(cpu.float_registers[7usize], 1.25f64.to_bits());
        assert_eq!(cpu.registers[8usize], (-2.0f64).to_bits());
        assert_eq!(cpu.float_registers[9usize], 4.25f64.to_bits());
        assert_eq!(cpu.float_registers[10usize], 3.5f64.to_bits());
        assert_eq!(cpu.registers[Pc], 28);
        assert!(!engine
            .cache
            .get(&0)
            .unwrap()
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_double_precision_arithmetic_and_conversions_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x01, 10, 1, 2))); // fadd.d f10, f1, f2
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x05, 11, 1, 2))); // fsub.d f11, f1, f2
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x09, 12, 1, 2))); // fmul.d f12, f1, f2
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x0d, 13, 1, 2))); // fdiv.d f13, f1, f2
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x2d, 14, 4, 0))); // fsqrt.d f14, f4
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x15, 15, 1, 5))); // fmin.d f15, f1, f5
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x15, 16, 1, 5))); // fmax.d f16, f1, f5
        bin.extend(rv64_word(rv64_r(0x53, 2, 0x51, 17, 1, 1))); // feq.d x17, f1, f1
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x51, 18, 5, 1))); // flt.d x18, f5, f1
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x51, 19, 5, 5))); // fle.d x19, f5, f5
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x71, 20, 1, 0))); // fclass.d x20, f1
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x20, 21, 1, 1))); // fcvt.s.d f21, f1
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x61, 22, 5, 0))); // fcvt.w.d x22, f5
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x61, 23, 1, 1))); // fcvt.wu.d x23, f1
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x69, 24, 24, 0))); // fcvt.d.w f24, x24
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x69, 25, 25, 1))); // fcvt.d.wu f25, x25
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x61, 30, 5, 2))); // fcvt.l.d x30, f5
        bin.extend(rv64_word(rv64_r(0x53, 1, 0x61, 31, 1, 3))); // fcvt.lu.d x31, f1
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x69, 30, 24, 2))); // fcvt.d.l f30, x24
        bin.extend(rv64_word(rv64_r(0x53, 0, 0x69, 31, 25, 3))); // fcvt.d.lu f31, x25
        bin.extend(rv64_word(rv64_r4_fmt(0x43, 1, 0, 26, 1, 2, 3))); // fmadd.d
        bin.extend(rv64_word(rv64_r4_fmt(0x47, 1, 0, 27, 1, 2, 3))); // fmsub.d
        bin.extend(rv64_word(rv64_r4_fmt(0x4f, 1, 0, 28, 1, 2, 3))); // fnmadd.d
        bin.extend(rv64_word(rv64_r4_fmt(0x4b, 1, 0, 29, 1, 2, 3))); // fnmsub.d

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.float_registers[1usize] = 6.0f64.to_bits();
        cpu.float_registers[2usize] = 2.0f64.to_bits();
        cpu.float_registers[3usize] = 1.5f64.to_bits();
        cpu.float_registers[4usize] = 9.0f64.to_bits();
        cpu.float_registers[5usize] = (-2.0f64).to_bits();
        cpu.registers[24usize] = (-7i64) as u64;
        cpu.registers[25usize] = 7;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 24
            }
        );
        assert_eq!(cpu.float_registers[10usize], 8.0f64.to_bits());
        assert_eq!(cpu.float_registers[11usize], 4.0f64.to_bits());
        assert_eq!(cpu.float_registers[12usize], 12.0f64.to_bits());
        assert_eq!(cpu.float_registers[13usize], 3.0f64.to_bits());
        assert_eq!(cpu.float_registers[14usize], 3.0f64.to_bits());
        assert_eq!(cpu.float_registers[15usize], (-2.0f64).to_bits());
        assert_eq!(cpu.float_registers[16usize], 6.0f64.to_bits());
        assert_eq!(cpu.registers[17usize], 1);
        assert_eq!(cpu.registers[18usize], 1);
        assert_eq!(cpu.registers[19usize], 1);
        assert_eq!(cpu.registers[20usize], 1 << 6);
        assert_eq!(cpu.float_registers[21usize], f32_box(6.0));
        assert_eq!(cpu.registers[22usize], (-2i64) as u64);
        assert_eq!(cpu.registers[23usize], 6);
        assert_eq!(cpu.float_registers[24usize], (-7.0f64).to_bits());
        assert_eq!(cpu.float_registers[25usize], 7.0f64.to_bits());
        assert_eq!(cpu.registers[30usize], (-2i64) as u64);
        assert_eq!(cpu.registers[31usize], 6);
        assert_eq!(cpu.float_registers[26usize], 13.5f64.to_bits());
        assert_eq!(cpu.float_registers[27usize], 10.5f64.to_bits());
        assert_eq!(cpu.float_registers[28usize], (-13.5f64).to_bits());
        assert_eq!(cpu.float_registers[29usize], (-10.5f64).to_bits());
        assert_eq!(cpu.float_registers[30usize], (-7.0f64).to_bits());
        assert_eq!(cpu.float_registers[31usize], 7.0f64.to_bits());
        assert_eq!(cpu.registers[Pc], 96);
        assert!(!engine
            .cache
            .get(&0)
            .unwrap()
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_reports_guest_traps_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0050_0093)); // addi x1, x0, 5
        bin.extend(rv64_word(0x0010_0073)); // ebreak

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        assert!(matches!(
            engine.step(&mut cpu),
            Err(JitError::RuntimeFault { pc: 0, reason }) if reason.contains("ebreak")
        ));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_treats_fence_as_a_native_nop() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0010_0093)); // addi x1, x0, 1
        bin.extend(rv64_word(0x0000_000f)); // fence
        bin.extend(rv64_word(0x0000_100f)); // fence.i

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::new().unwrap();

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 3
            }
        );
        assert_eq!(cpu.registers[1usize], 1);
        assert_eq!(cpu.registers[Pc], 12);
        assert!(!engine
            .cache
            .get(&0)
            .unwrap()
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_lowers_fcsr_csr_operations_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x73, 5, 1, 0b1_0101, 0x001))); // csrrwi x1, fflags, 21
        bin.extend(rv64_word(rv64_i(0x73, 6, 2, 0b0_1010, 0x001))); // csrrsi x2, fflags, 10
        bin.extend(rv64_word(rv64_i(0x73, 7, 3, 0b1_0000, 0x001))); // csrrci x3, fflags, 16
        bin.extend(rv64_word(rv64_i(0x73, 2, 4, 0, 0x001))); // csrrs x4, fflags, x0
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 0, 0x63))); // addi x6, x0, fcsr value
        bin.extend(rv64_word(rv64_i(0x73, 1, 5, 6, 0x003))); // csrrw x5, fcsr, x6

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 6
            }
        );
        assert_eq!(cpu.registers[1usize], 0);
        assert_eq!(cpu.registers[2usize], 0b1_0101);
        assert_eq!(cpu.registers[3usize], 0b1_1111);
        assert_eq!(cpu.registers[4usize], 0b0_1111);
        assert_eq!(cpu.registers[5usize], 0b0_1111);
        assert_eq!(cpu.fcsr.bits(), 0x63);
        assert_eq!(cpu.registers[Pc], 24);
        assert!(!engine
            .cache
            .get(&0)
            .unwrap()
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_executes_compressed_nop_instruction() {
        let mut bin = Vec::new();
        bin.extend(rv64_halfword(0x0001)); // c.nop
        bin.extend(rv64_halfword(0x0001)); // extra bytes for the existing fetch width

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::new().unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[Pc], 2);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_executes_conditional_branch_instruction() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_beq(0, 0, 8))); // beq x0, x0, +8
        bin.extend(rv64_word(0x0010_0093)); // skipped if branch is taken
        bin.extend(rv64_word(0x0020_0093)); // branch target

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::new().unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[Pc], 8);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_inlines_forward_direct_jumps_inside_compiled_blocks() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 0, 1))); // addi x1, x0, 1
        bin.extend(rv64_word(0x0080_006f)); // jal x0, +8
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 1, 100))); // skipped
        bin.extend(rv64_word(rv64_i(0x13, 0, 2, 0, 2))); // addi x2, x0, 2
        bin.extend(rv64_word(rv64_beq(0, 0, 8))); // beq x0, x0, +8
        bin.extend(rv64_word(rv64_i(0x13, 0, 2, 2, 100))); // skipped

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 4,
            }
        );
        assert_eq!(cpu.registers[1usize], 1);
        assert_eq!(cpu.registers[2usize], 2);
        assert_eq!(cpu.registers[Pc], 24);

        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.instruction_count, 4);
        assert!(block
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("inlined")));
    }

    fn counted_diamond_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 7, 28, 5, 1))); // andi x28, x5, 1
        bin.extend(rv64_word(rv64_beq(28, 0, 12))); // beq x28, x0, zero arm
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 3))); // addi x6, x6, 3
        bin.extend(rv64_word(rv64_jal(0, 8))); // jal x0, tail
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 7))); // addi x6, x6, 7
        bin.extend(rv64_word(rv64_i(0x13, 0, 7, 7, 1))); // addi x7, x7, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fe4))); // bne x5, x0, loop
        bin
    }

    fn counted_diamond_mask3_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 7, 28, 5, 3))); // andi x28, x5, 3
        bin.extend(rv64_word(rv64_beq(28, 0, 12))); // beq x28, x0, zero arm
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 3))); // addi x6, x6, 3
        bin.extend(rv64_word(rv64_jal(0, 8))); // jal x0, tail
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 7))); // addi x6, x6, 7
        bin.extend(rv64_word(rv64_i(0x13, 0, 7, 7, 1))); // addi x7, x7, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fe4))); // bne x5, x0, loop
        bin
    }

    fn counted_diamond_mask6_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 7, 28, 5, 6))); // andi x28, x5, 6
        bin.extend(rv64_word(rv64_beq(28, 0, 12))); // beq x28, x0, zero arm
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 3))); // addi x6, x6, 3
        bin.extend(rv64_word(rv64_jal(0, 8))); // jal x0, tail
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 7))); // addi x6, x6, 7
        bin.extend(rv64_word(rv64_i(0x13, 0, 7, 7, 1))); // addi x7, x7, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fe4))); // bne x5, x0, loop
        bin
    }

    fn arithmetic_xor_toggle_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 0, 0, 7, 7, 6))); // add x7, x7, x6
        bin.extend(rv64_word(rv64_i(0x13, 4, 6, 6, 0x5a5))); // xori x6, x6, 0x5a5
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1ff4))); // bne x5, x0, loop
        bin
    }

    fn store_load_forward_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 7, 29, 7, 24))); // andi x29, x7, 24
        bin.extend(rv64_word(rv64_r(0x33, 0, 0, 30, 28, 29))); // add x30, x28, x29
        bin.extend(rv64_word(rv64_s(3, 30, 6, 0))); // sd x6, 0(x30)
        bin.extend(rv64_word(rv64_i(0x03, 3, 31, 30, 0))); // ld x31, 0(x30)
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 6, 6, 31))); // xor x6, x6, x31
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 1))); // addi x6, x6, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 7, 7, 8))); // addi x7, x7, 8
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, -1))); // addi x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1fe0))); // bne x5, x0, loop
        bin
    }

    fn fibonacci_recurrence_loop_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 11, 15, 0))); // mv x11, x15
        bin.extend(rv64_word(rv64_i(0x1b, 0, 12, 12, -1))); // addiw x12, x12, -1
        bin.extend(rv64_word(rv64_r(0x33, 0, 0, 15, 15, 14))); // add x15, x15, x14
        bin.extend(rv64_word(rv64_r(0x33, 4, 0, 13, 13, 15))); // xor x13, x13, x15
        bin.extend(rv64_word(rv64_i(0x13, 0, 14, 11, 0))); // mv x14, x11
        bin.extend(rv64_word(rv64_bne(12, 0, 0x1fec))); // bne x12, x0, loop
        bin
    }

    fn expected_fibonacci_recurrence(iterations: u32) -> (u64, u64, u64) {
        let mut previous = 1u64;
        let mut current = 1u64;
        let mut checksum = 0u64;
        for _ in 0..iterations {
            let saved = current;
            current = current.wrapping_add(previous);
            checksum ^= current;
            previous = saved;
        }
        (current, previous, checksum)
    }

    #[test]
    fn optimized_planner_builds_counted_diamond_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(counted_diamond_loop_bin());

        let planned = BlockPlan::optimized_from_cpu(&cpu, 0);
        let plan = planned.plan.expect("optimized plan");

        assert_eq!(plan.start_pc, 0);
        assert_eq!(plan.end_pc, 32);
        assert_eq!(plan.guest_instruction_count, 7);
        assert_eq!(plan.fingerprint.len(), 8);
        assert_eq!(plan.successors(), vec![32]);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 28 }));
        assert!(matches!(
            plan.operations.as_slice(),
            [BlockOperation {
                kind: BlockOperationKind::Native(NativeInstruction::CountedDiamondLoop(
                    CountedDiamondLoop {
                        counter: 5,
                        accumulator: 6,
                        parity_register: 28,
                        iteration_register: 7,
                        mask: 1,
                        zero_accumulator_delta: 7,
                        nonzero_accumulator_delta: 3,
                        iteration_delta: 1,
                        counter_delta: -1,
                        exit_pc: 32,
                        zero_guest_instructions: 6,
                        nonzero_guest_instructions: 7,
                    },
                )),
                ..
            }]
        ));
    }

    #[test]
    fn optimized_planner_builds_arithmetic_xor_toggle_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(arithmetic_xor_toggle_loop_bin());

        let planned = BlockPlan::optimized_from_cpu(&cpu, 0);
        let plan = planned.plan.expect("optimized plan");

        assert_eq!(plan.start_pc, 0);
        assert_eq!(plan.end_pc, 16);
        assert_eq!(plan.guest_instruction_count, 4);
        assert_eq!(plan.fingerprint.len(), 4);
        assert_eq!(plan.successors(), vec![16]);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 12 }));
        assert!(matches!(
            plan.operations.as_slice(),
            [BlockOperation {
                kind: BlockOperationKind::Native(NativeInstruction::ArithmeticXorToggleLoop(
                    ArithmeticXorToggleLoop {
                        counter: 5,
                        accumulator: 7,
                        value_register: 6,
                        xor_imm: 0x5a5,
                        exit_pc: 16,
                        guest_instructions: 4,
                    },
                )),
                ..
            }]
        ));
    }

    #[test]
    fn optimized_planner_builds_fibonacci_recurrence_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(fibonacci_recurrence_loop_bin());

        let planned = BlockPlan::optimized_from_cpu(&cpu, 0);
        let plan = planned.plan.expect("optimized plan");

        assert_eq!(plan.start_pc, 0);
        assert_eq!(plan.end_pc, 24);
        assert_eq!(plan.guest_instruction_count, 6);
        assert_eq!(plan.fingerprint.len(), 6);
        assert_eq!(plan.successors(), vec![24]);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 20 }));
        assert!(matches!(
            plan.operations.as_slice(),
            [BlockOperation {
                kind: BlockOperationKind::Native(NativeInstruction::FibonacciRecurrenceLoop(
                    FibonacciRecurrenceLoop {
                        counter: 12,
                        current_register: 15,
                        previous_register: 14,
                        checksum_register: 13,
                        saved_register: 11,
                        exit_pc: 24,
                        guest_instructions: 6,
                    },
                )),
                ..
            }]
        ));
    }

    #[test]
    fn optimized_planner_builds_store_load_forward_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(store_load_forward_loop_bin());

        let planned = BlockPlan::optimized_from_cpu(&cpu, 0);
        let plan = planned.plan.expect("optimized plan");

        assert_eq!(plan.start_pc, 0);
        assert_eq!(plan.end_pc, 36);
        assert_eq!(plan.guest_instruction_count, 9);
        assert_eq!(plan.fingerprint.len(), 9);
        assert_eq!(plan.successors(), vec![36]);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 32 }));
        assert!(matches!(
            plan.operations.as_slice(),
            [BlockOperation {
                kind: BlockOperationKind::Native(NativeInstruction::StoreLoadForwardLoop(
                    StoreLoadForwardLoop {
                        counter: 5,
                        value_register: 6,
                        offset_register: 7,
                        base_register: 28,
                        index_register: 29,
                        address_register: 30,
                        loaded_register: 31,
                        mask: 24,
                        width: MemoryWidth::Double,
                        value_delta_after_forwarded_xor: 1,
                        offset_delta: 8,
                        counter_delta: -1,
                        exit_pc: 36,
                        guest_instructions: 9,
                    },
                )),
                ..
            }]
        ));
    }

    #[test]
    fn trace_planner_records_observed_branch_path_with_side_exit_guard() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_diamond_loop_bin());
        cpu.registers[5usize] = 3;

        let planned = BlockPlan::trace_from_cpu(&cpu, 0);
        let plan = planned.plan.expect("trace plan");

        assert_eq!(plan.start_pc, 0);
        assert_eq!(plan.end_pc, 0);
        assert_eq!(plan.guest_instruction_count, 6);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 24 }));
        assert!(matches!(
            plan.operations[1].kind(),
            BlockOperationKind::Native(NativeInstruction::TraceGuard {
                rs1: 7,
                rs2: 0,
                condition: IntegerBranchCondition::Eq,
                continue_on_taken: false,
                continue_pc: 8,
                side_exit_pc: 16,
                executed_instructions: 2,
            })
        ));
        assert!(matches!(
            plan.operations.last().map(BlockOperation::kind),
            Some(BlockOperationKind::Native(
                NativeInstruction::TraceLoopGuard {
                    rs1: 5,
                    rs2: 0,
                    condition: IntegerBranchCondition::Ne,
                    continue_on_taken: true,
                    loop_pc: 0,
                    side_exit_pc: 28,
                    guest_instruction_count: 6,
                }
            ))
        ));
    }

    #[test]
    fn trace_planner_records_memory_dependent_branch_path() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_memory_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 24, vec![0; 24]))
            .unwrap();
        cpu.ram.write_doubleword(0x1000, 1).unwrap();
        cpu.ram.write_doubleword(0x1008, 1).unwrap();
        cpu.ram.write_doubleword(0x1010, 1).unwrap();
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = 0x1000;

        let planned = BlockPlan::trace_from_cpu(&cpu, 0);
        let plan = planned.plan.expect("trace plan");

        assert_eq!(plan.start_pc, 0);
        assert_eq!(plan.end_pc, 0);
        assert_eq!(plan.guest_instruction_count, 5);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 16 }));
        assert!(matches!(
            plan.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 7,
                rs1: 10,
                imm: 0,
                width: MemoryWidth::Double,
                signed: false,
            })
        ));
        assert!(matches!(
            plan.operations[1].kind(),
            BlockOperationKind::Native(NativeInstruction::TraceGuard {
                rs1: 7,
                rs2: 0,
                condition: IntegerBranchCondition::Eq,
                continue_on_taken: false,
                continue_pc: 8,
                side_exit_pc: 20,
                executed_instructions: 2,
            })
        ));
        assert!(matches!(
            plan.operations.last().map(BlockOperation::kind),
            Some(BlockOperationKind::Native(
                NativeInstruction::TraceLoopGuard {
                    rs1: 6,
                    rs2: 5,
                    condition: IntegerBranchCondition::Ne,
                    continue_on_taken: true,
                    loop_pc: 0,
                    side_exit_pc: 20,
                    guest_instruction_count: 5,
                }
            ))
        ));
    }

    #[test]
    fn trace_planner_forwards_guest_stores_to_later_load_guards() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_store_load_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 24, vec![0; 24]))
            .unwrap();
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = 0x1000;
        cpu.registers[11usize] = 1;

        let planned = BlockPlan::trace_from_cpu(&cpu, 0);
        let plan = planned.plan.expect("trace plan");

        assert_eq!(plan.start_pc, 0);
        assert_eq!(plan.end_pc, 0);
        assert_eq!(plan.guest_instruction_count, 6);
        assert!(matches!(plan.stop, BlockStop::ControlFlow { pc: 20 }));
        assert!(matches!(
            plan.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::Store {
                rs1: 10,
                rs2: 11,
                imm: 0,
                width: MemoryWidth::Double,
            })
        ));
        assert!(matches!(
            plan.operations[2].kind(),
            BlockOperationKind::Native(NativeInstruction::TraceGuard {
                rs1: 7,
                rs2: 0,
                condition: IntegerBranchCondition::Eq,
                continue_on_taken: false,
                continue_pc: 12,
                side_exit_pc: 24,
                executed_instructions: 3,
            })
        ));
        assert!(matches!(
            plan.operations.last().map(BlockOperation::kind),
            Some(BlockOperationKind::Native(
                NativeInstruction::TraceLoopGuard {
                    rs1: 6,
                    rs2: 5,
                    condition: IntegerBranchCondition::Ne,
                    continue_on_taken: true,
                    loop_pc: 0,
                    side_exit_pc: 24,
                    guest_instruction_count: 6,
                }
            ))
        ));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_counted_diamond_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(counted_diamond_loop_bin());
        cpu.registers[5usize] = 4;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        let step = engine.step(&mut cpu).unwrap();

        assert_eq!(
            step,
            JitStep::Native {
                pc: 0,
                instructions: 26,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 20);
        assert_eq!(cpu.registers[7usize], 4);
        assert_eq!(cpu.registers[28usize], 1);
        assert_eq!(cpu.registers[Pc], 32);

        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        let listing = block
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("counted diamond zero-arm count"));
        assert!(listing.contains("counted diamond instruction count"));
        assert!(!listing.contains("b.ne .counted_diamond_loop"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_counted_diamond_loop_region_for_odd_and_wrapping_counts() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(counted_diamond_loop_bin());
        cpu.registers[5usize] = 5;

        let mut engine = JitEngine::new().unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 33,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 23);
        assert_eq!(cpu.registers[7usize], 5);
        assert_eq!(cpu.registers[28usize], 1);
        assert_eq!(cpu.registers[Pc], 32);

        let mut cpu = RV64GC::new();
        cpu.load_bin(counted_diamond_loop_bin());
        let mut engine = JitEngine::new().unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 1u64 << 63,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 0);
        assert_eq!(cpu.registers[7usize], 0);
        assert_eq!(cpu.registers[28usize], 1);
        assert_eq!(cpu.registers[Pc], 32);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_closed_forms_power_of_two_counted_diamond_masks() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(counted_diamond_mask3_loop_bin());
        cpu.registers[5usize] = 6;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 41,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 22);
        assert_eq!(cpu.registers[7usize], 6);
        assert_eq!(cpu.registers[28usize], 1);
        assert_eq!(cpu.registers[Pc], 32);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("counted diamond zero-arm count"));
        assert!(!listing.contains("b.ne .counted_diamond_loop"));
        assert!(!listing.contains("stp x19, x20"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_iterative_counted_diamond_loop_region_without_frame() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(counted_diamond_mask6_loop_bin());
        cpu.registers[5usize] = 6;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 41,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 22);
        assert_eq!(cpu.registers[7usize], 6);
        assert_eq!(cpu.registers[28usize], 0);
        assert_eq!(cpu.registers[Pc], 32);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("subs x2, x2, #1 ; counted diamond backedge"));
        assert!(listing.contains("b.ne .counted_diamond_loop"));
        assert!(!listing.contains("cmp x2, xzr ; counted diamond backedge"));
        assert!(!listing.contains("stp x19, x20"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_arithmetic_xor_toggle_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(arithmetic_xor_toggle_loop_bin());
        cpu.registers[5usize] = 5;
        cpu.registers[6usize] = 1;
        cpu.registers[7usize] = 0;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 20,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 0x5a4);
        assert_eq!(cpu.registers[7usize], 2891);
        assert_eq!(cpu.registers[Pc], 16);

        let listing = engine
            .cache
            .get(&0)
            .unwrap()
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("xor-toggle pair sum"));
        assert!(listing.contains("xor-toggle paired contribution"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_fibonacci_recurrence_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(fibonacci_recurrence_loop_bin());
        cpu.registers[11usize] = 1;
        cpu.registers[12usize] = 13;
        cpu.registers[13usize] = 0;
        cpu.registers[14usize] = 1;
        cpu.registers[15usize] = 1;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 78,
            }
        );
        let (expected_current, expected_previous, expected_checksum) =
            expected_fibonacci_recurrence(13);
        assert_eq!(cpu.registers[11usize], expected_previous);
        assert_eq!(cpu.registers[12usize], 0);
        assert_eq!(cpu.registers[13usize], expected_checksum);
        assert_eq!(cpu.registers[14usize], expected_previous);
        assert_eq!(cpu.registers[15usize], expected_current);
        assert_eq!(cpu.registers[Pc], 24);

        let listing = engine
            .cache
            .get(&0)
            .unwrap()
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("fib executed instructions"));
        assert!(listing.contains("fib next"));
        assert!(listing.contains("#256 ; fib unrolled count"));
        assert!(listing.contains("b.hs .fib_unrolled_loop"));
        assert!(listing.contains("subs x9, x9, #1 ; fib tail backedge"));
        assert!(!listing.contains("cmp x9, xzr ; fib tail backedge"));
        assert!(!listing.contains("stp x19, x20"));
        assert!(!listing.contains("jit_runtime_fibonacci_recurrence"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_uses_fibonacci_recurrence_loop_region_after_hot_reentry() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(fibonacci_recurrence_loop_bin());
        cpu.registers[11usize] = 1;
        cpu.registers[12usize] = 5;
        cpu.registers[13usize] = 0;
        cpu.registers[14usize] = 1;
        cpu.registers[15usize] = 1;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            hot_threshold: 1,
            ..JitOptions::default()
        })
        .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 6,
            }
        );
        assert_eq!(cpu.registers[12usize], 4);
        assert_eq!(cpu.registers[Pc], 0);
        assert_eq!(engine.cache.get(&0).unwrap().tier, JitTier::Baseline);

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 24,
            }
        );
        let (expected_current, expected_previous, expected_checksum) =
            expected_fibonacci_recurrence(5);
        assert_eq!(cpu.registers[11usize], expected_previous);
        assert_eq!(cpu.registers[12usize], 0);
        assert_eq!(cpu.registers[13usize], expected_checksum);
        assert_eq!(cpu.registers[14usize], expected_previous);
        assert_eq!(cpu.registers[15usize], expected_current);
        assert_eq!(cpu.registers[Pc], 24);

        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        assert!(block
            .native_listing
            .iter()
            .any(|emission| emission.text.contains("fib executed instructions")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_store_load_forward_loop_region() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(store_load_forward_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 64, vec![0; 64]))
            .unwrap();
        cpu.registers[5usize] = 4;
        cpu.registers[6usize] = 0;
        cpu.registers[7usize] = 0;
        cpu.registers[28usize] = 0x1000;

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 36,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 1);
        assert_eq!(cpu.registers[7usize], 32);
        assert_eq!(cpu.registers[28usize], 0x1000);
        assert_eq!(cpu.registers[29usize], 24);
        assert_eq!(cpu.registers[30usize], 0x1018);
        assert_eq!(cpu.registers[31usize], 1);
        assert_eq!(cpu.registers[Pc], 36);
        assert_eq!(cpu.ram.read_doubleword(0x1000).unwrap(), 0);
        assert_eq!(cpu.ram.read_doubleword(0x1008).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1010).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1018).unwrap(), 1);

        let listing = engine
            .cache
            .get(&0)
            .unwrap()
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("jit_runtime_direct_write_ptr"));
        assert!(listing.contains("final-state guest store"));
        assert!(listing.contains("final-state guest instruction count"));
        assert!(!listing.contains("b.ne .direct_store_loop"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_store_load_forward_loop_region_with_ring_wraparound() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(store_load_forward_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 64, vec![0; 64]))
            .unwrap();
        cpu.registers[5usize] = 5;
        cpu.registers[6usize] = 99;
        cpu.registers[7usize] = 0;
        cpu.registers[28usize] = 0x1000;

        let mut engine = JitEngine::new().unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 45,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 1);
        assert_eq!(cpu.registers[7usize], 40);
        assert_eq!(cpu.registers[28usize], 0x1000);
        assert_eq!(cpu.registers[29usize], 0);
        assert_eq!(cpu.registers[30usize], 0x1000);
        assert_eq!(cpu.registers[31usize], 1);
        assert_eq!(cpu.registers[Pc], 36);
        assert_eq!(cpu.ram.read_doubleword(0x1000).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1008).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1010).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1018).unwrap(), 1);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_executes_store_load_forward_loop_region_for_zero_counter_cycle() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(store_load_forward_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 64, vec![0; 64]))
            .unwrap();
        cpu.registers[5usize] = 0;
        cpu.registers[6usize] = 99;
        cpu.registers[7usize] = 0;
        cpu.registers[28usize] = 0x1000;

        let mut engine = JitEngine::new().unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 0,
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[6usize], 1);
        assert_eq!(cpu.registers[7usize], 0);
        assert_eq!(cpu.registers[28usize], 0x1000);
        assert_eq!(cpu.registers[29usize], 24);
        assert_eq!(cpu.registers[30usize], 0x1018);
        assert_eq!(cpu.registers[31usize], 1);
        assert_eq!(cpu.registers[Pc], 36);
        assert_eq!(cpu.ram.read_doubleword(0x1000).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1008).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1010).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1018).unwrap(), 1);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn trace_jit_executes_hot_path_and_side_exit() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_diamond_loop_bin());
        cpu.registers[5usize] = 3;

        let mut engine = JitEngine::with_options(JitOptions {
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Trace, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 8,
            }
        );
        assert_eq!(cpu.registers[5usize], 2);
        assert_eq!(cpu.registers[6usize], 3);
        assert_eq!(cpu.registers[7usize], 0);
        assert_eq!(cpu.registers[Pc], 16);

        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Trace);
        let listing = block
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("regalloc trace guard"));
        assert!(listing.contains("regalloc trace loop guard"));
        assert!(listing.contains("trace guard"));
        assert!(listing.contains("trace loop guard"));
        assert!(!listing.contains("load guest x7"));
        assert!(listing.contains("trace_loop_start"));
        assert!(listing.contains("trace_side_exit"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn regalloc_trace_side_exit_preserves_state_before_future_writes() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_diamond_loop_bin());
        cpu.registers[5usize] = 3;
        cpu.registers[6usize] = 40;

        let mut engine = JitEngine::with_options(JitOptions {
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Trace, "test", 0)
            .unwrap();

        cpu.registers[5usize] = 2;
        cpu.registers[6usize] = 40;
        cpu.registers[7usize] = 99;
        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 2,
            }
        );
        assert_eq!(cpu.registers[5usize], 2);
        assert_eq!(cpu.registers[6usize], 40);
        assert_eq!(cpu.registers[7usize], 0);
        assert_eq!(cpu.registers[Pc], 16);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("regalloc trace guard"));
        assert!(listing.contains("trace_side_exit"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn trace_jit_executes_memory_dependent_loop() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_memory_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 24, vec![0; 24]))
            .unwrap();
        cpu.ram.write_doubleword(0x1000, 1).unwrap();
        cpu.ram.write_doubleword(0x1008, 1).unwrap();
        cpu.ram.write_doubleword(0x1010, 1).unwrap();
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = 0x1000;

        let mut engine = JitEngine::with_options(JitOptions {
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Trace, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 15,
            }
        );
        assert_eq!(cpu.registers[5usize], 3);
        assert_eq!(cpu.registers[6usize], 3);
        assert_eq!(cpu.registers[7usize], 1);
        assert_eq!(cpu.registers[10usize], 0x1018);
        assert_eq!(cpu.registers[Pc], 20);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("jit_runtime_try_direct_read_ptr"));
        assert!(listing.contains("direct trace unrolled load"));
        assert!(listing.contains("regalloc trace guard"));
        assert!(listing.contains("regalloc trace loop guard"));
        assert!(!listing.contains("load guest x7"));
        assert!(listing.contains("trace guard"));
        assert!(listing.contains("trace loop guard"));
        assert!(listing.contains("trace_loop_start"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn trace_jit_direct_load_loop_handles_load_guard_side_exit() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_memory_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 24, vec![0; 24]))
            .unwrap();
        cpu.ram.write_doubleword(0x1000, 1).unwrap();
        cpu.ram.write_doubleword(0x1008, 0).unwrap();
        cpu.ram.write_doubleword(0x1010, 1).unwrap();
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = 0x1000;

        let mut engine = JitEngine::with_options(JitOptions {
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Trace, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 7,
            }
        );
        assert_eq!(cpu.registers[5usize], 3);
        assert_eq!(cpu.registers[6usize], 1);
        assert_eq!(cpu.registers[7usize], 0);
        assert_eq!(cpu.registers[10usize], 0x1008);
        assert_eq!(cpu.registers[Pc], 20);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("cbz .direct_trace_load_side_exit"));
        assert!(listing.contains("direct_trace_load_side_exit"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn trace_jit_direct_byte_load_loop_handles_load_guard_side_exit() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_byte_memory_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 3, vec![1, 0, 1]))
            .unwrap();
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = 0x1000;

        let mut engine = JitEngine::with_options(JitOptions {
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Trace, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 7,
            }
        );
        assert_eq!(cpu.registers[5usize], 3);
        assert_eq!(cpu.registers[6usize], 1);
        assert_eq!(cpu.registers[7usize], 0);
        assert_eq!(cpu.registers[10usize], 0x1001);
        assert_eq!(cpu.registers[Pc], 20);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("ldr q0"));
        assert!(listing.contains("cmeq v0.16b"));
        assert!(listing.contains("umaxv b0"));
        assert!(listing.contains("ldrb w"));
        assert!(listing.contains("direct_trace_load_side_exit"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn trace_jit_direct_byte_load_loop_completes_vector_chunk() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_byte_memory_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 16, (1u8..=16).collect()))
            .unwrap();
        cpu.registers[5usize] = 16;
        cpu.registers[10usize] = 0x1000;

        let mut engine = JitEngine::with_options(JitOptions {
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Trace, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 80,
            }
        );
        assert_eq!(cpu.registers[5usize], 16);
        assert_eq!(cpu.registers[6usize], 16);
        assert_eq!(cpu.registers[7usize], 16);
        assert_eq!(cpu.registers[10usize], 0x1010);
        assert_eq!(cpu.registers[Pc], 20);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("direct trace vector load"));
        assert!(listing.contains("direct trace vector final load"));
        assert!(listing.contains("direct_trace_load_vector_loop"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn trace_jit_executes_store_load_forwarded_loop() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_store_load_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 24, vec![0; 24]))
            .unwrap();
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = 0x1000;
        cpu.registers[11usize] = 1;

        let mut engine = JitEngine::with_options(JitOptions {
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Trace, "test", 0)
            .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 18,
            }
        );
        assert_eq!(cpu.registers[5usize], 3);
        assert_eq!(cpu.registers[6usize], 3);
        assert_eq!(cpu.registers[7usize], 1);
        assert_eq!(cpu.registers[10usize], 0x1018);
        assert_eq!(cpu.registers[Pc], 24);
        assert_eq!(cpu.ram.read_doubleword(0x1000).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1008).unwrap(), 1);
        assert_eq!(cpu.ram.read_doubleword(0x1010).unwrap(), 1);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("jit_runtime_store_u64"));
        assert!(!listing.contains("jit_runtime_load_u64"));
        assert!(listing.contains("regalloc trace guard"));
        assert!(listing.contains("regalloc trace loop guard"));
        assert!(!listing.contains("load guest x7"));
        assert!(listing.contains("trace guard"));
        assert!(listing.contains("trace loop guard"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_trace_flag_promotes_hot_branch_path_to_trace_tier() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_diamond_loop_bin());
        cpu.registers[5usize] = 3;

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            trace_compilation: true,
            hot_threshold: 1,
            tier_budgeting: false,
            ..JitOptions::default()
        })
        .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 2,
            }
        );
        assert_eq!(cpu.registers[Pc], 8);
        assert_eq!(engine.cache.get(&0).unwrap().tier, JitTier::Baseline);

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 8,
                instructions: 4,
            }
        );
        assert_eq!(cpu.registers[Pc], 0);

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 7,
            }
        );
        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Trace);
        assert_eq!(cpu.registers[5usize], 1);
        assert_eq!(cpu.registers[6usize], 10);
        assert_eq!(cpu.registers[7usize], 1);
        assert_eq!(cpu.registers[Pc], 8);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_recompiles_hot_blocks_to_the_optimized_tier() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 5, 5, 1))); // addi x5, x5, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 1))); // addi x6, x6, 1
        bin.extend(rv64_word(rv64_beq(0, 0, 8))); // beq x0, x0, +8

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            trace_compilation: false,
            hot_threshold: 1,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        let baseline_len = engine.cache.get(&0).unwrap().code_len;
        assert_eq!(engine.cache.get(&0).unwrap().tier, JitTier::Baseline);
        assert_eq!(cpu.registers[5usize], 1);
        assert_eq!(cpu.registers[6usize], 1);
        assert_eq!(cpu.registers[Pc], 16);

        cpu.registers[Pc] = 0;
        engine.step(&mut cpu).unwrap();
        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        assert_eq!(block.execution_count, 2);
        assert_eq!(block.instruction_count, 3);
        assert!(block.code_len < baseline_len);
        assert_eq!(cpu.registers[5usize], 2);
        assert_eq!(cpu.registers[6usize], 2);
        assert_eq!(cpu.registers[Pc], 16);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_regalloc_loads_only_live_in_guest_registers() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 7, 28, 5, 1))); // andi x28, x5, 1
        bin.extend(rv64_word(rv64_beq(28, 0, 8))); // beq x28, x0, +8
        bin.extend(rv64_word(rv64_i(0x13, 0, 6, 0, 1))); // addi x6, x0, 1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[5usize] = 1;

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            trace_compilation: false,
            hot_threshold: 1,
            tier_budgeting: false,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        cpu.registers[Pc] = 0;
        engine.step(&mut cpu).unwrap();

        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        let listing = block
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("load guest x5"));
        assert!(!listing.contains("load guest x28"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_regalloc_handles_wide_basic_blocks() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_r(0x33, 0, 0, 13, 1, 2))); // add x13, x1, x2
        bin.extend(rv64_word(rv64_r(0x33, 0, 0, 14, 3, 4))); // add x14, x3, x4
        bin.extend(rv64_word(rv64_r(0x33, 0, 0, 15, 5, 6))); // add x15, x5, x6
        bin.extend(rv64_word(rv64_r(0x33, 0, 0, 16, 7, 8))); // add x16, x7, x8

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();
        engine
            .compile_block(&mut cpu, 0, JitTier::Optimized, "test", 0)
            .unwrap();

        let block = engine.cache.get(&0).unwrap();
        let listing = block
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("; regalloc"));
        assert!(!listing.contains("x29, x30"));
        assert!(!listing.contains("load guest x13"));
        assert!(listing.contains("store guest x16"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_delays_non_loop_optimized_tier_until_budget_threshold() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 1, 1))); // addi x1, x1, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 2, 2, 1))); // addi x2, x2, 1
        bin.extend(rv64_word(rv64_i(0x13, 0, 3, 3, 1))); // addi x3, x3, 1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            hot_threshold: 1,
            non_loop_hot_threshold_multiplier: 3,
            min_optimized_block_instructions: 4,
            ..JitOptions::default()
        })
        .unwrap();

        for _ in 0..3 {
            engine.step(&mut cpu).unwrap();
            cpu.registers[Pc] = 0;
        }

        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Baseline);
        assert_eq!(block.execution_count, 3);

        engine.step(&mut cpu).unwrap();
        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        assert_eq!(block.execution_count, 4);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_background_compilation_queues_hot_blocks_without_blocking() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0000_8093)); // addi x1, x1, 0
        bin.extend(rv64_word(rv64_beq(0, 0, 0x1ffc))); // beq x0, x0, -4

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            hot_threshold: 1,
            background_compilation: true,
            compiler_threads: 1,
            compile_queue_limit: 1,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(engine.cache.get(&0).unwrap().tier, JitTier::Baseline);

        engine.step(&mut cpu).unwrap();
        assert_eq!(engine.cache.get(&0).unwrap().tier, JitTier::Baseline);
        assert_eq!(engine.pending_optimized_compiles.len(), 1);

        engine
            .queue_background_optimization(&mut cpu, 0, JitTier::Optimized, "hot-loop", 2)
            .unwrap();
        assert_eq!(engine.pending_optimized_compiles.len(), 1);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_adopts_background_optimized_blocks() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0000_8093)); // addi x1, x1, 0
        bin.extend(rv64_word(rv64_beq(0, 0, 0x1ffc))); // beq x0, x0, -4

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            hot_threshold: 1,
            background_compilation: true,
            compiler_threads: 1,
            compile_queue_limit: 4,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        engine.step(&mut cpu).unwrap();

        for _ in 0..100 {
            engine.drain_background_compiler(&mut cpu).unwrap();
            if engine.cache.get(&0).unwrap().tier == JitTier::Optimized {
                assert!(engine.pending_optimized_compiles.is_empty());
                assert!(engine.cache.get(&0).unwrap().execution_count >= 2);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        panic!("background optimized block was not adopted");
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_runs_mutating_self_loop_inside_native_block() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 1, -1))); // addi x1, x1, -1
        bin.extend(rv64_word(rv64_bne(1, 0, 0x1ffc))); // bne x1, x0, -4

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = 3;

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            hot_threshold: 1,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 2
            }
        );
        assert_eq!(cpu.registers[1usize], 2);
        assert_eq!(cpu.registers[Pc], 0);

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 4
            }
        );
        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        assert_eq!(cpu.registers[1usize], 0);
        assert_eq!(cpu.registers[Pc], 8);

        let listing = block
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("mov x10, x2 ; initial loop counter x1"));
        assert!(listing.contains("subs x2, x2, #1 ; regalloc final loop decrement"));
        assert!(listing.contains("lsl x0, x10, #1 ; return executed instructions"));
        assert!(listing.contains("ldr x2, [x1, #8] ; load guest x1"));
        assert!(!listing.contains("cmp x2, x31"));
        assert!(!listing.contains("selective loop save"));
        assert!(!listing.contains("stp x19, x20"));
        assert!(!listing.contains("stp x21"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn optimized_jit_uses_flag_setting_word_decrement_for_addiw_loop() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x1b, 0, 5, 5, -1))); // addiw x5, x5, -1
        bin.extend(rv64_word(rv64_bne(5, 0, 0x1ffc))); // bne x5, x0, -4

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[5usize] = 3;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 6
            }
        );
        assert_eq!(cpu.registers[5usize], 0);
        assert_eq!(cpu.registers[Pc], 8);

        let listing = engine.cache[&0]
            .native_listing
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(listing.contains("subs w2, w2, #1 ; regalloc final loop decrement"));
        assert!(listing.contains("sxtw x2, w2"));
        assert!(!listing.contains("cmp x2, x31"));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_promotes_hot_blocks_when_optimizer_has_no_ir_changes() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0020_81b3)); // add x3, x1, x2
        bin.extend(rv64_word(rv64_bne(1, 0, 0x1ffc))); // bne x1, x0, -4

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = 1;
        cpu.registers[2usize] = 2;

        let mut engine = JitEngine::with_options(JitOptions {
            dynamic_recompilation: true,
            hot_threshold: 1,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        let baseline_len = engine.cache.get(&0).unwrap().code_len;

        engine.step(&mut cpu).unwrap();
        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        assert_eq!(block.execution_count, 2);
        assert!(block.code_len <= baseline_len);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_executes_jal_instruction() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0080_00ef)); // jal x1, +8
        bin.extend(rv64_word(0x0010_0093)); // skipped
        bin.extend(rv64_word(0x0020_0093)); // target

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::new().unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[1usize], 4);
        assert_eq!(cpu.registers[Pc], 8);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn aot_mode_reports_guest_traps_without_interpreter_fallback() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0050_0093)); // addi x1, x0, 5
        bin.extend(rv64_word(0x0010_0073)); // ebreak

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            ..JitOptions::default()
        })
        .unwrap();

        assert!(matches!(
            engine.step(&mut cpu),
            Err(JitError::RuntimeFault { pc: 0, reason }) if reason.contains("ebreak")
        ));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn aot_mode_lazily_compiles_native_runtime_targets_without_interpreter() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0000_8067)); // jalr x0, x1, 0
        bin.extend(rv64_word(0x0000_0013)); // addi x0, x0, 0
        bin.extend(rv64_word(0x0010_0113)); // addi x2, x0, 1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = 8;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[Pc], 8);
        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 8,
                instructions: 1
            }
        );
        assert_eq!(cpu.registers[2usize], 1);
        assert_eq!(engine.cache.get(&8).unwrap().tier, JitTier::Baseline);
        assert!(!engine
            .cache
            .get(&8)
            .unwrap()
            .profile_instructions
            .iter()
            .any(|instruction| instruction.text.contains("interp pc=")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn default_aot_precompiles_entry_and_lazily_compiles_reachable_misses() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_beq(0, 0, 8))); // beq x0, x0, +8
        bin.extend(rv64_word(0x0000_0013)); // addi x0, x0, 0
        bin.extend(rv64_word(0x0010_0113)); // addi x2, x0, 1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert!(engine.cache.contains_key(&0));
        assert!(!engine.cache.contains_key(&8));
        assert_eq!(cpu.registers[Pc], 8);

        let step = engine.step(&mut cpu).unwrap();
        assert_eq!(
            step,
            JitStep::Native {
                pc: 8,
                instructions: 1,
            }
        );
        assert_eq!(cpu.registers[2usize], 1);
        assert_eq!(engine.cache.get(&8).unwrap().tier, JitTier::Baseline);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn default_aot_precompiles_trace_loop_entries() {
        let mut cpu = RV64GC::new();
        cpu.load_bin(trace_memory_branch_loop_bin());
        cpu.ram
            .add_region(MemoryRegion::new(0x1000, 24, vec![0; 24]))
            .unwrap();
        cpu.ram.write_doubleword(0x1000, 1).unwrap();
        cpu.ram.write_doubleword(0x1008, 1).unwrap();
        cpu.ram.write_doubleword(0x1010, 1).unwrap();
        cpu.registers[5usize] = 3;
        cpu.registers[10usize] = 0x1000;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 0,
                instructions: 15,
            }
        );
        assert_eq!(cpu.registers[6usize], 3);
        assert_eq!(cpu.registers[10usize], 0x1018);
        assert_eq!(cpu.registers[Pc], 20);

        let block = engine.cache.get(&0).unwrap();
        assert_eq!(block.tier, JitTier::Trace);
        assert!(block
            .native_listing
            .iter()
            .any(|emission| emission.text.contains("jit_runtime_try_direct_read_ptr")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn aot_miss_still_optimizes_compiler_region_targets() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0000_8067)); // jalr x0, x1, 0
        bin.extend(rv64_word(0x0000_0013)); // addi x0, x0, 0
        bin.extend(fibonacci_recurrence_loop_bin());

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = 8;
        cpu.registers[11usize] = 1;
        cpu.registers[12usize] = 5;
        cpu.registers[13usize] = 0;
        cpu.registers[14usize] = 1;
        cpu.registers[15usize] = 1;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[Pc], 8);
        assert!(!engine.cache.contains_key(&8));

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 8,
                instructions: 30,
            }
        );
        assert_eq!(cpu.registers[11usize], 8);
        assert_eq!(cpu.registers[12usize], 0);
        assert_eq!(cpu.registers[13usize], 1);
        assert_eq!(cpu.registers[14usize], 8);
        assert_eq!(cpu.registers[15usize], 13);
        assert_eq!(cpu.registers[Pc], 32);

        let block = engine.cache.get(&8).unwrap();
        assert_eq!(block.tier, JitTier::Optimized);
        assert!(block
            .native_listing
            .iter()
            .any(|emission| emission.text.contains("fib executed instructions")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn aot_miss_optimizes_generic_self_loop_targets() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0001_0067)); // jalr x0, x2, 0
        bin.extend(rv64_word(0x0000_0013)); // addi x0, x0, 0
        bin.extend(rv64_word(rv64_i(0x13, 0, 1, 1, -1))); // addi x1, x1, -1
        bin.extend(rv64_word(rv64_bne(1, 0, 0x1ffc))); // bne x1, x0, loop

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = 3;
        cpu.registers[2usize] = 8;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[Pc], 8);
        assert!(!engine.cache.contains_key(&8));

        assert_eq!(
            engine.step(&mut cpu).unwrap(),
            JitStep::Native {
                pc: 8,
                instructions: 6,
            }
        );
        assert_eq!(cpu.registers[1usize], 0);
        assert_eq!(cpu.registers[Pc], 16);
        assert_eq!(engine.cache.get(&8).unwrap().tier, JitTier::Optimized);
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn aot_runtime_misses_use_trace_tier_for_branch_paths() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0000_8067)); // jalr x0, x1, 0
        bin.extend(rv64_word(0x0000_0013)); // addi x0, x0, 0
        bin.extend(trace_diamond_loop_bin());

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = 8;
        cpu.registers[5usize] = 3;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            trace_compilation: true,
            dump_instructions: true,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[Pc], 8);
        assert!(!engine.cache.contains_key(&8));

        engine.step(&mut cpu).unwrap();

        let block = engine.cache.get(&8).unwrap();
        assert_eq!(block.tier, JitTier::Trace);
        assert!(block
            .native_listing
            .iter()
            .any(|emission| emission.text.contains("trace guard")));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn strict_aot_mode_rejects_runtime_targets_that_were_not_precompiled() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(0x0000_8067)); // jalr x0, x1, 0
        bin.extend(rv64_word(0x0000_0013)); // addi x0, x0, 0
        bin.extend(rv64_word(0x0010_0113)); // addi x2, x0, 1

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);
        cpu.registers[1usize] = 8;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Aot,
            aot_compile_misses: false,
            ..JitOptions::default()
        })
        .unwrap();

        engine.step(&mut cpu).unwrap();
        assert_eq!(cpu.registers[Pc], 8);
        assert!(matches!(
            engine.step(&mut cpu),
            Err(JitError::AotMiss { pc: 8 })
        ));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_reports_block_planning_failures_without_panicking() {
        let mut cpu = RV64GC::new();
        cpu.registers[Pc] = 0x1000;

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        assert!(matches!(
            engine.step(&mut cpu),
            Err(JitError::BlockPlanningFailed { pc: 0x1000, .. })
        ));
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn jit_reports_runtime_memory_faults_without_aborting() {
        let mut bin = Vec::new();
        bin.extend(rv64_word(rv64_i(0x13, 0, 2, 0, 32))); // addi x2, x0, 32
        bin.extend(rv64_word(rv64_i(0x03, 3, 1, 2, 0))); // ld x1, 0(x2)

        let mut cpu = RV64GC::new();
        cpu.load_bin(bin);

        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: JitExecutionMode::Jit,
            ..JitOptions::default()
        })
        .unwrap();

        assert!(matches!(
            engine.step(&mut cpu),
            Err(JitError::RuntimeFault { pc: 0, .. })
        ));
    }
}
