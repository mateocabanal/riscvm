use std::ffi::c_void;
use std::ptr;

use crate::cpu::RV64GC;

use super::{
    jit_runtime_atomic, jit_runtime_binary, jit_runtime_csr, jit_runtime_direct_write_ptr,
    jit_runtime_ecall, jit_runtime_float_load, jit_runtime_float_op, jit_runtime_float_store,
    jit_runtime_load_i16, jit_runtime_load_i32, jit_runtime_load_i8, jit_runtime_load_u16,
    jit_runtime_load_u32, jit_runtime_load_u64, jit_runtime_load_u8, jit_runtime_store_u16,
    jit_runtime_store_u32, jit_runtime_store_u64, jit_runtime_store_u8, jit_runtime_trap,
    ArithmeticXorToggleLoop, BlockOperationKind, BlockPlan, CompiledBlock, CountedDiamondLoop,
    FibonacciRecurrenceLoop, IntegerBranchCondition, JitError, JitTier, NativeEmission,
    NativeInstruction, RuntimeCsrOp, StoreLoadForwardLoop,
};

const CPU_PTR: u8 = 19;
const REG_PTR: u8 = 20;
const SCRATCH0: u8 = 9;
const SCRATCH1: u8 = 10;
const SCRATCH2: u8 = 11;
const SCRATCH3: u8 = 12;
const CALL_TARGET: u8 = 16;
const DYNAMIC_INSTRUCTION_COUNT: u8 = 21;
const PC_REGISTER: u8 = 32;
const A64_ZERO_REGISTER: u8 = 31;
const LOOP_TEMP: u8 = 16;
const BLOCK_HOST_REGISTERS: [u8; 8] = [9, 10, 11, 12, 13, 14, 15, 17];
const LOOP_HOST_REGISTERS: [u8; 7] = [22, 23, 24, 25, 26, 27, 28];
const UNMAPPED_GUEST_REGISTER: u8 = u8::MAX;

type JitFn = unsafe extern "C" fn(*mut RV64GC, *mut u64) -> u64;

fn jit_load_runtime(width: u64, signed: bool) -> (u64, &'static str) {
    match (width, signed) {
        (1, false) => (
            jit_runtime_load_u8 as *const () as usize as u64,
            "jit_runtime_load_u8",
        ),
        (1, true) => (
            jit_runtime_load_i8 as *const () as usize as u64,
            "jit_runtime_load_i8",
        ),
        (2, false) => (
            jit_runtime_load_u16 as *const () as usize as u64,
            "jit_runtime_load_u16",
        ),
        (2, true) => (
            jit_runtime_load_i16 as *const () as usize as u64,
            "jit_runtime_load_i16",
        ),
        (4, false) => (
            jit_runtime_load_u32 as *const () as usize as u64,
            "jit_runtime_load_u32",
        ),
        (4, true) => (
            jit_runtime_load_i32 as *const () as usize as u64,
            "jit_runtime_load_i32",
        ),
        (8, _) => (
            jit_runtime_load_u64 as *const () as usize as u64,
            "jit_runtime_load_u64",
        ),
        _ => {
            debug_assert!(
                matches!(width, 1 | 2 | 4 | 8),
                "invalid JIT load width: {width}"
            );
            (
                jit_runtime_load_u64 as *const () as usize as u64,
                "jit_runtime_load_u64",
            )
        }
    }
}

fn jit_store_runtime(width: u64) -> (u64, &'static str) {
    match width {
        1 => (
            jit_runtime_store_u8 as *const () as usize as u64,
            "jit_runtime_store_u8",
        ),
        2 => (
            jit_runtime_store_u16 as *const () as usize as u64,
            "jit_runtime_store_u16",
        ),
        4 => (
            jit_runtime_store_u32 as *const () as usize as u64,
            "jit_runtime_store_u32",
        ),
        8 => (
            jit_runtime_store_u64 as *const () as usize as u64,
            "jit_runtime_store_u64",
        ),
        _ => {
            debug_assert!(
                matches!(width, 1 | 2 | 4 | 8),
                "invalid JIT store width: {width}"
            );
            (
                jit_runtime_store_u64 as *const () as usize as u64,
                "jit_runtime_store_u64",
            )
        }
    }
}

pub(crate) struct AArch64Backend;

impl AArch64Backend {
    pub fn new() -> Self {
        Self
    }

    pub fn compile(
        &mut self,
        plan: &BlockPlan,
        tier: JitTier,
        include_listing: bool,
    ) -> Result<CompiledBlock, JitError> {
        if matches!(tier, JitTier::Optimized | JitTier::Trace) {
            if let Some(region) = arithmetic_xor_toggle_loop_region(plan) {
                let mut code = A64Emitter::new(include_listing);
                code.emit_arithmetic_xor_toggle_loop(region);
                let code_bytes = code.finish();
                let code_len = code_bytes.bytes.len();
                let native = NativeBlock::new(code_bytes.bytes)?;
                return Ok(CompiledBlock {
                    native,
                    fingerprint: plan.fingerprint.clone(),
                    code_version: plan.code_version,
                    instruction_count: plan.guest_instruction_count,
                    code_len,
                    native_listing: code_bytes.listing,
                    profile_instructions: plan.profile_instructions(),
                    tier: JitTier::Baseline,
                    execution_count: 0,
                    next_optimized_attempt_count: 0,
                    ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                    ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
                });
            }

            if let Some(region) = counted_diamond_loop_region(plan) {
                let mut code = A64Emitter::new(include_listing);
                code.emit_counted_diamond_loop(region);
                let code_bytes = code.finish();
                let code_len = code_bytes.bytes.len();
                let native = NativeBlock::new(code_bytes.bytes)?;
                return Ok(CompiledBlock {
                    native,
                    fingerprint: plan.fingerprint.clone(),
                    code_version: plan.code_version,
                    instruction_count: plan.guest_instruction_count,
                    code_len,
                    native_listing: code_bytes.listing,
                    profile_instructions: plan.profile_instructions(),
                    tier: JitTier::Baseline,
                    execution_count: 0,
                    next_optimized_attempt_count: 0,
                    ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                    ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
                });
            }

            if let Some(region) = fibonacci_recurrence_loop_region(plan) {
                let mut code = A64Emitter::new(include_listing);
                code.emit_fibonacci_recurrence_loop(region);
                let code_bytes = code.finish();
                let code_len = code_bytes.bytes.len();
                let native = NativeBlock::new(code_bytes.bytes)?;
                return Ok(CompiledBlock {
                    native,
                    fingerprint: plan.fingerprint.clone(),
                    code_version: plan.code_version,
                    instruction_count: plan.guest_instruction_count,
                    code_len,
                    native_listing: code_bytes.listing,
                    profile_instructions: plan.profile_instructions(),
                    tier: JitTier::Baseline,
                    execution_count: 0,
                    next_optimized_attempt_count: 0,
                    ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                    ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
                });
            }

            if let Some(region) = store_load_forward_loop_region(plan) {
                let mut code = A64Emitter::new(include_listing);
                code.emit_store_load_forward_loop(region);
                let code_bytes = code.finish();
                let code_len = code_bytes.bytes.len();
                let native = NativeBlock::new(code_bytes.bytes)?;
                return Ok(CompiledBlock {
                    native,
                    fingerprint: plan.fingerprint.clone(),
                    code_version: plan.code_version,
                    instruction_count: plan.guest_instruction_count,
                    code_len,
                    native_listing: code_bytes.listing,
                    profile_instructions: plan.profile_instructions(),
                    tier: JitTier::Baseline,
                    execution_count: 0,
                    next_optimized_attempt_count: 0,
                    ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                    ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
                });
            }
        }

        if tier == JitTier::Trace {
            let mut code = A64Emitter::new(include_listing);
            if let Some(trace_loop) = RegisterAllocatedTraceLoop::from_plan(plan) {
                code.emit_register_allocated_trace_loop(&trace_loop);
                let code_bytes = code.finish();
                let code_len = code_bytes.bytes.len();
                let native = NativeBlock::new(code_bytes.bytes)?;
                return Ok(CompiledBlock {
                    native,
                    fingerprint: plan.fingerprint.clone(),
                    code_version: plan.code_version,
                    instruction_count: plan.guest_instruction_count,
                    code_len,
                    native_listing: code_bytes.listing,
                    profile_instructions: plan.profile_instructions(),
                    tier: JitTier::Baseline,
                    execution_count: 0,
                    next_optimized_attempt_count: 0,
                    ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                    ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
                });
            }
            code.emit_trace_plan(plan);
            let code_bytes = code.finish();
            let code_len = code_bytes.bytes.len();
            let native = NativeBlock::new(code_bytes.bytes)?;
            return Ok(CompiledBlock {
                native,
                fingerprint: plan.fingerprint.clone(),
                code_version: plan.code_version,
                instruction_count: plan.guest_instruction_count,
                code_len,
                native_listing: code_bytes.listing,
                profile_instructions: plan.profile_instructions(),
                tier: JitTier::Baseline,
                execution_count: 0,
                next_optimized_attempt_count: 0,
                ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
            });
        }

        let self_loop_branch = if tier == JitTier::Optimized {
            optimized_self_loop_branch(plan)
        } else {
            None
        };
        let mut code = A64Emitter::new(include_listing);

        if let Some(loop_plan) = RegisterAllocatedLoop::from_plan(plan, self_loop_branch) {
            code.emit_register_allocated_self_loop(&loop_plan);
            let code_bytes = code.finish();
            let code_len = code_bytes.bytes.len();
            let native = NativeBlock::new(code_bytes.bytes)?;
            return Ok(CompiledBlock {
                native,
                fingerprint: plan.fingerprint.clone(),
                code_version: plan.code_version,
                instruction_count: plan.guest_instruction_count,
                code_len,
                native_listing: code_bytes.listing,
                profile_instructions: plan.profile_instructions(),
                tier: JitTier::Baseline,
                execution_count: 0,
                next_optimized_attempt_count: 0,
                ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
            });
        }

        if tier == JitTier::Optimized {
            if let Some(block_plan) = RegisterAllocatedBlock::from_plan(plan) {
                code.emit_register_allocated_block(&block_plan);
                let code_bytes = code.finish();
                let code_len = code_bytes.bytes.len();
                let native = NativeBlock::new(code_bytes.bytes)?;
                return Ok(CompiledBlock {
                    native,
                    fingerprint: plan.fingerprint.clone(),
                    code_version: plan.code_version,
                    instruction_count: plan.guest_instruction_count,
                    code_len,
                    native_listing: code_bytes.listing,
                    profile_instructions: plan.profile_instructions(),
                    tier: JitTier::Baseline,
                    execution_count: 0,
                    next_optimized_attempt_count: 0,
                    ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
                    ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
                });
            }
        }

        code.emit_prologue(self_loop_branch.is_some());

        if let Some(branch) = self_loop_branch {
            code.emit(
                mov_reg(DYNAMIC_INSTRUCTION_COUNT, A64_ZERO_REGISTER),
                format!("mov x{DYNAMIC_INSTRUCTION_COUNT}, xzr ; dynamic instruction count"),
            );
            let loop_start = code.current_offset();
            for operation in &plan.operations[..plan.operations.len() - 1] {
                match operation.kind() {
                    BlockOperationKind::Native(instruction) => code.emit_instruction(instruction),
                }
            }
            code.emit_counted_self_loop_branch(
                branch,
                loop_start,
                plan.guest_instruction_count as u64,
            );
            code.emit(
                mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
                format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return executed instructions"),
            );
        } else {
            for operation in &plan.operations {
                match operation.kind() {
                    BlockOperationKind::Native(instruction) => code.emit_instruction(instruction),
                }
            }
            if !matches!(
                plan.operations.last().map(|operation| operation.kind()),
                Some(BlockOperationKind::Native(instruction)) if instruction.terminates_block()
            ) {
                code.store_pc(plan.end_pc);
            }
            code.mov_imm64(0, plan.guest_instruction_count as u64);
        }
        code.emit_epilogue(self_loop_branch.is_some());
        code.ret();

        let code_bytes = code.finish();
        let code_len = code_bytes.bytes.len();
        let native = NativeBlock::new(code_bytes.bytes)?;
        Ok(CompiledBlock {
            native,
            fingerprint: plan.fingerprint.clone(),
            code_version: plan.code_version,
            instruction_count: plan.guest_instruction_count,
            code_len,
            native_listing: code_bytes.listing,
            profile_instructions: plan.profile_instructions(),
            tier: JitTier::Baseline,
            execution_count: 0,
            next_optimized_attempt_count: 0,
            ends_with_self_loop_branch: plan.ends_with_self_loop_branch(),
            ends_with_loop_back_edge: plan.ends_with_loop_back_edge(),
        })
    }
}

pub(crate) struct NativeBlock {
    memory: ExecutableMemory,
    entry: JitFn,
}

// Native blocks own immutable executable mappings after construction. Moving the
// ownership handle between compiler and execution threads does not share mutable
// guest state; execution still requires `&mut RV64GC`.
unsafe impl Send for NativeBlock {}

impl NativeBlock {
    fn new(code: Vec<u8>) -> Result<Self, JitError> {
        let memory = ExecutableMemory::new(&code)?;
        let entry = unsafe { std::mem::transmute::<*mut u8, JitFn>(memory.ptr) };

        Ok(Self { memory, entry })
    }

    pub fn execute(&self, cpu: &mut RV64GC) -> u64 {
        let _keep_memory_alive = &self.memory;
        let cpu_ptr = cpu as *mut RV64GC;
        let registers = cpu.registers.as_mut_ptr();
        unsafe { (self.entry)(cpu_ptr, registers) }
    }
}

#[derive(Debug, Clone, Copy)]
struct SelfLoopBranch {
    rs1: u8,
    rs2: u8,
    fallthrough: u64,
    condition: A64Cond,
}

fn counted_diamond_loop_region(plan: &BlockPlan) -> Option<CountedDiamondLoop> {
    let [operation] = plan.operations.as_slice() else {
        return None;
    };
    let BlockOperationKind::Native(NativeInstruction::CountedDiamondLoop(region)) =
        operation.kind()
    else {
        return None;
    };
    Some(region)
}

fn arithmetic_xor_toggle_loop_region(plan: &BlockPlan) -> Option<ArithmeticXorToggleLoop> {
    let [operation] = plan.operations.as_slice() else {
        return None;
    };
    let BlockOperationKind::Native(NativeInstruction::ArithmeticXorToggleLoop(region)) =
        operation.kind()
    else {
        return None;
    };
    Some(region)
}

fn fibonacci_recurrence_loop_region(plan: &BlockPlan) -> Option<FibonacciRecurrenceLoop> {
    let [operation] = plan.operations.as_slice() else {
        return None;
    };
    let BlockOperationKind::Native(NativeInstruction::FibonacciRecurrenceLoop(region)) =
        operation.kind()
    else {
        return None;
    };
    Some(region)
}

fn store_load_forward_loop_region(plan: &BlockPlan) -> Option<StoreLoadForwardLoop> {
    let [operation] = plan.operations.as_slice() else {
        return None;
    };
    let BlockOperationKind::Native(NativeInstruction::StoreLoadForwardLoop(region)) =
        operation.kind()
    else {
        return None;
    };
    Some(region)
}

fn trace_plan_has_native_loop(plan: &BlockPlan) -> bool {
    matches!(
        plan.operations.last().map(|operation| operation.kind()),
        Some(BlockOperationKind::Native(
            NativeInstruction::TraceLoopGuard { .. }
        ))
    )
}

fn optimized_self_loop_branch(plan: &BlockPlan) -> Option<SelfLoopBranch> {
    let Some(last_operation) = plan.operations.last() else {
        return None;
    };

    let BlockOperationKind::Native(instruction) = last_operation.kind();
    let branch = match instruction {
        NativeInstruction::Beq {
            rs1,
            rs2,
            target,
            fallthrough,
        } if target == plan.start_pc => Some(SelfLoopBranch {
            rs1,
            rs2,
            fallthrough,
            condition: A64Cond::Eq,
        }),
        NativeInstruction::Bge {
            rs1,
            rs2,
            target,
            fallthrough,
        } if target == plan.start_pc => Some(SelfLoopBranch {
            rs1,
            rs2,
            fallthrough,
            condition: A64Cond::Ge,
        }),
        NativeInstruction::Bgeu {
            rs1,
            rs2,
            target,
            fallthrough,
        } if target == plan.start_pc => Some(SelfLoopBranch {
            rs1,
            rs2,
            fallthrough,
            condition: A64Cond::Hs,
        }),
        NativeInstruction::Blt {
            rs1,
            rs2,
            target,
            fallthrough,
        } if target == plan.start_pc => Some(SelfLoopBranch {
            rs1,
            rs2,
            fallthrough,
            condition: A64Cond::Lt,
        }),
        NativeInstruction::Bltu {
            rs1,
            rs2,
            target,
            fallthrough,
        } if target == plan.start_pc => Some(SelfLoopBranch {
            rs1,
            rs2,
            fallthrough,
            condition: A64Cond::Lo,
        }),
        NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } if target == plan.start_pc => Some(SelfLoopBranch {
            rs1,
            rs2,
            fallthrough,
            condition: A64Cond::Ne,
        }),
        _ => None,
    }?;

    plan.operations[..plan.operations.len() - 1]
        .iter()
        .any(|operation| {
            let BlockOperationKind::Native(instruction) = operation.kind();
            writes_integer_register(instruction, branch.rs1)
                || writes_integer_register(instruction, branch.rs2)
        })
        .then_some(branch)
}

fn writes_integer_register(instruction: NativeInstruction, register: u8) -> bool {
    if register == 0 {
        return false;
    }

    match instruction {
        NativeInstruction::Add { rd, .. }
        | NativeInstruction::Addi { rd, .. }
        | NativeInstruction::Addiw { rd, .. }
        | NativeInstruction::Addw { rd, .. }
        | NativeInstruction::And { rd, .. }
        | NativeInstruction::AndBranch { rd, .. }
        | NativeInstruction::Andi { rd, .. }
        | NativeInstruction::Auipc { rd, .. }
        | NativeInstruction::Jal { rd, .. }
        | NativeInstruction::Jalr { rd, .. }
        | NativeInstruction::Load { rd, .. }
        | NativeInstruction::LoadImmediate { rd, .. }
        | NativeInstruction::Lui { rd, .. }
        | NativeInstruction::Move { rd, .. }
        | NativeInstruction::Mul { rd, .. }
        | NativeInstruction::Mulh { rd, .. }
        | NativeInstruction::Mulhu { rd, .. }
        | NativeInstruction::Mulw { rd, .. }
        | NativeInstruction::Or { rd, .. }
        | NativeInstruction::Ori { rd, .. }
        | NativeInstruction::RuntimeAtomic { rd, .. }
        | NativeInstruction::RuntimeBinary { rd, .. }
        | NativeInstruction::RuntimeCsr { rd, .. }
        | NativeInstruction::Sll { rd, .. }
        | NativeInstruction::Slli { rd, .. }
        | NativeInstruction::Slliw { rd, .. }
        | NativeInstruction::Sllw { rd, .. }
        | NativeInstruction::Slt { rd, .. }
        | NativeInstruction::Slti { rd, .. }
        | NativeInstruction::Sltiu { rd, .. }
        | NativeInstruction::Sltu { rd, .. }
        | NativeInstruction::Sra { rd, .. }
        | NativeInstruction::Srai { rd, .. }
        | NativeInstruction::Sraiw { rd, .. }
        | NativeInstruction::Sraw { rd, .. }
        | NativeInstruction::Srl { rd, .. }
        | NativeInstruction::Srli { rd, .. }
        | NativeInstruction::Srliw { rd, .. }
        | NativeInstruction::Srlw { rd, .. }
        | NativeInstruction::Sub { rd, .. }
        | NativeInstruction::Subw { rd, .. }
        | NativeInstruction::Xor { rd, .. }
        | NativeInstruction::Xori { rd, .. } => rd == register,
        NativeInstruction::Beq { .. }
        | NativeInstruction::Bge { .. }
        | NativeInstruction::Bgeu { .. }
        | NativeInstruction::Blt { .. }
        | NativeInstruction::Bltu { .. }
        | NativeInstruction::Bne { .. }
        | NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::Ecall { .. }
        | NativeInstruction::FibonacciRecurrenceLoop(_)
        | NativeInstruction::FloatLoad { .. }
        | NativeInstruction::FloatStore { .. }
        | NativeInstruction::InlinedJump { .. }
        | NativeInstruction::Jump { .. }
        | NativeInstruction::JumpReg { .. }
        | NativeInstruction::JumpRegLink { .. }
        | NativeInstruction::Nop
        | NativeInstruction::RuntimeFloat { .. }
        | NativeInstruction::RuntimeTrap { .. }
        | NativeInstruction::Store { .. }
        | NativeInstruction::StoreLoadForwardLoop(_)
        | NativeInstruction::TraceGuard { .. }
        | NativeInstruction::TraceLoopGuard { .. } => false,
    }
}

struct RegisterAllocatedLoop<'a> {
    branch: SelfLoopBranch,
    operations: &'a [super::BlockOperation],
    guest_to_host: [u8; 33],
    loaded_registers: Vec<(u8, u8)>,
    dirty_registers: Vec<(u8, u8)>,
    guest_instruction_count: u64,
}

impl<'a> RegisterAllocatedLoop<'a> {
    fn from_plan(plan: &'a BlockPlan, branch: Option<SelfLoopBranch>) -> Option<Self> {
        let branch = branch?;
        let loop_body_len = plan.operations.len().checked_sub(1)?;
        let operations = &plan.operations[..loop_body_len];
        if operations.is_empty() {
            return None;
        }

        let mut guest_registers = Vec::new();
        let mut dirty_registers = Vec::new();
        for operation in operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            collect_loop_instruction_registers(
                instruction,
                &mut guest_registers,
                &mut dirty_registers,
            )?;
        }
        push_guest_register(branch.rs1, &mut guest_registers);
        push_guest_register(branch.rs2, &mut guest_registers);

        if guest_registers.len() > LOOP_HOST_REGISTERS.len() {
            return None;
        }

        let mut guest_to_host = [UNMAPPED_GUEST_REGISTER; 33];
        for (index, guest) in guest_registers.into_iter().enumerate() {
            let host = LOOP_HOST_REGISTERS[index];
            guest_to_host[guest as usize] = host;
        }
        let loaded_registers = initial_load_registers(operations, Some(branch), &guest_to_host);

        dirty_registers.sort_unstable();
        dirty_registers.dedup();
        let dirty_registers = dirty_registers
            .into_iter()
            .filter_map(|guest| {
                let host = guest_to_host[guest as usize];
                (host != UNMAPPED_GUEST_REGISTER).then_some((guest, host))
            })
            .collect();

        Some(Self {
            branch,
            operations,
            guest_to_host,
            loaded_registers,
            dirty_registers,
            guest_instruction_count: plan.guest_instruction_count as u64,
        })
    }

    fn host_or_zero(&self, guest: u8) -> u8 {
        if guest == 0 {
            return A64_ZERO_REGISTER;
        }

        let host = self.guest_to_host[guest as usize];
        debug_assert_ne!(host, UNMAPPED_GUEST_REGISTER);
        host
    }

    fn host_for_write(&self, guest: u8) -> Option<u8> {
        if guest == 0 {
            return None;
        }

        let host = self.guest_to_host[guest as usize];
        debug_assert_ne!(host, UNMAPPED_GUEST_REGISTER);
        Some(host)
    }

    fn decrementing_counter(&self) -> Option<u8> {
        if self.branch.condition != A64Cond::Ne {
            return None;
        }

        let counter = if self.branch.rs2 == 0 {
            self.branch.rs1
        } else if self.branch.rs1 == 0 {
            self.branch.rs2
        } else {
            return None;
        };
        if counter == 0 {
            return None;
        }

        let mut matching_decrements = 0;
        for operation in self.operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            if !writes_integer_register(instruction, counter) {
                continue;
            }

            match instruction {
                NativeInstruction::Addi { rd, rs1, imm }
                | NativeInstruction::Addiw { rd, rs1, imm }
                    if rd == counter && rs1 == counter && imm == -1 =>
                {
                    matching_decrements += 1;
                }
                _ => return None,
            }
        }

        (matching_decrements == 1).then_some(counter)
    }
}

struct RegisterAllocatedTraceLoop<'a> {
    loop_guard: TraceLoopBranch,
    operations: &'a [super::BlockOperation],
    guest_to_host: [u8; 33],
    loaded_registers: Vec<(u8, u8)>,
    dirty_registers: Vec<(u8, u8)>,
}

impl<'a> RegisterAllocatedTraceLoop<'a> {
    fn from_plan(plan: &'a BlockPlan) -> Option<Self> {
        let BlockOperationKind::Native(NativeInstruction::TraceLoopGuard {
            rs1,
            rs2,
            condition,
            continue_on_taken,
            loop_pc,
            side_exit_pc,
            guest_instruction_count,
        }) = plan.operations.last()?.kind()
        else {
            return None;
        };
        if loop_pc != plan.start_pc {
            return None;
        }

        let operations = &plan.operations[..plan.operations.len() - 1];
        if operations.is_empty() {
            return None;
        }

        let mut guest_registers = Vec::new();
        let mut dirty_registers = Vec::new();
        for operation in operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            collect_loop_instruction_registers(
                instruction,
                &mut guest_registers,
                &mut dirty_registers,
            )?;
        }
        push_guest_register(rs1, &mut guest_registers);
        push_guest_register(rs2, &mut guest_registers);

        if guest_registers.len() > LOOP_HOST_REGISTERS.len() {
            return None;
        }

        let mut guest_to_host = [UNMAPPED_GUEST_REGISTER; 33];
        for (index, guest) in guest_registers.into_iter().enumerate() {
            let host = LOOP_HOST_REGISTERS[index];
            guest_to_host[guest as usize] = host;
        }

        dirty_registers.sort_unstable();
        dirty_registers.dedup();
        let dirty_registers: Vec<_> = dirty_registers
            .into_iter()
            .filter_map(|guest| {
                let host = guest_to_host[guest as usize];
                (host != UNMAPPED_GUEST_REGISTER).then_some((guest, host))
            })
            .collect();
        let mut loaded_registers = initial_load_registers(
            operations,
            Some(SelfLoopBranch {
                rs1,
                rs2,
                fallthrough: side_exit_pc,
                condition: trace_continue_condition(condition, continue_on_taken),
            }),
            &guest_to_host,
        );
        for (guest, host) in &dirty_registers {
            if trace_loop_may_side_exit_before_write(operations, *guest) {
                push_loaded_register(&mut loaded_registers, *guest, *host);
            }
        }

        Some(Self {
            loop_guard: TraceLoopBranch {
                rs1,
                rs2,
                side_exit_pc,
                condition: trace_continue_condition(condition, continue_on_taken),
                guest_instruction_count,
            },
            operations,
            guest_to_host,
            loaded_registers,
            dirty_registers,
        })
    }

    fn host_or_zero(&self, guest: u8) -> u8 {
        if guest == 0 {
            return A64_ZERO_REGISTER;
        }

        let host = self.guest_to_host[guest as usize];
        debug_assert_ne!(host, UNMAPPED_GUEST_REGISTER);
        host
    }
}

struct TraceLoopBranch {
    rs1: u8,
    rs2: u8,
    side_exit_pc: u64,
    condition: A64Cond,
    guest_instruction_count: u64,
}

struct RegisterAllocatedBlock<'a> {
    operations: &'a [super::BlockOperation],
    guest_to_host: [u8; 33],
    loaded_registers: Vec<(u8, u8)>,
    dirty_registers: Vec<(u8, u8)>,
    guest_instruction_count: u64,
}

impl<'a> RegisterAllocatedBlock<'a> {
    fn from_plan(plan: &'a BlockPlan) -> Option<Self> {
        if plan.operations.is_empty() {
            return None;
        }

        let mut guest_registers = Vec::new();
        let mut dirty_registers = Vec::new();
        for operation in &plan.operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            collect_block_instruction_registers(
                instruction,
                &mut guest_registers,
                &mut dirty_registers,
            )?;
        }

        if guest_registers.len() > BLOCK_HOST_REGISTERS.len() {
            return None;
        }

        let mut guest_to_host = [UNMAPPED_GUEST_REGISTER; 33];
        for (index, guest) in guest_registers.into_iter().enumerate() {
            let host = BLOCK_HOST_REGISTERS[index];
            guest_to_host[guest as usize] = host;
        }
        let loaded_registers = initial_load_registers(&plan.operations, None, &guest_to_host);

        dirty_registers.sort_unstable();
        dirty_registers.dedup();
        let dirty_registers = dirty_registers
            .into_iter()
            .filter_map(|guest| {
                let host = guest_to_host[guest as usize];
                (host != UNMAPPED_GUEST_REGISTER).then_some((guest, host))
            })
            .collect();

        Some(Self {
            operations: &plan.operations,
            guest_to_host,
            loaded_registers,
            dirty_registers,
            guest_instruction_count: plan.guest_instruction_count as u64,
        })
    }

    fn host_or_zero(&self, guest: u8) -> u8 {
        if guest == 0 {
            return A64_ZERO_REGISTER;
        }

        let host = self.guest_to_host[guest as usize];
        debug_assert_ne!(host, UNMAPPED_GUEST_REGISTER);
        host
    }

    fn host_for_write(&self, guest: u8) -> Option<u8> {
        if guest == 0 {
            return None;
        }

        let host = self.guest_to_host[guest as usize];
        debug_assert_ne!(host, UNMAPPED_GUEST_REGISTER);
        Some(host)
    }
}

fn initial_load_registers(
    operations: &[super::BlockOperation],
    self_loop_branch: Option<SelfLoopBranch>,
    guest_to_host: &[u8; 33],
) -> Vec<(u8, u8)> {
    let mut written = [false; 33];
    let mut loaded = Vec::new();

    for operation in operations {
        let BlockOperationKind::Native(instruction) = operation.kind();
        collect_read_integer_registers(instruction, |guest| {
            push_initial_load_register(guest, &written, &mut loaded);
        });
        collect_written_integer_registers(instruction, |guest| {
            if guest != 0 {
                written[guest as usize] = true;
            }
        });
    }

    if let Some(branch) = self_loop_branch {
        push_initial_load_register(branch.rs1, &written, &mut loaded);
        push_initial_load_register(branch.rs2, &written, &mut loaded);
    }

    loaded
        .into_iter()
        .filter_map(|guest| {
            let host = guest_to_host[guest as usize];
            (host != UNMAPPED_GUEST_REGISTER).then_some((guest, host))
        })
        .collect()
}

fn push_initial_load_register(guest: u8, written: &[bool; 33], loaded: &mut Vec<u8>) {
    if guest != 0 && !written[guest as usize] && !loaded.contains(&guest) {
        loaded.push(guest);
    }
}

fn push_loaded_register(loaded: &mut Vec<(u8, u8)>, guest: u8, host: u8) {
    if guest != 0
        && !loaded
            .iter()
            .any(|(loaded_guest, _)| *loaded_guest == guest)
    {
        loaded.push((guest, host));
    }
}

fn trace_loop_may_side_exit_before_write(operations: &[super::BlockOperation], guest: u8) -> bool {
    if guest == 0 {
        return false;
    }

    for operation in operations {
        let BlockOperationKind::Native(instruction) = operation.kind();
        if matches!(instruction, NativeInstruction::TraceGuard { .. }) {
            return true;
        }

        let mut written = false;
        collect_written_integer_registers(instruction, |written_guest| {
            written |= written_guest == guest;
        });
        if written {
            return false;
        }
    }

    false
}

fn collect_read_integer_registers<F>(instruction: NativeInstruction, mut push: F)
where
    F: FnMut(u8),
{
    match instruction {
        NativeInstruction::Add { rs1, rs2, .. }
        | NativeInstruction::Addw { rs1, rs2, .. }
        | NativeInstruction::And { rs1, rs2, .. }
        | NativeInstruction::Beq { rs1, rs2, .. }
        | NativeInstruction::Bge { rs1, rs2, .. }
        | NativeInstruction::Bgeu { rs1, rs2, .. }
        | NativeInstruction::Blt { rs1, rs2, .. }
        | NativeInstruction::Bltu { rs1, rs2, .. }
        | NativeInstruction::Bne { rs1, rs2, .. }
        | NativeInstruction::Mul { rs1, rs2, .. }
        | NativeInstruction::Mulh { rs1, rs2, .. }
        | NativeInstruction::Mulhu { rs1, rs2, .. }
        | NativeInstruction::Mulw { rs1, rs2, .. }
        | NativeInstruction::Or { rs1, rs2, .. }
        | NativeInstruction::RuntimeAtomic { rs1, rs2, .. }
        | NativeInstruction::RuntimeBinary { rs1, rs2, .. }
        | NativeInstruction::Sll { rs1, rs2, .. }
        | NativeInstruction::Sllw { rs1, rs2, .. }
        | NativeInstruction::Slt { rs1, rs2, .. }
        | NativeInstruction::Sltu { rs1, rs2, .. }
        | NativeInstruction::Sra { rs1, rs2, .. }
        | NativeInstruction::Sraw { rs1, rs2, .. }
        | NativeInstruction::Srl { rs1, rs2, .. }
        | NativeInstruction::Srlw { rs1, rs2, .. }
        | NativeInstruction::Sub { rs1, rs2, .. }
        | NativeInstruction::Subw { rs1, rs2, .. }
        | NativeInstruction::TraceGuard { rs1, rs2, .. }
        | NativeInstruction::TraceLoopGuard { rs1, rs2, .. }
        | NativeInstruction::Xor { rs1, rs2, .. } => {
            push(rs1);
            push(rs2);
        }
        NativeInstruction::Addi { rs1, .. }
        | NativeInstruction::Addiw { rs1, .. }
        | NativeInstruction::AndBranch { rs1, .. }
        | NativeInstruction::Andi { rs1, .. }
        | NativeInstruction::FloatLoad { rs1, .. }
        | NativeInstruction::Jalr { rs1, .. }
        | NativeInstruction::JumpReg { rs1 }
        | NativeInstruction::JumpRegLink { rs1, .. }
        | NativeInstruction::Load { rs1, .. }
        | NativeInstruction::Move { rs: rs1, .. }
        | NativeInstruction::Ori { rs1, .. }
        | NativeInstruction::Slli { rs1, .. }
        | NativeInstruction::Slliw { rs1, .. }
        | NativeInstruction::Slti { rs1, .. }
        | NativeInstruction::Sltiu { rs1, .. }
        | NativeInstruction::Srai { rs1, .. }
        | NativeInstruction::Sraiw { rs1, .. }
        | NativeInstruction::Srli { rs1, .. }
        | NativeInstruction::Srliw { rs1, .. }
        | NativeInstruction::Xori { rs1, .. } => push(rs1),
        NativeInstruction::FloatStore { rs1, .. } => push(rs1),
        NativeInstruction::Store { rs1, rs2, .. } => {
            push(rs1);
            push(rs2);
        }
        NativeInstruction::Auipc { .. }
        | NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::Ecall { .. }
        | NativeInstruction::FibonacciRecurrenceLoop(_)
        | NativeInstruction::InlinedJump { .. }
        | NativeInstruction::Jal { .. }
        | NativeInstruction::Jump { .. }
        | NativeInstruction::LoadImmediate { .. }
        | NativeInstruction::Lui { .. }
        | NativeInstruction::Nop
        | NativeInstruction::RuntimeCsr { .. }
        | NativeInstruction::RuntimeFloat { .. }
        | NativeInstruction::RuntimeTrap { .. }
        | NativeInstruction::StoreLoadForwardLoop(_) => {}
    }
}

fn collect_written_integer_registers<F>(instruction: NativeInstruction, mut push: F)
where
    F: FnMut(u8),
{
    match instruction {
        NativeInstruction::Add { rd, .. }
        | NativeInstruction::Addi { rd, .. }
        | NativeInstruction::Addiw { rd, .. }
        | NativeInstruction::Addw { rd, .. }
        | NativeInstruction::And { rd, .. }
        | NativeInstruction::AndBranch { rd, .. }
        | NativeInstruction::Andi { rd, .. }
        | NativeInstruction::Auipc { rd, .. }
        | NativeInstruction::Jal { rd, .. }
        | NativeInstruction::Jalr { rd, .. }
        | NativeInstruction::Load { rd, .. }
        | NativeInstruction::LoadImmediate { rd, .. }
        | NativeInstruction::Lui { rd, .. }
        | NativeInstruction::Move { rd, .. }
        | NativeInstruction::Mul { rd, .. }
        | NativeInstruction::Mulh { rd, .. }
        | NativeInstruction::Mulhu { rd, .. }
        | NativeInstruction::Mulw { rd, .. }
        | NativeInstruction::Or { rd, .. }
        | NativeInstruction::Ori { rd, .. }
        | NativeInstruction::RuntimeAtomic { rd, .. }
        | NativeInstruction::RuntimeBinary { rd, .. }
        | NativeInstruction::RuntimeCsr { rd, .. }
        | NativeInstruction::Sll { rd, .. }
        | NativeInstruction::Slli { rd, .. }
        | NativeInstruction::Slliw { rd, .. }
        | NativeInstruction::Sllw { rd, .. }
        | NativeInstruction::Slt { rd, .. }
        | NativeInstruction::Slti { rd, .. }
        | NativeInstruction::Sltiu { rd, .. }
        | NativeInstruction::Sltu { rd, .. }
        | NativeInstruction::Sra { rd, .. }
        | NativeInstruction::Srai { rd, .. }
        | NativeInstruction::Sraiw { rd, .. }
        | NativeInstruction::Sraw { rd, .. }
        | NativeInstruction::Srl { rd, .. }
        | NativeInstruction::Srli { rd, .. }
        | NativeInstruction::Srliw { rd, .. }
        | NativeInstruction::Srlw { rd, .. }
        | NativeInstruction::Sub { rd, .. }
        | NativeInstruction::Subw { rd, .. }
        | NativeInstruction::Xor { rd, .. }
        | NativeInstruction::Xori { rd, .. } => push(rd),
        NativeInstruction::JumpRegLink { .. } => push(1),
        NativeInstruction::Beq { .. }
        | NativeInstruction::Bge { .. }
        | NativeInstruction::Bgeu { .. }
        | NativeInstruction::Blt { .. }
        | NativeInstruction::Bltu { .. }
        | NativeInstruction::Bne { .. }
        | NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::Ecall { .. }
        | NativeInstruction::FibonacciRecurrenceLoop(_)
        | NativeInstruction::FloatLoad { .. }
        | NativeInstruction::FloatStore { .. }
        | NativeInstruction::InlinedJump { .. }
        | NativeInstruction::Jump { .. }
        | NativeInstruction::JumpReg { .. }
        | NativeInstruction::Nop
        | NativeInstruction::RuntimeFloat { .. }
        | NativeInstruction::RuntimeTrap { .. }
        | NativeInstruction::Store { .. }
        | NativeInstruction::StoreLoadForwardLoop(_)
        | NativeInstruction::TraceGuard { .. }
        | NativeInstruction::TraceLoopGuard { .. } => {}
    }
}

fn collect_block_instruction_registers(
    instruction: NativeInstruction,
    guest_registers: &mut Vec<u8>,
    dirty_registers: &mut Vec<u8>,
) -> Option<()> {
    match instruction {
        NativeInstruction::Beq { rs1, rs2, .. }
        | NativeInstruction::Bge { rs1, rs2, .. }
        | NativeInstruction::Bgeu { rs1, rs2, .. }
        | NativeInstruction::Blt { rs1, rs2, .. }
        | NativeInstruction::Bltu { rs1, rs2, .. }
        | NativeInstruction::Bne { rs1, rs2, .. } => {
            push_guest_register(rs1, guest_registers);
            push_guest_register(rs2, guest_registers);
            Some(())
        }
        NativeInstruction::AndBranch { rd, rs1, .. } => {
            push_guest_register(rs1, guest_registers);
            push_dirty_register(rd, guest_registers, dirty_registers);
            Some(())
        }
        NativeInstruction::Jal { rd, .. } => {
            push_dirty_register(rd, guest_registers, dirty_registers);
            Some(())
        }
        NativeInstruction::Jalr { rd, rs1, .. } => {
            push_guest_register(rs1, guest_registers);
            push_dirty_register(rd, guest_registers, dirty_registers);
            Some(())
        }
        NativeInstruction::InlinedJump { .. } | NativeInstruction::Jump { .. } => Some(()),
        NativeInstruction::JumpReg { rs1 } => {
            push_guest_register(rs1, guest_registers);
            Some(())
        }
        NativeInstruction::JumpRegLink { rs1, .. } => {
            push_guest_register(rs1, guest_registers);
            push_dirty_register(1, guest_registers, dirty_registers);
            Some(())
        }
        NativeInstruction::TraceGuard { rs1, rs2, .. }
        | NativeInstruction::TraceLoopGuard { rs1, rs2, .. } => {
            push_guest_register(rs1, guest_registers);
            push_guest_register(rs2, guest_registers);
            Some(())
        }
        NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::Load { .. }
        | NativeInstruction::Store { .. }
        | NativeInstruction::StoreLoadForwardLoop(_) => None,
        _ => collect_loop_instruction_registers(instruction, guest_registers, dirty_registers),
    }
}

fn collect_loop_instruction_registers(
    instruction: NativeInstruction,
    guest_registers: &mut Vec<u8>,
    dirty_registers: &mut Vec<u8>,
) -> Option<()> {
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
            push_guest_register(rs1, guest_registers);
            push_guest_register(rs2, guest_registers);
            push_dirty_register(rd, guest_registers, dirty_registers);
        }
        NativeInstruction::Addi { rd, rs1, .. }
        | NativeInstruction::Addiw { rd, rs1, .. }
        | NativeInstruction::AndBranch { rd, rs1, .. }
        | NativeInstruction::Andi { rd, rs1, .. }
        | NativeInstruction::Ori { rd, rs1, .. }
        | NativeInstruction::Slli { rd, rs1, .. }
        | NativeInstruction::Slliw { rd, rs1, .. }
        | NativeInstruction::Slti { rd, rs1, .. }
        | NativeInstruction::Sltiu { rd, rs1, .. }
        | NativeInstruction::Srai { rd, rs1, .. }
        | NativeInstruction::Sraiw { rd, rs1, .. }
        | NativeInstruction::Srli { rd, rs1, .. }
        | NativeInstruction::Srliw { rd, rs1, .. }
        | NativeInstruction::Xori { rd, rs1, .. } => {
            push_guest_register(rs1, guest_registers);
            push_dirty_register(rd, guest_registers, dirty_registers);
        }
        NativeInstruction::Auipc { rd, .. }
        | NativeInstruction::LoadImmediate { rd, .. }
        | NativeInstruction::Lui { rd, .. } => {
            push_dirty_register(rd, guest_registers, dirty_registers);
        }
        NativeInstruction::Move { rd, rs } => {
            push_guest_register(rs, guest_registers);
            push_dirty_register(rd, guest_registers, dirty_registers);
        }
        NativeInstruction::Load { rd, rs1, .. } => {
            push_guest_register(rs1, guest_registers);
            push_dirty_register(rd, guest_registers, dirty_registers);
        }
        NativeInstruction::Store { rs1, rs2, .. } => {
            push_guest_register(rs1, guest_registers);
            push_guest_register(rs2, guest_registers);
        }
        NativeInstruction::TraceGuard { rs1, rs2, .. }
        | NativeInstruction::TraceLoopGuard { rs1, rs2, .. } => {
            push_guest_register(rs1, guest_registers);
            push_guest_register(rs2, guest_registers);
        }
        NativeInstruction::InlinedJump { .. } | NativeInstruction::Nop => {}
        _ => return None,
    }

    Some(())
}

fn push_dirty_register(guest: u8, guest_registers: &mut Vec<u8>, dirty_registers: &mut Vec<u8>) {
    push_guest_register(guest, guest_registers);
    push_guest_register(guest, dirty_registers);
}

fn push_guest_register(guest: u8, registers: &mut Vec<u8>) {
    if guest != 0 && !registers.contains(&guest) {
        registers.push(guest);
    }
}

struct TraceSideExitPatch {
    branch_offset: usize,
    condition: A64Cond,
    side_exit_pc: u64,
    executed_instructions: u64,
    dynamic_base: bool,
    dirty_registers: Vec<(u8, u8)>,
}

struct A64Emitter {
    code: Vec<u8>,
    listing: Vec<NativeEmission>,
    include_listing: bool,
    trace_side_exits: Vec<TraceSideExitPatch>,
}

impl A64Emitter {
    fn new(include_listing: bool) -> Self {
        Self {
            code: Vec::new(),
            listing: Vec::new(),
            include_listing,
            trace_side_exits: Vec::new(),
        }
    }

    fn finish(self) -> EmittedCode {
        EmittedCode {
            bytes: self.code,
            listing: self.listing,
        }
    }

    fn current_offset(&self) -> usize {
        self.code.len()
    }

    fn emit_trace_plan(&mut self, plan: &BlockPlan) {
        let loops_in_native_trace = trace_plan_has_native_loop(plan);
        self.emit_prologue(loops_in_native_trace);
        if loops_in_native_trace {
            self.emit(
                mov_reg(DYNAMIC_INSTRUCTION_COUNT, A64_ZERO_REGISTER),
                format!("mov x{DYNAMIC_INSTRUCTION_COUNT}, xzr ; trace instruction count"),
            );
        }
        let loop_start = self.current_offset();

        for operation in &plan.operations {
            match operation.kind() {
                BlockOperationKind::Native(NativeInstruction::TraceGuard {
                    rs1,
                    rs2,
                    condition,
                    continue_on_taken,
                    side_exit_pc,
                    executed_instructions,
                    ..
                }) => self.emit_trace_guard(
                    rs1,
                    rs2,
                    condition,
                    continue_on_taken,
                    side_exit_pc,
                    executed_instructions,
                    loops_in_native_trace,
                ),
                BlockOperationKind::Native(NativeInstruction::TraceLoopGuard {
                    rs1,
                    rs2,
                    condition,
                    continue_on_taken,
                    side_exit_pc,
                    guest_instruction_count,
                    ..
                }) => self.emit_trace_loop_guard(
                    rs1,
                    rs2,
                    condition,
                    continue_on_taken,
                    side_exit_pc,
                    guest_instruction_count,
                    loop_start,
                ),
                BlockOperationKind::Native(instruction) => self.emit_instruction(instruction),
            }
        }

        if !loops_in_native_trace {
            if !matches!(
                plan.operations.last().map(|operation| operation.kind()),
                Some(BlockOperationKind::Native(instruction)) if instruction.terminates_block()
            ) {
                self.store_pc(plan.end_pc);
            }
            self.mov_imm64(0, plan.guest_instruction_count as u64);
            self.emit_epilogue(false);
            self.ret();
        }
        self.emit_trace_side_exits();
    }

    fn emit_trace_side_exits(&mut self) {
        let side_exits = std::mem::take(&mut self.trace_side_exits);
        for side_exit in side_exits {
            let side_exit_offset = self.current_offset();
            self.patch_branch(
                side_exit.branch_offset,
                b_cond(
                    side_exit.branch_offset,
                    side_exit_offset,
                    side_exit.condition,
                ),
            );
            self.flush_regalloc_dirty_registers(&side_exit.dirty_registers);
            self.store_pc(side_exit.side_exit_pc);
            if side_exit.dynamic_base {
                self.emit_host_add_sub_imm_or_move(
                    0,
                    DYNAMIC_INSTRUCTION_COUNT,
                    side_exit.executed_instructions as i64,
                );
            } else {
                self.mov_imm64(0, side_exit.executed_instructions);
            }
            self.emit_epilogue(side_exit.dynamic_base);
            self.ret();
        }
    }

    fn emit_instruction(&mut self, instruction: NativeInstruction) {
        match instruction {
            NativeInstruction::Add { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "add", add_reg)
            }
            NativeInstruction::Addi { rd, rs1, imm } => self.emit_add_sub_imm(rd, rs1, imm),
            NativeInstruction::Addiw { rd, rs1, imm } => self.emit_add_sub_imm32(rd, rs1, imm),
            NativeInstruction::Addw { rd, rs1, rs2 } => {
                self.emit_word_binary_reg(rd, rs1, rs2, "addw", add_reg32)
            }
            NativeInstruction::And { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "and", and_reg)
            }
            NativeInstruction::AndBranch {
                rd,
                rs1,
                imm,
                target,
                fallthrough,
                branch_if_zero,
            } => self.emit_masked_zero_branch(rd, rs1, imm, target, fallthrough, branch_if_zero),
            NativeInstruction::Andi { rd, rs1, imm } => {
                self.emit_logical_imm(rd, rs1, imm, "and", and_reg)
            }
            NativeInstruction::Auipc { rd, value }
            | NativeInstruction::Lui { rd, value }
            | NativeInstruction::LoadImmediate { rd, value } => {
                self.mov_imm64(SCRATCH0, value);
                self.store_guest_register(rd, SCRATCH0);
            }
            NativeInstruction::Beq {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_conditional_branch(rs1, rs2, target, fallthrough, A64Cond::Eq),
            NativeInstruction::Bge {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_conditional_branch(rs1, rs2, target, fallthrough, A64Cond::Ge),
            NativeInstruction::Bgeu {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_conditional_branch(rs1, rs2, target, fallthrough, A64Cond::Hs),
            NativeInstruction::Blt {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_conditional_branch(rs1, rs2, target, fallthrough, A64Cond::Lt),
            NativeInstruction::Bltu {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_conditional_branch(rs1, rs2, target, fallthrough, A64Cond::Lo),
            NativeInstruction::Bne {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_conditional_branch(rs1, rs2, target, fallthrough, A64Cond::Ne),
            NativeInstruction::ArithmeticXorToggleLoop(_) => debug_assert!(
                false,
                "arithmetic xor toggle loops must be emitted as whole regions"
            ),
            NativeInstruction::CountedDiamondLoop(_) => debug_assert!(
                false,
                "counted diamond loops must be emitted as whole regions"
            ),
            NativeInstruction::Ecall { next_pc } => self.emit_ecall(next_pc),
            NativeInstruction::FibonacciRecurrenceLoop(_) => debug_assert!(
                false,
                "fibonacci recurrence loops must be emitted as whole regions"
            ),
            NativeInstruction::FloatLoad {
                rd,
                rs1,
                imm,
                width,
            } => self.emit_float_load(rd, rs1, imm, width.bytes()),
            NativeInstruction::FloatStore {
                rs1,
                rs2,
                imm,
                width,
            } => self.emit_float_store(rs1, rs2, imm, width.bytes()),
            NativeInstruction::RuntimeFloat {
                rd,
                rm,
                rs1,
                rs2,
                rs3,
                op,
            } => self.emit_runtime_float(rd, rm, rs1, rs2, rs3, op as u64),
            NativeInstruction::InlinedJump { .. } => {}
            NativeInstruction::Jal {
                rd,
                target,
                return_pc,
            } => self.emit_jal(rd, target, return_pc),
            NativeInstruction::Jalr {
                rd,
                rs1,
                imm,
                return_pc,
            } => self.emit_jalr(rd, rs1, imm, return_pc),
            NativeInstruction::Jump { target } => self.store_pc(target),
            NativeInstruction::JumpReg { rs1 } => self.emit_jump_reg(rs1),
            NativeInstruction::JumpRegLink { rs1, return_pc } => {
                self.emit_jump_reg_link(rs1, return_pc)
            }
            NativeInstruction::Load {
                rd,
                rs1,
                imm,
                width,
                signed,
            } => self.emit_load(rd, rs1, imm, width.bytes(), signed),
            NativeInstruction::Move { rd, rs } => self.emit_move(rd, rs),
            NativeInstruction::Mul { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "mul", mul_reg)
            }
            NativeInstruction::Mulh { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "smulh", smulh_reg)
            }
            NativeInstruction::Mulhu { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "umulh", umulh_reg)
            }
            NativeInstruction::Mulw { rd, rs1, rs2 } => {
                self.emit_word_binary_reg(rd, rs1, rs2, "mulw", mul_reg32)
            }
            NativeInstruction::Nop => {}
            NativeInstruction::Or { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "orr", orr_reg)
            }
            NativeInstruction::Ori { rd, rs1, imm } => {
                self.emit_logical_imm(rd, rs1, imm, "orr", orr_reg)
            }
            NativeInstruction::Sll { rd, rs1, rs2 } => {
                self.emit_shift_reg(rd, rs1, rs2, "lslv", lslv_reg)
            }
            NativeInstruction::Slli { rd, rs1, shamt } => {
                self.emit_shift_imm(rd, rs1, shamt, "lslv", lslv_reg)
            }
            NativeInstruction::Slliw { rd, rs1, shamt } => {
                self.emit_shift_imm32(rd, rs1, shamt, "slliw", lslv_reg32)
            }
            NativeInstruction::Sllw { rd, rs1, rs2 } => {
                self.emit_word_binary_reg(rd, rs1, rs2, "sllw", lslv_reg32)
            }
            NativeInstruction::Slt { rd, rs1, rs2 } => {
                self.emit_compare_set_reg(rd, rs1, rs2, A64Cond::Lt, "slt")
            }
            NativeInstruction::Slti { rd, rs1, imm } => {
                self.emit_compare_set_imm(rd, rs1, imm, A64Cond::Lt, "slti")
            }
            NativeInstruction::Sltiu { rd, rs1, imm } => {
                self.emit_compare_set_imm(rd, rs1, imm, A64Cond::Lo, "sltiu")
            }
            NativeInstruction::Sltu { rd, rs1, rs2 } => {
                self.emit_compare_set_reg(rd, rs1, rs2, A64Cond::Lo, "sltu")
            }
            NativeInstruction::Sra { rd, rs1, rs2 } => {
                self.emit_shift_reg(rd, rs1, rs2, "asrv", asrv_reg)
            }
            NativeInstruction::Srai { rd, rs1, shamt } => {
                self.emit_shift_imm(rd, rs1, shamt, "asrv", asrv_reg)
            }
            NativeInstruction::Sraiw { rd, rs1, shamt } => {
                self.emit_shift_imm32(rd, rs1, shamt, "sraiw", asrv_reg32)
            }
            NativeInstruction::Sraw { rd, rs1, rs2 } => {
                self.emit_word_binary_reg(rd, rs1, rs2, "sraw", asrv_reg32)
            }
            NativeInstruction::Srl { rd, rs1, rs2 } => {
                self.emit_shift_reg(rd, rs1, rs2, "lsrv", lsrv_reg)
            }
            NativeInstruction::Srli { rd, rs1, shamt } => {
                self.emit_shift_imm(rd, rs1, shamt, "lsrv", lsrv_reg)
            }
            NativeInstruction::Srliw { rd, rs1, shamt } => {
                self.emit_shift_imm32(rd, rs1, shamt, "srliw", lsrv_reg32)
            }
            NativeInstruction::Srlw { rd, rs1, rs2 } => {
                self.emit_word_binary_reg(rd, rs1, rs2, "srlw", lsrv_reg32)
            }
            NativeInstruction::Sub { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "sub", sub_reg)
            }
            NativeInstruction::Subw { rd, rs1, rs2 } => {
                self.emit_word_binary_reg(rd, rs1, rs2, "subw", sub_reg32)
            }
            NativeInstruction::RuntimeBinary { rd, rs1, rs2, op } => {
                self.emit_runtime_binary(rd, rs1, rs2, op as u64)
            }
            NativeInstruction::RuntimeCsr {
                rd,
                rs1_or_uimm,
                csr,
                op,
            } => self.emit_runtime_csr(rd, rs1_or_uimm, csr, op),
            NativeInstruction::RuntimeTrap { pc, opcode, op } => {
                self.emit_runtime_trap(pc, opcode, op as u64)
            }
            NativeInstruction::RuntimeAtomic { rd, rs1, rs2, op } => {
                self.emit_runtime_atomic(rd, rs1, rs2, op as u64)
            }
            NativeInstruction::Store {
                rs1,
                rs2,
                imm,
                width,
            } => self.emit_store(rs1, rs2, imm, width.bytes()),
            NativeInstruction::StoreLoadForwardLoop(_) => debug_assert!(
                false,
                "store/load forwarding loops must be emitted as whole regions"
            ),
            NativeInstruction::TraceGuard {
                rs1,
                rs2,
                condition,
                continue_on_taken,
                side_exit_pc,
                executed_instructions,
                ..
            } => self.emit_trace_guard(
                rs1,
                rs2,
                condition,
                continue_on_taken,
                side_exit_pc,
                executed_instructions,
                false,
            ),
            NativeInstruction::TraceLoopGuard { .. } => debug_assert!(
                false,
                "trace loop guards must be emitted as part of a trace plan"
            ),
            NativeInstruction::Xor { rd, rs1, rs2 } => {
                self.emit_binary_reg(rd, rs1, rs2, "eor", eor_reg)
            }
            NativeInstruction::Xori { rd, rs1, imm } => {
                self.emit_logical_imm(rd, rs1, imm, "eor", eor_reg)
            }
        }
    }

    fn emit_arithmetic_xor_toggle_loop(&mut self, region: ArithmeticXorToggleLoop) {
        const COUNT_HOST: u8 = 17;
        const ACC_HOST: u8 = 9;
        const VALUE_HOST: u8 = 10;
        const TOGGLED_HOST: u8 = 11;
        const PAIR_SUM_HOST: u8 = 12;
        const PAIR_COUNT_HOST: u8 = 13;
        const ODD_HOST: u8 = 14;

        self.emit_prologue(false);
        self.load_guest_register(COUNT_HOST, region.counter);
        self.load_guest_register(ACC_HOST, region.accumulator);
        self.load_guest_register(VALUE_HOST, region.value_register);

        self.mov_imm64(LOOP_TEMP, region.xor_imm as u64);
        self.emit(
            eor_reg(TOGGLED_HOST, VALUE_HOST, LOOP_TEMP),
            format!(
                "eor x{TOGGLED_HOST}, x{VALUE_HOST}, x{LOOP_TEMP} ; xor-toggle alternate value"
            ),
        );
        self.emit(
            add_reg(PAIR_SUM_HOST, VALUE_HOST, TOGGLED_HOST),
            format!("add x{PAIR_SUM_HOST}, x{VALUE_HOST}, x{TOGGLED_HOST} ; xor-toggle pair sum"),
        );

        self.mov_imm64(LOOP_TEMP, 1);
        self.emit(
            lsrv_reg(PAIR_COUNT_HOST, COUNT_HOST, LOOP_TEMP),
            format!("lsr x{PAIR_COUNT_HOST}, x{COUNT_HOST}, #1 ; xor-toggle pair count"),
        );
        self.emit(
            cmp_reg(COUNT_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNT_HOST}, xzr ; zero count means 2^64 iterations"),
        );
        self.mov_imm64(LOOP_TEMP, 1u64 << 63);
        self.emit(
            csel(PAIR_COUNT_HOST, LOOP_TEMP, PAIR_COUNT_HOST, A64Cond::Eq),
            format!("csel x{PAIR_COUNT_HOST}, x{LOOP_TEMP}, x{PAIR_COUNT_HOST}, eq ; normalized pair count"),
        );
        self.emit(
            mul_reg(PAIR_SUM_HOST, PAIR_SUM_HOST, PAIR_COUNT_HOST),
            format!("mul x{PAIR_SUM_HOST}, x{PAIR_SUM_HOST}, x{PAIR_COUNT_HOST} ; xor-toggle paired contribution"),
        );
        self.emit(
            add_reg(ACC_HOST, ACC_HOST, PAIR_SUM_HOST),
            format!("add x{ACC_HOST}, x{ACC_HOST}, x{PAIR_SUM_HOST} ; xor-toggle accumulate pairs"),
        );

        self.mov_imm64(LOOP_TEMP, 1);
        self.emit(
            and_reg(ODD_HOST, COUNT_HOST, LOOP_TEMP),
            format!("and x{ODD_HOST}, x{COUNT_HOST}, x{LOOP_TEMP} ; xor-toggle odd iteration"),
        );
        self.emit(
            cmp_reg(ODD_HOST, A64_ZERO_REGISTER),
            format!("cmp x{ODD_HOST}, xzr ; xor-toggle odd check"),
        );
        self.emit(
            csel(LOOP_TEMP, VALUE_HOST, A64_ZERO_REGISTER, A64Cond::Ne),
            format!("csel x{LOOP_TEMP}, x{VALUE_HOST}, xzr, ne ; xor-toggle odd contribution"),
        );
        self.emit(
            add_reg(ACC_HOST, ACC_HOST, LOOP_TEMP),
            format!("add x{ACC_HOST}, x{ACC_HOST}, x{LOOP_TEMP} ; xor-toggle accumulate tail"),
        );
        self.emit(
            csel(VALUE_HOST, TOGGLED_HOST, VALUE_HOST, A64Cond::Ne),
            format!(
                "csel x{VALUE_HOST}, x{TOGGLED_HOST}, x{VALUE_HOST}, ne ; xor-toggle final value"
            ),
        );

        self.store_guest_register(region.counter, A64_ZERO_REGISTER);
        self.store_guest_register(region.accumulator, ACC_HOST);
        self.store_guest_register(region.value_register, VALUE_HOST);
        self.store_pc(region.exit_pc);
        self.mov_imm64(LOOP_TEMP, region.guest_instructions);
        self.emit(
            mul_reg(0, COUNT_HOST, LOOP_TEMP),
            format!("mul x0, x{COUNT_HOST}, x{LOOP_TEMP} ; xor-toggle guest instruction count"),
        );
        self.emit_epilogue(false);
        self.ret();
    }

    fn emit_counted_diamond_loop(&mut self, region: CountedDiamondLoop) {
        if self.emit_counted_diamond_loop_closed_form(region) {
            return;
        }
        self.emit_counted_diamond_loop_iterative(region);
    }

    fn emit_counted_diamond_loop_closed_form(&mut self, region: CountedDiamondLoop) -> bool {
        if region.mask != 1 || region.counter_delta != -1 {
            return false;
        }

        const COUNTER_HOST: u8 = 17;
        const ACCUMULATOR_HOST: u8 = 9;
        const ITERATION_HOST: u8 = 10;
        const PAIR_COUNT_HOST: u8 = 11;
        const ODD_COUNT_HOST: u8 = 12;
        const NONZERO_COUNT_HOST: u8 = 13;
        const CONTRIBUTION_HOST: u8 = 14;
        const INSTRUCTION_COUNT_HOST: u8 = 15;

        self.emit_prologue(false);
        self.load_guest_register(COUNTER_HOST, region.counter);
        self.load_guest_register(ACCUMULATOR_HOST, region.accumulator);
        self.load_guest_register(ITERATION_HOST, region.iteration_register);

        self.mov_imm64(LOOP_TEMP, 1);
        self.emit(
            lsrv_reg(PAIR_COUNT_HOST, COUNTER_HOST, LOOP_TEMP),
            format!("lsr x{PAIR_COUNT_HOST}, x{COUNTER_HOST}, #1 ; counted diamond pair count"),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; zero count means 2^64 iterations"),
        );
        self.mov_imm64(LOOP_TEMP, 1u64 << 63);
        self.emit(
            csel(PAIR_COUNT_HOST, LOOP_TEMP, PAIR_COUNT_HOST, A64Cond::Eq),
            format!("csel x{PAIR_COUNT_HOST}, x{LOOP_TEMP}, x{PAIR_COUNT_HOST}, eq ; normalized pair count"),
        );
        self.mov_imm64(LOOP_TEMP, 1);
        self.emit(
            and_reg(ODD_COUNT_HOST, COUNTER_HOST, LOOP_TEMP),
            format!(
                "and x{ODD_COUNT_HOST}, x{COUNTER_HOST}, x{LOOP_TEMP} ; counted diamond odd count"
            ),
        );
        self.emit(
            add_reg(NONZERO_COUNT_HOST, PAIR_COUNT_HOST, ODD_COUNT_HOST),
            format!("add x{NONZERO_COUNT_HOST}, x{PAIR_COUNT_HOST}, x{ODD_COUNT_HOST} ; counted diamond nonzero count"),
        );

        self.mov_imm64(LOOP_TEMP, region.zero_accumulator_delta as u64);
        self.emit(
            mul_reg(CONTRIBUTION_HOST, PAIR_COUNT_HOST, LOOP_TEMP),
            format!("mul x{CONTRIBUTION_HOST}, x{PAIR_COUNT_HOST}, x{LOOP_TEMP} ; counted diamond zero contribution"),
        );
        self.emit(
            add_reg(ACCUMULATOR_HOST, ACCUMULATOR_HOST, CONTRIBUTION_HOST),
            format!("add x{ACCUMULATOR_HOST}, x{ACCUMULATOR_HOST}, x{CONTRIBUTION_HOST} ; counted diamond zero accumulate"),
        );
        self.mov_imm64(LOOP_TEMP, region.nonzero_accumulator_delta as u64);
        self.emit(
            mul_reg(CONTRIBUTION_HOST, NONZERO_COUNT_HOST, LOOP_TEMP),
            format!("mul x{CONTRIBUTION_HOST}, x{NONZERO_COUNT_HOST}, x{LOOP_TEMP} ; counted diamond nonzero contribution"),
        );
        self.emit(
            add_reg(ACCUMULATOR_HOST, ACCUMULATOR_HOST, CONTRIBUTION_HOST),
            format!("add x{ACCUMULATOR_HOST}, x{ACCUMULATOR_HOST}, x{CONTRIBUTION_HOST} ; counted diamond nonzero accumulate"),
        );

        self.mov_imm64(LOOP_TEMP, region.iteration_delta as u64);
        self.emit(
            mul_reg(CONTRIBUTION_HOST, COUNTER_HOST, LOOP_TEMP),
            format!("mul x{CONTRIBUTION_HOST}, x{COUNTER_HOST}, x{LOOP_TEMP} ; counted diamond iteration delta"),
        );
        self.emit(
            add_reg(ITERATION_HOST, ITERATION_HOST, CONTRIBUTION_HOST),
            format!("add x{ITERATION_HOST}, x{ITERATION_HOST}, x{CONTRIBUTION_HOST} ; counted diamond iterations"),
        );

        self.mov_imm64(LOOP_TEMP, region.zero_guest_instructions);
        self.emit(
            mul_reg(INSTRUCTION_COUNT_HOST, PAIR_COUNT_HOST, LOOP_TEMP),
            format!("mul x{INSTRUCTION_COUNT_HOST}, x{PAIR_COUNT_HOST}, x{LOOP_TEMP} ; counted diamond zero instruction count"),
        );
        self.mov_imm64(LOOP_TEMP, region.nonzero_guest_instructions);
        self.emit(
            mul_reg(CONTRIBUTION_HOST, NONZERO_COUNT_HOST, LOOP_TEMP),
            format!("mul x{CONTRIBUTION_HOST}, x{NONZERO_COUNT_HOST}, x{LOOP_TEMP} ; counted diamond nonzero instruction count"),
        );
        self.emit(
            add_reg(
                INSTRUCTION_COUNT_HOST,
                INSTRUCTION_COUNT_HOST,
                CONTRIBUTION_HOST,
            ),
            format!("add x{INSTRUCTION_COUNT_HOST}, x{INSTRUCTION_COUNT_HOST}, x{CONTRIBUTION_HOST} ; counted diamond instruction count"),
        );

        self.store_guest_register(region.counter, A64_ZERO_REGISTER);
        self.store_guest_register(region.accumulator, ACCUMULATOR_HOST);
        self.store_guest_register(region.iteration_register, ITERATION_HOST);
        self.mov_imm64(LOOP_TEMP, 1);
        self.store_guest_register(region.parity_register, LOOP_TEMP);
        self.store_pc(region.exit_pc);
        self.emit(
            mov_reg(0, INSTRUCTION_COUNT_HOST),
            format!("mov x0, x{INSTRUCTION_COUNT_HOST} ; return executed instructions"),
        );
        self.emit_epilogue(false);
        self.ret();
        true
    }

    fn emit_counted_diamond_loop_iterative(&mut self, region: CountedDiamondLoop) {
        const COUNTER_HOST: u8 = 9;
        const ACCUMULATOR_HOST: u8 = 10;
        const ITERATION_HOST: u8 = 11;
        const PARITY_HOST: u8 = 12;
        const MASK_HOST: u8 = 13;
        const ZERO_DELTA_HOST: u8 = 14;
        const NONZERO_DELTA_HOST: u8 = 15;
        const SELECTED_HOST: u8 = 16;
        const ZERO_COUNT_HOST: u8 = 17;
        const NONZERO_COUNT_HOST: u8 = 22;

        self.emit_prologue(true);
        self.load_guest_register(COUNTER_HOST, region.counter);
        self.load_guest_register(ACCUMULATOR_HOST, region.accumulator);
        self.load_guest_register(ITERATION_HOST, region.iteration_register);
        self.mov_imm64(DYNAMIC_INSTRUCTION_COUNT, 0);
        self.mov_imm64(MASK_HOST, region.mask as u64);
        self.mov_imm64(ZERO_DELTA_HOST, region.zero_accumulator_delta as u64);
        self.mov_imm64(NONZERO_DELTA_HOST, region.nonzero_accumulator_delta as u64);
        self.mov_imm64(ZERO_COUNT_HOST, region.zero_guest_instructions);
        self.mov_imm64(NONZERO_COUNT_HOST, region.nonzero_guest_instructions);

        let loop_start = self.current_offset();
        self.emit(
            ands_reg(PARITY_HOST, COUNTER_HOST, MASK_HOST),
            format!("ands x{PARITY_HOST}, x{COUNTER_HOST}, x{MASK_HOST} ; counted diamond parity"),
        );
        self.emit(
            csel(
                SELECTED_HOST,
                ZERO_DELTA_HOST,
                NONZERO_DELTA_HOST,
                A64Cond::Eq,
            ),
            format!(
                "csel x{SELECTED_HOST}, x{ZERO_DELTA_HOST}, x{NONZERO_DELTA_HOST}, eq ; accumulator delta"
            ),
        );
        self.emit(
            add_reg(ACCUMULATOR_HOST, ACCUMULATOR_HOST, SELECTED_HOST),
            format!("add x{ACCUMULATOR_HOST}, x{ACCUMULATOR_HOST}, x{SELECTED_HOST}"),
        );
        self.emit(
            csel(
                SELECTED_HOST,
                ZERO_COUNT_HOST,
                NONZERO_COUNT_HOST,
                A64Cond::Eq,
            ),
            format!(
                "csel x{SELECTED_HOST}, x{ZERO_COUNT_HOST}, x{NONZERO_COUNT_HOST}, eq ; guest instruction count"
            ),
        );
        self.emit(
            add_reg(
                DYNAMIC_INSTRUCTION_COUNT,
                DYNAMIC_INSTRUCTION_COUNT,
                SELECTED_HOST,
            ),
            format!(
                "add x{DYNAMIC_INSTRUCTION_COUNT}, x{DYNAMIC_INSTRUCTION_COUNT}, x{SELECTED_HOST}"
            ),
        );
        self.emit_host_add_sub_imm_any(ITERATION_HOST, ITERATION_HOST, region.iteration_delta);
        self.emit_host_add_sub_imm_any(COUNTER_HOST, COUNTER_HOST, region.counter_delta);
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; counted diamond backedge"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, A64Cond::Ne),
            "b.ne .counted_diamond_loop".to_string(),
        );

        self.store_guest_register(region.counter, COUNTER_HOST);
        self.store_guest_register(region.accumulator, ACCUMULATOR_HOST);
        self.store_guest_register(region.iteration_register, ITERATION_HOST);
        self.store_guest_register(region.parity_register, PARITY_HOST);
        self.store_pc(region.exit_pc);
        self.emit(
            mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
            format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return executed instructions"),
        );
        self.emit_epilogue(true);
        self.ret();
    }

    fn emit_fibonacci_recurrence_loop(&mut self, region: FibonacciRecurrenceLoop) {
        const COUNTER_HOST: u8 = 22;
        const CURRENT_HOST: u8 = 23;
        const PREVIOUS_HOST: u8 = 24;
        const CHECKSUM_HOST: u8 = 25;
        const TRIP_COUNT_HOST: u8 = 26;
        const STEP_HOST: u8 = 27;
        const SAVED_HOST: u8 = 28;

        self.emit_prologue(true);
        self.load_guest_register(COUNTER_HOST, region.counter);
        self.mov_imm64(STEP_HOST, u64::from(u32::MAX));
        self.emit(
            and_reg(COUNTER_HOST, COUNTER_HOST, STEP_HOST),
            format!("and x{COUNTER_HOST}, x{COUNTER_HOST}, x{STEP_HOST} ; fib u32 trip count"),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; fib zero trip count"),
        );
        let nonzero_branch = self.emit_patchable_branch("b.ne .fib_count_ready".to_string());
        self.mov_imm64(COUNTER_HOST, 1u64 << 32);
        let count_ready = self.current_offset();
        self.patch_branch(
            nonzero_branch,
            b_cond(nonzero_branch, count_ready, A64Cond::Ne),
        );

        self.emit(
            mov_reg(TRIP_COUNT_HOST, COUNTER_HOST),
            format!("mov x{TRIP_COUNT_HOST}, x{COUNTER_HOST} ; fib trip count"),
        );
        self.load_guest_register(CURRENT_HOST, region.current_register);
        self.load_guest_register(PREVIOUS_HOST, region.previous_register);
        self.load_guest_register(CHECKSUM_HOST, region.checksum_register);

        self.mov_imm64(STEP_HOST, 16);
        self.emit(
            cmp_reg(COUNTER_HOST, STEP_HOST),
            format!("cmp x{COUNTER_HOST}, x{STEP_HOST} ; fib unrolled count"),
        );
        let tail_branch = self.emit_patchable_branch("b.lo .fib_tail".to_string());
        let unrolled_loop = self.current_offset();
        for _ in 0..8 {
            self.emit_fibonacci_recurrence_pair(CURRENT_HOST, PREVIOUS_HOST, CHECKSUM_HOST);
        }
        self.emit(
            sub_imm(COUNTER_HOST, COUNTER_HOST, 16),
            format!("sub x{COUNTER_HOST}, x{COUNTER_HOST}, #16 ; fib unrolled count"),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, STEP_HOST),
            format!("cmp x{COUNTER_HOST}, x{STEP_HOST} ; fib unrolled backedge"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, unrolled_loop, A64Cond::Hs),
            "b.hs .fib_unrolled_loop".to_string(),
        );
        let tail_offset = self.current_offset();
        self.patch_branch(tail_branch, b_cond(tail_branch, tail_offset, A64Cond::Lo));

        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; fib tail done"),
        );
        let done_branch = self.emit_patchable_branch("b.eq .fib_done".to_string());
        let tail_loop = self.current_offset();
        self.emit_fibonacci_recurrence_step(CURRENT_HOST, PREVIOUS_HOST, CHECKSUM_HOST, SAVED_HOST);
        self.emit(
            sub_imm(COUNTER_HOST, COUNTER_HOST, 1),
            format!("sub x{COUNTER_HOST}, x{COUNTER_HOST}, #1 ; fib tail count"),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; fib tail backedge"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, tail_loop, A64Cond::Ne),
            "b.ne .fib_tail_loop".to_string(),
        );

        let done_offset = self.current_offset();
        self.patch_branch(done_branch, b_cond(done_branch, done_offset, A64Cond::Eq));
        self.mov_imm64(SCRATCH0, 0);
        self.store_guest_register(region.counter, SCRATCH0);
        self.store_guest_register(region.current_register, CURRENT_HOST);
        self.store_guest_register(region.previous_register, PREVIOUS_HOST);
        self.store_guest_register(region.checksum_register, CHECKSUM_HOST);
        self.store_guest_register(region.saved_register, PREVIOUS_HOST);
        self.store_pc(region.exit_pc);
        self.mov_imm64(STEP_HOST, region.guest_instructions);
        self.emit(
            mul_reg(0, TRIP_COUNT_HOST, STEP_HOST),
            format!("mul x0, x{TRIP_COUNT_HOST}, x{STEP_HOST} ; fib executed instructions"),
        );
        self.emit_epilogue(true);
        self.ret();
    }

    fn emit_fibonacci_recurrence_step(
        &mut self,
        current: u8,
        previous: u8,
        checksum: u8,
        saved: u8,
    ) {
        self.emit(
            mov_reg(saved, current),
            format!("mov x{saved}, x{current} ; fib saved current"),
        );
        self.emit(
            add_reg(current, current, previous),
            format!("add x{current}, x{current}, x{previous} ; fib next"),
        );
        self.emit(
            eor_reg(checksum, checksum, current),
            format!("eor x{checksum}, x{checksum}, x{current} ; fib checksum"),
        );
        self.emit(
            mov_reg(previous, saved),
            format!("mov x{previous}, x{saved} ; fib rotate previous"),
        );
    }

    fn emit_fibonacci_recurrence_pair(&mut self, current: u8, previous: u8, checksum: u8) {
        self.emit(
            add_reg(previous, current, previous),
            format!("add x{previous}, x{current}, x{previous} ; fib next"),
        );
        self.emit(
            eor_reg(checksum, checksum, previous),
            format!("eor x{checksum}, x{checksum}, x{previous} ; fib checksum"),
        );
        self.emit(
            add_reg(current, previous, current),
            format!("add x{current}, x{previous}, x{current} ; fib next"),
        );
        self.emit(
            eor_reg(checksum, checksum, current),
            format!("eor x{checksum}, x{checksum}, x{current} ; fib checksum"),
        );
    }

    fn emit_store_load_forward_loop(&mut self, region: StoreLoadForwardLoop) {
        if self.emit_store_load_forward_loop_final_state(region) {
            return;
        }
        self.emit_store_load_forward_loop_iterative(region);
    }

    fn emit_store_load_forward_loop_final_state(&mut self, region: StoreLoadForwardLoop) -> bool {
        let width = region.width.bytes();
        if region.width != super::MemoryWidth::Double
            || region.offset_delta != width as i64
            || region.mask < 0
        {
            return false;
        }
        let ring_bytes = region.mask as u64 + width;
        if !ring_bytes.is_power_of_two() || ring_bytes % width != 0 {
            return false;
        }

        const DATA_PTR_HOST: u8 = 22;
        const COUNTER_HOST: u8 = 23;
        const VALUE_HOST: u8 = 24;
        const OFFSET_HOST: u8 = 25;
        const BASE_HOST: u8 = 26;
        const INDEX_HOST: u8 = 27;
        const ADDRESS_HOST: u8 = 28;
        const CURRENT_VALUE_HOST: u8 = 12;
        const MASK_HOST: u8 = 13;
        const DELTA_HOST: u8 = 14;
        const SLOT_COUNT_HOST: u8 = 15;
        const STORE_COUNT_HOST: u8 = 17;

        let slot_count = ring_bytes / width;
        debug_assert!(slot_count > 0);

        self.emit_prologue(true);
        self.load_guest_register(SCRATCH0, region.base_register);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; direct write cpu"),
        );
        self.emit(
            mov_reg(1, SCRATCH0),
            format!("mov x1, x{SCRATCH0} ; direct write base"),
        );
        self.mov_imm64(2, ring_bytes);
        self.emit_call(
            jit_runtime_direct_write_ptr as *const () as usize as u64,
            "jit_runtime_direct_write_ptr",
        );
        self.emit(
            mov_reg(DATA_PTR_HOST, 0),
            format!("mov x{DATA_PTR_HOST}, x0 ; direct write host pointer"),
        );
        self.emit(
            cmp_reg(DATA_PTR_HOST, A64_ZERO_REGISTER),
            format!("cmp x{DATA_PTR_HOST}, xzr ; direct write available"),
        );
        let ok_branch = self.emit_patchable_branch("b.ne .direct_final_state_ready".to_string());
        self.emit(mov_reg(0, A64_ZERO_REGISTER), "mov x0, xzr".to_string());
        self.emit_epilogue(true);
        self.ret();
        let ok_offset = self.current_offset();
        self.patch_branch(ok_branch, b_cond(ok_branch, ok_offset, A64Cond::Ne));

        self.load_guest_register(COUNTER_HOST, region.counter);
        self.load_guest_register(VALUE_HOST, region.value_register);
        self.load_guest_register(OFFSET_HOST, region.offset_register);
        self.load_guest_register(BASE_HOST, region.base_register);
        self.mov_imm64(MASK_HOST, region.mask as u64);
        self.mov_imm64(DELTA_HOST, region.value_delta_after_forwarded_xor as u64);
        self.mov_imm64(SLOT_COUNT_HOST, slot_count);

        self.emit(
            and_reg(INDEX_HOST, OFFSET_HOST, MASK_HOST),
            format!("and x{INDEX_HOST}, x{OFFSET_HOST}, x{MASK_HOST} ; final-state start index"),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; zero count covers full ring"),
        );
        self.emit(
            csel(
                STORE_COUNT_HOST,
                SLOT_COUNT_HOST,
                COUNTER_HOST,
                A64Cond::Eq,
            ),
            format!("csel x{STORE_COUNT_HOST}, x{SLOT_COUNT_HOST}, x{COUNTER_HOST}, eq ; final-state store count"),
        );
        self.emit(
            cmp_reg(STORE_COUNT_HOST, SLOT_COUNT_HOST),
            format!("cmp x{STORE_COUNT_HOST}, x{SLOT_COUNT_HOST} ; clamp store count"),
        );
        self.emit(
            csel(
                STORE_COUNT_HOST,
                SLOT_COUNT_HOST,
                STORE_COUNT_HOST,
                A64Cond::Hs,
            ),
            format!("csel x{STORE_COUNT_HOST}, x{SLOT_COUNT_HOST}, x{STORE_COUNT_HOST}, hs ; clamped store count"),
        );
        self.emit(
            mov_reg(CURRENT_VALUE_HOST, VALUE_HOST),
            format!("mov x{CURRENT_VALUE_HOST}, x{VALUE_HOST} ; first forwarded store value"),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; full-ring first value"),
        );
        self.emit(
            csel(
                CURRENT_VALUE_HOST,
                DELTA_HOST,
                CURRENT_VALUE_HOST,
                A64Cond::Eq,
            ),
            format!("csel x{CURRENT_VALUE_HOST}, x{DELTA_HOST}, x{CURRENT_VALUE_HOST}, eq ; zero-count first value"),
        );
        self.emit(
            cmp_reg(SLOT_COUNT_HOST, COUNTER_HOST),
            format!("cmp x{SLOT_COUNT_HOST}, x{COUNTER_HOST} ; first slot overwritten"),
        );
        self.emit(
            csel(
                CURRENT_VALUE_HOST,
                DELTA_HOST,
                CURRENT_VALUE_HOST,
                A64Cond::Lo,
            ),
            format!("csel x{CURRENT_VALUE_HOST}, x{DELTA_HOST}, x{CURRENT_VALUE_HOST}, lo ; wrapped first value"),
        );

        let loop_start = self.current_offset();
        self.emit(
            add_reg(ADDRESS_HOST, DATA_PTR_HOST, INDEX_HOST),
            format!(
                "add x{ADDRESS_HOST}, x{DATA_PTR_HOST}, x{INDEX_HOST} ; final-state host address"
            ),
        );
        self.emit(
            str_u64(CURRENT_VALUE_HOST, ADDRESS_HOST, 0),
            format!("str x{CURRENT_VALUE_HOST}, [x{ADDRESS_HOST}] ; final-state guest store"),
        );
        self.emit(
            mov_reg(CURRENT_VALUE_HOST, DELTA_HOST),
            format!("mov x{CURRENT_VALUE_HOST}, x{DELTA_HOST} ; forwarded steady-state value"),
        );
        self.emit_host_add_sub_imm_any(INDEX_HOST, INDEX_HOST, width as i64);
        self.emit(
            and_reg(INDEX_HOST, INDEX_HOST, MASK_HOST),
            format!("and x{INDEX_HOST}, x{INDEX_HOST}, x{MASK_HOST} ; final-state wrap index"),
        );
        self.emit(
            sub_imm(STORE_COUNT_HOST, STORE_COUNT_HOST, 1),
            format!(
                "sub x{STORE_COUNT_HOST}, x{STORE_COUNT_HOST}, #1 ; final-state stores remaining"
            ),
        );
        self.emit(
            cmp_reg(STORE_COUNT_HOST, A64_ZERO_REGISTER),
            format!("cmp x{STORE_COUNT_HOST}, xzr ; final-state store loop"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, A64Cond::Ne),
            "b.ne .direct_final_state_store_loop".to_string(),
        );

        self.mov_imm64(LOOP_TEMP, width);
        self.emit(
            mul_reg(ADDRESS_HOST, COUNTER_HOST, LOOP_TEMP),
            format!("mul x{ADDRESS_HOST}, x{COUNTER_HOST}, x{LOOP_TEMP} ; final offset delta"),
        );
        self.emit(
            add_reg(OFFSET_HOST, OFFSET_HOST, ADDRESS_HOST),
            format!("add x{OFFSET_HOST}, x{OFFSET_HOST}, x{ADDRESS_HOST} ; final guest offset"),
        );
        self.emit(
            sub_imm(INDEX_HOST, OFFSET_HOST, width as u16),
            format!(
                "sub x{INDEX_HOST}, x{OFFSET_HOST}, #{} ; final last offset",
                width
            ),
        );
        self.emit(
            and_reg(INDEX_HOST, INDEX_HOST, MASK_HOST),
            format!("and x{INDEX_HOST}, x{INDEX_HOST}, x{MASK_HOST} ; final guest index"),
        );
        self.emit(
            add_reg(ADDRESS_HOST, BASE_HOST, INDEX_HOST),
            format!("add x{ADDRESS_HOST}, x{BASE_HOST}, x{INDEX_HOST} ; final guest address"),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; final loaded value"),
        );
        self.emit(
            csel(CURRENT_VALUE_HOST, DELTA_HOST, VALUE_HOST, A64Cond::Eq),
            format!("csel x{CURRENT_VALUE_HOST}, x{DELTA_HOST}, x{VALUE_HOST}, eq ; zero-count loaded value"),
        );
        self.mov_imm64(LOOP_TEMP, 1);
        self.emit(
            cmp_reg(COUNTER_HOST, LOOP_TEMP),
            format!("cmp x{COUNTER_HOST}, x{LOOP_TEMP} ; single-iteration loaded value"),
        );
        self.emit(
            csel(CURRENT_VALUE_HOST, VALUE_HOST, DELTA_HOST, A64Cond::Eq),
            format!(
                "csel x{CURRENT_VALUE_HOST}, x{VALUE_HOST}, x{DELTA_HOST}, eq ; final loaded value"
            ),
        );

        self.store_guest_register(region.counter, A64_ZERO_REGISTER);
        self.store_guest_register(region.value_register, DELTA_HOST);
        self.store_guest_register(region.offset_register, OFFSET_HOST);
        self.store_guest_register(region.index_register, INDEX_HOST);
        self.store_guest_register(region.address_register, ADDRESS_HOST);
        self.store_guest_register(region.loaded_register, CURRENT_VALUE_HOST);
        self.store_pc(region.exit_pc);
        self.mov_imm64(LOOP_TEMP, region.guest_instructions);
        self.emit(
            mul_reg(0, COUNTER_HOST, LOOP_TEMP),
            format!("mul x0, x{COUNTER_HOST}, x{LOOP_TEMP} ; final-state guest instruction count"),
        );
        self.emit_epilogue(true);
        self.ret();
        true
    }

    fn emit_store_load_forward_loop_iterative(&mut self, region: StoreLoadForwardLoop) {
        const DATA_PTR_HOST: u8 = 22;
        const COUNTER_HOST: u8 = 23;
        const VALUE_HOST: u8 = 24;
        const OFFSET_HOST: u8 = 25;
        const BASE_HOST: u8 = 26;
        const INDEX_HOST: u8 = 27;
        const ADDRESS_HOST: u8 = 28;
        const LOADED_HOST: u8 = 17;
        const MASK_HOST: u8 = 15;
        const ITERATION_COUNT_HOST: u8 = 14;

        debug_assert_eq!(region.width, super::MemoryWidth::Double);
        self.emit_prologue(true);
        self.load_guest_register(SCRATCH0, region.base_register);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; direct write cpu"),
        );
        self.emit(
            mov_reg(1, SCRATCH0),
            format!("mov x1, x{SCRATCH0} ; direct write base"),
        );
        self.mov_imm64(2, region.mask as u64 + region.width.bytes());
        self.emit_call(
            jit_runtime_direct_write_ptr as *const () as usize as u64,
            "jit_runtime_direct_write_ptr",
        );
        self.emit(
            mov_reg(DATA_PTR_HOST, 0),
            format!("mov x{DATA_PTR_HOST}, x0 ; direct write host pointer"),
        );
        self.emit(
            cmp_reg(DATA_PTR_HOST, A64_ZERO_REGISTER),
            format!("cmp x{DATA_PTR_HOST}, xzr ; direct write available"),
        );
        let ok_branch = self.emit_patchable_branch("b.ne .direct_write_ready".to_string());
        self.emit(mov_reg(0, A64_ZERO_REGISTER), "mov x0, xzr".to_string());
        self.emit_epilogue(true);
        self.ret();
        let ok_offset = self.current_offset();
        self.patch_branch(ok_branch, b_cond(ok_branch, ok_offset, A64Cond::Ne));

        self.load_guest_register(COUNTER_HOST, region.counter);
        self.load_guest_register(VALUE_HOST, region.value_register);
        self.load_guest_register(OFFSET_HOST, region.offset_register);
        self.load_guest_register(BASE_HOST, region.base_register);
        self.mov_imm64(DYNAMIC_INSTRUCTION_COUNT, 0);
        self.mov_imm64(MASK_HOST, region.mask as u64);
        self.mov_imm64(ITERATION_COUNT_HOST, region.guest_instructions);

        let loop_start = self.current_offset();
        self.emit(
            and_reg(INDEX_HOST, OFFSET_HOST, MASK_HOST),
            format!("and x{INDEX_HOST}, x{OFFSET_HOST}, x{MASK_HOST} ; direct store index"),
        );
        self.emit(
            add_reg(ADDRESS_HOST, DATA_PTR_HOST, INDEX_HOST),
            format!("add x{ADDRESS_HOST}, x{DATA_PTR_HOST}, x{INDEX_HOST} ; host store address"),
        );
        self.emit(
            str_u64(VALUE_HOST, ADDRESS_HOST, 0),
            format!("str x{VALUE_HOST}, [x{ADDRESS_HOST}] ; direct guest store"),
        );
        self.emit(
            mov_reg(LOADED_HOST, VALUE_HOST),
            format!("mov x{LOADED_HOST}, x{VALUE_HOST} ; forwarded load"),
        );
        self.emit(
            mov_reg(VALUE_HOST, A64_ZERO_REGISTER),
            format!("mov x{VALUE_HOST}, xzr"),
        );
        self.emit_host_add_sub_imm_any(
            VALUE_HOST,
            VALUE_HOST,
            region.value_delta_after_forwarded_xor,
        );
        self.emit_host_add_sub_imm_any(OFFSET_HOST, OFFSET_HOST, region.offset_delta);
        self.emit_host_add_sub_imm_any(COUNTER_HOST, COUNTER_HOST, region.counter_delta);
        self.emit(
            add_reg(
                DYNAMIC_INSTRUCTION_COUNT,
                DYNAMIC_INSTRUCTION_COUNT,
                ITERATION_COUNT_HOST,
            ),
            format!(
                "add x{DYNAMIC_INSTRUCTION_COUNT}, x{DYNAMIC_INSTRUCTION_COUNT}, x{ITERATION_COUNT_HOST}"
            ),
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; direct store loop backedge"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, A64Cond::Ne),
            "b.ne .direct_store_loop".to_string(),
        );

        self.emit(
            add_reg(ADDRESS_HOST, BASE_HOST, INDEX_HOST),
            format!("add x{ADDRESS_HOST}, x{BASE_HOST}, x{INDEX_HOST} ; final guest address"),
        );
        self.store_guest_register(region.counter, COUNTER_HOST);
        self.store_guest_register(region.value_register, VALUE_HOST);
        self.store_guest_register(region.offset_register, OFFSET_HOST);
        self.store_guest_register(region.index_register, INDEX_HOST);
        self.store_guest_register(region.address_register, ADDRESS_HOST);
        self.store_guest_register(region.loaded_register, LOADED_HOST);
        self.store_pc(region.exit_pc);
        self.emit(
            mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
            format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return executed instructions"),
        );
        self.emit_epilogue(true);
        self.ret();
    }

    fn emit_register_allocated_self_loop(&mut self, loop_plan: &RegisterAllocatedLoop<'_>) {
        self.emit_prologue(true);
        let decrementing_counter = loop_plan.decrementing_counter();
        if decrementing_counter.is_none() {
            self.emit(
                mov_reg(DYNAMIC_INSTRUCTION_COUNT, A64_ZERO_REGISTER),
                format!("mov x{DYNAMIC_INSTRUCTION_COUNT}, xzr ; dynamic instruction count"),
            );
        }

        for (guest, host) in &loop_plan.loaded_registers {
            self.load_guest_register(*host, *guest);
        }
        if let Some(counter) = decrementing_counter {
            let counter_host = loop_plan.host_or_zero(counter);
            self.emit(
                mov_reg(DYNAMIC_INSTRUCTION_COUNT, counter_host),
                format!(
                    "mov x{DYNAMIC_INSTRUCTION_COUNT}, x{counter_host} ; initial loop counter x{counter}"
                ),
            );
        }

        let loop_start = self.current_offset();
        for operation in loop_plan.operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            self.emit_loop_instruction(instruction, loop_plan);
        }

        if decrementing_counter.is_none() {
            self.emit(
                add_imm(
                    DYNAMIC_INSTRUCTION_COUNT,
                    DYNAMIC_INSTRUCTION_COUNT,
                    loop_plan.guest_instruction_count as u16,
                ),
                format!(
                    "add x{DYNAMIC_INSTRUCTION_COUNT}, x{DYNAMIC_INSTRUCTION_COUNT}, #{}",
                    loop_plan.guest_instruction_count
                ),
            );
        }
        let lhs = loop_plan.host_or_zero(loop_plan.branch.rs1);
        let rhs = loop_plan.host_or_zero(loop_plan.branch.rs2);
        self.emit(cmp_reg(lhs, rhs), format!("cmp x{lhs}, x{rhs}"));
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, loop_plan.branch.condition),
            format!(
                "b.{} .jit_loop_start",
                loop_plan.branch.condition.mnemonic()
            ),
        );

        for (guest, host) in &loop_plan.dirty_registers {
            self.store_guest_register(*guest, *host);
        }
        self.store_pc(loop_plan.branch.fallthrough);
        if decrementing_counter.is_some() {
            self.mov_imm64(LOOP_TEMP, loop_plan.guest_instruction_count);
            self.emit(
                mul_reg(0, DYNAMIC_INSTRUCTION_COUNT, LOOP_TEMP),
                format!(
                    "mul x0, x{DYNAMIC_INSTRUCTION_COUNT}, x{LOOP_TEMP} ; return executed instructions"
                ),
            );
        } else {
            self.emit(
                mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
                format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return executed instructions"),
            );
        }
        self.emit_epilogue(true);
        self.ret();
    }

    fn emit_register_allocated_trace_loop(&mut self, trace_loop: &RegisterAllocatedTraceLoop<'_>) {
        self.emit_prologue(true);
        self.emit(
            mov_reg(DYNAMIC_INSTRUCTION_COUNT, A64_ZERO_REGISTER),
            format!("mov x{DYNAMIC_INSTRUCTION_COUNT}, xzr ; trace instruction count"),
        );

        for (guest, host) in &trace_loop.loaded_registers {
            self.load_guest_register(*host, *guest);
        }

        let loop_start = self.current_offset();
        for operation in trace_loop.operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            match instruction {
                NativeInstruction::TraceGuard {
                    rs1,
                    rs2,
                    condition,
                    continue_on_taken,
                    side_exit_pc,
                    executed_instructions,
                    ..
                } => self.emit_regalloc_trace_guard(
                    rs1,
                    rs2,
                    condition,
                    continue_on_taken,
                    side_exit_pc,
                    executed_instructions,
                    trace_loop,
                ),
                _ => self.emit_trace_loop_instruction(instruction, trace_loop),
            }
        }

        self.emit_host_add_sub_imm(
            DYNAMIC_INSTRUCTION_COUNT,
            DYNAMIC_INSTRUCTION_COUNT,
            trace_loop.loop_guard.guest_instruction_count as i64,
        );
        let lhs = trace_loop.host_or_zero(trace_loop.loop_guard.rs1);
        let rhs = trace_loop.host_or_zero(trace_loop.loop_guard.rs2);
        self.emit(
            cmp_reg(lhs, rhs),
            format!("cmp x{lhs}, x{rhs} ; regalloc trace loop guard"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, trace_loop.loop_guard.condition),
            format!(
                "b.{} .regalloc_trace_loop_start",
                trace_loop.loop_guard.condition.mnemonic()
            ),
        );
        self.flush_regalloc_dirty_registers(&trace_loop.dirty_registers);
        self.store_pc(trace_loop.loop_guard.side_exit_pc);
        self.emit(
            mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
            format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return executed trace instructions"),
        );
        self.emit_epilogue(true);
        self.ret();
        self.emit_trace_side_exits();
    }

    fn emit_register_allocated_block(&mut self, block_plan: &RegisterAllocatedBlock<'_>) {
        self.emit_prologue(false);

        for (guest, host) in &block_plan.loaded_registers {
            self.load_guest_register(*host, *guest);
        }

        for operation in block_plan.operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            self.emit_regalloc_instruction(instruction, block_plan);
        }

        for (guest, host) in &block_plan.dirty_registers {
            self.store_guest_register(*guest, *host);
        }

        self.mov_imm64(0, block_plan.guest_instruction_count);
        self.emit_epilogue(false);
        self.ret();
    }

    fn emit_trace_loop_instruction(
        &mut self,
        instruction: NativeInstruction,
        trace_loop: &RegisterAllocatedTraceLoop<'_>,
    ) {
        let loop_plan = RegisterAllocatedLoop {
            branch: SelfLoopBranch {
                rs1: 0,
                rs2: 0,
                fallthrough: 0,
                condition: A64Cond::Eq,
            },
            operations: &[],
            guest_to_host: trace_loop.guest_to_host,
            loaded_registers: Vec::new(),
            dirty_registers: Vec::new(),
            guest_instruction_count: trace_loop.loop_guard.guest_instruction_count,
        };
        self.emit_loop_instruction(instruction, &loop_plan);
    }

    fn emit_regalloc_trace_guard(
        &mut self,
        rs1: u8,
        rs2: u8,
        condition: IntegerBranchCondition,
        continue_on_taken: bool,
        side_exit_pc: u64,
        executed_instructions: u64,
        trace_loop: &RegisterAllocatedTraceLoop<'_>,
    ) {
        let lhs = trace_loop.host_or_zero(rs1);
        let rhs = trace_loop.host_or_zero(rs2);
        let continue_condition = trace_continue_condition(condition, continue_on_taken);
        self.emit(
            cmp_reg(lhs, rhs),
            format!("cmp x{lhs}, x{rhs} ; regalloc trace guard"),
        );
        let side_exit_condition = invert_condition(continue_condition);
        let branch_offset = self.emit_patchable_branch(format!(
            "b.{} .regalloc_trace_side_exit ; trace_side_exit",
            side_exit_condition.mnemonic()
        ));
        self.trace_side_exits.push(TraceSideExitPatch {
            branch_offset,
            condition: side_exit_condition,
            side_exit_pc,
            executed_instructions,
            dynamic_base: true,
            dirty_registers: trace_loop.dirty_registers.clone(),
        });
    }

    fn flush_regalloc_dirty_registers(&mut self, dirty_registers: &[(u8, u8)]) {
        for (guest, host) in dirty_registers {
            self.store_guest_register(*guest, *host);
        }
    }

    fn emit_regalloc_instruction(
        &mut self,
        instruction: NativeInstruction,
        block_plan: &RegisterAllocatedBlock<'_>,
    ) {
        match instruction {
            NativeInstruction::Beq {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_regalloc_conditional_branch(
                rs1,
                rs2,
                target,
                fallthrough,
                A64Cond::Eq,
                block_plan,
            ),
            NativeInstruction::Bge {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_regalloc_conditional_branch(
                rs1,
                rs2,
                target,
                fallthrough,
                A64Cond::Ge,
                block_plan,
            ),
            NativeInstruction::Bgeu {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_regalloc_conditional_branch(
                rs1,
                rs2,
                target,
                fallthrough,
                A64Cond::Hs,
                block_plan,
            ),
            NativeInstruction::Blt {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_regalloc_conditional_branch(
                rs1,
                rs2,
                target,
                fallthrough,
                A64Cond::Lt,
                block_plan,
            ),
            NativeInstruction::Bltu {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_regalloc_conditional_branch(
                rs1,
                rs2,
                target,
                fallthrough,
                A64Cond::Lo,
                block_plan,
            ),
            NativeInstruction::Bne {
                rs1,
                rs2,
                target,
                fallthrough,
            } => self.emit_regalloc_conditional_branch(
                rs1,
                rs2,
                target,
                fallthrough,
                A64Cond::Ne,
                block_plan,
            ),
            NativeInstruction::AndBranch {
                rd,
                rs1,
                imm,
                target,
                fallthrough,
                branch_if_zero,
            } => self.emit_regalloc_masked_zero_branch(
                rd,
                rs1,
                imm,
                target,
                fallthrough,
                branch_if_zero,
                block_plan,
            ),
            NativeInstruction::Jal {
                rd,
                target,
                return_pc,
            } => {
                if let Some(dst) = block_plan.host_for_write(rd) {
                    self.mov_imm64(dst, return_pc);
                }
                self.store_pc_immediate_regalloc(target);
            }
            NativeInstruction::Jalr {
                rd,
                rs1,
                imm,
                return_pc,
            } => {
                if rs1 == 0 {
                    self.mov_imm64(LOOP_TEMP, imm as u64);
                } else {
                    let src = block_plan.host_or_zero(rs1);
                    self.emit_host_add_sub_imm_any(LOOP_TEMP, src, imm);
                }
                self.mov_imm64(0, !1);
                self.emit(
                    and_reg(LOOP_TEMP, LOOP_TEMP, 0),
                    format!("and x{LOOP_TEMP}, x{LOOP_TEMP}, x0 ; clear jalr bit 0"),
                );
                if let Some(dst) = block_plan.host_for_write(rd) {
                    self.mov_imm64(dst, return_pc);
                }
                self.store_guest_register(PC_REGISTER, LOOP_TEMP);
            }
            NativeInstruction::InlinedJump { .. } => {}
            NativeInstruction::Jump { target } => self.store_pc_immediate_regalloc(target),
            NativeInstruction::JumpReg { rs1 } => {
                let target = block_plan.host_or_zero(rs1);
                self.store_guest_register(PC_REGISTER, target);
            }
            NativeInstruction::JumpRegLink { rs1, return_pc } => {
                let target = block_plan.host_or_zero(rs1);
                self.emit(
                    mov_reg(LOOP_TEMP, target),
                    format!("mov x{LOOP_TEMP}, x{target} ; preserve jalr target"),
                );
                if let Some(dst) = block_plan.host_for_write(1) {
                    self.mov_imm64(dst, return_pc);
                }
                self.store_guest_register(PC_REGISTER, LOOP_TEMP);
            }
            other => self.emit_loop_instruction_for_block(other, block_plan),
        }
    }

    fn emit_loop_instruction_for_block(
        &mut self,
        instruction: NativeInstruction,
        block_plan: &RegisterAllocatedBlock<'_>,
    ) {
        let loop_plan = RegisterAllocatedLoop {
            branch: SelfLoopBranch {
                rs1: 0,
                rs2: 0,
                fallthrough: 0,
                condition: A64Cond::Eq,
            },
            operations: &[],
            guest_to_host: block_plan.guest_to_host,
            loaded_registers: Vec::new(),
            dirty_registers: Vec::new(),
            guest_instruction_count: block_plan.guest_instruction_count,
        };
        self.emit_loop_instruction(instruction, &loop_plan);
    }

    fn emit_regalloc_conditional_branch(
        &mut self,
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
        condition: A64Cond,
        block_plan: &RegisterAllocatedBlock<'_>,
    ) {
        let lhs = block_plan.host_or_zero(rs1);
        let rhs = block_plan.host_or_zero(rs2);
        self.mov_imm64(LOOP_TEMP, target);
        self.mov_imm64(0, fallthrough);
        self.emit(cmp_reg(lhs, rhs), format!("cmp x{lhs}, x{rhs} ; regalloc"));
        self.emit(
            csel(LOOP_TEMP, LOOP_TEMP, 0, condition),
            format!(
                "csel x{LOOP_TEMP}, x{LOOP_TEMP}, x0, {} ; regalloc pc",
                condition.mnemonic()
            ),
        );
        self.store_guest_register(PC_REGISTER, LOOP_TEMP);
    }

    fn emit_regalloc_masked_zero_branch(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        target: u64,
        fallthrough: u64,
        branch_if_zero: bool,
        block_plan: &RegisterAllocatedBlock<'_>,
    ) {
        let src = block_plan.host_or_zero(rs1);
        let dst = block_plan.host_for_write(rd).unwrap_or(LOOP_TEMP);
        self.mov_imm64(LOOP_TEMP, imm as u64);
        self.emit(
            ands_reg(dst, src, LOOP_TEMP),
            format!("ands x{dst}, x{src}, x{LOOP_TEMP} ; regalloc masked branch"),
        );
        self.mov_imm64(LOOP_TEMP, target);
        self.mov_imm64(0, fallthrough);
        let condition = if branch_if_zero {
            A64Cond::Eq
        } else {
            A64Cond::Ne
        };
        self.emit(
            csel(LOOP_TEMP, LOOP_TEMP, 0, condition),
            format!(
                "csel x{LOOP_TEMP}, x{LOOP_TEMP}, x0, {} ; regalloc masked pc",
                condition.mnemonic()
            ),
        );
        self.store_guest_register(PC_REGISTER, LOOP_TEMP);
    }

    fn store_pc_immediate_regalloc(&mut self, pc: u64) {
        self.mov_imm64(LOOP_TEMP, pc);
        self.store_guest_register(PC_REGISTER, LOOP_TEMP);
    }

    fn emit_loop_instruction(
        &mut self,
        instruction: NativeInstruction,
        loop_plan: &RegisterAllocatedLoop<'_>,
    ) {
        match instruction {
            NativeInstruction::Add { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "add", add_reg)
            }
            NativeInstruction::Addi { rd, rs1, imm } => {
                self.emit_loop_add_sub_imm(rd, rs1, imm, loop_plan)
            }
            NativeInstruction::Addiw { rd, rs1, imm } => {
                self.emit_loop_add_sub_imm32(rd, rs1, imm, loop_plan)
            }
            NativeInstruction::Addw { rd, rs1, rs2 } => {
                self.emit_loop_word_binary_reg(rd, rs1, rs2, loop_plan, "addw", add_reg32)
            }
            NativeInstruction::And { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "and", and_reg)
            }
            NativeInstruction::Andi { rd, rs1, imm } => {
                self.emit_loop_logical_imm(rd, rs1, imm, loop_plan, "and", and_reg)
            }
            NativeInstruction::Auipc { rd, value }
            | NativeInstruction::Lui { rd, value }
            | NativeInstruction::LoadImmediate { rd, value } => {
                if let Some(dst) = loop_plan.host_for_write(rd) {
                    self.mov_imm64(dst, value);
                }
            }
            NativeInstruction::Move { rd, rs } => {
                if let Some(dst) = loop_plan.host_for_write(rd) {
                    let src = loop_plan.host_or_zero(rs);
                    if dst != src {
                        self.emit(mov_reg(dst, src), format!("mv x{rd}, x{rs} ; regalloc"));
                    }
                }
            }
            NativeInstruction::Load {
                rd,
                rs1,
                imm,
                width,
                signed,
            } => self.emit_loop_load(rd, rs1, imm, width.bytes(), signed, loop_plan),
            NativeInstruction::Mul { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "mul", mul_reg)
            }
            NativeInstruction::Mulh { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "smulh", smulh_reg)
            }
            NativeInstruction::Mulhu { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "umulh", umulh_reg)
            }
            NativeInstruction::Mulw { rd, rs1, rs2 } => {
                self.emit_loop_word_binary_reg(rd, rs1, rs2, loop_plan, "mulw", mul_reg32)
            }
            NativeInstruction::InlinedJump { .. } | NativeInstruction::Nop => {}
            NativeInstruction::Or { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "orr", orr_reg)
            }
            NativeInstruction::Ori { rd, rs1, imm } => {
                self.emit_loop_logical_imm(rd, rs1, imm, loop_plan, "orr", orr_reg)
            }
            NativeInstruction::Sll { rd, rs1, rs2 } => {
                self.emit_loop_shift_reg(rd, rs1, rs2, loop_plan, "lslv", lslv_reg)
            }
            NativeInstruction::Slli { rd, rs1, shamt } => {
                self.emit_loop_shift_imm(rd, rs1, shamt, loop_plan, "lslv", lslv_reg)
            }
            NativeInstruction::Slliw { rd, rs1, shamt } => {
                self.emit_loop_shift_imm32(rd, rs1, shamt, loop_plan, "slliw", lslv_reg32)
            }
            NativeInstruction::Sllw { rd, rs1, rs2 } => {
                self.emit_loop_word_binary_reg(rd, rs1, rs2, loop_plan, "sllw", lslv_reg32)
            }
            NativeInstruction::Slt { rd, rs1, rs2 } => {
                self.emit_loop_compare_set_reg(rd, rs1, rs2, loop_plan, A64Cond::Lt, "slt")
            }
            NativeInstruction::Slti { rd, rs1, imm } => {
                self.emit_loop_compare_set_imm(rd, rs1, imm, loop_plan, A64Cond::Lt, "slti")
            }
            NativeInstruction::Sltiu { rd, rs1, imm } => {
                self.emit_loop_compare_set_imm(rd, rs1, imm, loop_plan, A64Cond::Lo, "sltiu")
            }
            NativeInstruction::Sltu { rd, rs1, rs2 } => {
                self.emit_loop_compare_set_reg(rd, rs1, rs2, loop_plan, A64Cond::Lo, "sltu")
            }
            NativeInstruction::Sra { rd, rs1, rs2 } => {
                self.emit_loop_shift_reg(rd, rs1, rs2, loop_plan, "asrv", asrv_reg)
            }
            NativeInstruction::Srai { rd, rs1, shamt } => {
                self.emit_loop_shift_imm(rd, rs1, shamt, loop_plan, "asrv", asrv_reg)
            }
            NativeInstruction::Sraiw { rd, rs1, shamt } => {
                self.emit_loop_shift_imm32(rd, rs1, shamt, loop_plan, "sraiw", asrv_reg32)
            }
            NativeInstruction::Sraw { rd, rs1, rs2 } => {
                self.emit_loop_word_binary_reg(rd, rs1, rs2, loop_plan, "sraw", asrv_reg32)
            }
            NativeInstruction::Srl { rd, rs1, rs2 } => {
                self.emit_loop_shift_reg(rd, rs1, rs2, loop_plan, "lsrv", lsrv_reg)
            }
            NativeInstruction::Srli { rd, rs1, shamt } => {
                self.emit_loop_shift_imm(rd, rs1, shamt, loop_plan, "lsrv", lsrv_reg)
            }
            NativeInstruction::Srliw { rd, rs1, shamt } => {
                self.emit_loop_shift_imm32(rd, rs1, shamt, loop_plan, "srliw", lsrv_reg32)
            }
            NativeInstruction::Srlw { rd, rs1, rs2 } => {
                self.emit_loop_word_binary_reg(rd, rs1, rs2, loop_plan, "srlw", lsrv_reg32)
            }
            NativeInstruction::Sub { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "sub", sub_reg)
            }
            NativeInstruction::Subw { rd, rs1, rs2 } => {
                self.emit_loop_word_binary_reg(rd, rs1, rs2, loop_plan, "subw", sub_reg32)
            }
            NativeInstruction::Store {
                rs1,
                rs2,
                imm,
                width,
            } => self.emit_loop_store(rs1, rs2, imm, width.bytes(), loop_plan),
            NativeInstruction::Xor { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "eor", eor_reg)
            }
            NativeInstruction::Xori { rd, rs1, imm } => {
                self.emit_loop_logical_imm(rd, rs1, imm, loop_plan, "eor", eor_reg)
            }
            _ => debug_assert!(
                false,
                "register allocated loop accepted an unsupported instruction"
            ),
        }
    }

    fn emit_prologue(&mut self, preserve_dynamic_count: bool) {
        self.emit(
            stp_pre(29, 30, 31, -16),
            "stp x29, x30, [sp, #-16]!".to_string(),
        );
        self.emit(
            stp_pre(CPU_PTR, REG_PTR, 31, -16),
            format!("stp x{CPU_PTR}, x{REG_PTR}, [sp, #-16]!"),
        );
        if preserve_dynamic_count {
            self.emit(
                stp_pre(DYNAMIC_INSTRUCTION_COUNT, 22, 31, -16),
                format!("stp x{DYNAMIC_INSTRUCTION_COUNT}, x22, [sp, #-16]!"),
            );
            self.emit(
                stp_pre(23, 24, 31, -16),
                "stp x23, x24, [sp, #-16]!".to_string(),
            );
            self.emit(
                stp_pre(25, 26, 31, -16),
                "stp x25, x26, [sp, #-16]!".to_string(),
            );
            self.emit(
                stp_pre(27, 28, 31, -16),
                "stp x27, x28, [sp, #-16]!".to_string(),
            );
        }
        self.emit(add_imm(29, 31, 0), "mov x29, sp".to_string());
        self.emit(mov_reg(CPU_PTR, 0), format!("mov x{CPU_PTR}, x0"));
        self.emit(mov_reg(REG_PTR, 1), format!("mov x{REG_PTR}, x1"));
    }

    fn emit_epilogue(&mut self, restore_dynamic_count: bool) {
        if restore_dynamic_count {
            self.emit(
                ldp_post(27, 28, 31, 16),
                "ldp x27, x28, [sp], #16".to_string(),
            );
            self.emit(
                ldp_post(25, 26, 31, 16),
                "ldp x25, x26, [sp], #16".to_string(),
            );
            self.emit(
                ldp_post(23, 24, 31, 16),
                "ldp x23, x24, [sp], #16".to_string(),
            );
            self.emit(
                ldp_post(DYNAMIC_INSTRUCTION_COUNT, 22, 31, 16),
                format!("ldp x{DYNAMIC_INSTRUCTION_COUNT}, x22, [sp], #16"),
            );
        }
        self.emit(
            ldp_post(CPU_PTR, REG_PTR, 31, 16),
            format!("ldp x{CPU_PTR}, x{REG_PTR}, [sp], #16"),
        );
        self.emit(
            ldp_post(29, 30, 31, 16),
            "ldp x29, x30, [sp], #16".to_string(),
        );
    }

    fn store_pc(&mut self, pc: u64) {
        self.mov_imm64(SCRATCH0, pc);
        self.store_guest_register(PC_REGISTER, SCRATCH0);
    }

    fn emit_counted_self_loop_branch(
        &mut self,
        branch: SelfLoopBranch,
        loop_start: usize,
        guest_instruction_count: u64,
    ) {
        self.emit(
            add_imm(
                DYNAMIC_INSTRUCTION_COUNT,
                DYNAMIC_INSTRUCTION_COUNT,
                guest_instruction_count as u16,
            ),
            format!(
                "add x{DYNAMIC_INSTRUCTION_COUNT}, x{DYNAMIC_INSTRUCTION_COUNT}, #{guest_instruction_count}"
            ),
        );
        let lhs = self.load_guest_register_or_zero(SCRATCH0, branch.rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, branch.rs2);
        self.emit(cmp_reg(lhs, rhs), format!("cmp x{lhs}, x{rhs}"));
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, branch.condition),
            format!("b.{} .jit_loop_start", branch.condition.mnemonic()),
        );
        self.store_pc(branch.fallthrough);
    }

    fn emit_jal(&mut self, rd: u8, target: u64, return_pc: u64) {
        self.mov_imm64(SCRATCH0, return_pc);
        self.store_guest_register(rd, SCRATCH0);
        self.store_pc(target);
    }

    fn emit_jalr(&mut self, rd: u8, rs1: u8, imm: i64, return_pc: u64) {
        self.load_guest_register(SCRATCH0, rs1);
        self.emit_host_add_sub_imm(SCRATCH0, SCRATCH0, imm);
        self.mov_imm64(SCRATCH1, !1);
        self.emit(
            and_reg(SCRATCH0, SCRATCH0, SCRATCH1),
            format!("and x{SCRATCH0}, x{SCRATCH0}, x{SCRATCH1} ; clear jalr bit 0"),
        );
        self.mov_imm64(SCRATCH1, return_pc);
        self.store_guest_register(rd, SCRATCH1);
        self.store_guest_register(PC_REGISTER, SCRATCH0);
    }

    fn emit_jump_reg(&mut self, rs1: u8) {
        self.load_guest_register(SCRATCH0, rs1);
        self.store_guest_register(PC_REGISTER, SCRATCH0);
    }

    fn emit_jump_reg_link(&mut self, rs1: u8, return_pc: u64) {
        self.load_guest_register(SCRATCH0, rs1);
        self.mov_imm64(SCRATCH1, return_pc);
        self.store_guest_register(1, SCRATCH1);
        self.store_guest_register(PC_REGISTER, SCRATCH0);
    }

    fn emit_ecall(&mut self, next_pc: u64) {
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; ecall cpu"),
        );
        self.emit_call(
            jit_runtime_ecall as *const () as usize as u64,
            "jit_runtime_ecall",
        );
        self.store_pc(next_pc);
    }

    fn emit_conditional_branch(
        &mut self,
        rs1: u8,
        rs2: u8,
        target: u64,
        fallthrough: u64,
        condition: A64Cond,
    ) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.mov_imm64(SCRATCH2, target);
        self.mov_imm64(SCRATCH3, fallthrough);
        self.emit(cmp_reg(lhs, rhs), format!("cmp x{lhs}, x{rhs}"));
        self.emit(
            csel(SCRATCH0, SCRATCH2, SCRATCH3, condition),
            format!(
                "csel x{SCRATCH0}, x{SCRATCH2}, x{SCRATCH3}, {}",
                condition.mnemonic()
            ),
        );
        self.store_guest_register(PC_REGISTER, SCRATCH0);
    }

    fn emit_trace_guard(
        &mut self,
        rs1: u8,
        rs2: u8,
        condition: IntegerBranchCondition,
        continue_on_taken: bool,
        side_exit_pc: u64,
        executed_instructions: u64,
        dynamic_base: bool,
    ) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        let continue_condition = trace_continue_condition(condition, continue_on_taken);
        self.emit(
            cmp_reg(lhs, rhs),
            format!("cmp x{lhs}, x{rhs} ; trace guard"),
        );

        let side_exit_condition = invert_condition(continue_condition);
        let branch_offset = self.emit_patchable_branch(format!(
            "b.{} .trace_side_exit",
            side_exit_condition.mnemonic()
        ));
        self.trace_side_exits.push(TraceSideExitPatch {
            branch_offset,
            condition: side_exit_condition,
            side_exit_pc,
            executed_instructions,
            dynamic_base,
            dirty_registers: Vec::new(),
        });
    }

    fn emit_trace_loop_guard(
        &mut self,
        rs1: u8,
        rs2: u8,
        condition: IntegerBranchCondition,
        continue_on_taken: bool,
        side_exit_pc: u64,
        guest_instruction_count: u64,
        loop_start: usize,
    ) {
        self.emit_host_add_sub_imm(
            DYNAMIC_INSTRUCTION_COUNT,
            DYNAMIC_INSTRUCTION_COUNT,
            guest_instruction_count as i64,
        );
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        let continue_condition = trace_continue_condition(condition, continue_on_taken);
        self.emit(
            cmp_reg(lhs, rhs),
            format!("cmp x{lhs}, x{rhs} ; trace loop guard"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, continue_condition),
            format!("b.{} .trace_loop_start", continue_condition.mnemonic()),
        );
        self.store_pc(side_exit_pc);
        self.emit(
            mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
            format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return executed trace instructions"),
        );
        self.emit_epilogue(true);
        self.ret();
    }

    fn emit_masked_zero_branch(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        target: u64,
        fallthrough: u64,
        branch_if_zero: bool,
    ) {
        self.load_guest_register(SCRATCH0, rs1);
        self.mov_imm64(SCRATCH1, imm as u64);
        self.emit(
            ands_reg(SCRATCH0, SCRATCH0, SCRATCH1),
            format!("ands x{SCRATCH0}, x{SCRATCH0}, x{SCRATCH1} ; masked branch"),
        );
        self.store_guest_register(rd, SCRATCH0);
        self.mov_imm64(SCRATCH2, target);
        self.mov_imm64(SCRATCH3, fallthrough);
        let condition = if branch_if_zero {
            A64Cond::Eq
        } else {
            A64Cond::Ne
        };
        self.emit(
            csel(SCRATCH0, SCRATCH2, SCRATCH3, condition),
            format!(
                "csel x{SCRATCH0}, x{SCRATCH2}, x{SCRATCH3}, {} ; masked pc",
                condition.mnemonic()
            ),
        );
        self.store_guest_register(PC_REGISTER, SCRATCH0);
    }

    fn emit_load(&mut self, rd: u8, rs1: u8, imm: i64, width: u64, signed: bool) {
        let (target, name) = jit_load_runtime(width, signed);
        self.load_guest_register(SCRATCH0, rs1);
        self.emit_host_add_sub_imm_or_move(SCRATCH0, SCRATCH0, imm);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; load cpu"),
        );
        self.emit(mov_reg(1, SCRATCH0), format!("mov x1, x{SCRATCH0} ; addr"));
        self.emit_call(target, name);
        self.store_guest_register(rd, 0);
    }

    fn emit_store(&mut self, rs1: u8, rs2: u8, imm: i64, width: u64) {
        let (target, name) = jit_store_runtime(width);
        self.load_guest_register(SCRATCH0, rs1);
        self.emit_host_add_sub_imm_or_move(SCRATCH0, SCRATCH0, imm);
        let value = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; store cpu"),
        );
        self.emit(mov_reg(1, SCRATCH0), format!("mov x1, x{SCRATCH0} ; addr"));
        self.emit(mov_reg(2, value), format!("mov x2, x{value} ; value"));
        self.emit_call(target, name);
    }

    fn emit_float_load(&mut self, rd: u8, rs1: u8, imm: i64, width: u64) {
        self.load_guest_register(SCRATCH0, rs1);
        self.emit_host_add_sub_imm(SCRATCH0, SCRATCH0, imm);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; float load cpu"),
        );
        self.mov_imm64(1, u64::from(rd));
        self.emit(mov_reg(2, SCRATCH0), format!("mov x2, x{SCRATCH0} ; addr"));
        self.mov_imm64(3, width);
        self.emit_call(
            jit_runtime_float_load as *const () as usize as u64,
            "jit_runtime_float_load",
        );
    }

    fn emit_float_store(&mut self, rs1: u8, rs2: u8, imm: i64, width: u64) {
        self.load_guest_register(SCRATCH0, rs1);
        self.emit_host_add_sub_imm(SCRATCH0, SCRATCH0, imm);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; float store cpu"),
        );
        self.mov_imm64(1, u64::from(rs2));
        self.emit(mov_reg(2, SCRATCH0), format!("mov x2, x{SCRATCH0} ; addr"));
        self.mov_imm64(3, width);
        self.emit_call(
            jit_runtime_float_store as *const () as usize as u64,
            "jit_runtime_float_store",
        );
    }

    fn emit_runtime_binary(&mut self, rd: u8, rs1: u8, rs2: u8, op: u64) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.mov_imm64(0, op);
        self.emit(mov_reg(1, lhs), format!("mov x1, x{lhs} ; lhs"));
        self.emit(mov_reg(2, rhs), format!("mov x2, x{rhs} ; rhs"));
        self.emit_call(
            jit_runtime_binary as *const () as usize as u64,
            "jit_runtime_binary",
        );
        self.store_guest_register(rd, 0);
    }

    fn emit_runtime_atomic(&mut self, rd: u8, rs1: u8, rs2: u8, op: u64) {
        self.load_guest_register(SCRATCH0, rs1);
        let value = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; atomic cpu"),
        );
        self.mov_imm64(1, op);
        self.emit(mov_reg(2, SCRATCH0), format!("mov x2, x{SCRATCH0} ; addr"));
        self.emit(mov_reg(3, value), format!("mov x3, x{value} ; value"));
        self.emit_call(
            jit_runtime_atomic as *const () as usize as u64,
            "jit_runtime_atomic",
        );
        self.store_guest_register(rd, 0);
    }

    fn emit_runtime_csr(&mut self, rd: u8, rs1_or_uimm: u8, csr: u16, op: RuntimeCsrOp) {
        self.emit(mov_reg(0, CPU_PTR), format!("mov x0, x{CPU_PTR} ; csr cpu"));
        self.mov_imm64(1, op as u64);
        self.mov_imm64(2, u64::from(rd));
        self.mov_imm64(3, u64::from(rs1_or_uimm));
        self.mov_imm64(4, u64::from(csr));
        self.emit_call(
            jit_runtime_csr as *const () as usize as u64,
            "jit_runtime_csr",
        );
    }

    fn emit_runtime_trap(&mut self, pc: u64, opcode: u32, op: u64) {
        self.store_pc(pc);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; trap cpu"),
        );
        self.mov_imm64(1, op);
        self.mov_imm64(2, pc);
        self.mov_imm64(3, u64::from(opcode));
        self.emit_call(
            jit_runtime_trap as *const () as usize as u64,
            "jit_runtime_trap",
        );
    }

    fn emit_runtime_float(&mut self, rd: u8, rm: u8, rs1: u8, rs2: u8, rs3: u8, op: u64) {
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; float op cpu"),
        );
        self.mov_imm64(1, op);
        self.mov_imm64(2, u64::from(rd));
        self.mov_imm64(3, u64::from(rm));
        self.mov_imm64(4, u64::from(rs1));
        self.mov_imm64(5, u64::from(rs2));
        self.mov_imm64(6, u64::from(rs3));
        self.emit_call(
            jit_runtime_float_op as *const () as usize as u64,
            "jit_runtime_float_op",
        );
    }

    fn emit_loop_binary_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        loop_plan: &RegisterAllocatedLoop<'_>,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let lhs = loop_plan.host_or_zero(rs1);
        let rhs = loop_plan.host_or_zero(rs2);
        self.emit(
            op(dst, lhs, rhs),
            format!("{mnemonic} x{dst}, x{lhs}, x{rhs} ; regalloc"),
        );
    }

    fn emit_loop_word_binary_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        loop_plan: &RegisterAllocatedLoop<'_>,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let lhs = loop_plan.host_or_zero(rs1);
        let rhs = loop_plan.host_or_zero(rs2);
        self.emit(
            op(dst, lhs, rhs),
            format!("{mnemonic} w{dst}, w{lhs}, w{rhs} ; regalloc"),
        );
        self.sign_extend_word(dst, dst);
    }

    fn emit_loop_add_sub_imm(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        loop_plan: &RegisterAllocatedLoop<'_>,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        if rs1 == 0 {
            self.mov_imm64(dst, imm as u64);
            return;
        }

        let src = loop_plan.host_or_zero(rs1);
        self.emit_host_add_sub_imm_any(dst, src, imm);
    }

    fn emit_loop_add_sub_imm32(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        loop_plan: &RegisterAllocatedLoop<'_>,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        if rs1 == 0 {
            self.mov_imm64(dst, imm as u64);
        } else {
            let src = loop_plan.host_or_zero(rs1);
            self.emit_host_add_sub_imm32_any(dst, src, imm);
        }
        self.sign_extend_word(dst, dst);
    }

    fn emit_loop_logical_imm(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        loop_plan: &RegisterAllocatedLoop<'_>,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let src = loop_plan.host_or_zero(rs1);
        self.mov_imm64(LOOP_TEMP, imm as u64);
        self.emit(
            op(dst, src, LOOP_TEMP),
            format!("{mnemonic} x{dst}, x{src}, x{LOOP_TEMP} ; regalloc"),
        );
    }

    fn emit_loop_shift_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        loop_plan: &RegisterAllocatedLoop<'_>,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let lhs = loop_plan.host_or_zero(rs1);
        let rhs = loop_plan.host_or_zero(rs2);
        self.emit(
            op(dst, lhs, rhs),
            format!("{mnemonic} x{dst}, x{lhs}, x{rhs} ; regalloc"),
        );
    }

    fn emit_loop_shift_imm(
        &mut self,
        rd: u8,
        rs1: u8,
        shamt: u32,
        loop_plan: &RegisterAllocatedLoop<'_>,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let src = loop_plan.host_or_zero(rs1);
        self.mov_imm64(LOOP_TEMP, u64::from(shamt));
        self.emit(
            op(dst, src, LOOP_TEMP),
            format!("{mnemonic} x{dst}, x{src}, x{LOOP_TEMP} ; regalloc"),
        );
    }

    fn emit_loop_shift_imm32(
        &mut self,
        rd: u8,
        rs1: u8,
        shamt: u32,
        loop_plan: &RegisterAllocatedLoop<'_>,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let src = loop_plan.host_or_zero(rs1);
        self.mov_imm64(LOOP_TEMP, u64::from(shamt));
        self.emit(
            op(dst, src, LOOP_TEMP),
            format!("{mnemonic} w{dst}, w{src}, w{LOOP_TEMP} ; regalloc"),
        );
        self.sign_extend_word(dst, dst);
    }

    fn emit_loop_compare_set_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        loop_plan: &RegisterAllocatedLoop<'_>,
        condition: A64Cond,
        mnemonic: &str,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let lhs = loop_plan.host_or_zero(rs1);
        let rhs = loop_plan.host_or_zero(rs2);
        self.emit(
            cmp_reg(lhs, rhs),
            format!("cmp x{lhs}, x{rhs} ; {mnemonic} regalloc"),
        );
        self.mov_imm64(LOOP_TEMP, 1);
        self.emit(
            csel(dst, LOOP_TEMP, A64_ZERO_REGISTER, condition),
            format!("csel x{dst}, x{LOOP_TEMP}, xzr, {}", condition.mnemonic()),
        );
    }

    fn emit_loop_compare_set_imm(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        loop_plan: &RegisterAllocatedLoop<'_>,
        condition: A64Cond,
        mnemonic: &str,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let lhs = loop_plan.host_or_zero(rs1);
        self.mov_imm64(LOOP_TEMP, imm as u64);
        self.emit(
            cmp_reg(lhs, LOOP_TEMP),
            format!("cmp x{lhs}, x{LOOP_TEMP} ; {mnemonic} regalloc"),
        );
        self.mov_imm64(LOOP_TEMP, 1);
        self.emit(
            csel(dst, LOOP_TEMP, A64_ZERO_REGISTER, condition),
            format!("csel x{dst}, x{LOOP_TEMP}, xzr, {}", condition.mnemonic()),
        );
    }

    fn emit_loop_load(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        width: u64,
        signed: bool,
        loop_plan: &RegisterAllocatedLoop<'_>,
    ) {
        let (target, name) = jit_load_runtime(width, signed);
        let base = loop_plan.host_or_zero(rs1);
        self.emit_host_add_sub_imm_or_move(1, base, imm);
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; regalloc load cpu"),
        );
        self.emit_call(target, name);
        if let Some(dst) = loop_plan.host_for_write(rd) {
            self.emit(
                mov_reg(dst, 0),
                format!("mov x{dst}, x0 ; regalloc load x{rd}"),
            );
        }
    }

    fn emit_loop_store(
        &mut self,
        rs1: u8,
        rs2: u8,
        imm: i64,
        width: u64,
        loop_plan: &RegisterAllocatedLoop<'_>,
    ) {
        let (target, name) = jit_store_runtime(width);
        let base = loop_plan.host_or_zero(rs1);
        let value = loop_plan.host_or_zero(rs2);
        self.emit_host_add_sub_imm_or_move(1, base, imm);
        self.emit(
            mov_reg(2, value),
            format!("mov x2, x{value} ; regalloc store value"),
        );
        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; regalloc store cpu"),
        );
        self.emit_call(target, name);
    }

    fn emit_binary_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit(
            op(SCRATCH0, lhs, rhs),
            format!("{mnemonic} x{SCRATCH0}, x{lhs}, x{rhs}"),
        );
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_move(&mut self, rd: u8, rs: u8) {
        if rd == rs || rd == 0 {
            return;
        }
        self.load_guest_register(SCRATCH0, rs);
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_add_sub_imm(&mut self, rd: u8, rs1: u8, imm: i64) {
        if rs1 == 0 && imm >= 0 {
            self.mov_imm64(SCRATCH0, imm as u64);
            self.store_guest_register(rd, SCRATCH0);
            return;
        }

        self.load_guest_register(SCRATCH0, rs1);
        self.emit_host_add_sub_imm(SCRATCH0, SCRATCH0, imm);
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_add_sub_imm32(&mut self, rd: u8, rs1: u8, imm: i64) {
        if rs1 == 0 && imm >= 0 {
            self.mov_imm64(SCRATCH0, imm as u64);
            self.sign_extend_word(SCRATCH0, SCRATCH0);
            self.store_guest_register(rd, SCRATCH0);
            return;
        }

        self.load_guest_register(SCRATCH0, rs1);
        self.emit_host_add_sub_imm32(SCRATCH0, SCRATCH0, imm);
        self.sign_extend_word(SCRATCH0, SCRATCH0);
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_host_add_sub_imm(&mut self, rd: u8, rn: u8, imm: i64) {
        if imm >= 0 {
            self.emit(
                add_imm(rd, rn, imm as u16),
                format!("add x{rd}, x{rn}, #{}", imm),
            );
        } else {
            self.emit(
                sub_imm(rd, rn, (-imm) as u16),
                format!("sub x{rd}, x{rn}, #{}", -imm),
            );
        }
    }

    fn emit_host_add_sub_imm_or_move(&mut self, rd: u8, rn: u8, imm: i64) {
        if imm == 0 {
            if rd != rn {
                self.emit(mov_reg(rd, rn), format!("mov x{rd}, x{rn}"));
            }
            return;
        }

        self.emit_host_add_sub_imm(rd, rn, imm);
    }

    fn emit_host_add_sub_imm_any(&mut self, rd: u8, rn: u8, imm: i64) {
        if imm.unsigned_abs() < 4096 {
            self.emit_host_add_sub_imm_or_move(rd, rn, imm);
            return;
        }

        self.mov_imm64(LOOP_TEMP, imm.unsigned_abs());
        if imm >= 0 {
            self.emit(
                add_reg(rd, rn, LOOP_TEMP),
                format!("add x{rd}, x{rn}, x{LOOP_TEMP}"),
            );
        } else {
            self.emit(
                sub_reg(rd, rn, LOOP_TEMP),
                format!("sub x{rd}, x{rn}, x{LOOP_TEMP}"),
            );
        }
    }

    fn emit_host_add_sub_imm32(&mut self, rd: u8, rn: u8, imm: i64) {
        if imm >= 0 {
            self.emit(
                add_imm32(rd, rn, imm as u16),
                format!("add w{rd}, w{rn}, #{}", imm),
            );
        } else {
            self.emit(
                sub_imm32(rd, rn, (-imm) as u16),
                format!("sub w{rd}, w{rn}, #{}", -imm),
            );
        }
    }

    fn emit_host_add_sub_imm32_any(&mut self, rd: u8, rn: u8, imm: i64) {
        if imm.unsigned_abs() < 4096 {
            self.emit_host_add_sub_imm32(rd, rn, imm);
            return;
        }

        self.mov_imm64(LOOP_TEMP, imm.unsigned_abs());
        if imm >= 0 {
            self.emit(
                add_reg32(rd, rn, LOOP_TEMP),
                format!("add w{rd}, w{rn}, w{LOOP_TEMP}"),
            );
        } else {
            self.emit(
                sub_reg32(rd, rn, LOOP_TEMP),
                format!("sub w{rd}, w{rn}, w{LOOP_TEMP}"),
            );
        }
    }

    fn emit_logical_imm(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        self.load_guest_register(SCRATCH0, rs1);
        self.mov_imm64(SCRATCH1, imm as u64);
        self.emit(
            op(SCRATCH0, SCRATCH0, SCRATCH1),
            format!("{mnemonic} x{SCRATCH0}, x{SCRATCH0}, x{SCRATCH1}"),
        );
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_shift_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit(
            op(SCRATCH0, lhs, rhs),
            format!("{mnemonic} x{SCRATCH0}, x{lhs}, x{rhs}"),
        );
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_shift_imm(
        &mut self,
        rd: u8,
        rs1: u8,
        shamt: u32,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        self.load_guest_register(SCRATCH0, rs1);
        self.mov_imm64(SCRATCH1, u64::from(shamt));
        self.emit(
            op(SCRATCH0, SCRATCH0, SCRATCH1),
            format!("{mnemonic} x{SCRATCH0}, x{SCRATCH0}, x{SCRATCH1}"),
        );
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_word_binary_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit(
            op(SCRATCH0, lhs, rhs),
            format!("{mnemonic} w{SCRATCH0}, w{lhs}, w{rhs}"),
        );
        self.sign_extend_word(SCRATCH0, SCRATCH0);
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_shift_imm32(
        &mut self,
        rd: u8,
        rs1: u8,
        shamt: u32,
        mnemonic: &str,
        op: fn(u8, u8, u8) -> u32,
    ) {
        self.load_guest_register(SCRATCH0, rs1);
        self.mov_imm64(SCRATCH1, u64::from(shamt));
        self.emit(
            op(SCRATCH0, SCRATCH0, SCRATCH1),
            format!("{mnemonic} w{SCRATCH0}, w{SCRATCH0}, w{SCRATCH1}"),
        );
        self.sign_extend_word(SCRATCH0, SCRATCH0);
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_compare_set_reg(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        condition: A64Cond,
        mnemonic: &str,
    ) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit(
            cmp_reg(lhs, rhs),
            format!("cmp x{lhs}, x{rhs} ; {mnemonic}"),
        );
        self.emit_compare_result(rd, condition);
    }

    fn emit_compare_set_imm(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
        condition: A64Cond,
        mnemonic: &str,
    ) {
        self.load_guest_register(SCRATCH0, rs1);
        self.mov_imm64(SCRATCH1, imm as u64);
        self.emit(
            cmp_reg(SCRATCH0, SCRATCH1),
            format!("cmp x{SCRATCH0}, x{SCRATCH1} ; {mnemonic}"),
        );
        self.emit_compare_result(rd, condition);
    }

    fn emit_compare_result(&mut self, rd: u8, condition: A64Cond) {
        self.mov_imm64(SCRATCH1, 1);
        self.emit(
            csel(SCRATCH0, SCRATCH1, A64_ZERO_REGISTER, condition),
            format!(
                "csel x{SCRATCH0}, x{SCRATCH1}, xzr, {}",
                condition.mnemonic()
            ),
        );
        self.store_guest_register(rd, SCRATCH0);
    }

    fn sign_extend_word(&mut self, rd: u8, rn: u8) {
        self.emit(sxtw(rd, rn), format!("sxtw x{rd}, w{rn}"));
    }

    fn load_guest_register(&mut self, host: u8, guest: u8) {
        if guest == 0 {
            self.emit(
                mov_reg(host, A64_ZERO_REGISTER),
                format!("mov x{host}, xzr"),
            );
            return;
        }
        let offset = u16::from(guest) * 8;
        self.emit(
            ldr_u64(host, REG_PTR, offset),
            format!("ldr x{host}, [x{REG_PTR}, #{offset}] ; load guest x{guest}"),
        );
    }

    fn load_guest_register_or_zero(&mut self, host: u8, guest: u8) -> u8 {
        if guest == 0 {
            A64_ZERO_REGISTER
        } else {
            self.load_guest_register(host, guest);
            host
        }
    }

    fn store_guest_register(&mut self, guest: u8, host: u8) {
        if guest == 0 {
            return;
        }
        let offset = u16::from(guest) * 8;
        self.emit(
            str_u64(host, REG_PTR, offset),
            format!("str x{host}, [x{REG_PTR}, #{offset}] ; store guest x{guest}"),
        );
    }

    fn mov_imm64(&mut self, rd: u8, value: u64) {
        if value == 0 {
            self.emit(mov_reg(rd, A64_ZERO_REGISTER), format!("mov x{rd}, xzr"));
            return;
        }

        let mut emitted = false;
        for halfword in 0..4 {
            let imm = ((value >> (halfword * 16)) & 0xffff) as u16;
            if !emitted {
                self.emit(
                    movz(rd, imm, halfword),
                    format!("movz x{rd}, #0x{imm:04x}, lsl #{}", halfword * 16),
                );
                emitted = true;
            } else if imm != 0 {
                self.emit(
                    movk(rd, imm, halfword),
                    format!("movk x{rd}, #0x{imm:04x}, lsl #{}", halfword * 16),
                );
            }
        }
    }

    fn ret(&mut self) {
        self.emit(0xd65f_03c0, "ret".to_string());
    }

    fn emit_call(&mut self, target: u64, name: &str) {
        self.mov_imm64(CALL_TARGET, target);
        self.emit(blr(CALL_TARGET), format!("blr x{CALL_TARGET} ; {name}"));
    }

    fn emit_patchable_branch(&mut self, text: String) -> usize {
        let offset = self.code.len();
        self.code.extend(0u32.to_le_bytes());
        if self.include_listing {
            self.listing.push(NativeEmission {
                offset,
                word: 0,
                text,
            });
        }
        offset
    }

    fn patch_branch(&mut self, offset: usize, word: u32) {
        self.code[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
        if let Some(emission) = self
            .listing
            .iter_mut()
            .find(|emission| emission.offset == offset)
        {
            emission.word = word;
        }
    }

    fn emit(&mut self, word: u32, text: String) {
        let offset = self.code.len();
        self.code.extend(word.to_le_bytes());
        if self.include_listing {
            self.listing.push(NativeEmission { offset, word, text });
        }
    }
}

struct EmittedCode {
    bytes: Vec<u8>,
    listing: Vec<NativeEmission>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum A64Cond {
    Eq,
    Ne,
    Hs,
    Lo,
    Ge,
    Lt,
}

impl A64Cond {
    fn code(self) -> u8 {
        match self {
            Self::Eq => 0,
            Self::Ne => 1,
            Self::Hs => 2,
            Self::Lo => 3,
            Self::Ge => 10,
            Self::Lt => 11,
        }
    }

    fn mnemonic(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Hs => "hs",
            Self::Lo => "lo",
            Self::Ge => "ge",
            Self::Lt => "lt",
        }
    }
}

fn trace_continue_condition(condition: IntegerBranchCondition, continue_on_taken: bool) -> A64Cond {
    let taken = match condition {
        IntegerBranchCondition::Eq => A64Cond::Eq,
        IntegerBranchCondition::Ne => A64Cond::Ne,
        IntegerBranchCondition::Ge => A64Cond::Ge,
        IntegerBranchCondition::Geu => A64Cond::Hs,
        IntegerBranchCondition::Lt => A64Cond::Lt,
        IntegerBranchCondition::Ltu => A64Cond::Lo,
    };
    if continue_on_taken {
        taken
    } else {
        invert_condition(taken)
    }
}

fn invert_condition(condition: A64Cond) -> A64Cond {
    match condition {
        A64Cond::Eq => A64Cond::Ne,
        A64Cond::Ne => A64Cond::Eq,
        A64Cond::Hs => A64Cond::Lo,
        A64Cond::Lo => A64Cond::Hs,
        A64Cond::Ge => A64Cond::Lt,
        A64Cond::Lt => A64Cond::Ge,
    }
}

fn ldr_u64(rt: u8, rn: u8, offset: u16) -> u32 {
    debug_assert_eq!(offset % 8, 0);
    0xf940_0000 | (u32::from(offset / 8) << 10) | (u32::from(rn) << 5) | u32::from(rt)
}

fn str_u64(rt: u8, rn: u8, offset: u16) -> u32 {
    debug_assert_eq!(offset % 8, 0);
    0xf900_0000 | (u32::from(offset / 8) << 10) | (u32::from(rn) << 5) | u32::from(rt)
}

fn add_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x8b00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sub_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xcb00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn add_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x0b00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sub_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x4b00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn and_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x8a00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn ands_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xea00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn orr_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xaa00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn eor_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xca00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn mul_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9b00_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn mul_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1b00_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn smulh_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9b40_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn umulh_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9bc0_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn lslv_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9ac0_2000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn lsrv_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9ac0_2400 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn asrv_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9ac0_2800 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn lslv_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1ac0_2000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn lsrv_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1ac0_2400 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn asrv_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1ac0_2800 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn cmp_reg(rn: u8, rm: u8) -> u32 {
    0xeb00_001f | (u32::from(rm) << 16) | (u32::from(rn) << 5)
}

fn csel(rd: u8, rn: u8, rm: u8, cond: A64Cond) -> u32 {
    0x9a80_0000
        | (u32::from(rm) << 16)
        | (u32::from(cond.code()) << 12)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn b_cond(branch_offset: usize, target_offset: usize, cond: A64Cond) -> u32 {
    let byte_delta = target_offset as isize - branch_offset as isize;
    debug_assert_eq!(byte_delta % 4, 0);
    let instruction_delta = (byte_delta / 4) as i32;
    debug_assert!((-0x4_0000..0x4_0000).contains(&instruction_delta));
    0x5400_0000 | (((instruction_delta as u32) & 0x7ffff) << 5) | u32::from(cond.code())
}

fn blr(rn: u8) -> u32 {
    0xd63f_0000 | (u32::from(rn) << 5)
}

fn add_imm(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0x9100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sub_imm(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0xd100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn add_imm32(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0x1100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sub_imm32(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0x5100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sxtw(rd: u8, rn: u8) -> u32 {
    0x9340_7c00 | (u32::from(rn) << 5) | u32::from(rd)
}

fn movz(rd: u8, imm: u16, halfword: u64) -> u32 {
    0xd280_0000 | ((halfword as u32) << 21) | (u32::from(imm) << 5) | u32::from(rd)
}

fn movk(rd: u8, imm: u16, halfword: u64) -> u32 {
    0xf280_0000 | ((halfword as u32) << 21) | (u32::from(imm) << 5) | u32::from(rd)
}

fn mov_reg(rd: u8, rn: u8) -> u32 {
    orr_reg(rd, 31, rn)
}

fn stp_pre(rt: u8, rt2: u8, rn: u8, offset: i16) -> u32 {
    debug_assert_eq!(offset % 8, 0);
    let imm7 = ((offset / 8) as i32 & 0x7f) as u32;
    0xa980_0000 | (imm7 << 15) | (u32::from(rt2) << 10) | (u32::from(rn) << 5) | u32::from(rt)
}

fn ldp_post(rt: u8, rt2: u8, rn: u8, offset: i16) -> u32 {
    debug_assert_eq!(offset % 8, 0);
    let imm7 = ((offset / 8) as i32 & 0x7f) as u32;
    0xa8c0_0000 | (imm7 << 15) | (u32::from(rt2) << 10) | (u32::from(rn) << 5) | u32::from(rt)
}

struct ExecutableMemory {
    ptr: *mut u8,
    len: usize,
}

// The mapping is private to this owner and is never written after mprotect.
unsafe impl Send for ExecutableMemory {}

impl ExecutableMemory {
    fn new(code: &[u8]) -> Result<Self, JitError> {
        let page_size = 16 * 1024;
        let len = code.len().max(1).next_multiple_of(page_size);
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(JitError::AllocationFailed);
        }

        unsafe {
            ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len());
        }

        flush_instruction_cache(ptr, code.len());

        let rc = unsafe { mprotect(ptr, len, PROT_READ | PROT_EXEC) };
        if rc != 0 {
            unsafe {
                munmap(ptr, len);
            }
            return Err(JitError::PermissionFailed);
        }

        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.cast(), self.len);
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn flush_instruction_cache(ptr: *mut c_void, len: usize) {
    extern "C" {
        fn sys_icache_invalidate(start: *mut c_void, len: usize);
    }

    unsafe {
        sys_icache_invalidate(ptr, len);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn flush_instruction_cache(_ptr: *mut c_void, _len: usize) {}

const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const PROT_EXEC: i32 = 0x4;
const MAP_PRIVATE: i32 = 0x2;

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MAP_ANON: i32 = 0x1000;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
const MAP_ANON: i32 = 0x20;

const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: isize,
    ) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}
