use std::ffi::c_void;
use std::ptr;

use crate::cpu::RV64GC;

use super::{
    jit_runtime_atomic, jit_runtime_binary, jit_runtime_csr, jit_runtime_direct_write_ptr,
    jit_runtime_ecall, jit_runtime_float_load, jit_runtime_float_op, jit_runtime_float_store,
    jit_runtime_load_i16, jit_runtime_load_i32, jit_runtime_load_i8, jit_runtime_load_u16,
    jit_runtime_load_u32, jit_runtime_load_u64, jit_runtime_load_u8, jit_runtime_store_u16,
    jit_runtime_store_u32, jit_runtime_store_u64, jit_runtime_store_u8, jit_runtime_trap,
    jit_runtime_try_direct_read_ptr, ArithmeticXorToggleLoop, BlockOperationKind, BlockPlan,
    CompiledBlock, CountedDiamondLoop, DivisionRecurrenceLoop, FibonacciRecurrenceLoop,
    IntegerBranchCondition, JitError, JitTier, NativeEmission, NativeInstruction, RuntimeBinaryOp,
    RuntimeCsrOp, StoreLoadForwardLoop,
};

const CPU_PTR: u8 = 19;
const REG_PTR: u8 = 20;
const SCRATCH0: u8 = 9;
const SCRATCH1: u8 = 10;
const SCRATCH2: u8 = 11;
const SCRATCH3: u8 = 12;
const DIV_REM_CACHE: u8 = 13;
const CALL_TARGET: u8 = 16;
const DYNAMIC_INSTRUCTION_COUNT: u8 = 21;
const CALLER_LOOP_INSTRUCTION_COUNT: u8 = 10;
const DIVISION_RECURRENCE_INSTRUCTION_COUNT: u64 = 22;
const PC_REGISTER: u8 = 32;
const A64_ZERO_REGISTER: u8 = 31;
const LOOP_TEMP: u8 = 16;
const ARG_REG_PTR: u8 = 1;
const BLOCK_HOST_REGISTERS: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17];
const LOOP_HOST_REGISTERS: [u8; 7] = [22, 23, 24, 25, 26, 27, 28];
const LOOP_CALLER_HOST_REGISTERS: [u8; 7] = [2, 3, 4, 5, 6, 7, 8];
// Pure self-loops do not call out, so these caller-saved registers can hold
// invariant divisor masks without forcing the guest register bank into x22-x28.
const DIVISOR_MASK_HOST_REGISTERS: [u8; 5] = [9, 14, 15, 17, 0];
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

#[derive(Debug, Clone, Copy)]
struct ConstUnsignedDivisor {
    value: u64,
    shift: u32,
    multiplier: u64,
    add_indicator: bool,
    power_of_two_shift: Option<u32>,
}

impl ConstUnsignedDivisor {
    fn new(value: u64) -> Option<Self> {
        if value == 0 {
            return None;
        }
        if value.is_power_of_two() {
            return Some(Self {
                value,
                shift: value.trailing_zeros(),
                multiplier: 0,
                add_indicator: false,
                power_of_two_shift: Some(value.trailing_zeros()),
            });
        }

        let shift = u64::BITS - (value - 1).leading_zeros();
        if shift >= u64::BITS {
            return None;
        }
        let numerator = 1u128 << (u64::BITS + shift);
        let divisor = u128::from(value);
        let mut multiplier = (numerator + divisor - 1) / divisor;
        let add_indicator = multiplier >= (1u128 << u64::BITS);
        if add_indicator {
            multiplier -= 1u128 << u64::BITS;
        }

        Some(Self {
            value,
            shift,
            multiplier: multiplier as u64,
            add_indicator,
            power_of_two_shift: None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ConstSignedDivisor {
    value: u64,
    magic: i64,
    shift: u32,
}

impl ConstSignedDivisor {
    fn new_positive(value: u64) -> Option<Self> {
        if value == 0 || value > i64::MAX as u64 {
            return None;
        }
        if value == 1 {
            return Some(Self {
                value,
                magic: 1,
                shift: 0,
            });
        }

        let divisor = u128::from(value);
        let sign_bit = 1u128 << (u64::BITS - 1);
        let adjusted_nc = sign_bit - 1 - ((sign_bit - 1) % divisor);
        let mut p = u64::BITS - 1;
        let mut q1 = sign_bit / adjusted_nc;
        let mut r1 = sign_bit - q1 * adjusted_nc;
        let mut q2 = sign_bit / divisor;
        let mut r2 = sign_bit - q2 * divisor;

        loop {
            p += 1;
            q1 *= 2;
            r1 *= 2;
            if r1 >= adjusted_nc {
                q1 += 1;
                r1 -= adjusted_nc;
            }
            q2 *= 2;
            r2 *= 2;
            if r2 >= divisor {
                q2 += 1;
                r2 -= divisor;
            }

            let delta = divisor - r2;
            if q1 > delta || (q1 == delta && r1 != 0) {
                break;
            }
        }

        Some(Self {
            value,
            magic: (q2 + 1) as u64 as i64,
            shift: p - u64::BITS,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct DivisionRecurrenceReciprocalSpec {
    signed_divisor: u64,
    unsigned_divisor: u64,
    signed: ConstSignedDivisor,
    unsigned64: ConstUnsignedDivisor,
}

impl DivisionRecurrenceReciprocalSpec {
    fn from_region(region: DivisionRecurrenceLoop) -> Option<Self> {
        let signed_divisor = region.signed_divisor_value as i64;
        let signed_word_divisor = region.signed_divisor_value as u32 as i32;
        if signed_divisor <= 0 || signed_word_divisor <= 0 {
            return None;
        }
        let signed_divisor = signed_divisor as u64;
        if signed_divisor != signed_word_divisor as u64 {
            return None;
        }

        let unsigned_divisor = region.unsigned_divisor_value;
        let unsigned_word_divisor = region.unsigned_divisor_value as u32;
        if unsigned_divisor == 0 || unsigned_word_divisor == 0 {
            return None;
        }
        if unsigned_divisor != u64::from(unsigned_word_divisor) {
            return None;
        }

        Some(Self {
            signed_divisor,
            unsigned_divisor,
            signed: ConstSignedDivisor::new_positive(signed_divisor)?,
            unsigned64: ConstUnsignedDivisor::new(unsigned_divisor)?,
        })
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
                if let Some(region) = DirectLoadTraceLoop::from_trace_loop(&trace_loop) {
                    code.emit_direct_load_trace_loop(&trace_loop, region);
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

        if let Some(region) = division_recurrence_loop_region(plan) {
            code.emit_division_recurrence_loop(region);
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

fn counted_diamond_closed_form_mask_shift(mask: i64) -> Option<u32> {
    let mask = u64::try_from(mask).ok()?;
    let stride = mask.checked_add(1)?;
    if stride.is_power_of_two() && stride > 1 {
        Some(stride.trailing_zeros())
    } else {
        None
    }
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

fn division_recurrence_loop_region(plan: &BlockPlan) -> Option<DivisionRecurrenceLoop> {
    let [operation] = plan.operations.as_slice() else {
        return None;
    };
    let BlockOperationKind::Native(NativeInstruction::DivisionRecurrenceLoop(region)) =
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
        | NativeInstruction::DivisionRecurrenceLoop(_)
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

        let host_registers = if loop_can_use_caller_saved_registers(operations) {
            &LOOP_CALLER_HOST_REGISTERS
        } else {
            &LOOP_HOST_REGISTERS
        };

        if guest_registers.len() > host_registers.len() {
            return None;
        }

        let mut guest_to_host = [UNMAPPED_GUEST_REGISTER; 33];
        for (index, guest) in guest_registers.into_iter().enumerate() {
            let host = host_registers[index];
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

    fn final_decrementing_counter(&self) -> Option<(u8, bool)> {
        let counter = self.decrementing_counter()?;
        let final_operation = self.operations.last()?;
        let BlockOperationKind::Native(instruction) = final_operation.kind();
        match instruction {
            NativeInstruction::Addi { rd, rs1, imm }
                if rd == counter && rs1 == counter && imm == -1 =>
            {
                Some((counter, false))
            }
            NativeInstruction::Addiw { rd, rs1, imm }
                if rd == counter && rs1 == counter && imm == -1 =>
            {
                Some((counter, true))
            }
            _ => None,
        }
    }

    fn writes_guest_register(&self, guest: u8) -> bool {
        guest != 0
            && self
                .dirty_registers
                .iter()
                .any(|(dirty, _)| *dirty == guest)
    }

    fn calls_rust_runtime(&self) -> bool {
        self.operations.iter().any(|operation| {
            let BlockOperationKind::Native(instruction) = operation.kind();
            loop_instruction_calls_rust_runtime(instruction)
        })
    }

    fn uses_caller_saved_guest_registers(&self) -> bool {
        self.guest_to_host
            .iter()
            .any(|host| LOOP_CALLER_HOST_REGISTERS.contains(host))
    }

    fn used_callee_saved_registers(&self, instruction_count_host: u8) -> Vec<u8> {
        let mut registers = Vec::with_capacity(1 + LOOP_HOST_REGISTERS.len());
        if loop_host_register_is_callee_saved(instruction_count_host) {
            registers.push(instruction_count_host);
        }
        for host in self.guest_to_host {
            if loop_host_register_is_callee_saved(host) && !registers.contains(&host) {
                registers.push(host);
            }
        }
        registers.sort_unstable();
        registers
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

#[derive(Clone, Copy)]
struct DirectLoadTraceLoop {
    loaded_register: u8,
    address_register: u8,
    counter_register: u8,
    limit_register: u8,
    width_bytes: u64,
    guard_side_exit_pc: u64,
    guard_executed_instructions: u64,
    loop_exit_pc: u64,
    guest_instruction_count: u64,
}

impl DirectLoadTraceLoop {
    fn from_trace_loop(trace_loop: &RegisterAllocatedTraceLoop<'_>) -> Option<Self> {
        if trace_loop.operations.len() != 4 {
            return None;
        }

        let BlockOperationKind::Native(NativeInstruction::Load {
            rd: loaded_register,
            rs1: address_register,
            imm: load_imm,
            width,
            signed,
        }) = trace_loop.operations[0].kind()
        else {
            return None;
        };
        if loaded_register == 0 || address_register == 0 || load_imm != 0 || signed {
            return None;
        }

        let BlockOperationKind::Native(NativeInstruction::TraceGuard {
            rs1,
            rs2,
            condition,
            continue_on_taken,
            side_exit_pc: guard_side_exit_pc,
            executed_instructions: guard_executed_instructions,
            ..
        }) = trace_loop.operations[1].kind()
        else {
            return None;
        };
        if rs1 != loaded_register
            || rs2 != 0
            || condition != IntegerBranchCondition::Eq
            || continue_on_taken
        {
            return None;
        }

        let BlockOperationKind::Native(NativeInstruction::Addi {
            rd: counter_register,
            rs1: counter_source,
            imm: counter_delta,
        }) = trace_loop.operations[2].kind()
        else {
            return None;
        };
        if counter_register == 0 || counter_source != counter_register || counter_delta != 1 {
            return None;
        }

        let BlockOperationKind::Native(NativeInstruction::Addi {
            rd: address_dest,
            rs1: address_source,
            imm: address_delta,
        }) = trace_loop.operations[3].kind()
        else {
            return None;
        };
        let width_bytes = width.bytes();
        if address_dest != address_register
            || address_source != address_register
            || address_delta != width_bytes as i64
        {
            return None;
        }

        if trace_loop.loop_guard.rs1 != counter_register
            || trace_loop.loop_guard.rs2 == 0
            || trace_loop.loop_guard.condition != A64Cond::Ne
        {
            return None;
        }

        let reserved_hosts = [26, 27, 28];
        for guest in [
            loaded_register,
            address_register,
            counter_register,
            trace_loop.loop_guard.rs2,
        ] {
            if reserved_hosts.contains(&trace_loop.host_or_zero(guest)) {
                return None;
            }
        }

        Some(Self {
            loaded_register,
            address_register,
            counter_register,
            limit_register: trace_loop.loop_guard.rs2,
            width_bytes,
            guard_side_exit_pc,
            guard_executed_instructions,
            loop_exit_pc: trace_loop.loop_guard.side_exit_pc,
            guest_instruction_count: trace_loop.loop_guard.guest_instruction_count,
        })
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
        | NativeInstruction::FloatStore { rs1, rs2, .. }
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
        NativeInstruction::Store { rs1, rs2, .. } => {
            push(rs1);
            push(rs2);
        }
        NativeInstruction::Auipc { .. }
        | NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::DivisionRecurrenceLoop(_)
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
        | NativeInstruction::DivisionRecurrenceLoop(_)
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
        NativeInstruction::RuntimeBinary { .. } => None,
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
        | NativeInstruction::DivisionRecurrenceLoop(_)
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
        NativeInstruction::RuntimeBinary { rd, rs1, rs2, op }
            if runtime_binary_op_has_native_aarch64_lowering(op) =>
        {
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
                self.emit_logical_imm(rd, rs1, imm, LogicalImmediateOp::And)
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
            NativeInstruction::DivisionRecurrenceLoop(_) => debug_assert!(
                false,
                "division recurrence loops must be emitted as whole regions"
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
                self.emit_logical_imm(rd, rs1, imm, LogicalImmediateOp::Orr)
            }
            NativeInstruction::Sll { rd, rs1, rs2 } => {
                self.emit_shift_reg(rd, rs1, rs2, "lslv", lslv_reg)
            }
            NativeInstruction::Slli { rd, rs1, shamt } => {
                self.emit_shift_imm(rd, rs1, shamt, ShiftImmediateKind::Lsl)
            }
            NativeInstruction::Slliw { rd, rs1, shamt } => {
                self.emit_shift_imm32(rd, rs1, shamt, ShiftImmediateKind::Lsl)
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
                self.emit_shift_imm(rd, rs1, shamt, ShiftImmediateKind::Asr)
            }
            NativeInstruction::Sraiw { rd, rs1, shamt } => {
                self.emit_shift_imm32(rd, rs1, shamt, ShiftImmediateKind::Asr)
            }
            NativeInstruction::Sraw { rd, rs1, rs2 } => {
                self.emit_word_binary_reg(rd, rs1, rs2, "sraw", asrv_reg32)
            }
            NativeInstruction::Srl { rd, rs1, rs2 } => {
                self.emit_shift_reg(rd, rs1, rs2, "lsrv", lsrv_reg)
            }
            NativeInstruction::Srli { rd, rs1, shamt } => {
                self.emit_shift_imm(rd, rs1, shamt, ShiftImmediateKind::Lsr)
            }
            NativeInstruction::Srliw { rd, rs1, shamt } => {
                self.emit_shift_imm32(rd, rs1, shamt, ShiftImmediateKind::Lsr)
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
                self.emit_runtime_binary(rd, rs1, rs2, op)
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
                self.emit_logical_imm(rd, rs1, imm, LogicalImmediateOp::Eor)
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

        self.load_guest_register_from(COUNT_HOST, region.counter, ARG_REG_PTR);
        self.load_guest_register_from(ACC_HOST, region.accumulator, ARG_REG_PTR);
        self.load_guest_register_from(VALUE_HOST, region.value_register, ARG_REG_PTR);

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

        self.emit_shift_imm64_host(
            PAIR_COUNT_HOST,
            COUNT_HOST,
            1,
            ShiftImmediateKind::Lsr,
            " ; xor-toggle pair count",
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
            madd_reg(ACC_HOST, PAIR_SUM_HOST, PAIR_COUNT_HOST, ACC_HOST),
            format!(
                "madd x{ACC_HOST}, x{PAIR_SUM_HOST}, x{PAIR_COUNT_HOST}, x{ACC_HOST} ; xor-toggle paired contribution"
            ),
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

        self.store_guest_register_to(region.counter, A64_ZERO_REGISTER, ARG_REG_PTR);
        self.store_guest_register_to(region.accumulator, ACC_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.value_register, VALUE_HOST, ARG_REG_PTR);
        self.store_pc_to(region.exit_pc, ARG_REG_PTR);
        self.mov_imm64(LOOP_TEMP, region.guest_instructions);
        self.emit(
            mul_reg(0, COUNT_HOST, LOOP_TEMP),
            format!("mul x0, x{COUNT_HOST}, x{LOOP_TEMP} ; xor-toggle guest instruction count"),
        );
        self.ret();
    }

    fn emit_counted_diamond_loop(&mut self, region: CountedDiamondLoop) {
        if self.emit_counted_diamond_loop_closed_form(region) {
            return;
        }
        self.emit_counted_diamond_loop_iterative(region);
    }

    fn emit_counted_diamond_loop_closed_form(&mut self, region: CountedDiamondLoop) -> bool {
        if region.counter_delta != -1 {
            return false;
        }
        let Some(mask_shift) = counted_diamond_closed_form_mask_shift(region.mask) else {
            return false;
        };

        const COUNTER_HOST: u8 = 17;
        const ACCUMULATOR_HOST: u8 = 9;
        const ITERATION_HOST: u8 = 10;
        const ZERO_COUNT_HOST: u8 = 11;
        const NONZERO_COUNT_HOST: u8 = 12;
        const INSTRUCTION_COUNT_HOST: u8 = 15;

        self.load_guest_register_from(COUNTER_HOST, region.counter, ARG_REG_PTR);
        self.load_guest_register_from(ACCUMULATOR_HOST, region.accumulator, ARG_REG_PTR);
        self.load_guest_register_from(ITERATION_HOST, region.iteration_register, ARG_REG_PTR);

        self.emit_shift_imm64_host(
            ZERO_COUNT_HOST,
            COUNTER_HOST,
            mask_shift,
            ShiftImmediateKind::Lsr,
            " ; counted diamond zero-arm count",
        );
        self.emit(
            cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
            format!("cmp x{COUNTER_HOST}, xzr ; zero count means 2^64 iterations"),
        );
        self.mov_imm64(LOOP_TEMP, 1u64 << (64 - mask_shift));
        self.emit(
            csel(ZERO_COUNT_HOST, LOOP_TEMP, ZERO_COUNT_HOST, A64Cond::Eq),
            format!("csel x{ZERO_COUNT_HOST}, x{LOOP_TEMP}, x{ZERO_COUNT_HOST}, eq ; normalized zero-arm count"),
        );
        self.emit(
            sub_reg(NONZERO_COUNT_HOST, COUNTER_HOST, ZERO_COUNT_HOST),
            format!("sub x{NONZERO_COUNT_HOST}, x{COUNTER_HOST}, x{ZERO_COUNT_HOST} ; counted diamond nonzero count"),
        );

        self.mov_imm64(LOOP_TEMP, region.zero_accumulator_delta as u64);
        self.emit(
            madd_reg(
                ACCUMULATOR_HOST,
                ZERO_COUNT_HOST,
                LOOP_TEMP,
                ACCUMULATOR_HOST,
            ),
            format!(
                "madd x{ACCUMULATOR_HOST}, x{ZERO_COUNT_HOST}, x{LOOP_TEMP}, x{ACCUMULATOR_HOST} ; counted diamond zero contribution"
            ),
        );
        self.mov_imm64(LOOP_TEMP, region.nonzero_accumulator_delta as u64);
        self.emit(
            madd_reg(
                ACCUMULATOR_HOST,
                NONZERO_COUNT_HOST,
                LOOP_TEMP,
                ACCUMULATOR_HOST,
            ),
            format!(
                "madd x{ACCUMULATOR_HOST}, x{NONZERO_COUNT_HOST}, x{LOOP_TEMP}, x{ACCUMULATOR_HOST} ; counted diamond nonzero contribution"
            ),
        );

        self.mov_imm64(LOOP_TEMP, region.iteration_delta as u64);
        self.emit(
            madd_reg(ITERATION_HOST, COUNTER_HOST, LOOP_TEMP, ITERATION_HOST),
            format!(
                "madd x{ITERATION_HOST}, x{COUNTER_HOST}, x{LOOP_TEMP}, x{ITERATION_HOST} ; counted diamond iterations"
            ),
        );

        self.mov_imm64(LOOP_TEMP, region.zero_guest_instructions);
        self.emit(
            mul_reg(INSTRUCTION_COUNT_HOST, ZERO_COUNT_HOST, LOOP_TEMP),
            format!("mul x{INSTRUCTION_COUNT_HOST}, x{ZERO_COUNT_HOST}, x{LOOP_TEMP} ; counted diamond zero instruction count"),
        );
        self.mov_imm64(LOOP_TEMP, region.nonzero_guest_instructions);
        self.emit(
            madd_reg(
                INSTRUCTION_COUNT_HOST,
                NONZERO_COUNT_HOST,
                LOOP_TEMP,
                INSTRUCTION_COUNT_HOST,
            ),
            format!(
                "madd x{INSTRUCTION_COUNT_HOST}, x{NONZERO_COUNT_HOST}, x{LOOP_TEMP}, x{INSTRUCTION_COUNT_HOST} ; counted diamond instruction count"
            ),
        );

        self.store_guest_register_to(region.counter, A64_ZERO_REGISTER, ARG_REG_PTR);
        self.store_guest_register_to(region.accumulator, ACCUMULATOR_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.iteration_register, ITERATION_HOST, ARG_REG_PTR);
        self.mov_imm64(LOOP_TEMP, 1);
        self.store_guest_register_to(region.parity_register, LOOP_TEMP, ARG_REG_PTR);
        self.store_pc_to(region.exit_pc, ARG_REG_PTR);
        self.emit(
            mov_reg(0, INSTRUCTION_COUNT_HOST),
            format!("mov x0, x{INSTRUCTION_COUNT_HOST} ; return executed instructions"),
        );
        self.ret();
        true
    }

    fn emit_counted_diamond_loop_iterative(&mut self, region: CountedDiamondLoop) {
        const COUNTER_HOST: u8 = 2;
        const ACCUMULATOR_HOST: u8 = 3;
        const ITERATION_HOST: u8 = 4;
        const PARITY_HOST: u8 = 5;
        const MASK_HOST: u8 = 6;
        const ZERO_DELTA_HOST: u8 = 7;
        const NONZERO_DELTA_HOST: u8 = 8;
        const SELECTED_HOST: u8 = 10;
        const ZERO_COUNT_HOST: u8 = 11;
        const NONZERO_COUNT_HOST: u8 = 12;
        const INSTRUCTION_COUNT_HOST: u8 = 13;

        self.load_guest_register_from(COUNTER_HOST, region.counter, ARG_REG_PTR);
        self.load_guest_register_from(ACCUMULATOR_HOST, region.accumulator, ARG_REG_PTR);
        self.load_guest_register_from(ITERATION_HOST, region.iteration_register, ARG_REG_PTR);
        self.mov_imm64(INSTRUCTION_COUNT_HOST, 0);
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
                INSTRUCTION_COUNT_HOST,
                INSTRUCTION_COUNT_HOST,
                SELECTED_HOST,
            ),
            format!("add x{INSTRUCTION_COUNT_HOST}, x{INSTRUCTION_COUNT_HOST}, x{SELECTED_HOST}"),
        );
        self.emit_host_add_sub_imm_any(ITERATION_HOST, ITERATION_HOST, region.iteration_delta);
        if region.counter_delta == -1 {
            self.emit(
                subs_imm(COUNTER_HOST, COUNTER_HOST, 1),
                format!("subs x{COUNTER_HOST}, x{COUNTER_HOST}, #1 ; counted diamond backedge"),
            );
        } else {
            self.emit_host_add_sub_imm_any(COUNTER_HOST, COUNTER_HOST, region.counter_delta);
            self.emit(
                cmp_reg(COUNTER_HOST, A64_ZERO_REGISTER),
                format!("cmp x{COUNTER_HOST}, xzr ; counted diamond backedge"),
            );
        }
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, A64Cond::Ne),
            "b.ne .counted_diamond_loop".to_string(),
        );

        self.store_guest_register_to(region.counter, COUNTER_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.accumulator, ACCUMULATOR_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.iteration_register, ITERATION_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.parity_register, PARITY_HOST, ARG_REG_PTR);
        self.store_pc_to(region.exit_pc, ARG_REG_PTR);
        self.emit(
            mov_reg(0, INSTRUCTION_COUNT_HOST),
            format!("mov x0, x{INSTRUCTION_COUNT_HOST} ; return executed instructions"),
        );
        self.ret();
    }

    fn emit_fibonacci_recurrence_loop(&mut self, region: FibonacciRecurrenceLoop) {
        const UNROLLED_GUEST_ITERATIONS: u16 = 256;
        const UNROLLED_PAIRS: u16 = UNROLLED_GUEST_ITERATIONS / 2;
        const COUNTER_HOST: u8 = 9;
        const CURRENT_HOST: u8 = 10;
        const PREVIOUS_HOST: u8 = 11;
        const CHECKSUM_HOST: u8 = 12;
        const TRIP_COUNT_HOST: u8 = 13;
        const STEP_HOST: u8 = 14;
        const BLOCK_CHECKSUM_EVEN_HOST: u8 = 15;
        const BLOCK_CHECKSUM_ODD_HOST: u8 = 16;
        const SAVED_HOST: u8 = 17;

        self.load_guest_register_from(COUNTER_HOST, region.counter, ARG_REG_PTR);
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
        self.load_guest_register_from(CURRENT_HOST, region.current_register, ARG_REG_PTR);
        self.load_guest_register_from(PREVIOUS_HOST, region.previous_register, ARG_REG_PTR);
        self.load_guest_register_from(CHECKSUM_HOST, region.checksum_register, ARG_REG_PTR);

        self.mov_imm64(STEP_HOST, u64::from(UNROLLED_GUEST_ITERATIONS));
        self.emit(
            cmp_reg(COUNTER_HOST, STEP_HOST),
            format!("cmp x{COUNTER_HOST}, x{STEP_HOST} ; fib unrolled count"),
        );
        let tail_branch = self.emit_patchable_branch("b.lo .fib_tail".to_string());
        let unrolled_loop = self.current_offset();
        self.emit(
            mov_reg(BLOCK_CHECKSUM_EVEN_HOST, A64_ZERO_REGISTER),
            format!("mov x{BLOCK_CHECKSUM_EVEN_HOST}, xzr ; fib even block checksum"),
        );
        self.emit(
            mov_reg(BLOCK_CHECKSUM_ODD_HOST, A64_ZERO_REGISTER),
            format!("mov x{BLOCK_CHECKSUM_ODD_HOST}, xzr ; fib odd block checksum"),
        );
        for _ in 0..UNROLLED_PAIRS {
            self.emit_fibonacci_recurrence_pair(
                CURRENT_HOST,
                PREVIOUS_HOST,
                BLOCK_CHECKSUM_EVEN_HOST,
                BLOCK_CHECKSUM_ODD_HOST,
            );
        }
        self.emit(
            eor_reg(CHECKSUM_HOST, CHECKSUM_HOST, BLOCK_CHECKSUM_EVEN_HOST),
            format!("eor x{CHECKSUM_HOST}, x{CHECKSUM_HOST}, x{BLOCK_CHECKSUM_EVEN_HOST} ; fib merge even block checksum"),
        );
        self.emit(
            eor_reg(CHECKSUM_HOST, CHECKSUM_HOST, BLOCK_CHECKSUM_ODD_HOST),
            format!("eor x{CHECKSUM_HOST}, x{CHECKSUM_HOST}, x{BLOCK_CHECKSUM_ODD_HOST} ; fib merge odd block checksum"),
        );
        self.emit(
            sub_imm(COUNTER_HOST, COUNTER_HOST, UNROLLED_GUEST_ITERATIONS),
            format!(
                "sub x{COUNTER_HOST}, x{COUNTER_HOST}, #{UNROLLED_GUEST_ITERATIONS} ; fib unrolled count"
            ),
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
            subs_imm(COUNTER_HOST, COUNTER_HOST, 1),
            format!("subs x{COUNTER_HOST}, x{COUNTER_HOST}, #1 ; fib tail backedge"),
        );
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, tail_loop, A64Cond::Ne),
            "b.ne .fib_tail_loop".to_string(),
        );

        let done_offset = self.current_offset();
        self.patch_branch(done_branch, b_cond(done_branch, done_offset, A64Cond::Eq));
        self.store_guest_register_to(region.counter, A64_ZERO_REGISTER, ARG_REG_PTR);
        self.store_guest_register_to(region.current_register, CURRENT_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.previous_register, PREVIOUS_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.checksum_register, CHECKSUM_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.saved_register, PREVIOUS_HOST, ARG_REG_PTR);
        self.store_pc_to(region.exit_pc, ARG_REG_PTR);
        self.mov_imm64(STEP_HOST, region.guest_instructions);
        self.emit(
            mul_reg(0, TRIP_COUNT_HOST, STEP_HOST),
            format!("mul x0, x{TRIP_COUNT_HOST}, x{STEP_HOST} ; fib executed instructions"),
        );
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

    fn emit_fibonacci_recurrence_pair(
        &mut self,
        current: u8,
        previous: u8,
        even_checksum: u8,
        odd_checksum: u8,
    ) {
        self.emit(
            add_reg(previous, current, previous),
            format!("add x{previous}, x{current}, x{previous} ; fib next"),
        );
        self.emit(
            eor_reg(even_checksum, even_checksum, previous),
            format!("eor x{even_checksum}, x{even_checksum}, x{previous} ; fib even checksum"),
        );
        self.emit(
            add_reg(current, previous, current),
            format!("add x{current}, x{previous}, x{current} ; fib next"),
        );
        self.emit(
            eor_reg(odd_checksum, odd_checksum, current),
            format!("eor x{odd_checksum}, x{odd_checksum}, x{current} ; fib odd checksum"),
        );
    }

    fn emit_division_recurrence_loop(&mut self, region: DivisionRecurrenceLoop) {
        const COUNTER_HOST: u8 = 2;
        const SIGNED_HOST: u8 = 3;
        const SIGNED_DIVISOR_HOST: u8 = 4;
        const UNSIGNED_HOST: u8 = 5;
        const UNSIGNED_DIVISOR_HOST: u8 = 6;
        const CHECKSUM_HOST: u8 = 7;
        const TRIP_COUNT_HOST: u8 = 10;

        self.load_guest_register_from(COUNTER_HOST, region.counter, ARG_REG_PTR);
        self.load_guest_register_from(SIGNED_HOST, region.signed_value, ARG_REG_PTR);
        self.load_guest_register_from(SIGNED_DIVISOR_HOST, region.signed_divisor, ARG_REG_PTR);
        self.load_guest_register_from(UNSIGNED_HOST, region.unsigned_value, ARG_REG_PTR);
        self.load_guest_register_from(UNSIGNED_DIVISOR_HOST, region.unsigned_divisor, ARG_REG_PTR);
        self.load_guest_register_from(CHECKSUM_HOST, region.checksum, ARG_REG_PTR);
        self.emit(
            mov_reg(TRIP_COUNT_HOST, COUNTER_HOST),
            format!("mov x{TRIP_COUNT_HOST}, x{COUNTER_HOST} ; division recurrence trip count"),
        );

        let mut hardware_branches = Vec::new();
        if let Some(spec) = DivisionRecurrenceReciprocalSpec::from_region(region) {
            self.emit_division_recurrence_reciprocal_guards(
                SIGNED_DIVISOR_HOST,
                UNSIGNED_DIVISOR_HOST,
                spec,
                &mut hardware_branches,
            );
            self.emit_division_recurrence_reciprocal_constants(spec);

            let reciprocal_loop = self.current_offset();
            self.emit_division_recurrence_reciprocal_iteration(
                COUNTER_HOST,
                SIGNED_HOST,
                SIGNED_DIVISOR_HOST,
                UNSIGNED_DIVISOR_HOST,
                UNSIGNED_HOST,
                CHECKSUM_HOST,
                region,
                spec,
            );
            self.emit(
                subs_imm(COUNTER_HOST, COUNTER_HOST, 1),
                format!(
                    "subs x{COUNTER_HOST}, x{COUNTER_HOST}, #1 ; division recurrence reciprocal backedge"
                ),
            );
            let reciprocal_branch = self.current_offset();
            self.emit(
                b_cond(reciprocal_branch, reciprocal_loop, A64Cond::Ne),
                "b.ne .division_recurrence_reciprocal_loop".to_string(),
            );
            self.emit_division_recurrence_exit(
                region,
                COUNTER_HOST,
                SIGNED_HOST,
                UNSIGNED_HOST,
                CHECKSUM_HOST,
                TRIP_COUNT_HOST,
            );
        }

        let hardware_start = self.current_offset();
        for branch in hardware_branches {
            self.patch_branch(branch, b_cond(branch, hardware_start, A64Cond::Ne));
        }

        let mut slow_branches = Vec::with_capacity(4);
        self.emit(
            cmp_reg(SIGNED_DIVISOR_HOST, A64_ZERO_REGISTER),
            format!("cmp x{SIGNED_DIVISOR_HOST}, xzr ; division recurrence signed divisor"),
        );
        slow_branches
            .push(self.emit_patchable_branch("b.eq .division_recurrence_masked".to_string()));
        self.emit(
            cmp_reg32(SIGNED_DIVISOR_HOST, A64_ZERO_REGISTER),
            format!("cmp w{SIGNED_DIVISOR_HOST}, wzr ; division recurrence signed word divisor"),
        );
        slow_branches
            .push(self.emit_patchable_branch("b.eq .division_recurrence_masked".to_string()));
        self.emit(
            cmp_reg(UNSIGNED_DIVISOR_HOST, A64_ZERO_REGISTER),
            format!("cmp x{UNSIGNED_DIVISOR_HOST}, xzr ; division recurrence unsigned divisor"),
        );
        slow_branches
            .push(self.emit_patchable_branch("b.eq .division_recurrence_masked".to_string()));
        self.emit(
            cmp_reg32(UNSIGNED_DIVISOR_HOST, A64_ZERO_REGISTER),
            format!(
                "cmp w{UNSIGNED_DIVISOR_HOST}, wzr ; division recurrence unsigned word divisor"
            ),
        );
        slow_branches
            .push(self.emit_patchable_branch("b.eq .division_recurrence_masked".to_string()));

        let fast_loop = self.current_offset();
        self.emit_division_recurrence_fast_iteration(
            COUNTER_HOST,
            SIGNED_HOST,
            SIGNED_DIVISOR_HOST,
            UNSIGNED_HOST,
            UNSIGNED_DIVISOR_HOST,
            CHECKSUM_HOST,
            region,
        );
        self.emit(
            subs_imm(COUNTER_HOST, COUNTER_HOST, 1),
            format!("subs x{COUNTER_HOST}, x{COUNTER_HOST}, #1 ; division recurrence backedge"),
        );
        let fast_branch = self.current_offset();
        self.emit(
            b_cond(fast_branch, fast_loop, A64Cond::Ne),
            "b.ne .division_recurrence_loop".to_string(),
        );
        self.emit_division_recurrence_exit(
            region,
            COUNTER_HOST,
            SIGNED_HOST,
            UNSIGNED_HOST,
            CHECKSUM_HOST,
            TRIP_COUNT_HOST,
        );

        let slow_loop = self.current_offset();
        for branch in slow_branches {
            self.patch_branch(branch, b_cond(branch, slow_loop, A64Cond::Eq));
        }
        self.emit_division_recurrence_masked_iteration(
            COUNTER_HOST,
            SIGNED_HOST,
            SIGNED_DIVISOR_HOST,
            UNSIGNED_HOST,
            UNSIGNED_DIVISOR_HOST,
            CHECKSUM_HOST,
            region,
        );
        self.emit(
            subs_imm(COUNTER_HOST, COUNTER_HOST, 1),
            format!(
                "subs x{COUNTER_HOST}, x{COUNTER_HOST}, #1 ; division recurrence masked backedge"
            ),
        );
        let slow_branch = self.current_offset();
        self.emit(
            b_cond(slow_branch, slow_loop, A64Cond::Ne),
            "b.ne .division_recurrence_masked_loop".to_string(),
        );
        self.emit_division_recurrence_exit(
            region,
            COUNTER_HOST,
            SIGNED_HOST,
            UNSIGNED_HOST,
            CHECKSUM_HOST,
            TRIP_COUNT_HOST,
        );
    }

    fn emit_division_recurrence_reciprocal_guards(
        &mut self,
        signed_divisor: u8,
        unsigned_divisor: u8,
        spec: DivisionRecurrenceReciprocalSpec,
        hardware_branches: &mut Vec<usize>,
    ) {
        self.mov_imm64(LOOP_TEMP, spec.signed_divisor);
        self.emit(
            cmp_reg(signed_divisor, LOOP_TEMP),
            format!(
                "cmp x{signed_divisor}, x{LOOP_TEMP} ; division recurrence reciprocal signed guard"
            ),
        );
        hardware_branches
            .push(self.emit_patchable_branch("b.ne .division_recurrence_hardware".to_string()));

        self.mov_imm64(LOOP_TEMP, spec.unsigned_divisor);
        self.emit(
            cmp_reg(unsigned_divisor, LOOP_TEMP),
            format!(
                "cmp x{unsigned_divisor}, x{LOOP_TEMP} ; division recurrence reciprocal unsigned guard"
            ),
        );
        hardware_branches
            .push(self.emit_patchable_branch("b.ne .division_recurrence_hardware".to_string()));
    }

    fn emit_division_recurrence_reciprocal_constants(
        &mut self,
        spec: DivisionRecurrenceReciprocalSpec,
    ) {
        const SIGNED_MAGIC: u8 = 8;
        const UNSIGNED_MAGIC: u8 = 9;

        self.emit_const_signed_magic(SIGNED_MAGIC, spec.signed, "signed");
        self.emit_const_unsigned_magic(UNSIGNED_MAGIC, spec.unsigned64, "unsigned");
    }

    fn emit_division_recurrence_reciprocal_iteration(
        &mut self,
        counter: u8,
        signed_value: u8,
        signed_divisor: u8,
        unsigned_divisor: u8,
        unsigned_value: u8,
        checksum: u8,
        region: DivisionRecurrenceLoop,
        spec: DivisionRecurrenceReciprocalSpec,
    ) {
        const SIGNED_MAGIC: u8 = 8;
        const UNSIGNED_MAGIC: u8 = 9;
        const SIGNED_QUOTIENT: u8 = 11;
        const SIGNED_REMAINDER: u8 = 12;
        const UNSIGNED_QUOTIENT: u8 = 13;
        const UNSIGNED_REMAINDER: u8 = 14;
        const SIGNED_WORD_QUOTIENT: u8 = 15;
        const SIGNED_WORD_REMAINDER: u8 = 17;
        const UNSIGNED_WORD_QUOTIENT: u8 = 0;
        const WORD_INPUT: u8 = 12;
        const SIGN_MASK: u8 = 17;

        self.emit_const_signed_division_pair(
            SIGNED_QUOTIENT,
            SIGNED_REMAINDER,
            signed_value,
            SIGN_MASK,
            spec.signed,
            signed_divisor,
            SIGNED_MAGIC,
            "div",
        );
        self.emit_division_recurrence_checksum(checksum, SIGNED_QUOTIENT, "div");
        self.emit_division_recurrence_checksum(checksum, SIGNED_REMAINDER, "rem");

        self.emit_const_unsigned_division_pair(
            UNSIGNED_QUOTIENT,
            UNSIGNED_REMAINDER,
            unsigned_value,
            spec.unsigned64,
            unsigned_divisor,
            UNSIGNED_MAGIC,
            "divu",
        );
        self.emit_division_recurrence_checksum(checksum, UNSIGNED_QUOTIENT, "divu");
        self.emit_division_recurrence_checksum(checksum, UNSIGNED_REMAINDER, "remu");

        self.sign_extend_word(WORD_INPUT, signed_value);
        self.emit_const_signed_division_pair(
            SIGNED_WORD_QUOTIENT,
            SIGNED_WORD_REMAINDER,
            WORD_INPUT,
            SIGNED_QUOTIENT,
            spec.signed,
            signed_divisor,
            SIGNED_MAGIC,
            "divw",
        );
        self.emit_division_recurrence_checksum(checksum, SIGNED_WORD_QUOTIENT, "divw");
        self.emit_division_recurrence_checksum(checksum, SIGNED_WORD_REMAINDER, "remw");

        self.zero_extend_word(WORD_INPUT, unsigned_value);
        self.emit_const_unsigned_division_pair(
            UNSIGNED_WORD_QUOTIENT,
            UNSIGNED_REMAINDER,
            WORD_INPUT,
            spec.unsigned64,
            unsigned_divisor,
            UNSIGNED_MAGIC,
            "divuw",
        );
        self.sign_extend_word(UNSIGNED_WORD_QUOTIENT, UNSIGNED_WORD_QUOTIENT);
        self.sign_extend_word(UNSIGNED_REMAINDER, UNSIGNED_REMAINDER);
        self.emit_division_recurrence_checksum(checksum, UNSIGNED_WORD_QUOTIENT, "divuw");
        self.emit_division_recurrence_checksum(checksum, UNSIGNED_REMAINDER, "remuw");

        self.emit_mulhsu_host(SIGNED_QUOTIENT, signed_value, unsigned_divisor);
        self.emit(
            eor_reg(checksum, checksum, SIGNED_QUOTIENT),
            format!("eor x{checksum}, x{checksum}, x{SIGNED_QUOTIENT} ; division recurrence reciprocal mulhsu checksum"),
        );
        self.emit_host_add_sub_imm_any(signed_value, signed_value, region.signed_delta);
        self.emit_host_add_sub_imm_any(unsigned_value, unsigned_value, region.unsigned_delta);
        self.emit_host_move(
            counter,
            counter,
            " ; division recurrence reciprocal counter live",
        );
    }

    fn emit_division_recurrence_checksum(&mut self, checksum: u8, value: u8, suffix: &str) {
        self.emit(
            eor_reg(checksum, checksum, value),
            format!(
                "eor x{checksum}, x{checksum}, x{value} ; division recurrence reciprocal {suffix} checksum"
            ),
        );
    }

    fn emit_const_signed_division_pair(
        &mut self,
        quotient: u8,
        remainder: u8,
        numerator: u8,
        sign_mask: u8,
        divisor: ConstSignedDivisor,
        divisor_reg: u8,
        magic_reg: u8,
        mnemonic: &str,
    ) {
        self.emit_const_signed_quotient(
            quotient,
            numerator,
            sign_mask,
            divisor,
            magic_reg,
            &format!("division recurrence reciprocal {mnemonic} quotient"),
        );
        self.emit_const_remainder(
            remainder,
            numerator,
            quotient,
            divisor_reg,
            &format!("division recurrence reciprocal rem for {mnemonic}"),
        );
    }

    fn emit_const_unsigned_division_pair(
        &mut self,
        quotient: u8,
        remainder: u8,
        numerator: u8,
        divisor: ConstUnsignedDivisor,
        divisor_reg: u8,
        magic_reg: u8,
        mnemonic: &str,
    ) {
        self.emit_const_unsigned_quotient(
            quotient,
            numerator,
            divisor,
            magic_reg,
            &format!("division recurrence reciprocal {mnemonic} quotient"),
        );
        self.emit_const_remainder(
            remainder,
            numerator,
            quotient,
            divisor_reg,
            &format!("division recurrence reciprocal rem for {mnemonic}"),
        );
    }

    fn emit_const_signed_magic(&mut self, magic_reg: u8, divisor: ConstSignedDivisor, name: &str) {
        if divisor.value == 1 {
            return;
        }

        self.mov_imm64(magic_reg, divisor.magic as u64);
        self.emit_host_move(
            magic_reg,
            magic_reg,
            &format!(" ; division recurrence reciprocal {name} signed magic"),
        );
    }

    fn emit_const_unsigned_magic(
        &mut self,
        magic_reg: u8,
        divisor: ConstUnsignedDivisor,
        name: &str,
    ) {
        if divisor.value == 1 || divisor.power_of_two_shift.is_some() {
            return;
        }

        self.mov_imm64(magic_reg, divisor.multiplier);
        self.emit_host_move(
            magic_reg,
            magic_reg,
            &format!(" ; division recurrence reciprocal {name} magic"),
        );
    }

    fn emit_const_signed_quotient(
        &mut self,
        quotient: u8,
        numerator: u8,
        sign_adjust: u8,
        divisor: ConstSignedDivisor,
        magic_reg: u8,
        suffix: &str,
    ) {
        if divisor.value == 1 {
            self.emit_host_move(quotient, numerator, &format!(" ; {suffix}"));
            return;
        }

        self.emit(
            smulh_reg(quotient, numerator, magic_reg),
            format!("smulh x{quotient}, x{numerator}, x{magic_reg} ; {suffix}"),
        );
        if divisor.magic < 0 {
            self.emit(
                add_reg(quotient, quotient, numerator),
                format!("add x{quotient}, x{quotient}, x{numerator} ; {suffix} magic adjust"),
            );
        }
        self.emit_shift_imm64_host(
            quotient,
            quotient,
            divisor.shift,
            ShiftImmediateKind::Asr,
            &format!(" ; {suffix} shift"),
        );
        self.emit_shift_imm64_host(
            sign_adjust,
            numerator,
            63,
            ShiftImmediateKind::Asr,
            &format!(" ; {suffix} sign adjust"),
        );
        self.emit(
            sub_reg(quotient, quotient, sign_adjust),
            format!("sub x{quotient}, x{quotient}, x{sign_adjust} ; {suffix}"),
        );
    }

    fn emit_const_unsigned_quotient(
        &mut self,
        quotient: u8,
        numerator: u8,
        divisor: ConstUnsignedDivisor,
        magic_reg: u8,
        suffix: &str,
    ) {
        if divisor.value == 1 {
            self.emit_host_move(quotient, numerator, &format!(" ; {suffix}"));
            return;
        }

        if let Some(shift) = divisor.power_of_two_shift {
            self.emit_shift_imm64_host(
                quotient,
                numerator,
                shift,
                ShiftImmediateKind::Lsr,
                &format!(" ; {suffix} power-of-two"),
            );
            return;
        }

        self.emit(
            umulh_reg(quotient, numerator, magic_reg),
            format!("umulh x{quotient}, x{numerator}, x{magic_reg} ; {suffix}"),
        );
        if divisor.add_indicator {
            self.emit(
                sub_reg(LOOP_TEMP, numerator, quotient),
                format!("sub x{LOOP_TEMP}, x{numerator}, x{quotient} ; {suffix} add adjust"),
            );
            self.emit_shift_imm64_host(
                LOOP_TEMP,
                LOOP_TEMP,
                1,
                ShiftImmediateKind::Lsr,
                &format!(" ; {suffix} add adjust"),
            );
            self.emit(
                add_reg(quotient, LOOP_TEMP, quotient),
                format!("add x{quotient}, x{LOOP_TEMP}, x{quotient} ; {suffix} add adjust"),
            );
            self.emit_shift_imm64_host(
                quotient,
                quotient,
                divisor.shift - 1,
                ShiftImmediateKind::Lsr,
                &format!(" ; {suffix}"),
            );
        } else {
            self.emit_shift_imm64_host(
                quotient,
                quotient,
                divisor.shift,
                ShiftImmediateKind::Lsr,
                &format!(" ; {suffix}"),
            );
        }
    }

    fn emit_const_remainder(
        &mut self,
        remainder: u8,
        numerator: u8,
        quotient: u8,
        divisor_reg: u8,
        suffix: &str,
    ) {
        self.emit(
            msub_reg(remainder, quotient, divisor_reg, numerator),
            format!("msub x{remainder}, x{quotient}, x{divisor_reg}, x{numerator} ; {suffix}"),
        );
    }

    fn emit_division_recurrence_fast_iteration(
        &mut self,
        counter: u8,
        signed_value: u8,
        signed_divisor: u8,
        unsigned_value: u8,
        unsigned_divisor: u8,
        checksum: u8,
        region: DivisionRecurrenceLoop,
    ) {
        const SIGNED_QUOTIENT: u8 = 11;
        const SIGNED_REMAINDER: u8 = 12;
        const UNSIGNED_QUOTIENT: u8 = 13;
        const UNSIGNED_REMAINDER: u8 = 14;
        const SIGNED_WORD_QUOTIENT: u8 = 15;
        const SIGNED_WORD_REMAINDER: u8 = 17;
        const UNSIGNED_WORD_QUOTIENT: u8 = 0;
        const UNSIGNED_WORD_REMAINDER: u8 = 9;

        self.emit(
            sdiv_reg(SIGNED_QUOTIENT, signed_value, signed_divisor),
            format!(
                "sdiv x{SIGNED_QUOTIENT}, x{signed_value}, x{signed_divisor} ; division recurrence scheduled div"
            ),
        );
        self.emit(
            udiv_reg(UNSIGNED_QUOTIENT, unsigned_value, unsigned_divisor),
            format!(
                "udiv x{UNSIGNED_QUOTIENT}, x{unsigned_value}, x{unsigned_divisor} ; division recurrence scheduled divu"
            ),
        );
        self.emit(
            sdiv_reg32(SIGNED_WORD_QUOTIENT, signed_value, signed_divisor),
            format!(
                "sdiv w{SIGNED_WORD_QUOTIENT}, w{signed_value}, w{signed_divisor} ; division recurrence scheduled divw"
            ),
        );
        self.emit(
            udiv_reg32(UNSIGNED_WORD_QUOTIENT, unsigned_value, unsigned_divisor),
            format!(
                "udiv w{UNSIGNED_WORD_QUOTIENT}, w{unsigned_value}, w{unsigned_divisor} ; division recurrence scheduled divuw"
            ),
        );

        self.emit(
            msub_reg(
                SIGNED_REMAINDER,
                SIGNED_QUOTIENT,
                signed_divisor,
                signed_value,
            ),
            format!(
                "msub x{SIGNED_REMAINDER}, x{SIGNED_QUOTIENT}, x{signed_divisor}, x{signed_value} ; division recurrence rem"
            ),
        );
        self.emit(
            msub_reg(
                UNSIGNED_REMAINDER,
                UNSIGNED_QUOTIENT,
                unsigned_divisor,
                unsigned_value,
            ),
            format!(
                "msub x{UNSIGNED_REMAINDER}, x{UNSIGNED_QUOTIENT}, x{unsigned_divisor}, x{unsigned_value} ; division recurrence remu"
            ),
        );
        self.emit(
            msub_reg32(
                SIGNED_WORD_REMAINDER,
                SIGNED_WORD_QUOTIENT,
                signed_divisor,
                signed_value,
            ),
            format!(
                "msub w{SIGNED_WORD_REMAINDER}, w{SIGNED_WORD_QUOTIENT}, w{signed_divisor}, w{signed_value} ; division recurrence remw"
            ),
        );
        self.emit(
            msub_reg32(
                UNSIGNED_WORD_REMAINDER,
                UNSIGNED_WORD_QUOTIENT,
                unsigned_divisor,
                unsigned_value,
            ),
            format!(
                "msub w{UNSIGNED_WORD_REMAINDER}, w{UNSIGNED_WORD_QUOTIENT}, w{unsigned_divisor}, w{unsigned_value} ; division recurrence remuw"
            ),
        );

        self.sign_extend_word(SIGNED_WORD_QUOTIENT, SIGNED_WORD_QUOTIENT);
        self.sign_extend_word(SIGNED_WORD_REMAINDER, SIGNED_WORD_REMAINDER);
        self.sign_extend_word(UNSIGNED_WORD_QUOTIENT, UNSIGNED_WORD_QUOTIENT);
        self.sign_extend_word(UNSIGNED_WORD_REMAINDER, UNSIGNED_WORD_REMAINDER);

        for value in [
            SIGNED_QUOTIENT,
            SIGNED_REMAINDER,
            UNSIGNED_QUOTIENT,
            UNSIGNED_REMAINDER,
            SIGNED_WORD_QUOTIENT,
            SIGNED_WORD_REMAINDER,
            UNSIGNED_WORD_QUOTIENT,
            UNSIGNED_WORD_REMAINDER,
        ] {
            self.emit(
                eor_reg(checksum, checksum, value),
                format!("eor x{checksum}, x{checksum}, x{value} ; division recurrence checksum"),
            );
        }

        self.emit_mulhsu_host(SIGNED_QUOTIENT, signed_value, unsigned_divisor);
        self.emit(
            eor_reg(checksum, checksum, SIGNED_QUOTIENT),
            format!("eor x{checksum}, x{checksum}, x{SIGNED_QUOTIENT} ; division recurrence mulhsu checksum"),
        );
        self.emit_host_add_sub_imm_any(signed_value, signed_value, region.signed_delta);
        self.emit_host_add_sub_imm_any(unsigned_value, unsigned_value, region.unsigned_delta);
        self.emit_host_move(counter, counter, " ; division recurrence counter live");
    }

    fn emit_division_recurrence_masked_iteration(
        &mut self,
        counter: u8,
        signed_value: u8,
        signed_divisor: u8,
        unsigned_value: u8,
        unsigned_divisor: u8,
        checksum: u8,
        region: DivisionRecurrenceLoop,
    ) {
        const QUOTIENT: u8 = 11;
        const REMAINDER: u8 = 12;

        self.emit_division_recurrence_masked_pair(
            RuntimeBinaryOp::Div,
            RuntimeBinaryOp::Rem,
            signed_value,
            signed_divisor,
            checksum,
            QUOTIENT,
            REMAINDER,
        );
        self.emit_division_recurrence_masked_pair(
            RuntimeBinaryOp::Divu,
            RuntimeBinaryOp::Remu,
            unsigned_value,
            unsigned_divisor,
            checksum,
            QUOTIENT,
            REMAINDER,
        );
        self.emit_division_recurrence_masked_pair(
            RuntimeBinaryOp::Divw,
            RuntimeBinaryOp::Remw,
            signed_value,
            signed_divisor,
            checksum,
            QUOTIENT,
            REMAINDER,
        );
        self.emit_division_recurrence_masked_pair(
            RuntimeBinaryOp::Divuw,
            RuntimeBinaryOp::Remuw,
            unsigned_value,
            unsigned_divisor,
            checksum,
            QUOTIENT,
            REMAINDER,
        );
        self.emit_mulhsu_host(QUOTIENT, signed_value, unsigned_divisor);
        self.emit(
            eor_reg(checksum, checksum, QUOTIENT),
            format!("eor x{checksum}, x{checksum}, x{QUOTIENT} ; division recurrence masked mulhsu checksum"),
        );
        self.emit_host_add_sub_imm_any(signed_value, signed_value, region.signed_delta);
        self.emit_host_add_sub_imm_any(unsigned_value, unsigned_value, region.unsigned_delta);
        self.emit_host_move(
            counter,
            counter,
            " ; division recurrence masked counter live",
        );
    }

    fn emit_division_recurrence_masked_pair(
        &mut self,
        quotient_op: RuntimeBinaryOp,
        remainder_op: RuntimeBinaryOp,
        lhs: u8,
        rhs: u8,
        checksum: u8,
        quotient: u8,
        remainder: u8,
    ) {
        let quotient_kind = runtime_binary_division_kind(quotient_op)
            .expect("division recurrence quotient operation");
        let remainder_kind = runtime_binary_division_kind(remainder_op)
            .expect("division recurrence remainder operation");
        debug_assert!(quotient_kind.is_quotient());
        debug_assert!(!remainder_kind.is_quotient());
        debug_assert_eq!(quotient_kind.is_word(), remainder_kind.is_word());

        if quotient_kind.is_word() {
            self.emit(
                cmp_reg32(rhs, A64_ZERO_REGISTER),
                format!(
                    "cmp w{rhs}, wzr ; division recurrence masked {} divisor",
                    quotient_kind.mnemonic()
                ),
            );
            if quotient_kind.is_signed() {
                self.emit(
                    sdiv_reg32(quotient, lhs, rhs),
                    format!("sdiv w{quotient}, w{lhs}, w{rhs} ; division recurrence masked"),
                );
            } else {
                self.emit(
                    udiv_reg32(quotient, lhs, rhs),
                    format!("udiv w{quotient}, w{lhs}, w{rhs} ; division recurrence masked"),
                );
            }
            self.emit(
                msub_reg32(remainder, quotient, rhs, lhs),
                format!(
                    "msub w{remainder}, w{quotient}, w{rhs}, w{lhs} ; division recurrence masked remainder"
                ),
            );
            self.sign_extend_word(quotient, quotient);
            self.sign_extend_word(remainder, remainder);
        } else {
            self.emit(
                cmp_reg(rhs, A64_ZERO_REGISTER),
                format!(
                    "cmp x{rhs}, xzr ; division recurrence masked {} divisor",
                    quotient_kind.mnemonic()
                ),
            );
            if quotient_kind.is_signed() {
                self.emit(
                    sdiv_reg(quotient, lhs, rhs),
                    format!("sdiv x{quotient}, x{lhs}, x{rhs} ; division recurrence masked"),
                );
            } else {
                self.emit(
                    udiv_reg(quotient, lhs, rhs),
                    format!("udiv x{quotient}, x{lhs}, x{rhs} ; division recurrence masked"),
                );
            }
            self.emit(
                msub_reg(remainder, quotient, rhs, lhs),
                format!(
                    "msub x{remainder}, x{quotient}, x{rhs}, x{lhs} ; division recurrence masked remainder"
                ),
            );
        }
        self.emit(
            csinv(quotient, quotient, A64_ZERO_REGISTER, A64Cond::Ne),
            format!(
                "csinv x{quotient}, x{quotient}, xzr, ne ; division recurrence masked quotient"
            ),
        );
        self.emit(
            eor_reg(checksum, checksum, quotient),
            format!(
                "eor x{checksum}, x{checksum}, x{quotient} ; division recurrence masked checksum"
            ),
        );
        self.emit(
            eor_reg(checksum, checksum, remainder),
            format!(
                "eor x{checksum}, x{checksum}, x{remainder} ; division recurrence masked checksum"
            ),
        );
    }

    fn emit_division_recurrence_exit(
        &mut self,
        region: DivisionRecurrenceLoop,
        counter: u8,
        signed_value: u8,
        unsigned_value: u8,
        checksum: u8,
        trip_count: u8,
    ) {
        const RESULT_HOST: u8 = 11;

        self.store_guest_register_to(region.counter, counter, ARG_REG_PTR);
        self.store_guest_register_to(region.temp_register, RESULT_HOST, ARG_REG_PTR);
        self.store_guest_register_to(region.signed_value, signed_value, ARG_REG_PTR);
        self.store_guest_register_to(region.unsigned_value, unsigned_value, ARG_REG_PTR);
        self.store_guest_register_to(region.checksum, checksum, ARG_REG_PTR);
        self.store_pc_to(region.exit_pc, ARG_REG_PTR);
        self.mov_imm64(LOOP_TEMP, DIVISION_RECURRENCE_INSTRUCTION_COUNT);
        self.emit(
            mul_reg(0, trip_count, LOOP_TEMP),
            format!(
                "mul x0, x{trip_count}, x{LOOP_TEMP} ; division recurrence executed instructions"
            ),
        );
        self.ret();
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
            madd_reg(OFFSET_HOST, COUNTER_HOST, LOOP_TEMP, OFFSET_HOST),
            format!(
                "madd x{OFFSET_HOST}, x{COUNTER_HOST}, x{LOOP_TEMP}, x{OFFSET_HOST} ; final offset delta"
            ),
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

    fn emit_invariant_divisor_masks(
        &mut self,
        masks: &InvariantDivisorMasks,
        loop_plan: &RegisterAllocatedLoop<'_>,
    ) {
        for mask in &masks.masks {
            let divisor = loop_plan.host_or_zero(mask.guest);
            if mask.word {
                self.emit(
                    cmp_reg32(divisor, A64_ZERO_REGISTER),
                    format!(
                        "cmp w{divisor}, wzr ; invariant divisor x{} zero",
                        mask.guest
                    ),
                );
            } else {
                self.emit(
                    cmp_reg(divisor, A64_ZERO_REGISTER),
                    format!(
                        "cmp x{divisor}, xzr ; invariant divisor x{} zero",
                        mask.guest
                    ),
                );
            }
            self.emit(
                csinv(mask.host, A64_ZERO_REGISTER, A64_ZERO_REGISTER, A64Cond::Ne),
                format!(
                    "csinv x{}, xzr, xzr, ne ; invariant divisor x{} zero mask",
                    mask.host, mask.guest
                ),
            );
        }
    }

    fn emit_register_allocated_self_loop(&mut self, loop_plan: &RegisterAllocatedLoop<'_>) {
        let calls_rust_runtime = loop_plan.calls_rust_runtime();
        let instruction_count_host =
            if !calls_rust_runtime && loop_plan.uses_caller_saved_guest_registers() {
                CALLER_LOOP_INSTRUCTION_COUNT
            } else {
                DYNAMIC_INSTRUCTION_COUNT
            };
        if calls_rust_runtime {
            self.emit_prologue(true);
        } else {
            self.emit_selective_leaf_prologue(
                &loop_plan.used_callee_saved_registers(instruction_count_host),
            );
        }
        let reg_ptr = if calls_rust_runtime {
            REG_PTR
        } else {
            ARG_REG_PTR
        };
        let decrementing_counter = loop_plan.decrementing_counter();
        if decrementing_counter.is_none() {
            self.emit(
                mov_reg(instruction_count_host, A64_ZERO_REGISTER),
                format!("mov x{instruction_count_host}, xzr ; dynamic instruction count"),
            );
        }

        for (guest, host) in &loop_plan.loaded_registers {
            self.load_guest_register_from(*host, *guest, reg_ptr);
        }
        if let Some(counter) = decrementing_counter {
            let counter_host = loop_plan.host_or_zero(counter);
            self.emit(
                mov_reg(instruction_count_host, counter_host),
                format!(
                    "mov x{instruction_count_host}, x{counter_host} ; initial loop counter x{counter}"
                ),
            );
        }
        let divisor_masks = InvariantDivisorMasks::from_loop_plan(loop_plan);
        self.emit_invariant_divisor_masks(&divisor_masks, loop_plan);

        if divisor_masks.is_empty() {
            self.emit_register_allocated_self_loop_body(
                loop_plan,
                decrementing_counter,
                DivisorFixupMode::Compare,
                calls_rust_runtime,
                reg_ptr,
                instruction_count_host,
            );
        } else {
            let masked_loop_branch = self.emit_divisor_mask_branch(&divisor_masks);
            self.emit_register_allocated_self_loop_body(
                loop_plan,
                decrementing_counter,
                DivisorFixupMode::KnownNonZero(&divisor_masks),
                calls_rust_runtime,
                reg_ptr,
                instruction_count_host,
            );
            let masked_loop_start = self.current_offset();
            self.patch_branch(
                masked_loop_branch,
                b_cond(masked_loop_branch, masked_loop_start, A64Cond::Ne),
            );
            self.emit_register_allocated_self_loop_body(
                loop_plan,
                decrementing_counter,
                DivisorFixupMode::Masked(&divisor_masks),
                calls_rust_runtime,
                reg_ptr,
                instruction_count_host,
            );
        }
    }

    fn emit_divisor_mask_branch(&mut self, masks: &InvariantDivisorMasks) -> usize {
        debug_assert!(!masks.is_empty());

        if masks.masks.len() == 1 {
            let mask = masks.masks[0].host;
            self.emit(
                cmp_reg(mask, A64_ZERO_REGISTER),
                format!("cmp x{mask}, xzr ; invariant divisor mask check"),
            );
        } else {
            let mut masks_iter = masks.masks.iter();
            let first = masks_iter.next().expect("non-empty divisor mask set").host;
            let second = masks_iter.next().expect("two divisor masks").host;
            self.emit(
                orr_reg(LOOP_TEMP, first, second),
                format!("orr x{LOOP_TEMP}, x{first}, x{second} ; combined divisor masks"),
            );
            for mask in masks_iter {
                self.emit(
                    orr_reg(LOOP_TEMP, LOOP_TEMP, mask.host),
                    format!(
                        "orr x{LOOP_TEMP}, x{LOOP_TEMP}, x{} ; combined divisor masks",
                        mask.host
                    ),
                );
            }
            self.emit(
                cmp_reg(LOOP_TEMP, A64_ZERO_REGISTER),
                format!("cmp x{LOOP_TEMP}, xzr ; invariant divisor mask check"),
            );
        }

        self.emit_patchable_branch("b.ne .jit_masked_divisor_loop".to_string())
    }

    fn emit_register_allocated_self_loop_body(
        &mut self,
        loop_plan: &RegisterAllocatedLoop<'_>,
        decrementing_counter: Option<u8>,
        divisor_fixups: DivisorFixupMode<'_>,
        calls_rust_runtime: bool,
        reg_ptr: u8,
        instruction_count_host: u8,
    ) {
        let loop_start = self.current_offset();
        let final_decrementing_counter = loop_plan.final_decrementing_counter();
        let mut operation_index = 0;
        while operation_index < loop_plan.operations.len() {
            if operation_index + 1 == loop_plan.operations.len() {
                if let Some((counter, word)) = final_decrementing_counter {
                    self.emit_loop_final_decrement(counter, word, loop_plan);
                    operation_index += 1;
                    continue;
                }
            }

            if let Some((pair, quotient_consumer, remainder_consumer)) = loop_plan
                .operations
                .get(operation_index + 3)
                .and_then(|fourth| {
                    let BlockOperationKind::Native(first) =
                        loop_plan.operations[operation_index].kind();
                    let BlockOperationKind::Native(middle) =
                        loop_plan.operations[operation_index + 1].kind();
                    let BlockOperationKind::Native(third) =
                        loop_plan.operations[operation_index + 2].kind();
                    let BlockOperationKind::Native(fourth) = fourth.kind();
                    let pair = fused_div_rem_pair_across_pure_gap(first, middle, third)?;
                    let quotient_consumer =
                        binary_reg_gap_consumer(middle, pair.quotient_register)?;
                    let remainder_consumer =
                        binary_reg_gap_consumer(fourth, pair.remainder_register)?;
                    Some((pair, quotient_consumer, remainder_consumer))
                })
            {
                self.emit_loop_div_rem_pair_fully_deferred(pair, loop_plan, divisor_fixups);
                self.emit_loop_binary_reg_with_read_override(
                    quotient_consumer,
                    pair.quotient_register,
                    SCRATCH2,
                    loop_plan,
                    " ; regalloc fused quotient consumer",
                );
                self.emit_loop_binary_reg_with_read_override(
                    remainder_consumer,
                    pair.remainder_register,
                    DIV_REM_CACHE,
                    loop_plan,
                    " ; regalloc fused remainder consumer",
                );
                if guest_register_needed_before_next_write(
                    loop_plan.operations,
                    operation_index + 4,
                    pair.quotient_register,
                ) {
                    if let Some(dst) = loop_plan.host_for_write(pair.quotient_register) {
                        self.emit_host_move(dst, SCRATCH2, " ; fused deferred quotient");
                    }
                }
                if guest_register_needed_before_next_write(
                    loop_plan.operations,
                    operation_index + 4,
                    pair.remainder_register,
                ) {
                    if let Some(dst) = loop_plan.host_for_write(pair.remainder_register) {
                        self.emit_host_move(dst, DIV_REM_CACHE, " ; fused deferred remainder");
                    }
                }
                operation_index += 4;
                continue;
            }

            if let Some(pair) = loop_plan
                .operations
                .get(operation_index + 2)
                .and_then(|third| {
                    let BlockOperationKind::Native(first) =
                        loop_plan.operations[operation_index].kind();
                    let BlockOperationKind::Native(middle) =
                        loop_plan.operations[operation_index + 1].kind();
                    let BlockOperationKind::Native(third) = third.kind();
                    fused_div_rem_pair_across_pure_gap(first, middle, third)
                })
            {
                self.emit_loop_div_rem_pair_deferred(pair, loop_plan, divisor_fixups);
                let BlockOperationKind::Native(middle) =
                    loop_plan.operations[operation_index + 1].kind();
                self.emit_loop_instruction_with_divisor_fixups(middle, loop_plan, divisor_fixups);
                if let Some(dst) = loop_plan.host_for_write(pair.remainder_register) {
                    self.emit_host_move(dst, DIV_REM_CACHE, " ; fused deferred remainder");
                }
                operation_index += 3;
                continue;
            }

            if let Some(pair) = loop_plan
                .operations
                .get(operation_index + 1)
                .and_then(|next| {
                    let BlockOperationKind::Native(first) =
                        loop_plan.operations[operation_index].kind();
                    let BlockOperationKind::Native(second) = next.kind();
                    fused_div_rem_pair(first, second)
                })
            {
                self.emit_loop_div_rem_pair(pair, loop_plan, divisor_fixups);
                operation_index += 2;
                continue;
            }

            let operation = &loop_plan.operations[operation_index];
            let BlockOperationKind::Native(instruction) = operation.kind();
            self.emit_loop_instruction_with_divisor_fixups(instruction, loop_plan, divisor_fixups);
            operation_index += 1;
        }

        if decrementing_counter.is_none() {
            self.emit(
                add_imm(
                    instruction_count_host,
                    instruction_count_host,
                    loop_plan.guest_instruction_count as u16,
                ),
                format!(
                    "add x{instruction_count_host}, x{instruction_count_host}, #{}",
                    loop_plan.guest_instruction_count
                ),
            );
        }
        let lhs = loop_plan.host_or_zero(loop_plan.branch.rs1);
        let rhs = loop_plan.host_or_zero(loop_plan.branch.rs2);
        if final_decrementing_counter.is_none() {
            self.emit(cmp_reg(lhs, rhs), format!("cmp x{lhs}, x{rhs}"));
        }
        let branch_offset = self.current_offset();
        self.emit(
            b_cond(branch_offset, loop_start, loop_plan.branch.condition),
            format!(
                "b.{} .jit_loop_start",
                loop_plan.branch.condition.mnemonic()
            ),
        );

        for (guest, host) in &loop_plan.dirty_registers {
            self.store_guest_register_to(*guest, *host, reg_ptr);
        }
        self.store_pc_to(loop_plan.branch.fallthrough, reg_ptr);
        if decrementing_counter.is_some() {
            self.emit_loop_return_instruction_count(
                instruction_count_host,
                loop_plan.guest_instruction_count,
            );
        } else {
            self.emit(
                mov_reg(0, instruction_count_host),
                format!("mov x0, x{instruction_count_host} ; return executed instructions"),
            );
        }
        if calls_rust_runtime {
            self.emit_epilogue(true);
        } else {
            self.emit_selective_leaf_epilogue(
                &loop_plan.used_callee_saved_registers(instruction_count_host),
            );
        }
        self.ret();
    }

    fn emit_loop_final_decrement(
        &mut self,
        counter: u8,
        word: bool,
        loop_plan: &RegisterAllocatedLoop<'_>,
    ) {
        let dst = loop_plan.host_or_zero(counter);
        if word {
            self.emit(
                subs_imm32(dst, dst, 1),
                format!("subs w{dst}, w{dst}, #1 ; regalloc final loop decrement"),
            );
            self.sign_extend_word(dst, dst);
        } else {
            self.emit(
                subs_imm(dst, dst, 1),
                format!("subs x{dst}, x{dst}, #1 ; regalloc final loop decrement"),
            );
        }
    }

    fn emit_loop_return_instruction_count(&mut self, trip_count_host: u8, instruction_count: u64) {
        debug_assert!(instruction_count > 0);
        if instruction_count == 1 {
            self.emit(
                mov_reg(0, trip_count_host),
                format!("mov x0, x{trip_count_host} ; return executed instructions"),
            );
        } else if instruction_count.is_power_of_two() {
            let shift = instruction_count.trailing_zeros();
            self.emit(
                ShiftImmediateKind::Lsl.encode64(0, trip_count_host, shift),
                format!("lsl x0, x{trip_count_host}, #{shift} ; return executed instructions"),
            );
        } else {
            self.mov_imm64(LOOP_TEMP, instruction_count);
            self.emit(
                mul_reg(0, trip_count_host, LOOP_TEMP),
                format!("mul x0, x{trip_count_host}, x{LOOP_TEMP} ; return executed instructions"),
            );
        }
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

        self.emit_register_allocated_trace_loop_body(trace_loop);
    }

    fn emit_register_allocated_trace_loop_body(
        &mut self,
        trace_loop: &RegisterAllocatedTraceLoop<'_>,
    ) {
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

    fn emit_direct_load_trace_loop(
        &mut self,
        trace_loop: &RegisterAllocatedTraceLoop<'_>,
        region: DirectLoadTraceLoop,
    ) {
        const UNROLLED_LOADS: u16 = 4;
        const READ_PTR_HOST: u8 = 26;
        const ITERATIONS_HOST: u8 = 27;
        const BYTE_LEN_HOST: u8 = 28;

        let loaded_host = trace_loop.host_or_zero(region.loaded_register);
        let address_host = trace_loop.host_or_zero(region.address_register);
        let counter_host = trace_loop.host_or_zero(region.counter_register);
        let limit_host = trace_loop.host_or_zero(region.limit_register);

        self.emit_prologue(true);
        for (guest, host) in &trace_loop.loaded_registers {
            self.load_guest_register(*host, *guest);
        }

        self.emit(
            cmp_reg(counter_host, limit_host),
            format!("cmp x{counter_host}, x{limit_host} ; direct trace finite count"),
        );
        let generic_count_branch =
            self.emit_patchable_branch("b.hs .generic_trace_loop".to_string());

        self.emit(
            sub_reg(ITERATIONS_HOST, limit_host, counter_host),
            format!(
                "sub x{ITERATIONS_HOST}, x{limit_host}, x{counter_host} ; direct trace iterations"
            ),
        );
        let width_shift = direct_trace_width_shift(region.width_bytes);
        let generic_len_branch = if width_shift == 0 {
            self.emit(
                mov_reg(BYTE_LEN_HOST, ITERATIONS_HOST),
                format!("mov x{BYTE_LEN_HOST}, x{ITERATIONS_HOST} ; direct trace byte length"),
            );
            None
        } else {
            self.emit_shift_imm64_host(
                SCRATCH2,
                ITERATIONS_HOST,
                64 - width_shift,
                ShiftImmediateKind::Lsr,
                " ; direct trace byte length overflow",
            );
            self.emit(
                cmp_reg(SCRATCH2, A64_ZERO_REGISTER),
                format!("cmp x{SCRATCH2}, xzr ; direct trace byte length overflow"),
            );
            let branch = self.emit_patchable_branch("b.ne .generic_trace_loop".to_string());
            self.emit_shift_imm64_host(
                BYTE_LEN_HOST,
                ITERATIONS_HOST,
                width_shift,
                ShiftImmediateKind::Lsl,
                " ; direct trace byte length",
            );
            Some(branch)
        };
        self.mov_imm64(SCRATCH2, region.guest_instruction_count);
        self.emit(
            mul_reg(DYNAMIC_INSTRUCTION_COUNT, ITERATIONS_HOST, SCRATCH2),
            format!(
                "mul x{DYNAMIC_INSTRUCTION_COUNT}, x{ITERATIONS_HOST}, x{SCRATCH2} ; direct trace instruction count"
            ),
        );

        self.emit(
            mov_reg(0, CPU_PTR),
            format!("mov x0, x{CPU_PTR} ; direct read cpu"),
        );
        self.emit(
            mov_reg(1, address_host),
            format!("mov x1, x{address_host} ; direct read address"),
        );
        self.emit(
            mov_reg(2, BYTE_LEN_HOST),
            format!("mov x2, x{BYTE_LEN_HOST} ; direct read length"),
        );
        self.emit_call(
            jit_runtime_try_direct_read_ptr as *const () as usize as u64,
            "jit_runtime_try_direct_read_ptr",
        );
        self.emit(
            mov_reg(READ_PTR_HOST, 0),
            format!("mov x{READ_PTR_HOST}, x0 ; direct trace read host pointer"),
        );
        self.emit(
            cmp_reg(READ_PTR_HOST, A64_ZERO_REGISTER),
            format!("cmp x{READ_PTR_HOST}, xzr ; direct trace read available"),
        );
        let generic_ptr_branch = self.emit_patchable_branch("b.eq .generic_trace_loop".to_string());

        let mut side_exits = Vec::new();
        self.emit(
            cmp_imm(ITERATIONS_HOST, UNROLLED_LOADS),
            format!("cmp x{ITERATIONS_HOST}, #{UNROLLED_LOADS} ; direct trace unroll count"),
        );
        let tail_branch = self.emit_patchable_branch("b.lo .direct_trace_load_tail".to_string());

        let unrolled_loop = self.current_offset();
        for lane in 0..UNROLLED_LOADS {
            let offset = lane * region.width_bytes as u16;
            self.emit_direct_unsigned_load(
                loaded_host,
                READ_PTR_HOST,
                offset,
                region.width_bytes,
                format!("direct trace unrolled load {lane}"),
            );
            let branch = self.emit_patchable_cbz(
                loaded_host,
                format!("cbz .direct_trace_load_side_exit_{lane}"),
            );
            side_exits.push((branch, lane));
        }
        self.emit_host_add_sub_imm_any(
            READ_PTR_HOST,
            READ_PTR_HOST,
            i64::from(UNROLLED_LOADS) * region.width_bytes as i64,
        );
        self.emit(
            subs_imm(ITERATIONS_HOST, ITERATIONS_HOST, UNROLLED_LOADS),
            format!(
                "subs x{ITERATIONS_HOST}, x{ITERATIONS_HOST}, #{UNROLLED_LOADS} ; direct trace unrolled count"
            ),
        );
        self.emit(
            cmp_imm(ITERATIONS_HOST, UNROLLED_LOADS),
            format!("cmp x{ITERATIONS_HOST}, #{UNROLLED_LOADS} ; direct trace unrolled loop guard"),
        );
        let unrolled_loop_branch = self.current_offset();
        self.emit(
            b_cond(unrolled_loop_branch, unrolled_loop, A64Cond::Hs),
            "b.hs .direct_trace_load_unrolled_loop".to_string(),
        );

        let tail_offset = self.current_offset();
        self.patch_branch(tail_branch, b_cond(tail_branch, tail_offset, A64Cond::Lo));
        self.emit(
            cmp_reg(ITERATIONS_HOST, A64_ZERO_REGISTER),
            format!("cmp x{ITERATIONS_HOST}, xzr ; direct trace tail count"),
        );
        let done_branch = self.emit_patchable_branch("b.eq .direct_trace_load_done".to_string());

        let tail_loop = self.current_offset();
        self.emit_direct_unsigned_load(
            loaded_host,
            READ_PTR_HOST,
            0,
            region.width_bytes,
            "direct trace tail load".to_string(),
        );
        let tail_side_exit = self.emit_patchable_cbz(
            loaded_host,
            "cbz .direct_trace_load_tail_side_exit".to_string(),
        );
        side_exits.push((tail_side_exit, 0));
        self.emit_host_add_sub_imm_any(READ_PTR_HOST, READ_PTR_HOST, region.width_bytes as i64);
        self.emit(
            subs_imm(ITERATIONS_HOST, ITERATIONS_HOST, 1),
            format!("subs x{ITERATIONS_HOST}, x{ITERATIONS_HOST}, #1 ; direct trace tail guard"),
        );
        let tail_loop_branch = self.current_offset();
        self.emit(
            b_cond(tail_loop_branch, tail_loop, A64Cond::Ne),
            "b.ne .direct_trace_load_tail_loop".to_string(),
        );

        let done_offset = self.current_offset();
        self.patch_branch(done_branch, b_cond(done_branch, done_offset, A64Cond::Eq));
        self.emit(
            mov_reg(counter_host, limit_host),
            format!("mov x{counter_host}, x{limit_host} ; direct trace final counter"),
        );
        self.emit(
            add_reg(address_host, address_host, BYTE_LEN_HOST),
            format!("add x{address_host}, x{address_host}, x{BYTE_LEN_HOST} ; direct trace final address"),
        );
        self.flush_regalloc_dirty_registers(&trace_loop.dirty_registers);
        self.store_pc(region.loop_exit_pc);
        self.emit(
            mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
            format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return direct trace instructions"),
        );
        self.emit_epilogue(true);
        self.ret();

        for (side_exit_branch, completed_extra) in side_exits {
            let side_exit_offset = self.current_offset();
            self.patch_branch(
                side_exit_branch,
                cbz(side_exit_branch, side_exit_offset, loaded_host),
            );
            self.emit_direct_load_trace_side_exit(
                trace_loop,
                region,
                counter_host,
                address_host,
                BYTE_LEN_HOST,
                ITERATIONS_HOST,
                completed_extra,
            );
        }

        let generic_offset = self.current_offset();
        self.patch_branch(
            generic_count_branch,
            b_cond(generic_count_branch, generic_offset, A64Cond::Hs),
        );
        if let Some(generic_len_branch) = generic_len_branch {
            self.patch_branch(
                generic_len_branch,
                b_cond(generic_len_branch, generic_offset, A64Cond::Ne),
            );
        }
        self.patch_branch(
            generic_ptr_branch,
            b_cond(generic_ptr_branch, generic_offset, A64Cond::Eq),
        );
        self.emit_register_allocated_trace_loop_body(trace_loop);
    }

    fn emit_direct_unsigned_load(
        &mut self,
        dst: u8,
        base: u8,
        offset: u16,
        width_bytes: u64,
        detail: String,
    ) {
        let (word, mnemonic, register_prefix) = match width_bytes {
            1 => (ldr_u8(dst, base, offset), "ldrb", "w"),
            2 => (ldr_u16(dst, base, offset), "ldrh", "w"),
            4 => (ldr_u32(dst, base, offset), "ldr", "w"),
            8 => (ldr_u64(dst, base, offset), "ldr", "x"),
            _ => unreachable!("unsupported direct trace load width: {width_bytes}"),
        };
        self.emit(
            word,
            format!("{mnemonic} {register_prefix}{dst}, [x{base}, #{offset}] ; {detail}"),
        );
    }

    fn emit_direct_load_trace_side_exit(
        &mut self,
        trace_loop: &RegisterAllocatedTraceLoop<'_>,
        region: DirectLoadTraceLoop,
        counter_host: u8,
        address_host: u8,
        byte_len_host: u8,
        iterations_host: u8,
        completed_extra: u16,
    ) {
        let width_shift = direct_trace_width_shift(region.width_bytes);
        if width_shift == 0 {
            self.emit(
                mov_reg(SCRATCH2, byte_len_host),
                format!("mov x{SCRATCH2}, x{byte_len_host} ; direct trace total iterations"),
            );
        } else {
            self.emit_shift_imm64_host(
                SCRATCH2,
                byte_len_host,
                width_shift,
                ShiftImmediateKind::Lsr,
                " ; direct trace total iterations",
            );
        }
        self.emit(
            sub_reg(SCRATCH2, SCRATCH2, iterations_host),
            format!("sub x{SCRATCH2}, x{SCRATCH2}, x{iterations_host} ; direct trace completed iterations"),
        );
        self.emit_host_add_sub_imm_any(SCRATCH2, SCRATCH2, i64::from(completed_extra));
        if width_shift == 0 {
            self.emit(
                mov_reg(SCRATCH1, SCRATCH2),
                format!("mov x{SCRATCH1}, x{SCRATCH2} ; direct trace completed bytes"),
            );
        } else {
            self.emit_shift_imm64_host(
                SCRATCH1,
                SCRATCH2,
                width_shift,
                ShiftImmediateKind::Lsl,
                " ; direct trace completed bytes",
            );
        }
        self.emit(
            add_reg(counter_host, counter_host, SCRATCH2),
            format!(
                "add x{counter_host}, x{counter_host}, x{SCRATCH2} ; direct trace side counter"
            ),
        );
        self.emit(
            add_reg(address_host, address_host, SCRATCH1),
            format!(
                "add x{address_host}, x{address_host}, x{SCRATCH1} ; direct trace side address"
            ),
        );
        self.mov_imm64(SCRATCH1, region.guest_instruction_count);
        self.emit(
            mul_reg(DYNAMIC_INSTRUCTION_COUNT, SCRATCH2, SCRATCH1),
            format!(
                "mul x{DYNAMIC_INSTRUCTION_COUNT}, x{SCRATCH2}, x{SCRATCH1} ; direct trace side instruction count"
            ),
        );
        self.emit_host_add_sub_imm_any(
            DYNAMIC_INSTRUCTION_COUNT,
            DYNAMIC_INSTRUCTION_COUNT,
            region.guard_executed_instructions as i64,
        );
        self.flush_regalloc_dirty_registers(&trace_loop.dirty_registers);
        self.store_pc(region.guard_side_exit_pc);
        self.emit(
            mov_reg(0, DYNAMIC_INSTRUCTION_COUNT),
            format!("mov x0, x{DYNAMIC_INSTRUCTION_COUNT} ; return direct trace side instructions"),
        );
        self.emit_epilogue(true);
        self.ret();
    }

    fn emit_register_allocated_block(&mut self, block_plan: &RegisterAllocatedBlock<'_>) {
        self.emit_leaf_prologue(false);

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
        self.emit_leaf_epilogue(false);
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
        self.emit_loop_instruction_with_divisor_fixups(
            instruction,
            loop_plan,
            DivisorFixupMode::Compare,
        );
    }

    fn emit_loop_instruction_with_divisor_fixups(
        &mut self,
        instruction: NativeInstruction,
        loop_plan: &RegisterAllocatedLoop<'_>,
        divisor_fixups: DivisorFixupMode<'_>,
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
                self.emit_loop_logical_imm(rd, rs1, imm, loop_plan, LogicalImmediateOp::And)
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
            NativeInstruction::RuntimeBinary { rd, rs1, rs2, op }
                if runtime_binary_op_has_native_aarch64_lowering(op) =>
            {
                self.emit_loop_runtime_binary(rd, rs1, rs2, op, loop_plan, divisor_fixups)
            }
            NativeInstruction::InlinedJump { .. } | NativeInstruction::Nop => {}
            NativeInstruction::Or { rd, rs1, rs2 } => {
                self.emit_loop_binary_reg(rd, rs1, rs2, loop_plan, "orr", orr_reg)
            }
            NativeInstruction::Ori { rd, rs1, imm } => {
                self.emit_loop_logical_imm(rd, rs1, imm, loop_plan, LogicalImmediateOp::Orr)
            }
            NativeInstruction::Sll { rd, rs1, rs2 } => {
                self.emit_loop_shift_reg(rd, rs1, rs2, loop_plan, "lslv", lslv_reg)
            }
            NativeInstruction::Slli { rd, rs1, shamt } => {
                self.emit_loop_shift_imm(rd, rs1, shamt, loop_plan, ShiftImmediateKind::Lsl)
            }
            NativeInstruction::Slliw { rd, rs1, shamt } => {
                self.emit_loop_shift_imm32(rd, rs1, shamt, loop_plan, ShiftImmediateKind::Lsl)
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
                self.emit_loop_shift_imm(rd, rs1, shamt, loop_plan, ShiftImmediateKind::Asr)
            }
            NativeInstruction::Sraiw { rd, rs1, shamt } => {
                self.emit_loop_shift_imm32(rd, rs1, shamt, loop_plan, ShiftImmediateKind::Asr)
            }
            NativeInstruction::Sraw { rd, rs1, rs2 } => {
                self.emit_loop_word_binary_reg(rd, rs1, rs2, loop_plan, "sraw", asrv_reg32)
            }
            NativeInstruction::Srl { rd, rs1, rs2 } => {
                self.emit_loop_shift_reg(rd, rs1, rs2, loop_plan, "lsrv", lsrv_reg)
            }
            NativeInstruction::Srli { rd, rs1, shamt } => {
                self.emit_loop_shift_imm(rd, rs1, shamt, loop_plan, ShiftImmediateKind::Lsr)
            }
            NativeInstruction::Srliw { rd, rs1, shamt } => {
                self.emit_loop_shift_imm32(rd, rs1, shamt, loop_plan, ShiftImmediateKind::Lsr)
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
                self.emit_loop_logical_imm(rd, rs1, imm, loop_plan, LogicalImmediateOp::Eor)
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

    fn emit_leaf_prologue(&mut self, preserve_dynamic_count: bool) {
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
        self.emit(mov_reg(CPU_PTR, 0), format!("mov x{CPU_PTR}, x0"));
        self.emit(mov_reg(REG_PTR, 1), format!("mov x{REG_PTR}, x1"));
    }

    fn emit_selective_leaf_prologue(&mut self, registers: &[u8]) {
        for pair in registers.chunks(2) {
            let first = pair[0];
            let second = pair.get(1).copied().unwrap_or(A64_ZERO_REGISTER);
            self.emit(
                stp_pre(first, second, 31, -16),
                format!(
                    "stp x{first}, {}, [sp, #-16]! ; selective loop save",
                    a64_register_name(second)
                ),
            );
        }
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

    fn emit_leaf_epilogue(&mut self, restore_dynamic_count: bool) {
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
    }

    fn emit_selective_leaf_epilogue(&mut self, registers: &[u8]) {
        for pair in registers.chunks(2).rev() {
            let first = pair[0];
            let second = pair.get(1).copied().unwrap_or(A64_ZERO_REGISTER);
            self.emit(
                ldp_post(first, second, 31, 16),
                format!(
                    "ldp x{first}, {}, [sp], #16 ; selective loop restore",
                    a64_register_name(second)
                ),
            );
        }
    }

    fn store_pc(&mut self, pc: u64) {
        self.store_pc_to(pc, REG_PTR);
    }

    fn store_pc_to(&mut self, pc: u64, reg_ptr: u8) {
        self.mov_imm64(SCRATCH0, pc);
        self.store_guest_register_to(PC_REGISTER, SCRATCH0, reg_ptr);
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

    fn emit_runtime_binary(&mut self, rd: u8, rs1: u8, rs2: u8, op: RuntimeBinaryOp) {
        if let Some(kind) = runtime_binary_division_kind(op) {
            self.emit_divide_or_remainder(rd, rs1, rs2, kind);
        } else if op == RuntimeBinaryOp::Mulhsu {
            self.emit_mulhsu(rd, rs1, rs2);
        } else {
            self.emit_runtime_binary_call(rd, rs1, rs2, op);
        }
    }

    fn emit_runtime_binary_call(&mut self, rd: u8, rs1: u8, rs2: u8, op: RuntimeBinaryOp) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.mov_imm64(0, op as u64);
        self.emit(mov_reg(1, lhs), format!("mov x1, x{lhs} ; lhs"));
        self.emit(mov_reg(2, rhs), format!("mov x2, x{rhs} ; rhs"));
        self.emit_call(
            jit_runtime_binary as *const () as usize as u64,
            "jit_runtime_binary",
        );
        self.store_guest_register(rd, 0);
    }

    fn emit_mulhsu(&mut self, rd: u8, rs1: u8, rs2: u8) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit_mulhsu_host(SCRATCH2, lhs, rhs);
        self.store_guest_register(rd, SCRATCH2);
    }

    fn emit_mulhsu_host(&mut self, dst: u8, lhs: u8, rhs: u8) {
        let product = if dst != lhs && dst != rhs {
            dst
        } else {
            SCRATCH2
        };
        let correction = if product == SCRATCH3 {
            SCRATCH2
        } else {
            SCRATCH3
        };
        self.emit(
            umulh_reg(product, lhs, rhs),
            format!("umulh x{product}, x{lhs}, x{rhs} ; mulhsu unsigned high"),
        );
        self.emit_shift_imm64_host(
            correction,
            lhs,
            63,
            ShiftImmediateKind::Asr,
            " ; mulhsu lhs sign mask",
        );
        self.emit(
            and_reg(correction, correction, rhs),
            format!("and x{correction}, x{correction}, x{rhs} ; mulhsu signed correction"),
        );
        self.emit(
            sub_reg(product, product, correction),
            format!("sub x{product}, x{product}, x{correction} ; mulhsu signed high"),
        );
        self.emit_host_move(dst, product, " ; mulhsu result");
    }

    fn emit_divide_or_remainder(&mut self, rd: u8, rs1: u8, rs2: u8, kind: DivisionKind) {
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        let rhs = self.load_guest_register_or_zero(SCRATCH1, rs2);
        self.emit_divide_or_remainder_host(SCRATCH2, lhs, rhs, kind);
        self.store_guest_register(rd, SCRATCH2);
    }

    fn emit_divide_or_remainder_host(&mut self, dst: u8, lhs: u8, rhs: u8, kind: DivisionKind) {
        self.emit_divide_or_remainder_host_with_fixup(
            dst,
            lhs,
            rhs,
            kind,
            DivisionZeroFixup::Compare,
        );
    }

    fn emit_divide_or_remainder_host_with_fixup(
        &mut self,
        dst: u8,
        lhs: u8,
        rhs: u8,
        kind: DivisionKind,
        zero_fixup: DivisionZeroFixup,
    ) {
        if kind.is_quotient() && matches!(zero_fixup, DivisionZeroFixup::Compare) {
            if kind.is_word() {
                self.emit(
                    cmp_reg32(rhs, A64_ZERO_REGISTER),
                    format!("cmp w{rhs}, wzr ; {} divisor zero", kind.mnemonic()),
                );
            } else {
                self.emit(
                    cmp_reg(rhs, A64_ZERO_REGISTER),
                    format!("cmp x{rhs}, xzr ; {} divisor zero", kind.mnemonic()),
                );
            }
        }

        if kind.is_word() {
            let division = if kind.is_signed() {
                sdiv_reg32(SCRATCH2, lhs, rhs)
            } else {
                udiv_reg32(SCRATCH2, lhs, rhs)
            };
            self.emit(
                division,
                format!("{} w{SCRATCH2}, w{lhs}, w{rhs}", kind.division_mnemonic()),
            );
            if kind.is_remainder() {
                self.emit(
                    msub_reg32(SCRATCH2, SCRATCH2, rhs, lhs),
                    format!(
                        "msub w{SCRATCH2}, w{SCRATCH2}, w{rhs}, w{lhs} ; {}",
                        kind.mnemonic()
                    ),
                );
            }
            self.sign_extend_word(SCRATCH2, SCRATCH2);
        } else {
            let division = if kind.is_signed() {
                sdiv_reg(SCRATCH2, lhs, rhs)
            } else {
                udiv_reg(SCRATCH2, lhs, rhs)
            };
            self.emit(
                division,
                format!("{} x{SCRATCH2}, x{lhs}, x{rhs}", kind.division_mnemonic()),
            );
            if kind.is_remainder() {
                self.emit(
                    msub_reg(SCRATCH2, SCRATCH2, rhs, lhs),
                    format!(
                        "msub x{SCRATCH2}, x{SCRATCH2}, x{rhs}, x{lhs} ; {}",
                        kind.mnemonic()
                    ),
                );
            }
        }

        if kind.is_quotient() {
            match zero_fixup {
                DivisionZeroFixup::Compare => {
                    self.emit(
                        csinv(dst, SCRATCH2, A64_ZERO_REGISTER, A64Cond::Ne),
                        format!(
                            "csinv x{dst}, x{SCRATCH2}, xzr, ne ; {} division result",
                            kind.mnemonic()
                        ),
                    );
                }
                DivisionZeroFixup::Mask(mask) => {
                    self.emit(
                        orr_reg(dst, SCRATCH2, mask),
                        format!(
                            "orr x{dst}, x{SCRATCH2}, x{mask} ; {} masked division result",
                            kind.mnemonic()
                        ),
                    );
                }
                DivisionZeroFixup::KnownNonZero => {
                    self.emit_host_move(dst, SCRATCH2, " ; nonzero division result");
                }
            }
        } else {
            self.emit_host_move(dst, SCRATCH2, " ; division result");
        }
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

    fn emit_loop_binary_reg_with_read_override(
        &mut self,
        operation: LoopBinaryRegOperation,
        override_guest: u8,
        override_host: u8,
        loop_plan: &RegisterAllocatedLoop<'_>,
        suffix: &str,
    ) {
        let Some(dst) = loop_plan.host_for_write(operation.rd) else {
            return;
        };
        let lhs = if operation.rs1 == override_guest {
            override_host
        } else {
            loop_plan.host_or_zero(operation.rs1)
        };
        let rhs = if operation.rs2 == override_guest {
            override_host
        } else {
            loop_plan.host_or_zero(operation.rs2)
        };
        self.emit(
            (operation.op)(dst, lhs, rhs),
            format!("{} x{dst}, x{lhs}, x{rhs}{suffix}", operation.mnemonic),
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

    fn emit_loop_runtime_binary(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        op: RuntimeBinaryOp,
        loop_plan: &RegisterAllocatedLoop<'_>,
        divisor_fixups: DivisorFixupMode<'_>,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let lhs = loop_plan.host_or_zero(rs1);
        let rhs = loop_plan.host_or_zero(rs2);
        if let Some(kind) = runtime_binary_division_kind(op) {
            let zero_fixup = divisor_fixups.fixup_for(rs2, kind);
            self.emit_divide_or_remainder_host_with_fixup(dst, lhs, rhs, kind, zero_fixup);
        } else {
            debug_assert_eq!(op, RuntimeBinaryOp::Mulhsu);
            self.emit_mulhsu_host(dst, lhs, rhs);
        }
    }

    fn emit_loop_div_rem_pair(
        &mut self,
        pair: DivRemPair,
        loop_plan: &RegisterAllocatedLoop<'_>,
        divisor_fixups: DivisorFixupMode<'_>,
    ) {
        self.emit_loop_div_rem_pair_with_remainder_host(
            pair,
            loop_plan,
            divisor_fixups,
            SCRATCH3,
            true,
            true,
        );
    }

    fn emit_loop_div_rem_pair_deferred(
        &mut self,
        pair: DivRemPair,
        loop_plan: &RegisterAllocatedLoop<'_>,
        divisor_fixups: DivisorFixupMode<'_>,
    ) {
        self.emit_loop_div_rem_pair_with_remainder_host(
            pair,
            loop_plan,
            divisor_fixups,
            DIV_REM_CACHE,
            false,
            true,
        );
    }

    fn emit_loop_div_rem_pair_fully_deferred(
        &mut self,
        pair: DivRemPair,
        loop_plan: &RegisterAllocatedLoop<'_>,
        divisor_fixups: DivisorFixupMode<'_>,
    ) {
        self.emit_loop_div_rem_pair_with_remainder_host(
            pair,
            loop_plan,
            divisor_fixups,
            DIV_REM_CACHE,
            false,
            false,
        );
    }

    fn emit_loop_div_rem_pair_with_remainder_host(
        &mut self,
        pair: DivRemPair,
        loop_plan: &RegisterAllocatedLoop<'_>,
        divisor_fixups: DivisorFixupMode<'_>,
        remainder_host: u8,
        store_remainder: bool,
        store_quotient: bool,
    ) {
        let quotient_dst = loop_plan.host_for_write(pair.quotient_register);
        let remainder_dst = loop_plan.host_for_write(pair.remainder_register);
        let lhs = loop_plan.host_or_zero(pair.rs1);
        let rhs = loop_plan.host_or_zero(pair.rs2);
        let kind = pair.quotient_kind;
        let zero_fixup = divisor_fixups.fixup_for(pair.rs2, kind);

        if kind.is_word() {
            if matches!(zero_fixup, DivisionZeroFixup::Compare) {
                self.emit(
                    cmp_reg32(rhs, A64_ZERO_REGISTER),
                    format!("cmp w{rhs}, wzr ; fused {} divisor zero", kind.mnemonic()),
                );
            }
            let division = if kind.is_signed() {
                sdiv_reg32(SCRATCH2, lhs, rhs)
            } else {
                udiv_reg32(SCRATCH2, lhs, rhs)
            };
            self.emit(
                division,
                format!(
                    "{} w{SCRATCH2}, w{lhs}, w{rhs} ; fused div/rem",
                    kind.division_mnemonic()
                ),
            );
            if remainder_dst.is_some() {
                self.emit(
                    msub_reg32(remainder_host, SCRATCH2, rhs, lhs),
                    format!(
                        "msub w{remainder_host}, w{SCRATCH2}, w{rhs}, w{lhs} ; fused remainder"
                    ),
                );
                self.sign_extend_word(remainder_host, remainder_host);
            }
            if let Some(dst) = quotient_dst.filter(|_| store_quotient) {
                self.sign_extend_word(SCRATCH2, SCRATCH2);
                self.emit_fused_quotient_fixup(dst, kind, zero_fixup);
            } else if !store_quotient {
                self.sign_extend_word(SCRATCH2, SCRATCH2);
                self.emit_fused_quotient_fixup(SCRATCH2, kind, zero_fixup);
            }
        } else {
            if matches!(zero_fixup, DivisionZeroFixup::Compare) {
                self.emit(
                    cmp_reg(rhs, A64_ZERO_REGISTER),
                    format!("cmp x{rhs}, xzr ; fused {} divisor zero", kind.mnemonic()),
                );
            }
            let division = if kind.is_signed() {
                sdiv_reg(SCRATCH2, lhs, rhs)
            } else {
                udiv_reg(SCRATCH2, lhs, rhs)
            };
            self.emit(
                division,
                format!(
                    "{} x{SCRATCH2}, x{lhs}, x{rhs} ; fused div/rem",
                    kind.division_mnemonic()
                ),
            );
            if remainder_dst.is_some() {
                self.emit(
                    msub_reg(remainder_host, SCRATCH2, rhs, lhs),
                    format!(
                        "msub x{remainder_host}, x{SCRATCH2}, x{rhs}, x{lhs} ; fused remainder"
                    ),
                );
            }
            if let Some(dst) = quotient_dst.filter(|_| store_quotient) {
                self.emit_fused_quotient_fixup(dst, kind, zero_fixup);
            } else if !store_quotient {
                self.emit_fused_quotient_fixup(SCRATCH2, kind, zero_fixup);
            }
        }

        if store_remainder {
            if let Some(dst) = remainder_dst {
                self.emit_host_move(dst, remainder_host, " ; fused remainder");
            }
        }
    }

    fn emit_fused_quotient_fixup(
        &mut self,
        dst: u8,
        kind: DivisionKind,
        zero_fixup: DivisionZeroFixup,
    ) {
        match zero_fixup {
            DivisionZeroFixup::Compare => {
                self.emit(
                    csinv(dst, SCRATCH2, A64_ZERO_REGISTER, A64Cond::Ne),
                    format!(
                        "csinv x{dst}, x{SCRATCH2}, xzr, ne ; fused quotient {} divide by zero",
                        kind.mnemonic()
                    ),
                );
            }
            DivisionZeroFixup::Mask(mask) => {
                self.emit(
                    orr_reg(dst, SCRATCH2, mask),
                    format!(
                        "orr x{dst}, x{SCRATCH2}, x{mask} ; fused quotient {} divisor mask",
                        kind.mnemonic()
                    ),
                );
            }
            DivisionZeroFixup::KnownNonZero => {
                self.emit_host_move(dst, SCRATCH2, " ; fused quotient nonzero divisor");
            }
        }
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
        op: LogicalImmediateOp,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let value = imm as u64;
        let src = loop_plan.host_or_zero(rs1);

        if self.emit_loop_logical_imm_shortcut(dst, src, value, op) {
            return;
        }

        if let Some(encoded) = encode_logical_immediate(value, 64) {
            self.emit(
                op.imm64(dst, src, encoded),
                format!(
                    "{} x{dst}, x{src}, #0x{value:016x} ; regalloc",
                    op.mnemonic()
                ),
            );
            return;
        }

        self.mov_imm64(LOOP_TEMP, value);
        self.emit(
            op.reg(dst, src, LOOP_TEMP),
            format!("{} x{dst}, x{src}, x{LOOP_TEMP} ; regalloc", op.mnemonic()),
        );
    }

    fn emit_loop_logical_imm_shortcut(
        &mut self,
        dst: u8,
        src: u8,
        value: u64,
        op: LogicalImmediateOp,
    ) -> bool {
        match op {
            LogicalImmediateOp::And if value == 0 => {
                self.emit_host_move(dst, A64_ZERO_REGISTER, " ; regalloc");
                true
            }
            LogicalImmediateOp::And if value == u64::MAX => {
                self.emit_host_move(dst, src, " ; regalloc");
                true
            }
            LogicalImmediateOp::Orr | LogicalImmediateOp::Eor if value == 0 => {
                self.emit_host_move(dst, src, " ; regalloc");
                true
            }
            LogicalImmediateOp::Orr if value == u64::MAX => {
                self.mov_imm64(dst, value);
                true
            }
            LogicalImmediateOp::Eor if value == u64::MAX => {
                self.emit(
                    orn_reg(dst, A64_ZERO_REGISTER, src),
                    format!("mvn x{dst}, x{src}"),
                );
                true
            }
            _ => false,
        }
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
        kind: ShiftImmediateKind,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let src = loop_plan.host_or_zero(rs1);
        self.emit_shift_imm64_host(dst, src, shamt, kind, " ; regalloc");
    }

    fn emit_loop_shift_imm32(
        &mut self,
        rd: u8,
        rs1: u8,
        shamt: u32,
        loop_plan: &RegisterAllocatedLoop<'_>,
        kind: ShiftImmediateKind,
    ) {
        let Some(dst) = loop_plan.host_for_write(rd) else {
            return;
        };
        let src = loop_plan.host_or_zero(rs1);
        self.emit_shift_imm32_host(dst, src, shamt, kind, " ; regalloc");
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
        self.emit(
            cset(dst, condition),
            format!("cset x{dst}, {}", condition.mnemonic()),
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
        if rs1 == 0 {
            self.mov_imm64(dst, compare_zero_immediate(imm, condition));
            return;
        }
        let lhs = loop_plan.host_or_zero(rs1);
        if !self.emit_cmp_imm(lhs, imm, &format!(" ; {mnemonic} regalloc")) {
            self.mov_imm64(LOOP_TEMP, imm as u64);
            self.emit(
                cmp_reg(lhs, LOOP_TEMP),
                format!("cmp x{lhs}, x{LOOP_TEMP} ; {mnemonic} regalloc"),
            );
        }
        self.emit(
            cset(dst, condition),
            format!("cset x{dst}, {}", condition.mnemonic()),
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

    fn emit_logical_imm(&mut self, rd: u8, rs1: u8, imm: i64, op: LogicalImmediateOp) {
        if rd == 0 {
            return;
        }

        let value = imm as u64;
        let src = self.load_guest_register_or_zero(SCRATCH0, rs1);

        if self.emit_logical_imm_shortcut(src, value, op) {
            self.store_guest_register(rd, SCRATCH0);
            return;
        }

        if let Some(encoded) = encode_logical_immediate(value, 64) {
            self.emit(
                op.imm64(SCRATCH0, src, encoded),
                format!("{} x{SCRATCH0}, x{src}, #0x{value:016x}", op.mnemonic()),
            );
            self.store_guest_register(rd, SCRATCH0);
            return;
        }

        self.mov_imm64(SCRATCH1, value);
        self.emit(
            op.reg(SCRATCH0, src, SCRATCH1),
            format!("{} x{SCRATCH0}, x{src}, x{SCRATCH1}", op.mnemonic()),
        );
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_logical_imm_shortcut(&mut self, src: u8, value: u64, op: LogicalImmediateOp) -> bool {
        match op {
            LogicalImmediateOp::And if value == 0 => {
                self.emit_host_move(SCRATCH0, A64_ZERO_REGISTER, "");
                true
            }
            LogicalImmediateOp::And if value == u64::MAX => {
                self.emit_host_move(SCRATCH0, src, "");
                true
            }
            LogicalImmediateOp::Orr | LogicalImmediateOp::Eor if value == 0 => {
                self.emit_host_move(SCRATCH0, src, "");
                true
            }
            LogicalImmediateOp::Orr if value == u64::MAX => {
                self.mov_imm64(SCRATCH0, value);
                true
            }
            LogicalImmediateOp::Eor if value == u64::MAX => {
                self.emit(
                    orn_reg(SCRATCH0, A64_ZERO_REGISTER, src),
                    format!("mvn x{SCRATCH0}, x{src}"),
                );
                true
            }
            _ => false,
        }
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

    fn emit_shift_imm(&mut self, rd: u8, rs1: u8, shamt: u32, kind: ShiftImmediateKind) {
        self.load_guest_register(SCRATCH0, rs1);
        self.emit_shift_imm64_host(SCRATCH0, SCRATCH0, shamt, kind, "");
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

    fn emit_shift_imm32(&mut self, rd: u8, rs1: u8, shamt: u32, kind: ShiftImmediateKind) {
        self.load_guest_register(SCRATCH0, rs1);
        self.emit_shift_imm32_host(SCRATCH0, SCRATCH0, shamt, kind, "");
        self.sign_extend_word(SCRATCH0, SCRATCH0);
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_shift_imm64_host(
        &mut self,
        rd: u8,
        rn: u8,
        shamt: u32,
        kind: ShiftImmediateKind,
        suffix: &str,
    ) {
        debug_assert!(shamt < 64);
        if shamt == 0 {
            if rd != rn {
                self.emit(mov_reg(rd, rn), format!("mov x{rd}, x{rn}{suffix}"));
            }
            return;
        }

        self.emit(
            kind.encode64(rd, rn, shamt),
            format!("{} x{rd}, x{rn}, #{shamt}{suffix}", kind.mnemonic()),
        );
    }

    fn emit_shift_imm32_host(
        &mut self,
        rd: u8,
        rn: u8,
        shamt: u32,
        kind: ShiftImmediateKind,
        suffix: &str,
    ) {
        debug_assert!(shamt < 32);
        if shamt == 0 {
            if rd != rn {
                self.emit(mov_reg(rd, rn), format!("mov w{rd}, w{rn}{suffix}"));
            }
            return;
        }

        self.emit(
            kind.encode32(rd, rn, shamt),
            format!("{} w{rd}, w{rn}, #{shamt}{suffix}", kind.mnemonic()),
        );
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
        if rs1 == 0 {
            self.mov_imm64(SCRATCH0, compare_zero_immediate(imm, condition));
            self.store_guest_register(rd, SCRATCH0);
            return;
        }
        let lhs = self.load_guest_register_or_zero(SCRATCH0, rs1);
        if !self.emit_cmp_imm(lhs, imm, &format!(" ; {mnemonic}")) {
            self.mov_imm64(SCRATCH1, imm as u64);
            self.emit(
                cmp_reg(lhs, SCRATCH1),
                format!("cmp x{lhs}, x{SCRATCH1} ; {mnemonic}"),
            );
        }
        self.emit_compare_result(rd, condition);
    }

    fn emit_compare_result(&mut self, rd: u8, condition: A64Cond) {
        self.emit(
            cset(SCRATCH0, condition),
            format!("cset x{SCRATCH0}, {}", condition.mnemonic()),
        );
        self.store_guest_register(rd, SCRATCH0);
    }

    fn emit_cmp_imm(&mut self, rn: u8, imm: i64, suffix: &str) -> bool {
        if rn == A64_ZERO_REGISTER {
            return false;
        }
        if (0..4096).contains(&imm) {
            self.emit(
                cmp_imm(rn, imm as u16),
                format!("cmp x{rn}, #{imm}{suffix}"),
            );
            true
        } else if (-4095..0).contains(&imm) {
            let magnitude = (-imm) as u16;
            self.emit(
                cmn_imm(rn, magnitude),
                format!("cmn x{rn}, #{magnitude}{suffix}"),
            );
            true
        } else {
            false
        }
    }

    fn sign_extend_word(&mut self, rd: u8, rn: u8) {
        self.emit(sxtw(rd, rn), format!("sxtw x{rd}, w{rn}"));
    }

    fn zero_extend_word(&mut self, rd: u8, rn: u8) {
        self.emit(ubfm64(rd, rn, 0, 31), format!("uxtw x{rd}, w{rn}"));
    }

    fn load_guest_register(&mut self, host: u8, guest: u8) {
        self.load_guest_register_from(host, guest, REG_PTR);
    }

    fn load_guest_register_from(&mut self, host: u8, guest: u8, reg_ptr: u8) {
        if guest == 0 {
            self.emit(
                mov_reg(host, A64_ZERO_REGISTER),
                format!("mov x{host}, xzr"),
            );
            return;
        }
        let offset = u16::from(guest) * 8;
        self.emit(
            ldr_u64(host, reg_ptr, offset),
            format!("ldr x{host}, [x{reg_ptr}, #{offset}] ; load guest x{guest}"),
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
        self.store_guest_register_to(guest, host, REG_PTR);
    }

    fn store_guest_register_to(&mut self, guest: u8, host: u8, reg_ptr: u8) {
        if guest == 0 {
            return;
        }
        let offset = u16::from(guest) * 8;
        self.emit(
            str_u64(host, reg_ptr, offset),
            format!("str x{host}, [x{reg_ptr}, #{offset}] ; store guest x{guest}"),
        );
    }

    fn emit_host_move(&mut self, rd: u8, rn: u8, suffix: &str) {
        if rd == rn {
            return;
        }
        let src = if rn == A64_ZERO_REGISTER {
            "xzr".to_string()
        } else {
            format!("x{rn}")
        };
        self.emit(mov_reg(rd, rn), format!("mov x{rd}, {src}{suffix}"));
    }

    fn mov_imm64(&mut self, rd: u8, value: u64) {
        if value == 0 {
            self.emit(mov_reg(rd, A64_ZERO_REGISTER), format!("mov x{rd}, xzr"));
            return;
        }
        if value == u64::MAX {
            self.emit(
                orn_reg(rd, A64_ZERO_REGISTER, A64_ZERO_REGISTER),
                format!("mvn x{rd}, xzr"),
            );
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

    fn emit_patchable_cbz(&mut self, rt: u8, text: String) -> usize {
        let offset = self.code.len();
        self.code.extend(cbz(offset, offset, rt).to_le_bytes());
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

fn compare_zero_immediate(imm: i64, condition: A64Cond) -> u64 {
    let result = match condition {
        A64Cond::Eq => imm == 0,
        A64Cond::Ne => imm != 0,
        A64Cond::Hs => 0u64 >= imm as u64,
        A64Cond::Lo => 0u64 < imm as u64,
        A64Cond::Ge => 0 >= imm,
        A64Cond::Lt => 0 < imm,
    };
    u64::from(result)
}

#[derive(Debug, Clone, Copy)]
enum LogicalImmediateOp {
    And,
    Orr,
    Eor,
}

impl LogicalImmediateOp {
    fn mnemonic(self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Orr => "orr",
            Self::Eor => "eor",
        }
    }

    fn reg(self, rd: u8, rn: u8, rm: u8) -> u32 {
        match self {
            Self::And => and_reg(rd, rn, rm),
            Self::Orr => orr_reg(rd, rn, rm),
            Self::Eor => eor_reg(rd, rn, rm),
        }
    }

    fn imm64(self, rd: u8, rn: u8, imm: LogicalImmediate) -> u32 {
        logical_imm64(self.opc(), rd, rn, imm)
    }

    fn opc(self) -> u32 {
        match self {
            Self::And => 0,
            Self::Orr => 1,
            Self::Eor => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogicalImmediate {
    n: u32,
    immr: u32,
    imms: u32,
}

fn encode_logical_immediate(value: u64, width: u32) -> Option<LogicalImmediate> {
    debug_assert!(matches!(width, 32 | 64));
    let width_mask = bitmask(width);
    let value = value & width_mask;
    if value == 0 || value == width_mask {
        return None;
    }

    for element_size in [2, 4, 8, 16, 32, 64] {
        if element_size > width {
            break;
        }
        let element_mask = bitmask(element_size);
        let element = value & element_mask;
        if element == 0 || element == element_mask {
            continue;
        }
        if replicate_element(element, element_size, width) != value {
            continue;
        }

        for rotation in 0..element_size {
            let unrotated = rotate_left(element, rotation, element_size);
            if !is_low_ones(unrotated) {
                continue;
            }
            let ones = unrotated.count_ones();
            let imms = (!(element_size * 2 - 1) & 0x3f) | (ones - 1);
            return Some(LogicalImmediate {
                n: u32::from(element_size == 64),
                immr: rotation & (element_size - 1),
                imms,
            });
        }
    }

    None
}

fn bitmask(width: u32) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn replicate_element(element: u64, element_size: u32, width: u32) -> u64 {
    let mut replicated = 0;
    let mut offset = 0;
    while offset < width {
        replicated |= element << offset;
        offset += element_size;
    }
    replicated & bitmask(width)
}

fn rotate_left(value: u64, rotation: u32, width: u32) -> u64 {
    let rotation = rotation % width;
    let mask = bitmask(width);
    if rotation == 0 {
        value & mask
    } else {
        ((value << rotation) | (value >> (width - rotation))) & mask
    }
}

fn is_low_ones(value: u64) -> bool {
    value != 0 && (value & (value + 1)) == 0
}

#[derive(Debug, Clone, Copy)]
enum ShiftImmediateKind {
    Lsl,
    Lsr,
    Asr,
}

impl ShiftImmediateKind {
    fn mnemonic(self) -> &'static str {
        match self {
            Self::Lsl => "lsl",
            Self::Lsr => "lsr",
            Self::Asr => "asr",
        }
    }

    fn encode64(self, rd: u8, rn: u8, shamt: u32) -> u32 {
        debug_assert!((1..64).contains(&shamt));
        match self {
            Self::Lsl => ubfm64(rd, rn, 64 - shamt, 63 - shamt),
            Self::Lsr => ubfm64(rd, rn, shamt, 63),
            Self::Asr => sbfm64(rd, rn, shamt, 63),
        }
    }

    fn encode32(self, rd: u8, rn: u8, shamt: u32) -> u32 {
        debug_assert!((1..32).contains(&shamt));
        match self {
            Self::Lsl => ubfm32(rd, rn, 32 - shamt, 31 - shamt),
            Self::Lsr => ubfm32(rd, rn, shamt, 31),
            Self::Asr => sbfm32(rd, rn, shamt, 31),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DivisionKind {
    Signed64Quotient,
    Unsigned64Quotient,
    Signed32Quotient,
    Unsigned32Quotient,
    Signed64Remainder,
    Unsigned64Remainder,
    Signed32Remainder,
    Unsigned32Remainder,
}

impl DivisionKind {
    fn is_signed(self) -> bool {
        matches!(
            self,
            Self::Signed64Quotient
                | Self::Signed32Quotient
                | Self::Signed64Remainder
                | Self::Signed32Remainder
        )
    }

    fn is_word(self) -> bool {
        matches!(
            self,
            Self::Signed32Quotient
                | Self::Unsigned32Quotient
                | Self::Signed32Remainder
                | Self::Unsigned32Remainder
        )
    }

    fn is_quotient(self) -> bool {
        matches!(
            self,
            Self::Signed64Quotient
                | Self::Unsigned64Quotient
                | Self::Signed32Quotient
                | Self::Unsigned32Quotient
        )
    }

    fn is_remainder(self) -> bool {
        !self.is_quotient()
    }

    fn mnemonic(self) -> &'static str {
        match self {
            Self::Signed64Quotient => "div",
            Self::Unsigned64Quotient => "divu",
            Self::Signed32Quotient => "divw",
            Self::Unsigned32Quotient => "divuw",
            Self::Signed64Remainder => "rem",
            Self::Unsigned64Remainder => "remu",
            Self::Signed32Remainder => "remw",
            Self::Unsigned32Remainder => "remuw",
        }
    }

    fn division_mnemonic(self) -> &'static str {
        if self.is_signed() {
            "sdiv"
        } else {
            "udiv"
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DivRemPair {
    quotient_register: u8,
    remainder_register: u8,
    rs1: u8,
    rs2: u8,
    quotient_kind: DivisionKind,
}

#[derive(Clone, Copy)]
struct InvariantDivisorMask {
    guest: u8,
    word: bool,
    host: u8,
}

#[derive(Default)]
struct InvariantDivisorMasks {
    masks: Vec<InvariantDivisorMask>,
}

impl InvariantDivisorMasks {
    fn from_loop_plan(loop_plan: &RegisterAllocatedLoop<'_>) -> Self {
        if !loop_preserves_invariant_divisor_masks(loop_plan) {
            return Self::default();
        }

        let mut masks: Vec<InvariantDivisorMask> = Vec::new();
        for operation in loop_plan.operations {
            let BlockOperationKind::Native(instruction) = operation.kind();
            let NativeInstruction::RuntimeBinary { rs2, op, .. } = instruction else {
                continue;
            };
            let Some(kind) = runtime_binary_division_kind(op) else {
                continue;
            };
            if !kind.is_quotient() || loop_plan.writes_guest_register(rs2) {
                continue;
            }

            let word = kind.is_word();
            if masks
                .iter()
                .any(|mask| mask.guest == rs2 && mask.word == word)
            {
                continue;
            }

            let Some(&host) = DIVISOR_MASK_HOST_REGISTERS.get(masks.len()) else {
                return Self::default();
            };
            masks.push(InvariantDivisorMask {
                guest: rs2,
                word,
                host,
            });
        }

        Self { masks }
    }

    fn find(&self, guest: u8, kind: DivisionKind) -> Option<u8> {
        let word = kind.is_word();
        self.masks
            .iter()
            .find(|mask| mask.guest == guest && mask.word == word)
            .map(|mask| mask.host)
    }

    fn is_empty(&self) -> bool {
        self.masks.is_empty()
    }
}

#[derive(Clone, Copy)]
enum DivisorFixupMode<'a> {
    Compare,
    Masked(&'a InvariantDivisorMasks),
    KnownNonZero(&'a InvariantDivisorMasks),
}

impl DivisorFixupMode<'_> {
    fn fixup_for(self, guest: u8, kind: DivisionKind) -> DivisionZeroFixup {
        match self {
            Self::Compare => DivisionZeroFixup::Compare,
            Self::Masked(masks) => masks
                .find(guest, kind)
                .map(DivisionZeroFixup::Mask)
                .unwrap_or(DivisionZeroFixup::Compare),
            Self::KnownNonZero(masks) => {
                if masks.find(guest, kind).is_some() {
                    DivisionZeroFixup::KnownNonZero
                } else {
                    DivisionZeroFixup::Compare
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum DivisionZeroFixup {
    Compare,
    Mask(u8),
    KnownNonZero,
}

fn loop_preserves_invariant_divisor_masks(loop_plan: &RegisterAllocatedLoop<'_>) -> bool {
    loop_plan.operations.iter().all(|operation| {
        let BlockOperationKind::Native(instruction) = operation.kind();
        !matches!(
            instruction,
            NativeInstruction::Load { .. }
                | NativeInstruction::Store { .. }
                | NativeInstruction::FloatLoad { .. }
                | NativeInstruction::FloatStore { .. }
                | NativeInstruction::RuntimeAtomic { .. }
                | NativeInstruction::RuntimeCsr { .. }
                | NativeInstruction::RuntimeFloat { .. }
                | NativeInstruction::Ecall { .. }
                | NativeInstruction::RuntimeTrap { .. }
        )
    })
}

fn loop_instruction_calls_rust_runtime(instruction: NativeInstruction) -> bool {
    matches!(
        instruction,
        NativeInstruction::Load { .. } | NativeInstruction::Store { .. }
    )
}

fn loop_host_register_is_callee_saved(host: u8) -> bool {
    host == DYNAMIC_INSTRUCTION_COUNT || LOOP_HOST_REGISTERS.contains(&host)
}

fn loop_can_use_caller_saved_registers(operations: &[super::BlockOperation]) -> bool {
    operations.iter().all(|operation| {
        let BlockOperationKind::Native(instruction) = operation.kind();
        !loop_instruction_calls_rust_runtime(instruction)
    })
}

#[derive(Clone, Copy)]
struct LoopBinaryRegOperation {
    rd: u8,
    rs1: u8,
    rs2: u8,
    mnemonic: &'static str,
    op: fn(u8, u8, u8) -> u32,
}

fn loop_binary_reg_operation(instruction: NativeInstruction) -> Option<LoopBinaryRegOperation> {
    match instruction {
        NativeInstruction::Add { rd, rs1, rs2 } => Some(LoopBinaryRegOperation {
            rd,
            rs1,
            rs2,
            mnemonic: "add",
            op: add_reg,
        }),
        NativeInstruction::And { rd, rs1, rs2 } => Some(LoopBinaryRegOperation {
            rd,
            rs1,
            rs2,
            mnemonic: "and",
            op: and_reg,
        }),
        NativeInstruction::Mul { rd, rs1, rs2 } => Some(LoopBinaryRegOperation {
            rd,
            rs1,
            rs2,
            mnemonic: "mul",
            op: mul_reg,
        }),
        NativeInstruction::Or { rd, rs1, rs2 } => Some(LoopBinaryRegOperation {
            rd,
            rs1,
            rs2,
            mnemonic: "orr",
            op: orr_reg,
        }),
        NativeInstruction::Sub { rd, rs1, rs2 } => Some(LoopBinaryRegOperation {
            rd,
            rs1,
            rs2,
            mnemonic: "sub",
            op: sub_reg,
        }),
        NativeInstruction::Xor { rd, rs1, rs2 } => Some(LoopBinaryRegOperation {
            rd,
            rs1,
            rs2,
            mnemonic: "eor",
            op: eor_reg,
        }),
        _ => None,
    }
}

fn binary_reg_gap_consumer(
    instruction: NativeInstruction,
    remainder_register: u8,
) -> Option<LoopBinaryRegOperation> {
    let operation = loop_binary_reg_operation(instruction)?;
    if operation.rd == remainder_register
        || (operation.rs1 != remainder_register && operation.rs2 != remainder_register)
    {
        return None;
    }

    Some(operation)
}

fn fused_div_rem_pair(first: NativeInstruction, second: NativeInstruction) -> Option<DivRemPair> {
    let NativeInstruction::RuntimeBinary {
        rd: quotient_register,
        rs1,
        rs2,
        op: quotient_op,
    } = first
    else {
        return None;
    };
    let NativeInstruction::RuntimeBinary {
        rd: remainder_register,
        rs1: remainder_rs1,
        rs2: remainder_rs2,
        op: remainder_op,
    } = second
    else {
        return None;
    };

    if rs1 != remainder_rs1
        || rs2 != remainder_rs2
        || quotient_register == rs1
        || quotient_register == rs2
    {
        return None;
    }

    let quotient_kind = runtime_binary_division_kind(quotient_op)?;
    let remainder_kind = runtime_binary_division_kind(remainder_op)?;
    if !quotient_kind.is_quotient()
        || !remainder_kind.is_remainder()
        || quotient_kind.is_signed() != remainder_kind.is_signed()
        || quotient_kind.is_word() != remainder_kind.is_word()
    {
        return None;
    }

    Some(DivRemPair {
        quotient_register,
        remainder_register,
        rs1,
        rs2,
        quotient_kind,
    })
}

fn fused_div_rem_pair_across_pure_gap(
    first: NativeInstruction,
    middle: NativeInstruction,
    third: NativeInstruction,
) -> Option<DivRemPair> {
    let pair = fused_div_rem_pair(first, third)?;
    if !is_pure_integer_gap_instruction(middle)
        || writes_guest_register(middle, pair.rs1)
        || writes_guest_register(middle, pair.rs2)
    {
        return None;
    }

    Some(pair)
}

fn is_pure_integer_gap_instruction(instruction: NativeInstruction) -> bool {
    match instruction {
        NativeInstruction::Add { .. }
        | NativeInstruction::Addi { .. }
        | NativeInstruction::Addiw { .. }
        | NativeInstruction::Addw { .. }
        | NativeInstruction::And { .. }
        | NativeInstruction::Andi { .. }
        | NativeInstruction::Auipc { .. }
        | NativeInstruction::LoadImmediate { .. }
        | NativeInstruction::Lui { .. }
        | NativeInstruction::Move { .. }
        | NativeInstruction::Mul { .. }
        | NativeInstruction::Mulh { .. }
        | NativeInstruction::Mulhu { .. }
        | NativeInstruction::Mulw { .. }
        | NativeInstruction::Nop
        | NativeInstruction::Or { .. }
        | NativeInstruction::Ori { .. }
        | NativeInstruction::Sll { .. }
        | NativeInstruction::Slli { .. }
        | NativeInstruction::Slliw { .. }
        | NativeInstruction::Sllw { .. }
        | NativeInstruction::Slt { .. }
        | NativeInstruction::Slti { .. }
        | NativeInstruction::Sltiu { .. }
        | NativeInstruction::Sltu { .. }
        | NativeInstruction::Sra { .. }
        | NativeInstruction::Srai { .. }
        | NativeInstruction::Sraiw { .. }
        | NativeInstruction::Sraw { .. }
        | NativeInstruction::Srl { .. }
        | NativeInstruction::Srli { .. }
        | NativeInstruction::Srliw { .. }
        | NativeInstruction::Srlw { .. }
        | NativeInstruction::Sub { .. }
        | NativeInstruction::Subw { .. }
        | NativeInstruction::Xor { .. }
        | NativeInstruction::Xori { .. } => true,
        NativeInstruction::RuntimeBinary { op, .. } => {
            runtime_binary_op_has_native_aarch64_lowering(op)
        }
        _ => false,
    }
}

fn writes_guest_register(instruction: NativeInstruction, guest: u8) -> bool {
    if guest == 0 {
        return false;
    }

    let mut writes = false;
    collect_written_integer_registers(instruction, |written| {
        writes |= written == guest;
    });
    writes
}

fn guest_register_needed_before_next_write(
    operations: &[super::BlockOperation],
    start_index: usize,
    guest: u8,
) -> bool {
    for operation in &operations[start_index..] {
        let BlockOperationKind::Native(instruction) = operation.kind();
        if instruction_reads_guest_register(instruction, guest) {
            return true;
        }
        if writes_guest_register(instruction, guest) {
            return false;
        }
    }

    true
}

fn instruction_reads_guest_register(instruction: NativeInstruction, guest: u8) -> bool {
    if guest == 0 {
        return false;
    }

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
        | NativeInstruction::FloatStore { rs1, rs2, .. }
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
        | NativeInstruction::Store { rs1, rs2, .. }
        | NativeInstruction::Sub { rs1, rs2, .. }
        | NativeInstruction::Subw { rs1, rs2, .. }
        | NativeInstruction::TraceGuard { rs1, rs2, .. }
        | NativeInstruction::TraceLoopGuard { rs1, rs2, .. }
        | NativeInstruction::Xor { rs1, rs2, .. } => rs1 == guest || rs2 == guest,
        NativeInstruction::Addi { rs1, .. }
        | NativeInstruction::Addiw { rs1, .. }
        | NativeInstruction::AndBranch { rs1, .. }
        | NativeInstruction::Andi { rs1, .. }
        | NativeInstruction::FloatLoad { rs1, .. }
        | NativeInstruction::Jalr { rs1, .. }
        | NativeInstruction::JumpReg { rs1 }
        | NativeInstruction::JumpRegLink { rs1, .. }
        | NativeInstruction::Load { rs1, .. }
        | NativeInstruction::Ori { rs1, .. }
        | NativeInstruction::RuntimeCsr {
            rs1_or_uimm: rs1, ..
        }
        | NativeInstruction::Slli { rs1, .. }
        | NativeInstruction::Slliw { rs1, .. }
        | NativeInstruction::Slti { rs1, .. }
        | NativeInstruction::Sltiu { rs1, .. }
        | NativeInstruction::Srai { rs1, .. }
        | NativeInstruction::Sraiw { rs1, .. }
        | NativeInstruction::Srli { rs1, .. }
        | NativeInstruction::Srliw { rs1, .. }
        | NativeInstruction::Xori { rs1, .. } => rs1 == guest,
        NativeInstruction::Move { rs, .. } => rs == guest,
        NativeInstruction::RuntimeFloat { rs1, rs2, rs3, .. } => {
            rs1 == guest || rs2 == guest || rs3 == guest
        }
        NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::Auipc { .. }
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::DivisionRecurrenceLoop(_)
        | NativeInstruction::Ecall { .. }
        | NativeInstruction::FibonacciRecurrenceLoop(_)
        | NativeInstruction::InlinedJump { .. }
        | NativeInstruction::Jal { .. }
        | NativeInstruction::Jump { .. }
        | NativeInstruction::LoadImmediate { .. }
        | NativeInstruction::Lui { .. }
        | NativeInstruction::Nop
        | NativeInstruction::RuntimeTrap { .. }
        | NativeInstruction::StoreLoadForwardLoop(_) => false,
    }
}

fn runtime_binary_division_kind(op: RuntimeBinaryOp) -> Option<DivisionKind> {
    match op {
        RuntimeBinaryOp::Div => Some(DivisionKind::Signed64Quotient),
        RuntimeBinaryOp::Divu => Some(DivisionKind::Unsigned64Quotient),
        RuntimeBinaryOp::Divuw => Some(DivisionKind::Unsigned32Quotient),
        RuntimeBinaryOp::Divw => Some(DivisionKind::Signed32Quotient),
        RuntimeBinaryOp::Rem => Some(DivisionKind::Signed64Remainder),
        RuntimeBinaryOp::Remu => Some(DivisionKind::Unsigned64Remainder),
        RuntimeBinaryOp::Remuw => Some(DivisionKind::Unsigned32Remainder),
        RuntimeBinaryOp::Remw => Some(DivisionKind::Signed32Remainder),
        _ => None,
    }
}

fn runtime_binary_op_has_native_aarch64_lowering(op: RuntimeBinaryOp) -> bool {
    op == RuntimeBinaryOp::Mulhsu || runtime_binary_division_kind(op).is_some()
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

fn direct_trace_width_shift(width_bytes: u64) -> u32 {
    match width_bytes {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => unreachable!("unsupported direct trace load width: {width_bytes}"),
    }
}

fn a64_register_name(register: u8) -> String {
    if register == A64_ZERO_REGISTER {
        "xzr".to_string()
    } else {
        format!("x{register}")
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

fn ldr_u8(rt: u8, rn: u8, offset: u16) -> u32 {
    0x3940_0000 | (u32::from(offset) << 10) | (u32::from(rn) << 5) | u32::from(rt)
}

fn ldr_u16(rt: u8, rn: u8, offset: u16) -> u32 {
    debug_assert_eq!(offset % 2, 0);
    0x7940_0000 | (u32::from(offset / 2) << 10) | (u32::from(rn) << 5) | u32::from(rt)
}

fn ldr_u32(rt: u8, rn: u8, offset: u16) -> u32 {
    debug_assert_eq!(offset % 4, 0);
    0xb940_0000 | (u32::from(offset / 4) << 10) | (u32::from(rn) << 5) | u32::from(rt)
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

fn orn_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xaa20_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn eor_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xca00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn logical_imm64(opc: u32, rd: u8, rn: u8, imm: LogicalImmediate) -> u32 {
    debug_assert!(opc < 4);
    debug_assert!(imm.n <= 1);
    debug_assert!(imm.immr < 64);
    debug_assert!(imm.imms < 64);
    0x9200_0000
        | (opc << 29)
        | (imm.n << 22)
        | (imm.immr << 16)
        | (imm.imms << 10)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn mul_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9b00_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn mul_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1b00_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn madd_reg(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x9b00_0000
        | (u32::from(rm) << 16)
        | (u32::from(ra) << 10)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

#[cfg(test)]
fn madd_reg32(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x1b00_0000
        | (u32::from(rm) << 16)
        | (u32::from(ra) << 10)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn msub_reg(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x9b00_8000
        | (u32::from(rm) << 16)
        | (u32::from(ra) << 10)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn msub_reg32(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x1b00_8000
        | (u32::from(rm) << 16)
        | (u32::from(ra) << 10)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn smulh_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9b40_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn umulh_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9bc0_7c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sdiv_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9ac0_0c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn udiv_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9ac0_0800 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
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

fn sdiv_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1ac0_0c00 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn udiv_reg32(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1ac0_0800 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd)
}

fn ubfm64(rd: u8, rn: u8, immr: u32, imms: u32) -> u32 {
    debug_assert!(immr < 64);
    debug_assert!(imms < 64);
    0xd340_0000 | (immr << 16) | (imms << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sbfm64(rd: u8, rn: u8, immr: u32, imms: u32) -> u32 {
    debug_assert!(immr < 64);
    debug_assert!(imms < 64);
    0x9340_0000 | (immr << 16) | (imms << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn ubfm32(rd: u8, rn: u8, immr: u32, imms: u32) -> u32 {
    debug_assert!(immr < 32);
    debug_assert!(imms < 32);
    0x5300_0000 | (immr << 16) | (imms << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sbfm32(rd: u8, rn: u8, immr: u32, imms: u32) -> u32 {
    debug_assert!(immr < 32);
    debug_assert!(imms < 32);
    0x1300_0000 | (immr << 16) | (imms << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn cmp_reg(rn: u8, rm: u8) -> u32 {
    0xeb00_001f | (u32::from(rm) << 16) | (u32::from(rn) << 5)
}

fn cmp_reg32(rn: u8, rm: u8) -> u32 {
    0x6b00_001f | (u32::from(rm) << 16) | (u32::from(rn) << 5)
}

fn cmp_imm(rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0xf100_001f | (u32::from(imm) << 10) | (u32::from(rn) << 5)
}

fn cmn_imm(rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0xb100_001f | (u32::from(imm) << 10) | (u32::from(rn) << 5)
}

fn csel(rd: u8, rn: u8, rm: u8, cond: A64Cond) -> u32 {
    0x9a80_0000
        | (u32::from(rm) << 16)
        | (u32::from(cond.code()) << 12)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn csinc(rd: u8, rn: u8, rm: u8, cond: A64Cond) -> u32 {
    0x9a80_0400
        | (u32::from(rm) << 16)
        | (u32::from(cond.code()) << 12)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn csinv(rd: u8, rn: u8, rm: u8, cond: A64Cond) -> u32 {
    0xda80_0000
        | (u32::from(rm) << 16)
        | (u32::from(cond.code()) << 12)
        | (u32::from(rn) << 5)
        | u32::from(rd)
}

fn cset(rd: u8, cond: A64Cond) -> u32 {
    csinc(
        rd,
        A64_ZERO_REGISTER,
        A64_ZERO_REGISTER,
        invert_condition(cond),
    )
}

fn b_cond(branch_offset: usize, target_offset: usize, cond: A64Cond) -> u32 {
    let byte_delta = target_offset as isize - branch_offset as isize;
    debug_assert_eq!(byte_delta % 4, 0);
    let instruction_delta = (byte_delta / 4) as i32;
    debug_assert!((-0x4_0000..0x4_0000).contains(&instruction_delta));
    0x5400_0000 | (((instruction_delta as u32) & 0x7ffff) << 5) | u32::from(cond.code())
}

fn cbz(branch_offset: usize, target_offset: usize, rt: u8) -> u32 {
    let byte_delta = target_offset as isize - branch_offset as isize;
    debug_assert_eq!(byte_delta % 4, 0);
    let instruction_delta = (byte_delta / 4) as i32;
    debug_assert!((-0x4_0000..0x4_0000).contains(&instruction_delta));
    0xb400_0000 | (((instruction_delta as u32) & 0x7ffff) << 5) | u32::from(rt)
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

fn subs_imm(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0xf100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn add_imm32(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0x1100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn sub_imm32(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0x5100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
}

fn subs_imm32(rd: u8, rn: u8, imm: u16) -> u32 {
    debug_assert!(imm < 4096);
    0x7100_0000 | (u32::from(imm) << 10) | (u32::from(rn) << 5) | u32::from(rd)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_immediate_encoder_round_trips_common_masks() {
        for value in [
            0x0000_0000_0000_0001,
            0x0000_0000_0000_00ff,
            0xffff_ffff_ffff_f800,
            0x8000_0000_0000_0000,
            0x5555_5555_5555_5555,
            0xaaaa_aaaa_aaaa_aaaa,
            0x00ff_00ff_00ff_00ff,
            0xff00_ff00_ff00_ff00,
        ] {
            let encoded = encode_logical_immediate(value, 64).expect("encodable bitmask");
            assert_eq!(decode_logical_immediate_for_test(encoded, 64), value);
        }

        for value in [0x0000_00ff, 0x8000_0000, 0x00ff_00ff, 0xaaaa_aaaa] {
            let encoded = encode_logical_immediate(value, 32).expect("encodable bitmask");
            assert_eq!(decode_logical_immediate_for_test(encoded, 32), value);
        }

        assert!(encode_logical_immediate(0, 64).is_none());
        assert!(encode_logical_immediate(u64::MAX, 64).is_none());
        assert!(encode_logical_immediate(0x1234, 64).is_none());
    }

    #[test]
    fn direct_loop_instruction_helpers_match_known_a64_encodings() {
        assert_eq!(subs_imm(27, 27, 1), 0xf100_077b);
        assert_eq!(subs_imm32(27, 27, 1), 0x7100_077b);
        assert_eq!(cbz(0x100, 0x140, 23), 0xb400_0217);
        assert_eq!(ldr_u8(7, 26, 3), 0x3940_0f47);
        assert_eq!(ldr_u16(7, 26, 6), 0x7940_0f47);
        assert_eq!(ldr_u32(7, 26, 12), 0xb940_0f47);
        assert_eq!(ldr_u64(7, 26, 24), 0xf940_0f47);
        assert_eq!(sdiv_reg(7, 8, 9), 0x9ac9_0d07);
        assert_eq!(udiv_reg(7, 8, 9), 0x9ac9_0907);
        assert_eq!(sdiv_reg32(7, 8, 9), 0x1ac9_0d07);
        assert_eq!(udiv_reg32(7, 8, 9), 0x1ac9_0907);
        assert_eq!(madd_reg(7, 8, 9, 10), 0x9b09_2907);
        assert_eq!(madd_reg32(7, 8, 9, 10), 0x1b09_2907);
        assert_eq!(msub_reg(7, 8, 9, 10), 0x9b09_a907);
        assert_eq!(msub_reg32(7, 8, 9, 10), 0x1b09_a907);
        assert_eq!(cmp_reg32(8, 9), 0x6b09_011f);
        assert_eq!(csinv(11, 11, A64_ZERO_REGISTER, A64Cond::Ne), 0xda9f_116b);
    }

    fn decode_logical_immediate_for_test(imm: LogicalImmediate, width: u32) -> u64 {
        let len_input = (imm.n << 6) | ((!imm.imms) & 0x3f);
        let len = 31 - len_input.leading_zeros();
        let element_size = 1u32 << len;
        let levels = element_size - 1;
        let ones = (imm.imms & levels) + 1;
        let rotation = imm.immr & levels;
        let element = rotate_right_for_test(bitmask(ones), rotation, element_size);
        replicate_element(element, element_size, width)
    }

    fn rotate_right_for_test(value: u64, rotation: u32, width: u32) -> u64 {
        let rotation = rotation % width;
        let mask = bitmask(width);
        if rotation == 0 {
            value & mask
        } else {
            ((value >> rotation) | (value << (width - rotation))) & mask
        }
    }
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
