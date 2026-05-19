use std::fmt;

use super::{
    BlockOperation, BlockOperationKind, BlockPlan, IntegerBranchCondition, NativeInstruction,
    RuntimeBinaryOp,
};
use crate::cpu::RV64GCRegAbiName::Zero;
use crate::sign_extend;

const GUEST_INTEGER_REGISTERS: usize = 32;
const MAX_OPTIMIZER_PASSES: usize = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct OptimizationReport {
    pub removed_nops: u64,
    pub removed_dead_writes: u64,
    pub simplified_ops: u64,
    pub folded_branches: u64,
    pub propagated_values: u64,
    pub folded_constants: u64,
    pub eliminated_overwritten_writes: u64,
}

impl fmt::Display for OptimizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "removed_nops={} removed_dead_writes={} simplified_ops={} folded_branches={} propagated_values={} folded_constants={} eliminated_overwritten_writes={}",
            self.removed_nops,
            self.removed_dead_writes,
            self.simplified_ops,
            self.folded_branches,
            self.propagated_values,
            self.folded_constants,
            self.eliminated_overwritten_writes
        )
    }
}

pub(crate) fn optimize_plan(plan: &BlockPlan) -> (BlockPlan, OptimizationReport) {
    let mut report = OptimizationReport::default();
    let mut operations = plan.operations.clone();

    for _ in 0..MAX_OPTIMIZER_PASSES {
        let before = operations.clone();
        operations = peephole_pass(&operations, &mut report);
        operations = propagate_constants_and_copies(&operations, &mut report);
        operations = eliminate_overwritten_integer_writes(&operations, &mut report);
        if operations == before {
            break;
        }
    }

    let mut optimized = plan.clone();
    optimized.operations = operations;
    (optimized, report)
}

fn peephole_pass(input: &[BlockOperation], report: &mut OptimizationReport) -> Vec<BlockOperation> {
    let mut operations = Vec::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        let operation = &input[index];
        let BlockOperationKind::Native(instruction) = operation.kind();

        if let Some(next_operation) = input.get(index + 1) {
            if let Some(fused) =
                fuse_masked_zero_branch(operation.pc(), instruction, next_operation.kind())
            {
                report.simplified_ops += 1;
                operations.push(BlockOperation {
                    pc: operation.pc(),
                    opcode: operation.opcode(),
                    kind: BlockOperationKind::Native(fused),
                });
                index += 2;
                continue;
            }
        }

        let optimized = optimize_instruction(instruction, report);
        if matches!(optimized, NativeInstruction::Nop) {
            if matches!(instruction, NativeInstruction::Nop) {
                report.removed_nops += 1;
            }
            index += 1;
            continue;
        }

        operations.push(BlockOperation {
            pc: operation.pc(),
            opcode: operation.opcode(),
            kind: BlockOperationKind::Native(optimized),
        });

        if let Some(next_operation) = input.get(index + 1) {
            if let Some(forwarded) = forward_store_to_load(instruction, next_operation.kind()) {
                report.simplified_ops += 1;
                operations.push(BlockOperation {
                    pc: next_operation.pc(),
                    opcode: next_operation.opcode(),
                    kind: BlockOperationKind::Native(forwarded),
                });
                index += 2;
                continue;
            }
        }

        index += 1;
    }

    operations
}

fn fuse_masked_zero_branch(
    pc: u64,
    instruction: NativeInstruction,
    next: BlockOperationKind,
) -> Option<NativeInstruction> {
    let NativeInstruction::Andi { rd, rs1, imm } = instruction else {
        return None;
    };
    if imm == 0 || imm == -1 {
        return None;
    }

    let BlockOperationKind::Native(next_instruction) = next;
    match next_instruction {
        NativeInstruction::Beq {
            rs1: branch_lhs,
            rs2: branch_rhs,
            target,
            fallthrough,
        } if target > pc && is_zero_compare(rd, branch_lhs, branch_rhs) => {
            Some(NativeInstruction::AndBranch {
                rd,
                rs1,
                imm,
                target,
                fallthrough,
                branch_if_zero: true,
            })
        }
        NativeInstruction::Bne {
            rs1: branch_lhs,
            rs2: branch_rhs,
            target,
            fallthrough,
        } if target > pc && is_zero_compare(rd, branch_lhs, branch_rhs) => {
            Some(NativeInstruction::AndBranch {
                rd,
                rs1,
                imm,
                target,
                fallthrough,
                branch_if_zero: false,
            })
        }
        _ => None,
    }
}

fn is_zero_compare(register: u8, lhs: u8, rhs: u8) -> bool {
    (lhs == register && rhs == Zero as u8) || (rhs == register && lhs == Zero as u8)
}

fn forward_store_to_load(
    store: NativeInstruction,
    next: BlockOperationKind,
) -> Option<NativeInstruction> {
    let NativeInstruction::Store {
        rs1: store_base,
        rs2: store_value,
        imm: store_offset,
        width: store_width,
    } = store
    else {
        return None;
    };
    let BlockOperationKind::Native(next_instruction) = next;
    let NativeInstruction::Load {
        rd,
        rs1: load_base,
        imm: load_offset,
        width: load_width,
        ..
    } = next_instruction
    else {
        return None;
    };

    (store_base == load_base
        && store_offset == load_offset
        && store_width == load_width
        && store_width.bytes() == 8)
        .then_some(NativeInstruction::Move {
            rd,
            rs: store_value,
        })
}

fn optimize_instruction(
    instruction: NativeInstruction,
    report: &mut OptimizationReport,
) -> NativeInstruction {
    if dead_integer_write(instruction) {
        report.removed_dead_writes += 1;
        return NativeInstruction::Nop;
    }

    match instruction {
        NativeInstruction::Add { rd, rs1, rs2 } if rs1 == Zero as u8 => copy_from(rd, rs2, report),
        NativeInstruction::Add { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Add { rd, rs1, rs2 } if rs1 == rs2 && rs1 == Zero as u8 => {
            set_zero(rd, report)
        }
        NativeInstruction::Addi { rd, rs1, imm: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Addi { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, imm as u64, report)
        }
        NativeInstruction::Addiw { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, sign_extend_word(imm as u64), report)
        }
        NativeInstruction::Slti { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, u64::from(0 < imm), report)
        }
        NativeInstruction::Sltiu { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, u64::from(0 < imm as u64), report)
        }
        NativeInstruction::Sub { rd, rs1, rs2 } if rs1 == rs2 => set_zero(rd, report),
        NativeInstruction::Slt { rd, rs1, rs2 } if rs1 == rs2 => set_zero(rd, report),
        NativeInstruction::Sltu { rd, rs1, rs2 } if rs1 == rs2 => set_zero(rd, report),
        NativeInstruction::And { rd, rs1, .. } if rs1 == Zero as u8 => set_zero(rd, report),
        NativeInstruction::And { rd, rs2, .. } if rs2 == Zero as u8 => set_zero(rd, report),
        NativeInstruction::And { rd, rs1, rs2 } if rs1 == rs2 => copy_from(rd, rs1, report),
        NativeInstruction::Andi { rd, imm: 0, .. } => set_zero(rd, report),
        NativeInstruction::Andi { rd, rs1, imm: -1 } => copy_from(rd, rs1, report),
        NativeInstruction::Beq {
            rs1, rs2, target, ..
        } if rs1 == rs2 => {
            report.folded_branches += 1;
            NativeInstruction::Jump { target }
        }
        NativeInstruction::Bge {
            rs1, rs2, target, ..
        } if rs1 == rs2 => {
            report.folded_branches += 1;
            NativeInstruction::Jump { target }
        }
        NativeInstruction::Bgeu {
            rs1, rs2, target, ..
        } if rs1 == rs2 => {
            report.folded_branches += 1;
            NativeInstruction::Jump { target }
        }
        NativeInstruction::Blt {
            rs1,
            rs2,
            fallthrough,
            ..
        } if rs1 == rs2 => {
            report.folded_branches += 1;
            NativeInstruction::Jump {
                target: fallthrough,
            }
        }
        NativeInstruction::Bltu {
            rs1,
            rs2,
            fallthrough,
            ..
        } if rs1 == rs2 => {
            report.folded_branches += 1;
            NativeInstruction::Jump {
                target: fallthrough,
            }
        }
        NativeInstruction::Bne {
            rs1,
            rs2,
            fallthrough,
            ..
        } if rs1 == rs2 => {
            report.folded_branches += 1;
            NativeInstruction::Jump {
                target: fallthrough,
            }
        }
        NativeInstruction::Jal { rd, target, .. } if rd == Zero as u8 => {
            report.simplified_ops += 1;
            NativeInstruction::Jump { target }
        }
        NativeInstruction::Move { rd, rs } if rs == Zero as u8 => set_zero(rd, report),
        NativeInstruction::Or { rd, rs1, rs2 } if rs1 == Zero as u8 => copy_from(rd, rs2, report),
        NativeInstruction::Or { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Or { rd, rs1, rs2 } if rs1 == rs2 => copy_from(rd, rs1, report),
        NativeInstruction::Ori { rd, rs1, imm: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Mul { rd, rs1, .. } if rs1 == Zero as u8 => set_zero(rd, report),
        NativeInstruction::Mul { rd, rs2, .. } if rs2 == Zero as u8 => set_zero(rd, report),
        NativeInstruction::Mulw { rd, rs1, .. } if rs1 == Zero as u8 => set_zero(rd, report),
        NativeInstruction::Mulw { rd, rs2, .. } if rs2 == Zero as u8 => set_zero(rd, report),
        NativeInstruction::Sll { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Slli { rd, rs1, shamt: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Sra { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Srai { rd, rs1, shamt: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Srl { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Srli { rd, rs1, shamt: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Sub { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Xor { rd, rs1, rs2 } if rs1 == Zero as u8 => copy_from(rd, rs2, report),
        NativeInstruction::Xor { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Xor { rd, rs1, rs2 } if rs1 == rs2 => set_zero(rd, report),
        NativeInstruction::Xori { rd, rs1, imm: 0 } => copy_from(rd, rs1, report),
        other => other,
    }
}

fn dead_integer_write(instruction: NativeInstruction) -> bool {
    matches!(
        instruction,
        NativeInstruction::Add { rd: 0, .. }
            | NativeInstruction::Addi { rd: 0, .. }
            | NativeInstruction::Addiw { rd: 0, .. }
            | NativeInstruction::Addw { rd: 0, .. }
            | NativeInstruction::And { rd: 0, .. }
            | NativeInstruction::Andi { rd: 0, .. }
            | NativeInstruction::Auipc { rd: 0, .. }
            | NativeInstruction::Lui { rd: 0, .. }
            | NativeInstruction::LoadImmediate { rd: 0, .. }
            | NativeInstruction::Move { rd: 0, .. }
            | NativeInstruction::Mul { rd: 0, .. }
            | NativeInstruction::Mulh { rd: 0, .. }
            | NativeInstruction::Mulhu { rd: 0, .. }
            | NativeInstruction::Mulw { rd: 0, .. }
            | NativeInstruction::Or { rd: 0, .. }
            | NativeInstruction::Ori { rd: 0, .. }
            | NativeInstruction::RuntimeBinary { rd: 0, .. }
            | NativeInstruction::Sll { rd: 0, .. }
            | NativeInstruction::Slli { rd: 0, .. }
            | NativeInstruction::Slliw { rd: 0, .. }
            | NativeInstruction::Sllw { rd: 0, .. }
            | NativeInstruction::Slt { rd: 0, .. }
            | NativeInstruction::Slti { rd: 0, .. }
            | NativeInstruction::Sltiu { rd: 0, .. }
            | NativeInstruction::Sltu { rd: 0, .. }
            | NativeInstruction::Sra { rd: 0, .. }
            | NativeInstruction::Srai { rd: 0, .. }
            | NativeInstruction::Sraiw { rd: 0, .. }
            | NativeInstruction::Sraw { rd: 0, .. }
            | NativeInstruction::Srl { rd: 0, .. }
            | NativeInstruction::Srli { rd: 0, .. }
            | NativeInstruction::Srliw { rd: 0, .. }
            | NativeInstruction::Srlw { rd: 0, .. }
            | NativeInstruction::Sub { rd: 0, .. }
            | NativeInstruction::Subw { rd: 0, .. }
            | NativeInstruction::Xor { rd: 0, .. }
            | NativeInstruction::Xori { rd: 0, .. }
    )
}

fn copy_from(rd: u8, rs: u8, report: &mut OptimizationReport) -> NativeInstruction {
    if rd == rs {
        report.removed_dead_writes += 1;
        NativeInstruction::Nop
    } else {
        report.simplified_ops += 1;
        NativeInstruction::Move { rd, rs }
    }
}

fn set_zero(rd: u8, report: &mut OptimizationReport) -> NativeInstruction {
    load_immediate(rd, 0, report)
}

fn load_immediate(rd: u8, value: u64, report: &mut OptimizationReport) -> NativeInstruction {
    if rd == Zero as u8 {
        report.removed_dead_writes += 1;
        NativeInstruction::Nop
    } else {
        report.simplified_ops += 1;
        NativeInstruction::LoadImmediate { rd, value }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegisterValue {
    Unknown,
    Constant(u64),
    Copy(u8),
}

#[derive(Debug, Clone)]
struct RegisterFacts {
    values: [RegisterValue; GUEST_INTEGER_REGISTERS],
}

impl RegisterFacts {
    fn new() -> Self {
        let mut values = [RegisterValue::Unknown; GUEST_INTEGER_REGISTERS];
        values[Zero as usize] = RegisterValue::Constant(0);
        Self { values }
    }

    fn resolve(&self, register: u8) -> RegisterValue {
        if register == Zero as u8 {
            return RegisterValue::Constant(0);
        }

        let mut current = register;
        for _ in 0..GUEST_INTEGER_REGISTERS {
            match self.values[current as usize] {
                RegisterValue::Constant(value) => return RegisterValue::Constant(value),
                RegisterValue::Copy(next) if next == Zero as u8 => {
                    return RegisterValue::Constant(0);
                }
                RegisterValue::Copy(next) if next != current => current = next,
                RegisterValue::Copy(_) | RegisterValue::Unknown => {
                    return if current == register {
                        RegisterValue::Unknown
                    } else {
                        RegisterValue::Copy(current)
                    };
                }
            }
        }

        RegisterValue::Unknown
    }

    fn constant(&self, register: u8) -> Option<u64> {
        match self.resolve(register) {
            RegisterValue::Constant(value) => Some(value),
            RegisterValue::Unknown | RegisterValue::Copy(_) => None,
        }
    }

    fn canonical_register(&self, register: u8) -> u8 {
        match self.resolve(register) {
            RegisterValue::Constant(0) => Zero as u8,
            RegisterValue::Copy(source) => source,
            RegisterValue::Unknown | RegisterValue::Constant(_) => register,
        }
    }

    fn set(&mut self, register: u8, value: RegisterValue) {
        if register == Zero as u8 {
            return;
        }

        self.forget_register_and_copies(register);
        self.values[register as usize] = match value {
            RegisterValue::Copy(source) if source == Zero as u8 => RegisterValue::Constant(0),
            RegisterValue::Copy(source) if source == register => RegisterValue::Unknown,
            other => other,
        };
    }

    fn forget(&mut self, register: u8) {
        if register != Zero as u8 {
            self.forget_register_and_copies(register);
        }
    }

    fn forget_all(&mut self) {
        self.values = [RegisterValue::Unknown; GUEST_INTEGER_REGISTERS];
        self.values[Zero as usize] = RegisterValue::Constant(0);
    }

    fn forget_register_and_copies(&mut self, register: u8) {
        for candidate in 1..GUEST_INTEGER_REGISTERS {
            if candidate as u8 == register || self.copy_chain_contains(candidate as u8, register) {
                self.values[candidate] = RegisterValue::Unknown;
            }
        }
        self.values[Zero as usize] = RegisterValue::Constant(0);
    }

    fn copy_chain_contains(&self, candidate: u8, needle: u8) -> bool {
        let mut current = candidate;
        for _ in 0..GUEST_INTEGER_REGISTERS {
            match self.values[current as usize] {
                RegisterValue::Copy(next) => {
                    if next == needle {
                        return true;
                    }
                    if next == current {
                        return false;
                    }
                    current = next;
                }
                RegisterValue::Unknown | RegisterValue::Constant(_) => return false,
            }
        }

        false
    }
}

fn propagate_constants_and_copies(
    input: &[BlockOperation],
    report: &mut OptimizationReport,
) -> Vec<BlockOperation> {
    let mut facts = RegisterFacts::new();
    let mut output = Vec::with_capacity(input.len());

    for operation in input {
        let BlockOperationKind::Native(instruction) = operation.kind();
        let rewritten = rewrite_instruction_sources(instruction, &facts, report);
        let folded = constant_fold_instruction(rewritten, &facts, report);

        for native in folded {
            let optimized = optimize_instruction(native, report);
            if matches!(optimized, NativeInstruction::Nop) {
                if matches!(native, NativeInstruction::Nop) {
                    report.removed_nops += 1;
                }
                update_register_facts(native, &mut facts);
                continue;
            }

            update_register_facts(optimized, &mut facts);
            output.push(BlockOperation {
                pc: operation.pc(),
                opcode: operation.opcode(),
                kind: BlockOperationKind::Native(optimized),
            });
        }
    }

    output
}

fn rewrite_register(register: u8, facts: &RegisterFacts, changed: &mut u64) -> u8 {
    let rewritten = facts.canonical_register(register);
    if rewritten != register {
        *changed += 1;
    }
    rewritten
}

fn rewrite_instruction_sources(
    instruction: NativeInstruction,
    facts: &RegisterFacts,
    report: &mut OptimizationReport,
) -> NativeInstruction {
    let mut propagated = 0;
    let rewritten = match instruction {
        NativeInstruction::Add { rd, rs1, rs2 } => NativeInstruction::Add {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Addi { rd, rs1, imm } => NativeInstruction::Addi {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
        },
        NativeInstruction::Addiw { rd, rs1, imm } => NativeInstruction::Addiw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
        },
        NativeInstruction::Addw { rd, rs1, rs2 } => NativeInstruction::Addw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::And { rd, rs1, rs2 } => NativeInstruction::And {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::AndBranch {
            rd,
            rs1,
            imm,
            target,
            fallthrough,
            branch_if_zero,
        } => NativeInstruction::AndBranch {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
            target,
            fallthrough,
            branch_if_zero,
        },
        NativeInstruction::Andi { rd, rs1, imm } => NativeInstruction::Andi {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
        },
        NativeInstruction::Beq {
            rs1,
            rs2,
            target,
            fallthrough,
        } => NativeInstruction::Beq {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            target,
            fallthrough,
        },
        NativeInstruction::Bge {
            rs1,
            rs2,
            target,
            fallthrough,
        } => NativeInstruction::Bge {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            target,
            fallthrough,
        },
        NativeInstruction::Bgeu {
            rs1,
            rs2,
            target,
            fallthrough,
        } => NativeInstruction::Bgeu {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            target,
            fallthrough,
        },
        NativeInstruction::Blt {
            rs1,
            rs2,
            target,
            fallthrough,
        } => NativeInstruction::Blt {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            target,
            fallthrough,
        },
        NativeInstruction::Bltu {
            rs1,
            rs2,
            target,
            fallthrough,
        } => NativeInstruction::Bltu {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            target,
            fallthrough,
        },
        NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } => NativeInstruction::Bne {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            target,
            fallthrough,
        },
        NativeInstruction::FloatLoad {
            rd,
            rs1,
            imm,
            width,
        } => NativeInstruction::FloatLoad {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
            width,
        },
        NativeInstruction::FloatStore {
            rs1,
            rs2,
            imm,
            width,
        } => NativeInstruction::FloatStore {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2,
            imm,
            width,
        },
        NativeInstruction::Jalr {
            rd,
            rs1,
            imm,
            return_pc,
        } => NativeInstruction::Jalr {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
            return_pc,
        },
        NativeInstruction::JumpReg { rs1 } => NativeInstruction::JumpReg {
            rs1: rewrite_register(rs1, facts, &mut propagated),
        },
        NativeInstruction::JumpRegLink { rs1, return_pc } => NativeInstruction::JumpRegLink {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            return_pc,
        },
        NativeInstruction::Load {
            rd,
            rs1,
            imm,
            width,
            signed,
        } => NativeInstruction::Load {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
            width,
            signed,
        },
        NativeInstruction::Move { rd, rs } => NativeInstruction::Move {
            rd,
            rs: rewrite_register(rs, facts, &mut propagated),
        },
        NativeInstruction::Mul { rd, rs1, rs2 } => NativeInstruction::Mul {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Mulh { rd, rs1, rs2 } => NativeInstruction::Mulh {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Mulhu { rd, rs1, rs2 } => NativeInstruction::Mulhu {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Mulw { rd, rs1, rs2 } => NativeInstruction::Mulw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Or { rd, rs1, rs2 } => NativeInstruction::Or {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Ori { rd, rs1, imm } => NativeInstruction::Ori {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
        },
        NativeInstruction::RuntimeAtomic { rd, rs1, rs2, op } => NativeInstruction::RuntimeAtomic {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            op,
        },
        NativeInstruction::RuntimeBinary { rd, rs1, rs2, op } => NativeInstruction::RuntimeBinary {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            op,
        },
        NativeInstruction::Sll { rd, rs1, rs2 } => NativeInstruction::Sll {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Slli { rd, rs1, shamt } => NativeInstruction::Slli {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            shamt,
        },
        NativeInstruction::Slliw { rd, rs1, shamt } => NativeInstruction::Slliw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            shamt,
        },
        NativeInstruction::Sllw { rd, rs1, rs2 } => NativeInstruction::Sllw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Slt { rd, rs1, rs2 } => NativeInstruction::Slt {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Slti { rd, rs1, imm } => NativeInstruction::Slti {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
        },
        NativeInstruction::Sltiu { rd, rs1, imm } => NativeInstruction::Sltiu {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
        },
        NativeInstruction::Sltu { rd, rs1, rs2 } => NativeInstruction::Sltu {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Sra { rd, rs1, rs2 } => NativeInstruction::Sra {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Srai { rd, rs1, shamt } => NativeInstruction::Srai {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            shamt,
        },
        NativeInstruction::Sraiw { rd, rs1, shamt } => NativeInstruction::Sraiw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            shamt,
        },
        NativeInstruction::Sraw { rd, rs1, rs2 } => NativeInstruction::Sraw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Srl { rd, rs1, rs2 } => NativeInstruction::Srl {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Srli { rd, rs1, shamt } => NativeInstruction::Srli {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            shamt,
        },
        NativeInstruction::Srliw { rd, rs1, shamt } => NativeInstruction::Srliw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            shamt,
        },
        NativeInstruction::Srlw { rd, rs1, rs2 } => NativeInstruction::Srlw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Store {
            rs1,
            rs2,
            imm,
            width,
        } => NativeInstruction::Store {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            imm,
            width,
        },
        NativeInstruction::Sub { rd, rs1, rs2 } => NativeInstruction::Sub {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Subw { rd, rs1, rs2 } => NativeInstruction::Subw {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::TraceGuard {
            rs1,
            rs2,
            condition,
            continue_on_taken,
            continue_pc,
            side_exit_pc,
            executed_instructions,
        } => NativeInstruction::TraceGuard {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            condition,
            continue_on_taken,
            continue_pc,
            side_exit_pc,
            executed_instructions,
        },
        NativeInstruction::TraceLoopGuard {
            rs1,
            rs2,
            condition,
            continue_on_taken,
            loop_pc,
            side_exit_pc,
            guest_instruction_count,
        } => NativeInstruction::TraceLoopGuard {
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
            condition,
            continue_on_taken,
            loop_pc,
            side_exit_pc,
            guest_instruction_count,
        },
        NativeInstruction::Xor { rd, rs1, rs2 } => NativeInstruction::Xor {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            rs2: rewrite_register(rs2, facts, &mut propagated),
        },
        NativeInstruction::Xori { rd, rs1, imm } => NativeInstruction::Xori {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            imm,
        },
        other => other,
    };

    report.propagated_values += propagated;
    rewritten
}

fn constant_fold_instruction(
    instruction: NativeInstruction,
    facts: &RegisterFacts,
    report: &mut OptimizationReport,
) -> Vec<NativeInstruction> {
    if let NativeInstruction::AndBranch {
        rd,
        rs1,
        imm,
        target,
        fallthrough,
        branch_if_zero,
    } = instruction
    {
        if let Some(value) = facts.constant(rs1) {
            let masked = value & imm as u64;
            let branch_target = if (masked == 0) == branch_if_zero {
                target
            } else {
                fallthrough
            };
            report.folded_branches += 1;
            return vec![
                folded_constant(rd, masked, report),
                NativeInstruction::Jump {
                    target: branch_target,
                },
            ];
        }
    }

    let folded = match instruction {
        NativeInstruction::Add { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, u64::wrapping_add, report)
        }
        NativeInstruction::Addi { rd, rs1, imm } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, value.wrapping_add(imm as u64), report)),
        NativeInstruction::Addiw { rd, rs1, imm } => facts.constant(rs1).map(|value| {
            folded_constant(
                rd,
                sign_extend_word(u64::from((value as u32).wrapping_add(imm as u32))),
                report,
            )
        }),
        NativeInstruction::Addw { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, addw_value, report)
        }
        NativeInstruction::And { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, |lhs, rhs| lhs & rhs, report)
        }
        NativeInstruction::AndBranch { .. } => None,
        NativeInstruction::Andi { rd, rs1, imm } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, value & imm as u64, report)),
        NativeInstruction::Beq {
            rs1,
            rs2,
            target,
            fallthrough,
        } => fold_branch(
            rs1,
            rs2,
            target,
            fallthrough,
            IntegerBranchCondition::Eq,
            facts,
            report,
        ),
        NativeInstruction::Bge {
            rs1,
            rs2,
            target,
            fallthrough,
        } => fold_branch(
            rs1,
            rs2,
            target,
            fallthrough,
            IntegerBranchCondition::Ge,
            facts,
            report,
        ),
        NativeInstruction::Bgeu {
            rs1,
            rs2,
            target,
            fallthrough,
        } => fold_branch(
            rs1,
            rs2,
            target,
            fallthrough,
            IntegerBranchCondition::Geu,
            facts,
            report,
        ),
        NativeInstruction::Blt {
            rs1,
            rs2,
            target,
            fallthrough,
        } => fold_branch(
            rs1,
            rs2,
            target,
            fallthrough,
            IntegerBranchCondition::Lt,
            facts,
            report,
        ),
        NativeInstruction::Bltu {
            rs1,
            rs2,
            target,
            fallthrough,
        } => fold_branch(
            rs1,
            rs2,
            target,
            fallthrough,
            IntegerBranchCondition::Ltu,
            facts,
            report,
        ),
        NativeInstruction::Bne {
            rs1,
            rs2,
            target,
            fallthrough,
        } => fold_branch(
            rs1,
            rs2,
            target,
            fallthrough,
            IntegerBranchCondition::Ne,
            facts,
            report,
        ),
        NativeInstruction::Jalr {
            rd,
            rs1,
            imm,
            return_pc,
        } => facts.constant(rs1).map(|value| {
            report.folded_constants += 1;
            NativeInstruction::Jal {
                rd,
                target: value.wrapping_add(imm as u64) & !1,
                return_pc,
            }
        }),
        NativeInstruction::JumpReg { rs1 } => facts.constant(rs1).map(|target| {
            report.folded_constants += 1;
            NativeInstruction::Jump { target }
        }),
        NativeInstruction::JumpRegLink { rs1, return_pc } => facts.constant(rs1).map(|target| {
            report.folded_constants += 1;
            NativeInstruction::Jal {
                rd: 1,
                target,
                return_pc,
            }
        }),
        NativeInstruction::Move { rd, rs } => facts
            .constant(rs)
            .map(|value| folded_constant(rd, value, report)),
        NativeInstruction::Mul { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, u64::wrapping_mul, report)
        }
        NativeInstruction::Mulh { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, mulh_value, report)
        }
        NativeInstruction::Mulhu { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, mulhu_value, report)
        }
        NativeInstruction::Mulw { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, mulw_value, report)
        }
        NativeInstruction::Or { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, |lhs, rhs| lhs | rhs, report)
        }
        NativeInstruction::Ori { rd, rs1, imm } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, value | imm as u64, report)),
        NativeInstruction::RuntimeBinary { rd, rs1, rs2, op } => {
            let lhs = facts.constant(rs1);
            let rhs = facts.constant(rs2);
            lhs.zip(rhs)
                .map(|(lhs, rhs)| folded_constant(rd, runtime_binary_value(op, lhs, rhs), report))
        }
        NativeInstruction::Sll { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, sll_value, report)
        }
        NativeInstruction::Slli { rd, rs1, shamt } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, value.wrapping_shl(shamt & 0x3f), report)),
        NativeInstruction::Slliw { rd, rs1, shamt } => facts.constant(rs1).map(|value| {
            folded_constant(
                rd,
                sign_extend_word(u64::from((value as u32).wrapping_shl(shamt & 0x1f))),
                report,
            )
        }),
        NativeInstruction::Sllw { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, sllw_value, report)
        }
        NativeInstruction::Slt { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, slt_value, report)
        }
        NativeInstruction::Slti { rd, rs1, imm } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, u64::from((value as i64) < imm), report)),
        NativeInstruction::Sltiu { rd, rs1, imm } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, u64::from(value < imm as u64), report)),
        NativeInstruction::Sltu { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, |lhs, rhs| u64::from(lhs < rhs), report)
        }
        NativeInstruction::Sra { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, sra_value, report)
        }
        NativeInstruction::Srai { rd, rs1, shamt } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, ((value as i64) >> (shamt & 0x3f)) as u64, report)),
        NativeInstruction::Sraiw { rd, rs1, shamt } => facts.constant(rs1).map(|value| {
            folded_constant(
                rd,
                sign_extend_word(((value as i32) >> (shamt & 0x1f)) as u32 as u64),
                report,
            )
        }),
        NativeInstruction::Sraw { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, sraw_value, report)
        }
        NativeInstruction::Srl { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, srl_value, report)
        }
        NativeInstruction::Srli { rd, rs1, shamt } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, value.wrapping_shr(shamt & 0x3f), report)),
        NativeInstruction::Srliw { rd, rs1, shamt } => facts.constant(rs1).map(|value| {
            folded_constant(
                rd,
                sign_extend_word(u64::from((value as u32).wrapping_shr(shamt & 0x1f))),
                report,
            )
        }),
        NativeInstruction::Srlw { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, srlw_value, report)
        }
        NativeInstruction::Sub { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, u64::wrapping_sub, report)
        }
        NativeInstruction::Subw { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, subw_value, report)
        }
        NativeInstruction::TraceGuard {
            rs1,
            rs2,
            condition,
            continue_on_taken,
            ..
        } => {
            let lhs = facts.constant(rs1);
            let rhs = facts.constant(rs2);
            lhs.zip(rhs).and_then(|(lhs, rhs)| {
                if branch_condition_taken(condition, lhs, rhs) == continue_on_taken {
                    report.folded_branches += 1;
                    Some(NativeInstruction::Nop)
                } else {
                    None
                }
            })
        }
        NativeInstruction::Xor { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, |lhs, rhs| lhs ^ rhs, report)
        }
        NativeInstruction::Xori { rd, rs1, imm } => facts
            .constant(rs1)
            .map(|value| folded_constant(rd, value ^ imm as u64, report)),
        _ => None,
    };

    match folded {
        Some(NativeInstruction::Nop) => vec![NativeInstruction::Nop],
        Some(native) => vec![native],
        None => vec![instruction],
    }
}

fn fold_binary_value(
    rd: u8,
    rs1: u8,
    rs2: u8,
    facts: &RegisterFacts,
    op: fn(u64, u64) -> u64,
    report: &mut OptimizationReport,
) -> Option<NativeInstruction> {
    let lhs = facts.constant(rs1)?;
    let rhs = facts.constant(rs2)?;
    Some(folded_constant(rd, op(lhs, rhs), report))
}

fn fold_branch(
    rs1: u8,
    rs2: u8,
    target: u64,
    fallthrough: u64,
    condition: IntegerBranchCondition,
    facts: &RegisterFacts,
    report: &mut OptimizationReport,
) -> Option<NativeInstruction> {
    let lhs = facts.constant(rs1)?;
    let rhs = facts.constant(rs2)?;
    report.folded_branches += 1;
    Some(NativeInstruction::Jump {
        target: if branch_condition_taken(condition, lhs, rhs) {
            target
        } else {
            fallthrough
        },
    })
}

fn folded_constant(rd: u8, value: u64, report: &mut OptimizationReport) -> NativeInstruction {
    if rd == Zero as u8 {
        report.removed_dead_writes += 1;
        NativeInstruction::Nop
    } else {
        report.folded_constants += 1;
        NativeInstruction::LoadImmediate { rd, value }
    }
}

fn update_register_facts(instruction: NativeInstruction, facts: &mut RegisterFacts) {
    match instruction {
        NativeInstruction::Add { rd, .. }
        | NativeInstruction::Addi { rd, .. }
        | NativeInstruction::Addiw { rd, .. }
        | NativeInstruction::Addw { rd, .. }
        | NativeInstruction::And { rd, .. }
        | NativeInstruction::Andi { rd, .. }
        | NativeInstruction::Load { rd, .. }
        | NativeInstruction::Mul { rd, .. }
        | NativeInstruction::Mulh { rd, .. }
        | NativeInstruction::Mulhu { rd, .. }
        | NativeInstruction::Mulw { rd, .. }
        | NativeInstruction::Or { rd, .. }
        | NativeInstruction::Ori { rd, .. }
        | NativeInstruction::RuntimeAtomic { rd, .. }
        | NativeInstruction::RuntimeBinary { rd, .. }
        | NativeInstruction::RuntimeCsr { rd, .. }
        | NativeInstruction::RuntimeFloat { rd, .. }
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
        | NativeInstruction::Xori { rd, .. } => facts.forget(rd),
        NativeInstruction::AndBranch { rd, rs1, imm, .. } => {
            let value = facts
                .constant(rs1)
                .map(|value| RegisterValue::Constant(value & imm as u64))
                .unwrap_or(RegisterValue::Unknown);
            facts.set(rd, value);
        }
        NativeInstruction::Auipc { rd, value }
        | NativeInstruction::LoadImmediate { rd, value }
        | NativeInstruction::Lui { rd, value } => {
            facts.set(rd, RegisterValue::Constant(value));
        }
        NativeInstruction::Jal { rd, return_pc, .. }
        | NativeInstruction::Jalr { rd, return_pc, .. } => {
            facts.set(rd, RegisterValue::Constant(return_pc));
            facts.forget_all();
        }
        NativeInstruction::JumpRegLink { return_pc, .. } => {
            facts.set(1, RegisterValue::Constant(return_pc));
            facts.forget_all();
        }
        NativeInstruction::Move { rd, rs } => {
            let value = match facts.resolve(rs) {
                RegisterValue::Unknown if rs != Zero as u8 => RegisterValue::Copy(rs),
                other => other,
            };
            facts.set(rd, value);
        }
        NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::Beq { .. }
        | NativeInstruction::Bge { .. }
        | NativeInstruction::Bgeu { .. }
        | NativeInstruction::Blt { .. }
        | NativeInstruction::Bltu { .. }
        | NativeInstruction::Bne { .. }
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::DivisionRecurrenceLoop(_)
        | NativeInstruction::Ecall { .. }
        | NativeInstruction::FibonacciRecurrenceLoop(_)
        | NativeInstruction::FloatLoad { .. }
        | NativeInstruction::FloatStore { .. }
        | NativeInstruction::Jump { .. }
        | NativeInstruction::JumpReg { .. }
        | NativeInstruction::RuntimeTrap { .. }
        | NativeInstruction::StoreLoadForwardLoop(_)
        | NativeInstruction::TraceLoopGuard { .. } => facts.forget_all(),
        NativeInstruction::InlinedJump { .. }
        | NativeInstruction::Nop
        | NativeInstruction::Store { .. }
        | NativeInstruction::TraceGuard { .. } => {}
    }
}

fn eliminate_overwritten_integer_writes(
    input: &[BlockOperation],
    report: &mut OptimizationReport,
) -> Vec<BlockOperation> {
    let mut live = [true; GUEST_INTEGER_REGISTERS];
    let mut keep = vec![true; input.len()];

    for (index, operation) in input.iter().enumerate().rev() {
        let BlockOperationKind::Native(instruction) = operation.kind();

        if liveness_barrier(instruction) {
            live = [true; GUEST_INTEGER_REGISTERS];
            mark_read_registers(instruction, &mut live);
            continue;
        }

        if let Some(rd) = pure_integer_write(instruction) {
            if rd != Zero as u8 && !live[rd as usize] {
                keep[index] = false;
                report.removed_dead_writes += 1;
                report.eliminated_overwritten_writes += 1;
                continue;
            }
            if rd != Zero as u8 {
                live[rd as usize] = false;
            }
        }

        mark_read_registers(instruction, &mut live);
    }

    input
        .iter()
        .zip(keep)
        .filter_map(|(operation, keep)| keep.then_some(*operation))
        .collect()
}

fn pure_integer_write(instruction: NativeInstruction) -> Option<u8> {
    match instruction {
        NativeInstruction::Add { rd, .. }
        | NativeInstruction::Addi { rd, .. }
        | NativeInstruction::Addiw { rd, .. }
        | NativeInstruction::Addw { rd, .. }
        | NativeInstruction::And { rd, .. }
        | NativeInstruction::Andi { rd, .. }
        | NativeInstruction::Auipc { rd, .. }
        | NativeInstruction::LoadImmediate { rd, .. }
        | NativeInstruction::Lui { rd, .. }
        | NativeInstruction::Move { rd, .. }
        | NativeInstruction::Mul { rd, .. }
        | NativeInstruction::Mulh { rd, .. }
        | NativeInstruction::Mulhu { rd, .. }
        | NativeInstruction::Mulw { rd, .. }
        | NativeInstruction::Or { rd, .. }
        | NativeInstruction::Ori { rd, .. }
        | NativeInstruction::RuntimeBinary { rd, .. }
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
        | NativeInstruction::Xori { rd, .. } => Some(rd),
        _ => None,
    }
}

fn liveness_barrier(instruction: NativeInstruction) -> bool {
    matches!(
        instruction,
        NativeInstruction::AndBranch { .. }
            | NativeInstruction::ArithmeticXorToggleLoop(_)
            | NativeInstruction::Beq { .. }
            | NativeInstruction::Bge { .. }
            | NativeInstruction::Bgeu { .. }
            | NativeInstruction::Blt { .. }
            | NativeInstruction::Bltu { .. }
            | NativeInstruction::Bne { .. }
            | NativeInstruction::CountedDiamondLoop(_)
            | NativeInstruction::DivisionRecurrenceLoop(_)
            | NativeInstruction::Ecall { .. }
            | NativeInstruction::FibonacciRecurrenceLoop(_)
            | NativeInstruction::FloatLoad { .. }
            | NativeInstruction::FloatStore { .. }
            | NativeInstruction::Jal { .. }
            | NativeInstruction::Jalr { .. }
            | NativeInstruction::Jump { .. }
            | NativeInstruction::JumpReg { .. }
            | NativeInstruction::JumpRegLink { .. }
            | NativeInstruction::Load { .. }
            | NativeInstruction::RuntimeAtomic { .. }
            | NativeInstruction::RuntimeCsr { .. }
            | NativeInstruction::RuntimeFloat { .. }
            | NativeInstruction::RuntimeTrap { .. }
            | NativeInstruction::Store { .. }
            | NativeInstruction::StoreLoadForwardLoop(_)
            | NativeInstruction::TraceGuard { .. }
            | NativeInstruction::TraceLoopGuard { .. }
    )
}

fn mark_read_registers(instruction: NativeInstruction, live: &mut [bool; GUEST_INTEGER_REGISTERS]) {
    let mut mark = |register: u8| {
        if register != Zero as u8 && usize::from(register) < GUEST_INTEGER_REGISTERS {
            live[register as usize] = true;
        }
    };

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
            mark(rs1);
            mark(rs2);
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
        | NativeInstruction::Xori { rs1, .. } => mark(rs1),
        NativeInstruction::Store { rs1, rs2, .. } => {
            mark(rs1);
            mark(rs2);
        }
        NativeInstruction::Auipc { .. }
        | NativeInstruction::InlinedJump { .. }
        | NativeInstruction::Jal { .. }
        | NativeInstruction::Jump { .. }
        | NativeInstruction::LoadImmediate { .. }
        | NativeInstruction::Lui { .. }
        | NativeInstruction::Nop
        | NativeInstruction::ArithmeticXorToggleLoop(_)
        | NativeInstruction::CountedDiamondLoop(_)
        | NativeInstruction::DivisionRecurrenceLoop(_)
        | NativeInstruction::Ecall { .. }
        | NativeInstruction::FibonacciRecurrenceLoop(_)
        | NativeInstruction::FloatStore { .. }
        | NativeInstruction::RuntimeFloat { .. }
        | NativeInstruction::RuntimeTrap { .. }
        | NativeInstruction::StoreLoadForwardLoop(_) => {}
    }
}

fn branch_condition_taken(condition: IntegerBranchCondition, lhs: u64, rhs: u64) -> bool {
    match condition {
        IntegerBranchCondition::Eq => lhs == rhs,
        IntegerBranchCondition::Ne => lhs != rhs,
        IntegerBranchCondition::Ge => (lhs as i64) >= (rhs as i64),
        IntegerBranchCondition::Geu => lhs >= rhs,
        IntegerBranchCondition::Lt => (lhs as i64) < (rhs as i64),
        IntegerBranchCondition::Ltu => lhs < rhs,
    }
}

fn sign_extend_word(value: u64) -> u64 {
    sign_extend(value & u64::from(u32::MAX), 32) as u64
}

fn addw_value(lhs: u64, rhs: u64) -> u64 {
    sign_extend_word(u64::from((lhs as u32).wrapping_add(rhs as u32)))
}

fn subw_value(lhs: u64, rhs: u64) -> u64 {
    sign_extend_word(u64::from((lhs as u32).wrapping_sub(rhs as u32)))
}

fn mulh_value(lhs: u64, rhs: u64) -> u64 {
    let multiplicand = lhs as i64 as i128;
    let multiplier = rhs as i64 as i128;
    ((multiplicand * multiplier) >> 64) as u64
}

fn mulhu_value(lhs: u64, rhs: u64) -> u64 {
    let multiplicand = lhs as u128;
    let multiplier = rhs as u128;
    ((multiplicand * multiplier) >> 64) as u64
}

fn mulw_value(lhs: u64, rhs: u64) -> u64 {
    sign_extend_word(((lhs as i64).wrapping_mul(rhs as i64) as u64) & u64::from(u32::MAX))
}

fn sll_value(lhs: u64, rhs: u64) -> u64 {
    lhs.wrapping_shl((rhs & 0x3f) as u32)
}

fn sllw_value(lhs: u64, rhs: u64) -> u64 {
    sign_extend_word(u64::from((lhs as u32).wrapping_shl((rhs & 0x1f) as u32)))
}

fn slt_value(lhs: u64, rhs: u64) -> u64 {
    u64::from((lhs as i64) < (rhs as i64))
}

fn sra_value(lhs: u64, rhs: u64) -> u64 {
    ((lhs as i64) >> ((rhs & 0x3f) as u32)) as u64
}

fn sraw_value(lhs: u64, rhs: u64) -> u64 {
    sign_extend_word(((lhs as i32) >> ((rhs & 0x1f) as u32)) as u32 as u64)
}

fn srl_value(lhs: u64, rhs: u64) -> u64 {
    lhs.wrapping_shr((rhs & 0x3f) as u32)
}

fn srlw_value(lhs: u64, rhs: u64) -> u64 {
    sign_extend_word(u64::from((lhs as u32).wrapping_shr((rhs & 0x1f) as u32)))
}

fn runtime_binary_value(op: RuntimeBinaryOp, lhs: u64, rhs: u64) -> u64 {
    match op {
        RuntimeBinaryOp::Addw => addw_value(lhs, rhs),
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
                sign_extend_word(u64::from(dividend.wrapping_div(divisor)))
            }
        }
        RuntimeBinaryOp::Divw => {
            let dividend = lhs as i32;
            let divisor = rhs as i32;
            if divisor == 0 {
                u64::MAX
            } else {
                sign_extend_word(dividend.wrapping_div(divisor) as u32 as u64)
            }
        }
        RuntimeBinaryOp::Mul => lhs.wrapping_mul(rhs),
        RuntimeBinaryOp::Mulh => mulh_value(lhs, rhs),
        RuntimeBinaryOp::Mulhsu => {
            let multiplicand = lhs as i64 as i128;
            let multiplier = rhs as u128 as i128;
            ((multiplicand * multiplier) >> 64) as u64
        }
        RuntimeBinaryOp::Mulhu => mulhu_value(lhs, rhs),
        RuntimeBinaryOp::Mulw => mulw_value(lhs, rhs),
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
                sign_extend_word(u64::from(dividend))
            } else {
                sign_extend_word(u64::from(dividend.wrapping_rem(divisor)))
            }
        }
        RuntimeBinaryOp::Remw => {
            let dividend = lhs as i32;
            let divisor = rhs as i32;
            if divisor == 0 {
                sign_extend_word(dividend as u32 as u64)
            } else {
                sign_extend_word(dividend.wrapping_rem(divisor) as u32 as u64)
            }
        }
        RuntimeBinaryOp::Sllw => sllw_value(lhs, rhs),
        RuntimeBinaryOp::Slt => slt_value(lhs, rhs),
        RuntimeBinaryOp::Sltu => u64::from(lhs < rhs),
        RuntimeBinaryOp::Sraw => sraw_value(lhs, rhs),
        RuntimeBinaryOp::Srlw => srlw_value(lhs, rhs),
        RuntimeBinaryOp::Subw => subw_value(lhs, rhs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::InstructionTrace;

    fn plan(operations: Vec<BlockOperation>) -> BlockPlan {
        BlockPlan {
            start_pc: 0x1000,
            end_pc: 0x2000,
            operations,
            fingerprint: vec![(0x1000, 0x13)],
            code_version: 0,
            stop: super::super::BlockStop::MaxInstructions,
            guest_instruction_count: 1,
            profile_instructions: vec![InstructionTrace {
                pc: 0x1000,
                opcode: 0x13,
                text: "addi x1, x1, 0".to_string(),
            }],
        }
    }

    fn native(pc: u64, instruction: NativeInstruction) -> BlockOperation {
        BlockOperation {
            pc,
            opcode: 0x13,
            kind: BlockOperationKind::Native(instruction),
        }
    }

    #[test]
    fn removes_identity_self_moves_without_losing_guest_instruction_count() {
        let input = plan(vec![native(
            0x1000,
            NativeInstruction::Addi {
                rd: 1,
                rs1: 1,
                imm: 0,
            },
        )]);

        let (optimized, report) = optimize_plan(&input);

        assert!(optimized.operations.is_empty());
        assert_eq!(optimized.guest_instruction_count, 1);
        assert_eq!(optimized.profile_instructions.len(), 1);
        assert_eq!(report.removed_dead_writes, 1);
    }

    #[test]
    fn folds_branches_with_identical_sources() {
        let input = plan(vec![native(
            0x1000,
            NativeInstruction::Bne {
                rs1: 3,
                rs2: 3,
                target: 0x3000,
                fallthrough: 0x1004,
            },
        )]);

        let (optimized, report) = optimize_plan(&input);

        assert_eq!(report.folded_branches, 1);
        assert!(matches!(
            optimized.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::Jump { target: 0x1004 })
        ));
    }

    #[test]
    fn fuses_masked_zero_branches_without_dropping_architectural_write() {
        let mut input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Andi {
                    rd: 28,
                    rs1: 5,
                    imm: 1,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Beq {
                    rs1: 28,
                    rs2: 0,
                    target: 0x2000,
                    fallthrough: 0x1008,
                },
            ),
        ]);
        input.guest_instruction_count = 2;

        let (optimized, report) = optimize_plan(&input);

        assert_eq!(report.simplified_ops, 1);
        assert_eq!(optimized.guest_instruction_count, 2);
        assert_eq!(optimized.operations.len(), 1);
        assert!(matches!(
            optimized.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::AndBranch {
                rd: 28,
                rs1: 5,
                imm: 1,
                target: 0x2000,
                fallthrough: 0x1008,
                branch_if_zero: true,
            })
        ));
    }

    #[test]
    fn simplifies_register_copies_to_move_operations() {
        let input = plan(vec![native(
            0x1000,
            NativeInstruction::Addi {
                rd: 12,
                rs1: 15,
                imm: 0,
            },
        )]);

        let (optimized, report) = optimize_plan(&input);

        assert_eq!(report.simplified_ops, 1);
        assert!(matches!(
            optimized.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::Move { rd: 12, rs: 15 })
        ));
    }

    #[test]
    fn forwards_identical_doubleword_store_load_pairs() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Store {
                    rs1: 30,
                    rs2: 6,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Load {
                    rd: 31,
                    rs1: 30,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                    signed: false,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert_eq!(report.simplified_ops, 1);
        assert!(matches!(
            optimized.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::Store { .. })
        ));
        assert!(matches!(
            optimized.operations[1].kind(),
            BlockOperationKind::Native(NativeInstruction::Move { rd: 31, rs: 6 })
        ));
    }

    #[test]
    fn propagates_constants_and_folds_branch_targets() {
        let mut input = plan(vec![
            native(
                0x1000,
                NativeInstruction::LoadImmediate { rd: 5, value: 40 },
            ),
            native(
                0x1004,
                NativeInstruction::Addi {
                    rd: 6,
                    rs1: 5,
                    imm: 2,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Beq {
                    rs1: 6,
                    rs2: 0,
                    target: 0x3000,
                    fallthrough: 0x100c,
                },
            ),
        ]);
        input.guest_instruction_count = 3;

        let (optimized, report) = optimize_plan(&input);

        assert!(report.folded_constants >= 1);
        assert!(report.folded_branches >= 1);
        assert!(matches!(
            optimized.operations[1].kind(),
            BlockOperationKind::Native(NativeInstruction::LoadImmediate { rd: 6, value: 42 })
        ));
        assert!(matches!(
            optimized.operations[2].kind(),
            BlockOperationKind::Native(NativeInstruction::Jump { target: 0x100c })
        ));
    }

    #[test]
    fn propagates_copy_sources_into_later_arithmetic() {
        let input = plan(vec![
            native(0x1000, NativeInstruction::Move { rd: 10, rs: 11 }),
            native(
                0x1004,
                NativeInstruction::Add {
                    rd: 12,
                    rs1: 10,
                    rs2: 0,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.propagated_values >= 1);
        assert!(matches!(
            optimized.operations[1].kind(),
            BlockOperationKind::Native(NativeInstruction::Move { rd: 12, rs: 11 })
        ));
    }

    #[test]
    fn removes_overwritten_pure_integer_writes() {
        let input = plan(vec![
            native(0x1000, NativeInstruction::LoadImmediate { rd: 5, value: 1 }),
            native(
                0x1004,
                NativeInstruction::Addi {
                    rd: 5,
                    rs1: 5,
                    imm: 1,
                },
            ),
            native(0x1008, NativeInstruction::LoadImmediate { rd: 5, value: 7 }),
            native(
                0x100c,
                NativeInstruction::Add {
                    rd: 6,
                    rs1: 5,
                    rs2: 0,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.eliminated_overwritten_writes >= 2);
        assert!(optimized.operations.iter().all(|operation| {
            !matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::LoadImmediate {
                    rd: 5,
                    value: 1 | 2
                })
            )
        }));
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::LoadImmediate { rd: 5, value: 7 })
        )));
    }

    #[test]
    fn keeps_writes_before_faultable_memory_operations() {
        let input = plan(vec![
            native(0x1000, NativeInstruction::LoadImmediate { rd: 5, value: 1 }),
            native(
                0x1004,
                NativeInstruction::Store {
                    rs1: 10,
                    rs2: 11,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
            native(0x1008, NativeInstruction::LoadImmediate { rd: 5, value: 2 }),
        ]);

        let (optimized, _) = optimize_plan(&input);

        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::LoadImmediate { rd: 5, value: 1 })
        )));
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::LoadImmediate { rd: 5, value: 2 })
        )));
    }

    #[test]
    fn folds_constant_masked_branch_without_losing_destination_write() {
        let input = plan(vec![
            native(0x1000, NativeInstruction::LoadImmediate { rd: 4, value: 2 }),
            native(
                0x1004,
                NativeInstruction::AndBranch {
                    rd: 5,
                    rs1: 4,
                    imm: 1,
                    target: 0x3000,
                    fallthrough: 0x1008,
                    branch_if_zero: true,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.folded_branches >= 1);
        assert!(matches!(
            optimized.operations[1].kind(),
            BlockOperationKind::Native(NativeInstruction::LoadImmediate { rd: 5, value: 0 })
        ));
        assert!(matches!(
            optimized.operations[2].kind(),
            BlockOperationKind::Native(NativeInstruction::Jump { target: 0x3000 })
        ));
    }
}
