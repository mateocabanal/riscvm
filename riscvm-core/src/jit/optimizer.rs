use std::fmt;

use super::{BlockOperation, BlockOperationKind, BlockPlan, NativeInstruction};
use crate::cpu::RV64GCRegAbiName::Zero;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct OptimizationReport {
    pub removed_nops: u64,
    pub removed_dead_writes: u64,
    pub simplified_ops: u64,
    pub folded_branches: u64,
}

impl fmt::Display for OptimizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "removed_nops={} removed_dead_writes={} simplified_ops={} folded_branches={}",
            self.removed_nops, self.removed_dead_writes, self.simplified_ops, self.folded_branches
        )
    }
}

pub(crate) fn optimize_plan(plan: &BlockPlan) -> (BlockPlan, OptimizationReport) {
    let mut report = OptimizationReport::default();
    let mut operations = Vec::with_capacity(plan.operations.len());

    let mut index = 0;
    while index < plan.operations.len() {
        let operation = &plan.operations[index];
        let BlockOperationKind::Native(instruction) = operation.kind();

        if let Some(next_operation) = plan.operations.get(index + 1) {
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

        let optimized = optimize_instruction(instruction, &mut report);
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

        if let Some(next_operation) = plan.operations.get(index + 1) {
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

    let mut optimized = plan.clone();
    optimized.operations = operations;
    (optimized, report)
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
        NativeInstruction::Addi { rd, rs1, imm: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Addi { rd, rs1, imm } if rs1 == Zero as u8 && imm > 0 => {
            load_immediate(rd, imm as u64, report)
        }
        NativeInstruction::Slti { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, u64::from(0 < imm), report)
        }
        NativeInstruction::Sltiu { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, u64::from(0 < imm as u64), report)
        }
        NativeInstruction::Slt { rd, rs1, rs2 } if rs1 == rs2 => set_zero(rd, report),
        NativeInstruction::Sltu { rd, rs1, rs2 } if rs1 == rs2 => set_zero(rd, report),
        NativeInstruction::And { rd, rs1, .. } if rs1 == Zero as u8 => set_zero(rd, report),
        NativeInstruction::And { rd, rs2, .. } if rs2 == Zero as u8 => set_zero(rd, report),
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
        NativeInstruction::Or { rd, rs1, rs2 } if rs1 == Zero as u8 => copy_from(rd, rs2, report),
        NativeInstruction::Or { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Ori { rd, rs1, imm: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Sll { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Slli { rd, rs1, shamt: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Sra { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Srai { rd, rs1, shamt: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Srl { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Srli { rd, rs1, shamt: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Sub { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
        NativeInstruction::Xor { rd, rs1, rs2 } if rs1 == Zero as u8 => copy_from(rd, rs2, report),
        NativeInstruction::Xor { rd, rs1, rs2 } if rs2 == Zero as u8 => copy_from(rd, rs1, report),
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
}
