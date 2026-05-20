use std::{collections::HashMap, fmt};

use super::{
    BlockOperation, BlockOperationKind, BlockPlan, IntegerBranchCondition, MemoryWidth,
    NativeInstruction, RuntimeBinaryOp,
};
use crate::cpu::RV64GCRegAbiName::Zero;
use crate::sign_extend;

const GUEST_INTEGER_REGISTERS: usize = 32;
const MAX_OPTIMIZER_PASSES: usize = 4;
const MAX_SHIFTED_WORD_OR_DISTANCE: usize = 32;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct OptimizationReport {
    pub removed_nops: u64,
    pub removed_dead_writes: u64,
    pub simplified_ops: u64,
    pub folded_branches: u64,
    pub propagated_values: u64,
    pub folded_constants: u64,
    pub strength_reduced: u64,
    pub reused_expressions: u64,
    pub eliminated_overwritten_writes: u64,
}

impl fmt::Display for OptimizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "removed_nops={} removed_dead_writes={} simplified_ops={} folded_branches={} propagated_values={} folded_constants={} strength_reduced={} reused_expressions={} eliminated_overwritten_writes={}",
            self.removed_nops,
            self.removed_dead_writes,
            self.simplified_ops,
            self.folded_branches,
            self.propagated_values,
            self.folded_constants,
            self.strength_reduced,
            self.reused_expressions,
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
        operations = fold_shifted_word_or(&operations, &mut report);
        operations = fold_byte_copy_runs(&operations, &mut report);
        operations = fold_byte_pack_loads(&operations, &mut report);
        operations = propagate_constants_and_copies(&operations, &mut report);
        operations = eliminate_common_integer_expressions(&operations, &mut report);
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

#[derive(Debug, Clone, Copy)]
struct ShiftedWordOrFold {
    right_index: usize,
    left_index: usize,
    or_index: usize,
    replacement_index: usize,
    source: u8,
    right_temp: u8,
    left_temp: u8,
    final_register: u8,
    left_shamt: u32,
    right_shamt: u32,
}

fn fold_shifted_word_or(
    input: &[BlockOperation],
    report: &mut OptimizationReport,
) -> Vec<BlockOperation> {
    let mut output = input.to_vec();
    let mut consumed = vec![false; input.len()];

    for index in 0..input.len() {
        if consumed[index] {
            continue;
        }
        let Some(fold) = match_shifted_word_or(input, index) else {
            continue;
        };
        if consumed[fold.right_index] || consumed[fold.left_index] {
            continue;
        }
        if !byte_pack_temporaries_are_dead(
            input,
            fold.or_index + 1,
            &[fold.right_temp, fold.left_temp],
            fold.final_register,
        ) {
            continue;
        }

        output[fold.right_index] = nopped_operation(input[fold.right_index]);
        output[fold.left_index] = nopped_operation(input[fold.left_index]);
        output[fold.or_index] = nopped_operation(input[fold.or_index]);
        output[fold.replacement_index] = BlockOperation {
            pc: input[fold.replacement_index].pc(),
            opcode: input[fold.replacement_index].opcode(),
            kind: BlockOperationKind::Native(NativeInstruction::ShiftedWordOr {
                rd: fold.final_register,
                rs1: fold.source,
                left_shamt: fold.left_shamt,
                right_shamt: fold.right_shamt,
            }),
        };
        consumed[fold.right_index] = true;
        consumed[fold.left_index] = true;
        consumed[fold.or_index] = true;
        report.simplified_ops += 2;
        report.strength_reduced += 1;
    }

    output
}

fn match_shifted_word_or(input: &[BlockOperation], or_index: usize) -> Option<ShiftedWordOrFold> {
    let BlockOperationKind::Native(NativeInstruction::Or { rd, rs1, rs2 }) = input[or_index].kind()
    else {
        return None;
    };
    if rd == Zero as u8 || rs1 == rs2 {
        return None;
    }

    match (
        find_shifted_word_or_input(input, or_index, rs1)?,
        find_shifted_word_or_input(input, or_index, rs2)?,
    ) {
        (ShiftedWordOrInput::Left(left), ShiftedWordOrInput::Right(right))
        | (ShiftedWordOrInput::Right(right), ShiftedWordOrInput::Left(left)) => {
            build_shifted_word_or_fold(input, or_index, rd, left, right)
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct ShiftedWordOrInputDef {
    index: usize,
    temp: u8,
    source: u8,
    shamt: u32,
}

#[derive(Debug, Clone, Copy)]
enum ShiftedWordOrInput {
    Left(ShiftedWordOrInputDef),
    Right(ShiftedWordOrInputDef),
}

fn find_shifted_word_or_input(
    input: &[BlockOperation],
    or_index: usize,
    register: u8,
) -> Option<ShiftedWordOrInput> {
    if register == Zero as u8 {
        return None;
    }
    let start = or_index.saturating_sub(MAX_SHIFTED_WORD_OR_DISTANCE);
    for index in (start..or_index).rev() {
        let BlockOperationKind::Native(instruction) = input[index].kind();
        if shifted_word_or_tracking_barrier(instruction) {
            return None;
        }
        if integer_write_register(instruction) != Some(register) {
            continue;
        }
        return match instruction {
            NativeInstruction::Slli { rd, rs1, shamt } => {
                Some(ShiftedWordOrInput::Left(ShiftedWordOrInputDef {
                    index,
                    temp: rd,
                    source: rs1,
                    shamt,
                }))
            }
            NativeInstruction::Srliw { rd, rs1, shamt } => {
                Some(ShiftedWordOrInput::Right(ShiftedWordOrInputDef {
                    index,
                    temp: rd,
                    source: rs1,
                    shamt,
                }))
            }
            _ => None,
        };
    }

    None
}

fn build_shifted_word_or_fold(
    input: &[BlockOperation],
    or_index: usize,
    final_register: u8,
    left: ShiftedWordOrInputDef,
    right: ShiftedWordOrInputDef,
) -> Option<ShiftedWordOrFold> {
    if left.source != right.source
        || left.source == Zero as u8
        || left.temp == Zero as u8
        || right.temp == Zero as u8
        || left.temp == right.temp
        || right.temp == right.source
        || left.shamt == 0
        || left.shamt >= 32
        || right.shamt == 0
        || right.shamt >= 32
        || left.shamt + right.shamt != 32
    {
        return None;
    }
    let replacement_index = if left.temp == left.source {
        if final_register != left.temp || right.index > left.index {
            return None;
        }
        left.index
    } else {
        or_index
    };
    if !shifted_word_or_window_is_safe(input, or_index, left, right) {
        return None;
    }

    Some(ShiftedWordOrFold {
        right_index: right.index,
        left_index: left.index,
        or_index,
        replacement_index,
        source: left.source,
        right_temp: right.temp,
        left_temp: left.temp,
        final_register,
        left_shamt: left.shamt,
        right_shamt: right.shamt,
    })
}

fn shifted_word_or_window_is_safe(
    input: &[BlockOperation],
    or_index: usize,
    left: ShiftedWordOrInputDef,
    right: ShiftedWordOrInputDef,
) -> bool {
    let first_shift = left.index.min(right.index);
    for index in first_shift + 1..or_index {
        if index == left.index || index == right.index {
            continue;
        }
        let BlockOperationKind::Native(instruction) = input[index].kind();
        if shifted_word_or_tracking_barrier(instruction)
            || instruction_reads_any(instruction, &[left.temp, right.temp])
        {
            return false;
        }
        if let Some(rd) = integer_write_register(instruction) {
            if rd == left.source || rd == left.temp || rd == right.temp {
                return false;
            }
        }
    }

    true
}

fn shifted_word_or_tracking_barrier(instruction: NativeInstruction) -> bool {
    if matches!(instruction, NativeInstruction::Load { .. }) {
        return false;
    }
    liveness_barrier(instruction)
}

fn instruction_reads_any(instruction: NativeInstruction, registers: &[u8]) -> bool {
    let mut reads = [false; GUEST_INTEGER_REGISTERS];
    mark_read_registers(instruction, &mut reads);
    registers
        .iter()
        .any(|&register| register != Zero as u8 && reads[usize::from(register)])
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BytePackEntry {
    imm: i64,
    byte_index: u8,
    signed: bool,
}

impl BytePackEntry {
    const fn empty() -> Self {
        Self {
            imm: 0,
            byte_index: 0,
            signed: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BytePackExpr {
    base: u8,
    entries: [BytePackEntry; 8],
    len: usize,
}

impl BytePackExpr {
    fn single(base: u8, imm: i64, signed: bool) -> Self {
        let mut entries = [BytePackEntry::empty(); 8];
        entries[0] = BytePackEntry {
            imm,
            byte_index: 0,
            signed,
        };
        Self {
            base,
            entries,
            len: 1,
        }
    }

    fn shifted(self, shamt: u32) -> Option<Self> {
        if shamt % 8 != 0 {
            return None;
        }
        let byte_shift = u8::try_from(shamt / 8).ok()?;
        let mut shifted = self;
        for entry in &mut shifted.entries[..shifted.len] {
            entry.byte_index = entry.byte_index.checked_add(byte_shift)?;
            if entry.byte_index >= 8 {
                return None;
            }
        }
        Some(shifted)
    }

    fn or(self, other: Self) -> Option<Self> {
        if self.base != other.base || self.len + other.len > 8 {
            return None;
        }

        let mut combined = self;
        for &entry in &other.entries[..other.len] {
            if combined
                .entries
                .iter()
                .take(combined.len)
                .any(|existing| existing.byte_index == entry.byte_index)
            {
                return None;
            }
            combined.entries[combined.len] = entry;
            combined.len += 1;
        }
        Some(combined)
    }

    fn little_endian_load(self) -> Option<(i64, MemoryWidth, bool)> {
        let mut by_index = [None; 8];
        for &entry in &self.entries[..self.len] {
            by_index[usize::from(entry.byte_index)] = Some(entry);
        }

        match self.len {
            4 => {
                let base_imm = by_index[0]?.imm;
                if !by_index[3]?.signed {
                    return None;
                }
                for (byte_index, entry) in by_index.iter().take(4).enumerate() {
                    let entry = (*entry)?;
                    if entry.imm != base_imm.checked_add(byte_index as i64)? {
                        return None;
                    }
                    if byte_index < 3 && entry.signed {
                        return None;
                    }
                }
                Some((base_imm, MemoryWidth::Word, by_index[3]?.signed))
            }
            8 => {
                let base_imm = by_index[0]?.imm;
                for (byte_index, entry) in by_index.iter().enumerate() {
                    let entry = (*entry)?;
                    if entry.imm != base_imm.checked_add(byte_index as i64)? {
                        return None;
                    }
                    if entry.signed && byte_index != 7 {
                        return None;
                    }
                }
                Some((base_imm, MemoryWidth::Double, false))
            }
            _ => None,
        }
    }
}

fn byte_pack_expr_uses_only_low_bytes(expr: &BytePackExpr, bytes: usize) -> bool {
    if expr.len != bytes {
        return false;
    }
    let mut seen = [false; 8];
    for entry in &expr.entries[..expr.len] {
        let index = usize::from(entry.byte_index);
        if index >= bytes {
            return false;
        }
        seen[index] = true;
    }
    seen.iter().take(bytes).all(|seen| *seen)
}

fn folded_byte_load_report_count(width: MemoryWidth) -> u64 {
    match width {
        MemoryWidth::Word => 4,
        MemoryWidth::Double => 8,
        MemoryWidth::Byte | MemoryWidth::Half => 1,
    }
}

impl BytePackExpr {
    fn can_fold_to_load(self) -> Option<(i64, MemoryWidth, bool)> {
        let load = self.little_endian_load()?;
        let bytes = match load.1 {
            MemoryWidth::Word => 4,
            MemoryWidth::Double => 8,
            MemoryWidth::Byte | MemoryWidth::Half => {
                return None;
            }
        };
        byte_pack_expr_uses_only_low_bytes(&self, bytes).then_some(load)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrackedBytePackExpr {
    expr: BytePackExpr,
    contributors: Vec<usize>,
    base_clobbers: Vec<usize>,
    written: Vec<u8>,
}

impl TrackedBytePackExpr {
    fn single(base: u8, imm: i64, signed: bool, index: usize, rd: u8) -> Self {
        Self {
            expr: BytePackExpr::single(base, imm, signed),
            contributors: vec![index],
            base_clobbers: Vec::new(),
            written: vec![rd],
        }
    }

    fn shifted(&self, shamt: u32, index: usize, rd: u8) -> Option<Self> {
        let mut shifted = Self {
            expr: self.expr.shifted(shamt)?,
            contributors: self.contributors.clone(),
            base_clobbers: self.base_clobbers.clone(),
            written: self.written.clone(),
        };
        push_unique_usize(&mut shifted.contributors, index);
        push_unique_u8(&mut shifted.written, rd);
        Some(shifted)
    }

    fn or(&self, other: &Self, index: usize, rd: u8) -> Option<Self> {
        let mut combined = Self {
            expr: self.expr.or(other.expr)?,
            contributors: self.contributors.clone(),
            base_clobbers: self.base_clobbers.clone(),
            written: self.written.clone(),
        };
        for &contributor in &other.contributors {
            push_unique_usize(&mut combined.contributors, contributor);
        }
        for &base_clobber in &other.base_clobbers {
            push_unique_usize(&mut combined.base_clobbers, base_clobber);
        }
        push_unique_usize(&mut combined.contributors, index);
        for &register in &other.written {
            push_unique_u8(&mut combined.written, register);
        }
        push_unique_u8(&mut combined.written, rd);
        Some(combined)
    }
}

fn fold_byte_pack_loads(
    input: &[BlockOperation],
    report: &mut OptimizationReport,
) -> Vec<BlockOperation> {
    let mut output = input.to_vec();
    let mut expressions = vec![None::<TrackedBytePackExpr>; GUEST_INTEGER_REGISTERS];

    for (index, operation) in input.iter().enumerate() {
        let BlockOperationKind::Native(instruction) = operation.kind();

        if byte_pack_tracking_barrier(instruction) {
            expressions.fill(None);
            continue;
        }

        let tracked = track_byte_pack_instruction(instruction, index, &expressions);
        if tracked.is_none() {
            invalidate_byte_pack_reads(instruction, &mut expressions);
        }

        if let Some(rd) = integer_write_register(instruction) {
            if tracked
                .as_ref()
                .is_some_and(|(_, tracked)| tracked.expr.base == rd)
            {
                remember_byte_pack_base_clobber(rd, index, &mut expressions);
            } else {
                invalidate_byte_pack_base_write(rd, &mut expressions);
            }
            drop_visible_byte_pack_write(rd, &mut expressions);
            if tracked.is_none() {
                expressions[usize::from(rd)] = None;
            }
        }

        let Some((rd, tracked)) = tracked else {
            continue;
        };
        expressions[usize::from(rd)] = Some(tracked.clone());

        let Some((base_imm, width, signed)) = tracked.expr.can_fold_to_load() else {
            continue;
        };
        if !byte_pack_base_clobbers_are_contributors(&tracked) {
            continue;
        }
        let dead_writes = byte_pack_dead_writes_after(input, index + 1, &tracked.written, rd);
        let Some(contributors_to_nop) =
            byte_pack_contributors_to_nop(input, index, rd, &tracked, &dead_writes)
        else {
            continue;
        };

        for &contributor in &contributors_to_nop {
            output[contributor] = nopped_operation(input[contributor]);
        }
        output[index] = BlockOperation {
            pc: operation.pc(),
            opcode: operation.opcode(),
            kind: BlockOperationKind::Native(NativeInstruction::Load {
                rd,
                rs1: tracked.expr.base,
                imm: base_imm,
                width,
                signed,
            }),
        };
        report.simplified_ops += contributors_to_nop.len() as u64 + 1;
        report.strength_reduced += folded_byte_load_report_count(width);
        invalidate_byte_pack_contributors(&tracked.contributors, &mut expressions);
    }

    output
}

fn track_byte_pack_instruction(
    instruction: NativeInstruction,
    index: usize,
    expressions: &[Option<TrackedBytePackExpr>],
) -> Option<(u8, TrackedBytePackExpr)> {
    match instruction {
        NativeInstruction::Load {
            rd,
            rs1,
            imm,
            width: MemoryWidth::Byte,
            signed,
        } if rd != Zero as u8 && rs1 != Zero as u8 => {
            if expressions
                .get(usize::from(rs1))
                .is_some_and(Option::is_some)
            {
                return None;
            }
            Some((rd, TrackedBytePackExpr::single(rs1, imm, signed, index, rd)))
        }
        NativeInstruction::Slli { rd, rs1, shamt } if rd != Zero as u8 => {
            let expression = expressions
                .get(usize::from(rs1))
                .and_then(Option::as_ref)?
                .shifted(shamt, index, rd)?;
            Some((rd, expression))
        }
        NativeInstruction::Or { rd, rs1, rs2 } if rd != Zero as u8 => {
            let lhs = expressions.get(usize::from(rs1)).and_then(Option::as_ref)?;
            let rhs = expressions.get(usize::from(rs2)).and_then(Option::as_ref)?;
            Some((rd, lhs.or(rhs, index, rd)?))
        }
        NativeInstruction::Move { rd, rs } if rd != Zero as u8 => {
            let mut expression = expressions
                .get(usize::from(rs))
                .and_then(Option::as_ref)?
                .clone();
            push_unique_usize(&mut expression.contributors, index);
            push_unique_u8(&mut expression.written, rd);
            Some((rd, expression))
        }
        NativeInstruction::Nop => None,
        _ => None,
    }
}

fn nopped_operation(operation: BlockOperation) -> BlockOperation {
    BlockOperation {
        pc: operation.pc(),
        opcode: operation.opcode(),
        kind: BlockOperationKind::Native(NativeInstruction::Nop),
    }
}

#[derive(Debug)]
struct ByteCopyRun {
    load_indices: Vec<usize>,
    store_indices: Vec<usize>,
    load_base: u8,
    load_imm: i64,
    store_base: u8,
    store_imm: i64,
    temp_register: u8,
    registers: [u8; 8],
    visible_registers: Vec<u8>,
    end: usize,
}

fn fold_byte_copy_runs(
    input: &[BlockOperation],
    report: &mut OptimizationReport,
) -> Vec<BlockOperation> {
    let mut output = input.to_vec();
    let mut index = 0;

    while index < input.len() {
        let Some(run) = match_byte_copy_run(input, index) else {
            index += 1;
            continue;
        };
        let load_slot = run.load_indices[0];
        let store_slot = run.store_indices[0];
        let temporaries_are_dead =
            byte_pack_temporaries_are_dead(input, run.end, &run.visible_registers, Zero as u8);

        if temporaries_are_dead {
            output[load_slot] = BlockOperation {
                pc: input[load_slot].pc(),
                opcode: input[load_slot].opcode(),
                kind: BlockOperationKind::Native(NativeInstruction::Load {
                    rd: run.temp_register,
                    rs1: run.load_base,
                    imm: run.load_imm,
                    width: MemoryWidth::Double,
                    signed: false,
                }),
            };
            output[store_slot] = BlockOperation {
                pc: input[store_slot].pc(),
                opcode: input[store_slot].opcode(),
                kind: BlockOperationKind::Native(NativeInstruction::Store {
                    rs1: run.store_base,
                    rs2: run.temp_register,
                    imm: run.store_imm,
                    width: MemoryWidth::Double,
                }),
            };
        } else {
            output[store_slot] = BlockOperation {
                pc: input[store_slot].pc(),
                opcode: input[store_slot].opcode(),
                kind: BlockOperationKind::Native(NativeInstruction::ByteCopy8 {
                    registers: run.registers,
                    load_base: run.load_base,
                    load_imm: run.load_imm,
                    store_base: run.store_base,
                    store_imm: run.store_imm,
                }),
            };
        }

        for &load_index in &run.load_indices {
            if !temporaries_are_dead || load_index != load_slot {
                output[load_index] = nopped_operation(input[load_index]);
            }
        }
        for &store_index in &run.store_indices {
            if store_index != store_slot {
                output[store_index] = nopped_operation(input[store_index]);
            }
        }
        let retained_operations = if temporaries_are_dead { 2 } else { 1 };
        report.simplified_ops +=
            (run.load_indices.len() + run.store_indices.len() - retained_operations) as u64;
        index = run.end;
    }

    output
}

fn match_byte_copy_run(input: &[BlockOperation], start: usize) -> Option<ByteCopyRun> {
    if start + 8 > input.len() {
        return None;
    }

    let mut load_base = None;
    let mut loads = Vec::with_capacity(8);
    let mut load_registers = [false; GUEST_INTEGER_REGISTERS];
    for index in start..start + 8 {
        let BlockOperationKind::Native(instruction) = input[index].kind();
        let NativeInstruction::Load {
            rd,
            rs1,
            imm,
            width: MemoryWidth::Byte,
            signed: false,
        } = instruction
        else {
            return None;
        };
        if rd == Zero as u8 || rs1 == Zero as u8 || rd == rs1 || load_registers[usize::from(rd)] {
            return None;
        }
        if let Some(base) = load_base {
            if base != rs1 {
                return None;
            }
        } else {
            load_base = Some(rs1);
        }
        load_registers[usize::from(rd)] = true;
        loads.push((imm, rd, index));
    }

    loads.sort_by_key(|(imm, _, _)| *imm);
    let load_imm = loads.first()?.0;
    let mut registers = [Zero as u8; 8];
    for (byte_index, (imm, _, _)) in loads.iter().enumerate() {
        if *imm != load_imm.checked_add(byte_index as i64)? {
            return None;
        }
        registers[byte_index] = loads[byte_index].1;
    }

    let mut source_imm_by_register = [None; GUEST_INTEGER_REGISTERS];
    let mut stored_registers = [false; GUEST_INTEGER_REGISTERS];
    let mut visible_registers = Vec::with_capacity(8);
    for &(imm, rd, _) in &loads {
        source_imm_by_register[usize::from(rd)] = Some(imm);
        visible_registers.push(rd);
    }

    let load_indices = loads.iter().map(|(_, _, index)| *index).collect::<Vec<_>>();
    let mut store_indices = Vec::with_capacity(8);
    let mut store_base = None;
    let mut store_delta = None;
    let mut scan = start + 8;

    while scan < input.len() && scan < start + 32 {
        let BlockOperationKind::Native(instruction) = input[scan].kind();

        if let NativeInstruction::Store {
            rs1,
            rs2,
            imm,
            width: MemoryWidth::Byte,
        } = instruction
        {
            let source_imm = source_imm_by_register[usize::from(rs2)]?;
            if stored_registers[usize::from(rs2)] {
                return None;
            }
            if let Some(base) = store_base {
                if base != rs1 {
                    return None;
                }
            } else if rs1 == Zero as u8 || load_registers[usize::from(rs1)] {
                return None;
            } else {
                store_base = Some(rs1);
            }

            let delta = imm.checked_sub(source_imm)?;
            if let Some(existing_delta) = store_delta {
                if existing_delta != delta {
                    return None;
                }
            } else {
                store_delta = Some(delta);
            }
            stored_registers[usize::from(rs2)] = true;
            store_indices.push(scan);
            if store_indices.len() == 8 {
                return Some(ByteCopyRun {
                    load_indices,
                    store_indices,
                    load_base: load_base?,
                    load_imm,
                    store_base: store_base?,
                    store_imm: load_imm.checked_add(store_delta?)?,
                    temp_register: loads[0].1,
                    registers,
                    visible_registers,
                    end: scan + 1,
                });
            }
            scan += 1;
            continue;
        }

        if byte_copy_scan_barrier(instruction) {
            return None;
        }

        let mut reads = [false; GUEST_INTEGER_REGISTERS];
        mark_read_registers(instruction, &mut reads);
        if visible_registers
            .iter()
            .any(|&register| reads[usize::from(register)])
        {
            return None;
        }

        if let Some(rd) = integer_write_register(instruction) {
            if Some(rd) == load_base || Some(rd) == store_base {
                return None;
            }
            if load_registers[usize::from(rd)] {
                if !stored_registers[usize::from(rd)] {
                    return None;
                }
                visible_registers.retain(|&register| register != rd);
            }
        }

        scan += 1;
    }

    None
}

fn byte_copy_scan_barrier(instruction: NativeInstruction) -> bool {
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
            | NativeInstruction::ByteCopy8 { .. }
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
            | NativeInstruction::StoreLoadForwardLoop(_)
            | NativeInstruction::TraceGuard { .. }
            | NativeInstruction::TraceLoopGuard { .. }
    )
}

fn invalidate_byte_pack_reads(
    instruction: NativeInstruction,
    expressions: &mut [Option<TrackedBytePackExpr>],
) {
    let mut reads = [false; GUEST_INTEGER_REGISTERS];
    mark_read_registers(instruction, &mut reads);
    if !reads.iter().any(|read| *read) {
        return;
    }

    for expression in expressions {
        if expression.as_ref().is_some_and(|expression| {
            expression
                .written
                .iter()
                .any(|&register| reads[usize::from(register)])
        }) {
            *expression = None;
        }
    }
}

fn invalidate_byte_pack_base_write(rd: u8, expressions: &mut [Option<TrackedBytePackExpr>]) {
    if rd == Zero as u8 {
        return;
    }
    for expression in expressions {
        if expression
            .as_ref()
            .is_some_and(|expression| expression.expr.base == rd)
        {
            *expression = None;
        }
    }
}

fn remember_byte_pack_base_clobber(
    rd: u8,
    index: usize,
    expressions: &mut [Option<TrackedBytePackExpr>],
) {
    if rd == Zero as u8 {
        return;
    }
    for expression in expressions.iter_mut().flatten() {
        if expression.expr.base == rd {
            push_unique_usize(&mut expression.base_clobbers, index);
        }
    }
}

fn byte_pack_base_clobbers_are_contributors(expression: &TrackedBytePackExpr) -> bool {
    expression
        .base_clobbers
        .iter()
        .all(|clobber| expression.contributors.contains(clobber))
}

fn byte_pack_dead_writes_after(
    input: &[BlockOperation],
    after_pack: usize,
    written: &[u8],
    final_register: u8,
) -> [bool; GUEST_INTEGER_REGISTERS] {
    let mut pending = [false; GUEST_INTEGER_REGISTERS];
    let mut dead = [false; GUEST_INTEGER_REGISTERS];
    for &register in written {
        if register != final_register && register != Zero as u8 {
            pending[usize::from(register)] = true;
        }
    }
    if !pending.iter().any(|live| *live) {
        return dead;
    }

    for operation in &input[after_pack..] {
        let BlockOperationKind::Native(instruction) = operation.kind();

        let mut reads = [false; GUEST_INTEGER_REGISTERS];
        mark_read_registers(instruction, &mut reads);
        for register in 0..GUEST_INTEGER_REGISTERS {
            if pending[register] && reads[register] {
                pending[register] = false;
            }
        }
        if !pending.iter().any(|live| *live) {
            return dead;
        }

        if byte_pack_liveness_barrier(instruction) {
            return dead;
        }

        if let Some(rd) = integer_write_register(instruction) {
            let register = usize::from(rd);
            if pending[register] {
                pending[register] = false;
                dead[register] = true;
                if !pending.iter().any(|live| *live) {
                    return dead;
                }
            }
        }
    }

    dead
}

fn byte_pack_contributors_to_nop(
    input: &[BlockOperation],
    final_index: usize,
    final_register: u8,
    tracked: &TrackedBytePackExpr,
    dead_writes: &[bool; GUEST_INTEGER_REGISTERS],
) -> Option<Vec<usize>> {
    let mut preserved = Vec::new();
    let mut contributors = Vec::new();
    for &contributor in &tracked.contributors {
        if contributor == final_index {
            continue;
        }
        let BlockOperationKind::Native(instruction) = input[contributor].kind();
        let Some(rd) = integer_write_register(instruction) else {
            contributors.push(contributor);
            continue;
        };
        if rd == final_register || !tracked.written.contains(&rd) || dead_writes[usize::from(rd)] {
            contributors.push(contributor);
            continue;
        }
        preserve_byte_pack_contributor(input, tracked, contributor, &mut preserved)?;
    }

    contributors.clear();
    for &contributor in &tracked.contributors {
        if contributor != final_index && !preserved.contains(&contributor) {
            contributors.push(contributor);
        }
    }
    Some(contributors)
}

fn preserve_byte_pack_contributor(
    input: &[BlockOperation],
    tracked: &TrackedBytePackExpr,
    contributor: usize,
    preserved: &mut Vec<usize>,
) -> Option<()> {
    if preserved.contains(&contributor) {
        return Some(());
    }
    let BlockOperationKind::Native(instruction) = input[contributor].kind();
    if !byte_pack_preserved_contributor_is_safe(instruction, tracked.expr.base) {
        return None;
    }
    preserved.push(contributor);

    let mut reads = [false; GUEST_INTEGER_REGISTERS];
    mark_read_registers(instruction, &mut reads);
    for register in 0..GUEST_INTEGER_REGISTERS {
        if !reads[register] {
            continue;
        }
        let Some(&dependency) = tracked.contributors.iter().rev().find(|&&candidate| {
            candidate < contributor
                && matches!(
                    input[candidate].kind(),
                    BlockOperationKind::Native(candidate_instruction)
                        if integer_write_register(candidate_instruction)
                            == Some(register as u8)
                )
        }) else {
            continue;
        };
        preserve_byte_pack_contributor(input, tracked, dependency, preserved)?;
    }
    Some(())
}

fn byte_pack_preserved_contributor_is_safe(instruction: NativeInstruction, base: u8) -> bool {
    match instruction {
        NativeInstruction::Load {
            rd,
            width: MemoryWidth::Byte,
            ..
        }
        | NativeInstruction::Move { rd, .. }
        | NativeInstruction::Or { rd, .. }
        | NativeInstruction::Slli { rd, .. } => rd != base,
        _ => false,
    }
}

fn drop_visible_byte_pack_write(rd: u8, expressions: &mut [Option<TrackedBytePackExpr>]) {
    if rd == Zero as u8 {
        return;
    }
    for expression in expressions.iter_mut().flatten() {
        expression.written.retain(|&register| register != rd);
    }
}

fn invalidate_byte_pack_contributors(
    contributors: &[usize],
    expressions: &mut [Option<TrackedBytePackExpr>],
) {
    for expression in expressions {
        if expression.as_ref().is_some_and(|expression| {
            expression
                .contributors
                .iter()
                .any(|contributor| contributors.contains(contributor))
        }) {
            *expression = None;
        }
    }
}

fn push_unique_usize(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn push_unique_u8(values: &mut Vec<u8>, value: u8) {
    if value != Zero as u8 && !values.contains(&value) {
        values.push(value);
    }
}

fn byte_pack_temporaries_are_dead(
    input: &[BlockOperation],
    after_pack: usize,
    written: &[u8],
    final_register: u8,
) -> bool {
    let mut pending = [false; GUEST_INTEGER_REGISTERS];
    for &register in written {
        if register != final_register && register != Zero as u8 {
            pending[usize::from(register)] = true;
        }
    }
    if !pending.iter().any(|live| *live) {
        return true;
    }

    for operation in &input[after_pack..] {
        let BlockOperationKind::Native(instruction) = operation.kind();

        let mut reads = [false; GUEST_INTEGER_REGISTERS];
        mark_read_registers(instruction, &mut reads);
        if pending
            .iter()
            .zip(reads)
            .any(|(still_pending, is_read)| *still_pending && is_read)
        {
            return false;
        }

        if byte_pack_liveness_barrier(instruction) {
            return !pending.iter().any(|live| *live);
        }

        if let Some(rd) = integer_write_register(instruction) {
            pending[usize::from(rd)] = false;
            if !pending.iter().any(|live| *live) {
                return true;
            }
        }
    }

    !pending.iter().any(|live| *live)
}

fn byte_pack_liveness_barrier(instruction: NativeInstruction) -> bool {
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
            | NativeInstruction::ByteCopy8 { .. }
            | NativeInstruction::CountedDiamondLoop(_)
            | NativeInstruction::DivisionRecurrenceLoop(_)
            | NativeInstruction::Ecall { .. }
            | NativeInstruction::FibonacciRecurrenceLoop(_)
            | NativeInstruction::Jal { .. }
            | NativeInstruction::Jalr { .. }
            | NativeInstruction::Jump { .. }
            | NativeInstruction::JumpReg { .. }
            | NativeInstruction::JumpRegLink { .. }
            | NativeInstruction::RuntimeAtomic { .. }
            | NativeInstruction::RuntimeCsr { .. }
            | NativeInstruction::RuntimeFloat { .. }
            | NativeInstruction::RuntimeTrap { .. }
            | NativeInstruction::StoreLoadForwardLoop(_)
            | NativeInstruction::TraceGuard { .. }
            | NativeInstruction::TraceLoopGuard { .. }
    )
}

fn byte_pack_tracking_barrier(instruction: NativeInstruction) -> bool {
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
            | NativeInstruction::ByteCopy8 { .. }
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

fn integer_write_register(instruction: NativeInstruction) -> Option<u8> {
    match instruction {
        NativeInstruction::Load { rd, .. } => Some(rd),
        _ => pure_integer_write(instruction),
    }
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
        NativeInstruction::Add { rd, rs1, rs2 } if rs1 == rs2 => {
            report.strength_reduced += 1;
            NativeInstruction::Slli { rd, rs1, shamt: 1 }
        }
        NativeInstruction::Addi { rd, rs1, imm: 0 } => copy_from(rd, rs1, report),
        NativeInstruction::Addi { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, imm as u64, report)
        }
        NativeInstruction::Addiw { rd, rs1, imm } if rs1 == Zero as u8 => {
            load_immediate(rd, sign_extend_word(imm as u64), report)
        }
        NativeInstruction::Addw { rd, rs1, rs2 } if rs1 == rs2 => {
            report.strength_reduced += 1;
            NativeInstruction::Slliw { rd, rs1, shamt: 1 }
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
            | NativeInstruction::ShiftedWordOr { rd: 0, .. }
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
        NativeInstruction::ShiftedWordOr {
            rd,
            rs1,
            left_shamt,
            right_shamt,
        } => NativeInstruction::ShiftedWordOr {
            rd,
            rs1: rewrite_register(rs1, facts, &mut propagated),
            left_shamt,
            right_shamt,
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
            match fold_binary_value(rd, rs1, rs2, facts, u64::wrapping_mul, report) {
                Some(folded) => Some(folded),
                None => strength_reduce_multiply(rd, rs1, rs2, facts, false, report),
            }
        }
        NativeInstruction::Mulh { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, mulh_value, report)
        }
        NativeInstruction::Mulhu { rd, rs1, rs2 } => {
            fold_binary_value(rd, rs1, rs2, facts, mulhu_value, report)
        }
        NativeInstruction::Mulw { rd, rs1, rs2 } => {
            match fold_binary_value(rd, rs1, rs2, facts, mulw_value, report) {
                Some(folded) => Some(folded),
                None => strength_reduce_multiply(rd, rs1, rs2, facts, true, report),
            }
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
        NativeInstruction::ShiftedWordOr {
            rd,
            rs1,
            left_shamt,
            right_shamt,
        } => facts.constant(rs1).map(|value| {
            let right =
                sign_extend_word(u64::from((value as u32).wrapping_shr(right_shamt & 0x1f)));
            folded_constant(rd, value.wrapping_shl(left_shamt & 0x3f) | right, report)
        }),
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

fn strength_reduce_multiply(
    rd: u8,
    rs1: u8,
    rs2: u8,
    facts: &RegisterFacts,
    word: bool,
    report: &mut OptimizationReport,
) -> Option<NativeInstruction> {
    let lhs = facts.constant(rs1);
    let rhs = facts.constant(rs2);
    match (lhs, rhs) {
        (Some(constant), None) => {
            strength_reduce_multiply_by_constant(rd, rs2, constant, word, report)
        }
        (None, Some(constant)) => {
            strength_reduce_multiply_by_constant(rd, rs1, constant, word, report)
        }
        _ => None,
    }
}

fn strength_reduce_multiply_by_constant(
    rd: u8,
    variable: u8,
    constant: u64,
    word: bool,
    report: &mut OptimizationReport,
) -> Option<NativeInstruction> {
    let effective = if word {
        u64::from(constant as u32)
    } else {
        constant
    };

    let reduced = match effective {
        0 => folded_constant(rd, 0, report),
        1 if word => NativeInstruction::Addiw {
            rd,
            rs1: variable,
            imm: 0,
        },
        1 => copy_from(rd, variable, report),
        value if value.is_power_of_two() => {
            let shamt = value.trailing_zeros();
            if word {
                NativeInstruction::Slliw {
                    rd,
                    rs1: variable,
                    shamt,
                }
            } else {
                NativeInstruction::Slli {
                    rd,
                    rs1: variable,
                    shamt,
                }
            }
        }
        _ => return None,
    };

    report.strength_reduced += 1;
    Some(reduced)
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
        | NativeInstruction::ShiftedWordOr { rd, .. }
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
        | NativeInstruction::ByteCopy8 { .. }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CommonExpression {
    Binary {
        op: BinaryExpressionOp,
        lhs: u8,
        rhs: u8,
    },
    Immediate {
        op: ImmediateExpressionOp,
        rs: u8,
        imm: i64,
    },
}

impl CommonExpression {
    fn from_instruction(instruction: NativeInstruction) -> Option<Self> {
        match instruction {
            NativeInstruction::Add { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Add, rs1, rs2, true))
            }
            NativeInstruction::Addi { rs1, imm, .. } => {
                Some(Self::immediate(ImmediateExpressionOp::Addi, rs1, imm))
            }
            NativeInstruction::Addiw { rs1, imm, .. } => {
                Some(Self::immediate(ImmediateExpressionOp::Addiw, rs1, imm))
            }
            NativeInstruction::Addw { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Addw, rs1, rs2, true))
            }
            NativeInstruction::And { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::And, rs1, rs2, true))
            }
            NativeInstruction::Andi { rs1, imm, .. } => {
                Some(Self::immediate(ImmediateExpressionOp::Andi, rs1, imm))
            }
            NativeInstruction::Mul { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Mul, rs1, rs2, true))
            }
            NativeInstruction::Mulh { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Mulh, rs1, rs2, true))
            }
            NativeInstruction::Mulhu { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Mulhu, rs1, rs2, true))
            }
            NativeInstruction::Mulw { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Mulw, rs1, rs2, true))
            }
            NativeInstruction::Or { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Or, rs1, rs2, true))
            }
            NativeInstruction::Ori { rs1, imm, .. } => {
                Some(Self::immediate(ImmediateExpressionOp::Ori, rs1, imm))
            }
            NativeInstruction::Sll { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Sll, rs1, rs2, false))
            }
            NativeInstruction::Slli { rs1, shamt, .. } => Some(Self::immediate(
                ImmediateExpressionOp::Slli,
                rs1,
                i64::from(shamt),
            )),
            NativeInstruction::ShiftedWordOr { .. } => None,
            NativeInstruction::Slliw { rs1, shamt, .. } => Some(Self::immediate(
                ImmediateExpressionOp::Slliw,
                rs1,
                i64::from(shamt),
            )),
            NativeInstruction::Sllw { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Sllw, rs1, rs2, false))
            }
            NativeInstruction::Slt { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Slt, rs1, rs2, false))
            }
            NativeInstruction::Slti { rs1, imm, .. } => {
                Some(Self::immediate(ImmediateExpressionOp::Slti, rs1, imm))
            }
            NativeInstruction::Sltiu { rs1, imm, .. } => {
                Some(Self::immediate(ImmediateExpressionOp::Sltiu, rs1, imm))
            }
            NativeInstruction::Sltu { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Sltu, rs1, rs2, false))
            }
            NativeInstruction::Sra { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Sra, rs1, rs2, false))
            }
            NativeInstruction::Srai { rs1, shamt, .. } => Some(Self::immediate(
                ImmediateExpressionOp::Srai,
                rs1,
                i64::from(shamt),
            )),
            NativeInstruction::Sraiw { rs1, shamt, .. } => Some(Self::immediate(
                ImmediateExpressionOp::Sraiw,
                rs1,
                i64::from(shamt),
            )),
            NativeInstruction::Sraw { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Sraw, rs1, rs2, false))
            }
            NativeInstruction::Srl { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Srl, rs1, rs2, false))
            }
            NativeInstruction::Srli { rs1, shamt, .. } => Some(Self::immediate(
                ImmediateExpressionOp::Srli,
                rs1,
                i64::from(shamt),
            )),
            NativeInstruction::Srliw { rs1, shamt, .. } => Some(Self::immediate(
                ImmediateExpressionOp::Srliw,
                rs1,
                i64::from(shamt),
            )),
            NativeInstruction::Srlw { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Srlw, rs1, rs2, false))
            }
            NativeInstruction::Sub { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Sub, rs1, rs2, false))
            }
            NativeInstruction::Subw { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Subw, rs1, rs2, false))
            }
            NativeInstruction::Xor { rs1, rs2, .. } => {
                Some(Self::binary(BinaryExpressionOp::Xor, rs1, rs2, true))
            }
            NativeInstruction::Xori { rs1, imm, .. } => {
                Some(Self::immediate(ImmediateExpressionOp::Xori, rs1, imm))
            }
            _ => None,
        }
    }

    fn binary(op: BinaryExpressionOp, lhs: u8, rhs: u8, commutative: bool) -> Self {
        let (lhs, rhs) = if commutative && rhs < lhs {
            (rhs, lhs)
        } else {
            (lhs, rhs)
        };
        Self::Binary { op, lhs, rhs }
    }

    fn immediate(op: ImmediateExpressionOp, rs: u8, imm: i64) -> Self {
        Self::Immediate { op, rs, imm }
    }

    fn uses_register(self, register: u8) -> bool {
        match self {
            Self::Binary { lhs, rhs, .. } => lhs == register || rhs == register,
            Self::Immediate { rs, .. } => rs == register,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BinaryExpressionOp {
    Add,
    Addw,
    And,
    Mul,
    Mulh,
    Mulhu,
    Mulw,
    Or,
    Sll,
    Sllw,
    Slt,
    Sltu,
    Sra,
    Sraw,
    Srl,
    Srlw,
    Sub,
    Subw,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ImmediateExpressionOp {
    Addi,
    Addiw,
    Andi,
    Ori,
    Slli,
    Slliw,
    Slti,
    Sltiu,
    Srai,
    Sraiw,
    Srli,
    Srliw,
    Xori,
}

fn eliminate_common_integer_expressions(
    input: &[BlockOperation],
    report: &mut OptimizationReport,
) -> Vec<BlockOperation> {
    let mut available = HashMap::new();
    let mut output = Vec::with_capacity(input.len());

    for operation in input {
        let BlockOperationKind::Native(instruction) = operation.kind();

        if liveness_barrier(instruction) {
            available.clear();
            output.push(*operation);
            continue;
        }

        let Some(rd) = pure_integer_write(instruction) else {
            output.push(*operation);
            continue;
        };
        if rd == Zero as u8 {
            output.push(*operation);
            continue;
        }

        let Some(expression) = CommonExpression::from_instruction(instruction) else {
            invalidate_available_expressions(&mut available, rd);
            output.push(*operation);
            continue;
        };

        if let Some(source) = available.get(&expression).copied() {
            report.reused_expressions += 1;
            if source == rd {
                output.push(BlockOperation {
                    pc: operation.pc(),
                    opcode: operation.opcode(),
                    kind: BlockOperationKind::Native(NativeInstruction::Nop),
                });
            } else {
                invalidate_available_expressions(&mut available, rd);
                report.simplified_ops += 1;
                output.push(BlockOperation {
                    pc: operation.pc(),
                    opcode: operation.opcode(),
                    kind: BlockOperationKind::Native(NativeInstruction::Move { rd, rs: source }),
                });
            }
            continue;
        }

        invalidate_available_expressions(&mut available, rd);
        if !expression.uses_register(rd) {
            available.insert(expression, rd);
        }
        output.push(*operation);
    }

    output
}

fn invalidate_available_expressions(available: &mut HashMap<CommonExpression, u8>, register: u8) {
    if register == Zero as u8 {
        return;
    }

    available
        .retain(|expression, result| *result != register && !expression.uses_register(register));
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
        | NativeInstruction::ShiftedWordOr { rd, .. }
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
            | NativeInstruction::ByteCopy8 { .. }
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
        | NativeInstruction::ShiftedWordOr { rs1, .. }
        | NativeInstruction::Slliw { rs1, .. }
        | NativeInstruction::Slti { rs1, .. }
        | NativeInstruction::Sltiu { rs1, .. }
        | NativeInstruction::Srai { rs1, .. }
        | NativeInstruction::Sraiw { rs1, .. }
        | NativeInstruction::Srli { rs1, .. }
        | NativeInstruction::Srliw { rs1, .. }
        | NativeInstruction::Xori { rs1, .. } => mark(rs1),
        NativeInstruction::ByteCopy8 {
            load_base,
            store_base,
            ..
        } => {
            mark(load_base);
            mark(store_base);
        }
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
    fn folds_shifted_word_or_rotate_idiom() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 10,
                    shamt: 20,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Slli {
                    rd: 11,
                    rs1: 10,
                    shamt: 12,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Or {
                    rd: 12,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Store {
                    rs1: 2,
                    rs2: 12,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Addi {
                    rd: 8,
                    rs1: 3,
                    imm: 1,
                },
            ),
            native(
                0x1014,
                NativeInstruction::Addi {
                    rd: 11,
                    rs1: 4,
                    imm: 1,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.strength_reduced >= 1);
        assert!(matches!(
            optimized.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::ShiftedWordOr {
                rd: 12,
                rs1: 10,
                left_shamt: 12,
                right_shamt: 20,
            })
        ));
        assert!(matches!(
            optimized.operations[1].kind(),
            BlockOperationKind::Native(NativeInstruction::Store { rs2: 12, .. })
        ));
    }

    #[test]
    fn folds_interleaved_shifted_word_or_rotate_idiom() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 10,
                    shamt: 27,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Addi {
                    rd: 13,
                    rs1: 14,
                    imm: 3,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Slli {
                    rd: 11,
                    rs1: 10,
                    shamt: 5,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Xor {
                    rd: 15,
                    rs1: 16,
                    rs2: 17,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Or {
                    rd: 12,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x1014,
                NativeInstruction::Store {
                    rs1: 2,
                    rs2: 12,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
            native(
                0x1018,
                NativeInstruction::Addi {
                    rd: 8,
                    rs1: 3,
                    imm: 1,
                },
            ),
            native(
                0x101c,
                NativeInstruction::Addi {
                    rd: 11,
                    rs1: 4,
                    imm: 1,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.strength_reduced >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::ShiftedWordOr {
                rd: 12,
                rs1: 10,
                left_shamt: 5,
                right_shamt: 27,
            })
        )));
        assert!(optimized.operations.iter().all(|operation| {
            !matches!(
                operation.kind(),
                BlockOperationKind::Native(
                    NativeInstruction::Srliw {
                        rd: 8,
                        rs1: 10,
                        shamt: 27,
                    } | NativeInstruction::Slli {
                        rd: 11,
                        rs1: 10,
                        shamt: 5,
                    }
                )
            )
        }));
    }

    #[test]
    fn folds_shifted_word_or_across_unrelated_load() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 10,
                    shamt: 20,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Slli {
                    rd: 11,
                    rs1: 10,
                    shamt: 12,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Load {
                    rd: 13,
                    rs1: 14,
                    imm: 4,
                    width: super::super::MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Or {
                    rd: 12,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Store {
                    rs1: 2,
                    rs2: 12,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
            native(
                0x1014,
                NativeInstruction::Addi {
                    rd: 8,
                    rs1: 3,
                    imm: 1,
                },
            ),
            native(
                0x1018,
                NativeInstruction::Addi {
                    rd: 11,
                    rs1: 4,
                    imm: 1,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.strength_reduced >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::ShiftedWordOr {
                rd: 12,
                rs1: 10,
                left_shamt: 12,
                right_shamt: 20,
            })
        )));
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 13,
                rs1: 14,
                ..
            })
        )));
    }

    #[test]
    fn folds_in_place_shifted_word_or_across_unrelated_load() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 10,
                    shamt: 20,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Slli {
                    rd: 10,
                    rs1: 10,
                    shamt: 12,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Load {
                    rd: 13,
                    rs1: 14,
                    imm: 4,
                    width: super::super::MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Or {
                    rd: 10,
                    rs1: 10,
                    rs2: 8,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Store {
                    rs1: 2,
                    rs2: 10,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
            native(
                0x1014,
                NativeInstruction::Addi {
                    rd: 8,
                    rs1: 3,
                    imm: 1,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.strength_reduced >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::ShiftedWordOr {
                rd: 10,
                rs1: 10,
                left_shamt: 12,
                right_shamt: 20,
            })
        )));
        assert!(optimized.operations.iter().all(|operation| {
            !matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::Or {
                    rd: 10,
                    rs1: 10,
                    rs2: 8,
                })
            )
        }));
    }

    #[test]
    fn keeps_interleaved_shifted_word_or_when_temp_is_observed_before_or() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 10,
                    shamt: 27,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Addi {
                    rd: 13,
                    rs1: 8,
                    imm: 3,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Slli {
                    rd: 11,
                    rs1: 10,
                    shamt: 5,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Or {
                    rd: 12,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Store {
                    rs1: 2,
                    rs2: 12,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
        ]);

        let (optimized, _) = optimize_plan(&input);

        assert!(optimized.operations.iter().all(|operation| {
            !matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::ShiftedWordOr { .. })
            )
        }));
    }

    #[test]
    fn keeps_interleaved_shifted_word_or_when_source_is_rewritten_before_or() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 10,
                    shamt: 27,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Slli {
                    rd: 11,
                    rs1: 10,
                    shamt: 5,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Addi {
                    rd: 10,
                    rs1: 10,
                    imm: 1,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Or {
                    rd: 12,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Store {
                    rs1: 2,
                    rs2: 12,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
        ]);

        let (optimized, _) = optimize_plan(&input);

        assert!(optimized.operations.iter().all(|operation| {
            !matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::ShiftedWordOr { .. })
            )
        }));
    }

    #[test]
    fn keeps_shifted_word_or_sources_when_temporary_is_still_live() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 10,
                    shamt: 20,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Slli {
                    rd: 11,
                    rs1: 10,
                    shamt: 12,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Or {
                    rd: 12,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Store {
                    rs1: 2,
                    rs2: 8,
                    imm: 0,
                    width: super::super::MemoryWidth::Double,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Addi {
                    rd: 8,
                    rs1: 3,
                    imm: 1,
                },
            ),
            native(
                0x1014,
                NativeInstruction::Addi {
                    rd: 11,
                    rs1: 4,
                    imm: 1,
                },
            ),
        ]);

        let (optimized, _) = optimize_plan(&input);

        assert!(optimized.operations.iter().all(|operation| {
            !matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::ShiftedWordOr { .. })
            )
        }));
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Srliw {
                rd: 8,
                rs1: 10,
                shamt: 20,
            })
        )));
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
    fn reuses_common_integer_expressions_with_commuted_operands() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Add {
                    rd: 5,
                    rs1: 1,
                    rs2: 2,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Add {
                    rd: 6,
                    rs1: 2,
                    rs2: 1,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.reused_expressions >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Move { rd: 6, rs: 5 })
        )));
    }

    #[test]
    fn common_expression_reuse_invalidates_overwritten_inputs() {
        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Add {
                    rd: 5,
                    rs1: 1,
                    rs2: 2,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Addi {
                    rd: 1,
                    rs1: 1,
                    imm: 1,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Add {
                    rd: 6,
                    rs1: 2,
                    rs2: 1,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert_eq!(report.reused_expressions, 0);
        assert!(matches!(
            optimized.operations[2].kind(),
            BlockOperationKind::Native(NativeInstruction::Add {
                rd: 6,
                rs1: 2,
                rs2: 1,
            })
        ));
    }

    #[test]
    fn strength_reduces_self_add_to_shift() {
        let input = plan(vec![native(
            0x1000,
            NativeInstruction::Add {
                rd: 5,
                rs1: 6,
                rs2: 6,
            },
        )]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.strength_reduced >= 1);
        assert!(matches!(
            optimized.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::Slli {
                rd: 5,
                rs1: 6,
                shamt: 1,
            })
        ));
    }

    #[test]
    fn strength_reduces_multiply_by_constant_power_of_two() {
        let input = plan(vec![
            native(0x1000, NativeInstruction::LoadImmediate { rd: 5, value: 8 }),
            native(
                0x1004,
                NativeInstruction::Mul {
                    rd: 6,
                    rs1: 7,
                    rs2: 5,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.strength_reduced >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Slli {
                rd: 6,
                rs1: 7,
                shamt: 3,
            })
        )));
    }

    #[test]
    fn strength_reduces_word_multiply_by_one_to_word_sign_extend() {
        let input = plan(vec![
            native(0x1000, NativeInstruction::LoadImmediate { rd: 5, value: 1 }),
            native(
                0x1004,
                NativeInstruction::Mulw {
                    rd: 6,
                    rs1: 7,
                    rs2: 5,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.strength_reduced >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Addiw {
                rd: 6,
                rs1: 7,
                imm: 0,
            })
        )));
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Move { rd: 6, rs: 7 })
        )));
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

    fn unsigned_byte_pack_operations() -> Vec<BlockOperation> {
        use super::super::MemoryWidth;

        vec![
            native(
                0x1000,
                NativeInstruction::Load {
                    rd: 5,
                    rs1: 10,
                    imm: 0,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Load {
                    rd: 6,
                    rs1: 10,
                    imm: 1,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Load {
                    rd: 7,
                    rs1: 10,
                    imm: 2,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Load {
                    rd: 8,
                    rs1: 10,
                    imm: 3,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Load {
                    rd: 9,
                    rs1: 10,
                    imm: 4,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1014,
                NativeInstruction::Load {
                    rd: 11,
                    rs1: 10,
                    imm: 5,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1018,
                NativeInstruction::Load {
                    rd: 12,
                    rs1: 10,
                    imm: 6,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x101c,
                NativeInstruction::Load {
                    rd: 13,
                    rs1: 10,
                    imm: 7,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1020,
                NativeInstruction::Slli {
                    rd: 6,
                    rs1: 6,
                    shamt: 8,
                },
            ),
            native(
                0x1024,
                NativeInstruction::Slli {
                    rd: 7,
                    rs1: 7,
                    shamt: 16,
                },
            ),
            native(
                0x1028,
                NativeInstruction::Slli {
                    rd: 8,
                    rs1: 8,
                    shamt: 24,
                },
            ),
            native(
                0x102c,
                NativeInstruction::Slli {
                    rd: 11,
                    rs1: 11,
                    shamt: 8,
                },
            ),
            native(
                0x1030,
                NativeInstruction::Slli {
                    rd: 12,
                    rs1: 12,
                    shamt: 16,
                },
            ),
            native(
                0x1034,
                NativeInstruction::Slli {
                    rd: 13,
                    rs1: 13,
                    shamt: 24,
                },
            ),
            native(
                0x1038,
                NativeInstruction::Or {
                    rd: 5,
                    rs1: 5,
                    rs2: 6,
                },
            ),
            native(
                0x103c,
                NativeInstruction::Or {
                    rd: 7,
                    rs1: 7,
                    rs2: 8,
                },
            ),
            native(
                0x1040,
                NativeInstruction::Or {
                    rd: 9,
                    rs1: 9,
                    rs2: 11,
                },
            ),
            native(
                0x1044,
                NativeInstruction::Or {
                    rd: 12,
                    rs1: 12,
                    rs2: 13,
                },
            ),
            native(
                0x1048,
                NativeInstruction::Or {
                    rd: 5,
                    rs1: 5,
                    rs2: 7,
                },
            ),
            native(
                0x104c,
                NativeInstruction::Or {
                    rd: 9,
                    rs1: 9,
                    rs2: 12,
                },
            ),
            native(
                0x1050,
                NativeInstruction::Slli {
                    rd: 9,
                    rs1: 9,
                    shamt: 32,
                },
            ),
            native(
                0x1054,
                NativeInstruction::Or {
                    rd: 17,
                    rs1: 9,
                    rs2: 5,
                },
            ),
        ]
    }

    #[test]
    fn folds_unsigned_byte_pack_to_doubleword_load_when_temps_die() {
        let mut operations = unsigned_byte_pack_operations();
        for register in [5, 6, 7, 8, 9, 11, 12, 13] {
            operations.push(native(
                0x1100 + u64::from(register) * 4,
                NativeInstruction::LoadImmediate {
                    rd: register,
                    value: 0,
                },
            ));
        }
        operations.push(native(
            0x1200,
            NativeInstruction::Bne {
                rs1: 17,
                rs2: 0,
                target: 0x1300,
                fallthrough: 0x1204,
            },
        ));
        let input = plan(operations);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.simplified_ops >= 1);
        assert!(matches!(
            optimized.operations[0].kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 17,
                rs1: 10,
                imm: 0,
                width: super::super::MemoryWidth::Double,
                signed: false,
            })
        ));
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                width: super::super::MemoryWidth::Byte,
                signed: false,
                ..
            })
        )));
    }

    #[test]
    fn keeps_unsigned_byte_pack_when_temporary_is_read_later() {
        let mut operations = unsigned_byte_pack_operations();
        operations.push(native(
            0x1100,
            NativeInstruction::Add {
                rd: 18,
                rs1: 5,
                rs2: 17,
            },
        ));
        let input = plan(operations);

        let (optimized, _) = optimize_plan(&input);

        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                width: super::super::MemoryWidth::Byte,
                signed: false,
                ..
            })
        )));
    }

    #[test]
    fn keeps_unsigned_byte_pack_when_temps_die_only_after_control_flow() {
        use super::super::MemoryWidth;

        let mut operations = unsigned_byte_pack_operations();
        operations.push(native(
            0x1100,
            NativeInstruction::Bne {
                rs1: 17,
                rs2: 0,
                target: 0x1200,
                fallthrough: 0x1104,
            },
        ));
        for register in [5, 6, 7, 8, 9, 11, 12, 13] {
            operations.push(native(
                0x1200 + u64::from(register) * 4,
                NativeInstruction::LoadImmediate {
                    rd: register,
                    value: 0,
                },
            ));
        }

        let (optimized, _) = optimize_plan(&plan(operations));

        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                width: MemoryWidth::Byte,
                signed: false,
                ..
            })
        )));
    }

    #[test]
    fn folds_signed_word_byte_pack_to_word_load_when_temps_die() {
        use super::super::MemoryWidth;

        let input = plan(vec![
            native(
                0x1000,
                NativeInstruction::Load {
                    rd: 5,
                    rs1: 10,
                    imm: 0,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1004,
                NativeInstruction::Load {
                    rd: 6,
                    rs1: 10,
                    imm: 1,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1008,
                NativeInstruction::Load {
                    rd: 7,
                    rs1: 10,
                    imm: 2,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x100c,
                NativeInstruction::Load {
                    rd: 8,
                    rs1: 10,
                    imm: 3,
                    width: MemoryWidth::Byte,
                    signed: true,
                },
            ),
            native(
                0x1010,
                NativeInstruction::Slli {
                    rd: 6,
                    rs1: 6,
                    shamt: 8,
                },
            ),
            native(
                0x1014,
                NativeInstruction::Slli {
                    rd: 7,
                    rs1: 7,
                    shamt: 16,
                },
            ),
            native(
                0x1018,
                NativeInstruction::Slli {
                    rd: 8,
                    rs1: 8,
                    shamt: 24,
                },
            ),
            native(
                0x101c,
                NativeInstruction::Or {
                    rd: 5,
                    rs1: 5,
                    rs2: 6,
                },
            ),
            native(
                0x1020,
                NativeInstruction::Or {
                    rd: 7,
                    rs1: 7,
                    rs2: 8,
                },
            ),
            native(
                0x1024,
                NativeInstruction::Or {
                    rd: 17,
                    rs1: 5,
                    rs2: 7,
                },
            ),
            native(0x1028, NativeInstruction::LoadImmediate { rd: 5, value: 0 }),
            native(0x102c, NativeInstruction::LoadImmediate { rd: 6, value: 0 }),
            native(0x1030, NativeInstruction::LoadImmediate { rd: 7, value: 0 }),
            native(0x1034, NativeInstruction::LoadImmediate { rd: 8, value: 0 }),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.simplified_ops >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 17,
                rs1: 10,
                imm: 0,
                width: MemoryWidth::Word,
                signed: true,
            })
        )));
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                width: MemoryWidth::Byte,
                ..
            })
        )));
    }

    #[test]
    fn folds_md5_style_interleaved_signed_byte_pack() {
        use super::super::MemoryWidth;

        let input = plan(vec![
            native(
                0x1585d16,
                NativeInstruction::Load {
                    rd: 8,
                    rs1: 11,
                    imm: 57,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d1a,
                NativeInstruction::Load {
                    rd: 25,
                    rs1: 11,
                    imm: 58,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d1e,
                NativeInstruction::Add {
                    rd: 13,
                    rs1: 13,
                    rs2: 9,
                },
            ),
            native(
                0x1585d20,
                NativeInstruction::Load {
                    rd: 9,
                    rs1: 11,
                    imm: 59,
                    width: MemoryWidth::Byte,
                    signed: true,
                },
            ),
            native(
                0x1585d24,
                NativeInstruction::Slli {
                    rd: 1,
                    rs1: 8,
                    shamt: 8,
                },
            ),
            native(
                0x1585d28,
                NativeInstruction::Slli {
                    rd: 25,
                    rs1: 25,
                    shamt: 16,
                },
            ),
            native(
                0x1585d2a,
                NativeInstruction::Srliw {
                    rd: 8,
                    rs1: 13,
                    shamt: 20,
                },
            ),
            native(
                0x1585d2e,
                NativeInstruction::Slli {
                    rd: 13,
                    rs1: 13,
                    shamt: 12,
                },
            ),
            native(
                0x1585d30,
                NativeInstruction::Slli {
                    rd: 9,
                    rs1: 9,
                    shamt: 24,
                },
            ),
            native(
                0x1585d32,
                NativeInstruction::Load {
                    rd: 10,
                    rs1: 11,
                    imm: 56,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d36,
                NativeInstruction::Or {
                    rd: 13,
                    rs1: 13,
                    rs2: 8,
                },
            ),
            native(
                0x1585d38,
                NativeInstruction::Or {
                    rd: 10,
                    rs1: 1,
                    rs2: 10,
                },
            ),
            native(
                0x1585d3c,
                NativeInstruction::Or {
                    rd: 8,
                    rs1: 25,
                    rs2: 9,
                },
            ),
            native(
                0x1585d40,
                NativeInstruction::Add {
                    rd: 13,
                    rs1: 13,
                    rs2: 12,
                },
            ),
            native(
                0x1585d42,
                NativeInstruction::Xor {
                    rd: 9,
                    rs1: 12,
                    rs2: 15,
                },
            ),
            native(
                0x1585d46,
                NativeInstruction::Or {
                    rd: 25,
                    rs1: 10,
                    rs2: 8,
                },
            ),
            native(
                0x1585d4a,
                NativeInstruction::Lui {
                    rd: 10,
                    value: 0xa6794,
                },
            ),
            native(
                0x1585d4e,
                NativeInstruction::And {
                    rd: 9,
                    rs1: 9,
                    rs2: 13,
                },
            ),
            native(
                0x1585d5e,
                NativeInstruction::LoadImmediate { rd: 8, value: 0 },
            ),
            native(
                0x1585d68,
                NativeInstruction::LoadImmediate { rd: 1, value: 0 },
            ),
            native(
                0x1585d6c,
                NativeInstruction::Bne {
                    rs1: 25,
                    rs2: 0,
                    target: 0x1585d80,
                    fallthrough: 0x1585d70,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.simplified_ops >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 25,
                rs1: 11,
                imm: 56,
                width: MemoryWidth::Word,
                signed: true,
            })
        )));
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                width: MemoryWidth::Byte,
                ..
            })
        )));
    }

    #[test]
    fn folds_byte_pack_when_final_value_clobbers_base_register() {
        use super::super::MemoryWidth;

        let input = plan(vec![
            native(
                0x1585d4e,
                NativeInstruction::Load {
                    rd: 14,
                    rs1: 11,
                    imm: 61,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d52,
                NativeInstruction::Load {
                    rd: 8,
                    rs1: 11,
                    imm: 62,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d54,
                NativeInstruction::Add {
                    rd: 10,
                    rs1: 10,
                    rs2: 9,
                },
            ),
            native(
                0x1585d56,
                NativeInstruction::Load {
                    rd: 9,
                    rs1: 11,
                    imm: 63,
                    width: MemoryWidth::Byte,
                    signed: true,
                },
            ),
            native(
                0x1585d5a,
                NativeInstruction::Load {
                    rd: 1,
                    rs1: 11,
                    imm: 60,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d5e,
                NativeInstruction::Slli {
                    rd: 14,
                    rs1: 14,
                    shamt: 8,
                },
            ),
            native(
                0x1585d62,
                NativeInstruction::Slli {
                    rd: 8,
                    rs1: 8,
                    shamt: 16,
                },
            ),
            native(
                0x1585d64,
                NativeInstruction::Slli {
                    rd: 9,
                    rs1: 9,
                    shamt: 24,
                },
            ),
            native(
                0x1585d66,
                NativeInstruction::Or {
                    rd: 11,
                    rs1: 14,
                    rs2: 1,
                },
            ),
            native(
                0x1585d6a,
                NativeInstruction::Or {
                    rd: 8,
                    rs1: 8,
                    rs2: 9,
                },
            ),
            native(
                0x1585d6e,
                NativeInstruction::Add {
                    rd: 10,
                    rs1: 10,
                    rs2: 13,
                },
            ),
            native(
                0x1585d70,
                NativeInstruction::Xor {
                    rd: 14,
                    rs1: 13,
                    rs2: 12,
                },
            ),
            native(
                0x1585d72,
                NativeInstruction::Or {
                    rd: 11,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x1585d76,
                NativeInstruction::LoadImmediate { rd: 1, value: 0 },
            ),
            native(
                0x1585d7a,
                NativeInstruction::LoadImmediate { rd: 8, value: 0 },
            ),
            native(
                0x1585d7e,
                NativeInstruction::LoadImmediate { rd: 9, value: 0 },
            ),
            native(
                0x1585d82,
                NativeInstruction::Bne {
                    rs1: 11,
                    rs2: 0,
                    target: 0x1585d90,
                    fallthrough: 0x1585d86,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.simplified_ops >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 11,
                rs1: 11,
                imm: 60,
                width: MemoryWidth::Word,
                signed: true,
            })
        )));
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                width: MemoryWidth::Byte,
                ..
            })
        )));
    }

    #[test]
    fn folds_byte_pack_while_preserving_live_byte_temps() {
        use super::super::MemoryWidth;

        let input = plan(vec![
            native(
                0x1585d5a,
                NativeInstruction::Load {
                    rd: 14,
                    rs1: 11,
                    imm: 61,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d5e,
                NativeInstruction::Load {
                    rd: 8,
                    rs1: 11,
                    imm: 62,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d62,
                NativeInstruction::Add {
                    rd: 10,
                    rs1: 10,
                    rs2: 9,
                },
            ),
            native(
                0x1585d64,
                NativeInstruction::Load {
                    rd: 9,
                    rs1: 11,
                    imm: 63,
                    width: MemoryWidth::Byte,
                    signed: true,
                },
            ),
            native(
                0x1585d68,
                NativeInstruction::Load {
                    rd: 1,
                    rs1: 11,
                    imm: 60,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
            native(
                0x1585d6c,
                NativeInstruction::Slli {
                    rd: 14,
                    rs1: 14,
                    shamt: 8,
                },
            ),
            native(
                0x1585d72,
                NativeInstruction::ShiftedWordOr {
                    rd: 10,
                    rs1: 10,
                    left_shamt: 17,
                    right_shamt: 15,
                },
            ),
            native(
                0x1585d74,
                NativeInstruction::Slli {
                    rd: 8,
                    rs1: 8,
                    shamt: 16,
                },
            ),
            native(
                0x1585d76,
                NativeInstruction::Slli {
                    rd: 9,
                    rs1: 9,
                    shamt: 24,
                },
            ),
            native(
                0x1585d7a,
                NativeInstruction::Or {
                    rd: 11,
                    rs1: 14,
                    rs2: 1,
                },
            ),
            native(
                0x1585d7e,
                NativeInstruction::Or {
                    rd: 8,
                    rs1: 8,
                    rs2: 9,
                },
            ),
            native(
                0x1585d80,
                NativeInstruction::Add {
                    rd: 10,
                    rs1: 10,
                    rs2: 13,
                },
            ),
            native(
                0x1585d82,
                NativeInstruction::Xor {
                    rd: 14,
                    rs1: 13,
                    rs2: 12,
                },
            ),
            native(
                0x1585d86,
                NativeInstruction::Or {
                    rd: 11,
                    rs1: 11,
                    rs2: 8,
                },
            ),
            native(
                0x1585d90,
                NativeInstruction::LoadImmediate { rd: 8, value: 0 },
            ),
            native(
                0x1585d98,
                NativeInstruction::Bne {
                    rs1: 11,
                    rs2: 0,
                    target: 0x1585da0,
                    fallthrough: 0x1585d9c,
                },
            ),
        ]);

        let (optimized, report) = optimize_plan(&input);

        assert!(report.simplified_ops >= 1);
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 11,
                rs1: 11,
                imm: 60,
                width: MemoryWidth::Word,
                signed: true,
            })
        )));
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                rd: 1,
                rs1: 11,
                imm: 60,
                width: MemoryWidth::Byte,
                signed: false,
            })
        )));
        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Slli {
                rd: 9,
                rs1: 9,
                shamt: 24,
            })
        )));
        assert_eq!(
            optimized
                .operations
                .iter()
                .filter(|operation| matches!(
                    operation.kind(),
                    BlockOperationKind::Native(NativeInstruction::Load {
                        width: MemoryWidth::Byte,
                        ..
                    })
                ))
                .count(),
            2
        );
    }

    #[test]
    fn folds_byte_pack_when_consumed_temp_is_overwritten_before_final_value() {
        use super::super::MemoryWidth;

        let mut operations = unsigned_byte_pack_operations();
        let insert_at = operations
            .iter()
            .position(|operation| {
                matches!(
                    operation.kind(),
                    BlockOperationKind::Native(NativeInstruction::Or {
                        rd: 5,
                        rs1: 5,
                        rs2: 7,
                    })
                )
            })
            .unwrap()
            + 1;
        operations.insert(
            insert_at,
            native(
                0x104a,
                NativeInstruction::Load {
                    rd: 8,
                    rs1: 20,
                    imm: 0,
                    width: MemoryWidth::Byte,
                    signed: false,
                },
            ),
        );
        for register in [5, 6, 7, 9, 11, 12, 13] {
            operations.push(native(
                0x1100 + u64::from(register) * 4,
                NativeInstruction::LoadImmediate {
                    rd: register,
                    value: 0,
                },
            ));
        }
        operations.push(native(
            0x1200,
            NativeInstruction::Bne {
                rs1: 17,
                rs2: 0,
                target: 0x1300,
                fallthrough: 0x1204,
            },
        ));

        let (optimized, report) = optimize_plan(&plan(operations));

        assert!(report.simplified_ops >= 1);
        assert!(optimized.operations.iter().any(|operation| {
            matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::Load {
                    rd: 17,
                    rs1: 10,
                    imm: 0,
                    width: MemoryWidth::Double,
                    signed: false,
                })
            )
        }));
    }

    #[test]
    fn folds_dead_byte_copy_run_to_doubleword_copy() {
        use super::super::MemoryWidth;

        let mut pc = 0x1000;
        let mut operations = Vec::new();
        let mut push = |instruction| {
            operations.push(native(pc, instruction));
            pc += 4;
        };

        for (rd, imm) in [
            (6, 1),
            (5, 0),
            (8, 3),
            (7, 2),
            (11, 5),
            (12, 6),
            (13, 7),
            (9, 4),
        ] {
            push(NativeInstruction::Load {
                rd,
                rs1: 10,
                imm,
                width: MemoryWidth::Byte,
                signed: false,
            });
        }
        for (rs2, imm) in [(12, 6), (13, 7), (9, 4), (11, 5), (7, 2)] {
            push(NativeInstruction::Store {
                rs1: 20,
                rs2,
                imm,
                width: MemoryWidth::Byte,
            });
        }
        push(NativeInstruction::Addi {
            rd: 7,
            rs1: 20,
            imm: 8,
        });
        for (rs2, imm) in [(8, 3), (5, 0), (6, 1)] {
            push(NativeInstruction::Store {
                rs1: 20,
                rs2,
                imm,
                width: MemoryWidth::Byte,
            });
        }
        for register in [5, 6, 8, 9, 11, 12, 13] {
            push(NativeInstruction::LoadImmediate {
                rd: register,
                value: 0,
            });
        }

        let (optimized, report) = optimize_plan(&plan(operations));

        assert!(report.simplified_ops >= 14);
        assert!(optimized.operations.iter().any(|operation| {
            matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::Load {
                    rd: 5,
                    rs1: 10,
                    imm: 0,
                    width: MemoryWidth::Double,
                    signed: false,
                })
            )
        }));
        assert!(optimized.operations.iter().any(|operation| {
            matches!(
                operation.kind(),
                BlockOperationKind::Native(NativeInstruction::Store {
                    rs1: 20,
                    rs2: 5,
                    imm: 0,
                    width: MemoryWidth::Double,
                })
            )
        }));
        let double_load = optimized
            .operations
            .iter()
            .position(|operation| {
                matches!(
                    operation.kind(),
                    BlockOperationKind::Native(NativeInstruction::Load {
                        width: MemoryWidth::Double,
                        signed: false,
                        ..
                    })
                )
            })
            .unwrap();
        let double_store = optimized
            .operations
            .iter()
            .position(|operation| {
                matches!(
                    operation.kind(),
                    BlockOperationKind::Native(NativeInstruction::Store {
                        width: MemoryWidth::Double,
                        ..
                    })
                )
            })
            .unwrap();
        assert!(
            double_load < double_store,
            "folded byte copy must load the full source before storing it"
        );
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Store {
                width: MemoryWidth::Byte,
                ..
            })
        )));
    }

    #[test]
    fn folds_byte_copy_with_live_temps_to_preserving_copy() {
        use super::super::MemoryWidth;

        let mut pc = 0x1000;
        let mut operations = Vec::new();
        let mut push = |instruction| {
            operations.push(native(pc, instruction));
            pc += 4;
        };

        for (rd, imm) in [
            (6, 1),
            (5, 0),
            (8, 3),
            (7, 2),
            (11, 5),
            (12, 6),
            (13, 7),
            (9, 4),
        ] {
            push(NativeInstruction::Load {
                rd,
                rs1: 10,
                imm,
                width: MemoryWidth::Byte,
                signed: false,
            });
        }
        for (rs2, imm) in [
            (12, 6),
            (13, 7),
            (9, 4),
            (11, 5),
            (7, 2),
            (8, 3),
            (5, 0),
            (6, 1),
        ] {
            push(NativeInstruction::Store {
                rs1: 20,
                rs2,
                imm,
                width: MemoryWidth::Byte,
            });
        }
        push(NativeInstruction::Bne {
            rs1: 1,
            rs2: 2,
            target: 0x2000,
            fallthrough: 0x2004,
        });
        for register in [5, 6, 7, 8, 9, 11, 12, 13] {
            push(NativeInstruction::LoadImmediate {
                rd: register,
                value: 0,
            });
        }

        let (optimized, _) = optimize_plan(&plan(operations));

        assert!(optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::ByteCopy8 {
                registers: [5, 6, 7, 8, 9, 11, 12, 13],
                load_base: 10,
                load_imm: 0,
                store_base: 20,
                store_imm: 0,
            })
        )));
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Store {
                width: MemoryWidth::Byte,
                ..
            })
        )));
    }

    #[test]
    fn folds_interleaved_unsigned_byte_packs() {
        use super::super::MemoryWidth;

        let left = [5, 6, 7, 8, 9, 11, 12, 13];
        let right = [14, 15, 16, 18, 19, 21, 22, 23];
        let mut pc = 0x1000;
        let mut operations = Vec::new();
        let mut push = |instruction| {
            operations.push(native(pc, instruction));
            pc += 4;
        };

        for byte in 0..8 {
            push(NativeInstruction::Load {
                rd: left[byte],
                rs1: 10,
                imm: byte as i64,
                width: MemoryWidth::Byte,
                signed: false,
            });
            push(NativeInstruction::Load {
                rd: right[byte],
                rs1: 20,
                imm: byte as i64,
                width: MemoryWidth::Byte,
                signed: false,
            });
        }
        for byte in 1..8 {
            push(NativeInstruction::Slli {
                rd: left[byte],
                rs1: left[byte],
                shamt: (byte * 8) as u32,
            });
            push(NativeInstruction::Slli {
                rd: right[byte],
                rs1: right[byte],
                shamt: (byte * 8) as u32,
            });
        }
        for byte in 1..8 {
            push(NativeInstruction::Or {
                rd: left[0],
                rs1: left[0],
                rs2: left[byte],
            });
            push(NativeInstruction::Or {
                rd: right[0],
                rs1: right[0],
                rs2: right[byte],
            });
        }
        push(NativeInstruction::Move {
            rd: 17,
            rs: left[0],
        });
        push(NativeInstruction::Move {
            rd: 24,
            rs: right[0],
        });
        for register in left.into_iter().chain(right) {
            push(NativeInstruction::LoadImmediate {
                rd: register,
                value: 0,
            });
        }
        push(NativeInstruction::Bne {
            rs1: 17,
            rs2: 24,
            target: 0x2000,
            fallthrough: 0x2004,
        });

        let (optimized, report) = optimize_plan(&plan(operations));

        assert!(report.simplified_ops >= 2);
        let folded_loads = optimized
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation.kind(),
                    BlockOperationKind::Native(NativeInstruction::Load {
                        width: MemoryWidth::Double,
                        signed: false,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(folded_loads, 2);
        assert!(!optimized.operations.iter().any(|operation| matches!(
            operation.kind(),
            BlockOperationKind::Native(NativeInstruction::Load {
                width: MemoryWidth::Byte,
                signed: false,
                ..
            })
        )));
    }
}
