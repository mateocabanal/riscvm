use crate::cpu::RV64GC;
use crate::sign_extend;
use crate::tracer::InstructionTrace;

use super::{
    lower_native_at, BlockOperation, BlockOperationKind, BlockPlan, BlockPlanResult, BlockStop,
    IntegerBranchCondition, MemoryWidth, NativeInstruction,
};

const MAX_TRACE_INSTRUCTIONS: usize = 128;

pub(super) fn trace_from_cpu(cpu: &RV64GC, start_pc: u64) -> BlockPlanResult {
    let mut state = TraceRegisterState::from_cpu(cpu);
    let mut pc = start_pc;
    let mut operations = Vec::new();
    let mut fingerprint = Vec::new();
    let mut seen_pcs = Vec::new();
    let mut stop = BlockStop::MaxInstructions;
    let mut has_side_exit_guard = false;

    for _ in 0..MAX_TRACE_INSTRUCTIONS {
        if seen_pcs.contains(&pc) {
            stop = BlockStop::ControlFlow { pc };
            break;
        }
        seen_pcs.push(pc);

        let Some(lowered) = lower_native_at(cpu, pc) else {
            if operations.is_empty() {
                return BlockPlanResult {
                    plan: None,
                    stop: BlockStop::FetchFault { pc },
                };
            }
            stop = BlockStop::MaxInstructions;
            break;
        };

        fingerprint.push((lowered.pc, lowered.opcode));

        if let Some(branch) = IntegerBranch::from_instruction(lowered.instruction) {
            let taken = branch.evaluate(&state);
            let continue_pc = if taken {
                branch.target
            } else {
                branch.fallthrough
            };
            let side_exit_pc = if taken {
                branch.fallthrough
            } else {
                branch.target
            };
            let executed_instructions = operations.len() as u64 + 1;

            if continue_pc == start_pc && !operations.is_empty() {
                push_operation(
                    &mut operations,
                    lowered.pc,
                    lowered.opcode,
                    NativeInstruction::TraceLoopGuard {
                        rs1: branch.rs1,
                        rs2: branch.rs2,
                        condition: branch.condition,
                        continue_on_taken: taken,
                        loop_pc: continue_pc,
                        side_exit_pc,
                        guest_instruction_count: executed_instructions,
                    },
                );
                pc = continue_pc;
                stop = BlockStop::ControlFlow { pc: lowered.pc };
                break;
            }

            push_operation(
                &mut operations,
                lowered.pc,
                lowered.opcode,
                NativeInstruction::TraceGuard {
                    rs1: branch.rs1,
                    rs2: branch.rs2,
                    condition: branch.condition,
                    continue_on_taken: taken,
                    continue_pc,
                    side_exit_pc,
                    executed_instructions,
                },
            );
            has_side_exit_guard = true;
            pc = continue_pc;
            continue;
        }

        match direct_jump_target(lowered.instruction) {
            Some(target) if target == start_pc && !operations.is_empty() => {
                push_operation(
                    &mut operations,
                    lowered.pc,
                    lowered.opcode,
                    lowered.instruction,
                );
                pc = target;
                stop = BlockStop::ControlFlow { pc: lowered.pc };
                break;
            }
            Some(target) => {
                push_operation(
                    &mut operations,
                    lowered.pc,
                    lowered.opcode,
                    NativeInstruction::InlinedJump { target },
                );
                pc = target;
                continue;
            }
            None => {}
        }

        let terminates = lowered.instruction.terminates_block();
        push_operation(
            &mut operations,
            lowered.pc,
            lowered.opcode,
            lowered.instruction,
        );

        if terminates {
            stop = BlockStop::ControlFlow { pc: lowered.pc };
            pc = lowered.next_pc;
            break;
        }

        pc = lowered.next_pc;
        if !state.apply(cpu, lowered.instruction) {
            stop = BlockStop::MaxInstructions;
            break;
        }
    }

    if operations.is_empty() || !has_side_exit_guard {
        return BlockPlanResult { plan: None, stop };
    }

    let profile_instructions: Vec<_> = operations
        .iter()
        .map(|operation| InstructionTrace {
            pc: operation.pc(),
            opcode: operation.opcode(),
            text: operation.to_string(),
        })
        .collect();

    BlockPlanResult {
        plan: Some(BlockPlan {
            start_pc,
            end_pc: pc,
            operations,
            fingerprint,
            code_version: cpu.ram.code_version(),
            stop,
            guest_instruction_count: profile_instructions_len(&profile_instructions),
            profile_instructions,
        }),
        stop,
    }
}

fn profile_instructions_len(instructions: &[InstructionTrace]) -> usize {
    instructions.len()
}

fn push_operation(
    operations: &mut Vec<BlockOperation>,
    pc: u64,
    opcode: u32,
    instruction: NativeInstruction,
) {
    operations.push(BlockOperation {
        pc,
        opcode,
        kind: BlockOperationKind::Native(instruction),
    });
}

fn direct_jump_target(instruction: NativeInstruction) -> Option<u64> {
    match instruction {
        NativeInstruction::Jal { rd: 0, target, .. } | NativeInstruction::Jump { target } => {
            Some(target)
        }
        _ => None,
    }
}

#[derive(Clone)]
struct TraceRegisterState {
    registers: [u64; 32],
    memory_overlay: Vec<(u64, u8)>,
}

impl TraceRegisterState {
    fn from_cpu(cpu: &RV64GC) -> Self {
        let mut registers = [0; 32];
        for (index, register) in registers.iter_mut().enumerate() {
            *register = cpu.registers[index];
        }
        registers[0] = 0;
        Self {
            registers,
            memory_overlay: Vec::new(),
        }
    }

    fn read(&self, register: u8) -> u64 {
        if register == 0 {
            0
        } else {
            self.registers[register as usize]
        }
    }

    fn write(&mut self, register: u8, value: u64) {
        if register != 0 {
            self.registers[register as usize] = value;
        }
    }

    fn read_memory(&self, cpu: &RV64GC, address: u64, width: MemoryWidth) -> Option<u64> {
        let mut value = 0;
        for offset in 0..width.bytes() {
            let byte = self.read_memory_byte(cpu, address.wrapping_add(offset))?;
            value |= u64::from(byte) << (offset * 8);
        }
        Some(value)
    }

    fn read_memory_byte(&self, cpu: &RV64GC, address: u64) -> Option<u8> {
        self.memory_overlay
            .iter()
            .rev()
            .find_map(|(stored_address, value)| (*stored_address == address).then_some(*value))
            .or_else(|| cpu.ram.read_byte(address).ok())
    }

    fn write_memory(&mut self, address: u64, value: u64, width: MemoryWidth) {
        for offset in 0..width.bytes() {
            let byte_address = address.wrapping_add(offset);
            let byte = ((value >> (offset * 8)) & 0xff) as u8;
            if let Some((_, existing)) = self
                .memory_overlay
                .iter_mut()
                .rev()
                .find(|(stored_address, _)| *stored_address == byte_address)
            {
                *existing = byte;
            } else {
                self.memory_overlay.push((byte_address, byte));
            }
        }
    }

    fn apply(&mut self, cpu: &RV64GC, instruction: NativeInstruction) -> bool {
        match instruction {
            NativeInstruction::Add { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1).wrapping_add(self.read(rs2)));
            }
            NativeInstruction::Addi { rd, rs1, imm } => {
                self.write(rd, self.read(rs1).wrapping_add_signed(imm));
            }
            NativeInstruction::Addiw { rd, rs1, imm } => {
                let value = (self.read(rs1) as u32).wrapping_add(imm as u32);
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::Addw { rd, rs1, rs2 } => {
                let value = (self.read(rs1) as u32).wrapping_add(self.read(rs2) as u32);
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::And { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1) & self.read(rs2));
            }
            NativeInstruction::Andi { rd, rs1, imm } => {
                self.write(rd, self.read(rs1) & imm as u64);
            }
            NativeInstruction::Auipc { rd, value }
            | NativeInstruction::LoadImmediate { rd, value }
            | NativeInstruction::Lui { rd, value } => {
                self.write(rd, value);
            }
            NativeInstruction::Load {
                rd,
                rs1,
                imm,
                width,
                signed,
            } => {
                let address = self.read(rs1).wrapping_add_signed(imm);
                let Some(raw) = self.read_memory(cpu, address, width) else {
                    return false;
                };
                self.write(rd, extend_load(raw, width, signed));
            }
            NativeInstruction::Move { rd, rs } => {
                self.write(rd, self.read(rs));
            }
            NativeInstruction::Mul { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1).wrapping_mul(self.read(rs2)));
            }
            NativeInstruction::Mulh { rd, rs1, rs2 } => {
                let lhs = self.read(rs1) as i64 as i128;
                let rhs = self.read(rs2) as i64 as i128;
                self.write(rd, ((lhs * rhs) >> 64) as u64);
            }
            NativeInstruction::Mulhu { rd, rs1, rs2 } => {
                let lhs = self.read(rs1) as u128;
                let rhs = self.read(rs2) as u128;
                self.write(rd, ((lhs * rhs) >> 64) as u64);
            }
            NativeInstruction::Mulw { rd, rs1, rs2 } => {
                let value = (self.read(rs1) as u32).wrapping_mul(self.read(rs2) as u32);
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::Or { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1) | self.read(rs2));
            }
            NativeInstruction::Ori { rd, rs1, imm } => {
                self.write(rd, self.read(rs1) | imm as u64);
            }
            NativeInstruction::Sll { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1) << (self.read(rs2) & 0x3f));
            }
            NativeInstruction::Slli { rd, rs1, shamt } => {
                self.write(rd, self.read(rs1) << shamt);
            }
            NativeInstruction::Slliw { rd, rs1, shamt } => {
                let value = (self.read(rs1) as u32).wrapping_shl(shamt);
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::Sllw { rd, rs1, rs2 } => {
                let value = (self.read(rs1) as u32).wrapping_shl((self.read(rs2) & 0x1f) as u32);
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::Slt { rd, rs1, rs2 } => {
                self.write(
                    rd,
                    u64::from((self.read(rs1) as i64) < (self.read(rs2) as i64)),
                );
            }
            NativeInstruction::Slti { rd, rs1, imm } => {
                self.write(rd, u64::from((self.read(rs1) as i64) < imm));
            }
            NativeInstruction::Sltiu { rd, rs1, imm } => {
                self.write(rd, u64::from(self.read(rs1) < imm as u64));
            }
            NativeInstruction::Sltu { rd, rs1, rs2 } => {
                self.write(rd, u64::from(self.read(rs1) < self.read(rs2)));
            }
            NativeInstruction::Sra { rd, rs1, rs2 } => {
                self.write(
                    rd,
                    ((self.read(rs1) as i64) >> (self.read(rs2) & 0x3f)) as u64,
                );
            }
            NativeInstruction::Srai { rd, rs1, shamt } => {
                self.write(rd, ((self.read(rs1) as i64) >> shamt) as u64);
            }
            NativeInstruction::Sraiw { rd, rs1, shamt } => {
                let value = ((self.read(rs1) as u32) as i32) >> shamt;
                self.write(rd, value as i64 as u64);
            }
            NativeInstruction::Sraw { rd, rs1, rs2 } => {
                let value = ((self.read(rs1) as u32) as i32) >> (self.read(rs2) & 0x1f);
                self.write(rd, value as i64 as u64);
            }
            NativeInstruction::Srl { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1) >> (self.read(rs2) & 0x3f));
            }
            NativeInstruction::Srli { rd, rs1, shamt } => {
                self.write(rd, self.read(rs1) >> shamt);
            }
            NativeInstruction::Srliw { rd, rs1, shamt } => {
                let value = (self.read(rs1) as u32) >> shamt;
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::Srlw { rd, rs1, rs2 } => {
                let value = (self.read(rs1) as u32) >> (self.read(rs2) & 0x1f);
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::Sub { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1).wrapping_sub(self.read(rs2)));
            }
            NativeInstruction::Subw { rd, rs1, rs2 } => {
                let value = (self.read(rs1) as u32).wrapping_sub(self.read(rs2) as u32);
                self.write(rd, sign_extend(u64::from(value), 32) as u64);
            }
            NativeInstruction::Store {
                rs1,
                rs2,
                imm,
                width,
            } => {
                let address = self.read(rs1).wrapping_add_signed(imm);
                self.write_memory(address, self.read(rs2), width);
            }
            NativeInstruction::Xor { rd, rs1, rs2 } => {
                self.write(rd, self.read(rs1) ^ self.read(rs2));
            }
            NativeInstruction::Xori { rd, rs1, imm } => {
                self.write(rd, self.read(rs1) ^ imm as u64);
            }
            NativeInstruction::InlinedJump { .. } | NativeInstruction::Nop => {}
            _ => return false,
        }

        self.registers[0] = 0;
        true
    }
}

fn extend_load(value: u64, width: MemoryWidth, signed: bool) -> u64 {
    if !signed || width == MemoryWidth::Double {
        return value;
    }

    let bits = match width {
        MemoryWidth::Byte => 8,
        MemoryWidth::Half => 16,
        MemoryWidth::Word => 32,
        MemoryWidth::Double => 64,
    };
    sign_extend(value, bits) as u64
}

#[derive(Debug, Clone, Copy)]
struct IntegerBranch {
    rs1: u8,
    rs2: u8,
    target: u64,
    fallthrough: u64,
    condition: IntegerBranchCondition,
}

impl IntegerBranch {
    fn from_instruction(instruction: NativeInstruction) -> Option<Self> {
        match instruction {
            NativeInstruction::Beq {
                rs1,
                rs2,
                target,
                fallthrough,
            } => Some(Self {
                rs1,
                rs2,
                target,
                fallthrough,
                condition: IntegerBranchCondition::Eq,
            }),
            NativeInstruction::Bge {
                rs1,
                rs2,
                target,
                fallthrough,
            } => Some(Self {
                rs1,
                rs2,
                target,
                fallthrough,
                condition: IntegerBranchCondition::Ge,
            }),
            NativeInstruction::Bgeu {
                rs1,
                rs2,
                target,
                fallthrough,
            } => Some(Self {
                rs1,
                rs2,
                target,
                fallthrough,
                condition: IntegerBranchCondition::Geu,
            }),
            NativeInstruction::Blt {
                rs1,
                rs2,
                target,
                fallthrough,
            } => Some(Self {
                rs1,
                rs2,
                target,
                fallthrough,
                condition: IntegerBranchCondition::Lt,
            }),
            NativeInstruction::Bltu {
                rs1,
                rs2,
                target,
                fallthrough,
            } => Some(Self {
                rs1,
                rs2,
                target,
                fallthrough,
                condition: IntegerBranchCondition::Ltu,
            }),
            NativeInstruction::Bne {
                rs1,
                rs2,
                target,
                fallthrough,
            } => Some(Self {
                rs1,
                rs2,
                target,
                fallthrough,
                condition: IntegerBranchCondition::Ne,
            }),
            _ => None,
        }
    }

    fn evaluate(self, state: &TraceRegisterState) -> bool {
        let lhs = state.read(self.rs1);
        let rhs = state.read(self.rs2);
        match self.condition {
            IntegerBranchCondition::Eq => lhs == rhs,
            IntegerBranchCondition::Ne => lhs != rhs,
            IntegerBranchCondition::Ge => (lhs as i64) >= (rhs as i64),
            IntegerBranchCondition::Geu => lhs >= rhs,
            IntegerBranchCondition::Lt => (lhs as i64) < (rhs as i64),
            IntegerBranchCondition::Ltu => lhs < rhs,
        }
    }
}
