use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{self, Write};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionEngine {
    Interpreter,
    Jit,
    Hybrid,
    Aot,
}

impl ExecutionEngine {
    fn as_str(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Jit => "jit",
            Self::Hybrid => "hybrid",
            Self::Aot => "aot",
        }
    }

    fn uses_jit_blocks(self) -> bool {
        matches!(self, Self::Jit | Self::Hybrid | Self::Aot)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceOptions {
    pub trace: bool,
    pub profile: bool,
    pub top_limit: usize,
    pub trace_limit: Option<u64>,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self {
            trace: false,
            profile: false,
            top_limit: 20,
            trace_limit: None,
        }
    }
}

impl TraceOptions {
    pub fn enabled(self) -> bool {
        self.trace || self.profile
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionTrace {
    pub pc: u64,
    pub opcode: u32,
    pub text: String,
}

#[derive(Debug, Clone)]
struct InstructionProfile {
    pc: u64,
    opcode: u32,
    text: String,
    count: u64,
}

#[derive(Debug, Clone)]
struct BlockProfile {
    pc: u64,
    count: u64,
    instruction_repetitions: u64,
    instruction_traces: Vec<InstructionTrace>,
    instructions: usize,
    code_bytes: usize,
    compile_count: u64,
    compile_time: Duration,
    execute_time: Duration,
}

#[derive(Debug)]
pub struct ExecutionTracer {
    options: TraceOptions,
    engine: Option<ExecutionEngine>,
    started_at: Option<Instant>,
    elapsed: Duration,
    instruction_count: u64,
    interpreter_steps: u64,
    jit_block_entries: u64,
    jit_compiles: u64,
    jit_cache_hits: u64,
    jit_cache_misses: u64,
    jit_code_bytes: u64,
    jit_compile_time: Duration,
    jit_execute_time: Duration,
    jit_background_queued: u64,
    jit_background_adopted: u64,
    jit_background_discarded: u64,
    jit_background_queue_full: u64,
    trace_events_emitted: u64,
    trace_limit_reported: bool,
    instructions: HashMap<u64, InstructionProfile>,
    blocks: HashMap<u64, BlockProfile>,
    syscalls: HashMap<u64, u64>,
}

impl ExecutionTracer {
    pub fn new(options: TraceOptions) -> Self {
        Self {
            options,
            engine: None,
            started_at: None,
            elapsed: Duration::ZERO,
            instruction_count: 0,
            interpreter_steps: 0,
            jit_block_entries: 0,
            jit_compiles: 0,
            jit_cache_hits: 0,
            jit_cache_misses: 0,
            jit_code_bytes: 0,
            jit_compile_time: Duration::ZERO,
            jit_execute_time: Duration::ZERO,
            jit_background_queued: 0,
            jit_background_adopted: 0,
            jit_background_discarded: 0,
            jit_background_queue_full: 0,
            trace_events_emitted: 0,
            trace_limit_reported: false,
            instructions: HashMap::new(),
            blocks: HashMap::new(),
            syscalls: HashMap::new(),
        }
    }

    pub fn options(&self) -> TraceOptions {
        self.options
    }

    pub fn start(&mut self, engine: ExecutionEngine) {
        let options = self.options;
        *self = Self::new(options);
        self.engine = Some(engine);
        self.started_at = Some(Instant::now());
        self.trace_line(format_args!("[trace:{}] start", engine.as_str()));
    }

    pub fn finish(&mut self) {
        if let Some(started_at) = self.started_at.take() {
            self.elapsed += started_at.elapsed();
        }
        if let Some(engine) = self.engine {
            self.trace_line(format_args!(
                "[trace:{}] stop elapsed={}",
                engine.as_str(),
                format_duration(self.elapsed)
            ));
        }
    }

    pub fn record_interpreter_instruction(&mut self, pc: u64, opcode: u32, text: &str) {
        self.record_interpreter_instruction_lazy(pc, opcode, || text.to_string());
    }

    pub fn record_interpreter_instruction_lazy<F>(&mut self, pc: u64, opcode: u32, text: F)
    where
        F: FnOnce() -> String,
    {
        self.instruction_count = self.instruction_count.saturating_add(1);
        self.interpreter_steps = self.interpreter_steps.saturating_add(1);
        if self.options.trace {
            let text = text();
            self.record_instruction(pc, opcode, &text, 1);
            self.trace_line(format_args!(
                "[trace:interp] pc=0x{pc:016x} opcode=0x{opcode:08x} {text}"
            ));
        } else {
            self.record_instruction_lazy(pc, opcode, 1, text);
        }
    }

    pub fn record_jit_cache_hit(&mut self) {
        self.jit_cache_hits = self.jit_cache_hits.saturating_add(1);
    }

    pub fn record_jit_background_queue(&mut self) {
        self.jit_background_queued = self.jit_background_queued.saturating_add(1);
    }

    pub fn record_jit_background_adoption(&mut self) {
        self.jit_background_adopted = self.jit_background_adopted.saturating_add(1);
    }

    pub fn record_jit_background_discard(&mut self) {
        self.jit_background_discarded = self.jit_background_discarded.saturating_add(1);
    }

    pub fn record_jit_background_queue_full(&mut self) {
        self.jit_background_queue_full = self.jit_background_queue_full.saturating_add(1);
    }

    pub fn record_jit_compile(
        &mut self,
        pc: u64,
        instruction_count: usize,
        code_bytes: usize,
        stop: &str,
        duration: Duration,
    ) {
        self.jit_compiles = self.jit_compiles.saturating_add(1);
        self.jit_cache_misses = self.jit_cache_misses.saturating_add(1);
        self.jit_code_bytes = self.jit_code_bytes.saturating_add(code_bytes as u64);
        self.jit_compile_time += duration;

        let block = self.blocks.entry(pc).or_insert_with(|| BlockProfile {
            pc,
            count: 0,
            instruction_repetitions: 0,
            instruction_traces: Vec::new(),
            instructions: instruction_count,
            code_bytes,
            compile_count: 0,
            compile_time: Duration::ZERO,
            execute_time: Duration::ZERO,
        });
        block.instructions = instruction_count;
        block.code_bytes = code_bytes;
        block.compile_count = block.compile_count.saturating_add(1);
        block.compile_time += duration;

        self.trace_line(format_args!(
            "[trace:jit] compile pc=0x{pc:016x} instructions={instruction_count} code_bytes={code_bytes} stop={stop} elapsed={}",
            format_duration(duration)
        ));
    }

    pub fn record_jit_block(
        &mut self,
        pc: u64,
        instruction_count: usize,
        executed_instructions: u64,
        code_bytes: usize,
        next_pc: u64,
        duration: Duration,
        instructions: &[InstructionTrace],
    ) {
        self.jit_block_entries = self.jit_block_entries.saturating_add(1);
        self.instruction_count = self.instruction_count.saturating_add(executed_instructions);
        self.jit_execute_time += duration;

        let block = self.blocks.entry(pc).or_insert_with(|| BlockProfile {
            pc,
            count: 0,
            instruction_repetitions: 0,
            instruction_traces: Vec::new(),
            instructions: instruction_count,
            code_bytes,
            compile_count: 0,
            compile_time: Duration::ZERO,
            execute_time: Duration::ZERO,
        });
        block.count = block.count.saturating_add(1);
        block.instructions = instruction_count;
        block.code_bytes = code_bytes;
        block.execute_time += duration;
        if block.instruction_traces.is_empty() {
            block.instruction_traces.extend_from_slice(instructions);
        }

        let instruction_repetitions = executed_instructions
            .checked_div(instruction_count as u64)
            .unwrap_or(1)
            .max(1);
        block.instruction_repetitions = block
            .instruction_repetitions
            .saturating_add(instruction_repetitions);

        self.trace_line(format_args!(
            "[trace:jit] block pc=0x{pc:016x} instructions={executed_instructions} code_bytes={code_bytes} next_pc=0x{next_pc:016x} elapsed={}",
            format_duration(duration)
        ));
    }

    pub fn record_syscall(&mut self, syscall_id: u64) {
        *self.syscalls.entry(syscall_id).or_insert(0) += 1;
        self.trace_line(format_args!("[trace:syscall] id={syscall_id}"));
    }

    pub fn report(&self) -> String {
        let elapsed = self.elapsed();
        let mut report = String::new();
        let engine = self
            .engine
            .map(ExecutionEngine::as_str)
            .unwrap_or("unknown");
        let mips = if elapsed.is_zero() {
            0.0
        } else {
            self.instruction_count as f64 / elapsed.as_secs_f64() / 1_000_000.0
        };

        let _ = writeln!(report, "[profile] engine={engine}");
        let _ = writeln!(
            report,
            "[profile] elapsed={} instructions={} throughput={mips:.3} MIPS",
            format_duration(elapsed),
            self.instruction_count
        );
        let _ = writeln!(
            report,
            "[profile] interpreter_steps={} jit_block_entries={}",
            self.interpreter_steps, self.jit_block_entries
        );

        if self.engine.is_some_and(ExecutionEngine::uses_jit_blocks) {
            let _ = writeln!(
                report,
                "[profile] jit_compiles={} cache_hits={} cache_misses={} code_bytes={} compile_time={} execute_time={}",
                self.jit_compiles,
                self.jit_cache_hits,
                self.jit_cache_misses,
                self.jit_code_bytes,
                format_duration(self.jit_compile_time),
                format_duration(self.jit_execute_time)
            );
            if self.jit_background_queued > 0
                || self.jit_background_adopted > 0
                || self.jit_background_discarded > 0
                || self.jit_background_queue_full > 0
            {
                let _ = writeln!(
                    report,
                    "[profile] jit_background queued={} adopted={} discarded={} queue_full={}",
                    self.jit_background_queued,
                    self.jit_background_adopted,
                    self.jit_background_discarded,
                    self.jit_background_queue_full
                );
            }
        }

        if !self.syscalls.is_empty() {
            let _ = writeln!(report, "[profile] syscalls:");
            let mut syscalls: Vec<_> = self.syscalls.iter().collect();
            syscalls.sort_by(|(lhs_id, lhs_count), (rhs_id, rhs_count)| {
                rhs_count.cmp(lhs_count).then(lhs_id.cmp(rhs_id))
            });
            for (syscall_id, count) in syscalls.into_iter().take(self.options.top_limit) {
                let _ = writeln!(report, "[profile]   id={syscall_id} count={count}");
            }
        }

        let instructions = self.hot_instruction_profiles();
        if !instructions.is_empty() {
            let _ = writeln!(report, "[profile] hot instructions:");
            for instruction in instructions.into_iter().take(self.options.top_limit) {
                let _ = writeln!(
                    report,
                    "[profile]   pc=0x{:016x} count={} opcode=0x{:08x} {}",
                    instruction.pc, instruction.count, instruction.opcode, instruction.text
                );
            }
        }

        if !self.blocks.is_empty() {
            let _ = writeln!(report, "[profile] hot jit blocks:");
            let mut blocks: Vec<_> = self.blocks.values().collect();
            blocks.sort_by(|lhs, rhs| rhs.count.cmp(&lhs.count).then(lhs.pc.cmp(&rhs.pc)));
            for block in blocks.into_iter().take(self.options.top_limit) {
                let _ = writeln!(
                    report,
                    "[profile]   pc=0x{:016x} count={} instructions={} code_bytes={} compiles={} compile_time={} execute_time={}",
                    block.pc,
                    block.count,
                    block.instructions,
                    block.code_bytes,
                    block.compile_count,
                    format_duration(block.compile_time),
                    format_duration(block.execute_time)
                );
            }
        }

        report
    }

    fn elapsed(&self) -> Duration {
        let running = self
            .started_at
            .map(|started_at| started_at.elapsed())
            .unwrap_or(Duration::ZERO);
        self.elapsed + running
    }

    fn hot_instruction_profiles(&self) -> Vec<InstructionProfile> {
        let mut profiles = self.instructions.clone();

        if self.engine.is_some_and(ExecutionEngine::uses_jit_blocks) {
            for block in self.blocks.values() {
                for instruction in &block.instruction_traces {
                    let profile =
                        profiles
                            .entry(instruction.pc)
                            .or_insert_with(|| InstructionProfile {
                                pc: instruction.pc,
                                opcode: instruction.opcode,
                                text: instruction.text.clone(),
                                count: 0,
                            });
                    profile.count = profile.count.saturating_add(block.instruction_repetitions);
                }
            }
        }

        let mut profiles: Vec<_> = profiles.into_values().collect();
        profiles.sort_by(|lhs, rhs| rhs.count.cmp(&lhs.count).then(lhs.pc.cmp(&rhs.pc)));
        profiles
    }

    fn record_instruction(&mut self, pc: u64, opcode: u32, text: &str, count: u64) {
        self.record_instruction_lazy(pc, opcode, count, || text.to_string());
    }

    fn record_instruction_lazy<F>(&mut self, pc: u64, opcode: u32, count: u64, text: F)
    where
        F: FnOnce() -> String,
    {
        match self.instructions.entry(pc) {
            Entry::Occupied(mut entry) => {
                let instruction = entry.get_mut();
                instruction.count = instruction.count.saturating_add(count);
                if instruction.opcode != opcode {
                    instruction.opcode = opcode;
                    instruction.text = text();
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(InstructionProfile {
                    pc,
                    opcode,
                    text: text(),
                    count,
                });
            }
        }
    }

    fn trace_line(&mut self, args: fmt::Arguments<'_>) {
        if !self.options.trace {
            return;
        }

        if let Some(limit) = self.options.trace_limit {
            if self.trace_events_emitted >= limit {
                if !self.trace_limit_reported {
                    eprintln!("[trace] trace limit reached after {limit} events");
                    self.trace_limit_reported = true;
                }
                return;
            }
        }

        eprintln!("{args}");
        self.trace_events_emitted = self.trace_events_emitted.saturating_add(1);
    }
}

fn format_duration(duration: Duration) -> String {
    format!("{:.6}s", duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::{ExecutionEngine, ExecutionTracer, TraceOptions};

    #[test]
    fn profile_report_includes_hot_instruction_counts() {
        let mut tracer = ExecutionTracer::new(TraceOptions {
            profile: true,
            top_limit: 4,
            ..TraceOptions::default()
        });

        tracer.start(ExecutionEngine::Interpreter);
        tracer.record_interpreter_instruction(0x1000, 0x0010_0093, "addi x1, x0, 1");
        tracer.record_interpreter_instruction(0x1000, 0x0010_0093, "addi x1, x0, 1");
        tracer.record_syscall(93);
        tracer.finish();

        let report = tracer.report();
        assert!(report.contains("engine=interpreter"));
        assert!(report.contains("instructions=2"));
        assert!(report.contains("pc=0x0000000000001000 count=2"));
        assert!(report.contains("id=93 count=1"));
    }
}
