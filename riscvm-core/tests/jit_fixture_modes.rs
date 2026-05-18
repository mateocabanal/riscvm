#![cfg(all(target_arch = "aarch64", unix))]

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use goblin::elf::{header, Elf};
use riscvm_core::cpu::{RV64GCRegAbiName::Pc, RV64GC};
use riscvm_core::jit::{JitEngine, JitError, JitExecutionMode, JitOptions, JitStep};

const INTENTIONAL_FAULT_FIXTURE: &str = "tests/rv64gc/c/bin/invalid_deref";
const MAX_FIXTURE_GUEST_INSTRUCTIONS: u64 = 100_000_000;
const MAX_FIXTURE_ELAPSED: Duration = Duration::from_secs(15);

#[test]
fn all_executable_elf_test_binaries_match_interpreter_in_every_runtime_mode() {
    let fixtures = executable_elf_fixtures();
    assert!(
        fixtures.len() >= 36,
        "fixture discovery unexpectedly found only {} executable ELFs",
        fixtures.len()
    );

    for fixture in fixtures {
        let relative_path = relative_fixture_path(&fixture);
        if relative_path == INTENTIONAL_FAULT_FIXTURE {
            continue;
        }

        let interpreter = run_fixture(&fixture, None);

        for mode in [
            JitExecutionMode::Jit,
            JitExecutionMode::Hybrid,
            JitExecutionMode::Aot,
        ] {
            let output = run_fixture(&fixture, Some(mode));
            assert_eq!(
                output, interpreter,
                "{mode:?} diverged from interpreter for {relative_path}"
            );
        }
    }
}

#[test]
fn intentional_fault_fixture_fails_in_every_runtime_mode() {
    let fixture = workspace_root().join(INTENTIONAL_FAULT_FIXTURE);
    let interpreter = run_fixture_failure(&fixture, None);
    assert!(
        interpreter.contains("Invalid address"),
        "interpreter reported unexpected failure for {INTENTIONAL_FAULT_FIXTURE}: {interpreter}"
    );

    for mode in [
        JitExecutionMode::Jit,
        JitExecutionMode::Hybrid,
        JitExecutionMode::Aot,
    ] {
        let failure = run_fixture_failure(&fixture, Some(mode));
        assert!(
            failure.contains("Invalid address"),
            "{mode:?} reported unexpected failure for {INTENTIONAL_FAULT_FIXTURE}: {failure}"
        );
    }
}

#[test]
fn jit_modes_match_interpreter_for_fence_and_fcsr_csr_blocks() {
    let mut bin = Vec::new();
    bin.extend(rv64_word(rv64_i(0x73, 5, 1, 0b1_0101, 0x001))); // csrrwi x1, fflags, 21
    bin.extend(rv64_word(rv64_i(0x73, 6, 2, 0b0_1010, 0x001))); // csrrsi x2, fflags, 10
    bin.extend(rv64_word(rv64_i(0x73, 7, 3, 0b1_0000, 0x001))); // csrrci x3, fflags, 16
    bin.extend(rv64_word(rv64_i(0x73, 2, 4, 0, 0x001))); // csrrs x4, fflags, x0
    bin.extend(rv64_word(rv64_i(0x13, 0, 6, 0, 0x63))); // addi x6, x0, 0x63
    bin.extend(rv64_word(rv64_i(0x73, 1, 5, 6, 0x003))); // csrrw x5, fcsr, x6
    bin.extend(rv64_word(0x0000_000f)); // fence
    bin.extend(rv64_word(0x0000_100f)); // fence.i

    let interpreter = run_in_memory_program(&bin, None);
    assert_eq!(interpreter.pc, bin.len() as u64);
    assert_eq!(interpreter.fcsr, 0x63);

    for mode in [
        JitExecutionMode::Jit,
        JitExecutionMode::Hybrid,
        JitExecutionMode::Aot,
    ] {
        let snapshot = run_in_memory_program(&bin, Some(mode));
        assert_eq!(snapshot, interpreter, "{mode:?} diverged from interpreter");
    }
}

#[test]
fn jit_modes_report_guest_traps_without_interpreter_fallback() {
    let mut bin = Vec::new();
    bin.extend(rv64_word(rv64_i(0x13, 0, 1, 0, 5))); // addi x1, x0, 5
    bin.extend(rv64_word(0x0010_0073)); // ebreak

    let interpreter = run_in_memory_program(&bin, None);
    assert!(interpreter.should_quit);
    assert_eq!(interpreter.pc, 4);
    assert_eq!(interpreter.registers[1], 5);

    for mode in [
        JitExecutionMode::Jit,
        JitExecutionMode::Hybrid,
        JitExecutionMode::Aot,
    ] {
        let mut cpu = load_in_memory_program(&bin);
        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: mode,
            ..JitOptions::default()
        })
        .unwrap();

        let err = engine
            .step(&mut cpu)
            .expect_err("guest trap should be reported as a JIT runtime fault");
        assert!(
            matches!(err, JitError::RuntimeFault { ref reason, .. } if reason.contains("ebreak")),
            "{mode:?} reported unexpected error: {err}"
        );
        assert!(cpu.should_quit);
        assert_eq!(cpu.registers[Pc], interpreter.pc);
        assert_eq!(cpu.registers[1usize], interpreter.registers[1]);
    }
}

#[derive(Debug, PartialEq, Eq)]
struct FixtureOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    exit_register: u64,
    should_quit: bool,
}

fn run_fixture(path: &Path, mode: Option<JitExecutionMode>) -> FixtureOutput {
    run_fixture_result(path, mode).unwrap_or_else(|error| {
        let mode_name = mode
            .map(|mode| format!("{mode:?}"))
            .unwrap_or_else(|| "Interpreter".to_string());
        panic!("{mode_name} failed for {}: {error}", path.display());
    })
}

fn run_fixture_failure(path: &Path, mode: Option<JitExecutionMode>) -> String {
    run_fixture_result(path, mode)
        .expect_err("fixture should fail instead of completing successfully")
}

fn run_fixture_result(
    path: &Path,
    mode: Option<JitExecutionMode>,
) -> Result<FixtureOutput, String> {
    let bytes = fs::read(path).unwrap_or_else(|error| {
        panic!("failed to read fixture {}: {error}", path.display());
    });
    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.set_executable_path(path);
    cpu.set_argv([path.to_string_lossy().into_owned()]);
    let host_root = isolated_host_root();
    cpu.mount_host_directory(&host_root).unwrap();
    cpu.load_elf(bytes).unwrap();

    let result = if let Some(mode) = mode {
        run_jit_fixture_with_budget(&mut cpu, mode)
    } else {
        run_interpreter_fixture_with_budget(&mut cpu)
    };

    fs::remove_dir_all(host_root).unwrap();
    result.map(|_| FixtureOutput {
        stdout: cpu.stdout().to_vec(),
        stderr: cpu.stderr().to_vec(),
        exit_register: cpu.registers[10],
        should_quit: cpu.should_quit,
    })
}

#[derive(Debug, PartialEq, Eq)]
struct CpuSnapshot {
    registers: Vec<u64>,
    pc: u64,
    fcsr: u64,
    should_quit: bool,
}

fn run_in_memory_program(bin: &[u8], mode: Option<JitExecutionMode>) -> CpuSnapshot {
    let mut cpu = load_in_memory_program(bin);
    let end_pc = bin.len() as u64;
    if let Some(mode) = mode {
        let mut engine = JitEngine::with_options(JitOptions {
            execution_mode: mode,
            ..JitOptions::default()
        })
        .unwrap();
        run_jit_until_end(&mut cpu, &mut engine, end_pc, mode);
    } else {
        run_interpreter_until_end(&mut cpu, end_pc);
    }
    snapshot(&cpu)
}

fn load_in_memory_program(bin: &[u8]) -> RV64GC {
    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.load_bin(bin.to_vec());
    cpu
}

fn run_interpreter_until_end(cpu: &mut RV64GC, end_pc: u64) {
    for _ in 0..64 {
        if cpu.should_quit || cpu.registers[Pc] >= end_pc {
            return;
        }
        cpu.step();
    }
    panic!("interpreter did not reach end pc 0x{end_pc:x}");
}

fn run_jit_until_end(
    cpu: &mut RV64GC,
    engine: &mut JitEngine,
    end_pc: u64,
    mode: JitExecutionMode,
) {
    for _ in 0..64 {
        if cpu.should_quit || cpu.registers[Pc] >= end_pc {
            return;
        }
        engine.step(cpu).unwrap_or_else(|error| {
            panic!("{mode:?} failed at pc=0x{:x}: {error}", cpu.registers[Pc])
        });
    }
    panic!("{mode:?} did not reach end pc 0x{end_pc:x}");
}

fn run_interpreter_fixture_with_budget(cpu: &mut RV64GC) -> Result<(), String> {
    catch_interpreter_panic_silently(|| {
        let start = Instant::now();
        for executed in 0..MAX_FIXTURE_GUEST_INSTRUCTIONS {
            if cpu.should_quit {
                return Ok(());
            }
            if start.elapsed() > MAX_FIXTURE_ELAPSED {
                return Err(format!(
                    "interpreter fixture exceeded {:?} at pc=0x{:016x} after {executed} guest instructions",
                    MAX_FIXTURE_ELAPSED,
                    cpu.registers[Pc]
                ));
            }
            cpu.step();
        }
        Err(format!(
            "interpreter fixture exceeded {MAX_FIXTURE_GUEST_INSTRUCTIONS} guest instructions at pc=0x{:016x}",
            cpu.registers[Pc]
        ))
    })
}

fn run_jit_fixture_with_budget(cpu: &mut RV64GC, mode: JitExecutionMode) -> Result<(), String> {
    let mut engine = JitEngine::with_options(JitOptions {
        execution_mode: mode,
        ..JitOptions::default()
    })
    .map_err(|error| error.to_string())?;
    let start = Instant::now();
    let mut executed = 0u64;

    while !cpu.should_quit {
        if start.elapsed() > MAX_FIXTURE_ELAPSED {
            return Err(format!(
                "{mode:?} fixture exceeded {:?} at pc=0x{:016x} after {executed} guest instructions",
                MAX_FIXTURE_ELAPSED,
                cpu.registers[Pc]
            ));
        }
        if executed > MAX_FIXTURE_GUEST_INSTRUCTIONS {
            return Err(format!(
                "{mode:?} fixture exceeded {MAX_FIXTURE_GUEST_INSTRUCTIONS} guest instructions at pc=0x{:016x}",
                cpu.registers[Pc]
            ));
        }

        match engine.step(cpu).map_err(|error| error.to_string())? {
            JitStep::Native { instructions, .. } => {
                executed = executed.saturating_add(instructions.max(1));
            }
        }
    }

    Ok(())
}

fn snapshot(cpu: &RV64GC) -> CpuSnapshot {
    CpuSnapshot {
        registers: (0..33).map(|index| cpu.registers[index]).collect(),
        pc: cpu.registers[Pc],
        fcsr: cpu.fcsr.bits(),
        should_quit: cpu.should_quit,
    }
}

fn rv64_word(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

fn rv64_i(opcode: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
    ((imm as u32 & 0xfff) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | (u32::from(rd) << 7)
        | opcode
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("riscvm-core should be inside the workspace")
        .to_path_buf()
}

fn executable_elf_fixtures() -> Vec<PathBuf> {
    let mut fixtures = Vec::new();
    collect_executable_elf_fixtures(&workspace_root().join("tests"), &mut fixtures);
    fixtures.sort();
    fixtures
}

fn collect_executable_elf_fixtures(dir: &Path, fixtures: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();

    for path in entries {
        if path.is_dir() {
            collect_executable_elf_fixtures(&path, fixtures);
        } else if is_riscv_executable_elf(&path) {
            fixtures.push(path);
        }
    }
}

fn is_riscv_executable_elf(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let Ok(elf) = Elf::parse(&bytes) else {
        return false;
    };

    elf.header.e_machine == header::EM_RISCV
        && matches!(elf.header.e_type, header::ET_EXEC | header::ET_DYN)
}

fn relative_fixture_path(path: &Path) -> String {
    path.strip_prefix(workspace_root())
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    "panic with non-string payload".to_string()
}

fn catch_interpreter_panic_silently(f: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = PANIC_HOOK_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f)).map_err(panic_payload_to_string);
    std::panic::set_hook(hook);
    match result {
        Ok(result) => result,
        Err(error) => Err(error),
    }
}

fn isolated_host_root() -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "riscvm-jit-fixture-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}
