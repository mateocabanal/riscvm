use std::hint::black_box;
#[cfg(unix)]
use std::os::raw::{c_char, c_int, c_ulonglong};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{env, fs};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use riscvm_core::cpu::RV64GC;
use riscvm_core::jit::{JitExecutionMode, JitOptions};

const SYNTHETIC_FIXTURES: &[(&str, &str)] = &[
    ("arith", "tests/rv64gc/synthetic/bin/arith"),
    ("branch", "tests/rv64gc/synthetic/bin/branch"),
    ("memory", "tests/rv64gc/synthetic/bin/memory"),
    ("fibonacci", "tests/rv64gc/synthetic/bin/fibonacci"),
    (
        "fibonacci-core",
        "tests/rv64gc/synthetic/bin/fibonacci_core",
    ),
];

const NESTED_RISCVM_SOURCE_PATHS: &[&str] = &[
    "Cargo.lock",
    "Cargo.toml",
    "riscvm-core/Cargo.toml",
    "riscvm-core/src",
    "riscvm-runner/Cargo.toml",
    "riscvm-runner/src",
];

#[derive(Clone)]
struct BenchBinary {
    name: &'static str,
    path: PathBuf,
    argv: Vec<String>,
    bytes: Vec<u8>,
}

impl BenchBinary {
    fn load_direct(name: &'static str, relative_path: &str) -> Option<Self> {
        let path = workspace_root().join(relative_path);
        let bytes = load_binary(&path)?;
        Some(Self {
            name,
            argv: vec![path.to_string_lossy().into_owned()],
            path,
            bytes,
        })
    }

    fn load_nested(
        name: &'static str,
        nested_riscvm_path: &Path,
        guest_binary_relative_path: &str,
    ) -> Option<Self> {
        let bytes = load_binary(nested_riscvm_path)?;
        Some(Self {
            name,
            path: nested_riscvm_path.to_path_buf(),
            argv: vec![
                nested_riscvm_path.to_string_lossy().into_owned(),
                guest_binary_relative_path.to_string(),
            ],
            bytes,
        })
    }

    fn cpu(&self) -> RV64GC {
        let mut cpu = RV64GC::new();
        cpu.filesystem.set_output_mirroring(false);
        cpu.set_executable_path(&self.path);
        cpu.set_argv(self.argv.clone());
        cpu.mount_host_directory(workspace_root()).unwrap();
        cpu.load_elf(self.bytes.clone()).unwrap();
        cpu
    }
}

fn synthetic_benches(c: &mut Criterion) {
    bench_direct_synthetic(c);
    bench_nested_synthetic(c);
    bench_host_libc_synthetic(c);
}

fn bench_direct_synthetic(c: &mut Criterion) {
    let cases = synthetic_cases();
    if cases.is_empty() {
        eprintln!(
            "skipping direct synthetic benchmarks: no fixture binaries found under tests/rv64gc/synthetic/bin"
        );
        return;
    }

    let mut group = c.benchmark_group("synthetic/direct");
    for case in cases {
        bench_case(&mut group, "interpreter", case.clone(), run_interpreter);
        bench_case(&mut group, "jit", case.clone(), run_jit);
        bench_case(&mut group, "jit-trace", case.clone(), run_trace_jit);
        bench_case(&mut group, "hybrid", case.clone(), run_hybrid);
        bench_case(&mut group, "aot", case.clone(), run_aot);
        if case.name == "fibonacci-core" {
            bench_case(
                &mut group,
                "jit-libc-entry-shortcut",
                case.clone(),
                run_jit_libc_entry_shortcut,
            );
            bench_case(
                &mut group,
                "aot-libc-entry-shortcut",
                case,
                run_aot_libc_entry_shortcut,
            );
        }
    }
    group.finish();
}

fn bench_nested_synthetic(c: &mut Criterion) {
    if env::var_os("RISCVM_BENCH_NESTED").is_none() {
        eprintln!("skipping nested synthetic benchmarks: set RISCVM_BENCH_NESTED=1 to enable");
        return;
    }

    let nested_riscvm = nested_riscvm_path();
    if let Err(reason) = ensure_nested_riscvm_is_current(&nested_riscvm) {
        eprintln!("skipping nested synthetic benchmarks: {reason}");
        return;
    }

    let cases: Vec<_> = SYNTHETIC_FIXTURES
        .iter()
        .filter_map(|(name, relative_path)| {
            BenchBinary::load_nested(name, &nested_riscvm, relative_path)
        })
        .collect();
    if cases.is_empty() {
        eprintln!(
            "skipping nested synthetic benchmarks: RV64 riscvm binary not found at {}",
            nested_riscvm.display()
        );
        return;
    }

    let mut group = c.benchmark_group("synthetic/nested-rv64-riscvm");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(10));
    for case in cases {
        bench_case(
            &mut group,
            "outer-interpreter",
            case.clone(),
            run_interpreter,
        );
        bench_case(&mut group, "outer-jit", case.clone(), run_jit);
        bench_case(&mut group, "outer-hybrid", case, run_hybrid);
    }
    group.finish();
}

#[cfg(unix)]
unsafe extern "C" {
    fn snprintf(buffer: *mut c_char, count: usize, format: *const c_char, ...) -> c_int;
}

#[cfg(unix)]
fn bench_host_libc_synthetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic/host-libc");
    group.bench_function("fibonacci", |b| b.iter(run_host_libc_fibonacci));
    group.finish();
}

#[cfg(not(unix))]
fn bench_host_libc_synthetic(_c: &mut Criterion) {
    eprintln!("skipping host libc synthetic benchmarks: host libc FFI requires Unix");
}

fn bench_case<F>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    engine: &'static str,
    case: BenchBinary,
    run: F,
) where
    F: Fn(RV64GC) + Copy + 'static,
{
    group.bench_function(BenchmarkId::new(engine, case.name), move |b| {
        b.iter_batched(|| case.cpu(), run, BatchSize::SmallInput);
    });
}

fn run_interpreter(mut cpu: RV64GC) {
    cpu.start();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

fn run_jit(mut cpu: RV64GC) {
    // Direct benches use the freshly compiled riscvm_core crate from this cargo bench run.
    cpu.start_jit_with_options(jit_options(JitExecutionMode::Jit))
        .unwrap();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

fn run_trace_jit(mut cpu: RV64GC) {
    cpu.start_jit_with_options(JitOptions {
        execution_mode: JitExecutionMode::Jit,
        trace_compilation: true,
        tier_budgeting: false,
        ..JitOptions::default()
    })
    .unwrap();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

fn run_hybrid(mut cpu: RV64GC) {
    cpu.start_jit_with_options(jit_options(JitExecutionMode::Hybrid))
        .unwrap();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

fn run_aot(mut cpu: RV64GC) {
    cpu.start_jit_with_options(jit_options(JitExecutionMode::Aot))
        .unwrap();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

fn run_jit_libc_entry_shortcut(mut cpu: RV64GC) {
    cpu.start_jit_with_options(libc_entry_shortcut_options(JitExecutionMode::Jit))
        .unwrap();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

fn run_aot_libc_entry_shortcut(mut cpu: RV64GC) {
    cpu.start_jit_with_options(libc_entry_shortcut_options(JitExecutionMode::Aot))
        .unwrap();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

#[cfg(unix)]
fn run_host_libc_fibonacci() {
    let checksum = host_fibonacci_rounds(5_000_000);
    let mut buffer = [0 as c_char; 64];
    let written = unsafe {
        snprintf(
            buffer.as_mut_ptr(),
            buffer.len(),
            c"fibonacci checksum: %llu\n".as_ptr(),
            checksum as c_ulonglong,
        )
    };
    black_box(checksum);
    black_box(written);
    black_box(buffer);
}

fn jit_options(execution_mode: JitExecutionMode) -> JitOptions {
    JitOptions {
        execution_mode,
        ..JitOptions::default()
    }
}

fn libc_entry_shortcut_options(execution_mode: JitExecutionMode) -> JitOptions {
    JitOptions {
        execution_mode,
        libc_start_main_shortcut: true,
        ..JitOptions::default()
    }
}

#[cfg(unix)]
fn host_fibonacci_rounds(rounds: u32) -> u64 {
    let mut previous = 1u64;
    let mut current = 1u64;
    let mut checksum = 0u64;

    for _ in 0..rounds {
        let next = previous.wrapping_add(current);
        checksum ^= next;
        previous = current;
        current = next;
    }

    checksum
}

fn synthetic_cases() -> Vec<BenchBinary> {
    SYNTHETIC_FIXTURES
        .iter()
        .filter_map(|(name, relative_path)| BenchBinary::load_direct(name, relative_path))
        .collect()
}

fn load_binary(path: &Path) -> Option<Vec<u8>> {
    match fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(error) => {
            eprintln!("skipping benchmark binary {}: {error}", path.display());
            None
        }
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("riscvm-core should live under the workspace root")
        .to_path_buf()
}

fn nested_riscvm_path() -> PathBuf {
    env::var_os("RISCVM_NESTED_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            workspace_root().join("target/riscv64gc-unknown-linux-gnu/release/riscvm")
        })
}

fn ensure_nested_riscvm_is_current(path: &Path) -> Result<(), String> {
    let binary_modified = modified_time(path).map_err(|error| {
        format!(
            "RV64 riscvm binary not found at {}: {error}",
            path.display()
        )
    })?;
    let sources_modified = newest_nested_riscvm_source_time()
        .map_err(|error| format!("could not inspect workspace sources: {error}"))?;

    if binary_modified < sources_modified {
        return Err(format!(
            "RV64 riscvm binary at {} is older than the workspace sources; rebuild it or set RISCVM_NESTED_BIN to a current binary",
            path.display()
        ));
    }

    Ok(())
}

fn newest_nested_riscvm_source_time() -> std::io::Result<SystemTime> {
    let root = workspace_root();
    let mut newest = UNIX_EPOCH;
    for relative_path in NESTED_RISCVM_SOURCE_PATHS {
        update_newest_modified_time(&root.join(relative_path), &mut newest)?;
    }
    Ok(newest)
}

fn update_newest_modified_time(path: &Path, newest: &mut SystemTime) -> std::io::Result<()> {
    let metadata = fs::metadata(path)?;
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            update_newest_modified_time(&entry?.path(), newest)?;
        }
        return Ok(());
    }

    *newest = (*newest).max(metadata.modified()?);
    Ok(())
}

fn modified_time(path: &Path) -> std::io::Result<SystemTime> {
    fs::metadata(path)?.modified()
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = synthetic_benches
}
criterion_main!(benches);
