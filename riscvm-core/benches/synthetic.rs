use std::hint::black_box;
#[cfg(unix)]
use std::os::raw::{c_char, c_int, c_ulonglong};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{env, fs};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use riscvm_core::cpu::RV64GC;
use riscvm_core::jit::{JitEngine, JitExecutionMode, JitOptions};
use riscvm_core::ram::MemoryRegion;
use riscvm_core::sign_extend;

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
    bench_direct_trace_load(c);
    bench_counted_diamond_region(c);
    bench_store_load_forward_region(c);
    bench_fibonacci_region(c);
    bench_division_region(c);
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
        if let Some(host_runner) = host_direct_runner(case.name) {
            group.bench_function(BenchmarkId::new("host-rust", case.name), |b| {
                b.iter(host_runner);
            });
        }
        bench_case(&mut group, "interpreter", case.clone(), run_interpreter);
        bench_case(&mut group, "jit", case.clone(), run_jit);
        bench_case(&mut group, "jit-trace", case.clone(), run_trace_jit);
        bench_case(&mut group, "hybrid", case.clone(), run_hybrid);
        bench_case(&mut group, "aot", case.clone(), run_aot);
        bench_case_cached_engine(
            &mut group,
            "jit-cached",
            case.clone(),
            jit_options(JitExecutionMode::Jit),
        );
        bench_case_cached_engine(
            &mut group,
            "jit-trace-cached",
            case.clone(),
            trace_jit_options(),
        );
        bench_case_cached_engine(
            &mut group,
            "hybrid-cached",
            case.clone(),
            jit_options(JitExecutionMode::Hybrid),
        );
        bench_case_cached_engine(
            &mut group,
            "aot-cached",
            case.clone(),
            jit_options(JitExecutionMode::Aot),
        );
        if case.name == "fibonacci-core" {
            bench_case_cached_engine(
                &mut group,
                "jit-cached-libc-entry-shortcut",
                case.clone(),
                libc_entry_shortcut_options(JitExecutionMode::Jit),
            );
            bench_case_cached_engine(
                &mut group,
                "aot-cached-libc-entry-shortcut",
                case.clone(),
                libc_entry_shortcut_options(JitExecutionMode::Aot),
            );
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

fn bench_direct_trace_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic/direct-trace-load");
    group.bench_function("host-rust/load-branch", |b| {
        b.iter_batched(
            trace_load_host_data,
            run_host_direct_trace_load,
            BatchSize::SmallInput,
        );
    });
    group.bench_function("host-rust/load-byte-branch", |b| {
        b.iter_batched(
            trace_load_host_byte_data,
            run_host_direct_trace_load_bytes,
            BatchSize::SmallInput,
        );
    });
    group.bench_function("jit-no-trace/load-branch", |b| {
        b.iter_batched(trace_load_cpu, run_jit_without_trace, BatchSize::SmallInput);
    });
    group.bench_function("jit/load-branch", |b| {
        b.iter_batched(trace_load_cpu, run_jit, BatchSize::SmallInput);
    });
    group.bench_function("jit-trace/load-branch", |b| {
        b.iter_batched(trace_load_cpu, run_trace_jit, BatchSize::SmallInput);
    });
    group.bench_function("hybrid/load-branch", |b| {
        b.iter_batched(trace_load_cpu, run_hybrid, BatchSize::SmallInput);
    });
    group.bench_function("aot/load-branch", |b| {
        b.iter_batched(trace_load_cpu, run_aot, BatchSize::SmallInput);
    });
    group.bench_function("jit-no-trace/load-byte-branch", |b| {
        b.iter_batched(
            trace_load_byte_cpu,
            run_jit_without_trace,
            BatchSize::SmallInput,
        );
    });
    group.bench_function("jit-trace/load-byte-branch", |b| {
        b.iter_batched(trace_load_byte_cpu, run_trace_jit, BatchSize::SmallInput);
    });
    bench_trace_load_cached_engine(
        &mut group,
        "jit-cached/load-branch",
        jit_options(JitExecutionMode::Jit),
    );
    bench_trace_load_cached_engine(
        &mut group,
        "jit-trace-cached/load-branch",
        trace_jit_options(),
    );
    bench_trace_load_cached_engine(
        &mut group,
        "hybrid-cached/load-branch",
        jit_options(JitExecutionMode::Hybrid),
    );
    bench_trace_load_cached_engine(
        &mut group,
        "aot-cached/load-branch",
        jit_options(JitExecutionMode::Aot),
    );
    bench_trace_load_cached_engine_with(
        &mut group,
        "jit-trace-cached/load-byte-branch",
        trace_jit_options(),
        trace_load_byte_cpu,
    );
    group.finish();
}

fn bench_fibonacci_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic/fibonacci-region");
    group.bench_function("host-rust/recurrence", |b| {
        b.iter(run_host_fibonacci_core);
    });
    bench_fibonacci_region_cached_engine(
        &mut group,
        "jit-cached/recurrence",
        jit_options(JitExecutionMode::Jit),
    );
    bench_fibonacci_region_cached_engine(
        &mut group,
        "jit-trace-cached/recurrence",
        trace_jit_options(),
    );
    bench_fibonacci_region_cached_engine(
        &mut group,
        "hybrid-cached/recurrence",
        jit_options(JitExecutionMode::Hybrid),
    );
    bench_fibonacci_region_cached_engine(
        &mut group,
        "aot-cached/recurrence",
        jit_options(JitExecutionMode::Aot),
    );
    group.finish();
}

fn bench_counted_diamond_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic/counted-diamond-region");
    group.bench_function("host-rust/mask3", |b| {
        b.iter(run_host_counted_diamond_mask3)
    });
    bench_counted_diamond_region_cached_engine(
        &mut group,
        "jit-cached/mask3",
        jit_options(JitExecutionMode::Jit),
    );
    bench_counted_diamond_region_cached_engine(
        &mut group,
        "aot-cached/mask3",
        jit_options(JitExecutionMode::Aot),
    );
    group.finish();
}

fn bench_store_load_forward_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic/store-load-forward-region");
    group.bench_function("host-rust/ring", |b| {
        b.iter(run_host_store_load_forward_ring)
    });
    bench_store_load_forward_region_cached_engine(
        &mut group,
        "jit-cached/ring",
        jit_options(JitExecutionMode::Jit),
    );
    bench_store_load_forward_region_cached_engine(
        &mut group,
        "aot-cached/ring",
        jit_options(JitExecutionMode::Aot),
    );
    group.finish();
}

fn bench_division_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic/division-region");
    group.bench_function("host-rust/div-rem", |b| b.iter(run_host_division_core));
    bench_division_region_cached_engine(
        &mut group,
        "aot-cached/div-rem",
        jit_options(JitExecutionMode::Aot),
    );
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

fn trace_load_cpu() -> RV64GC {
    const BASE: u64 = 0x1000;
    const ITERATIONS: u64 = 250_000;

    let mut data = vec![0u8; (ITERATIONS * 8) as usize];
    for slot in data.chunks_exact_mut(8) {
        slot.copy_from_slice(&1u64.to_le_bytes());
    }

    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.load_bin(trace_load_branch_bin());
    cpu.ram
        .add_region(MemoryRegion::new(BASE, data.len() as u64, data))
        .unwrap();
    cpu.registers[5usize] = ITERATIONS;
    cpu.registers[10usize] = BASE;
    cpu
}

fn trace_load_byte_cpu() -> RV64GC {
    const BASE: u64 = 0x1000;
    const ITERATIONS: u64 = 250_000;

    let data = vec![1u8; ITERATIONS as usize];
    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.load_bin(trace_load_byte_branch_bin());
    cpu.ram
        .add_region(MemoryRegion::new(BASE, data.len() as u64, data))
        .unwrap();
    cpu.registers[5usize] = ITERATIONS;
    cpu.registers[10usize] = BASE;
    cpu
}

fn trace_load_host_data() -> Vec<u64> {
    vec![1; 250_000]
}

fn trace_load_host_byte_data() -> Vec<u8> {
    vec![1; 250_000]
}

fn run_host_direct_trace_load(data: Vec<u64>) {
    let data = black_box(data);
    let mut loaded = 0;
    let mut counter = 0usize;
    while counter < data.len() {
        loaded = data[counter];
        if loaded == 0 {
            break;
        }
        counter += 1;
    }
    black_box((counter, loaded));
}

fn run_host_direct_trace_load_bytes(data: Vec<u8>) {
    let data = black_box(data);
    let mut loaded = 0;
    let mut counter = 0usize;
    while counter < data.len() {
        loaded = data[counter];
        if loaded == 0 {
            break;
        }
        counter += 1;
    }
    black_box((counter, loaded));
}

fn run_host_fibonacci_core() {
    black_box(host_fibonacci_rounds(5_000_000));
}

fn run_host_division_core() {
    let mut count = black_box(50_000u64);
    let mut signed = black_box((-123_456_789i64) as u64);
    let signed_divisor = black_box(37u64);
    let mut unsigned = black_box(0xabcd_ef01_2345_6789u64);
    let unsigned_divisor = black_box(97u64);
    let mut checksum = 0u64;

    while count != 0 {
        let div = ((signed as i64).wrapping_div(signed_divisor as i64)) as u64;
        let rem = ((signed as i64).wrapping_rem(signed_divisor as i64)) as u64;
        let divu = unsigned.wrapping_div(unsigned_divisor);
        let remu = unsigned.wrapping_rem(unsigned_divisor);
        let divw = sign_extend(
            (signed as i32).wrapping_div(signed_divisor as i32) as u32 as u64,
            32,
        ) as u64;
        let remw = sign_extend(
            (signed as i32).wrapping_rem(signed_divisor as i32) as u32 as u64,
            32,
        ) as u64;
        let divuw = sign_extend(
            u64::from((unsigned as u32).wrapping_div(unsigned_divisor as u32)),
            32,
        ) as u64;
        let remuw = sign_extend(
            u64::from((unsigned as u32).wrapping_rem(unsigned_divisor as u32)),
            32,
        ) as u64;
        let mulhsu = host_mulhsu(signed, unsigned_divisor);

        checksum ^= div ^ rem ^ divu ^ remu ^ divw ^ remw ^ divuw ^ remuw ^ mulhsu;
        signed = signed.wrapping_add(3);
        unsigned = unsigned.wrapping_add(5);
        count = count.wrapping_sub(1);
    }

    black_box((checksum, signed, unsigned, count));
}

fn run_host_counted_diamond_mask3() {
    let mut count = black_box(5_000_000u64);
    let mut accumulator = black_box(0u64);
    let mut iteration = black_box(0u64);
    let mut parity = 0u64;
    let mut executed = 0u64;

    while count != 0 {
        parity = count & 3;
        if parity == 0 {
            accumulator = accumulator.wrapping_add(7);
            executed = executed.wrapping_add(6);
        } else {
            accumulator = accumulator.wrapping_add(3);
            executed = executed.wrapping_add(7);
        }
        iteration = iteration.wrapping_add(1);
        count = count.wrapping_sub(1);
    }

    black_box((count, accumulator, iteration, parity, executed));
}

fn run_host_store_load_forward_ring() {
    let mut slots = black_box([0u64; 4]);
    let mut count = black_box(250_000u64);
    let mut value = black_box(0u64);
    let mut offset = black_box(0u64);
    let mut index = 0usize;
    let mut loaded = 0u64;

    while count != 0 {
        index = ((offset & 24) / 8) as usize;
        slots[index] = value;
        loaded = slots[index];
        value = (value ^ loaded).wrapping_add(1);
        offset = offset.wrapping_add(8);
        count = count.wrapping_sub(1);
    }

    black_box((slots, count, value, offset, index, loaded));
}

fn host_mulhsu(lhs: u64, rhs: u64) -> u64 {
    (((lhs as i64 as i128) * (rhs as u128 as i128)) >> 64) as u64
}

fn fibonacci_region_cpu() -> RV64GC {
    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.load_bin(fibonacci_recurrence_loop_bin());
    reset_fibonacci_region_registers(&mut cpu);
    cpu
}

fn counted_diamond_region_cpu() -> RV64GC {
    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.load_bin(counted_diamond_mask3_loop_bin());
    reset_counted_diamond_region_registers(&mut cpu);
    cpu
}

fn store_load_forward_region_cpu() -> RV64GC {
    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.load_bin(store_load_forward_loop_bin());
    cpu.ram
        .add_region(MemoryRegion::new(0x1000, 64, vec![0; 64]))
        .unwrap();
    reset_store_load_forward_region_registers(&mut cpu);
    cpu
}

fn division_region_cpu() -> RV64GC {
    let mut cpu = RV64GC::new();
    cpu.filesystem.set_output_mirroring(false);
    cpu.load_bin(division_region_loop_bin());
    reset_division_region_registers(&mut cpu);
    cpu
}

fn reset_fibonacci_region_registers(cpu: &mut RV64GC) {
    cpu.registers[32usize] = 0;
    cpu.registers[11usize] = 1;
    cpu.registers[12usize] = 5_000_000;
    cpu.registers[13usize] = 0;
    cpu.registers[14usize] = 1;
    cpu.registers[15usize] = 1;
}

fn reset_counted_diamond_region_registers(cpu: &mut RV64GC) {
    cpu.registers[32usize] = 0;
    cpu.registers[5usize] = 5_000_000;
    cpu.registers[6usize] = 0;
    cpu.registers[7usize] = 0;
    cpu.registers[28usize] = 0;
}

fn reset_store_load_forward_region_registers(cpu: &mut RV64GC) {
    cpu.registers[32usize] = 0;
    cpu.registers[5usize] = 250_000;
    cpu.registers[6usize] = 0;
    cpu.registers[7usize] = 0;
    cpu.registers[28usize] = 0x1000;
}

fn reset_division_region_registers(cpu: &mut RV64GC) {
    cpu.registers[32usize] = 0;
    cpu.registers[5usize] = 50_000;
    cpu.registers[10usize] = (-123_456_789i64) as u64;
    cpu.registers[11usize] = 37;
    cpu.registers[12usize] = 0xabcd_ef01_2345_6789;
    cpu.registers[13usize] = 97;
    cpu.registers[20usize] = 0;
}

fn host_direct_runner(name: &str) -> Option<fn()> {
    match name {
        "arith" => Some(run_host_arith),
        "branch" => Some(run_host_branch),
        "memory" => Some(run_host_memory),
        "fibonacci-core" => Some(run_host_fibonacci_core),
        _ => None,
    }
}

fn run_host_arith() {
    let mut count = black_box(500_000u64);
    let mut value = black_box(1u64);
    let mut accumulator = 0u64;

    while count != 0 {
        accumulator = accumulator.wrapping_add(value);
        value ^= 0x5a5;
        count = count.wrapping_sub(1);
    }

    black_box((accumulator, value, count));
}

fn run_host_branch() {
    let mut count = black_box(500_000u64);
    let mut accumulator = 0u64;
    let mut iterations = 0u64;

    while count != 0 {
        if count & 1 == 0 {
            accumulator = accumulator.wrapping_add(7);
        } else {
            accumulator = accumulator.wrapping_add(3);
        }
        iterations = iterations.wrapping_add(1);
        count = count.wrapping_sub(1);
    }

    black_box((accumulator, iterations, count));
}

fn run_host_memory() {
    let mut buffer = [0u64; 256];
    let mut count = black_box(250_000u64);
    let mut value = 0u64;
    let mut offset = 0u64;

    while count != 0 {
        let index = ((offset & 2040) / 8) as usize;
        buffer[index] = value;
        let loaded = buffer[index];
        value ^= loaded;
        value = value.wrapping_add(1);
        offset = offset.wrapping_add(8);
        count = count.wrapping_sub(1);
    }

    black_box((buffer, value, offset, count));
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

#[cfg(all(target_arch = "aarch64", unix))]
fn bench_case_cached_engine(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    engine: &'static str,
    case: BenchBinary,
    options: JitOptions,
) {
    let mut jit = JitEngine::with_options(options).unwrap();
    group.bench_function(BenchmarkId::new(engine, case.name), move |b| {
        b.iter_batched(
            || case.cpu(),
            |mut cpu| {
                jit.run(&mut cpu).unwrap();
                black_box(cpu.registers[10]);
                black_box(cpu.stdout().len());
            },
            BatchSize::SmallInput,
        );
    });
}

#[cfg(not(all(target_arch = "aarch64", unix)))]
fn bench_case_cached_engine(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    engine: &'static str,
    case: BenchBinary,
    _options: JitOptions,
) {
    eprintln!(
        "skipping cached {engine}/{} benchmark: JIT requires Unix AArch64",
        case.name
    );
}

#[cfg(all(target_arch = "aarch64", unix))]
fn bench_trace_load_cached_engine(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    options: JitOptions,
) {
    bench_trace_load_cached_engine_with(group, name, options, trace_load_cpu);
}

#[cfg(all(target_arch = "aarch64", unix))]
fn bench_trace_load_cached_engine_with(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    options: JitOptions,
    setup: fn() -> RV64GC,
) {
    let mut jit = JitEngine::with_options(options).unwrap();
    group.bench_function(name, move |b| {
        b.iter_batched(
            setup,
            |mut cpu| {
                jit.run(&mut cpu).unwrap();
                black_box(cpu.registers[10]);
                black_box(cpu.stdout().len());
            },
            BatchSize::SmallInput,
        );
    });
}

#[cfg(not(all(target_arch = "aarch64", unix)))]
fn bench_trace_load_cached_engine(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    _options: JitOptions,
) {
    eprintln!("skipping cached {name} benchmark: JIT requires Unix AArch64");
}

#[cfg(not(all(target_arch = "aarch64", unix)))]
fn bench_trace_load_cached_engine_with(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    _options: JitOptions,
    _setup: fn() -> RV64GC,
) {
    eprintln!("skipping cached {name} benchmark: JIT requires Unix AArch64");
}

#[cfg(all(target_arch = "aarch64", unix))]
fn bench_fibonacci_region_cached_engine(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    options: JitOptions,
) {
    let mut jit = JitEngine::with_options(options).unwrap();
    group.bench_function(name, move |b| {
        b.iter_batched(
            fibonacci_region_cpu,
            |mut cpu| {
                jit.step(&mut cpu).unwrap();
                black_box(cpu.registers[12]);
                black_box(cpu.registers[13]);
                black_box(cpu.registers[14]);
                black_box(cpu.registers[15]);
            },
            BatchSize::SmallInput,
        );
    });
}

#[cfg(not(all(target_arch = "aarch64", unix)))]
fn bench_fibonacci_region_cached_engine(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    _options: JitOptions,
) {
    eprintln!("skipping cached {name} benchmark: JIT requires Unix AArch64");
}

#[cfg(all(target_arch = "aarch64", unix))]
fn bench_counted_diamond_region_cached_engine(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    options: JitOptions,
) {
    let mut jit = JitEngine::with_options(options).unwrap();
    group.bench_function(name, move |b| {
        b.iter_batched(
            counted_diamond_region_cpu,
            |mut cpu| {
                jit.step(&mut cpu).unwrap();
                black_box(cpu.registers[5]);
                black_box(cpu.registers[6]);
                black_box(cpu.registers[7]);
                black_box(cpu.registers[28]);
            },
            BatchSize::SmallInput,
        );
    });
}

#[cfg(not(all(target_arch = "aarch64", unix)))]
fn bench_counted_diamond_region_cached_engine(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    _options: JitOptions,
) {
    eprintln!("skipping cached {name} benchmark: JIT requires Unix AArch64");
}

#[cfg(all(target_arch = "aarch64", unix))]
fn bench_store_load_forward_region_cached_engine(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    options: JitOptions,
) {
    let mut jit = JitEngine::with_options(options).unwrap();
    group.bench_function(name, move |b| {
        b.iter_batched(
            store_load_forward_region_cpu,
            |mut cpu| {
                jit.step(&mut cpu).unwrap();
                black_box(cpu.registers[6]);
                black_box(cpu.registers[7]);
                black_box(cpu.registers[29]);
                black_box(cpu.registers[30]);
                black_box(cpu.registers[31]);
            },
            BatchSize::SmallInput,
        );
    });
}

#[cfg(not(all(target_arch = "aarch64", unix)))]
fn bench_store_load_forward_region_cached_engine(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    _options: JitOptions,
) {
    eprintln!("skipping cached {name} benchmark: JIT requires Unix AArch64");
}

#[cfg(all(target_arch = "aarch64", unix))]
fn bench_division_region_cached_engine(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    options: JitOptions,
) {
    let mut jit = JitEngine::with_options(options).unwrap();
    group.bench_function(name, move |b| {
        b.iter_batched(
            division_region_cpu,
            |mut cpu| {
                jit.step(&mut cpu).unwrap();
                black_box(cpu.registers[20]);
                black_box(cpu.registers[32]);
            },
            BatchSize::SmallInput,
        );
    });
}

#[cfg(not(all(target_arch = "aarch64", unix)))]
fn bench_division_region_cached_engine(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &'static str,
    _options: JitOptions,
) {
    eprintln!("skipping cached {name} benchmark: JIT requires Unix AArch64");
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

fn run_jit_without_trace(mut cpu: RV64GC) {
    cpu.start_jit_with_options(JitOptions {
        execution_mode: JitExecutionMode::Jit,
        trace_compilation: false,
        ..JitOptions::default()
    })
    .unwrap();
    black_box(cpu.registers[10]);
    black_box(cpu.stdout().len());
}

fn run_trace_jit(mut cpu: RV64GC) {
    cpu.start_jit_with_options(trace_jit_options()).unwrap();
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

fn trace_load_branch_bin() -> Vec<u8> {
    let mut bin = Vec::new();
    bin.extend(rv64_word(rv64_i(0x03, 3, 7, 10, 0))); // ld x7, 0(x10)
    bin.extend(rv64_word(rv64_beq(7, 0, 16))); // beq x7, x0, exit
    bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 1))); // addi x6, x6, 1
    bin.extend(rv64_word(rv64_i(0x13, 0, 10, 10, 8))); // addi x10, x10, 8
    bin.extend(rv64_word(rv64_bne(6, 5, 0x1ff0))); // bne x6, x5, loop
    bin.extend(rv64_word(rv64_i(0x13, 7, 10, 6, 0))); // andi a0, x6, 0
    bin.extend(rv64_word(rv64_i(0x13, 0, 17, 0, 93))); // addi a7, x0, 93
    bin.extend(rv64_word(0x0000_0073)); // ecall
    bin
}

fn trace_load_byte_branch_bin() -> Vec<u8> {
    let mut bin = Vec::new();
    bin.extend(rv64_word(rv64_i(0x03, 4, 7, 10, 0))); // lbu x7, 0(x10)
    bin.extend(rv64_word(rv64_beq(7, 0, 16))); // beq x7, x0, exit
    bin.extend(rv64_word(rv64_i(0x13, 0, 6, 6, 1))); // addi x6, x6, 1
    bin.extend(rv64_word(rv64_i(0x13, 0, 10, 10, 1))); // addi x10, x10, 1
    bin.extend(rv64_word(rv64_bne(6, 5, 0x1ff0))); // bne x6, x5, loop
    bin.extend(rv64_word(rv64_i(0x13, 7, 10, 6, 0))); // andi a0, x6, 0
    bin.extend(rv64_word(rv64_i(0x13, 0, 17, 0, 93))); // addi a7, x0, 93
    bin.extend(rv64_word(0x0000_0073)); // ecall
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

fn division_region_loop_bin() -> Vec<u8> {
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

fn rv64_word(value: u32) -> [u8; 4] {
    value.to_le_bytes()
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

fn rv64_s(funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let imm = imm as u32;
    ((imm >> 5) & 0x7f) << 25
        | (u32::from(rs2) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | 0x23
}

fn rv64_jal(rd: u8, offset: u32) -> u32 {
    ((offset >> 20) & 0x1) << 31
        | ((offset >> 1) & 0x3ff) << 21
        | ((offset >> 11) & 0x1) << 20
        | ((offset >> 12) & 0xff) << 12
        | (u32::from(rd) << 7)
        | 0x6f
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

fn trace_jit_options() -> JitOptions {
    JitOptions {
        execution_mode: JitExecutionMode::Jit,
        trace_compilation: true,
        tier_budgeting: false,
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
