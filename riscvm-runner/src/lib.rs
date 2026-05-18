use std::{io::Read, path::PathBuf};

use riscvm_core::cpu::RV64GC;
use riscvm_core::jit::{JitExecutionMode, JitOptions};
use riscvm_core::tracer::TraceOptions;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::EnvFilter;

pub fn run_from_env() {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_env_filter(EnvFilter::from_default_env())
        .with_ansi(true)
        .without_time()
        .init();

    run(std::env::args().skip(1));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunnerEngineMode {
    Interpreter,
    Jit,
    Hybrid,
    Aot,
}

impl RunnerEngineMode {
    fn jit_execution_mode(self) -> Option<JitExecutionMode> {
        match self {
            Self::Interpreter => None,
            Self::Jit => Some(JitExecutionMode::Jit),
            Self::Hybrid => Some(JitExecutionMode::Hybrid),
            Self::Aot => Some(JitExecutionMode::Aot),
        }
    }
}

fn activate_jit_flag(engine_mode: &mut RunnerEngineMode, engine_mode_explicit: bool) {
    if !engine_mode_explicit {
        *engine_mode = RunnerEngineMode::Jit;
    }
}

fn activate_aot_flag(engine_mode: &mut RunnerEngineMode, engine_mode_explicit: bool) {
    if !engine_mode_explicit {
        *engine_mode = RunnerEngineMode::Aot;
    }
}

fn default_engine_mode() -> RunnerEngineMode {
    if cfg!(all(target_arch = "aarch64", unix)) {
        RunnerEngineMode::Hybrid
    } else {
        RunnerEngineMode::Interpreter
    }
}

pub fn run(args: impl IntoIterator<Item = String>) {
    let mut engine_mode = default_engine_mode();
    let mut engine_mode_explicit = false;
    let mut jit_log = false;
    let mut jit_dump = false;
    let default_jit_options = JitOptions::default();
    let mut jit_host_libc = default_jit_options.host_libc;
    let mut jit_libc_start_main_shortcut = default_jit_options.libc_start_main_shortcut;
    let mut jit_dynamic_recompilation = default_jit_options.dynamic_recompilation;
    let mut jit_trace_compilation = default_jit_options.trace_compilation;
    let mut jit_hot_threshold = default_jit_options.hot_threshold;
    let mut jit_tier_budgeting = default_jit_options.tier_budgeting;
    let mut jit_non_loop_hot_threshold_multiplier =
        default_jit_options.non_loop_hot_threshold_multiplier;
    let mut jit_min_optimized_block_instructions =
        default_jit_options.min_optimized_block_instructions;
    let mut jit_background_compilation = default_jit_options.background_compilation;
    let mut jit_compiler_threads = default_jit_options.compiler_threads;
    let mut jit_compile_queue_limit = default_jit_options.compile_queue_limit;
    let mut aot_compile_misses = default_jit_options.aot_compile_misses;
    let mut aot_symbol_entries = default_jit_options.aot_symbol_entries;
    let mut aot_linear_sweep = default_jit_options.aot_linear_sweep;
    let mut trace_options = TraceOptions::default();
    let mut file_path = None;
    let mut guest_args = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if file_path.is_some() {
            guest_args.push(arg);
            guest_args.extend(args);
            break;
        }

        match arg.as_str() {
            "--interp" | "--interpreter" => {
                engine_mode = RunnerEngineMode::Interpreter;
                engine_mode_explicit = true;
            }
            "--hybrid" => {
                engine_mode = RunnerEngineMode::Hybrid;
                engine_mode_explicit = true;
            }
            "--jit" => {
                engine_mode = RunnerEngineMode::Jit;
                engine_mode_explicit = true;
            }
            "--aot" => {
                engine_mode = RunnerEngineMode::Aot;
                engine_mode_explicit = true;
            }
            "--jit-log" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_log = true;
            }
            "--jit-dump" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dump = true;
            }
            "--jit-host-libc" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_host_libc = true;
            }
            "--jit-no-host-libc" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_host_libc = false;
            }
            "--jit-libc-start-main-shortcut" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_host_libc = true;
                jit_libc_start_main_shortcut = true;
            }
            "--jit-no-libc-start-main-shortcut" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_libc_start_main_shortcut = false;
            }
            "--jit-dynarec" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
            }
            "--jit-no-dynarec" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = false;
            }
            "--jit-trace" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_trace_compilation = true;
            }
            "--jit-no-trace" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_trace_compilation = false;
            }
            "--jit-no-tier-budget" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_tier_budgeting = false;
            }
            "--aot-linear-sweep" => {
                activate_aot_flag(&mut engine_mode, engine_mode_explicit);
                aot_linear_sweep = true;
            }
            "--aot-no-linear-sweep" => {
                activate_aot_flag(&mut engine_mode, engine_mode_explicit);
                aot_linear_sweep = false;
            }
            "--aot-compile-misses" => {
                activate_aot_flag(&mut engine_mode, engine_mode_explicit);
                aot_compile_misses = true;
            }
            "--aot-no-compile-misses" => {
                activate_aot_flag(&mut engine_mode, engine_mode_explicit);
                aot_compile_misses = false;
            }
            "--aot-symbol-entries" => {
                activate_aot_flag(&mut engine_mode, engine_mode_explicit);
                aot_symbol_entries = true;
            }
            "--aot-no-symbol-entries" => {
                activate_aot_flag(&mut engine_mode, engine_mode_explicit);
                aot_symbol_entries = false;
            }
            "--jit-non-loop-hot-multiplier" => {
                let Some(value) = args.next() else {
                    eprintln!("--jit-non-loop-hot-multiplier requires a value");
                    std::process::exit(2);
                };
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_non_loop_hot_threshold_multiplier =
                    parse_u64_flag("--jit-non-loop-hot-multiplier", &value);
            }
            arg if arg.starts_with("--jit-non-loop-hot-multiplier=") => {
                let value = arg.trim_start_matches("--jit-non-loop-hot-multiplier=");
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_non_loop_hot_threshold_multiplier =
                    parse_u64_flag("--jit-non-loop-hot-multiplier", value);
            }
            "--jit-min-optimized-block-instructions" => {
                let Some(value) = args.next() else {
                    eprintln!("--jit-min-optimized-block-instructions requires a value");
                    std::process::exit(2);
                };
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_min_optimized_block_instructions =
                    parse_usize_flag("--jit-min-optimized-block-instructions", &value);
            }
            arg if arg.starts_with("--jit-min-optimized-block-instructions=") => {
                let value = arg.trim_start_matches("--jit-min-optimized-block-instructions=");
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_min_optimized_block_instructions =
                    parse_usize_flag("--jit-min-optimized-block-instructions", value);
            }
            "--jit-hot-threshold" => {
                let Some(value) = args.next() else {
                    eprintln!("--jit-hot-threshold requires a value");
                    std::process::exit(2);
                };
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_hot_threshold = parse_u64_flag("--jit-hot-threshold", &value);
            }
            arg if arg.starts_with("--jit-hot-threshold=") => {
                let value = arg.trim_start_matches("--jit-hot-threshold=");
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_hot_threshold = parse_u64_flag("--jit-hot-threshold", value);
            }
            "--jit-bg-compile" => {
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_background_compilation = true;
            }
            "--jit-compiler-threads" => {
                let Some(value) = args.next() else {
                    eprintln!("--jit-compiler-threads requires a value");
                    std::process::exit(2);
                };
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_background_compilation = true;
                jit_compiler_threads = parse_usize_flag("--jit-compiler-threads", &value);
            }
            arg if arg.starts_with("--jit-compiler-threads=") => {
                let value = arg.trim_start_matches("--jit-compiler-threads=");
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_background_compilation = true;
                jit_compiler_threads = parse_usize_flag("--jit-compiler-threads", value);
            }
            "--jit-compile-queue-limit" => {
                let Some(value) = args.next() else {
                    eprintln!("--jit-compile-queue-limit requires a value");
                    std::process::exit(2);
                };
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_background_compilation = true;
                jit_compile_queue_limit = parse_usize_flag("--jit-compile-queue-limit", &value);
            }
            arg if arg.starts_with("--jit-compile-queue-limit=") => {
                let value = arg.trim_start_matches("--jit-compile-queue-limit=");
                activate_jit_flag(&mut engine_mode, engine_mode_explicit);
                jit_dynamic_recompilation = true;
                jit_background_compilation = true;
                jit_compile_queue_limit = parse_usize_flag("--jit-compile-queue-limit", value);
            }
            "--trace" => trace_options.trace = true,
            "--profile" => trace_options.profile = true,
            "--profile-top" => {
                let Some(value) = args.next() else {
                    eprintln!("--profile-top requires a value");
                    std::process::exit(2);
                };
                trace_options.top_limit = parse_usize_flag("--profile-top", &value);
            }
            arg if arg.starts_with("--profile-top=") => {
                let value = arg.trim_start_matches("--profile-top=");
                trace_options.top_limit = parse_usize_flag("--profile-top", value);
            }
            "--trace-limit" => {
                let Some(value) = args.next() else {
                    eprintln!("--trace-limit requires a value");
                    std::process::exit(2);
                };
                trace_options.trace_limit = Some(parse_u64_flag("--trace-limit", &value));
            }
            arg if arg.starts_with("--trace-limit=") => {
                let value = arg.trim_start_matches("--trace-limit=");
                trace_options.trace_limit = Some(parse_u64_flag("--trace-limit", value));
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            "--" => {
                file_path = args.next();
                guest_args.extend(args);
                break;
            }
            _ => file_path = Some(arg),
        }
    }

    let Some(file_path) = file_path else {
        eprintln!("No binary specified!\n");
        print_usage();
        return;
    };
    let executable_path = absolute_executable_path(&file_path);
    let executable_argv0 = executable_path.to_string_lossy().into_owned();

    let mut bin = Vec::new();
    std::fs::File::open(&file_path)
        .unwrap()
        .read_to_end(&mut bin)
        .unwrap();

    let mut riscvm = RV64GC::new();
    riscvm.set_executable_path(&executable_path);
    let mut argv = vec![executable_argv0];
    argv.extend(guest_args);
    riscvm.set_argv(argv);
    riscvm
        .mount_host_directory(std::env::current_dir().unwrap())
        .unwrap();

    if file_path.ends_with(".bin") {
        riscvm.load_bin(bin);
    } else {
        riscvm.load_elf(bin).unwrap();
    }

    if trace_options.enabled() {
        riscvm.set_trace_options(trace_options);
    }

    if let Some(execution_mode) = engine_mode.jit_execution_mode() {
        let jit_options = JitOptions {
            execution_mode,
            host_libc: jit_host_libc,
            libc_start_main_shortcut: jit_libc_start_main_shortcut,
            debug_log: jit_log,
            dump_instructions: jit_dump,
            dynamic_recompilation: jit_dynamic_recompilation,
            trace_compilation: jit_trace_compilation,
            hot_threshold: jit_hot_threshold,
            tier_budgeting: jit_tier_budgeting,
            non_loop_hot_threshold_multiplier: jit_non_loop_hot_threshold_multiplier,
            min_optimized_block_instructions: jit_min_optimized_block_instructions,
            background_compilation: jit_background_compilation,
            compiler_threads: jit_compiler_threads,
            compile_queue_limit: jit_compile_queue_limit,
            aot_compile_misses,
            aot_symbol_entries,
            aot_linear_sweep,
        };
        if let Err(error) = riscvm.start_jit_with_options(jit_options) {
            eprintln!("{execution_mode:?} engine failed: {error}");
            std::process::exit(1);
        }
    } else {
        riscvm.start();
    }

    if trace_options.profile {
        if let Some(report) = riscvm.trace_report() {
            eprint!("{report}");
        }
    }
}

fn print_usage() {
    let default_engine = match default_engine_mode() {
        RunnerEngineMode::Interpreter => "--interp",
        RunnerEngineMode::Jit => "--jit",
        RunnerEngineMode::Hybrid => "--hybrid",
        RunnerEngineMode::Aot => "--aot",
    };
    eprintln!(
        "Usage: riscvm [--hybrid|--jit|--aot|--interp] [--jit-log] [--jit-dump] [--jit-host-libc] [--jit-no-host-libc] [--jit-libc-start-main-shortcut] [--jit-no-libc-start-main-shortcut] [--jit-dynarec] [--jit-no-dynarec] [--jit-trace] [--jit-no-trace] [--jit-hot-threshold N] [--jit-no-tier-budget] [--aot-linear-sweep] [--aot-no-linear-sweep] [--aot-compile-misses] [--aot-no-compile-misses] [--aot-symbol-entries] [--aot-no-symbol-entries] [--jit-non-loop-hot-multiplier N] [--jit-min-optimized-block-instructions N] [--jit-bg-compile] [--jit-compiler-threads N] [--jit-compile-queue-limit N] [--trace] [--trace-limit N] [--profile] [--profile-top N] <binary> [guest-args...]\nDefault engine: {default_engine}"
    );
}

fn parse_usize_flag(flag: &str, value: &str) -> usize {
    value.parse().unwrap_or_else(|_| {
        eprintln!("{flag} expects a positive integer, got {value:?}");
        std::process::exit(2);
    })
}

fn parse_u64_flag(flag: &str, value: &str) -> u64 {
    value.parse().unwrap_or_else(|_| {
        eprintln!("{flag} expects a positive integer, got {value:?}");
        std::process::exit(2);
    })
}

fn absolute_executable_path(path: &str) -> PathBuf {
    let raw_path = PathBuf::from(path);
    if let Ok(canonical) = std::fs::canonicalize(&raw_path) {
        return canonical;
    }
    if raw_path.is_absolute() {
        return raw_path;
    }

    std::env::current_dir()
        .map(|cwd| cwd.join(raw_path))
        .unwrap_or_else(|_| PathBuf::from("/").join(path))
}

#[cfg(test)]
mod tests {
    use super::{default_engine_mode, RunnerEngineMode};

    #[test]
    fn default_engine_matches_host_backend_availability() {
        let expected = if cfg!(all(target_arch = "aarch64", unix)) {
            RunnerEngineMode::Hybrid
        } else {
            RunnerEngineMode::Interpreter
        };
        assert_eq!(default_engine_mode(), expected);
    }
}
