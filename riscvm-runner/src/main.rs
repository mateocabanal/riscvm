use std::io::Read;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::EnvFilter;

use cpu::RV64GC;
use riscvm_core::*;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_env_filter(EnvFilter::from_default_env())
        .with_ansi(true)
        .without_time()
        .init();

    if std::env::args().len() < 2 {
        eprintln!("No binary specified!\n");
        return;
    }

    let file_path = std::env::args().nth(1).unwrap();
    let mut bin = Vec::new();
    std::fs::File::open(&file_path)
        .unwrap()
        .read_to_end(&mut bin)
        .unwrap();

    let mut riscvm = RV64GC::new();

    if file_path.ends_with(".bin") {
        riscvm.load_bin(bin);
    } else {
        riscvm.load_elf(bin).unwrap();
    }

    riscvm.start();
}
