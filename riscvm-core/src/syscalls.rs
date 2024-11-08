use crate::cpu::RV64GCRegAbiName::*;
use crate::cpu::RV64GC;
use crate::ram::MemoryRegion;
use rand::Rng;
use tracing::error;
use tracing::info;
use tracing::span;
use tracing::trace;
use tracing::warn;
use tracing::Level;

pub fn writev(cpu: &mut RV64GC) {
    #[derive(Debug)]
    struct Iovec {
        iov_base: u64,
        iov_len: u64,
    }

    trace!("writev");

    let mut total_bytes_written = 0;
    let fd = cpu.registers[A0];
    trace!("fd: {fd}");
    let iovec_ptr = cpu.registers[A1];
    let iovec_cnt = cpu.registers[A2];

    for i in 0..iovec_cnt {
        let lptr = iovec_ptr + (16 * i);
        let iov_base = cpu.ram.read_doubleword(lptr).unwrap();
        let iov_len = cpu.ram.read_doubleword(lptr + 8).unwrap();
        let iov = Iovec { iov_base, iov_len };

        let mut val = vec![];

        if iov_len > 0 {
            for j in iov.iov_base..iov.iov_base + iov.iov_len {
                val.push(cpu.ram.read_byte(j).unwrap());
                total_bytes_written += 1;
            }
        } else {
            let mut base_addr = iov.iov_base;
            while let Ok(b) = cpu.ram.read_byte(base_addr) {
                if b != 0 {
                    val.push(b);
                    base_addr += 1;
                } else {
                    break;
                }
            }
        }

        if let Ok(s) = String::from_utf8(val) {
            match fd {
                2 => eprint!("\x1b[31m{s}\x1b[0m"),
                _ => {
                    print!("{s}");
                }
            }
        }
    }

    trace!("wrote {total_bytes_written} bytes");

    cpu.registers[A0] = total_bytes_written
}

pub fn mmap(cpu: &mut RV64GC) {
    let span = span!(Level::TRACE, "mmap");
    let _guard = span.enter();

    let addr = cpu.registers[A0];
    let len = cpu.registers[A1];
    let prot = cpu.registers[A2] as i64;
    let flags = cpu.registers[A3] as i64;
    let fd = cpu.registers[A4] as i64;
    let offset = cpu.registers[A5];

    trace!("mmap\n\taddr: {addr}\n\tlen: {len}\n\tprot: {prot}\n\tflags: {flags}\n\tfd: {fd}\n\toffset: {offset}");

    if len == 0 {
        panic!("mmap called with no len!")
    }

    let mmap_addr = if addr == 0 {
        cpu.ram.lowest_unalloced_addr + 16
    } else {
        addr
    };

    let map = MemoryRegion::new(mmap_addr, len, vec![0u8; len as usize]);
    let res = cpu.ram.add_region(map);

    trace!("mmap_addr: {mmap_addr:08x}");

    if res.is_ok() {
        cpu.registers[A0] = mmap_addr;
    } else {
        cpu.registers[A0] = u64::MAX;
    }
}

pub fn brk(cpu: &mut RV64GC) {
    info!("brk");
    let addr = cpu.registers[A0];

    if addr == 0 {
        cpu.registers[A0] = cpu.ram.find_end_of_text_region();
        return;
    }

    if cpu.ram.extend_text_region_to(addr).is_ok() {
        cpu.registers[A0] = 0;
    } else {
        error!("brk failed: 0x:{addr:08x}");
        cpu.registers[A0] = u64::MAX;
    }
}

// 278
pub fn getrandom(cpu: &mut RV64GC) {
    let mut rng = rand::thread_rng();
    let addr = cpu.registers[A0];
    let len = cpu.registers[A1];
    let _flags = cpu.registers[A2];

    for i in 0..len {
        if cpu.ram.write_byte(addr + i, rng.gen()).is_err() {
            warn!("getrandom failed!");
            cpu.registers[A0] = u64::MAX;
            return;
        }
    }

    cpu.registers[A0] = 0;
}

// 172
pub fn getpid(cpu: &mut RV64GC) {
    cpu.registers[A0] = std::process::id().into();
}

// 178
pub fn gettid(cpu: &mut RV64GC) {
    getpid(cpu);
}

// 113
pub fn clock_gettime(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}

// 135
pub fn rt_sigprocmask(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}

// 131
pub fn tgkll(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}

// 261
pub fn prlimit64(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}

// 134
pub fn sig_action(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}

// 78
pub fn readlink(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}

// 226
pub fn mprotect(cpu: &mut RV64GC) {
    cpu.registers[A0] = 0;
}

// 80
pub fn lstat(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}
