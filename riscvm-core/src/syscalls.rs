use std::io;
use std::io::Read;
use std::io::Write;

use crate::cpu::RV64GCRegAbiName::*;
use crate::cpu::RV64GC;
use crate::ram::MemoryRegion;
use rand::Rng;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::span;
use tracing::trace;
use tracing::warn;
use tracing::Level;

#[allow(unused, clippy::upper_case_acronyms)]
enum Errno {
    EPERM = 1,
    ENOENT = 2,
    ESRCH = 3,
    EINTR = 4,
    EIO = 5,
    EBADF = 9,
    EAGAIN = 11,
    ENOMEM = 12,
    EACCCES = 13,
    EFAULT = 14,
}

impl From<Errno> for i64 {
    fn from(val: Errno) -> Self {
        val as u64 as i64
    }
}

impl Errno {
    pub fn into_err(self) -> u64 {
        let sval: i64 = self.into();
        -sval as u64
    }
}

#[derive(Debug)]
enum SeekMode {
    Set = 0,
    Cur = 1,
    End = 2,
}

impl From<i64> for SeekMode {
    fn from(value: i64) -> Self {
        match value {
            0 => SeekMode::Set,
            1 => SeekMode::Cur,
            2 => SeekMode::End,
            _ => panic!("invalid seek mode!"),
        }
    }
}

impl From<SeekMode> for i64 {
    fn from(value: SeekMode) -> Self {
        value as i64
    }
}

// 63
pub fn read(cpu: &mut RV64GC) {
    let span = span!(Level::TRACE, "syscall_read");
    let _guard = span.enter();

    let fd = cpu.registers[A0];
    let buf = cpu.registers[A1];
    let count = cpu.registers[A2];

    debug!("fd: {fd}");
    debug!("buf: 0x{buf:08x}");
    debug!("count: {count}");

    if count == 0 {
        cpu.registers[A0] = 0;
        return;
    }

    if fd != 0 {
        panic!("riscvm only supports the read syscall with the file descriptor 0 (stdin)");
    }

    let mut buffer = vec![0u8; count as usize];

    if fd == 0 {
        let bytes_read = match std::io::stdin().read(&mut buffer) {
            Ok(n) => n,
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {
                // Read was interrupted
                cpu.registers[A0] = Errno::EINTR.into_err();
                return;
            }
            Err(_) => {
                // Other I/O error
                cpu.registers[A0] = Errno::EIO.into_err();
                return;
            }
        };

        for (idx, b) in buffer.into_iter().enumerate() {
            if cpu.ram.write_byte(buf + idx as u64, b).is_err() {
                cpu.registers[A0] = Errno::EFAULT.into_err();
                return;
            }
        }

        cpu.registers[A0] = bytes_read as u64;
    }
}

// 64
pub fn write(cpu: &mut RV64GC) {
    let span = span!(Level::TRACE, "syscall_write");
    let _guard = span.enter();

    debug!("write");

    let ptr = cpu.registers[A1];
    let len = cpu.registers[A2];

    trace!("ptr: {ptr:#08x}");
    trace!("len: {len}");

    let msg = if len != 0 {
        (ptr..ptr + len)
            .map(|addr| {
                cpu.ram
                    .read_byte(addr)
                    .inspect_err(|e| panic!("{e}"))
                    .unwrap() as char
            })
            .collect::<String>()
    } else {
        let mut i = 0;
        let mut string = String::new();
        while let Ok(c) = cpu.ram.read_byte(i).map(|i| i as char) {
            if c == '\0' {
                break;
            }

            string.push(c);
            i += 1;
        }

        string
    };

    print!("{msg}");

    std::io::stdout().flush().unwrap();
    cpu.registers[A0] = len;
}

// 66
pub fn writev(cpu: &mut RV64GC) {
    #[derive(Debug)]
    struct Iovec {
        iov_base: u64,
        iov_len: u64,
    }

    debug!("writev");

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

    std::io::stdout().flush().unwrap();
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

    debug!("mmap");
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
    debug!("brk");
    let addr = cpu.registers[A0];
    if addr == 0 {
        cpu.registers[A0] = cpu.ram.find_end_of_text_region();
        return;
    }

    if cpu.ram.extend_text_region_to(addr).is_ok() {
        cpu.registers[A0] = 0;
    } else {
        error!("brk failed: 0x:{addr:08x}");
        cpu.registers[A0] = Errno::ENOMEM.into_err();
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
pub fn tgkill(cpu: &mut RV64GC) {
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

// 258
pub fn riscv_hwprobe(cpu: &mut RV64GC) {
    cpu.registers[A0] = u64::MAX;
}

// 62
pub fn lseek(cpu: &mut RV64GC) {
    let span = span!(Level::TRACE, "lseek");
    let _guard = span.enter();

    let fd = cpu.registers[A0] as i64;
    let offset = cpu.registers[A1] as i64;
    let whence = cpu.registers[A2] as i64;
    let seek_mode = SeekMode::from(whence);

    trace!("lseek");
    trace!("fd: {fd}");
    trace!("offset: {offset}");
    trace!("seek mode: {seek_mode:?}");

    if fd == 0 {
        warn!("lseek on stdin is not supported!");
        cpu.registers[A0] = Errno::EBADF.into_err();
    } else {
        warn!("lseek is not yet implemented!");
        cpu.registers[A0] = Errno::EBADF.into_err();
    }
}

// 98
// https://www.man7.org/linux/man-pages/man2/futex.2.html
// NOTE: Just return FUTEX_WAIT for now
pub fn futex(cpu: &mut RV64GC) {
    cpu.registers[A0] = 0;
}
