use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cpu::RV64GCRegAbiName::*;
use crate::cpu::RV64GC;
use crate::filesystem::{FileMetadata, FileSystemError, SeekWhence};
use crate::ram::PAGE_SIZE;
use rand::Rng;
use tracing::debug;
use tracing::error;
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
    EACCES = 13,
    EFAULT = 14,
    EEXIST = 17,
    ENOTDIR = 20,
    EISDIR = 21,
    EINVAL = 22,
    EMFILE = 24,
    ENOTTY = 25,
    ESPIPE = 29,
    ERANGE = 34,
    ENOSYS = 38,
    EOVERFLOW = 75,
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

const AT_FDCWD: i64 = -100;
const AT_EMPTY_PATH: u64 = 0x1000;
const UTS_FIELD_LEN: u64 = 65;
const IOV_MAX: u64 = 1024;
const O_CLOEXEC: u64 = 0o2000000;
const PROT_EXEC: i64 = 0x4;
const MAP_FIXED: i64 = 0x10;
const MAP_ANONYMOUS: i64 = 0x20;
const EMULATED_CPU_COUNT: u64 = 1;
const CLONE_VM: u64 = 0x0000_0100;
const CLONE_PARENT_SETTID: u64 = 0x0010_0000;
const CLONE_CHILD_CLEARTID: u64 = 0x0020_0000;
const CLONE_CHILD_SETTID: u64 = 0x0100_0000;
const FIRST_SYNTHETIC_TID: u64 = 10_000;
const CLONE_ARGS_FLAGS_OFFSET: u64 = 0;
const CLONE_ARGS_CHILD_TID_OFFSET: u64 = 16;
const CLONE_ARGS_PARENT_TID_OFFSET: u64 = 24;
const CLONE_ARGS_STACK_OFFSET: u64 = 40;
const CLONE_ARGS_STACK_SIZE_OFFSET: u64 = 48;
const CLONE_ARGS_TLS_OFFSET: u64 = 56;

static NEXT_SYNTHETIC_TID: AtomicU64 = AtomicU64::new(FIRST_SYNTHETIC_TID);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyntheticThreadRun {
    Completed,
    Yielded,
}

fn read_c_string(cpu: &RV64GC, ptr: u64) -> Result<String, Errno> {
    let mut bytes = Vec::new();
    for offset in 0..4096 {
        let byte = cpu.ram.read_byte(ptr + offset).map_err(|_| Errno::EFAULT)?;
        if byte == 0 {
            return String::from_utf8(bytes).map_err(|_| Errno::EFAULT);
        }
        bytes.push(byte);
    }

    Err(Errno::EFAULT)
}

fn write_bytes(cpu: &mut RV64GC, ptr: u64, bytes: &[u8]) -> Result<(), Errno> {
    if bytes.is_empty() {
        return Ok(());
    }
    if let Ok(target) = cpu.ram.write_slice_range(ptr, bytes.len() as u64) {
        target.copy_from_slice(bytes);
        return Ok(());
    }

    for (idx, byte) in bytes.iter().enumerate() {
        let addr = ptr.checked_add(idx as u64).ok_or(Errno::EFAULT)?;
        cpu.ram.write_byte(addr, *byte).map_err(|_| Errno::EFAULT)?;
    }

    Ok(())
}

fn write_c_string(cpu: &mut RV64GC, ptr: u64, value: &str) -> Result<usize, Errno> {
    write_bytes(cpu, ptr, value.as_bytes())?;
    cpu.ram
        .write_byte(ptr + value.len() as u64, 0)
        .map_err(|_| Errno::EFAULT)?;

    Ok(value.len() + 1)
}

fn read_bytes(cpu: &RV64GC, ptr: u64, len: u64) -> Result<Vec<u8>, Errno> {
    let len = usize::try_from(len).map_err(|_| Errno::ENOMEM)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    if let Ok(slice) = cpu.ram.read_slice_range(ptr, len as u64) {
        return Ok(slice.to_vec());
    }

    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|_| Errno::ENOMEM)?;

    for idx in 0..len {
        let addr = ptr.checked_add(idx as u64).ok_or(Errno::EFAULT)?;
        bytes.push(cpu.ram.read_byte(addr).map_err(|_| Errno::EFAULT)?);
    }

    Ok(bytes)
}

fn validate_guest_buffer(cpu: &RV64GC, ptr: u64, len: u64) -> Result<(), Errno> {
    let len = usize::try_from(len).map_err(|_| Errno::ENOMEM)?;
    if len == 0 || cpu.ram.read_slice_range(ptr, len as u64).is_ok() {
        return Ok(());
    }

    for idx in 0..len {
        let addr = ptr.checked_add(idx as u64).ok_or(Errno::EFAULT)?;
        cpu.ram.read_byte(addr).map_err(|_| Errno::EFAULT)?;
    }

    Ok(())
}

fn zeroed_buffer(len: u64) -> Result<Vec<u8>, Errno> {
    let len = usize::try_from(len).map_err(|_| Errno::ENOMEM)?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).map_err(|_| Errno::ENOMEM)?;
    buffer.resize(len, 0);

    Ok(buffer)
}

fn fs_error_to_errno(error: FileSystemError) -> Errno {
    match error {
        FileSystemError::BadFileDescriptor => Errno::EBADF,
        FileSystemError::FileExists => Errno::EEXIST,
        FileSystemError::IllegalSeek => Errno::ESPIPE,
        FileSystemError::InvalidInput => Errno::EINVAL,
        FileSystemError::Io => Errno::EIO,
        FileSystemError::IsDirectory => Errno::EISDIR,
        FileSystemError::NoSuchFile => Errno::ENOENT,
        FileSystemError::PermissionDenied => Errno::EACCES,
        FileSystemError::TooManyOpenFiles => Errno::EMFILE,
    }
}

fn seek_whence(value: u64) -> Result<SeekWhence, Errno> {
    match value {
        0 => Ok(SeekWhence::Set),
        1 => Ok(SeekWhence::Current),
        2 => Ok(SeekWhence::End),
        _ => Err(Errno::EINVAL),
    }
}

fn valid_dirfd_for_path(dirfd: i64, path: &str) -> bool {
    path.starts_with('/') || dirfd == AT_FDCWD
}

fn read_iovecs(cpu: &RV64GC, iovec_ptr: u64, iovec_cnt: u64) -> Result<Vec<(u64, u64)>, Errno> {
    if iovec_cnt > IOV_MAX {
        return Err(Errno::EINVAL);
    }

    let mut iovecs = Vec::new();
    iovecs
        .try_reserve_exact(iovec_cnt as usize)
        .map_err(|_| Errno::ENOMEM)?;

    for i in 0..iovec_cnt {
        let iovec_offset = 16u64.checked_mul(i).ok_or(Errno::EFAULT)?;
        let lptr = iovec_ptr.checked_add(iovec_offset).ok_or(Errno::EFAULT)?;
        let iov_base = cpu.ram.read_doubleword(lptr).map_err(|_| Errno::EFAULT)?;
        let iov_len = cpu
            .ram
            .read_doubleword(lptr.checked_add(8).ok_or(Errno::EFAULT)?)
            .map_err(|_| Errno::EFAULT)?;
        iovecs.push((iov_base, iov_len));
    }

    Ok(iovecs)
}

fn write_guest_tid(cpu: &mut RV64GC, ptr: u64, tid: u64) -> Result<(), Errno> {
    if ptr == 0 {
        return Ok(());
    }
    cpu.ram
        .write_word(ptr, tid as u32)
        .map_err(|_| Errno::EFAULT)
}

fn read_clone_arg(cpu: &RV64GC, args: u64, size: u64, offset: u64) -> Result<u64, Errno> {
    if args == 0 {
        return Err(Errno::EFAULT);
    }
    if size < offset + 8 {
        return Ok(0);
    }

    cpu.ram
        .read_doubleword(args + offset)
        .map_err(|_| Errno::EFAULT)
}

fn enqueue_synthetic_clone(
    cpu: &mut RV64GC,
    flags: u64,
    stack: u64,
    parent_tid: u64,
    tls: u64,
    child_tid: u64,
) -> Result<u64, Errno> {
    if flags & CLONE_VM == 0 {
        warn!("clone without CLONE_VM is not supported: flags=0x{flags:x}");
        return Err(Errno::ENOSYS);
    }

    let tid = NEXT_SYNTHETIC_TID.fetch_add(1, Ordering::Relaxed);
    if flags & CLONE_PARENT_SETTID != 0 {
        write_guest_tid(cpu, parent_tid, tid)?;
    }
    if flags & CLONE_CHILD_SETTID != 0 {
        write_guest_tid(cpu, child_tid, tid)?;
    }

    let mut child = cpu.clone();
    child.set_thread_id(tid);
    child.set_synthetic_thread(true);
    child.should_quit = false;
    child.registers[A0] = 0;
    child.registers[Pc] = child.registers[Pc].wrapping_add(4);
    if stack != 0 {
        child.registers[Sp] = stack;
    }
    if tls != 0 {
        child.registers[Tp] = tls;
    }
    if flags & CLONE_CHILD_SETTID != 0 {
        write_guest_tid(&mut child, child_tid, tid)?;
    }
    child.set_clear_child_tid((flags & CLONE_CHILD_CLEARTID != 0).then_some(child_tid));

    cpu.enqueue_synthetic_thread(child);
    Ok(tid)
}

fn run_synthetic_thread(child: &mut RV64GC) -> Result<SyntheticThreadRun, Errno> {
    child.prepare_synthetic_thread_run();
    if let Some(options) = child.synthetic_thread_jit_options() {
        child.start_jit_with_options(options).map_err(|error| {
            child.set_jit_runtime_fault(format!("synthetic clone thread JIT failed: {error}"));
            Errno::EFAULT
        })?;
    } else {
        child.start();
    }
    run_pending_synthetic_threads(child)?;
    if child.synthetic_thread_yielded() {
        return Ok(SyntheticThreadRun::Yielded);
    }
    if let Some(clear_child_tid) = child.clear_child_tid() {
        write_guest_tid(child, clear_child_tid, 0)?;
    }

    Ok(SyntheticThreadRun::Completed)
}

fn run_pending_synthetic_threads(cpu: &mut RV64GC) -> Result<(), Errno> {
    let mut yielded_threads = Vec::new();
    for mut child in cpu.take_synthetic_threads() {
        match run_synthetic_thread(&mut child)? {
            SyntheticThreadRun::Completed => cpu.merge_synthetic_thread(child),
            SyntheticThreadRun::Yielded => {
                cpu.merge_synthetic_thread_trace(&mut child);
                yielded_threads.push(child);
            }
        }
    }
    for child in yielded_threads {
        cpu.enqueue_synthetic_thread(child);
    }
    Ok(())
}

pub(crate) fn run_ready_synthetic_threads(cpu: &mut RV64GC) {
    cpu.tick_synthetic_threads();
    if !cpu.synthetic_threads_ready() {
        return;
    }
    if let Err(errno) = run_pending_synthetic_threads(cpu) {
        let result = errno.into_err();
        cpu.set_jit_runtime_fault(format!(
            "synthetic clone thread scheduler failed with result {result}"
        ));
    }
}

fn write_stat(cpu: &mut RV64GC, statbuf: u64, metadata: FileMetadata) -> Result<(), Errno> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let sec = now.as_secs();
    let nsec = u64::from(now.subsec_nanos());

    cpu.ram
        .write_doubleword(statbuf, metadata.dev)
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 8, metadata.ino))
        .and_then(|_| cpu.ram.write_word(statbuf + 16, metadata.mode))
        .and_then(|_| cpu.ram.write_word(statbuf + 20, 1))
        .and_then(|_| cpu.ram.write_word(statbuf + 24, 1000))
        .and_then(|_| cpu.ram.write_word(statbuf + 28, 1000))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 32, 0))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 40, 0))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 48, metadata.size))
        .and_then(|_| cpu.ram.write_word(statbuf + 56, metadata.block_size))
        .and_then(|_| cpu.ram.write_word(statbuf + 60, 0))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 64, metadata.blocks))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 72, sec))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 80, nsec))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 88, sec))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 96, nsec))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 104, sec))
        .and_then(|_| cpu.ram.write_doubleword(statbuf + 112, nsec))
        .and_then(|_| cpu.ram.write_word(statbuf + 120, 0))
        .and_then(|_| cpu.ram.write_word(statbuf + 124, 0))
        .map_err(|_| Errno::EFAULT)
}

fn write_uts_field(cpu: &mut RV64GC, addr: u64, value: &str) -> Result<(), Errno> {
    for i in 0..UTS_FIELD_LEN {
        cpu.ram.write_byte(addr + i, 0).map_err(|_| Errno::EFAULT)?;
    }

    let bytes = value.as_bytes();
    let len = bytes.len().min(UTS_FIELD_LEN as usize - 1);
    write_bytes(cpu, addr, &bytes[..len])
}

// 29
pub fn ioctl(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let request = cpu.registers[A1];
    debug!("ioctl: fd {fd}, request 0x{request:x}");

    cpu.registers[A0] = Errno::ENOTTY.into_err();
}

// 17
pub fn getcwd(cpu: &mut RV64GC) {
    let buf = cpu.registers[A0];
    let size = cpu.registers[A1];
    if size < 2 {
        cpu.registers[A0] = Errno::ERANGE.into_err();
        return;
    }

    match write_c_string(cpu, buf, "/") {
        Ok(len) => cpu.registers[A0] = len as u64,
        Err(errno) => cpu.registers[A0] = errno.into_err(),
    }
}

// 23
pub fn dup(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    match cpu.filesystem.duplicate(fd) {
        Ok(new_fd) => cpu.registers[A0] = new_fd,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 24
pub fn dup3(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let new_fd = cpu.registers[A1];
    let flags = cpu.registers[A2];
    if fd == new_fd || flags & !O_CLOEXEC != 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    match cpu.filesystem.duplicate_to(fd, new_fd) {
        Ok(new_fd) => cpu.registers[A0] = new_fd,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 20
pub fn epoll_create1(cpu: &mut RV64GC) {
    let flags = cpu.registers[A0];
    if flags & !O_CLOEXEC != 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    cpu.registers[A0] = Errno::ENOSYS.into_err();
}

// 21
pub fn epoll_ctl(cpu: &mut RV64GC) {
    let _epfd = cpu.registers[A0];
    let _op = cpu.registers[A1];
    let _fd = cpu.registers[A2];
    let event = cpu.registers[A3];
    if event != 0 && validate_guest_buffer(cpu, event, 16).is_err() {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    }

    cpu.registers[A0] = 0;
}

// 22
pub fn epoll_pwait(cpu: &mut RV64GC) {
    let _epfd = cpu.registers[A0];
    let events = cpu.registers[A1];
    let maxevents = cpu.registers[A2] as i64;
    let _timeout = cpu.registers[A3] as i64;
    let _sigmask = cpu.registers[A4];
    let _sigsetsize = cpu.registers[A5];

    if maxevents <= 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }
    if validate_guest_buffer(cpu, events, maxevents as u64 * 16).is_err() {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    }

    cpu.registers[A0] = 0;
}

// 25
pub fn fcntl(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let cmd = cpu.registers[A1];
    let arg = cpu.registers[A2];
    match cpu.filesystem.fcntl(fd, cmd, arg) {
        Ok(value) => cpu.registers[A0] = value,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 61
pub fn getdents64(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let dirp = cpu.registers[A1];
    let count = cpu.registers[A2];
    if count == 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }
    if let Err(errno) = validate_guest_buffer(cpu, dirp, count) {
        cpu.registers[A0] = errno.into_err();
        return;
    }

    match cpu.filesystem.getdents64(fd, count as usize) {
        Ok(bytes) => match write_bytes(cpu, dirp, &bytes) {
            Ok(()) => cpu.registers[A0] = bytes.len() as u64,
            Err(errno) => cpu.registers[A0] = errno.into_err(),
        },
        Err(FileSystemError::BadFileDescriptor) => cpu.registers[A0] = Errno::EBADF.into_err(),
        Err(FileSystemError::InvalidInput) => cpu.registers[A0] = Errno::ENOTDIR.into_err(),
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
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

    if let Err(errno) = validate_guest_buffer(cpu, buf, count) {
        cpu.registers[A0] = errno.into_err();
        return;
    }

    let mut buffer = match zeroed_buffer(count) {
        Ok(buffer) => buffer,
        Err(errno) => {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    };

    let bytes_read = match cpu.filesystem.read(fd, &mut buffer) {
        Ok(bytes_read) => bytes_read,
        Err(error) => {
            cpu.registers[A0] = fs_error_to_errno(error).into_err();
            return;
        }
    };

    if write_bytes(cpu, buf, &buffer[..bytes_read]).is_err() {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    }

    cpu.registers[A0] = bytes_read as u64;
}

// 65
pub fn readv(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let iovec_ptr = cpu.registers[A1];
    let iovec_cnt = cpu.registers[A2];
    let iovecs = match read_iovecs(cpu, iovec_ptr, iovec_cnt) {
        Ok(iovecs) => iovecs,
        Err(errno) => {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    };

    for (iov_base, iov_len) in &iovecs {
        if let Err(errno) = validate_guest_buffer(cpu, *iov_base, *iov_len) {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    }

    let mut total = 0u64;
    for (iov_base, iov_len) in iovecs {
        let mut buffer = match zeroed_buffer(iov_len) {
            Ok(buffer) => buffer,
            Err(errno) => {
                cpu.registers[A0] = errno.into_err();
                return;
            }
        };
        let bytes_read = match cpu.filesystem.read(fd, &mut buffer) {
            Ok(bytes_read) => bytes_read,
            Err(error) => {
                cpu.registers[A0] = fs_error_to_errno(error).into_err();
                return;
            }
        };

        if write_bytes(cpu, iov_base, &buffer[..bytes_read]).is_err() {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
        total += bytes_read as u64;
        if bytes_read < iov_len as usize {
            break;
        }
    }

    cpu.registers[A0] = total;
}

// 67
pub fn pread64(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let buf = cpu.registers[A1];
    let count = cpu.registers[A2];
    let offset = cpu.registers[A3] as i64;
    if offset < 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }
    if let Err(errno) = validate_guest_buffer(cpu, buf, count) {
        cpu.registers[A0] = errno.into_err();
        return;
    }

    let mut buffer = match zeroed_buffer(count) {
        Ok(buffer) => buffer,
        Err(errno) => {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    };
    let bytes_read = match cpu.filesystem.read_at(fd, offset as u64, &mut buffer) {
        Ok(bytes_read) => bytes_read,
        Err(error) => {
            cpu.registers[A0] = fs_error_to_errno(error).into_err();
            return;
        }
    };
    if write_bytes(cpu, buf, &buffer[..bytes_read]).is_err() {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    }

    cpu.registers[A0] = bytes_read as u64;
}

// 68
pub fn pwrite64(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let ptr = cpu.registers[A1];
    let len = cpu.registers[A2];
    let offset = cpu.registers[A3] as i64;
    if offset < 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    let bytes = match read_bytes(cpu, ptr, len) {
        Ok(bytes) => bytes,
        Err(errno) => {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    };

    match cpu.filesystem.write_at(fd, offset as u64, &bytes) {
        Ok(bytes_written) => cpu.registers[A0] = bytes_written as u64,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 56
pub fn openat(cpu: &mut RV64GC) {
    let dirfd = cpu.registers[A0] as i64;
    let pathname = cpu.registers[A1];
    let flags = cpu.registers[A2] as i32;
    let mode = cpu.registers[A3];
    let path = match read_c_string(cpu, pathname) {
        Ok(path) => path,
        Err(_) => {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    };

    debug!("openat: dirfd {dirfd}, path {path}, flags 0x{flags:x}, mode 0o{mode:o}");

    if !path.starts_with('/') && dirfd != AT_FDCWD {
        cpu.registers[A0] = Errno::EBADF.into_err();
        return;
    }

    match cpu.filesystem.open(&path, flags) {
        Ok(fd) => cpu.registers[A0] = fd,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 57
pub fn close(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    debug!("close: fd {fd}");

    match cpu.filesystem.close(fd) {
        Ok(()) => cpu.registers[A0] = 0,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 48
pub fn faccessat(cpu: &mut RV64GC) {
    let dirfd = cpu.registers[A0] as i64;
    let pathname = cpu.registers[A1];
    let mode = cpu.registers[A2] as i32;
    let path = match read_c_string(cpu, pathname) {
        Ok(path) => path,
        Err(_) => {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    };

    if !valid_dirfd_for_path(dirfd, &path) {
        cpu.registers[A0] = Errno::EBADF.into_err();
        return;
    }

    match cpu.filesystem.access(&path, mode) {
        Ok(()) => cpu.registers[A0] = 0,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 64
pub fn write(cpu: &mut RV64GC) {
    let span = span!(Level::TRACE, "syscall_write");
    let _guard = span.enter();

    debug!("write");

    let fd = cpu.registers[A0];
    let ptr = cpu.registers[A1];
    let len = cpu.registers[A2];

    trace!("fd: {fd}");
    trace!("ptr: {ptr:#08x}");
    trace!("len: {len}");

    let bytes = match read_bytes(cpu, ptr, len) {
        Ok(bytes) => bytes,
        Err(errno) => {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    };

    match cpu.filesystem.write(fd, &bytes) {
        Ok(bytes_written) => cpu.registers[A0] = bytes_written as u64,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

fn checked_extend(target: &mut Vec<u8>, bytes: Vec<u8>) -> Result<(), Errno> {
    target
        .try_reserve_exact(bytes.len())
        .map_err(|_| Errno::ENOMEM)?;
    target.extend(bytes);

    Ok(())
}

// 66
pub fn writev(cpu: &mut RV64GC) {
    debug!("writev");

    let fd = cpu.registers[A0];
    trace!("fd: {fd}");
    let iovec_ptr = cpu.registers[A1];
    let iovec_cnt = cpu.registers[A2];

    if iovec_cnt > 1024 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    let mut bytes = Vec::new();
    for i in 0..iovec_cnt {
        let Some(iovec_offset) = 16u64.checked_mul(i) else {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        };
        let Some(lptr) = iovec_ptr.checked_add(iovec_offset) else {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        };
        let iov_base = match cpu.ram.read_doubleword(lptr) {
            Ok(iov_base) => iov_base,
            Err(_) => {
                cpu.registers[A0] = Errno::EFAULT.into_err();
                return;
            }
        };
        let Some(iov_len_ptr) = lptr.checked_add(8) else {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        };
        let iov_len = match cpu.ram.read_doubleword(iov_len_ptr) {
            Ok(iov_len) => iov_len,
            Err(_) => {
                cpu.registers[A0] = Errno::EFAULT.into_err();
                return;
            }
        };

        let iov_bytes = match read_bytes(cpu, iov_base, iov_len) {
            Ok(iov_bytes) => iov_bytes,
            Err(errno) => {
                cpu.registers[A0] = errno.into_err();
                return;
            }
        };

        if let Err(errno) = checked_extend(&mut bytes, iov_bytes) {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    }

    match cpu.filesystem.write(fd, &bytes) {
        Ok(bytes_written) => {
            trace!("wrote {bytes_written} bytes");
            cpu.registers[A0] = bytes_written as u64;
        }
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 73
pub fn ppoll(cpu: &mut RV64GC) {
    const POLLIN: u16 = 0x0001;
    const POLLOUT: u16 = 0x0004;

    let fds = cpu.registers[A0];
    let nfds = cpu.registers[A1];
    let mut ready = 0;

    for idx in 0..nfds {
        let pollfd = fds + idx * 8;
        let fd = cpu.ram.read_word(pollfd).unwrap_or_default() as i32;
        let events = cpu.ram.read_halfword(pollfd + 4).unwrap_or_default() as u16;
        let mut revents = 0;

        if fd == 0 && events & POLLIN != 0 {
            revents |= POLLIN;
        }

        if (fd == 1 || fd == 2) && events & POLLOUT != 0 {
            revents |= POLLOUT;
        }

        if cpu.ram.write_halfword(pollfd + 6, revents.into()).is_err() {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }

        if revents != 0 {
            ready += 1;
        }
    }

    cpu.registers[A0] = ready;
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

    debug!("mmap\n\taddr: {addr}\n\tlen: {len}\n\tprot: {prot}\n\tflags: {flags}\n\tfd: {fd}\n\toffset: {offset}");

    if len == 0 || offset % PAGE_SIZE != 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    if flags & MAP_FIXED != 0 && addr % PAGE_SIZE != 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    let requested_addr = (flags & MAP_FIXED != 0).then_some(addr);
    let region_flags = if prot & PROT_EXEC != 0 { 1 } else { 0 };
    let replace = flags & MAP_FIXED != 0;
    let mmap_result = if fd != -1 && flags & MAP_ANONYMOUS == 0 {
        let Ok(mut data) = zeroed_buffer(len) else {
            cpu.registers[A0] = Errno::ENOMEM.into_err();
            return;
        };
        match cpu.filesystem.read_at(fd as u64, offset, &mut data) {
            Ok(_) => cpu
                .ram
                .mmap_bytes(requested_addr, len, data, region_flags, replace),
            Err(error) => {
                cpu.registers[A0] = fs_error_to_errno(error).into_err();
                return;
            }
        }
    } else {
        cpu.ram
            .mmap_anonymous_with_flags(requested_addr, len, region_flags, replace)
    };

    match mmap_result {
        Ok(mmap_addr) => {
            debug!("mmap_addr: {mmap_addr:08x}");
            cpu.registers[A0] = mmap_addr;
        }
        Err(err) => {
            error!("mmap failed: {err}");
            cpu.registers[A0] = Errno::ENOMEM.into_err();
        }
    }
}

pub fn brk(cpu: &mut RV64GC) {
    let addr = cpu.registers[A0];
    debug!("brk: addr 0x{addr:08x}");
    if addr == 0 {
        cpu.registers[A0] = cpu.ram.program_break();
        return;
    }

    match cpu.ram.set_program_break(addr) {
        Ok(new_break) => cpu.registers[A0] = new_break,
        Err(err) => {
            error!("brk failed for 0x{addr:08x}: {err}");
            cpu.registers[A0] = cpu.ram.program_break();
        }
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
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    }

    cpu.registers[A0] = len;
}

// 172
pub fn getpid(cpu: &mut RV64GC) {
    cpu.registers[A0] = std::process::id().into();
}

// 173
pub fn getppid(cpu: &mut RV64GC) {
    cpu.registers[A0] = 1;
}

// 174
pub fn getuid(cpu: &mut RV64GC) {
    cpu.registers[A0] = 1000;
}

// 175
pub fn geteuid(cpu: &mut RV64GC) {
    cpu.registers[A0] = 1000;
}

// 176
pub fn getgid(cpu: &mut RV64GC) {
    cpu.registers[A0] = 1000;
}

// 177
pub fn getegid(cpu: &mut RV64GC) {
    cpu.registers[A0] = 1000;
}

// 178
pub fn gettid(cpu: &mut RV64GC) {
    cpu.registers[A0] = cpu.thread_id();
}

// 96
pub fn set_tid_address(cpu: &mut RV64GC) {
    let clear_child_tid = cpu.registers[A0];
    cpu.set_clear_child_tid((clear_child_tid != 0).then_some(clear_child_tid));
    cpu.registers[A0] = cpu.thread_id();
}

// 179
pub fn sysinfo(cpu: &mut RV64GC) {
    let info = cpu.registers[A0];
    if let Err(errno) = validate_guest_buffer(cpu, info, 112) {
        cpu.registers[A0] = errno.into_err();
        return;
    }

    let total_ram = 8 * 1024 * 1024 * 1024u64;
    let free_ram = 4 * 1024 * 1024 * 1024u64;

    let result = cpu
        .ram
        .write_doubleword(info, 3600)
        .and_then(|_| cpu.ram.write_doubleword(info + 8, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 16, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 24, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 32, total_ram))
        .and_then(|_| cpu.ram.write_doubleword(info + 40, free_ram))
        .and_then(|_| cpu.ram.write_doubleword(info + 48, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 56, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 64, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 72, 0))
        .and_then(|_| cpu.ram.write_halfword(info + 80, EMULATED_CPU_COUNT))
        .and_then(|_| cpu.ram.write_halfword(info + 82, 0))
        .and_then(|_| cpu.ram.write_word(info + 84, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 88, 0))
        .and_then(|_| cpu.ram.write_doubleword(info + 96, 0))
        .and_then(|_| cpu.ram.write_word(info + 104, 1))
        .and_then(|_| cpu.ram.write_word(info + 108, 0));

    cpu.registers[A0] = if result.is_ok() {
        0
    } else {
        Errno::EFAULT.into_err()
    };
}

// 113
pub fn clock_gettime(cpu: &mut RV64GC) {
    let timespec = cpu.registers[A1];
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    if cpu
        .ram
        .write_doubleword(timespec, now.as_secs())
        .and_then(|_| {
            cpu.ram
                .write_doubleword(timespec + 8, now.subsec_nanos().into())
        })
        .is_err()
    {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    }

    cpu.registers[A0] = 0;
}

// 115
pub fn clock_getres(cpu: &mut RV64GC) {
    let timespec = cpu.registers[A1];
    if timespec == 0 {
        cpu.registers[A0] = 0;
        return;
    }

    if cpu
        .ram
        .write_doubleword(timespec, 0)
        .and_then(|_| cpu.ram.write_doubleword(timespec + 8, 1))
        .is_err()
    {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    }

    cpu.registers[A0] = 0;
}

// 123
pub fn sched_getaffinity(cpu: &mut RV64GC) {
    let _pid = cpu.registers[A0];
    let cpusetsize = cpu.registers[A1];
    let mask = cpu.registers[A2];

    if cpusetsize == 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    if let Err(errno) = validate_guest_buffer(cpu, mask, cpusetsize) {
        cpu.registers[A0] = errno.into_err();
        return;
    }

    for idx in 0..cpusetsize {
        if cpu.ram.write_byte(mask + idx, 0).is_err() {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    }

    let bits = (cpusetsize as usize)
        .saturating_mul(8)
        .min(EMULATED_CPU_COUNT as usize);
    for cpu_idx in 0..bits {
        let byte_addr = mask + (cpu_idx / 8) as u64;
        let Ok(byte) = cpu.ram.read_byte(byte_addr) else {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        };
        let bit = 1u8 << (cpu_idx % 8);
        if cpu.ram.write_byte(byte_addr, byte | bit).is_err() {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    }

    cpu.registers[A0] = 0;
}

// 122
pub fn sched_setaffinity(cpu: &mut RV64GC) {
    let _pid = cpu.registers[A0];
    let cpusetsize = cpu.registers[A1];
    let mask = cpu.registers[A2];

    if cpusetsize == 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }
    if let Err(errno) = validate_guest_buffer(cpu, mask, cpusetsize) {
        cpu.registers[A0] = errno.into_err();
        return;
    }

    cpu.registers[A0] = Errno::ENOSYS.into_err();
}

// 220
pub fn sys_clone(cpu: &mut RV64GC) {
    let flags = cpu.registers[A0];
    let stack = cpu.registers[A1];
    let parent_tid = cpu.registers[A2];
    let tls = cpu.registers[A3];
    let child_tid = cpu.registers[A4];

    match enqueue_synthetic_clone(cpu, flags, stack, parent_tid, tls, child_tid) {
        Ok(tid) => cpu.registers[A0] = tid,
        Err(errno) => cpu.registers[A0] = errno.into_err(),
    }
}

// 435
pub fn clone3(cpu: &mut RV64GC) {
    let args = cpu.registers[A0];
    let size = cpu.registers[A1];

    let result = (|| {
        let flags = read_clone_arg(cpu, args, size, CLONE_ARGS_FLAGS_OFFSET)?;
        let child_tid = read_clone_arg(cpu, args, size, CLONE_ARGS_CHILD_TID_OFFSET)?;
        let parent_tid = read_clone_arg(cpu, args, size, CLONE_ARGS_PARENT_TID_OFFSET)?;
        let stack = read_clone_arg(cpu, args, size, CLONE_ARGS_STACK_OFFSET)?;
        let stack_size = read_clone_arg(cpu, args, size, CLONE_ARGS_STACK_SIZE_OFFSET)?;
        let tls = read_clone_arg(cpu, args, size, CLONE_ARGS_TLS_OFFSET)?;
        let child_stack = if stack != 0 && stack_size != 0 {
            stack.checked_add(stack_size).ok_or(Errno::EINVAL)?
        } else {
            stack
        };
        enqueue_synthetic_clone(cpu, flags, child_stack, parent_tid, tls, child_tid)
    })();

    match result {
        Ok(tid) => cpu.registers[A0] = tid,
        Err(errno) => cpu.registers[A0] = errno.into_err(),
    }
}

// 135
pub fn rt_sigprocmask(cpu: &mut RV64GC) {
    cpu.registers[A0] = 0;
}

// 131
pub fn tgkill(cpu: &mut RV64GC) {
    let tgid = cpu.registers[A0];
    let tid = cpu.registers[A1];
    let sig = cpu.registers[A2];
    debug!("tgkill: tgid {tgid}, tid {tid}, sig {sig}");
    cpu.registers[A0] = Errno::ENOSYS.into_err();
}

// 132
pub fn sigaltstack(cpu: &mut RV64GC) {
    cpu.registers[A0] = 0;
}

// 261
pub fn prlimit64(cpu: &mut RV64GC) {
    let resource = cpu.registers[A1];
    let new_limit = cpu.registers[A2];
    let old_limit = cpu.registers[A3];

    if new_limit != 0 {
        warn!("prlimit64 set operation is not supported; ignoring requested update");
    }

    if old_limit != 0 {
        let (cur, max) = match resource {
            // RLIMIT_STACK
            3 => (8 * 1024 * 1024, u64::MAX),
            _ => (u64::MAX, u64::MAX),
        };

        if cpu
            .ram
            .write_doubleword(old_limit, cur)
            .and_then(|_| cpu.ram.write_doubleword(old_limit + 8, max))
            .is_err()
        {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    }

    cpu.registers[A0] = 0;
}

// 134
pub fn sig_action(cpu: &mut RV64GC) {
    let signum = cpu.registers[A0];
    let act = cpu.registers[A1];
    let oldact = cpu.registers[A2];
    let sigsetsize = cpu.registers[A3];
    debug!("rt_sigaction: signum {signum}, act 0x{act:08x}, oldact 0x{oldact:08x}, sigsetsize {sigsetsize}");
    cpu.registers[A0] = 0;
}

// 160
pub fn uname(cpu: &mut RV64GC) {
    let ptr = cpu.registers[A0];
    let fields = ["Linux", "riscvm", "6.0.0", "#1", "riscv64", "(none)"];
    for (idx, field) in fields.iter().enumerate() {
        if write_uts_field(cpu, ptr + idx as u64 * UTS_FIELD_LEN, field).is_err() {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    }

    cpu.registers[A0] = 0;
}

// 78
// readlinkat
pub fn readlink(cpu: &mut RV64GC) {
    let pathname = cpu.registers[A1];
    let buf = cpu.registers[A2];
    let bufsiz = cpu.registers[A3] as usize;

    let Ok(pathname) = read_c_string(cpu, pathname) else {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    };

    let target = match pathname.as_str() {
        "/proc/self/exe" => {
            let exe = cpu
                .executable_path()
                .map(|path| path.to_path_buf())
                .unwrap_or_else(|| "riscvm".into());
            std::fs::canonicalize(&exe)
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_else(|_| exe.to_string_lossy().into_owned())
        }
        _ => {
            cpu.registers[A0] = Errno::ENOENT.into_err();
            return;
        }
    };

    let bytes = target.as_bytes();
    let write_len = bytes.len().min(bufsiz);
    if write_bytes(cpu, buf, &bytes[..write_len]).is_err() {
        cpu.registers[A0] = Errno::EFAULT.into_err();
        return;
    }

    cpu.registers[A0] = write_len as u64;
}

// 79
pub fn newfstatat(cpu: &mut RV64GC) {
    let dirfd = cpu.registers[A0] as i64;
    let pathname = cpu.registers[A1];
    let statbuf = cpu.registers[A2];
    let flags = cpu.registers[A3];
    let path = match read_c_string(cpu, pathname) {
        Ok(path) => path,
        Err(_) => {
            cpu.registers[A0] = Errno::EFAULT.into_err();
            return;
        }
    };

    let metadata = if path.is_empty() && flags & AT_EMPTY_PATH != 0 {
        cpu.filesystem.stat_fd(dirfd as u64)
    } else if valid_dirfd_for_path(dirfd, &path) {
        cpu.filesystem.stat_path(&path)
    } else {
        Err(FileSystemError::BadFileDescriptor)
    };

    match metadata {
        Ok(metadata) => match write_stat(cpu, statbuf, metadata) {
            Ok(()) => cpu.registers[A0] = 0,
            Err(errno) => cpu.registers[A0] = errno.into_err(),
        },
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 226
pub fn mprotect(cpu: &mut RV64GC) {
    cpu.registers[A0] = 0;
}

// 215
pub fn munmap(cpu: &mut RV64GC) {
    let addr = cpu.registers[A0];
    let len = cpu.registers[A1];
    if len == 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    if addr % PAGE_SIZE != 0 {
        cpu.registers[A0] = Errno::EINVAL.into_err();
        return;
    }

    match cpu.ram.munmap(addr, len) {
        Ok(()) => cpu.registers[A0] = 0,
        Err(_) => cpu.registers[A0] = Errno::EINVAL.into_err(),
    }
}

// 233
pub fn madvise(cpu: &mut RV64GC) {
    cpu.registers[A0] = 0;
}

// 80
pub fn fstat(cpu: &mut RV64GC) {
    let fd = cpu.registers[A0];
    let statbuf = cpu.registers[A1];
    match cpu.filesystem.stat_fd(fd) {
        Ok(metadata) => match write_stat(cpu, statbuf, metadata) {
            Ok(()) => cpu.registers[A0] = 0,
            Err(errno) => cpu.registers[A0] = errno.into_err(),
        },
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 258
pub fn riscv_hwprobe(cpu: &mut RV64GC) {
    cpu.registers[A0] = Errno::ENOSYS.into_err();
}

// 62
pub fn lseek(cpu: &mut RV64GC) {
    let span = span!(Level::TRACE, "lseek");
    let _guard = span.enter();

    let fd = cpu.registers[A0];
    let offset = cpu.registers[A1] as i64;
    let whence = match seek_whence(cpu.registers[A2]) {
        Ok(whence) => whence,
        Err(errno) => {
            cpu.registers[A0] = errno.into_err();
            return;
        }
    };

    trace!("lseek");
    trace!("fd: {fd}");
    trace!("offset: {offset}");
    trace!("seek mode: {whence:?}");

    match cpu.filesystem.seek(fd, offset, whence) {
        Ok(new_offset) => cpu.registers[A0] = new_offset,
        Err(error) => cpu.registers[A0] = fs_error_to_errno(error).into_err(),
    }
}

// 98
// https://www.man7.org/linux/man-pages/man2/futex.2.html
// NOTE: Just return FUTEX_WAIT for now
pub fn futex(cpu: &mut RV64GC) {
    if cpu.is_synthetic_thread() {
        cpu.registers[A0] = 0;
        cpu.yield_synthetic_thread();
        return;
    }

    cpu.make_synthetic_threads_ready();
    cpu.registers[A0] = match run_pending_synthetic_threads(cpu) {
        Ok(()) => 0,
        Err(errno) => errno.into_err(),
    };
}

// 293
pub fn rseq(cpu: &mut RV64GC) {
    cpu.registers[A0] = Errno::ENOSYS.into_err();
}

pub fn unimplemented_syscall(cpu: &mut RV64GC, id: u64) {
    warn!("syscall {id} is not implemented");
    cpu.registers[A0] = Errno::ENOSYS.into_err();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ram::MemoryRegion;

    const BASE: u64 = 0x1000;

    fn cpu_with_memory() -> RV64GC {
        let mut cpu = RV64GC::new();
        cpu.filesystem.set_output_mirroring(false);
        cpu.ram
            .add_region(MemoryRegion::new(BASE, 0x1000, vec![0; 0x1000]))
            .unwrap();
        cpu
    }

    fn write_guest_bytes(cpu: &mut RV64GC, addr: u64, bytes: &[u8]) {
        for (idx, byte) in bytes.iter().enumerate() {
            cpu.ram.write_byte(addr + idx as u64, *byte).unwrap();
        }
    }

    fn write_guest_word(cpu: &mut RV64GC, addr: u64, value: u32) {
        cpu.ram.write_word(addr, value).unwrap();
    }

    fn read_guest_bytes(cpu: &RV64GC, addr: u64, len: usize) -> Vec<u8> {
        (0..len)
            .map(|idx| cpu.ram.read_byte(addr + idx as u64).unwrap())
            .collect()
    }

    #[test]
    fn read_syscall_copies_from_emulated_stdin() {
        let mut cpu = cpu_with_memory();
        cpu.set_stdin(b"abcdef".to_vec());
        cpu.registers[A0] = 0;
        cpu.registers[A1] = BASE;
        cpu.registers[A2] = 4;

        read(&mut cpu);

        assert_eq!(cpu.registers[A0], 4);
        assert_eq!(read_guest_bytes(&cpu, BASE, 4), b"abcd");

        cpu.registers[A0] = 0;
        cpu.registers[A1] = BASE + 8;
        cpu.registers[A2] = 8;

        read(&mut cpu);

        assert_eq!(cpu.registers[A0], 2);
        assert_eq!(read_guest_bytes(&cpu, BASE + 8, 2), b"ef");
    }

    #[test]
    fn read_syscall_rejects_invalid_descriptors_without_panicking() {
        let mut cpu = cpu_with_memory();
        cpu.registers[A0] = 99;
        cpu.registers[A1] = BASE;
        cpu.registers[A2] = 1;

        read(&mut cpu);

        assert_eq!(cpu.registers[A0], Errno::EBADF.into_err());
    }

    #[test]
    fn read_syscall_rejects_invalid_guest_buffers_before_consuming_input() {
        let mut cpu = cpu_with_memory();
        cpu.set_stdin(b"xyz".to_vec());
        cpu.registers[A0] = 0;
        cpu.registers[A1] = 0xdead_beef;
        cpu.registers[A2] = 1;

        read(&mut cpu);

        assert_eq!(cpu.registers[A0], Errno::EFAULT.into_err());

        cpu.registers[A0] = 0;
        cpu.registers[A1] = BASE;
        cpu.registers[A2] = 1;

        read(&mut cpu);

        assert_eq!(cpu.registers[A0], 1);
        assert_eq!(read_guest_bytes(&cpu, BASE, 1), b"x");
    }

    #[test]
    fn write_syscall_copies_to_emulated_stdout() {
        let mut cpu = cpu_with_memory();
        write_guest_bytes(&mut cpu, BASE, b"hello");
        cpu.registers[A0] = 1;
        cpu.registers[A1] = BASE;
        cpu.registers[A2] = 5;

        write(&mut cpu);

        assert_eq!(cpu.registers[A0], 5);
        assert_eq!(cpu.stdout(), b"hello");
    }

    #[test]
    fn write_syscall_rejects_invalid_guest_buffers_without_panicking() {
        let mut cpu = cpu_with_memory();
        cpu.registers[A0] = 1;
        cpu.registers[A1] = 0xdead_beef;
        cpu.registers[A2] = 4;

        write(&mut cpu);

        assert_eq!(cpu.registers[A0], Errno::EFAULT.into_err());
        assert!(cpu.stdout().is_empty());
    }

    #[test]
    fn writev_syscall_concatenates_iovecs_and_preserves_binary_bytes() {
        let mut cpu = cpu_with_memory();
        write_guest_bytes(&mut cpu, BASE + 0x80, b"ab\0");
        write_guest_bytes(&mut cpu, BASE + 0x90, b"cd");
        cpu.ram.write_doubleword(BASE, BASE + 0x80).unwrap();
        cpu.ram.write_doubleword(BASE + 8, 3).unwrap();
        cpu.ram.write_doubleword(BASE + 16, BASE + 0x90).unwrap();
        cpu.ram.write_doubleword(BASE + 24, 2).unwrap();
        cpu.registers[A0] = 1;
        cpu.registers[A1] = BASE;
        cpu.registers[A2] = 2;

        writev(&mut cpu);

        assert_eq!(cpu.registers[A0], 5);
        assert_eq!(cpu.stdout(), b"ab\0cd");
    }

    #[test]
    fn sched_getaffinity_writes_a_non_empty_cpu_mask() {
        let mut cpu = cpu_with_memory();
        cpu.registers[A0] = 0;
        cpu.registers[A1] = 16;
        cpu.registers[A2] = BASE;

        sched_getaffinity(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);
        assert_ne!(read_guest_bytes(&cpu, BASE, 16), vec![0; 16]);
    }

    #[test]
    fn sched_setaffinity_reports_unavailable_for_guest_fallback() {
        let mut cpu = cpu_with_memory();
        write_guest_bytes(&mut cpu, BASE, &[1, 0, 0, 0]);
        cpu.registers[A0] = 0;
        cpu.registers[A1] = 4;
        cpu.registers[A2] = BASE;

        sched_setaffinity(&mut cpu);

        assert_eq!(cpu.registers[A0], Errno::ENOSYS.into_err());
    }

    #[test]
    fn epoll_create1_reports_unavailable_for_guest_fallback() {
        let mut cpu = cpu_with_memory();
        cpu.registers[A0] = O_CLOEXEC;

        epoll_create1(&mut cpu);

        assert_eq!(cpu.registers[A0], Errno::ENOSYS.into_err());
    }

    #[test]
    fn clone_enqueues_child_and_futex_drains_it() {
        let mut cpu = cpu_with_memory();
        write_guest_word(&mut cpu, BASE + 4, 0x0000_0513); // li a0, 0
        write_guest_word(&mut cpu, BASE + 8, 0x05d0_0893); // li a7, 93
        write_guest_word(&mut cpu, BASE + 12, 0x0000_0073); // ecall

        let parent_tid = BASE + 0x100;
        let child_tid = BASE + 0x104;
        cpu.registers[Pc] = BASE;
        cpu.registers[A0] =
            CLONE_VM | CLONE_PARENT_SETTID | CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID;
        cpu.registers[A1] = 0;
        cpu.registers[A2] = parent_tid;
        cpu.registers[A3] = 0;
        cpu.registers[A4] = child_tid;

        sys_clone(&mut cpu);

        let tid = cpu.registers[A0];
        assert!(tid >= FIRST_SYNTHETIC_TID);
        assert_eq!(cpu.ram.read_word(parent_tid).unwrap(), tid as u32);
        assert_eq!(cpu.ram.read_word(child_tid).unwrap(), tid as u32);

        futex(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);
        assert_eq!(cpu.ram.read_word(child_tid).unwrap(), 0);
        assert!(!cpu.should_quit);
    }

    #[test]
    fn synthetic_child_futex_yields_without_clearing_tid() {
        let mut cpu = cpu_with_memory();
        write_guest_word(&mut cpu, BASE + 4, 0x0620_0893); // li a7, 98
        write_guest_word(&mut cpu, BASE + 8, 0x0000_0073); // ecall
        write_guest_word(&mut cpu, BASE + 12, 0x0000_0513); // li a0, 0
        write_guest_word(&mut cpu, BASE + 16, 0x05d0_0893); // li a7, 93
        write_guest_word(&mut cpu, BASE + 20, 0x0000_0073); // ecall

        let child_tid = BASE + 0x104;
        cpu.registers[Pc] = BASE;
        cpu.registers[A0] = CLONE_VM | CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID;
        cpu.registers[A1] = 0;
        cpu.registers[A2] = 0;
        cpu.registers[A3] = 0;
        cpu.registers[A4] = child_tid;

        sys_clone(&mut cpu);
        let tid = cpu.registers[A0];
        assert_eq!(cpu.ram.read_word(child_tid).unwrap(), tid as u32);

        futex(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);
        assert_eq!(cpu.ram.read_word(child_tid).unwrap(), tid as u32);
        assert!(!cpu.should_quit);

        futex(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);
        assert_eq!(cpu.ram.read_word(child_tid).unwrap(), 0);
        assert!(!cpu.should_quit);
    }

    #[test]
    fn sysinfo_writes_basic_memory_and_process_info() {
        let mut cpu = cpu_with_memory();
        cpu.registers[A0] = BASE;

        sysinfo(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);
        assert_eq!(
            cpu.ram.read_doubleword(BASE + 32).unwrap(),
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(cpu.ram.read_word(BASE + 104).unwrap(), 1);
    }

    #[test]
    fn clock_getres_writes_one_nanosecond_resolution() {
        let mut cpu = cpu_with_memory();
        cpu.registers[A0] = 1;
        cpu.registers[A1] = BASE;

        clock_getres(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);
        assert_eq!(cpu.ram.read_doubleword(BASE).unwrap(), 0);
        assert_eq!(cpu.ram.read_doubleword(BASE + 8).unwrap(), 1);
    }

    #[test]
    fn openat_write_lseek_read_and_close_use_emulated_files() {
        let mut cpu = cpu_with_memory();
        write_guest_bytes(&mut cpu, BASE, b"file-rw.txt\0");
        cpu.registers[A0] = AT_FDCWD as u64;
        cpu.registers[A1] = BASE;
        cpu.registers[A2] = 0o100 | 0o2;
        cpu.registers[A3] = 0o644;

        openat(&mut cpu);

        let fd = cpu.registers[A0];
        assert!(fd >= 3);

        write_guest_bytes(&mut cpu, BASE + 0x80, b"riscvm");
        cpu.registers[A0] = fd;
        cpu.registers[A1] = BASE + 0x80;
        cpu.registers[A2] = 6;

        write(&mut cpu);

        assert_eq!(cpu.registers[A0], 6);

        cpu.registers[A0] = fd;
        cpu.registers[A1] = 0;
        cpu.registers[A2] = 0;

        lseek(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);

        cpu.registers[A0] = fd;
        cpu.registers[A1] = BASE + 0x100;
        cpu.registers[A2] = 16;

        read(&mut cpu);

        assert_eq!(cpu.registers[A0], 6);
        assert_eq!(read_guest_bytes(&cpu, BASE + 0x100, 6), b"riscvm");

        cpu.registers[A0] = fd;
        close(&mut cpu);

        assert_eq!(cpu.registers[A0], 0);

        cpu.registers[A0] = fd;
        cpu.registers[A1] = BASE + 0x100;
        cpu.registers[A2] = 1;
        read(&mut cpu);

        assert_eq!(cpu.registers[A0], Errno::EBADF.into_err());
    }

    #[test]
    fn openat_without_create_reports_missing_file() {
        let mut cpu = cpu_with_memory();
        write_guest_bytes(&mut cpu, BASE, b"missing.txt\0");
        cpu.registers[A0] = AT_FDCWD as u64;
        cpu.registers[A1] = BASE;
        cpu.registers[A2] = 0;

        openat(&mut cpu);

        assert_eq!(cpu.registers[A0], Errno::ENOENT.into_err());
    }
}
