use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::rc::Rc;

pub type GuestFd = u64;

const FIRST_FILE_FD: GuestFd = 3;

const O_ACCMODE: i32 = 0o3;
const O_WRONLY: i32 = 0o1;
const O_RDWR: i32 = 0o2;
const O_CREAT: i32 = 0o100;
const O_EXCL: i32 = 0o200;
const O_TRUNC: i32 = 0o1000;
const O_APPEND: i32 = 0o2000;

const F_GETFD: u64 = 1;
const F_SETFD: u64 = 2;
const F_GETFL: u64 = 3;
const F_SETFL: u64 = 4;

const R_OK: i32 = 4;
const W_OK: i32 = 2;
const X_OK: i32 = 1;

const S_IFCHR: u32 = 0o020000;
const S_IFDIR: u32 = 0o040000;
const S_IFREG: u32 = 0o100000;

type FileHandle = Rc<RefCell<FileDescription>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSystemError {
    BadFileDescriptor,
    FileExists,
    IllegalSeek,
    InvalidInput,
    Io,
    IsDirectory,
    NoSuchFile,
    PermissionDenied,
    TooManyOpenFiles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekWhence {
    Set,
    Current,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileMetadata {
    pub mode: u32,
    pub size: u64,
    pub block_size: u32,
    pub blocks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMirror {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FileBacking {
    Stdin,
    Stdout,
    Stderr,
    File(String),
}

#[derive(Debug, Clone)]
struct FileDescription {
    backing: FileBacking,
    offset: usize,
    readable: bool,
    writable: bool,
    append: bool,
}

impl FileDescription {
    fn stdin() -> Self {
        Self {
            backing: FileBacking::Stdin,
            offset: 0,
            readable: true,
            writable: false,
            append: false,
        }
    }

    fn output(backing: FileBacking) -> Self {
        Self {
            backing,
            offset: 0,
            readable: false,
            writable: true,
            append: true,
        }
    }

    fn file(path: String, readable: bool, writable: bool, append: bool, len: usize) -> Self {
        Self {
            backing: FileBacking::File(path),
            offset: if append { len } else { 0 },
            readable,
            writable,
            append,
        }
    }

    fn flags(&self) -> i32 {
        let access = match (self.readable, self.writable) {
            (true, true) => O_RDWR,
            (false, true) => O_WRONLY,
            _ => 0,
        };
        let append = if self.append { O_APPEND } else { 0 };

        access | append
    }

    fn set_status_flags(&mut self, flags: i32) {
        self.append = flags & O_APPEND != 0;
    }
}

fn file_handle(description: FileDescription) -> FileHandle {
    Rc::new(RefCell::new(description))
}

#[derive(Debug, Clone)]
pub struct GuestFileSystem {
    descriptors: BTreeMap<GuestFd, FileHandle>,
    files: BTreeMap<String, Vec<u8>>,
    stdin: Vec<u8>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    host_root: Option<PathBuf>,
    next_fd: GuestFd,
    mirror_output: bool,
}

impl Default for GuestFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl GuestFileSystem {
    pub fn new() -> Self {
        let mut descriptors = BTreeMap::new();
        descriptors.insert(0, file_handle(FileDescription::stdin()));
        descriptors.insert(1, file_handle(FileDescription::output(FileBacking::Stdout)));
        descriptors.insert(2, file_handle(FileDescription::output(FileBacking::Stderr)));

        Self {
            descriptors,
            files: BTreeMap::new(),
            stdin: Vec::new(),
            stdout: Vec::new(),
            stderr: Vec::new(),
            host_root: None,
            next_fd: FIRST_FILE_FD,
            mirror_output: true,
        }
    }

    pub fn set_stdin(&mut self, bytes: impl Into<Vec<u8>>) {
        self.stdin = bytes.into();
        self.descriptors
            .entry(0)
            .or_insert_with(|| file_handle(FileDescription::stdin()))
            .borrow_mut()
            .offset = 0;
    }

    pub fn set_output_mirroring(&mut self, enabled: bool) {
        self.mirror_output = enabled;
    }

    pub fn mount_host_directory(
        &mut self,
        root: impl Into<PathBuf>,
    ) -> Result<(), FileSystemError> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|_| FileSystemError::Io)?;
        self.host_root = Some(root);

        Ok(())
    }

    pub fn open(&mut self, path: &str, flags: i32) -> Result<GuestFd, FileSystemError> {
        let path = normalize_path(path)?;
        let access_mode = flags & O_ACCMODE;
        let (readable, writable) = match access_mode {
            0 => (true, false),
            O_WRONLY => (false, true),
            O_RDWR => (true, true),
            _ => return Err(FileSystemError::InvalidInput),
        };

        let create = flags & O_CREAT != 0;
        let exclusive = flags & O_EXCL != 0;
        let truncate = flags & O_TRUNC != 0;
        let append = flags & O_APPEND != 0;
        let exists = self.load_host_file_if_present(&path)?;

        if exists && create && exclusive {
            return Err(FileSystemError::FileExists);
        }

        if !exists {
            if !create {
                return Err(FileSystemError::NoSuchFile);
            }
            self.files.insert(path.clone(), Vec::new());
            self.persist_file(&path)?;
        }

        if truncate && writable {
            self.files
                .get_mut(&path)
                .ok_or(FileSystemError::NoSuchFile)?
                .clear();
            self.persist_file(&path)?;
        }

        let len = self
            .files
            .get(&path)
            .ok_or(FileSystemError::NoSuchFile)?
            .len();
        let fd = self.allocate_fd()?;
        self.descriptors.insert(
            fd,
            file_handle(FileDescription::file(path, readable, writable, append, len)),
        );

        Ok(fd)
    }

    pub fn close(&mut self, fd: GuestFd) -> Result<(), FileSystemError> {
        self.descriptors
            .remove(&fd)
            .map(|_| ())
            .ok_or(FileSystemError::BadFileDescriptor)
    }

    pub fn duplicate(&mut self, fd: GuestFd) -> Result<GuestFd, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        let new_fd = self.allocate_fd()?;
        self.descriptors.insert(new_fd, description);

        Ok(new_fd)
    }

    pub fn duplicate_to(
        &mut self,
        fd: GuestFd,
        new_fd: GuestFd,
    ) -> Result<GuestFd, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        self.descriptors.insert(new_fd, description);

        Ok(new_fd)
    }

    pub fn fcntl(&mut self, fd: GuestFd, cmd: u64, arg: u64) -> Result<u64, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        let mut description = description.borrow_mut();

        match cmd {
            F_GETFD => Ok(0),
            F_SETFD => Ok(0),
            F_GETFL => Ok(description.flags() as u64),
            F_SETFL => {
                description.set_status_flags(arg as i32);
                Ok(0)
            }
            _ => Err(FileSystemError::InvalidInput),
        }
    }

    pub fn read(&mut self, fd: GuestFd, buffer: &mut [u8]) -> Result<usize, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        let mut description = description.borrow_mut();
        if !description.readable {
            return Err(FileSystemError::BadFileDescriptor);
        }

        let source = match &description.backing {
            FileBacking::Stdin => self.stdin.as_slice(),
            FileBacking::File(path) => self
                .files
                .get(path)
                .ok_or(FileSystemError::BadFileDescriptor)?
                .as_slice(),
            FileBacking::Stdout | FileBacking::Stderr => {
                return Err(FileSystemError::BadFileDescriptor)
            }
        };

        let bytes_read = read_from_buffer(source, &mut description.offset, buffer);
        Ok(bytes_read)
    }

    pub fn read_at(
        &self,
        fd: GuestFd,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<usize, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        let description = description.borrow();
        if !description.readable {
            return Err(FileSystemError::BadFileDescriptor);
        }

        let source = match &description.backing {
            FileBacking::File(path) => self
                .files
                .get(path)
                .ok_or(FileSystemError::BadFileDescriptor)?
                .as_slice(),
            FileBacking::Stdin | FileBacking::Stdout | FileBacking::Stderr => {
                return Err(FileSystemError::IllegalSeek)
            }
        };

        let mut offset = usize::try_from(offset).map_err(|_| FileSystemError::InvalidInput)?;
        Ok(read_from_buffer(source, &mut offset, buffer))
    }

    pub fn write(&mut self, fd: GuestFd, bytes: &[u8]) -> Result<usize, FileSystemError> {
        let (path_to_persist, bytes_written) = {
            let description = self
                .descriptors
                .get(&fd)
                .ok_or(FileSystemError::BadFileDescriptor)?
                .clone();
            let mut description = description.borrow_mut();
            if !description.writable {
                return Err(FileSystemError::BadFileDescriptor);
            }

            let backing = description.backing.clone();
            let append = description.append;
            match backing {
                FileBacking::Stdout => {
                    if self.mirror_output {
                        write_mirror(OutputMirror::Stdout, bytes)
                            .map_err(|_| FileSystemError::Io)?;
                    }
                    let bytes_written =
                        write_to_buffer(&mut self.stdout, &mut description.offset, bytes, append)?;
                    (None, bytes_written)
                }
                FileBacking::Stderr => {
                    if self.mirror_output {
                        write_mirror(OutputMirror::Stderr, bytes)
                            .map_err(|_| FileSystemError::Io)?;
                    }
                    let bytes_written =
                        write_to_buffer(&mut self.stderr, &mut description.offset, bytes, append)?;
                    (None, bytes_written)
                }
                FileBacking::File(path) => {
                    let file = self
                        .files
                        .get_mut(&path)
                        .ok_or(FileSystemError::BadFileDescriptor)?;
                    let bytes_written =
                        write_to_buffer(file, &mut description.offset, bytes, append)?;
                    (Some(path), bytes_written)
                }
                FileBacking::Stdin => return Err(FileSystemError::BadFileDescriptor),
            }
        };

        if let Some(path) = path_to_persist {
            self.persist_file(&path)?;
        }

        Ok(bytes_written)
    }

    pub fn write_at(
        &mut self,
        fd: GuestFd,
        offset: u64,
        bytes: &[u8],
    ) -> Result<usize, FileSystemError> {
        let path_to_persist;
        let bytes_written;
        {
            let description = self
                .descriptors
                .get(&fd)
                .ok_or(FileSystemError::BadFileDescriptor)?
                .clone();
            let description = description.borrow();
            if !description.writable {
                return Err(FileSystemError::BadFileDescriptor);
            }

            let FileBacking::File(path) = description.backing.clone() else {
                return Err(FileSystemError::IllegalSeek);
            };
            let file = self
                .files
                .get_mut(&path)
                .ok_or(FileSystemError::BadFileDescriptor)?;
            let mut offset = usize::try_from(offset).map_err(|_| FileSystemError::InvalidInput)?;
            bytes_written = write_to_buffer(file, &mut offset, bytes, false)?;
            path_to_persist = path;
        }

        self.persist_file(&path_to_persist)?;
        Ok(bytes_written)
    }

    pub fn seek(
        &mut self,
        fd: GuestFd,
        offset: i64,
        whence: SeekWhence,
    ) -> Result<u64, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        let mut description = description.borrow_mut();
        let file_len = match &description.backing {
            FileBacking::File(path) => self
                .files
                .get(path)
                .ok_or(FileSystemError::BadFileDescriptor)?
                .len(),
            FileBacking::Stdin | FileBacking::Stdout | FileBacking::Stderr => {
                return Err(FileSystemError::IllegalSeek)
            }
        };

        let base = match whence {
            SeekWhence::Set => 0,
            SeekWhence::Current => description.offset as i64,
            SeekWhence::End => file_len as i64,
        };
        let new_offset = base
            .checked_add(offset)
            .ok_or(FileSystemError::InvalidInput)?;
        if new_offset < 0 {
            return Err(FileSystemError::InvalidInput);
        }
        let new_offset = usize::try_from(new_offset).map_err(|_| FileSystemError::InvalidInput)?;
        description.offset = new_offset;

        Ok(new_offset as u64)
    }

    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }

    pub fn file_contents(&self, path: &str) -> Option<&[u8]> {
        let path = normalize_path(path).ok()?;
        self.files.get(&path).map(Vec::as_slice)
    }

    pub fn stat_fd(&self, fd: GuestFd) -> Result<FileMetadata, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        let description = description.borrow();

        match &description.backing {
            FileBacking::Stdin | FileBacking::Stdout | FileBacking::Stderr => Ok(char_metadata()),
            FileBacking::File(path) => self
                .files
                .get(path)
                .map(|bytes| regular_metadata(bytes.len() as u64))
                .ok_or(FileSystemError::BadFileDescriptor),
        }
    }

    pub fn stat_path(&mut self, path: &str) -> Result<FileMetadata, FileSystemError> {
        if is_directory_path(path) {
            return Ok(directory_metadata());
        }

        let path = normalize_path(path)?;
        if self.load_host_file_if_present(&path)? {
            return self
                .files
                .get(&path)
                .map(|bytes| regular_metadata(bytes.len() as u64))
                .ok_or(FileSystemError::NoSuchFile);
        }

        Err(FileSystemError::NoSuchFile)
    }

    pub fn access(&mut self, path: &str, mode: i32) -> Result<(), FileSystemError> {
        if mode & !(R_OK | W_OK | X_OK) != 0 {
            return Err(FileSystemError::InvalidInput);
        }

        let metadata = self.stat_path(path)?;
        if mode & X_OK != 0 && metadata.mode & S_IFDIR == 0 {
            return Err(FileSystemError::PermissionDenied);
        }

        Ok(())
    }

    fn allocate_fd(&mut self) -> Result<GuestFd, FileSystemError> {
        for _ in FIRST_FILE_FD..GuestFd::MAX {
            let fd = self.next_fd;
            self.next_fd = self.next_fd.checked_add(1).unwrap_or(FIRST_FILE_FD);
            if self.next_fd < FIRST_FILE_FD {
                self.next_fd = FIRST_FILE_FD;
            }
            if !self.descriptors.contains_key(&fd) {
                return Ok(fd);
            }
        }

        Err(FileSystemError::TooManyOpenFiles)
    }

    fn load_host_file_if_present(&mut self, path: &str) -> Result<bool, FileSystemError> {
        let Some(host_path) = self.host_path(path) else {
            return Ok(self.files.contains_key(path));
        };

        if host_path.is_dir() {
            return Err(FileSystemError::IsDirectory);
        }

        if !host_path.exists() {
            return Ok(self.files.contains_key(path));
        }

        let bytes = fs::read(&host_path).map_err(|_| FileSystemError::PermissionDenied)?;
        self.files.insert(path.to_string(), bytes);

        Ok(true)
    }

    fn persist_file(&self, path: &str) -> Result<(), FileSystemError> {
        let Some(host_path) = self.host_path(path) else {
            return Ok(());
        };

        let Some(bytes) = self.files.get(path) else {
            return Err(FileSystemError::NoSuchFile);
        };

        if let Some(parent) = host_path.parent() {
            if !parent.exists() {
                return Err(FileSystemError::NoSuchFile);
            }
        }

        fs::write(host_path, bytes).map_err(|_| FileSystemError::PermissionDenied)
    }

    fn host_path(&self, path: &str) -> Option<PathBuf> {
        let mut host_path = self.host_root.clone()?;
        for component in path.split('/') {
            host_path.push(component);
        }

        Some(host_path)
    }
}

fn regular_metadata(size: u64) -> FileMetadata {
    FileMetadata {
        mode: S_IFREG | 0o666,
        size,
        block_size: 4096,
        blocks: size.div_ceil(512),
    }
}

fn directory_metadata() -> FileMetadata {
    FileMetadata {
        mode: S_IFDIR | 0o777,
        size: 0,
        block_size: 4096,
        blocks: 0,
    }
}

fn char_metadata() -> FileMetadata {
    FileMetadata {
        mode: S_IFCHR | 0o666,
        size: 0,
        block_size: 4096,
        blocks: 0,
    }
}

fn is_directory_path(path: &str) -> bool {
    path == "/" || path == "." || path == "./"
}

fn normalize_path(path: &str) -> Result<String, FileSystemError> {
    if path.is_empty() {
        return Err(FileSystemError::NoSuchFile);
    }

    if path == "/" || path.ends_with('/') {
        return Err(FileSystemError::IsDirectory);
    }

    let mut parts = Vec::new();
    for part in path.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." || part.contains('\\') {
            return Err(FileSystemError::PermissionDenied);
        }
        parts.push(part);
    }

    if parts.is_empty() {
        return Err(FileSystemError::IsDirectory);
    }

    Ok(parts.join("/"))
}

fn read_from_buffer(source: &[u8], offset: &mut usize, buffer: &mut [u8]) -> usize {
    let available = source.len().saturating_sub(*offset);
    let bytes_read = available.min(buffer.len());
    buffer[..bytes_read].copy_from_slice(&source[*offset..*offset + bytes_read]);
    *offset += bytes_read;

    bytes_read
}

fn write_to_buffer(
    target: &mut Vec<u8>,
    offset: &mut usize,
    bytes: &[u8],
    append: bool,
) -> Result<usize, FileSystemError> {
    if append {
        *offset = target.len();
    }

    let end = (*offset)
        .checked_add(bytes.len())
        .ok_or(FileSystemError::InvalidInput)?;
    if end > target.len() {
        target.resize(end, 0);
    }
    target[*offset..end].copy_from_slice(bytes);
    *offset = end;

    Ok(bytes.len())
}

fn write_mirror(mirror: OutputMirror, bytes: &[u8]) -> io::Result<()> {
    match mirror {
        OutputMirror::Stdout => {
            let mut stdout = io::stdout().lock();
            stdout.write_all(bytes)?;
            stdout.flush()
        }
        OutputMirror::Stderr => {
            let mut stderr = io::stderr().lock();
            stderr.write_all(bytes)?;
            stderr.flush()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FileSystemError, GuestFileSystem, SeekWhence};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_mount(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("riscvm-{name}-{}-{nonce}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn stdin_reads_from_configured_buffer_and_advances_offset() {
        let mut fs = GuestFileSystem::new();
        fs.set_stdin(b"abcdef".to_vec());

        let mut first = [0; 2];
        assert_eq!(fs.read(0, &mut first), Ok(2));
        assert_eq!(&first, b"ab");

        let mut second = [0; 8];
        assert_eq!(fs.read(0, &mut second), Ok(4));
        assert_eq!(&second[..4], b"cdef");

        assert_eq!(fs.read(0, &mut second), Ok(0));
    }

    #[test]
    fn stdout_and_stderr_store_written_bytes() {
        let mut fs = GuestFileSystem::new();
        fs.set_output_mirroring(false);

        assert_eq!(fs.write(1, b"hello"), Ok(5));
        assert_eq!(fs.write(2, b"err"), Ok(3));

        assert_eq!(fs.stdout(), b"hello");
        assert_eq!(fs.stderr(), b"err");
    }

    #[test]
    fn unreadable_or_unwritable_descriptors_return_ebadf_equivalent() {
        let mut fs = GuestFileSystem::new();
        let mut byte = [0];

        assert_eq!(fs.write(0, b"x"), Err(FileSystemError::BadFileDescriptor));
        assert_eq!(
            fs.read(1, &mut byte),
            Err(FileSystemError::BadFileDescriptor)
        );
        assert_eq!(
            fs.read(99, &mut byte),
            Err(FileSystemError::BadFileDescriptor)
        );
    }

    #[test]
    fn open_write_seek_read_and_close_use_virtual_file_contents() {
        let mut fs = GuestFileSystem::new();
        let fd = fs.open("notes.txt", 0o100 | 0o2).unwrap();

        assert_eq!(fs.write(fd, b"hello"), Ok(5));
        assert_eq!(fs.seek(fd, 0, SeekWhence::Set), Ok(0));

        let mut buffer = [0; 8];
        assert_eq!(fs.read(fd, &mut buffer), Ok(5));
        assert_eq!(&buffer[..5], b"hello");
        assert_eq!(fs.close(fd), Ok(()));

        let reopened = fs.open("notes.txt", 0).unwrap();
        let mut reopened_buffer = [0; 5];
        assert_eq!(fs.read(reopened, &mut reopened_buffer), Ok(5));
        assert_eq!(&reopened_buffer, b"hello");
    }

    #[test]
    fn append_mode_writes_at_end_even_after_seek() {
        let mut fs = GuestFileSystem::new();
        let fd = fs.open("append.txt", 0o100 | 0o2).unwrap();
        assert_eq!(fs.write(fd, b"ab"), Ok(2));
        assert_eq!(fs.close(fd), Ok(()));

        let fd = fs.open("append.txt", 0o2000 | 0o1).unwrap();
        assert_eq!(fs.seek(fd, 0, SeekWhence::Set), Ok(0));
        assert_eq!(fs.write(fd, b"cd"), Ok(2));

        assert_eq!(fs.file_contents("append.txt"), Some(b"abcd".as_slice()));
    }

    #[test]
    fn duplicated_file_descriptors_share_the_file_offset() {
        let mut fs = GuestFileSystem::new();
        let fd = fs.open("dup.txt", 0o100 | 0o2).unwrap();
        assert_eq!(fs.write(fd, b"abcdef"), Ok(6));
        assert_eq!(fs.seek(fd, 0, SeekWhence::Set), Ok(0));

        let dup_fd = fs.duplicate(fd).unwrap();
        let mut first = [0; 2];
        let mut second = [0; 2];

        assert_eq!(fs.read(fd, &mut first), Ok(2));
        assert_eq!(fs.read(dup_fd, &mut second), Ok(2));
        assert_eq!(&first, b"ab");
        assert_eq!(&second, b"cd");
    }

    #[test]
    fn duplicate_to_can_replace_standard_descriptors() {
        let mut fs = GuestFileSystem::new();
        let fd = fs.open("stdin-replacement.txt", 0o100 | 0o2).unwrap();
        assert_eq!(fs.write(fd, b"replacement"), Ok(11));
        assert_eq!(fs.seek(fd, 0, SeekWhence::Set), Ok(0));
        assert_eq!(fs.duplicate_to(fd, 0), Ok(0));

        let mut buffer = [0; 16];
        assert_eq!(fs.read(0, &mut buffer), Ok(11));
        assert_eq!(&buffer[..11], b"replacement");
    }

    #[test]
    fn mounted_host_directory_persists_created_and_written_files() {
        let root = temp_mount("host-persist");
        let host_file = root.join("guest.txt");
        let mut fs = GuestFileSystem::new();
        fs.mount_host_directory(root.clone()).unwrap();

        let fd = fs.open("guest.txt", 0o100 | 0o2).unwrap();
        assert!(host_file.exists());
        assert_eq!(fs.write(fd, b"host-visible"), Ok(12));
        assert_eq!(fs.seek(fd, 0, SeekWhence::Set), Ok(0));

        let mut buffer = [0; 16];
        assert_eq!(fs.read(fd, &mut buffer), Ok(12));
        assert_eq!(&buffer[..12], b"host-visible");
        assert_eq!(fs::read(&host_file).unwrap(), b"host-visible");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn mounted_host_directory_loads_existing_files() {
        let root = temp_mount("host-load");
        fs::write(root.join("existing.txt"), b"from-host").unwrap();
        let mut fs = GuestFileSystem::new();
        fs.mount_host_directory(root.clone()).unwrap();

        let fd = fs.open("existing.txt", 0).unwrap();
        let mut buffer = [0; 16];
        assert_eq!(fs.read(fd, &mut buffer), Ok(9));
        assert_eq!(&buffer[..9], b"from-host");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn mounted_host_directory_rejects_path_traversal() {
        let root = temp_mount("host-sandbox");
        let mut fs = GuestFileSystem::new();
        fs.mount_host_directory(root.clone()).unwrap();

        assert_eq!(
            fs.open("../outside.txt", 0o100 | 0o2),
            Err(FileSystemError::PermissionDenied)
        );
        assert!(!root.parent().unwrap().join("outside.txt").exists());

        let _ = fs::remove_dir_all(root);
    }
}
