use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::rc::Rc;

use rand::RngCore;

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
    pub dev: u64,
    pub ino: u64,
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
    Null,
    Random,
    Directory(String),
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

    fn device(backing: FileBacking, readable: bool, writable: bool) -> Self {
        Self {
            backing,
            offset: 0,
            readable,
            writable,
            append: false,
        }
    }

    fn directory(path: String) -> Self {
        Self {
            backing: FileBacking::Directory(path),
            offset: 0,
            readable: false,
            writable: false,
            append: false,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct HostBindMount {
    guest_prefix: String,
    host_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct GuestFileSystem {
    descriptors: BTreeMap<GuestFd, FileHandle>,
    files: BTreeMap<String, Vec<u8>>,
    stdin: Vec<u8>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    host_roots: Vec<PathBuf>,
    host_bind_mounts: Vec<HostBindMount>,
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
            host_roots: Vec::new(),
            host_bind_mounts: Vec::new(),
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
        if !self.host_roots.contains(&root) {
            self.host_roots.push(root);
        }

        Ok(())
    }

    pub fn mount_host_directory_at(
        &mut self,
        guest_prefix: &str,
        root: impl Into<PathBuf>,
    ) -> Result<(), FileSystemError> {
        let guest_prefix = normalize_mount_path(guest_prefix)?;
        let root = root.into();
        fs::create_dir_all(&root).map_err(|_| FileSystemError::Io)?;
        let mount = HostBindMount {
            guest_prefix,
            host_root: root,
        };
        if !self.host_bind_mounts.contains(&mount) {
            self.host_bind_mounts.push(mount);
        }

        Ok(())
    }

    pub fn open(&mut self, path: &str, flags: i32) -> Result<GuestFd, FileSystemError> {
        let directory_path = normalize_mount_path(path)?;
        if self.is_directory(&directory_path) {
            let fd = self.allocate_fd()?;
            self.descriptors
                .insert(fd, file_handle(FileDescription::directory(directory_path)));
            return Ok(fd);
        }

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

        if is_random_device_path(&path) || is_null_device_path(&path) {
            if create && exclusive {
                return Err(FileSystemError::FileExists);
            }
            let backing = if is_random_device_path(&path) {
                FileBacking::Random
            } else {
                FileBacking::Null
            };
            let fd = self.allocate_fd()?;
            self.descriptors.insert(
                fd,
                file_handle(FileDescription::device(backing, readable, writable)),
            );
            return Ok(fd);
        }

        let exists = self.load_file_if_present(&path)?;

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
            FileBacking::Null => return Ok(0),
            FileBacking::Random => {
                rand::thread_rng().fill_bytes(buffer);
                description.offset = description.offset.saturating_add(buffer.len());
                return Ok(buffer.len());
            }
            FileBacking::Directory(_) => return Err(FileSystemError::IsDirectory),
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
            FileBacking::Stdin
            | FileBacking::Stdout
            | FileBacking::Stderr
            | FileBacking::Null
            | FileBacking::Random
            | FileBacking::Directory(_) => return Err(FileSystemError::IllegalSeek),
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
                FileBacking::Null | FileBacking::Random => (None, bytes.len()),
                FileBacking::Directory(_) => return Err(FileSystemError::IsDirectory),
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
            FileBacking::Stdin
            | FileBacking::Stdout
            | FileBacking::Stderr
            | FileBacking::Null
            | FileBacking::Random
            | FileBacking::Directory(_) => return Err(FileSystemError::IllegalSeek),
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
            FileBacking::Stdin
            | FileBacking::Stdout
            | FileBacking::Stderr
            | FileBacking::Null
            | FileBacking::Random => Ok(char_metadata()),
            FileBacking::Directory(path) => Ok(directory_metadata(path)),
            FileBacking::File(path) => self
                .files
                .get(path)
                .map(|bytes| regular_metadata(path, bytes.len() as u64))
                .ok_or(FileSystemError::BadFileDescriptor),
        }
    }

    pub fn stat_path(&mut self, path: &str) -> Result<FileMetadata, FileSystemError> {
        let directory_path = normalize_mount_path(path)?;
        if self.is_directory(&directory_path) {
            return Ok(directory_metadata(&directory_path));
        }

        let path = normalize_path(path)?;
        if is_random_device_path(&path) || is_null_device_path(&path) {
            return Ok(char_metadata());
        }

        if self.load_file_if_present(&path)? {
            return self
                .files
                .get(&path)
                .map(|bytes| regular_metadata(&path, bytes.len() as u64))
                .ok_or(FileSystemError::NoSuchFile);
        }

        Err(FileSystemError::NoSuchFile)
    }

    pub fn getdents64(
        &mut self,
        fd: GuestFd,
        max_bytes: usize,
    ) -> Result<Vec<u8>, FileSystemError> {
        let description = self
            .descriptors
            .get(&fd)
            .ok_or(FileSystemError::BadFileDescriptor)?
            .clone();
        let mut description = description.borrow_mut();
        let FileBacking::Directory(path) = description.backing.clone() else {
            return Err(FileSystemError::InvalidInput);
        };

        let entries = self.directory_entries(&path)?;
        let mut bytes = Vec::new();
        while description.offset < entries.len() {
            let entry = &entries[description.offset];
            let record_len = linux_dirent64_reclen(entry.name.len());
            if !bytes.is_empty() && bytes.len() + record_len > max_bytes {
                break;
            }
            if bytes.is_empty() && record_len > max_bytes {
                break;
            }

            let next_offset = description.offset + 1;
            push_linux_dirent64(&mut bytes, entry, next_offset as u64, record_len);
            description.offset = next_offset;
        }

        Ok(bytes)
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

    fn load_file_if_present(&mut self, path: &str) -> Result<bool, FileSystemError> {
        if let Some(contents) = virtual_file_contents(path) {
            self.files.insert(path.to_string(), contents);
            return Ok(true);
        }

        let Some(host_path) = self.host_path_for_read(path) else {
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

    fn is_directory(&self, path: &str) -> bool {
        if is_virtual_directory(path) {
            return true;
        }
        if path.is_empty() {
            return true;
        }

        self.host_path_for_read(path)
            .is_some_and(|host_path| host_path.is_dir())
    }

    fn directory_entries(&self, path: &str) -> Result<Vec<DirectoryEntry>, FileSystemError> {
        if let Some(entries) = virtual_directory_entries(path) {
            return Ok(entries);
        }

        let Some(host_path) = self.host_path_for_read(path) else {
            return Err(FileSystemError::NoSuchFile);
        };
        if !host_path.is_dir() {
            return Err(FileSystemError::InvalidInput);
        }

        let mut entries = vec![
            DirectoryEntry::directory(".", path),
            DirectoryEntry::directory("..", parent_path(path).unwrap_or("")),
        ];
        for entry in fs::read_dir(host_path).map_err(|_| FileSystemError::PermissionDenied)? {
            let entry = entry.map_err(|_| FileSystemError::PermissionDenied)?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy().into_owned();
            let entry_path = join_path(path, &name);
            let file_type = entry
                .file_type()
                .map_err(|_| FileSystemError::PermissionDenied)?;
            entries.push(if file_type.is_dir() {
                DirectoryEntry::directory(name, entry_path)
            } else {
                DirectoryEntry::regular(name, entry_path)
            });
        }

        Ok(entries)
    }

    fn persist_file(&self, path: &str) -> Result<(), FileSystemError> {
        let Some(host_path) = self.host_path_for_write(path) else {
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

    fn host_path_for_read(&self, path: &str) -> Option<PathBuf> {
        let mut fallback = None;
        for root in &self.host_roots {
            let host_path = host_path(root, path);
            if host_path.exists() {
                return Some(host_path);
            }
            fallback.get_or_insert(host_path);
        }
        for mount in &self.host_bind_mounts {
            let Some(host_path) = bind_host_path(mount, path) else {
                continue;
            };
            if host_path.exists() {
                return Some(host_path);
            }
            fallback.get_or_insert(host_path);
        }
        fallback
    }

    fn host_path_for_write(&self, path: &str) -> Option<PathBuf> {
        for mount in &self.host_bind_mounts {
            if let Some(host_path) = bind_host_path(mount, path) {
                return Some(host_path);
            }
        }

        self.host_roots.first().map(|root| host_path(root, path))
    }
}

fn bind_host_path(mount: &HostBindMount, path: &str) -> Option<PathBuf> {
    let suffix = if mount.guest_prefix.is_empty() {
        path
    } else if path == mount.guest_prefix {
        ""
    } else {
        path.strip_prefix(&mount.guest_prefix)
            .and_then(|path| path.strip_prefix('/'))?
    };

    Some(host_path(&mount.host_root, suffix))
}

fn host_path(root: &std::path::Path, path: &str) -> PathBuf {
    let mut host_path = root.to_path_buf();
    for component in path.split('/') {
        if component.is_empty() {
            continue;
        }
        host_path.push(component);
    }

    host_path
}

fn regular_metadata(path: &str, size: u64) -> FileMetadata {
    FileMetadata {
        dev: 1,
        ino: stable_inode(path),
        mode: S_IFREG | 0o666,
        size,
        block_size: 4096,
        blocks: size.div_ceil(512),
    }
}

fn directory_metadata(path: &str) -> FileMetadata {
    FileMetadata {
        dev: 1,
        ino: stable_inode(path),
        mode: S_IFDIR | 0o777,
        size: 0,
        block_size: 4096,
        blocks: 0,
    }
}

fn char_metadata() -> FileMetadata {
    FileMetadata {
        dev: 2,
        ino: 1,
        mode: S_IFCHR | 0o666,
        size: 0,
        block_size: 4096,
        blocks: 0,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryEntry {
    name: String,
    path: String,
    kind: DirectoryEntryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryEntryKind {
    Directory,
    Regular,
}

impl DirectoryEntry {
    fn directory(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            kind: DirectoryEntryKind::Directory,
        }
    }

    fn regular(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            kind: DirectoryEntryKind::Regular,
        }
    }

    fn dtype(&self) -> u8 {
        match self.kind {
            DirectoryEntryKind::Directory => 4,
            DirectoryEntryKind::Regular => 8,
        }
    }
}

fn linux_dirent64_reclen(name_len: usize) -> usize {
    (19 + name_len + 1).next_multiple_of(8)
}

fn push_linux_dirent64(
    bytes: &mut Vec<u8>,
    entry: &DirectoryEntry,
    offset: u64,
    record_len: usize,
) {
    let start = bytes.len();
    bytes.extend_from_slice(&stable_inode(&entry.path).to_le_bytes());
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.extend_from_slice(&(record_len as u16).to_le_bytes());
    bytes.push(entry.dtype());
    bytes.extend_from_slice(entry.name.as_bytes());
    bytes.push(0);
    bytes.resize(start + record_len, 0);
}

fn stable_inode(path: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in path.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    hash.max(2)
}

fn is_random_device_path(path: &str) -> bool {
    path == "dev/random" || path == "dev/urandom"
}

fn is_null_device_path(path: &str) -> bool {
    path == "dev/null"
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

fn normalize_mount_path(path: &str) -> Result<String, FileSystemError> {
    if path.is_empty() || path == "/" {
        return Ok(String::new());
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

    Ok(parts.join("/"))
}

fn parent_path(path: &str) -> Option<&str> {
    if path.is_empty() {
        return None;
    }
    path.rsplit_once('/').map(|(parent, _)| parent).or(Some(""))
}

fn join_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

const EMULATED_CPU_COUNT: usize = 1;

fn emulated_cpu_count() -> usize {
    EMULATED_CPU_COUNT
}

fn cpu_range_string() -> String {
    let count = emulated_cpu_count();
    if count == 1 {
        "0".to_string()
    } else {
        format!("0-{}", count - 1)
    }
}

fn virtual_file_contents(path: &str) -> Option<Vec<u8>> {
    let cpus = emulated_cpu_count();
    let text = match path {
        "proc/cpuinfo" => {
            let mut text = String::new();
            for cpu in 0..cpus {
                text.push_str(&format!(
                    "processor\t: {cpu}\nhart\t\t: {cpu}\nisa\t\t: rv64imafdc_zicsr_zifencei\nmmu\t\t: sv57\nuarch\t\t: riscvm\nmvendorid\t: 0x0\nmarchid\t\t: 0x0\nmimpid\t\t: 0x0\n\n"
                ));
            }
            text
        }
        "proc/stat" => {
            let mut text = "cpu  100 0 100 100000 0 0 0 0 0 0\n".to_string();
            for cpu in 0..cpus {
                text.push_str(&format!("cpu{cpu} 100 0 100 100000 0 0 0 0 0 0\n"));
            }
            text.push_str("intr 0\nctxt 0\nbtime 1700000000\nprocesses 1\nprocs_running 1\nprocs_blocked 0\n");
            text
        }
        "proc/meminfo" => {
            "MemTotal:        8388608 kB\nMemFree:         4194304 kB\nMemAvailable:    4194304 kB\n".to_string()
        }
        "proc/sys/kernel/random/uuid" => "00000000-0000-4000-8000-000000000000\n".to_string(),
        "sys/devices/system/cpu/online"
        | "sys/devices/system/cpu/possible"
        | "sys/devices/system/cpu/present" => format!("{}\n", cpu_range_string()),
        "sys/devices/system/cpu/kernel_max" => format!("{}\n", cpus - 1),
        "etc/os-release" => "NAME=\"riscvm Linux\"\nID=riscvm\nPRETTY_NAME=\"riscvm Linux\"\n".to_string(),
        "etc/lsb-release" => "DISTRIB_ID=riscvm\nDISTRIB_DESCRIPTION=\"riscvm Linux\"\n".to_string(),
        "sys/class/dmi/id/sys_vendor" => "riscvm\n".to_string(),
        "sys/class/dmi/id/product_name" => "RV64GC virtual machine\n".to_string(),
        "sys/class/dmi/id/board_vendor" => "riscvm\n".to_string(),
        "sys/class/dmi/id/board_name" => "RV64GC\n".to_string(),
        "sys/firmware/devicetree/base/model" => "riscvm RV64GC virtual machine\n".to_string(),
        _ => {
            if let Some(cpu) = cpu_sysfs_file(path, "online") {
                return (cpu < cpus).then(|| b"1\n".to_vec());
            }
            if let Some(cpu) = cpu_sysfs_file(path, "topology/core_id") {
                return (cpu < cpus).then(|| format!("{cpu}\n").into_bytes());
            }
            if let Some(cpu) = cpu_sysfs_file(path, "topology/physical_package_id") {
                return (cpu < cpus).then(|| b"0\n".to_vec());
            }
            if let Some(cpu) = cpu_sysfs_file(path, "topology/thread_siblings_list") {
                return (cpu < cpus).then(|| format!("{cpu}\n").into_bytes());
            }
            if let Some(cpu) = cpu_sysfs_file(path, "topology/core_siblings_list") {
                return (cpu < cpus).then(|| format!("{}\n", cpu_range_string()).into_bytes());
            }
            if let Some(cpu) = cpu_sysfs_file(path, "cpufreq/cpuinfo_max_freq")
                .or_else(|| cpu_sysfs_file(path, "cpufreq/scaling_max_freq"))
                .or_else(|| cpu_sysfs_file(path, "cpufreq/base_frequency"))
            {
                return (cpu < cpus).then(|| b"3200000\n".to_vec());
            }
            return None;
        }
    };

    Some(text.into_bytes())
}

fn cpu_sysfs_file(path: &str, suffix: &str) -> Option<usize> {
    let path = path.strip_prefix("sys/devices/system/cpu/cpu")?;
    let (cpu, rest) = path.split_once('/')?;
    (rest == suffix).then(|| cpu.parse().ok()).flatten()
}

fn is_virtual_directory(path: &str) -> bool {
    virtual_directory_entries(path).is_some()
}

fn virtual_directory_entries(path: &str) -> Option<Vec<DirectoryEntry>> {
    let cpus = emulated_cpu_count();
    let mut entries = vec![
        DirectoryEntry::directory(".", path),
        DirectoryEntry::directory("..", parent_path(path).unwrap_or("")),
    ];

    match path {
        "" => {
            for name in ["dev", "etc", "proc", "sys"] {
                entries.push(DirectoryEntry::directory(name, name));
            }
        }
        "proc" => {
            for name in ["cpuinfo", "meminfo", "stat", "sys"] {
                let entry_path = join_path(path, name);
                let entry = if name == "sys" {
                    DirectoryEntry::directory(name, entry_path)
                } else {
                    DirectoryEntry::regular(name, entry_path)
                };
                entries.push(entry);
            }
        }
        "proc/sys" => entries.push(DirectoryEntry::directory("kernel", "proc/sys/kernel")),
        "proc/sys/kernel" => entries.push(DirectoryEntry::directory(
            "random",
            "proc/sys/kernel/random",
        )),
        "proc/sys/kernel/random" => entries.push(DirectoryEntry::regular(
            "uuid",
            "proc/sys/kernel/random/uuid",
        )),
        "sys" => {
            for name in ["class", "devices", "firmware"] {
                entries.push(DirectoryEntry::directory(name, join_path(path, name)));
            }
        }
        "sys/class" => entries.push(DirectoryEntry::directory("net", "sys/class/net")),
        "sys/class/net" => entries.push(DirectoryEntry::directory("lo", "sys/class/net/lo")),
        "sys/class/net/lo" => {}
        "sys/class/dmi" => entries.push(DirectoryEntry::directory("id", "sys/class/dmi/id")),
        "sys/class/dmi/id" => {
            for name in ["board_name", "board_vendor", "product_name", "sys_vendor"] {
                entries.push(DirectoryEntry::regular(name, join_path(path, name)));
            }
        }
        "sys/devices" => entries.push(DirectoryEntry::directory("system", "sys/devices/system")),
        "sys/devices/system" => {
            entries.push(DirectoryEntry::directory("cpu", "sys/devices/system/cpu"))
        }
        "sys/devices/system/cpu" => {
            for cpu in 0..cpus {
                let name = format!("cpu{cpu}");
                entries.push(DirectoryEntry::directory(
                    name.clone(),
                    join_path(path, &name),
                ));
            }
            for name in ["kernel_max", "online", "possible", "present"] {
                entries.push(DirectoryEntry::regular(name, join_path(path, name)));
            }
        }
        "sys/firmware" => entries.push(DirectoryEntry::directory(
            "devicetree",
            "sys/firmware/devicetree",
        )),
        "sys/firmware/devicetree" => entries.push(DirectoryEntry::directory(
            "base",
            "sys/firmware/devicetree/base",
        )),
        "sys/firmware/devicetree/base" => entries.push(DirectoryEntry::regular(
            "model",
            "sys/firmware/devicetree/base/model",
        )),
        "etc" => {
            for name in ["lsb-release", "os-release"] {
                entries.push(DirectoryEntry::regular(name, join_path(path, name)));
            }
        }
        _ => {
            if let Some(cpu) = path
                .strip_prefix("sys/devices/system/cpu/cpu")
                .and_then(|value| value.parse::<usize>().ok())
            {
                if cpu >= cpus {
                    return None;
                }
                entries.push(DirectoryEntry::regular("online", join_path(path, "online")));
                entries.push(DirectoryEntry::directory(
                    "topology",
                    join_path(path, "topology"),
                ));
                entries.push(DirectoryEntry::directory(
                    "cpufreq",
                    join_path(path, "cpufreq"),
                ));
            } else if let Some(cpu) = path
                .strip_prefix("sys/devices/system/cpu/cpu")
                .and_then(|path| path.strip_suffix("/topology"))
                .and_then(|value| value.parse::<usize>().ok())
            {
                if cpu >= cpus {
                    return None;
                }
                for name in [
                    "core_id",
                    "core_siblings_list",
                    "physical_package_id",
                    "thread_siblings_list",
                ] {
                    entries.push(DirectoryEntry::regular(name, join_path(path, name)));
                }
            } else if let Some(cpu) = path
                .strip_prefix("sys/devices/system/cpu/cpu")
                .and_then(|path| path.strip_suffix("/cpufreq"))
                .and_then(|value| value.parse::<usize>().ok())
            {
                if cpu >= cpus {
                    return None;
                }
                for name in ["base_frequency", "cpuinfo_max_freq", "scaling_max_freq"] {
                    entries.push(DirectoryEntry::regular(name, join_path(path, name)));
                }
            } else {
                return None;
            }
        }
    }

    Some(entries)
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
    fn mounted_host_directory_at_maps_absolute_guest_prefixes() {
        let root = temp_mount("host-bind");
        fs::write(root.join("resource.plar"), b"archive").unwrap();
        let mut fs = GuestFileSystem::new();
        fs.mount_host_directory_at("/private/tmp/app", root.clone())
            .unwrap();

        let fd = fs.open("/private/tmp/app/resource.plar", 0).unwrap();
        let mut buffer = [0; 16];
        assert_eq!(fs.read(fd, &mut buffer), Ok(7));
        assert_eq!(&buffer[..7], b"archive");

        let out_fd = fs.open("/private/tmp/app/output.txt", 0o100 | 0o2).unwrap();
        assert_eq!(fs.write(out_fd, b"host-visible"), Ok(12));
        assert_eq!(fs::read(root.join("output.txt")).unwrap(), b"host-visible");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn virtual_dev_random_and_null_behave_like_character_devices() {
        let mut fs = GuestFileSystem::new();

        let random_fd = fs.open("/dev/urandom", 0).unwrap();
        let mut random = [0; 32];
        assert_eq!(fs.read(random_fd, &mut random), Ok(32));
        assert_eq!(fs.stat_fd(random_fd).unwrap().mode & 0o170000, 0o020000);

        let null_fd = fs.open("/dev/null", 0o2).unwrap();
        assert_eq!(fs.write(null_fd, b"discarded"), Ok(9));
        assert_eq!(fs.stat_fd(null_fd).unwrap().mode & 0o170000, 0o020000);
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
