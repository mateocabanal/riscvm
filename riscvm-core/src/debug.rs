use std::{
    fmt,
    fs::{File, OpenOptions},
    io::{self, Write},
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::Duration,
};

static DEBUG_FILE: OnceLock<Mutex<File>> = OnceLock::new();
static TERMINATION_SIGNAL: OnceLock<Arc<AtomicUsize>> = OnceLock::new();
static SIGNAL_HANDLERS_INSTALLED: OnceLock<()> = OnceLock::new();

const SIGNAL_EXIT_GRACE: Duration = Duration::from_secs(2);

pub fn init_debug_file(path: impl AsRef<Path>) -> io::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    DEBUG_FILE
        .set(Mutex::new(file))
        .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "debug file already set"))?;
    line(format_args!("[debug] debug file initialized"));
    Ok(())
}

pub fn debug_file_enabled() -> bool {
    DEBUG_FILE.get().is_some()
}

pub fn install_signal_handlers() -> io::Result<()> {
    if SIGNAL_HANDLERS_INSTALLED.get().is_some() {
        return Ok(());
    }

    #[cfg(unix)]
    {
        use signal_hook::{
            consts::{SIGHUP, SIGINT, SIGQUIT, SIGTERM},
            flag,
            iterator::Signals,
        };

        let signal = termination_signal_flag();
        let watched_signals = [SIGINT, SIGTERM, SIGHUP, SIGQUIT];
        for sig in watched_signals {
            flag::register_usize(sig, Arc::clone(signal), sig as usize)
                .map_err(io::Error::other)?;
        }

        let mut signals = Signals::new(watched_signals).map_err(io::Error::other)?;
        let signal = Arc::clone(signal);
        thread::Builder::new()
            .name("riscvm-signal-flusher".to_string())
            .spawn(move || {
                for sig in signals.forever() {
                    signal.store(sig as usize, Ordering::SeqCst);
                    flush();
                    thread::sleep(SIGNAL_EXIT_GRACE);
                    if signal.load(Ordering::SeqCst) == sig as usize {
                        line(format_args!(
                            "[signal] received {} ({sig}); flushed debug output before forced exit",
                            signal_name(sig)
                        ));
                        flush();
                        std::process::exit(128 + sig);
                    }
                }
            })
            .map_err(io::Error::other)?;
    }

    let _ = SIGNAL_HANDLERS_INSTALLED.set(());
    Ok(())
}

pub fn termination_signal() -> Option<i32> {
    let signal = termination_signal_flag().load(Ordering::SeqCst);
    (signal != 0).then_some(signal as i32)
}

pub fn termination_requested() -> bool {
    termination_signal().is_some()
}

pub fn signal_name(signal: i32) -> &'static str {
    #[cfg(unix)]
    {
        use signal_hook::consts::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};

        match signal {
            SIGINT => "SIGINT",
            SIGTERM => "SIGTERM",
            SIGHUP => "SIGHUP",
            SIGQUIT => "SIGQUIT",
            _ => "signal",
        }
    }

    #[cfg(not(unix))]
    {
        let _ = signal;
        "signal"
    }
}

pub fn line(args: fmt::Arguments<'_>) {
    let _ = write_with_newline(args);
}

pub fn write(args: fmt::Arguments<'_>) {
    let _ = write_args(args);
}

pub fn flush() {
    if let Some(file) = DEBUG_FILE.get() {
        if let Ok(mut file) = file.lock() {
            let _ = file.flush();
            let _ = file.sync_all();
        }
    } else {
        let _ = io::stderr().lock().flush();
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DebugWriter;

impl Write for DebugWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write_bytes(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        flush();
        Ok(())
    }
}

fn termination_signal_flag() -> &'static Arc<AtomicUsize> {
    TERMINATION_SIGNAL.get_or_init(|| Arc::new(AtomicUsize::new(0)))
}

fn write_with_newline(args: fmt::Arguments<'_>) -> io::Result<()> {
    if let Some(file) = DEBUG_FILE.get() {
        let mut file = file
            .lock()
            .map_err(|_| io::Error::other("debug file lock poisoned"))?;
        file.write_fmt(args)?;
        file.write_all(b"\n")?;
        file.flush()?;
        let _ = file.sync_all();
        return Ok(());
    }

    let mut stderr = io::stderr().lock();
    stderr.write_fmt(args)?;
    stderr.write_all(b"\n")?;
    stderr.flush()
}

fn write_args(args: fmt::Arguments<'_>) -> io::Result<()> {
    let mut writer = DebugWriter;
    writer.write_fmt(args)?;
    writer.flush()
}

fn write_bytes(bytes: &[u8]) -> io::Result<()> {
    if let Some(file) = DEBUG_FILE.get() {
        let mut file = file
            .lock()
            .map_err(|_| io::Error::other("debug file lock poisoned"))?;
        file.write_all(bytes)?;
        file.flush()?;
        let _ = file.sync_all();
        return Ok(());
    }

    let mut stderr = io::stderr().lock();
    stderr.write_all(bytes)?;
    stderr.flush()
}
