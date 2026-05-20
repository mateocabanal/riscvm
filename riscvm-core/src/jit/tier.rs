use std::fmt;

pub(crate) const DEFAULT_HOT_THRESHOLD: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitTier {
    Baseline,
    Optimized,
    Trace,
}

impl JitTier {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Optimized => "optimized",
            Self::Trace => "trace",
        }
    }
}

impl fmt::Display for JitTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
