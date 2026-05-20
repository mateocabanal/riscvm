use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitStartupProfileEntry {
    pub pc: u64,
    pub count: u64,
}

impl JitStartupProfileEntry {
    pub fn new(pc: u64, count: u64) -> Self {
        Self {
            pc,
            count: count.max(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JitStartupProfileParseError {
    line: usize,
    field: &'static str,
    value: String,
}

impl fmt::Display for JitStartupProfileParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid JIT startup profile {} on line {}: {}",
            self.field, self.line, self.value
        )
    }
}

impl std::error::Error for JitStartupProfileParseError {}

pub fn parse_startup_profile(
    text: &str,
) -> Result<Vec<JitStartupProfileEntry>, JitStartupProfileParseError> {
    let mut entries = Vec::new();

    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut pc = None;
        let mut count = None;
        for token in line
            .split_whitespace()
            .map(|token| token.trim_end_matches(','))
        {
            if let Some(value) = token.strip_prefix("pc=") {
                pc = Some(parse_u64_field(line_number, "pc", value)?);
                continue;
            }
            if let Some(value) = token.strip_prefix("count=") {
                count = Some(parse_u64_field(line_number, "count", value)?);
                continue;
            }

            if pc.is_none() {
                if let Some(value) = parse_u64_literal(token) {
                    pc = Some(value);
                }
            } else if count.is_none() {
                if let Some(value) = parse_u64_literal(token) {
                    count = Some(value);
                }
            }
        }

        if let Some(pc) = pc {
            entries.push(JitStartupProfileEntry::new(pc, count.unwrap_or(1)));
        }
    }

    Ok(normalize_startup_profile(entries))
}

pub fn format_startup_profile(entries: &[JitStartupProfileEntry]) -> String {
    let mut profile = String::from("# riscvm jit startup profile v1\n");
    for entry in normalize_startup_profile(entries.iter().copied()) {
        profile.push_str(&format!("pc=0x{:016x} count={}\n", entry.pc, entry.count));
    }
    profile
}

pub fn normalize_startup_profile<I>(entries: I) -> Vec<JitStartupProfileEntry>
where
    I: IntoIterator<Item = JitStartupProfileEntry>,
{
    let mut entries: Vec<_> = entries.into_iter().collect();
    entries.sort_by(|lhs, rhs| rhs.count.cmp(&lhs.count).then(lhs.pc.cmp(&rhs.pc)));

    let mut normalized: Vec<JitStartupProfileEntry> = Vec::with_capacity(entries.len());
    for entry in entries {
        if normalized.iter().any(|existing| existing.pc == entry.pc) {
            continue;
        }
        normalized.push(entry);
    }
    normalized
}

fn parse_u64_field(
    line: usize,
    field: &'static str,
    value: &str,
) -> Result<u64, JitStartupProfileParseError> {
    parse_u64_literal(value).ok_or_else(|| JitStartupProfileParseError {
        line,
        field,
        value: value.to_string(),
    })
}

fn parse_u64_literal(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_startup_profile_and_profile_report_lines() {
        let profile = "\
# riscvm jit startup profile v1
pc=0x20 count=9
0x10 4
[profile]   pc=0x0000000000000030 count=2 instructions=3
";

        let entries = parse_startup_profile(profile).unwrap();

        assert_eq!(
            entries,
            vec![
                JitStartupProfileEntry::new(0x20, 9),
                JitStartupProfileEntry::new(0x10, 4),
                JitStartupProfileEntry::new(0x30, 2),
            ]
        );
    }

    #[test]
    fn formats_startup_profile_hot_first() {
        let profile = format_startup_profile(&[
            JitStartupProfileEntry::new(0x10, 1),
            JitStartupProfileEntry::new(0x20, 7),
        ]);

        assert!(profile.starts_with("# riscvm jit startup profile v1\n"));
        assert!(profile.contains("pc=0x0000000000000020 count=7\n"));
        assert!(
            profile.find("0x0000000000000020").unwrap()
                < profile.find("0x0000000000000010").unwrap()
        );
    }
}
