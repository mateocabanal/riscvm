#[derive(Debug)]
pub struct FCSR {
    pub frm: RoundingMode,
    fflags: u8,
}

impl FCSR {
    pub fn new() -> Self {
        FCSR {
            frm: RoundingMode::Rne,
            fflags: 0,
        }
    }

    pub const NV: u8 = 1 << 4;
    pub const DZ: u8 = 1 << 3;
    pub const OF: u8 = 1 << 2;
    pub const UF: u8 = 1 << 1;
    pub const NX: u8 = 1 << 0;

    pub fn set_flag(&mut self, flag: u8) {
        self.fflags |= flag;
    }

    pub fn flags(&self) -> u8 {
        self.fflags
    }

    pub fn set_flags(&mut self, value: u8) {
        self.fflags = value & 0x1f;
    }

    pub fn bits(&self) -> u64 {
        u64::from(self.fflags) | (u64::from(self.frm.bits()) << 5)
    }

    pub fn write_bits(&mut self, value: u64) -> Result<(), ()> {
        let frm_bits = ((value >> 5) & 0b111) as u8;
        let Some(frm) = RoundingMode::from_bits(frm_bits) else {
            return Err(());
        };
        self.frm = frm;
        self.set_flags(value as u8);
        Ok(())
    }

    pub fn set_rounding_mode_bits(&mut self, value: u8) -> Result<(), ()> {
        let Some(frm) = RoundingMode::from_bits(value & 0b111) else {
            return Err(());
        };
        self.frm = frm;
        Ok(())
    }

    pub fn handle_exceptions(&mut self, exact: f64, rounded: f64) {
        if exact.is_nan() {
            self.set_flag(FCSR::NV);
        }
        if exact.is_infinite() {
            self.set_flag(FCSR::OF);
        }
        if (rounded - exact).abs() > 0.0 {
            self.set_flag(FCSR::NX);
        }
    }
}

/// Represents the RISC-V floating-point rounding modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    Rne, // Round to Nearest, ties to Even
    Rtz, // Round Toward Zero
    Rdn, // Round Down (toward -∞)
    Rup, // Round Up (toward +∞)
    Rmm, // Round to Nearest, ties to Max Magnitude
}

impl RoundingMode {
    pub fn bits(self) -> u8 {
        match self {
            Self::Rne => 0b000,
            Self::Rtz => 0b001,
            Self::Rdn => 0b010,
            Self::Rup => 0b011,
            Self::Rmm => 0b100,
        }
    }

    pub fn from_bits(value: u8) -> Option<Self> {
        Some(match value {
            0b000 => Self::Rne,
            0b001 => Self::Rtz,
            0b010 => Self::Rdn,
            0b011 => Self::Rup,
            0b100 => Self::Rmm,
            _ => return None,
        })
    }
}

impl From<&u8> for RoundingMode {
    fn from(value: &u8) -> Self {
        Self::from_bits(*value).expect("unknown rounding mode")
    }
}

impl From<u8> for RoundingMode {
    fn from(value: u8) -> Self {
        Self::from_bits(value).expect("unknown rounding mode")
    }
}

fn fround_rtz(value: f32) -> f32 {
    value.trunc()
}

fn fround_rdn(value: f32) -> f32 {
    value.floor()
}

fn fround_rup(value: f32) -> f32 {
    value.ceil()
}

fn fround_rmm(value: f32) -> f32 {
    let rounded = value.round();
    let frac_part = value.fract();

    if frac_part.abs() == 0.5 {
        // Tie case: round away from zero
        if value > 0.0 {
            rounded + 1.0
        } else {
            rounded - 1.0
        }
    } else {
        rounded
    }
}

fn fround_rtz64(value: f64) -> f64 {
    value.trunc()
}

fn fround_rdn64(value: f64) -> f64 {
    value.floor()
}

fn fround_rup64(value: f64) -> f64 {
    value.ceil()
}

fn fround_rmm64(value: f64) -> f64 {
    let rounded = value.round();
    let frac_part = value.fract();

    if frac_part.abs() == 0.5 {
        if value > 0.0 {
            rounded + 1.0
        } else {
            rounded - 1.0
        }
    } else {
        rounded
    }
}

pub fn round_f32(value: f32, rounding_mode: RoundingMode) -> f32 {
    match rounding_mode {
        RoundingMode::Rne => value,
        RoundingMode::Rtz => fround_rtz(value),
        RoundingMode::Rdn => fround_rdn(value),
        RoundingMode::Rup => fround_rup(value),
        RoundingMode::Rmm => fround_rmm(value),
    }
}

pub fn round_f64(value: f64, rounding_mode: RoundingMode) -> f64 {
    match rounding_mode {
        RoundingMode::Rne => value,
        RoundingMode::Rtz => fround_rtz64(value),
        RoundingMode::Rdn => fround_rdn64(value),
        RoundingMode::Rup => fround_rup64(value),
        RoundingMode::Rmm => fround_rmm64(value),
    }
}

pub fn classify_f32(value: f32) -> u32 {
    let mut result = 0u32;

    if value.is_nan() {
        if value.is_snan() {
            result |= 1 << 8; // Signaling NaN
        } else {
            result |= 1 << 9; // Quiet NaN
        }
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            result |= 1 << 0; // Negative infinity
        } else {
            result |= 1 << 7; // Positive infinity
        }
    } else if value == 0.0 {
        if value.is_sign_negative() {
            result |= 1 << 3; // Negative zero
        } else {
            result |= 1 << 4; // Positive zero
        }
    } else if value.is_subnormal() {
        if value.is_sign_negative() {
            result |= 1 << 2; // Negative subnormal
        } else {
            result |= 1 << 5; // Positive subnormal
        }
    } else {
        // Normal number
        if value.is_sign_negative() {
            result |= 1 << 1; // Negative normal
        } else {
            result |= 1 << 6; // Positive normal
        }
    }

    result
}

pub fn classify_f64(value: f64) -> u32 {
    let mut result = 0u32;

    if value.is_nan() {
        if value.is_snan() {
            result |= 1 << 8;
        } else {
            result |= 1 << 9;
        }
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            result |= 1 << 0;
        } else {
            result |= 1 << 7;
        }
    } else if value == 0.0 {
        if value.is_sign_negative() {
            result |= 1 << 3;
        } else {
            result |= 1 << 4;
        }
    } else if value.is_subnormal() {
        if value.is_sign_negative() {
            result |= 1 << 2;
        } else {
            result |= 1 << 5;
        }
    } else if value.is_sign_negative() {
        result |= 1 << 1;
    } else {
        result |= 1 << 6;
    }

    result
}

pub trait FloatExtends {
    fn is_snan(&self) -> bool;
}

impl FloatExtends for f32 {
    fn is_snan(&self) -> bool {
        let bits = self.to_bits();
        let exponent = (bits >> 23) & 0xFF;
        let fraction = bits & 0x7FFFFF;

        exponent == 0xFF && fraction != 0 && (fraction & (1 << 22)) == 0
    }
}

impl FloatExtends for f64 {
    fn is_snan(&self) -> bool {
        let bits = self.to_bits();
        let exponent = (bits >> 52) & 0x7ff;
        let fraction = bits & 0x000f_ffff_ffff_ffff;

        exponent == 0x7ff && fraction != 0 && (fraction & (1 << 51)) == 0
    }
}
