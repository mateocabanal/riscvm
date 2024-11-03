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
#[derive(Debug, Clone, Copy)]
pub enum RoundingMode {
    Rne, // Round to Nearest, ties to Even
    Rtz, // Round Toward Zero
    Rdn, // Round Down (toward -∞)
    Rup, // Round Up (toward +∞)
    Rmm, // Round to Nearest, ties to Max Magnitude
}

impl From<&u8> for RoundingMode {
    fn from(value: &u8) -> Self {
        use RoundingMode::*;
        match value {
            0b000 => Rne,
            0b001 => Rtz,
            0b010 => Rdn,
            0b011 => Rup,
            0b100 => Rmm,
            _ => panic!("unknown rounding mode"),
        }
    }
}

impl From<u8> for RoundingMode {
    fn from(value: u8) -> Self {
        use RoundingMode::*;
        match value {
            0b000 => Rne,
            0b001 => Rtz,
            0b010 => Rdn,
            0b011 => Rup,
            0b100 => Rmm,
            _ => panic!("unknown rounding mode"),
        }
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

fn fround_rne(value: f32) -> f32 {
    let rounded = value.round();
    let frac_part = value.fract();

    if frac_part.abs() == 0.5 {
        // Tie case: round to even
        if rounded % 2.0 != 0.0 {
            if value > 0.0 {
                rounded - 1.0
            } else {
                rounded + 1.0
            }
        } else {
            rounded
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
