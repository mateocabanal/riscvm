use super::common::*;
use crate::cpu::RV64GCInstruction::*;

type Amo = fn(u8, u8, u8) -> crate::cpu::RV64GCInstruction;

fn run_word_amo(make: Amo, initial: u32, operand: u64) -> (u64, u32) {
    let mut cpu = cpu_with_memory();
    cpu.registers[RS1 as usize] = BASE;
    cpu.registers[RS2 as usize] = operand;
    cpu.ram.write_word(BASE, initial).unwrap();

    make(RD, RS1, RS2).execute_instruction(&mut cpu);

    (cpu.registers[RD as usize], cpu.ram.read_word(BASE).unwrap())
}

fn run_doubleword_amo(make: Amo, initial: u64, operand: u64) -> (u64, u64) {
    let mut cpu = cpu_with_memory();
    cpu.registers[RS1 as usize] = BASE;
    cpu.registers[RS2 as usize] = operand;
    cpu.ram.write_doubleword(BASE, initial).unwrap();

    make(RD, RS1, RS2).execute_instruction(&mut cpu);

    (
        cpu.registers[RD as usize],
        cpu.ram.read_doubleword(BASE).unwrap(),
    )
}

#[test]
fn rv64a_lr_sc_word_and_doubleword_single_hart_success_path() {
    let mut cpu = cpu_with_memory();
    cpu.registers[RS1 as usize] = BASE;
    cpu.registers[RS2 as usize] = 0x1122_3344_5566_7788;
    cpu.ram.write_word(BASE, 0x8000_0001).unwrap();

    Lrw(RD, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], sx32(0x8000_0001));

    Scw(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);
    assert_eq!(cpu.ram.read_word(BASE).unwrap(), 0x5566_7788);

    cpu.ram
        .write_doubleword(BASE, 0x8877_6655_4433_2211)
        .unwrap();
    Lrd(RD, RS1).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0x8877_6655_4433_2211);

    Scd(RD, RS1, RS2).execute_instruction(&mut cpu);
    assert_eq!(cpu.registers[RD as usize], 0);
    assert_eq!(
        cpu.ram.read_doubleword(BASE).unwrap(),
        0x1122_3344_5566_7788
    );
}

#[test]
fn rv64a_word_amo_ops_return_original_sign_extended_word() {
    let cases: &[(Amo, u32, u64, u32)] = &[
        (Amoswapw, 0x8000_0001, 0x1234_5678, 0x1234_5678),
        (Amoaddw, 10, 5, 15),
        (Amoxorw, 0b1010, 0b1100, 0b0110),
        (Amoandw, 0b1010, 0b1100, 0b1000),
        (Amoorw, 0b1010, 0b1100, 0b1110),
        (Amominw, 0xffff_fff0, 5, 0xffff_fff0),
        (Amomaxw, 0xffff_fff0, 5, 5),
        (Amominuw, 10, 5, 5),
        (Amomaxuw, 10, 5, 10),
    ];

    for (make, initial, operand, expected_mem) in cases {
        let (rd, mem) = run_word_amo(*make, *initial, *operand);
        assert_eq!(rd, sx32(*initial));
        assert_eq!(mem, *expected_mem);
    }
}

#[test]
fn rv64a_doubleword_amo_ops_return_original_doubleword() {
    let cases: &[(Amo, u64, u64, u64)] = &[
        (
            Amoswapd,
            0x8000_0000_0000_0001,
            0x1234_5678_9abc_def0,
            0x1234_5678_9abc_def0,
        ),
        (Amoaddd, 10, 5, 15),
        (Amoxord, 0b1010, 0b1100, 0b0110),
        (Amoandd, 0b1010, 0b1100, 0b1000),
        (Amoord, 0b1010, 0b1100, 0b1110),
        (Amomind, u64::MAX - 15, 5, u64::MAX - 15),
        (Amomaxd, u64::MAX - 15, 5, 5),
        (Amominud, 10, 5, 5),
        (Amomaxud, 10, 5, 10),
    ];

    for (make, initial, operand, expected_mem) in cases {
        let (rd, mem) = run_doubleword_amo(*make, *initial, *operand);
        assert_eq!(rd, *initial);
        assert_eq!(mem, *expected_mem);
    }
}
