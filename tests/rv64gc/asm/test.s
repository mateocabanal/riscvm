.section .text
.global _start

# Entry point
_start:
    # Set up the stack pointer and use x8 as a temporary base register
    addi sp, sp, -64           # Allocate 64 bytes on the stack
    add x8, sp, x0             # Use x8 as a base register for stack operations

    # Call each test subroutine
    jal test_addw
    jal test_subw
    jal test_sllw
    jal test_sraw
    jal test_slti
    jal test_sltiu
    jal test_slt
    jal test_sltu
    jal test_clw
    jal test_caddi16sp

    # If all tests pass, branch to success
    jal pass

# Subroutines for each test

test_addw:
    li x9, 5
    li x10, 10
    addw x13, x10, x9          # x13 = x10 + x9 = 15
    li x17, 15
    li x10, 1                  # Set exit code to 1 for addw test failure
    bne x13, x17, fail
    ret

test_subw:
    li x9, 5
    li x11, 20
    subw x14, x11, x9          # x14 = x11 - x9 = 15
    li x17, 15
    li x10, 2                  # Set exit code to 2 for subw test failure
    bne x14, x17, fail
    ret

test_sllw:
    li x9, 5
    sllw x15, x9, 1            # x15 = x9 << 1 = 10
    li x17, 10
    li x10, 3                  # Set exit code to 3 for sllw test failure
    bne x15, x17, fail
    ret

test_sraw:
    li x12, -15
    sraw x16, x12, 1           # x16 = x12 >> 1 = -8
    li x17, -8
    li x10, 4                  # Set exit code to 4 for sraw test failure
    bne x16, x17, fail
    ret

test_slti:
    li x9, 5
    slti x20, x9, 10           # x20 = (x9 < 10) ? 1 : 0 = 1
    li x17, 1
    li x10, 5                  # Set exit code to 5 for slti test failure
    bne x20, x17, fail
    ret

test_sltiu:
    li x9, 5
    sltiu x21, x9, 10          # x21 = (x9 < 10 unsigned) ? 1 : 0 = 1
    li x17, 1
    li x10, 6                  # Set exit code to 6 for sltiu test failure
    bne x21, x17, fail
    ret

test_slt:
    li x9, 5
    li x10, 10
    slt x22, x9, x10           # x22 = (x9 < x10) ? 1 : 0 = 1
    li x17, 1
    li x10, 7                  # Set exit code to 7 for slt test failure
    bne x22, x17, fail
    ret

test_sltu:
    li x9, 5
    li x10, 10
    sltu x23, x9, x10          # x23 = (x9 < x10 unsigned) ? 1 : 0 = 1
    li x17, 1
    li x10, 8                  # Set exit code to 8 for sltu test failure
    bne x23, x17, fail
    ret

test_clw:
    # Store and load a value using compressed instructions
    li x17, 15
    sd x17, 0(x8)              # Store 15 at stack[0]
    c.lw x13, 0(x8)            # Load x13 from stack[0] (should be 15)
    li x10, 10                 # Set exit code to 10 for c.lw test failure
    bne x13, x17, fail
    ret

test_caddi16sp:
    li x10, 11
    c.mv x9, x8
    c.addi16sp sp, 16
    c.addi x9, 16
    bne x9, sp, fail
    ret


# Common success and failure handlers

dump_regs:
    li x17, 1001
    ecall
    ret

fail:
    jal dump_regs
    li x17, 93                 # Syscall number for exit
    ecall                      # Exit with failure

pass:
    li x10, 0                  # Success exit code
    li x17, 93                 # Syscall number for exit
    ecall                      # Exit with success

