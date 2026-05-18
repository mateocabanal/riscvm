    .section .text
    .global _start
    .option push
    .option norvc

_start:
    li t0, 500000
    li t1, 1
    li t2, 0

1:
    add t2, t2, t1
    xori t1, t1, 0x5a5
    addi t0, t0, -1
    bne t0, zero, 1b

    andi a0, t2, 0
    li a7, 93
    ecall

    .option pop
