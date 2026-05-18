    .section .bss
    .balign 8
buffer:
    .zero 2048

    .section .text
    .global _start
    .option push
    .option norvc
    .option norelax

_start:
    la t3, buffer
    li t0, 250000
    li t1, 0
    li t2, 0

1:
    andi t4, t2, 2040
    add t5, t3, t4
    sd t1, 0(t5)
    ld t6, 0(t5)
    xor t1, t1, t6
    addi t1, t1, 1
    addi t2, t2, 8
    addi t0, t0, -1
    bne t0, zero, 1b

    andi a0, t1, 0
    li a7, 93
    ecall

    .option pop
