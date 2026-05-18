    .section .text
    .global _start
    .option push
    .option norvc

_start:
    li t0, 500000
    li t1, 0
    li t2, 0

1:
    andi t3, t0, 1
    beq t3, zero, 2f
    addi t1, t1, 3
    jal zero, 3f

2:
    addi t1, t1, 7

3:
    addi t2, t2, 1
    addi t0, t0, -1
    bne t0, zero, 1b

    andi a0, t1, 0
    li a7, 93
    ecall

    .option pop
