.global _start
_start:
  addi sp, sp, -32
  addi t0, t0, 97
  sb t0, 28(sp)
  addi t0, t0, 1
  sb t0, 29(sp)
  addi t0, t0, 1
  sb t0, 30(sp)
  addi t0, t0, 1
  sb t0, 31(sp)

  addi a7, zero, 64
  addi a1, sp, 28
  addi a2, zero, 4
  ecall

  addi a7, zero, 93
  addi a1, zero, 0
  ecall
