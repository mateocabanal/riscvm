.global _start
_start:
  addi sp, sp, -32

  addi x9, x0, 2
  addi x10, x0, 34
  sb x9, 28(sp)
  sb x9, 29(sp)
  sb x9, 30(sp)
  sb x10, 31(sp)
  lw x11, 28(sp)

  addi a7, zero, 93
  addi a0, zero, 0
  ecall
