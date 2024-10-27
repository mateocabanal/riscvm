.global _start
_start:
  addi x5, x5, 50
  add x6, x5, x5
  sub x7, x6, x5
  xori x7, x7, 18
  addi x4, x4, -10
  srai x4, x4, 2

_exit:
  xor a0, a0, a0
  xor a7, a7, a7
  addi a7, a7, 93
  ecall
