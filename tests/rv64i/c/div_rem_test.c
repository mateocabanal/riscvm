void rvm_print_u64_mem(unsigned long long a) {
  asm("addi a7, x0, 1000\n"
      "add a0, x0, %[ptr]\n"
      "ecall" ::[ptr] "r"(a));
}

void exit(int code) {
  asm("addi a7, x0, 93\n"
      "add a0, zero, %[code]\n"
      "ecall" ::[code] "r"(code));
  __builtin_unreachable();
}

void _start() {
  int divident = 50;
  int divisor = 5;

  int result = 50 % 35;
  rvm_print_u64_mem(result);

  exit(0);
}
