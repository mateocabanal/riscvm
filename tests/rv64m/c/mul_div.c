void rvm_print_u64_mem(unsigned long long a) {
  asm("addi a7, x0, 1000\n"
      "add a0, x0, %[ptr]\n"
      "ecall" ::[ptr] "r"(a));
}

void _exit(int code) {
  asm("addi a7, x0, 93\n"
      "add a0, zero, %[code]\n"
      "ecall" ::[code] "r"(code));
  __builtin_unreachable();
}

void _start() {
  int a = 100;
  int b = 30;

  int res = a / b;
  int mul = a * b;
  rvm_print_u64_mem(res);
  rvm_print_u64_mem(mul);
  _exit(0);
}
