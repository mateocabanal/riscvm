void print_str(char *msg, int len) {
  asm("addi a7, x0, 64\n"
      "add a0, zero, 1\n"
      "add a1, zero, %[ptr]\n"
      "add a2, x0, %1\n"
      "ecall" ::[ptr] "r"(msg),
      "r"(len));
}

void rvm_print_u64_mem(unsigned long long a) {
  asm("addi a7, x0, 1000\n"
      "add a0, x0, %[ptr]\n"
      "ecall" ::[ptr] "r"(a));
}

int hw_mul(int a, int b) {
  int val = 0;

  asm("mul %[output], %[rs1], %[rs2]\n"
      : [output] "=r"(val)
      : [rs1] "r"(a), [rs2] "r"(b));

  return val;
}

int hw_div(int a, int b) {
  int val = 0;

  asm("div %[output], %[rs1], %[rs2]\n"
      : [output] "=r"(val)
      : [rs1] "r"(a), [rs2] "r"(b));

  return val;
}

int hw_mod(int a, int b) {
  int val = 0;

  asm("rem %[output], %[rs1], %[rs2]\n"
      : [output] "=r"(val)
      : [rs1] "r"(a), [rs2] "r"(b));

  return val;
}

int reverse_int(int num) {
  int reversed = 0;

  while (num != 0) {
    int digit = hw_mod(num, 10); // Extract the last digit
    reversed =
        hw_mul(reversed, 10) + digit; // Append the digit to reversed number
    num = hw_div(num, 10);            // Remove the last digit from num
  }

  return reversed;
}

void print_int(int num) {
  char buf[50] = {};
  int reversed = reverse_int(num);

  int i = 0;
  print_str("rem: ", 5);
  while (reversed != 0) {
    int rem = hw_mod(reversed, 10);
    rvm_print_u64_mem(rem);
    buf[i] = rem + 48;
    reversed = hw_div(reversed, 10);
    i++;
  }

  print_str(buf, i);
}

void exit(int code) {
  asm("addi a7, x0, 93\n"
      "add a0, zero, %[code]\n"
      "ecall" ::[code] "r"(code));
  __builtin_unreachable();
}

void fib(int n) {
  int t1 = 0, t2 = 1, nextTerm = 1;
  for (int i = 3; i <= n; ++i) {
    print_int(nextTerm);

    if (i != n)
      print_str(", ", 2);

    t1 = t2;
    t2 = nextTerm;
    nextTerm = t1 + t2;
  }
  print_str("\n", 1);
}

void _start() {
  char *msg = "Hello, World\n";
  print_str(msg, 13);
  print_int(10);
  print_str("\n", 1);
  fib(35);
  exit(0);
}
