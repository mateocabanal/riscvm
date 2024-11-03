void __exit(void) {
  asm("addi a7, zero, 93\n"
      "addi a0, zero, 0\n"
      "ecall" ::
          :);
  __builtin_unreachable();
}

void print_i32(int *a) {
  asm("addi a7, zero, 1101\n"
      "add a0, zero, %[ptr]\n"
      "ecall" ::[ptr] "r"((void *)a));
}

void print_float(float *a) {
  asm("addi a7, zero, 1110\n"
      "add a0, zero, %[ptr]\n"
      "ecall" ::[ptr] "r"((void *)a));
}

void _start() {
  float a = 1.3f;
  float b = 15.4f;
  float c = a + b;

  float sub_c = a - b;

  int z = c;

  print_float(&c);
  print_i32(&z);

  print_float(&sub_c);
  __exit();
}
