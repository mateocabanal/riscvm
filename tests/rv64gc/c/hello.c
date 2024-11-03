
void print_str(char *str, int len) {
  asm("addi a7, zero, 64\n"
      "add a1, zero, %[ptr]\n"
      "add a2, zero, %[len]\n"
      "ecall" ::[ptr] "r"(str),
      [len] "r"(len));
}

int main() {
  print_str("Hello World\n", 13);
  return 0;
}
