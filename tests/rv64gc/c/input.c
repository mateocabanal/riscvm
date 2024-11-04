#include <stdio.h>
#include <stdlib.h>
void _exit(int code) {
  asm("addi a7, zero, 93\n"
      "mv a0, %[code]\n"
      "ecall" ::[code] "r"(code));
  __builtin_unreachable();
}

int main() {
  printf("Hello World\n");
  _exit(0);
}
