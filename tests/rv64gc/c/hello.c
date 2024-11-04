#include <stdio.h>
#include <stdlib.h>

void print_str(char *str, int len) {
  asm("addi a7, zero, 64\n"
      "add a1, zero, %[ptr]\n"
      "add a2, zero, %[len]\n"
      "ecall" ::[ptr] "r"(str),
      [len] "r"(len));
}

void _start() {
  printf("Hello World\n");
  exit(0);
}
