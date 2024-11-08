#include <stdio.h>
#include <stdlib.h>

void print_str(char *str) {
  int len = 0;
  char *ptr = str;
  while (*ptr++ != '\0')
    len++;

  asm("addi a7, zero, 64\n"
      "add a1, zero, %[ptr]\n"
      "add a2, zero, %[len]\n"
      "ecall" ::[ptr] "r"(str),
      [len] "r"(len));
}

void _start() {
  printf("Hello World\n");

  asm("li a7, 93\n"
      "li a0, 0\n"
      "ecall" ::);
}
