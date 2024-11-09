#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void _sys_print_i64(int64_t i) {
  asm("mv a0, %[num]\n"
      "li a7, 1000\n"
      "ecall" ::[num] "r"(i));
}

int main(int argc, char **argv) {
  printf("at main!\n");

  char *mem = malloc(50);
  strcpy(mem, "Hello, World");

  printf("%s\n", mem);
  _sys_print_i64(strlen(mem));

  free(mem);
  return 0;
}
