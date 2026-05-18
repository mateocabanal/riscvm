#include <string.h>

static void riscvm_print_pointer(void *addr) {
  asm("li a7, 1004\n"
      "mv a0, %[ptr]\n"
      "ecall\n" ::[ptr] "r"(addr));
}

static void print_str(const char *str) {
  int len = strlen(str);
  __asm__("addi a7, x0, 1006\n"
          "add a0, x0, %0\n"
          "add a1, x0, %1\n"
          "ecall" ::"r"(str),
          "r"(len));
}
