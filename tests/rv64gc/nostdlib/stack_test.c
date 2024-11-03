// test.c
// Compile with: riscv64-linux-gnu-gcc -nostdlib -static -o test test.c

#include <stddef.h>
#include <stdint.h>

#define NULL ((void *)0)

// System call numbers
#define SYS_exit 93
#define SYS_write 64

// Auxiliary vector types
#define AT_NULL 0

// Function prototypes
void _start(void) __attribute__((naked));
void exit(int status);
size_t write(int fd, const void *buf, size_t count);

// Helper functions
void write_str(const char *str);
void write_hex(uint64_t val);

// _start function: Entry point
void _start(void) {
  // Stack pointer
  uint64_t *sp;

  // Get the stack pointer
  asm volatile("mv %0, sp" : "=r"(sp));

  // Extract argc
  int argc = *sp++;

  // Extract argv pointers
  char **argv = (char **)sp;
  sp += argc + 1; // Move past argv pointers and NULL terminator

  // Extract envp pointers
  char **envp = (char **)sp;
  while (*sp++ != 0)
    ; // Move past envp pointers and NULL terminator

  // Now sp points to auxv
  uint64_t *auxv = sp;

  // Output argc
  write_str("argc = ");
  write_hex(argc);
  write_str("\n");

  // Output argv
  for (int i = 0; i < argc; i++) {
    write_str("argv[");
    write_hex(i);
    write_str("] = ");
    write_str(argv[i]);
    write_str("\n");
  }

  // Output auxv
  write_str("auxv entries:\n");
  uint64_t *auxp = auxv;
  while (auxp[0] != AT_NULL) {
    write_str("Type: ");
    write_hex(auxp[0]);
    write_str(", Value: ");
    write_hex(auxp[1]);
    write_str("\n");
    auxp += 2;
  }
  write_str("Done auxv\n");

  // Exit the program
  exit(0);
}

// Implement exit syscall
void exit(int status) {
  asm volatile("mv a0, %0\n"
               "li a7, %1\n"
               "ecall\n"
               :
               : "r"(status), "i"(SYS_exit)
               : "a0", "a7");
  __builtin_unreachable();
}

// Implement write syscall
size_t write(int fd, const void *buf, size_t count) {
  size_t ret;
  asm volatile("mv a0, %1\n"
               "mv a1, %2\n"
               "mv a2, %3\n"
               "li a7, %4\n"
               "ecall\n"
               "mv %0, a0\n"
               : "=r"(ret)
               : "r"(fd), "r"(buf), "r"(count), "i"(SYS_write)
               : "a0", "a1", "a2", "a7");
  return ret;
}

// Helper function to write a string
void write_str(const char *str) {
  const char *s = str;
  size_t len = 0;
  while (s[len] != '\0') {
    len++;
  }
  write(1, str, len);
}

// Helper function to write a 64-bit unsigned integer in hexadecimal
void write_hex(uint64_t val) {
  char buf[17];
  buf[16] = '\0';
  for (int i = 15; i >= 0; i--) {
    int digit = val & 0xF;
    buf[i] = (digit < 10) ? ('0' + digit) : ('a' + digit - 10);
    val >>= 4;
  }
  write_str(buf);
}
