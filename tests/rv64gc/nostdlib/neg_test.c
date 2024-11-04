#include <stddef.h>
#include <stdint.h>

#define SYS_write (64)
#define SYS_mmap (222)

void *mmap(void *addr, size_t size, int prot, int flags, int fd, int offset) {
  size_t sz_addr = (size_t)addr;
  size_t ret_addr = 0;
  asm("li a7, %[mmap]\n"
      "mv a0, %[sz_addr]\n"
      "mv a1, %[size]\n"
      "mv a2, %[prot]\n"
      "mv a3, %[flags]\n"
      "mv a4, %[fd]\n"
      "mv a5, %[offset]\n"
      "ecall\n"
      "mv %[addr], a0"
      : [addr] "=r"(ret_addr)
      : [mmap] "i"(SYS_mmap), [sz_addr] "r"(sz_addr), [size] "r"(size),
        [prot] "r"(prot), [flags] "r"(flags), [fd] "r"(fd), [offset] "r"(offset)
      : "a0", "a1", "a2", "a3", "a4", "a5", "a7");

  return (void *)ret_addr;
};

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
  char buf[19];
  buf[18] = '\0';
  for (int i = 17; i >= 2; i--) {
    int digit = val & 0xF;
    buf[i] = (digit < 10) ? ('0' + digit) : ('a' + digit - 10);
    val >>= 4;
  }

  buf[0] = '0';
  buf[1] = 'x';
  write_str(buf);
}

void _memcpy(void *a, void *b, size_t len) {
  uint8_t *a_bit = (uint8_t *)a;
  uint8_t *b_bit = (uint8_t *)b;
  for (uint64_t i = 0; i < len; i++)
    *(a_bit + i) = *(b_bit + i);
}

void _start() {
  int64_t a = 10;
  write_hex(a);
  write_str("\n");
  write_hex(-a);
  write_str("\n");

  int64_t *og = mmap(NULL, 4096, 0, 0, -1, 0);
  og[0] = -1;

  int64_t *cpy = mmap(NULL, 4096, 0, 0, -1, 0);

  _memcpy(cpy, og, 8);

  if (og[0] == cpy[0]) {
    write_str("successfully copied!\n");
  } else {
    write_str("failed to copy!\n");
  }

  asm("li a7, 93\n"
      "li a0, 0\n"
      "ecall" ::);
}
