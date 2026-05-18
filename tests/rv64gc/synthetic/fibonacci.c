#include <stdint.h>
#include <stdio.h>

static uint64_t fibonacci_rounds(unsigned rounds) {
  uint64_t previous = 1;
  uint64_t current = 1;
  uint64_t checksum = 0;

  for (unsigned i = 0; i < rounds; i++) {
    uint64_t next = previous + current;
    checksum ^= next;
    previous = current;
    current = next;
  }

  return checksum;
}

int main(void) {
  uint64_t checksum = fibonacci_rounds(5000000);
  printf("fibonacci checksum: %llu\n", (unsigned long long)checksum);
  return 0;
}
