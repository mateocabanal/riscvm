#include <stdio.h>
#include <unistd.h>

int main() {
  int val = isatty(1);
  if (val == 1) {
    printf("isatty: true\n");
  } else {
    printf("isatty: false\n");
  }
}
