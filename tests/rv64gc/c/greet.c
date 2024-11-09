#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// FIXME: Printf fails when including more than 1 variadic argument (even tho
// variadic arguments work) For now, you have to separate the printf calls
int main() {
  char name[50];
  scanf("%s", name);
  printf("Hello, ");
  printf("%s\n", name);
  return 0;
}
