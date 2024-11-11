#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  // char **char_buf = malloc(5);
  // char_buf = realloc(char_buf, 9);
  //
  // for (int i = 0; i < argc; i++) {
  //   char *arg = argv[i];
  //   size_t len = strlen(arg);
  //   char *new_buf = malloc(len + 1);
  //   strcpy(new_buf, arg);
  //
  //   char_buf[i] = new_buf;
  // }

  for (int i = 0; i < argc; i++)
    printf("%s\n", argv[i]);

  return 0;
}
