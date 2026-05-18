#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

int main() {
  const char *path = "riscvm-file-rw.txt";
  const char *message = "riscvm virtual filesystem\n";
  char buffer[64] = {0};

  int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
  if (fd < 0) {
    puts("file-rw: open failed");
    return 1;
  }

  ssize_t written = write(fd, message, strlen(message));
  if (written != (ssize_t)strlen(message)) {
    puts("file-rw: write failed");
    return 2;
  }

  int file_flags = fcntl(fd, F_GETFL);
  if (file_flags < 0) {
    puts("file-rw: fcntl failed");
    return 3;
  }

  if (lseek(fd, 0, SEEK_SET) != 0) {
    puts("file-rw: lseek failed");
    return 4;
  }

  ssize_t read_len = read(fd, buffer, sizeof(buffer) - 1);
  if (read_len != (ssize_t)strlen(message)) {
    puts("file-rw: read failed");
    return 5;
  }
  buffer[read_len] = '\0';

  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size != (off_t)strlen(message)) {
    puts("file-rw: fstat failed");
    return 6;
  }

  char pread_buffer[64] = {0};
  read_len = pread(fd, pread_buffer, sizeof(pread_buffer) - 1, 0);
  if (read_len != (ssize_t)strlen(message) ||
      strcmp(pread_buffer, message) != 0) {
    puts("file-rw: pread failed");
    return 7;
  }

  if (close(fd) != 0) {
    puts("file-rw: close failed");
    return 8;
  }

  if (strcmp(buffer, message) != 0) {
    puts("file-rw: mismatch");
    return 9;
  }

  if (access(path, F_OK) != 0) {
    puts("file-rw: access failed");
    return 10;
  }

  if (stat(path, &st) != 0 || st.st_size != (off_t)strlen(message)) {
    puts("file-rw: stat failed");
    return 11;
  }

  char cwd[8];
  if (getcwd(cwd, sizeof(cwd)) == NULL || strcmp(cwd, "/") != 0) {
    puts("file-rw: getcwd failed");
    return 12;
  }

  struct utsname uts;
  if (uname(&uts) != 0 || strcmp(uts.machine, "riscv64") != 0) {
    puts("file-rw: uname failed");
    return 13;
  }

  errno = 0;
  int missing = open("riscvm-missing-file.txt", O_RDONLY);
  if (missing != -1 || errno != ENOENT) {
    puts("file-rw: errno ENOENT failed");
    return 14;
  }

  errno = 0;
  char bad_read;
  if (read(fd, &bad_read, 1) != -1 || errno != EBADF) {
    puts("file-rw: errno EBADF failed");
    return 15;
  }

  printf("file-rw: %s", buffer);
  return 0;
}
