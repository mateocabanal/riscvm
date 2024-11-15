#include <elf.h>
#include <stdio.h>
#include <unistd.h>

// Function to dump the auxiliary vector
void dump_auxv(char **envp) {
  printf("\nAuxiliary Vector:\n");
  Elf64_auxv_t *auxv;

  // Find the auxiliary vector after the environment pointers
  while (*envp++) {
  }
  auxv = (Elf64_auxv_t *)envp;

  while (auxv->a_type != AT_NULL) {
    printf("Type: %ld, Value: 0x%lx\n", auxv->a_type, auxv->a_un.a_val);
    auxv++;
  }
}

int main(int argc, char *argv[], char *envp[]) {
  // Dump argc and argv
  printf("Argument Count (argc): %d\n", argc);
  printf("\nArguments (argv):\n");
  for (int i = 0; i < argc; i++) {
    printf("[%d]: %s\n", i, argv[i]);
  }

  // Dump envp
  printf("\nEnvironment Variables (envp):\n");
  for (char **env = envp; *env != NULL; env++) {
    printf("%s\n", *env);
  }

  // Dump auxv
  dump_auxv(envp);

  return 0;
}
