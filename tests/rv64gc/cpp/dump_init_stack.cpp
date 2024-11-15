#include <elf.h>
#include <iostream>
#include <unistd.h>

void dump_auxv(char **envp) {
  std::cout << "\nAuxiliary Vector:" << std::endl;

  // Move past environment variables to find auxv
  while (*envp++) {
  }
  Elf64_auxv_t *auxv = reinterpret_cast<Elf64_auxv_t *>(envp);

  while (auxv->a_type != AT_NULL) {
    std::cout << "Type: " << auxv->a_type << ", Value: 0x" << std::hex
              << auxv->a_un.a_val << std::dec << '\n';
    auxv++;
  }
}

int main(int argc, char *argv[], char *envp[]) {
  // Dump argc
  std::cout << "Argument Count (argc): " << argc << std::endl;

  // Dump argv
  std::cout << "\nArguments (argv):" << std::endl;
  for (int i = 0; i < argc; i++) {
    std::cout << "[" << i << "]: " << argv[i] << std::endl;
  }

  // Dump envp
  std::cout << "\nEnvironment Variables (envp):" << std::endl;
  for (char **env = envp; *env != nullptr; env++) {
    std::cout << *env << std::endl;
  }

  // Dump auxv
  dump_auxv(envp);

  return 0;
}
