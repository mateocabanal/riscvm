#include <iostream>
#include <string>

int main() {
  std::string a = "";
  std::cout << "What\'s your name: ";
  std::getline(std::cin, a);
  std::cout << "Hello, " << a << '\n';
  return 0;
}
