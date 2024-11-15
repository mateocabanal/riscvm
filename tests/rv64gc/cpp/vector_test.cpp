#include <iostream>
#include <vector>

int main() {
  std::vector<char> char_buf = {'H', 'e', 'l', 'l', 'o', ',', ' '};
  std::string input_buf = "";

  std::cout << "What's your name? ";
  std::cin >> input_buf;
  std::cout << std::endl;

  for (char c : input_buf)
    char_buf.push_back(c);

  for (char c : char_buf)
    std::cout << c;

  std::cout << std::endl;

  return 0;
}
