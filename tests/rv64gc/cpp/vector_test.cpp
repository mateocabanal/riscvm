#include <iostream>
#include <vector>

int main() {
  std::vector<char> char_buf = {'H', 'e', 'l', 'l', 'o'};

  for (char c : ", Mateo!")
    char_buf.push_back(c);

  for (char c : char_buf)
    std::cout << c << ' ';

  std::cout << std::endl;

  return 0;
}
