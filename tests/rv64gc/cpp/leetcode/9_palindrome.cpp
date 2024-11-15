#include <iostream>
#include <utility>
#include <vector>
int reverse_num(int x) {
  int64_t rev = 0;

  while (x > 0) {
    int digit = x % 10;
    rev *= 10;
    rev += digit;
    x /= 10;
  }

  return rev;
}

bool is_palindrome(int x) {
  int reversed_num = reverse_num(x);

  return x == reversed_num;
}

int main() {
  std::vector<std::pair<int, bool>> testCases = {
      std::make_pair(121, true),
      std::make_pair(-121, false),
  };

  for (auto p : testCases) {
    if (is_palindrome(p.first) != p.second) {
      std::cout << "Test " << p.first << " failed!\n";
      return -1;
    }
  }

  std::cout << "All tests passed!\n";

  return 0;
}
