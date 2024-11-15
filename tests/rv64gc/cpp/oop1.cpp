#include <iostream>
class Animal {
public:
  virtual void speak() { std::cout << "...\n"; }
};

class Dog : public Animal {
public:
  void speak() override { std::cout << "Woof\n"; }
};

class Human : public Animal {
public:
  void speak() override { std::cout << "Hello\n"; }
};

int main() {
  Animal *arr[] = {new Dog(), new Dog(), new Human()};

  static_cast<Human *>(arr[2])->speak();

  return 0;
}
