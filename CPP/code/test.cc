#include <iostream>

int main() {
  int a = 0;
  int b = 1;
  auto c = [&]{
    return a+b;
  }();
  std::cout << "c: " << c << "\n";
}
