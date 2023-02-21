#include <exception>
#include <vector>
#include <iostream>

int fun() noexcept {
  throw int(1);
}

int main() {
  std::vector<int>vec(10);
  try{
    fun();
  }catch(int i){
    std::cout << "error int: " << i;
  }
}