#include "../variant.h"
#include <assert.h>
#include <iostream>

int main() {
  paddle::variant<bool, int, float> var;
  var = true;
  assert(std::string(var.type().name()) == std::string("b"));
  var = 1;
  assert(std::string(var.type().name()) == std::string("i"));
  var = 2.0f;
  assert(std::string(var.type().name()) == std::string("f"));
}

