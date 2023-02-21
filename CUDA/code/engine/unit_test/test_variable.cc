#include "../variable.h"

int main() {
  Variable v;
  v.GetMutable<int>();
  assert(v.IsType<int>());
  int *a = v.GetMutable<int>();
  *a = 1;
  assert(v.Get<int>() == 1);
}