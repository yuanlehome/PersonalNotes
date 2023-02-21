1. 在使用gcc/g++编译器时，通过添加 -D 选项，添加用户自定义的宏。

```cpp
#include <iostream>

int main() {
#ifdef PRINT
  std::cout << "hello\n";
#endif
}
```

编译命令: `g++ -D PRINT main.cc `

输出： hello。