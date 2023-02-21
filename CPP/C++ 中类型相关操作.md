# 1. std::type_info
[参考](https://cplusplus.com/reference/typeinfo/type_info/)

主要内容如下：

`An object of this class is returned by the typeid operator (as a const-qualified lvalue). Although its actual dynamic type may be of a derived class.`

`typeid can be applied to any type or any expression that has a type.`

`If applied to a reference type (lvalue), the type_info returned identifies the referenced type. Any const or volatile qualified type is identified as its unqualified equivalent.`

`When typeid is applied to a reference or dereferenced pointer to an object of a polymorphic class type (a class declaring or inheriting a virtual function), it considers its dynamic type.`

demo
```cpp
#include <iostream>
#include <typeinfo>
using namespace std;

//自定义类型的定义
struct myType {
 virtual void function() {}
};

struct deriveType : myType
{
};

int main()
{
  int i = 0;
  const int &j = i;
  const std::type_info &i_info = typeid(i);
  std::cout << i_info.name() << "\n";
  const std::type_info &j_info = typeid(j);
  std::cout << j_info.name() << "\n";

  myType base;
  deriveType derive;
  myType *pbase = &base;
  myType *pbase_1 = &derive;
  std::cout << typeid(pbase).name() << "\n";
  std::cout << typeid(pbase_1).name() << "\n";
  std::cout << typeid(*pbase).name() << "\n";
  std::cout << typeid(*pbase_1).name() << "\n";
}
```
输出
```bash
i
i
P6myType
P6myType
6myType
6myType
```

# 2. std::type_index
`Class that wraps a type_info object so that it can be copied (copy-constructed and copy-assigned) and/or be used used as index by means of a standard hash function.`

demo
```cpp
#include <iostream>       // std::cout
#include <typeinfo>       // operator typeid
#include <typeindex>      // std::type_index
#include <unordered_map>  // std::unordered_map
#include <string>         // std::string

struct C {};

int main()
{
  std::unordered_map<std::type_index,std::string> mytypes;

  mytypes[std::type_index(typeid(int))]="Integer type";
  mytypes[typeid(double)]="Floating-point type";
  mytypes[typeid(C)]="Custom class named C";

  std::cout << "int: " << mytypes[typeid(int)] << '\n';
  std::cout << "double: " << mytypes[typeid(double)] << '\n';
  std::cout << "C: " << mytypes[typeid(C)] << '\n';

  return 0;
}
```

输出
```bash
```
