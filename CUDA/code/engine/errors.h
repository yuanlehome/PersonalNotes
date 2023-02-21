#pragma once

#include <iostream>
#include <assert.h>
#include <stdarg.h>

#define ENFORCE_NOT_NULL(condition, ...)                  \
  if(!condition) {                                        \
    printf(__VA_ARGS__);                                  \
    std::cout << std::endl;                               \
    assert(false);                                        \
  }               

#define InvalidArgument(format, ...) printf(format, __VA_ARGS__)
#define NotFound(format, ...) printf(format, __VA_ARGS__)

#define ENFORCE_EQ(l_expression, r_expression, invalid_argument_)   \
  if(l_expression != r_expression) {                                \
    invalid_argument_;                                              \                
    std::cout << std::endl;                                         \
    assert(false);                                                  \
  }

