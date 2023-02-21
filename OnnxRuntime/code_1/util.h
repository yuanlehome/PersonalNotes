
#include <chrono>
#include <iostream>
namespace util{

using namespace std::chrono;
using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };

// ms
double time_diff(Time start, Time end) {
  auto counter = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return counter.count() / 1000.0;
}

}