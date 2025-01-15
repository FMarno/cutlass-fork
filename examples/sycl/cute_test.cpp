#include <iostream>
#include "cute/tensor.hpp"                     // cute::Shape

int main() {
  std::puts("hello\n");
  auto shape = Shape<_3,_4,_5>{};
  auto layout = make_layout(shape);
  cute::print(layout);
  cute::print(coshape(layout));
}
