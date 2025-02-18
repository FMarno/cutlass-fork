#include <iostream>
#include "cute/tensor.hpp"                     // cute::Shape

using namespace cute;

int main() {
  std::puts("hello\n");
  using CtaTileMNK= Shape<_32,_16,_4>;
  using VMNK = Shape<_7,_4,_2,_1>;

  static constexpr auto BLK_M = get<0>(CtaTileMNK{});
  static constexpr auto BLK_N = get<1>(CtaTileMNK{});
  static constexpr auto BLK_K = get<2>(CtaTileMNK{});

  int m = 1, n = 2, l = 3;

  auto M = 512, N = 128, K=128, L=3;

  auto MNL = make_shape(M,N,L);
  auto cD =  make_counting_tensor(make_layout(MNL, make_stride(E<0>(), E<1>(), E<2>())));
  auto cta_tiling = local_tile(cD, CtaTileMNK{}, make_coord(_,_,_), Step<_1,_1, X>{});
  auto alt_cta_tiling = local_tile(cD, make_shape(BLK_M, BLK_N), make_coord(_,_));
  auto alt_2_cta_tiling = flatten(zipped_divide(cD, take<0,2>(CtaTileMNK{})));
 // cute::print(cta_tiling);
 // cute::print("\n");
 // cute::print(alt_cta_tiling);
 // cute::print("\n");
 // cute::print(alt_2_cta_tiling);
 // cute::print("\n");

 // auto tile = cta_tiling(_,_,m,n,l);
 // cute::print(tile);
 // cute::print("\n");
 // auto alt_tile = local_tile(cD, take<0,2>(CtaTileMNK{}), make_coord(m,n,l));
 // cute::print(alt_tile);
 // cute::print("\n");

 // cute::print(select<0,2>(CtaTileMNK{}));
 // cute::print("\n");

  cute::print(ceil_div(CtaTileMNK{}, take<1,4>(VMNK{})));
  cute::print("\n");
}

