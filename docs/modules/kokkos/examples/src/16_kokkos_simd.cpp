#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using simd_type = Kokkos::Experimental::native_simd<double>;
    using tag_type = Kokkos::Experimental::element_aligned_tag;
    constexpr int width = int(simd_type::size());
    int n = 1000;
    Kokkos::View<double *> x("x", n);
    Kokkos::View<double *> y("y", n);
    Kokkos::View<double *> z("z", n);
    Kokkos::View<double *> r("r", n);
    Kokkos::parallel_for(
        "init", n, KOKKOS_LAMBDA(const int i) {
          x(i) = static_cast<double>(i);
          y(i) = static_cast<double>(i * 2);
          z(i) = static_cast<double>(i * 3);
        });
    Kokkos::parallel_for(
        "compute", n / width, KOKKOS_LAMBDA(const int i) {
          int idx = i * width;
          simd_type sx([&x, idx](std::size_t j) { return x(idx + j); });
          simd_type sy([&y, idx](std::size_t j) { return y(idx + j); });
          simd_type sz([&z, idx](std::size_t j) { return z(idx + j); });
          simd_type sr = Kokkos::sqrt(sx * sx + sy * sy + sz * sz);
          sr.copy_to(r.data() + idx, tag_type());
        });
    Kokkos::fence();
    auto h_r = Kokkos::create_mirror_view(r);
    Kokkos::deep_copy(h_r, r);
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
      printf("r[%d] = %f\n", i, h_r(i));
    }
  }
  Kokkos::finalize();
  return 0;
}