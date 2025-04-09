#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
     Kokkos::Timer timer;
    const int N = 1000000;

    Kokkos::View<double*, Kokkos::HIP> a("a", N);
    Kokkos::View<double*, Kokkos::HIP> b("b", N);
    Kokkos::View<double*, Kokkos::HIP> c("c", N);

    // Initialize View 'a'
    Kokkos::parallel_for("init_a", Kokkos::RangePolicy<Kokkos::HIP>(0, N), KOKKOS_LAMBDA(const int i) {
      a(i) = i * 0.1;
    });

    // Initialize View 'b'
    Kokkos::parallel_for("init_b", Kokkos::RangePolicy<Kokkos::HIP>(0, N), KOKKOS_LAMBDA(const int i) {
      b(i) = i * 0.2;
    });

    // Compute View 'c' as the sum of 'a' and 'b'
    Kokkos::parallel_for("compute_c", Kokkos::RangePolicy<Kokkos::HIP>(0, N), KOKKOS_LAMBDA(const int i) {
      c(i) = a(i) + b(i);
    });

    // Ensure all computations are complete
    Kokkos::fence();

    // Compute the sum of all elements in 'c'
    double sum = 0.0;
    Kokkos::parallel_reduce("sum_c", Kokkos::RangePolicy<Kokkos::HIP>(0, N), KOKKOS_LAMBDA(const int i, double& lsum) {
      lsum += c(i);
    }, sum);

    // Ensure all reductions are complete
    Kokkos::fence();

    std::cout << "Sum c : " << sum << std::endl;

    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
