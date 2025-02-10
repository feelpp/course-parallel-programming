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
    const int num_streams = 2;

    Kokkos::View<double*, Kokkos::HIP> a("a", N);
    Kokkos::View<double*, Kokkos::HIP> b("b", N);
    Kokkos::View<double*, Kokkos::HIP> c("c", N);

    std::vector<Kokkos::HIP> streams(num_streams);

    for (int i = 0; i < num_streams; ++i) {
      streams[i] = Kokkos::HIP();
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::HIP>(streams[0], 0, N), KOKKOS_LAMBDA(const int i) {
      a(i) = i * 0.1;
    });

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::HIP>(streams[1], 0, N), KOKKOS_LAMBDA(const int i) {
      b(i) = i * 0.2;
    });

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::HIP>(streams[0], 0, N), KOKKOS_LAMBDA(const int i) {
      c(i) = a(i) + b(i);
    });

    Kokkos::fence();

    double sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::HIP>(0, N), KOKKOS_LAMBDA(const int i, double& lsum) {
      lsum += c(i);
    }, sum);

    std::cout << "Sum c : " << sum << std::endl;

    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
