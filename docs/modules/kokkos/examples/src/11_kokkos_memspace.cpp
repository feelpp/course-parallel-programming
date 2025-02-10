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
#ifdef KOKKOS_ENABLE_CUDA
    std::cout << "Kokkos::CudaSpace" << std::endl;
#define MemSpace Kokkos::CudaSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
    std::cout << "Kokkos::HIPSpace" << std::endl;
#define MemSpace Kokkos::Experimental::HIPSpace
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    std::cout << "Kokkos::OpenMPTargetSpace" << std::endl;
#define MemSpace Kokkos::OpenMPTargetSpace
#endif

#ifndef MemSpace
    std::cout << "Kokkos::HostSpace" << std::endl;
#define MemSpace Kokkos::HostSpace
#endif

    const int N = 1000000;
    Kokkos::View<double *, MemSpace> data("data", N);

    Kokkos::Timer timer;

    Kokkos::parallel_for(
        "init", N, KOKKOS_LAMBDA(const int i) { data(i) = i * 0.000001; });

    double sum = 0.0;
    Kokkos::parallel_reduce(
        "sum", N,
        KOKKOS_LAMBDA(const int i, double &partial_sum) {
          partial_sum += data(i);
        },
        sum);

    double elapsed_time = timer.seconds();

    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
