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

    Kokkos::View<double *> data("Data", N);

    // Data initialization
    Kokkos::parallel_for(
        "Init", N,
        KOKKOS_LAMBDA(const int i) { data(i) = std::sin(i * 0.001) * 100; });

    double sum = 0.0;
    double max_val = -std::numeric_limits<double>::max();
    int count_positive = 0;

    // Multiple reductions
    Kokkos::parallel_reduce(
        "MultipleReductions", N,
        KOKKOS_LAMBDA(const int i, double &lsum, double &lmax, int &lcount) {
          lsum += data(i);
          lmax = std::max(lmax, data(i));
          if (data(i) > 0)
            lcount++;
        },
        sum, Kokkos::Max<double>(max_val), count_positive);

    std::cout << "Sum : " << sum << std::endl;
    std::cout << "Maximum value : " << max_val << std::endl;
    std::cout << "Number of positive values : " << count_positive << std::endl;
    std::cout << "Average : " << sum / N << std::endl;

    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
