#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>


#include <Kokkos_Core.hpp>



int main( int argc, char* argv[] )
{
    Kokkos::initialize(argc, argv);
    {

 Kokkos::Timer timer;
    const int N = 1000000;

    Kokkos::View<double*> data("Data", N);

    // Data initialization
    Kokkos::parallel_for("Init", N, KOKKOS_LAMBDA(const int i) {
      data(i) = std::sin(i * 0.001) * 100;
    });

    CustomReduction result;

    // Advanced reduction
    Kokkos::parallel_reduce("AdvancedReduction", N,
      KOKKOS_LAMBDA(const int i, CustomReduction& local) {
        local.max_value = std::max(local.max_value, data(i));
        local.sum += data(i);
        if (data(i) > 50) local.count++;
      },
      Kokkos::Sum<CustomReduction>(result)
    );

    std::cout << "Maximum value : " << result.max_value << std::endl;
    std::cout << "Sum : " << result.sum << std::endl;
    std::cout << "Number of values > 50 : " << result.count << std::endl;
    std::cout << "Average : " << result.sum / N << std::endl;

    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}

