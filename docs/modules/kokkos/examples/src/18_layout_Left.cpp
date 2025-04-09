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
    const int N = 5;
    const int M = 4;

    Kokkos::Timer timer;

    // Creating a 2D View with LayoutLeft
    Kokkos::View<double **, Kokkos::LayoutLeft> matrix("Matrix", N, M);

    // Filling the matrix
    Kokkos::parallel_for(
        "FillMatrix", N, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < M; ++j) {
            matrix(i, j) = i * 10 + j;
          }
        });

    // Create a mirror on the host to display the results
    auto h_matrix = Kokkos::create_mirror_view(matrix);
    Kokkos::deep_copy(h_matrix, matrix);

    // Display the matrix
    std::cout << "Matrix with LayoutLeft :" << std::endl;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        std::cout << h_matrix(i, j) << " ";
      }
      std::cout << std::endl;
    }

    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
