#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const size_t N = 100000;
    // Create a 1D View of doubles
    Kokkos::View<int *> myView("MyView", N);
    // Fill the View with data
    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) { myView(i) = i; });
    // Compute the sum of all elements
    int sum = 0.0;
    Kokkos::parallel_reduce(
        N,
        KOKKOS_LAMBDA(const int i, int &partial_sum) {
          partial_sum += myView(i);
        },
        sum);

    std::cout << "Sum: " << sum << std::endl;
  }
  Kokkos::finalize();
  return 0;
}