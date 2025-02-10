#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  int N = 5, K = 10;
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double *[5]> a("a", N), b("b", K);
    a = b; // a gets deallocated and both a and b are points to the same thing
    Kokkos::View<double **> c(b); // copy constructor
    std::cout << "Label of c: " << c.label()
              << std::endl; // The label of c is the same as the label of b
    a(0, 2) = 1;
    b(0, 2) = 2;
    c(0, 2) = 3;
    std::cout << "a(0, 2) = " << a(0, 2) << std::endl;
  }
  Kokkos::finalize();
  return 0;
}