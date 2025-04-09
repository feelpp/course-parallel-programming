#include <Kokkos_Core.hpp>
#include <cassert>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int N0 = 10;
    Kokkos::View<double *[5]> a("A", N0);
    assert(a.extent(0) == N0);
    assert(a.extent(1) == 5);
    assert(a.size() == N0 * 5);
    assert(a.rank() == 2);
    assert(a.span() == N0 * 5);
    assert(a.data() != nullptr);
    assert(a.label() == "A");
  }
  Kokkos::finalize();
  return 0;
}