#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

struct DualViewExample {
  // Define the dual view type
  using dual_view_type = Kokkos::DualView<double *, Kokkos::LayoutLeft>;

  // Function to initialize device view
  static void initialize(dual_view_type &dv) {
    // Initialize the device view with values
    Kokkos::parallel_for(
        "Initialize DeviceView", dv.d_view.extent(0),
        KOKKOS_LAMBDA(const int i) {
          dv.d_view(i) = static_cast<double>(i); // Assign values based on index
        });
    // Synchronize to update the host mirror
    dv.template sync<Kokkos::HostSpace>();
  }
  // Function to print values from both views
  static void printValues(const dual_view_type &dv) {
    std::cout << "Host View Values: ";
    for (int i = 0; i < dv.h_view.extent(0); ++i) {
      std::cout << dv.h_view(i) << " "; // Access host view
    }
    std::cout << std::endl;
    std::cout << "Device View Values: ";
    Kokkos::parallel_for(
        "Print DeviceView", dv.d_view.extent(0), KOKKOS_LAMBDA(const int i) {
          printf("%f ", dv.d_view(i)); // Access device view
        });
    std::cout << std::endl;
  }
};

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 10; // Size of the DualView
    // Create a DualView with N elements
    DualViewExample::dual_view_type dv("MyDualView", N);
    // Initialize the device view
    DualViewExample::initialize(dv);
    // Print values from both views
    DualViewExample::printValues(dv);
  }
  Kokkos::finalize();
  return 0;
}