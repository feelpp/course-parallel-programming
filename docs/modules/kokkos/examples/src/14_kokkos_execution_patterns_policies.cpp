#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

struct VectorAdd {
  // Member variables for the vectors
  Kokkos::View<double *> a;
  Kokkos::View<double *> b;
  Kokkos::View<double *> c;
  // Constructor to initialize the vectors
  VectorAdd(Kokkos::View<double *> a_, Kokkos::View<double *> b_,
            Kokkos::View<double *> c_)
      : a(a_), b(b_), c(c_) {}
  // Functor to perform vector addition
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    c(i) = a(i) + b(i); // Perform addition
  }
};

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 1000; // Size of the vectors
    // Allocate vectors on the device
    Kokkos::View<double *> a("A", N);
    Kokkos::View<double *> b("B", N);
    Kokkos::View<double *> c("C", N);
    // Initialize vectors a and b on the host
    Kokkos::parallel_for(
        "InitializeVectors", N, KOKKOS_LAMBDA(const int i) {
          a(i) = static_cast<double>(i); // Fill vector A with values 0 to N-1
          b(i) =
              static_cast<double>(N - i); // Fill vector B with values N-1 to 0
        });
    // Perform vector addition using default execution policy
    Kokkos::parallel_for("VectorAdd", N, VectorAdd(a, b, c));
    // Synchronize to ensure all computations are complete
    Kokkos::fence();

    auto h_c = Kokkos::create_mirror_view(c);

    // Output the first 10 results for verification
    Kokkos::deep_copy(h_c, c);
    std::cout << "Result of vector addition (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "c[" << i << "] = " << h_c(i)
                << std::endl; // Print results from vector C
    }
    // Perform vector addition using a different execution policy (Dynamic
    // Scheduling)
    Kokkos::TeamPolicy<> teamPolicy(N, 32); // League size: N, Team size: 32
    Kokkos::parallel_for(
        teamPolicy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &teamMember) {
          const int teamSize = teamMember.team_size();
          const int i =
              teamMember.league_rank() * teamSize + teamMember.team_rank();
          if (i < N) {
            c(i) = a(i) + b(i); // Perform addition within the team
          }
        });

    // Synchronize again after using the team policy
    Kokkos::fence();
    // Output the results after using the team policy
    Kokkos::deep_copy(h_c, c);
    std::cout
        << "Result of vector addition using Team Policy (first 10 elements):"
        << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "c[" << i << "] = " << h_c(i)
                << std::endl; // Print results from vector C
    }
  }
  Kokkos::finalize();
  return 0;
}