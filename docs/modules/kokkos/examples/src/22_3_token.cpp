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
      // Size of the array
      const int N = 100;
      // Kokkos view to store the results
      Kokkos::View<int*> results("results", N);
      // Create a UniqueToken (based on thread execution)
      Kokkos::Experimental::UniqueToken<Kokkos::DefaultExecutionSpace> unique_token;
      // Number of available threads
      const int num_threads = unique_token.size();
      std::cout << "Number of threads: " << num_threads << std::endl;
      Kokkos::parallel_for("UniqueTokenExample", N, KOKKOS_LAMBDA(const int i) {
          // Get a unique identifier for this thread
          int token = unique_token.acquire();
          results(i) = i;
          unique_token.release(token);
      });
      // Copy the results to the host for display
      auto host_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
      std::cout << "Results: ";
      for (int i = 0; i < N; ++i) {
          std::cout << host_results(i) << " ";
      }
      std::cout << std::endl;
  }
  Kokkos::finalize();
}
