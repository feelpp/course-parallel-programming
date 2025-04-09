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
    const int N = 1000;
    const int M = 100;

    // Create a 2D Kokkos View for storing matrix data
    Kokkos::View<double **> matrix("Matrix", N, M);

    // Define a TeamPolicy for parallelism
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    // Fill the matrix using nested parallelism
    Kokkos::parallel_for(
        "OuterLoop", team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int i = team_member.league_rank(); // Get the "outer" index
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, M),
                               [=](const int j) { matrix(i, j) = i * M + j; });
        });

    // Compute the sum of all elements in the matrix
    double sum = 0.0;
    Kokkos::parallel_reduce(
        "Sum", team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type &team_member, double &lsum) {
          const int i = team_member.league_rank(); // Get the "outer" index
          double row_sum = 0.0;

          // Compute the row sum using nested parallelism
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team_member, M),
              [=](const int j, double &thread_sum) {
                thread_sum += matrix(i, j);
              },
              row_sum);

          lsum += row_sum; // Accumulate row sums into lsum
        },
        sum);

    // Print results
    std::cout << "Total Sum : " << sum << std::endl;
    std::cout << "Average : " << sum / (N * M) << std::endl;

    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
