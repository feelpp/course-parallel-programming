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
    const int N = 100;
    const int M = 64;
    const int K = 16;

    Kokkos::View<double***> A("A", N, M, K);

    Kokkos::parallel_for("Init", N*M*K, KOKKOS_LAMBDA(const int idx) {
      int i = idx / (M*K);
      int j = (idx / K) % M;
      int k = idx % K;
      A(i,j,k) = i*M*K + j*K + k;
    });

    double total_sum = 0.0;

    Kokkos::parallel_reduce("CombinedRanges", 
      Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member, double& team_sum) {
        const int n = team_member.league_rank();

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, M),
          [&](const int m, double& thread_sum) {
            double vector_sum = 0.0;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, K),
              [&](const int k, double& inner_sum) {
                inner_sum += A(n,m,k);
              }, vector_sum);
            thread_sum += vector_sum;
          }, team_sum);
      }, total_sum);

    std::cout << "Total Sum : " << total_sum << std::endl;
    std::cout << "Average : " << total_sum / (N*M*K) << std::endl;

    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}



