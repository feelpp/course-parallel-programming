#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

struct ScratchMemoryExample {
    Kokkos::View<double*> data;
    ScratchMemoryExample(int N) : data("data", N) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type& team_member) const {
        const int team_size = team_member.team_size();
        const int team_rank = team_member.team_rank();
        const int league_rank = team_member.league_rank();

        // Allocate team scratch memory
        double* team_scratch = (double*)team_member.team_shmem().get_shmem(team_size * sizeof(double));

        // Each thread initializes its scratch memory
        team_scratch[team_rank] = league_rank * team_size + team_rank;

        // Synchronize to ensure all threads have written to scratch memory
        team_member.team_barrier();

        // Perform a reduction within the team
        double team_sum = 0.0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, team_size), [&](const int i, double& lsum) {
            lsum += team_scratch[i];
        }, team_sum);

        // Only one thread writes the result back to global memory
        if (team_rank == 0) {
            data(league_rank) = team_sum;
        }
    }

    // Specify the amount of scratch memory needed
    size_t team_shmem_size(int team_size) const {
        return team_size * sizeof(double);
    }
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int N = 1000;
        ScratchMemoryExample functor(N);
        Kokkos::parallel_for(Kokkos::TeamPolicy<>(N / 10, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(functor.team_shmem_size(10))), functor);
    }
    Kokkos::finalize();
    return 0;
}