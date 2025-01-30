#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>


#include <Kokkos_Core.hpp>


struct VectorAddFunctor
{
    Kokkos::View<double*, Kokkos::HIPSpace> a;
    Kokkos::View<double*, Kokkos::HIPSpace> b;
    Kokkos::View<double*, Kokkos::HIPSpace> c;

    VectorAddFunctor( Kokkos::View<double*, Kokkos::HIPSpace> a_,
                      Kokkos::View<double*, Kokkos::HIPSpace> b_,
                      Kokkos::View<double*, Kokkos::HIPSpace> c_ )
        : a( a_ ), b( b_ ), c( c_ ) {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int i ) const
    {
        c( i ) = a( i ) + b( i );
    }
};



int main( int argc, char* argv[] )
{
    Kokkos::initialize(argc, argv);
    { 
        const int vector_size = 1000000;
        const int num_gpus = Kokkos::HIP::detect_device_count();
        // const int num_gpus = Kokkos::HIP::impl_internal_space_instance()->m_device_count; //another approach
        Kokkos::InitializationSettings settings;
        std::cout << "Number of GPUs available : " << num_gpus << std::endl;
        // #define MemSpace Kokkos::Experimental::HIPSpace

        Kokkos::Timer timer;

        for ( int gpu = 0; gpu < num_gpus; ++gpu )
        {

            settings.set_device_id( gpu );
            Kokkos::HIP::impl_initialize( settings );
            Kokkos::fence();

            Kokkos::View<double*, Kokkos::HIPSpace> a( "a", vector_size );
            Kokkos::View<double*, Kokkos::HIPSpace> b( "b", vector_size );
            Kokkos::View<double*, Kokkos::HIPSpace> c( "c", vector_size );

            Kokkos::parallel_for(
                Kokkos::RangePolicy<Kokkos::HIP>( 0, vector_size ),
                KOKKOS_LAMBDA( const int i ) {
                    a( i ) = 1.0;
                    b( i ) = 2.0;
                } );

            Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::HIP>( 0, vector_size ),
                                VectorAddFunctor( a, b, c ) );

            Kokkos::View<double*>::HostMirror h_c = Kokkos::create_mirror_view( c );
            Kokkos::deep_copy( h_c, c );

            bool correct = true;
            for ( int i = 0; i < vector_size; ++i )
            {
                if ( h_c( i ) != 3.0 )
                {
                    correct = false;
                    break;
                }
            }

            std::cout << "Result on GPU " << gpu << " : " << ( correct ? "Correct" : "Incorrect" ) << std::endl;
        }

        double elapsed_time = timer.seconds();

        std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;

    }
    Kokkos::finalize();
    return 0;
}
