#include <feel/feel.hpp>
// #include <Kokkos_Core.hpp>
#include <feel/feelcore/kokkos.hpp>

using namespace Feel;

int main(int argc, char**argv )
{
    Environment env( _argc=argc, _argv=argv,
                     _desc=feel_options(),
                     _about=about(_name="mylaplacian",
                                  _author="Feel++ Consortium",
                                  _email="feelpp-devel at feelpp.org"));


    Kokkos::initialize(argc, argv);
    // create mesh
    auto mesh = unitSquare();

    // function space
    auto Vh = Pch<1>( mesh );
    auto u = Vh->element();
    auto v = Vh->element();

    // left hand side
    auto a = form2( _trial=Vh, _test=Vh );
    a = integrate(_range=elements(mesh), _expr=gradt(u)*trans(grad(v)) );

    // right hand side
    auto l = form1( _test=Vh );
    l = integrate(_range=elements(mesh), _expr=id(v));

    // boundary condition
    a+=on(_range=boundaryfaces(mesh), _rhs=l, _element=u, _expr=constant(0.) );

    // solve the equation a(u,v) = l(v)
    a.solve(_rhs=l,_solution=u);

    Kokkos::finalize();
}