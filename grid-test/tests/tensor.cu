#include <Grid/Grid.h>
#include <Grid/tensors/Tensor_class.h>
#include <Grid/tensors/Tensor_traits.h>


using namespace Grid;


int main () {
    // FILE: Tensor_traits.h
    // FILE: Tensor_traits.h
    // FILE: Tensor_traits.h
    // Base type mapper
    GridTypeMapper<RealF>::Complexified c;

    GridTypeMapper<vRealF>::scalar_type a;
    GridTypeMapper<vRealF>::vector_type d;
    GridTypeMapper<vRealF>::Complexified b;
    //GridTypeMapper<vRealF>::Rank;
    // GridTypeMapper<vRealF>::TensorLevel;
    // GridTypeMapper<vRealF>::count; // total number of elements

    // for tensor types
    GridTypeMapper<iScalar<RealF>>::scalar_type t1; // gets fundamental type as scalar
    GridTypeMapper<iScalar<RealF>>::vector_type t3; // gets fundamental type as scalar
    GridTypeMapper<iScalar<RealF>>::Complexified t5; // gets fundamental type as scalar

    GridTypeMapper<iScalar<vRealF>>::scalar_type t2;
    GridTypeMapper<iScalar<vRealF>>::vector_type t4;

    // GridTypeMapper<iMatrix<iMatrix<vRealF ,2>, 2>>::count;
    // iMatrix<iMatrix<vRealF ,2>, 2> m2;
    // sizeof(vRealF);
    // sizeof(m2);
    


    // FILE: Tensor_class.h
    // FILE: Tensor_class.h
    // FILE: Tensor_class.h
    iScalar<iScalar<iScalar<RealF>>> s1(0);
    std::cout << s1 << std::endl;
    std::cout << TensorRemove(s1) << std::endl;

    iMatrix<iMatrix<vRealF, 2>, 2> m1;
    decltype(m1)::Traits::Rank;
    m1.Nsimd();


    // FILE: Tensor_arithm_mul.h
    // FILE: Tensor_arithm_mul.h
    // FILE: Tensor_arithm_mul.h
    iVector<RealF, 2> v1;
    v1(0) = 1;
    v1(1) = 1;

    iMatrix<RealF, 2> m2;
    m2(0, 0) = 1;
    m2(0, 1) = 0;
    m2(1, 0) = 0;
    m2(1, 1) = 1;

    std::cout << m2 * v1 << std::endl;

    // FILE: Tensor_index.h
    iVector<iVector<RealF, 3>, 30> v;
    auto vres = TensorIndexRecursion<1>::peekIndex(v, 0);
}
