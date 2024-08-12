#include <Grid/Grid.h>
#include <Grid/simd/Simd.h>

#include <iostream>

int main () {
    // FILE: Simd.h: RealF, RealD, ComplexF, ComplexD
    // FILE: Simd.h: RealF, RealD, ComplexF, ComplexD
    // FILE: Simd.h: RealF, RealD, ComplexF, ComplexD
    Grid::ComplexF z(1, 2);
    z = Grid::timesI(z);
    std::cout << z << std::endl; // should default to thrusts cout impl
    Grid::ComplexF c;
    Grid::vstream(c, z);
    std::cout << c << std::endl;
    
    c = Grid::adj(c);
    std::cout << c << std::endl;







    // FILE: Grid_vector_types.h: 
    // FILE: Grid_vector_types.h: 
    // FILE: Grid_vector_types.h: 
#ifdef GPU_VEC
    std::cout << GPU_VEC << std::endl;
#endif
#ifdef GEN
    std::cout << GEN << std::endl;
#endif

    // Type traits
    std::cout << Grid::is_complex<Grid::ComplexF>::value << std::endl;
    std::cout << Grid::is_complex<Grid::RealD>::value << std::endl;

    // Grid_simd
    // Grid::Grid_simd<Scalar_type, Vector_type>
    Grid::vRealF a(1), b(2); // Grid_simd<float, GpuVector<16, float>>
    std::cout << a + b << std::endl;
    Grid::RealF s = 5.6;
    std::cout << s * a << std::endl;

    // Access Grid_simd types
    a.putlane(123,3);
    std::cout << a.getlane(3) << std::endl;





    // FILE: Grid_gpu_vec.h
    // FILE: Grid_gpu_vec.h
    // FILE: Grid_gpu_vec.h
    std::cout << Grid::NSIMD_RealF << std::endl;
    std::cout << Grid::NSIMD_RealD << std::endl;
    std::cout << Grid::NSIMD_ComplexF << std::endl;
    std::cout << Grid::NSIMD_ComplexD << std::endl;

    // this is used as Vector_type for Grid_simd
    std::cout << Grid::SIMD_CFtype::N << std::endl;
    // they are all 64 bytes long (lol? -> nah makes sense -> BUT NOT IN THE CONTEXT OF GPUS wtf)
    std::cout << sizeof(Grid::SIMD_Ftype) << sizeof(Grid::SIMD_Dtype) << sizeof(Grid::SIMD_CFtype) << sizeof(Grid::SIMD_CDtype) << std::endl;
}
