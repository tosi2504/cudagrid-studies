#include <Grid/Grid.h>
#include <Grid/algorithms/blas/BatchedBlas.h>
#include <Grid/util/Coordinate.h>
#include <array>

using namespace Grid;

constexpr int nbasis = 32;

int main (int argc, char * argv[]) {
    Grid_init(&argc, &argv);
    
    std::array<int, 8> nrhss = {8, 16, 24, 32, 40, 48, 56, 64};

    for (auto nrhs: nrhss) {
        
        // const int nbasis = 64; // N
        // const int nrhs = 64; // M
        Coordinate coarseLatt({8,8,8,8});

        Coordinate rhMpi ({1,1,1,1,1,1});
        Coordinate rhLatt({nrhs,1,coarseLatt[0],coarseLatt[1],coarseLatt[2],coarseLatt[3]});
        Coordinate rhSimd({vComplexF::Nsimd(),1, 1,1,1,1});

        GridCartesian *Coarse4d = SpaceTimeGrid::makeFourDimGrid (
                coarseLatt,
                GridDefaultSimd(Nd,vComplexF::Nsimd()),
                GridDefaultMpi()
        );
        GridCartesian *Coarse5d =  SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);
        GridCartesian *CoarseMrhs = new GridCartesian(rhLatt,rhSimd,rhMpi); 

        NearestStencilGeometry5D geom(Coarse5d);
        
        typedef MultiGeneralCoarsenedMatrix<vSpinColourVectorF,vTComplexF,nbasis> MultiGeneralCoarsenedMatrix_t;
        MultiGeneralCoarsenedMatrix_t mrhs(geom,CoarseMrhs);

        MultiGeneralCoarsenedMatrix_t::CoarseMatrix A(Coarse5d);
        MultiGeneralCoarsenedMatrix_t::CoarseVector B(CoarseMrhs);
        MultiGeneralCoarsenedMatrix_t::CoarseVector C(CoarseMrhs);
        
        GridParallelRNG rhRNG(CoarseMrhs); rhRNG.SeedFixedIntegers({1,2,3,4});
        GridParallelRNG lhRNG(Coarse5d); lhRNG.SeedFixedIntegers({5,6,7,8});
        random(rhRNG, B);
        random(lhRNG, A);

        for (int p = 0; p < geom.npoint; p++) {
            mrhs.SetMatrix(p, A);
        }
        for (int i = 0; i < 100; i++) {
            std::cout << "Iteration: " << i << std::endl;
            std::cout << "nrhs:   " << nrhs << std::endl;
            std::cout << "nbasis: " << nbasis << std::endl;
            mrhs.M(B, C);
        }
    }
    
    Grid_finalize();
}
