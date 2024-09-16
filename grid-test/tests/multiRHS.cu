// #include "multigrid/GeneralCoarsenedMatrixMultiRHS.h"
#include <Grid/Grid.h>
#include "multigrid/MultiGrid.h"

using namespace Grid;

int main (int argc, char * argv[]) {
    Grid_init(&argc, &argv);
    MultiGeneralCoarsenedMatrix<iMatrix<vComplexD, 3>, vComplexD, 10> M;
    Grid_finalize();
}


