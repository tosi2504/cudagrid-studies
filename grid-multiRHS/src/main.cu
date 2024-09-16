#include <Grid/Grid.h>
#include <Grid/util/Init.h>



int main (int argc, char * argv[]) {
    Grid::Grid_init(&argc, &argv);
    

    Grid::Grid_finalize();
}
