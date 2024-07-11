#include <Grid/Grid.h>
#include <Grid/DisableWarnings.h>
#include <Grid/Namespace.h>
#include <Grid/GridStd.h>
#include <Grid/simd/Grid_vector_types.h>
#include <Grid/threads/Pragmas.h>

using namespace Grid;


int main () {
    accelerator_for(i, 10000000, 64, {
        printf("iter1: %lu %lu\n", i, lane);
    });
}
